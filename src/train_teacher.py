"""
IPD Phase 3: Field-Robust Teacher Model Training Pipeline
=========================================================
Trains crop-specific EfficientNetB3 Teacher Classifiers (Potato, Tomato, Rice).
Implements the 4-Phase Plan to bridge the Studio -> Field domain gap:
  - Phase A: Field-Robust Data Augmentation (Zoom-In Crop, Color Jitter, Random Erasing)
  - Phase B: Fresh Retraining with Regularized Head (Dropout 0.4) & Deep Fine-Tuning (Top 40 layers, LR 5e-5)
  - Phase C: Test set benchmark regression audit + Model Manifest Locking

Usage:
    # Train Potato Teacher (default: 20 epochs Stage A, 15 epochs Stage B):
    python -m src.train_teacher --crop potato

    # Train with custom epochs or on GPU:
    python -m src.train_teacher --crop potato --epochs-a 20 --epochs-b 15 --gpu

    # Train Tomato or Rice:
    python -m src.train_teacher --crop tomato
    python -m src.train_teacher --crop rice
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from src.augmentations import visualize_augmentations
from src.dataset import load_crop_datasets
from src.evaluate import evaluate_teacher_model
from src.model import build_teacher_model, setup_stage_b_fine_tuning


# Set Reproducibility Seed
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hex digest of a model artifact."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def setup_hardware(enable_gpu_mixed_precision: bool = True):
    """Detects and configures GPU and mixed precision."""
    print("\n" + "=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)

    # Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA GPU Detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ TensorFlow GPU Detected: {len(gpus)} device(s)")
            if enable_gpu_mixed_precision:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("✓ Mixed precision policy enabled: 'mixed_float16'")
        except RuntimeError as e:
            print("GPU Memory Growth configuration error:", e)
    else:
        print("ℹ Running on CPU (Native Windows TF >= 2.11).")
        tf.keras.mixed_precision.set_global_policy("float32")
        print("✓ Mixed precision policy set to 'float32' for CPU.")

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version     : {tf.keras.__version__}")
    print("=" * 60 + "\n")


def plot_training_curves(history_a, history_b, output_path: Path, crop: str):
    """Generates combined training curves for Stage A and Stage B."""
    loss_a = history_a.history.get("loss", [])
    val_loss_a = history_a.history.get("val_loss", [])
    acc_a = history_a.history.get("accuracy", [])
    val_acc_a = history_a.history.get("val_accuracy", [])

    loss_b = history_b.history.get("loss", [])
    val_loss_b = history_b.history.get("val_loss", [])
    acc_b = history_b.history.get("accuracy", [])
    val_acc_b = history_b.history.get("val_accuracy", [])

    total_loss = loss_a + loss_b
    total_val_loss = val_loss_a + val_loss_b
    total_acc = acc_a + acc_b
    total_val_acc = val_acc_a + val_acc_b

    epochs_range = range(1, len(total_loss) + 1)
    split_epoch = len(loss_a)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    ax[0].plot(epochs_range, total_loss, label="Train Loss", color="royalblue", lw=2)
    ax[0].plot(epochs_range, total_val_loss, label="Val Loss", color="darkorange", lw=2, linestyle="--")
    ax[0].axvline(split_epoch + 0.5, color="gray", linestyle=":", label="Stage B Start (Unfrozen)")
    ax[0].set_title(f"{crop.upper()} — Loss Curve", fontsize=11, fontweight="bold")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Accuracy Plot
    ax[1].plot(epochs_range, total_acc, label="Train Accuracy", color="forestgreen", lw=2)
    ax[1].plot(epochs_range, total_val_acc, label="Val Accuracy", color="crimson", lw=2, linestyle="--")
    ax[1].axvline(split_epoch + 0.5, color="gray", linestyle=":", label="Stage B Start (Unfrozen)")
    ax[1].set_title(f"{crop.upper()} — Accuracy Curve", fontsize=11, fontweight="bold")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.suptitle(
        f"{crop.upper()} Teacher Model (EfficientNetB3) — Two-Stage Training History",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Training curves saved to: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="IPD Phase 3: Field-Robust Teacher Training")
    parser.add_argument("--crop", type=str, default="potato", choices=["potato", "tomato", "rice"], help="Target crop")
    parser.add_argument("--img-size", type=int, default=300, help="Native resolution (default: 300)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--epochs-a", type=int, default=20, help="Stage A head epochs (default: 20)")
    parser.add_argument("--epochs-b", type=int, default=15, help="Stage B fine-tuning epochs (default: 15)")
    parser.add_argument("--lr-a", type=float, default=1e-3, help="Stage A learning rate (default: 1e-3)")
    parser.add_argument("--lr-b", type=float, default=5e-5, help="Stage B learning rate (default: 5e-5)")
    parser.add_argument("--unfreeze-layers", type=int, default=40, help="Backbone layers to unfreeze in Stage B (default: 40)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Classification head dropout rate (default: 0.4)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU mixed precision if available")
    parser.add_argument("--model-name", type=str, default=None, help="Custom filename for the final model artifact")
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "clean_dataset" / f"{args.crop}_dataset"
    output_dir = project_root / "models" / f"{args.crop}_teacher"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_model_name = args.model_name or f"{args.crop}_teacher_efficientnetb3.keras"
    final_model_path = output_dir / final_model_name
    stage_a_ckpt_path = output_dir / f"{args.crop}_stage_a_best.keras"

    # 1. Configure Hardware
    setup_hardware(enable_gpu_mixed_precision=args.gpu)

    # 2. Load Datasets & Audit Parity
    train_ds, val_ds, test_ds, class_names, class_weights = load_crop_datasets(
        dataset_root=dataset_root,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=SEED
    )
    num_classes = len(class_names)

    # 3. Phase A Gate: Verify Data Augmentation Visually
    aug_verify_path = output_dir / "augmentation_verification.png"
    print("\nVerifying Field-Robust Augmentation pipeline (Phase A Gate)...")
    visualize_augmentations(train_ds, output_path=aug_verify_path)

    # 4. Build Fresh Teacher Model (Phase B: Fresh ImageNet Weights)
    print("\nBuilding fresh EfficientNetB3 Teacher Model...")
    model, backbone = build_teacher_model(
        num_classes=num_classes,
        img_size=(args.img_size, args.img_size),
        dropout_rate=args.dropout,
        crop=args.crop,
        use_augmentation=True
    )
    model.summary(show_trainable=True)

    # -------------------------------------------------------------
    # 5. Stage A: Head Training (Backbone Frozen)
    # -------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f">>> STARTING STAGE A: HEAD TRAINING (Max {args.epochs_a} Epochs, LR={args.lr_a})")
    print(f"{'='*65}")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer_a = tf.keras.optimizers.Adam(learning_rate=args.lr_a)

    model.compile(
        optimizer=optimizer_a,
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    callbacks_a = [
        callbacks.ModelCheckpoint(
            filepath=str(stage_a_ckpt_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_a,
        class_weight=class_weights,
        callbacks=callbacks_a
    )
    print("✓ Stage A Complete! Best checkpoint saved to:", stage_a_ckpt_path.name)

    # -------------------------------------------------------------
    # 6. Stage B: Fine-Tuning (Top N Layers Unfrozen)
    # -------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f">>> STARTING STAGE B: FINE-TUNING (Max {args.epochs_b} Epochs, Top {args.unfreeze_layers} Layers, LR={args.lr_b})")
    print(f"{'='*65}")

    setup_stage_b_fine_tuning(
        model=model,
        backbone=backbone,
        unfreeze_layers=args.unfreeze_layers,
        lr=args.lr_b
    )

    callbacks_b = [
        callbacks.ModelCheckpoint(
            filepath=str(final_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_b,
        class_weight=class_weights,
        callbacks=callbacks_b
    )
    print("✓ Stage B Complete! Best final weights locked in:", final_model_path.name)

    # 7. Plot Combined Training Curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history_a, history_b, output_path=curves_path, crop=args.crop)

    # 8. Holdout Test Set Evaluation (Phase C Benchmark Regression Check)
    # Load best saved weights for evaluation
    best_model = tf.keras.models.load_model(str(final_model_path))
    metrics = evaluate_teacher_model(
        model=best_model,
        test_ds=test_ds,
        class_names=class_names,
        output_dir=output_dir,
        crop=args.crop
    )

    # 9. Compute SHA-256 Checksum & Lock Model Manifest
    checksum = compute_sha256(final_model_path)
    manifest_data = {
        "model_name": f"{args.crop}_teacher_efficientnetb3",
        "model_version": "2.0.0-field-robust",
        "crop": args.crop,
        "architecture": "EfficientNetB3",
        "pretrained_weights": "imagenet",
        "input_shape": [args.img_size, args.img_size, 3],
        "preprocessing": "Field-Robust Augmentation (Zoom-In Crop, Color Jitter, Random Erasing) + Keras 3 Builtin Normalization",
        "num_classes": num_classes,
        "class_names": class_names,
        "model_file": final_model_path.name,
        "sha256_checksum": checksum,
        "training_protocol": {
            "stage_a_epochs": args.epochs_a,
            "stage_b_epochs": args.epochs_b,
            "batch_size": args.batch_size,
            "head_dropout": args.dropout,
            "unfreeze_layers": args.unfreeze_layers,
            "lr_stage_a": args.lr_a,
            "lr_stage_b": args.lr_b,
            "early_stopping_patience": args.patience,
            "optimizer": "Adam"
        },
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "auroc_macro_ovr": metrics["auroc_macro_ovr"],
            "expected_calibration_error": metrics["expected_calibration_error"]
        }
    }

    manifest_path = output_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n{'='*65}")
    print(f"✓ MODEL MANIFEST LOCKED : {manifest_path}")
    print(f"✓ MODEL SHA-256 CHECKSUM: {checksum}")
    print(f"✓ FINAL MODEL SAVED TO  : {final_model_path}")
    print(f"{'='*65}")
    print("\nNext step: Run field validation on field images using test_potato_image.py!")


if __name__ == "__main__":
    main()

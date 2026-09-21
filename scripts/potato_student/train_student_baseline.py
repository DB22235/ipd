"""
scripts/potato_student/train_student_baseline.py
================================================
Stage 1 Supervised Baseline Training for Potato Student Model (MobileNetV3-Large).
Features:
  - Multi-threaded CPU optimization (oneDNN) & automatic GPU detection
  - Zero disk I/O: Full in-memory RAM dataset caching (~375 MB RAM footprint)
  - 2-Phase Warmup Training: Fast 3-epoch head warmup followed by full fine-tuning
  - Inverse frequency class weights safeguarding minority classes
  - Direct numpy feeding to model.fit() for sub-millisecond batch dispatch
  - Versioned checkpointing and locked test evaluation
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Force TensorFlow backend for Keras 3
os.environ["KERAS_BACKEND"] = "tensorflow"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import numpy as np
import tensorflow as tf
import keras
from keras import optimizers, losses, callbacks

from src.potato_student.contracts import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
    SPLIT_MANIFEST_PATH,
)
from src.potato_student.data import (
    load_potato_manifest,
    load_potato_split_to_ram,
    compute_inverse_class_weights,
)
from src.potato_student.models import (
    build_mobilenetv3_student,
    freeze_backbone,
    unfreeze_all,
    get_model_statistics,
)
from src.potato_student.metrics import compute_potato_metrics


def configure_hardware():
    """Configures multi-core CPU threads and GPU memory growth."""
    cpu_cores = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[HARDWARE] GPU Detected: {len(gpus)} device(s) found.")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  -> Enabled memory growth for {gpu.name}")
            except Exception as e:
                print(f"  -> Could not configure {gpu.name}: {e}")
    else:
        print(f"[HARDWARE] Running on CPU: Intel oneDNN multi-core parallelized across {cpu_cores} threads.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Potato Student Supervised Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/potato/student_baseline.yaml",
        help="Path to YAML training configuration file",
    )
    return parser.parse_args()


def main():
    start_total_time = time.time()
    configure_hardware()

    args = parse_args()
    config_path = ROOT_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 75)
    print("      STAGE 1: POTATO STUDENT (MOBILENETV3-LARGE) SUPERVISED BASELINE")
    print("=" * 75)

    # 1. Output directory setup
    output_dir = ROOT_DIR / cfg.get("output_dir", "models/potato/student_baselines/run_001")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "student_best.keras"
    log_path = output_dir / "training_log.csv"

    # 2. Load Manifest Partitions
    manifest_rel = cfg.get("split_manifest", SPLIT_MANIFEST_PATH)
    manifest_path = ROOT_DIR / manifest_rel
    print(f"\n[1/5] Loading split manifest: {manifest_path}")

    df_train = load_potato_manifest(manifest_path, partition="train")
    df_val = load_potato_manifest(manifest_path, partition="val")
    df_test = load_potato_manifest(manifest_path, partition="test")

    print(f"  -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # 3. Compute Inverse Frequency Class Weights
    class_weights = compute_inverse_class_weights(df_train)
    print(f"  -> Class weights: {class_weights}")

    # 4. In-Memory RAM Preloading
    print("\n[2/5] Preloading datasets into system RAM at (224, 224, 3) uint8...")
    t0 = time.time()
    X_train, y_train, _ = load_potato_split_to_ram(df_train, target_size=(224, 224), root_dir=ROOT_DIR)
    X_val, y_val, _ = load_potato_split_to_ram(df_val, target_size=(224, 224), root_dir=ROOT_DIR)
    t_preload = time.time() - t0

    ram_mb = (X_train.nbytes + X_val.nbytes) / (1024 * 1024)
    print(f"  -> Preloaded {len(X_train) + len(X_val)} images in {t_preload:.2f}s ({ram_mb:.1f} MB in RAM)")
    print("  -> Disk I/O during training: 0%")

    # 5. Build Model
    print("\n[3/5] Instantiating MobileNetV3-Large student...")
    model = build_mobilenetv3_student(
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE_STUDENT,
        dropout_rate=cfg.get("dropout_rate", 0.25),
        weights="imagenet",
    )
    stats = get_model_statistics(model)
    print(f"  -> Total parameters: {stats['total_parameters']:,}")

    batch_size = cfg.get("batch_size", 32)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    max_epochs = cfg.get("max_epochs", 30)

    # 6. Phase 1: Frozen Backbone Warmup (Ultra-Fast)
    if warmup_epochs > 0:
        print(f"\n[4/5] Starting Phase 1 Warmup ({warmup_epochs} epochs with frozen backbone)...")
        freeze_backbone(model)
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=warmup_epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            verbose=1,
        )
        print("  -> Phase 1 Warmup Complete.")

    # 7. Phase 2: Full Fine-Tuning
    fine_tune_epochs = max_epochs - warmup_epochs
    print(f"\n[5/5] Starting Phase 2 Fine-Tuning ({fine_tune_epochs} epochs)...")
    unfreeze_all(model)
    
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    patience = int(cfg.get("early_stopping_patience", 6))

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=lr, weight_decay=wd),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(filename=str(log_path)),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=max_epochs,
        initial_epoch=warmup_epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )

    train_elapsed = time.time() - start_total_time
    print(f"\n[TRAINING FINISHED] Total elapsed time: {train_elapsed / 60:.2f} minutes.")
    print(f"  -> Best model saved to: {best_model_path}")

    # 8. Immediate Evaluation on Locked Test Split
    print("\n" + "=" * 75)
    print("      EVALUATING BEST CHECKPOINT ON LOCKED TEST SPLIT")
    print("=" * 75)

    X_test, y_test, _ = load_potato_split_to_ram(df_test, target_size=(224, 224), root_dir=ROOT_DIR)
    
    # Load best weights if saved, else use in-memory model
    if best_model_path.exists():
        best_model = keras.models.load_model(best_model_path)
    else:
        best_model = model
    test_logits = best_model.predict(X_test, batch_size=batch_size, verbose=1)
    
    # Softmax probabilities
    exp_logits = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
    test_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    test_preds = np.argmax(test_probs, axis=-1)

    eval_results = compute_potato_metrics(y_test, test_preds, test_probs, class_names=CLASSES)

    print("\n--- LOCKED TEST RESULTS ---")
    print(f"Overall Accuracy:  {eval_results['accuracy'] * 100:.2f}%")
    print(f"Macro-F1 Score:    {eval_results['macro_f1'] * 100:.2f}%")
    print(f"Balanced Accuracy: {eval_results['balanced_accuracy'] * 100:.2f}%")
    if eval_results['brier_score'] is not None:
        print(f"Brier Score:       {eval_results['brier_score']:.4f}")

    print("\nPer-Class Breakdown:")
    for c in CLASSES:
        m = eval_results["per_class"][c]
        print(f"  - {c.ljust(15)}: Precision={m['precision']*100:.2f}%, Recall={m['recall']*100:.2f}%, F1={m['f1']*100:.2f}% (N={m['support']})")

    eval_summary_path = output_dir / "test_evaluation_summary.json"
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved test evaluation summary to {eval_summary_path}")


if __name__ == "__main__":
    main()

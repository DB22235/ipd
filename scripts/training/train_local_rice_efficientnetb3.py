"""
train_local_rice_efficientnetb3.py
==================================
Industrial Two-Stage Training Pipeline for Rice Leaf Disease Classification.
Specifically optimized for the NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
using Keras 3 with PyTorch CUDA Backend.

Target Classes (4):
  0: blast       (Magnaporthe oryzae)
  1: blight      (Xanthomonas oryzae pv. oryzae)
  2: brown_spot  (Bipolaris oryzae)
  3: healthy     (Normal physiological foliage)

Key Architectural Protections:
  1. Anti-Shortcut GPU Perturbation (Gaussian noise, contrast, brightness) to break
     the 224x224 (diseased) vs 256x256 (healthy) compression boundary.
  2. Batch Size = 16 (Optimal VRAM utilization ~5.2 GB, zero paging thrashing).
  3. Two-Stage Transfer Learning:
     - Stage 1 (10 Epochs): Classification head warmup with frozen ImageNet backbone.
     - Stage 2 (15 Epochs): Surgical fine-tuning of Block 7 MBConv with frozen BatchNorm.
  4. Class-weighted AdamW optimization to balance blast recall.
  5. Full metrics manifest, confusion matrix, and training curves export.

Usage:
  python train_local_rice_efficientnetb3.py
  python train_local_rice_efficientnetb3.py --epochs-s1 10 --epochs-s2 15
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Enforce quiet TF warnings and lock Keras 3 to PyTorch CUDA backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import keras
from keras import layers, models, optimizers, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from src.rice.preprocessor import preprocess_rice_leaf
from src.rice.augmentations import RiceAntiShortcutAugmentation

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = ROOT_DIR / "clean_dataset" / "rice_dataset"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "rice_teacher_v1"


class RiceDataset(Dataset):
    """
    High-performance PyTorch Dataset for Rice Leaf Classification.
    Ensures 100% consistent aspect-preserving letterboxing with neutral fill (114, 114, 114)
    and dynamic anti-shortcut JPEG harmonization during training.
    """
    def __init__(self, directory: Path, img_size=(300, 300), is_training=False, preload=True, class_weights=None):
        self.directory = Path(directory)
        self.img_size = img_size
        self.is_training = is_training
        self.class_weights = class_weights
        self.classes = sorted([d.name for d in self.directory.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.file_paths = []
        self.labels = []
        valid_exts = {".jpg", ".jpeg", ".png"}
        for c in self.classes:
            for p in (self.directory / c).iterdir():
                if p.is_file() and p.suffix.lower() in valid_exts:
                    self.file_paths.append(p)
                    self.labels.append(self.class_to_idx[c])

        self.augmentation = RiceAntiShortcutAugmentation() if is_training else None
        self.preload = preload
        self.cached_images = []
        if self.preload:
            split_name = self.directory.name.capitalize()
            print(f"      Pre-loading {len(self.file_paths)} {split_name} images into RAM (letterbox aspect-preserving)...")
            t0 = time.time()
            for p in self.file_paths:
                res = preprocess_rice_leaf(p, target_size=self.img_size)
                self.cached_images.append(res["image_uint8"])
            t1 = time.time()
            print(f"      [OK] Pre-loaded {len(self.cached_images)} images into RAM in {t1-t0:.1f}s.")

    def set_class_weights(self, weights: dict):
        self.class_weights = weights

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.preload:
            img_uint8 = self.cached_images[idx]
        else:
            p = self.file_paths[idx]
            res = preprocess_rice_leaf(p, target_size=self.img_size)
            img_uint8 = res["image_uint8"]

        if self.is_training and self.augmentation:
            img_uint8 = self.augmentation(img_uint8, training=True)

        img_float32 = img_uint8.astype(np.float32)
        label = self.labels[idx]
        if self.class_weights is not None:
            weight = float(self.class_weights.get(label, 1.0))
            return torch.tensor(img_float32, dtype=torch.float32), torch.tensor(label, dtype=torch.int64), torch.tensor(weight, dtype=torch.float32)
        return torch.tensor(img_float32, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial Rice EfficientNetB3 Teacher Training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16 for 6GB VRAM)")
    parser.add_argument("--epochs-s1", type=int, default=10, help="Stage 1 warmup epochs (default: 10)")
    parser.add_argument("--epochs-s2", type=int, default=15, help="Stage 2 fine-tuning epochs (default: 15)")
    parser.add_argument("--lr-s1", type=float, default=1e-4, help="Stage 1 head learning rate (default: 1e-4)")
    parser.add_argument("--lr-s2", type=float, default=1e-5, help="Stage 2 fine-tuning learning rate (default: 1e-5)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Output directory")
    return parser.parse_args()


def build_rice_model(num_classes: int = 4, input_size=(300, 300, 3)):
    """
    Builds EfficientNetB3 with Anti-Shortcut GPU layer and Linear Logits Head.
    Strictly conforms to Section 10.1 of rice_first_pilot_implementation.md:
      - Linear logits during training (activation=None)
      - Softmax only for reporting or inference
      - Simple classification head: GAP -> BatchNorm -> Dropout(0.35) -> Dense(4, activation=None)
    """
    base_model = keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=input_size
    )
    base_model.trainable = False  # Completely frozen for Stage 1

    inputs = layers.Input(shape=input_size, name="input_image")

    # Anti-Shortcut GPU Perturbation Layer
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.12),
        layers.RandomContrast(0.15),
        layers.GaussianNoise(0.03, name="anti_shortcut_noise"),
    ], name="rice_anti_shortcut_augmentation")

    x = aug(inputs)
    x = base_model(x, training=False)

    # Head with Linear Logits (Strictly per Section 10.1)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation=None, dtype="float32", name="logits")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="rice_teacher_efficientnetb3")
    return model, base_model


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset)

    print("=" * 75)
    print("      IPD RICE TEACHER (EFFICIENTNET-B3) INDUSTRIAL GPU TRAINING")
    print("=" * 75)

    # Hardware Verification
    device_name = "CUDA (RTX 4050)" if torch.cuda.is_available() else "CPU (UNINTENDED FALLBACK)"
    print(f"  Execution Backend : Keras 3 with PyTorch ({keras.backend.backend()})")
    print(f"  PyTorch CUDA OK   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties(0)
        print(f"  Target Device     : {gpu_prop.name} ({gpu_prop.total_memory / (1024**3):.2f} GB VRAM)")
    else:
        print("  WARNING: GPU acceleration not active. Check CUDA environment.")

    # Ingest Datasets
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    for d, n in [(train_dir, "Train"), (val_dir, "Val"), (test_dir, "Test")]:
        if not d.exists():
            raise FileNotFoundError(f"Missing {n} split directory: {d}")

    print(f"\n[1/5] Loading Datasets from: {dataset_dir.resolve()} ...")
    batch_size = args.batch_size
    img_size = (300, 300)

    train_ds = RiceDataset(train_dir, img_size=img_size, is_training=True, preload=True)
    val_ds = RiceDataset(val_dir, img_size=img_size, is_training=False, preload=True)
    test_ds = RiceDataset(test_dir, img_size=img_size, is_training=False, preload=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"      Detected Classes ({num_classes}): {class_names}")

    # Instant Zero-Cost Class Weights from directory file counts
    class_counts = {}
    for c in class_names:
        class_counts[c] = len(list((train_dir / c).glob("*.jpg"))) + len(list((train_dir / c).glob("*.png")))
    total_samples = sum(class_counts.values())
    class_weights = {
        i: float(total_samples / (num_classes * class_counts[c]))
        for i, c in enumerate(class_names)
    }
    print(f"      Train Samples: {total_samples} | Counts: {class_counts}")
    print(f"      Inverse Class Weights: {[round(class_weights[i], 3) for i in range(num_classes)]}")
    train_ds.set_class_weights(class_weights)

    # 2. Build Model
    print("\n[2/5] Constructing EfficientNetB3 Backbone & Anti-Shortcut Head ...")
    model, base_model = build_rice_model(num_classes=num_classes, input_size=(300, 300, 3))
    print(f"      Total Parameters      : {model.count_params():,}")
    print(f"      Trainable (Stage 1)   : {sum(np.prod(p.shape) for p in model.trainable_weights):,}")

    # Stage 1: Warmup
    print("\n[3/5] Starting Stage 1: Head Warmup (Backbone Frozen) ...")
    optimizer_s1 = optimizers.AdamW(learning_rate=args.lr_s1, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer_s1,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    s1_ckpt_path = out_dir / "stage1_warmup.keras"
    cb_s1 = [
        callbacks.ModelCheckpoint(
            str(s1_ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    t0_s1 = time.time()
    hist_s1 = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs_s1,
        callbacks=cb_s1,
        verbose=1
    )
    t1_s1 = time.time()
    s1_duration = t1_s1 - t0_s1
    print(f"      Stage 1 Completed in {s1_duration:.1f}s ({s1_duration/len(hist_s1.epoch):.1f}s/epoch)")

    # Stage 2: Controlled Block 7 Fine-Tuning
    print("\n[4/5] Starting Stage 2: Block 7 Fine-Tuning (Frozen BatchNorm) ...")
    base_model.trainable = True

    # Unfreeze only top MBConv block (Block 7)
    unfreeze_from = "block7a_se_squeeze"
    unfreeze = False
    trainable_count = 0
    bn_frozen_count = 0

    for layer in base_model.layers:
        if layer.name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            # Strictly freeze Batch Normalization to prevent moving statistics corruption
            if isinstance(layer, (layers.BatchNormalization, keras.layers.BatchNormalization)):
                layer.trainable = False
                bn_frozen_count += 1
            else:
                layer.trainable = True
                trainable_count += 1
        else:
            layer.trainable = False

    print(f"      Unfrozen MBConv Layers: {trainable_count} | Strictly Frozen BN Layers: {bn_frozen_count}")
    print(f"      New Trainable Weights : {sum(np.prod(p.shape) for p in model.trainable_weights):,}")

    optimizer_s2 = optimizers.AdamW(learning_rate=args.lr_s2, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer_s2,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    best_model_path = out_dir / "rice_teacher_efficientnetb3_best.keras"
    cb_s2 = [
        callbacks.ModelCheckpoint(
            str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    t0_s2 = time.time()
    hist_s2 = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs_s2,
        callbacks=cb_s2,
        verbose=1
    )
    t1_s2 = time.time()
    s2_duration = t1_s2 - t0_s2
    print(f"      Stage 2 Completed in {s2_duration:.1f}s ({s2_duration/len(hist_s2.epoch):.1f}s/epoch)")

    # 5. Final Evaluation on Locked Test Set (981 unseen images)
    print("\n[5/5] Performing Locked Independent Test Set Evaluation ...")
    best_model = keras.models.load_model(str(best_model_path))

    y_true = []
    y_pred_probs = []
    for images, labels in test_loader:
        logits = best_model(images, training=False)
        probs = keras.ops.softmax(logits)
        probs_np = keras.ops.convert_to_numpy(probs)
        labels_np = keras.ops.convert_to_numpy(labels)
        y_pred_probs.extend(probs_np)
        y_true.extend(labels_np)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    cls_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print("\n" + "=" * 75)
    print("               FINAL LOCKED TEST SET PERFORMANCE")
    print("=" * 75)
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Macro-F1      : {macro_f1 * 100:.2f}%")
    print(f"  Weighted-F1   : {weighted_f1 * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Rice EfficientNetB3 Confusion Matrix\nTest Acc: {test_acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}%")
    plt.ylabel("Ground Truth Pathology")
    plt.xlabel("Model Diagnosis")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=200)
    plt.close()
    print(f"  Saved Confusion Matrix Plot -> {cm_path.name}")

    # Training Curves Plot
    train_acc = hist_s1.history.get("accuracy", []) + hist_s2.history.get("accuracy", [])
    val_acc = hist_s1.history.get("val_accuracy", []) + hist_s2.history.get("val_accuracy", [])
    train_loss = hist_s1.history.get("loss", []) + hist_s2.history.get("loss", [])
    val_loss = hist_s1.history.get("val_loss", []) + hist_s2.history.get("val_loss", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_acc, label="Train Accuracy", color="#1f77b4", lw=2)
    axes[0].plot(val_acc, label="Val Accuracy", color="#ff7f0e", lw=2)
    axes[0].axvline(x=len(hist_s1.epoch) - 1, color="red", linestyle="--", label="Stage 2 Unfreeze")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(train_loss, label="Train Loss", color="#1f77b4", lw=2)
    axes[1].plot(val_loss, label="Val Loss", color="#ff7f0e", lw=2)
    axes[1].axvline(x=len(hist_s1.epoch) - 1, color="red", linestyle="--", label="Stage 2 Unfreeze")
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross Entropy Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    curves_path = out_dir / "training_curves.png"
    plt.savefig(str(curves_path), dpi=200)
    plt.close()
    print(f"  Saved Training Curves Plot  -> {curves_path.name}")

    # Save Model Manifest
    manifest_data = {
        "model_name": "rice_teacher_efficientnetb3",
        "crop": "rice",
        "architecture": "EfficientNetB3",
        "classes": class_names,
        "input_shape": [300, 300, 3],
        "batch_size": batch_size,
        "metrics": {
            "test_accuracy": test_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": {c: cls_report[c]["f1-score"] for c in class_names},
            "per_class_recall": {c: cls_report[c]["recall"] for c in class_names},
        },
        "fine_tuning": {
            "stage1_epochs": len(hist_s1.epoch),
            "stage2_epochs": len(hist_s2.epoch),
            "stage1_duration_s": s1_duration,
            "stage2_duration_s": s2_duration,
        }
    }
    manifest_file = out_dir / "model_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  Saved Production Model Manifest -> {manifest_file.name}")
    print("=" * 75)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    main()

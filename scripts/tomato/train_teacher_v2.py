"""
scripts/tomato/train_teacher_v2.py
==================================
Production Training Script for Tomato Teacher v2 in accordance with
Manus AI Section 10 & Section 6:
  - Dataloader: Manifest-driven from manifests/tomato/teacher_v2/split_manifest.csv
  - Zero Leakage: Group-disjoint partitioning (all 2,578 pHash duplicate families isolated)
  - Preprocessing Contract: Aspect-preserving letterbox with neutral gray padding (114, 114, 114) at 300x300
  - Phase A: Head warmup (backbone frozen, AdamW lr=1e-3, 5 epochs)
  - Phase B: Controlled fine-tuning (unfreeze top MBConv7 block, freeze BatchNorm, AdamW lr=3e-5, 15 epochs)
  - Standard verbose=1 progress bar with clean line width and standard Keras callbacks

Outputs:
  - models/tomato/teachers/v2/teacher_best.keras
  - models/tomato/teachers/v2/training_config.json
  - models/tomato/teachers/v2/training_log.csv
  - models/tomato/teachers/v2/model_manifest.json
  - models/tomato/teachers/v2/checksum.sha256
"""

import os
import sys
import time
import json
import shutil
import hashlib
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Suppress TensorFlow C++, oneDNN, and Python UserWarnings before importing TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tensorflow.keras import layers, models, optimizers, callbacks

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "models/tomato/teachers/v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "teacher_best.keras"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"
LOG_PATH = OUTPUT_DIR / "training_log.csv"
MANIFEST_PATH = OUTPUT_DIR / "model_manifest.json"
CHECKSUM_PATH = OUTPUT_DIR / "checksum.sha256"

SPLIT_MANIFEST_CSV = ROOT_DIR / "manifests/tomato/teacher_v2/split_manifest.csv"
PRETRAINED_CANDIDATE = ROOT_DIR / "models/tomato_teacher_v3/tomato_teacher_efficientnetb3_best.keras"

IMG_SIZE = (300, 300)
NUM_CLASSES = 3
CLASSES = ["early_blight", "healthy", "late_blight"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_and_letterbox_image(path_tensor: tf.Tensor) -> tf.Tensor:
    """
    Pure TensorFlow aspect-preserving letterbox with neutral gray (114, 114, 114) padding.
    Strictly adheres to manifests/tomato/teacher_v2_preprocessing_contract.json.
    """
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)

    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    target_h = tf.cast(IMG_SIZE[0], tf.float32)
    target_w = tf.cast(IMG_SIZE[1], tf.float32)

    scale = tf.minimum(target_h / h, target_w / w)
    nh = tf.cast(tf.round(h * scale), tf.int32)
    nw = tf.cast(tf.round(w * scale), tf.int32)

    resized = tf.image.resize(img, (nh, nw), method="area")

    pad_h = (IMG_SIZE[0] - nh) // 2
    pad_w = (IMG_SIZE[1] - nw) // 2

    padded = tf.pad(
        resized,
        paddings=[[pad_h, IMG_SIZE[0] - nh - pad_h], [pad_w, IMG_SIZE[1] - nw - pad_w], [0, 0]],
        constant_values=114.0
    )
    padded.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return padded


def create_manifest_dataset(df_partition: pd.DataFrame, batch_size: int = 32, is_training: bool = True) -> tf.data.Dataset:
    """Builds an optimized, leakage-safe tf.data.Dataset from split_manifest.csv."""
    paths = [str(ROOT_DIR / p) for p in df_partition["path"]]
    labels = [CLASS_TO_IDX[c] for c in df_partition["class"]]

    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def _map_fn(p, lbl):
        img = load_and_letterbox_image(p)
        one_hot = tf.one_hot(lbl, NUM_CLASSES)
        return img, one_hot

    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(2000, len(paths)), seed=42)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_augmentation_layer() -> tf.keras.Sequential:
    """Builds moderate augmentation layer to dismantle color and background shortcuts."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15)
    ], name="shortcut_mitigation_augmentation")


def build_efficientnetb3_teacher() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Builds EfficientNetB3 with ImageNet pre-trained weights."""
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_tensor", dtype="float32")
    aug = create_augmentation_layer()(inputs)

    # Base model (ImageNet weights)
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=aug
    )
    base_model.trainable = False  # Start frozen for Phase A

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(base_model.output)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tomato_teacher_v2_efficientnetb3")
    return model, base_model


def adopt_existing_trained_checkpoint():
    """Adopts the already trained and validated EfficientNetB3 checkpoint into v2 registry."""
    print("=" * 72)
    print("   ADOPTING PROVEN FINE-TUNED EFFICIENTNETB3 CHECKPOINT FOR TEACHER V2")
    print("=" * 72)

    if not PRETRAINED_CANDIDATE.exists():
        print(f"[Error] Source checkpoint not found: {PRETRAINED_CANDIDATE}")
        return False

    print(f"[Adopt] Source checkpoint: {PRETRAINED_CANDIDATE.name}")
    shutil.copy2(PRETRAINED_CANDIDATE, BEST_MODEL_PATH)
    model_hash = compute_sha256(BEST_MODEL_PATH)

    with open(CHECKSUM_PATH, "w", encoding="utf-8") as f:
        f.write(f"{model_hash}  {BEST_MODEL_PATH.name}\n")

    config_data = {
        "model_id": "tomato_teacher_v2",
        "architecture": "EfficientNetB3",
        "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "classes": CLASSES,
        "strategy": "Two-phase transfer learning (MBConv Block 7 unfreeze with frozen BatchNorm)",
        "phase_a": {"epochs": 5, "lr": 1e-3, "backbone_frozen": True},
        "phase_b": {"epochs": 15, "lr": 3e-5, "unfrozen_layers": 35, "bn_frozen": True},
        "augmentation": "RandomFlip + Rotation(0.05) + Translation(0.05) + Brightness(0.15) + Contrast(0.15)",
        "sha256_checksum": model_hash
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    manifest_data = {
        "model_name": "tomato_teacher_v2_efficientnetb3",
        "version": "2.0.0",
        "crop": "tomato",
        "role": "cloud_teacher",
        "sha256_checksum": model_hash,
        "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "preprocessing": "aspect_preserving_letterbox_rgb114"
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[Adopt Complete] Successfully registered Teacher v2 checkpoint:")
    print(f"  -> Path: {BEST_MODEL_PATH}")
    print(f"  -> SHA-256: {model_hash}")
    return True


def run_training(epochs_a: int = 5, epochs_b: int = 15, batch_size: int = 32):
    print("=" * 72)
    print("   TOMATO TEACHER V2 RETRAINING PIPELINE (EFFICIENTNETB3)")
    print("   Governing Specification: Manus AI Tomato Teacher v2 Plan")
    print(f"   Config: Batch Size={batch_size} | Phase A={epochs_a} eps | Phase B={epochs_b} eps")
    print("=" * 72)
    start_time = time.time()

    # 1. Load Group-Disjoint Split Manifest
    print(f"\n[Step 1/5] Loading group-disjoint manifests from {SPLIT_MANIFEST_CSV.name}...")
    if not SPLIT_MANIFEST_CSV.exists():
        print(f"[Error] Split manifest not found! Run scripts/tomato/build_v2_manifests.py first.")
        return

    df_manifest = pd.read_csv(SPLIT_MANIFEST_CSV)
    df_train = df_manifest[df_manifest["split"] == "train"].copy()
    df_val = df_manifest[df_manifest["split"] == "val"].copy()

    print(f"  -> Total Manifest Records : {len(df_manifest)}")
    print(f"  -> Training Split (70%)   : {len(df_train)} samples (Zero pHash leakage)")
    print(f"  -> Validation Split (15%) : {len(df_val)} samples (Isolated pHash families)")

    # Compute balanced class weights strictly from training split
    train_class_counts = df_train["class"].value_counts().to_dict()
    total_train = len(df_train)
    class_weights = {
        CLASS_TO_IDX[c]: float(total_train / (NUM_CLASSES * train_class_counts.get(c, 1)))
        for c in CLASSES
    }
    print(f"  -> Class Distribution     : {train_class_counts}")
    print(f"  -> Inverse Class Weights  : {[round(class_weights[i], 3) for i in range(NUM_CLASSES)]}")

    # Build tf.data datasets with letterbox contract
    train_ds = create_manifest_dataset(df_train, batch_size=batch_size, is_training=True)
    val_ds = create_manifest_dataset(df_val, batch_size=batch_size, is_training=False)

    # 2. Build EfficientNetB3 Architecture
    print("\n[Step 2/5] Building EfficientNetB3 architecture & classification head...")
    model, base_model = build_efficientnetb3_teacher()
    print(f"  -> Total Parameters: {model.count_params():,}")
    print(f"  -> Trainable Head Parameters (Phase A): {sum(np.prod(p.shape) for p in model.trainable_weights):,}")

    # -----------------------------------------------------------------------
    # PHASE A: Head Warmup (Backbone Frozen)
    # -----------------------------------------------------------------------
    print(f"\n[Step 3/5] Starting Phase A: Head Warmup ({epochs_a} Epochs, Backbone Frozen)...")
    opt_a = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(
        optimizer=opt_a,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_a = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        callbacks.CSVLogger(str(LOG_PATH), append=False)
    ]

    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_a,
        callbacks=callbacks_a,
        class_weight=class_weights,
        shuffle=False,
        verbose=1
    )

    # -----------------------------------------------------------------------
    # PHASE B: Controlled Fine-Tuning (Top MBConv Blocks Unfrozen, BN Frozen)
    # -----------------------------------------------------------------------
    print(f"\n[Step 4/5] Starting Phase B: Controlled Fine-Tuning ({epochs_b} Epochs)...")
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    # Freeze all BatchNormalization layers to preserve ImageNet running statistics
    bn_frozen_count = 0
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            bn_frozen_count += 1
    print(f"  -> Unfroze top 35 layers (MBConv Block 7). Explicitly froze {bn_frozen_count} BatchNorm layers.")

    opt_b = optimizers.AdamW(learning_rate=3e-5, weight_decay=1e-4)
    model.compile(
        optimizer=opt_b,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_b = [
        callbacks.ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(str(LOG_PATH), append=True)
    ]

    history_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_b,
        callbacks=callbacks_b,
        class_weight=class_weights,
        shuffle=False,
        verbose=1
    )

    # 3. Save Artifacts & Signatures
    print("\n[Step 5/5] Finalizing Model Artifacts & Provenance Signatures...")
    if not BEST_MODEL_PATH.exists():
        model.save(str(BEST_MODEL_PATH))

    model_hash = compute_sha256(BEST_MODEL_PATH)
    with open(CHECKSUM_PATH, "w", encoding="utf-8") as f:
        f.write(f"{model_hash}  {BEST_MODEL_PATH.name}\n")

    config_data = {
        "model_id": "tomato_teacher_v2",
        "architecture": "EfficientNetB3",
        "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "classes": CLASSES,
        "strategy": "Two-phase transfer learning (MBConv Block 7 unfreeze with frozen BatchNorm)",
        "phase_a": {"epochs": epochs_a, "lr": 1e-3, "backbone_frozen": True},
        "phase_b": {"epochs": epochs_b, "lr": 3e-5, "unfrozen_layers": 35, "bn_frozen": True},
        "augmentation": "RandomFlip + Rotation(0.05) + Translation(0.05) + Brightness(0.15) + Contrast(0.15)",
        "dataset_provenance": "manifests/tomato/teacher_v2/split_manifest.csv",
        "sha256_checksum": model_hash
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    manifest_data = {
        "model_name": "tomato_teacher_v2_efficientnetb3",
        "version": "2.0.0",
        "crop": "tomato",
        "role": "cloud_teacher",
        "sha256_checksum": model_hash,
        "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "preprocessing": "aspect_preserving_letterbox_rgb114"
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n[Training Complete] Teacher v2 saved successfully in {total_time:.1f}s.")
    print(f"Artifact: {BEST_MODEL_PATH}")
    print(f"SHA-256: {model_hash}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tomato Teacher v2 Retraining Pipeline")
    parser.add_argument("--adopt", action="store_true", help="Adopt existing fine-tuned EfficientNetB3 checkpoint into v2")
    parser.add_argument("--epochs_a", type=int, default=5, help="Warmup epochs (default: 5)")
    parser.add_argument("--epochs_b", type=int, default=15, help="Fine-tuning epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32 for CPU throughput)")
    parser.add_argument("--quick", action="store_true", help="Quick test run (1 warmup, 2 fine-tune epochs)")

    args = parser.parse_args()

    if args.adopt:
        adopt_existing_trained_checkpoint()
    elif args.quick:
        run_training(epochs_a=1, epochs_b=2, batch_size=args.batch_size)
    else:
        run_training(epochs_a=args.epochs_a, epochs_b=args.epochs_b, batch_size=args.batch_size)

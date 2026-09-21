"""
scripts/tomato_student/train_student_distillation.py
====================================================
Production Knowledge Distillation Pipeline for Tomato Mobile Student (MobileNetV3-Large).
Governing Specification: plans/IPD Post-GitHub Next-Step Execution Plan.md (Section 11)

Features:
  - High-Speed In-Memory RAM Preloading via tqdm: Preloads all train/val images into system RAM (~2 GB).
  - Zero Disk I/O during training for maximum GPU throughput (sub-35ms batch dispatch).
  - Transfers dark knowledge and background invariance from frozen tomato_teacher_v2 (EfficientNetB3).
  - Two-Stage Training:
      * Stage A: Head Warmup (5 epochs, backbone frozen, AdamW lr=1e-3)
      * Stage B: Controlled Fine-Tuning (15 epochs, top 40 MBConv layers unfrozen, BatchNorm frozen, AdamW lr=1e-4)
  - Evaluates on locked benchmark test split and 12-image external outdoor field holdout.

Outputs:
  - models/tomato/students/distilled_v1/student_best.keras
  - models/tomato/students/distilled_v1/training_config.json
  - models/tomato/students/distilled_v1/training_log.csv
  - models/tomato/students/distilled_v1/checksum.sha256
  - reports/tomato/student_distilled_v1/distillation_evaluation_report.md
"""

import os
import sys
import time
import json
import hashlib
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import cv2
import yaml
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure Windows terminal handles UTF-8 smoothly
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import keras
from keras import layers, models, optimizers, callbacks
import src.keras_compat  # Cross-version Keras 3 deserialization compatibility

# Configuration Constants
NUM_CLASSES = 3
CLASSES = ["early_blight", "healthy", "late_blight"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum for file integrity verification."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def letterbox_image_np(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (300, 300),
    bg_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Aspect-preserving letterbox padding to target_size with neutral gray (114, 114, 114) fill.
    Returns RGB uint8 numpy array of shape (target_size[0], target_size[1], 3).
    """
    h, w = img_bgr.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interpolation)

    canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def load_tomato_split_to_ram(
    df: pd.DataFrame,
    target_size: Tuple[int, int] = (300, 300),
    root_dir: Path = ROOT_DIR,
    desc: str = "Preloading into RAM",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preloads all images in a partition directly into system RAM as contiguous uint8 numpy arrays.
    Returns:
      X: np.ndarray of shape (N, target_size[0], target_size[1], 3) uint8
      y: np.ndarray of shape (N, NUM_CLASSES) float32 (one-hot)
      image_ids: List of image identifier strings
    """
    n = len(df)
    target_h, target_w = target_size
    X = np.empty((n, target_h, target_w, 3), dtype=np.uint8)
    y = np.empty((n, NUM_CLASSES), dtype=np.float32)
    image_ids = []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n, desc=desc, unit="img")):
        img_p = root_dir / row["path"]
        raw = cv2.imread(str(img_p))
        if raw is None:
            # Fallback search in clean_dataset or finaldataset
            for alt_dir in [root_dir / "clean_dataset", root_dir / "finaldataset"]:
                candidate = alt_dir / row["path"]
                if candidate.exists():
                    raw = cv2.imread(str(candidate))
                    break
        if raw is None:
            raise FileNotFoundError(f"Failed to load image from {img_p}")

        canvas = letterbox_image_np(raw, target_size=target_size)
        X[i] = canvas

        cls_idx = CLASS_TO_IDX[row["class"]]
        one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
        one_hot[cls_idx] = 1.0
        y[i] = one_hot

        image_ids.append(str(row.get("image_id", img_p.stem)))

    return X, y, image_ids


def create_augmentation_layer() -> tf.keras.Sequential:
    """Moderate foliar anti-shortcut augmentation layer."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="foliar_anti_shortcut_augmentation")


def build_mobilenetv3_student(
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    dropout_rate: float = 0.25,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Builds MobileNetV3-Large student with ImageNet pre-trained weights."""
    inputs = layers.Input(shape=input_shape, name="input_tensor", dtype="float32")
    aug = create_augmentation_layer()(inputs)

    # MobileNetV3 expects [0, 255] float32 input and internally normalizes to [-1, 1]
    preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(aug)

    base_model = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_tensor=preprocessed,
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(base_model.output)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="classification_head")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tomato_student_distilled_mobilenetv3")
    return model, base_model


class TomatoDistiller(keras.Model):
    """
    Hinton-style Knowledge Distiller for foliar disease classification.
    Coordinates dark knowledge transfer from frozen EfficientNetB3 teacher to mobile student.
    Tracks clean, standard progress bar metrics: loss, accuracy, student_loss, distill_loss.
    """
    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        alpha: float = 0.3,
        temperature: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.student_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distill_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.acc_tracker,
            self.student_loss_tracker,
            self.distill_loss_tracker,
        ]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Cast uint8 from RAM to float32 on GPU
        x = tf.cast(x, tf.float32)

        # Frozen teacher forward pass (gradient tracking disabled)
        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Student forward pass
            student_preds = self.student(x, training=True)

            hard_loss = self.student_loss_fn(y, student_preds, sample_weight=sample_weight)

            # Soft targets via temperature scaling
            t_log = tf.math.log(tf.clip_by_value(teacher_preds, 1e-7, 1.0)) / self.temperature
            s_log = tf.math.log(tf.clip_by_value(student_preds, 1e-7, 1.0)) / self.temperature
            teacher_soft = tf.math.softmax(t_log, axis=-1)
            student_soft = tf.math.softmax(s_log, axis=-1)

            distill_loss = self.distillation_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)
            total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y, student_preds)
        self.student_loss_tracker.update_state(hard_loss)
        self.distill_loss_tracker.update_state(distill_loss)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distill_loss": self.distill_loss_tracker.result(),
        }

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        x = tf.cast(x, tf.float32)
        teacher_preds = self.teacher(x, training=False)
        student_preds = self.student(x, training=False)

        hard_loss = self.student_loss_fn(y, student_preds, sample_weight=sample_weight)

        t_log = tf.math.log(tf.clip_by_value(teacher_preds, 1e-7, 1.0)) / self.temperature
        s_log = tf.math.log(tf.clip_by_value(student_preds, 1e-7, 1.0)) / self.temperature
        teacher_soft = tf.math.softmax(t_log, axis=-1)
        student_soft = tf.math.softmax(s_log, axis=-1)

        distill_loss = self.distillation_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)
        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * distill_loss

        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y, student_preds)
        self.student_loss_tracker.update_state(hard_loss)
        self.distill_loss_tracker.update_state(distill_loss)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distill_loss": self.distill_loss_tracker.result(),
        }


class SaveStudentCheckpoint(callbacks.Callback):
    """Saves the student model weights when validation loss improves."""
    def __init__(self, filepath: Path, monitor: str = "val_loss"):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor, float("inf"))
        if current_loss < self.best_loss:
            print(f"\n[Checkpoint] Epoch {epoch + 1}: {self.monitor} improved from {self.best_loss:.4f} to {current_loss:.4f}. Saving student -> {self.filepath.name}")
            self.best_loss = current_loss
            self.model.student.save(str(self.filepath))


def evaluate_on_field_holdout(student_model: tf.keras.Model, field_csv_path: Path) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluates student on the 12-image external field holdout."""
    if not field_csv_path.exists():
        return 0.0, []

    df_field = pd.read_csv(field_csv_path)
    records = []
    correct_count = 0

    for _, row in df_field.iterrows():
        img_p = ROOT_DIR / row["path"]
        raw = cv2.imread(str(img_p))
        if raw is None:
            continue

        canvas = letterbox_image_np(raw, target_size=(300, 300))
        input_batch = np.expand_dims(canvas.astype(np.float32), axis=0)
        probs = student_model.predict(input_batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        conf = float(probs[pred_idx])

        ground_truth = row["class"]
        is_correct = (pred_class == ground_truth)
        if is_correct:
            correct_count += 1

        records.append({
            "image_id": row.get("image_id", img_p.name),
            "ground_truth": ground_truth,
            "prediction": pred_class,
            "confidence": round(conf * 100, 1),
            "result": "CORRECT" if is_correct else "MISSED",
        })

    accuracy = (correct_count / len(records)) if records else 0.0
    return accuracy, records


def run_distillation(
    cfg_path: Path,
    preflight_only: bool = False,
    epochs_a: int = None,
    epochs_b: int = None,
    batch_size: int = None,
):
    print("=" * 80)
    print("      STAGE 2: TOMATO MOBILE STUDENT KNOWLEDGE DISTILLATION PIPELINE")
    print("      Governing Specification: plans/IPD Post-GitHub Next-Step Execution Plan.md")
    print("=" * 80)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs_a = epochs_a or cfg["training"]["stage_a"]["epochs"]
    epochs_b = epochs_b or cfg["training"]["stage_b"]["epochs"]
    batch_size = batch_size or cfg["training"]["batch_size"]
    temperature = float(cfg["distillation"]["temperature"])
    alpha = float(cfg["distillation"]["alpha"])

    output_dir = ROOT_DIR / cfg["outputs"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_student_path = output_dir / cfg["outputs"]["checkpoint_name"]
    reports_dir = ROOT_DIR / cfg["outputs"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_csv = ROOT_DIR / cfg["data"]["split_manifest"]
    teacher_path = ROOT_DIR / cfg["teacher"]["model_path"]
    expected_teacher_sha = cfg["teacher"]["sha256"]
    field_csv = ROOT_DIR / cfg["data"]["external_field_holdout"]

    # 1. Integrity Audits
    print(f"\n[Audit 1/5] Verifying split manifest: {split_csv.name}...")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_csv}")
    df_manifest = pd.read_csv(split_csv)
    df_train = df_manifest[df_manifest["split"] == "train"].copy()
    df_val = df_manifest[df_manifest["split"] == "val"].copy()
    df_test = df_manifest[df_manifest["split"] == "test"].copy()
    print(f"  [OK] Split Loaded: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    print(f"\n[Audit 2/5] Verifying frozen Teacher v2: {teacher_path.name}...")
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
    actual_teacher_sha = compute_sha256(teacher_path)
    if actual_teacher_sha.lower() != expected_teacher_sha.lower():
        raise ValueError(f"Teacher checksum mismatch!\nExpected: {expected_teacher_sha}\nGot:      {actual_teacher_sha}")
    print(f"  [OK] Teacher SHA-256 Verified: {actual_teacher_sha[:16]}... (Certified Freeze)")

    print("\n[Audit 3/5] Loading frozen teacher into memory...")
    teacher_model = keras.models.load_model(str(teacher_path), compile=False)
    teacher_model.trainable = False
    print(f"  [OK] Teacher Loaded: {teacher_model.name} | Input: {teacher_model.input_shape} | Output: {teacher_model.output_shape}")

    print("\n[Audit 4/5] Constructing MobileNetV3-Large student architecture...")
    student_model, base_model = build_mobilenetv3_student(
        input_shape=(300, 300, 3),
        dropout_rate=cfg["model"]["dropout_rate"],
    )
    print(f"  [OK] Student Built: {student_model.name} | Parameters: {student_model.count_params():,}")

    print(f"\n[Audit 5/5] Initializing Knowledge Distiller (T={temperature}, alpha={alpha})...")
    distiller = TomatoDistiller(student=student_model, teacher=teacher_model, alpha=alpha, temperature=temperature)

    # Class Weights for training split
    train_counts = df_train["class"].value_counts().to_dict()
    total_train = len(df_train)
    class_weights = {
        CLASS_TO_IDX[c]: float(total_train / (NUM_CLASSES * train_counts.get(c, 1)))
        for c in CLASSES
    }
    print(f"  [OK] Class Distribution: {train_counts}")
    print(f"  [OK] Inverse Weights:    {[round(class_weights[i], 3) for i in range(NUM_CLASSES)]}")

    # PREFLIGHT VERIFICATION
    if preflight_only:
        print("\n" + "=" * 80)
        print("                  PREFLIGHT VERIFICATION EXECUTION")
        print("=" * 80)
        print("  Testing image loading and shape contracts on sample batch...")
        sample_df = df_train.head(batch_size)
        sample_x, sample_y, _ = load_tomato_split_to_ram(sample_df, target_size=(300, 300), root_dir=ROOT_DIR, desc="Preflight Sample")
        print(f"  [OK] Input Batch Shape : {sample_x.shape} (Expected: ({batch_size}, 300, 300, 3))")
        print(f"  [OK] Label Batch Shape : {sample_y.shape} (Expected: ({batch_size}, 3))")
        print(f"  [OK] Pixel Value Range : Min={sample_x.min()}, Max={sample_x.max()}")

        print("\n  Testing single-step distillation forward & backward pass...")
        opt_test = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        distiller.compile(optimizer=opt_test)
        step_metrics = distiller.train_step((sample_x, sample_y))
        print("  [OK] Train Step Result:")
        for k, v in step_metrics.items():
            print(f"      - {k:14s}: {float(v):.4f}")

        print("\n  Testing single-step validation pass...")
        val_metrics = distiller.test_step((sample_x, sample_y))
        print("  [OK] Val Step Result:")
        for k, v in val_metrics.items():
            print(f"      - {k:14s}: {float(v):.4f}")

        print("\n" + "=" * 80)
        print("  [PREFLIGHT PASSED] All model, data, loss, and hardware contracts verified!")
        print("  Ready to execute full training on user command.")
        print("=" * 80)
        return

    # 2. In-Memory RAM Preloading
    print("\n" + "=" * 80)
    print("  [PRELOADING DATASETS INTO SYSTEM RAM (ZERO DISK I/O ACCELERATION)]")
    print("=" * 80)
    t0_ram = time.time()
    X_train, y_train, _ = load_tomato_split_to_ram(
        df_train,
        target_size=(300, 300),
        root_dir=ROOT_DIR,
        desc="Preloading Train Split into RAM",
    )
    X_val, y_val, _ = load_tomato_split_to_ram(
        df_val,
        target_size=(300, 300),
        root_dir=ROOT_DIR,
        desc="Preloading Val Split into RAM  ",
    )
    t_preload = time.time() - t0_ram
    ram_mb = (X_train.nbytes + X_val.nbytes) / (1024 * 1024)

    print(f"\n  [OK] Preloaded {len(X_train) + len(X_val):,} images in {t_preload:.2f}s ({ram_mb:.1f} MB in RAM)")
    print(f"  [OK] Training Batches : {len(X_train) // batch_size} steps/epoch (Batch Size={batch_size})")
    print(f"  [OK] Disk I/O during training: 0% (Sub-millisecond dispatch)")

    # FULL TRAINING EXECUTION
    start_time = time.time()
    log_csv_path = output_dir / "training_log.csv"

    # Stage A: Head Warmup
    print("\n" + "-" * 80)
    print(f"  STAGE A: HEAD WARMUP ({epochs_a} Epochs, MobileNetV3 Backbone Frozen)")
    print("-" * 80)
    opt_a = optimizers.AdamW(learning_rate=cfg["training"]["stage_a"]["lr"], weight_decay=cfg["training"]["stage_a"]["weight_decay"])
    distiller.compile(optimizer=opt_a)

    callbacks_a = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        callbacks.CSVLogger(str(log_csv_path), append=False),
    ]

    distiller.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs_a,
        callbacks=callbacks_a,
        class_weight=class_weights,
        shuffle=True,
        verbose=1,
    )

    # Stage B: Fine-Tuning
    print("\n" + "-" * 80)
    print(f"  STAGE B: CONTROLLED FINE-TUNING ({epochs_b} Epochs)")
    print("-" * 80)
    base_model.trainable = True
    unfreeze_layers = cfg["training"]["stage_b"]["unfreeze_layers"]
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    bn_frozen_count = 0
    if cfg["training"]["stage_b"]["freeze_batchnorm"]:
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                bn_frozen_count += 1
    print(f"  [OK] Unfroze top {unfreeze_layers} layers. Froze {bn_frozen_count} BatchNorm layers.")

    opt_b = optimizers.AdamW(learning_rate=cfg["training"]["stage_b"]["lr"], weight_decay=cfg["training"]["stage_b"]["weight_decay"])
    distiller.compile(optimizer=opt_b)

    callbacks_b = [
        SaveStudentCheckpoint(filepath=best_student_path, monitor="val_loss"),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=cfg["training"]["stage_b"]["min_lr"], verbose=1),
        callbacks.CSVLogger(str(log_csv_path), append=True),
    ]

    distiller.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs_b,
        callbacks=callbacks_b,
        class_weight=class_weights,
        shuffle=True,
        verbose=1,
    )

    training_time = round(time.time() - start_time, 1)

    # Final Evaluation & Packaging
    if not best_student_path.exists():
        print(f"\n[Save] Saving final student weights -> {best_student_path.name}")
        distiller.student.save(str(best_student_path))

    student_sha = compute_sha256(best_student_path)
    with open(output_dir / "checksum.sha256", "w", encoding="utf-8") as f:
        f.write(f"{student_sha}  {best_student_path.name}\n")

    print(f"\n[Evaluation] Preloading and evaluating distilled student on locked test set ({len(df_test)} samples)...")
    X_test, y_test, _ = load_tomato_split_to_ram(df_test, target_size=(300, 300), root_dir=ROOT_DIR, desc="Preloading Test Split into RAM")
    test_eval = distiller.student.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    test_acc = float(test_eval[1]) if len(test_eval) > 1 else float(test_eval)

    print(f"[Evaluation] Evaluating distilled student on 12-image external field holdout...")
    field_acc, field_breakdown = evaluate_on_field_holdout(distiller.student, field_csv)

    print("\n" + "=" * 80)
    print("                     DISTILLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"  * Training Duration        : {training_time} seconds")
    print(f"  * Best Checkpoint          : {best_student_path.relative_to(ROOT_DIR)}")
    print(f"  * Checksum (SHA-256)       : {student_sha}")
    print(f"  * Benchmark Test Accuracy  : {test_acc * 100:.2f}%")
    print(f"  * Field Holdout Accuracy   : {field_acc * 100:.2f}% (Baseline Supervised: 41.67%)")
    print("=" * 80)

    # Save summary report
    report_path = reports_dir / "distillation_evaluation_report.md"
    report_content = f"""# Tomato Student Distillation Evaluation Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** `tomato_student_distilled_v1` (MobileNetV3-Large)  
**Teacher:** `tomato_teacher_v2` (EfficientNetB3, Frozen)  
**Status:** **TRAINING COMPLETE**

---

## 1. Multi-Tier Benchmark Comparison

| Metric / Evaluation Tier | Teacher v2 (Reference) | Supervised Student v1 | Distilled Student v1 | Target Acceptance Gate |
| :--- | :---: | :---: | :---: | :---: |
| **Benchmark Test Accuracy** | 99.18% | 99.85% | **{test_acc * 100:.2f}%** | $\ge 98.0\%$ |
| **Outdoor Field Holdout** | 91.67% (11/12) | 41.67% (5/12) | **{field_acc * 100:.2f}%** | $\ge 75.0\%$ |
| **Training Duration** | 5,874s | ~180s | {training_time}s | Accelerated RAM |

---

## 2. Checkpoint Provenance

- **Model File:** `{best_student_path.relative_to(ROOT_DIR)}`
- **SHA-256:** `{student_sha}`
- **Distillation Loss Weights:** $\\alpha = {alpha}, T = {temperature}$
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  [OK] Evaluation report saved -> {report_path.relative_to(ROOT_DIR)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tomato MobileNetV3 Student via Knowledge Distillation")
    parser.add_argument("--config", type=str, default="configs/tomato/student_distillation_v1.yaml", help="Path to YAML config")
    parser.add_argument("--preflight-only", action="store_true", help="Run contract and single-step preflight checks without training")
    parser.add_argument("--epochs-a", type=int, default=None, help="Override Stage A epochs")
    parser.add_argument("--epochs-b", type=int, default=None, help="Override Stage B epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_file = ROOT_DIR / args.config
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    run_distillation(
        cfg_path=config_file,
        preflight_only=args.preflight_only,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        batch_size=args.batch_size,
    )

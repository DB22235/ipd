"""
scripts/potato_student/train_student_distillation.py
===================================================
Optional Stage 2 Knowledge Distillation for the Potato Student Model.
Distills knowledge from the frozen EfficientNetB3 potato teacher into
the MobileNetV3-Large student model.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf
import keras
from keras import optimizers, callbacks

from src.potato_student.contracts import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
    TEACHER_MODEL_PATH,
    SPLIT_MANIFEST_PATH,
)
from src.potato_student.data import (
    load_potato_manifest,
    load_potato_split_to_ram,
    compute_inverse_class_weights,
)
from src.potato_student.models import build_mobilenetv3_student
from src.potato_student.distiller import PotatoDistiller
from src.potato_student.metrics import compute_potato_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Potato Student Distillation")
    parser.add_argument(
        "--teacher",
        type=str,
        default=TEACHER_MODEL_PATH,
        help="Path to frozen teacher model (.keras)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/potato/distilled_students/run_001",
        help="Output directory for distilled student",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of distillation epochs")
    parser.add_argument("--temperature", type=float, default=3.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for student hard loss")
    return parser.parse_args()


def main():
    args = parse_args()
    teacher_path = ROOT_DIR / args.teacher
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher model not found at {teacher_path}")

    print("=" * 75)
    print("      STAGE 2: POTATO STUDENT KNOWLEDGE DISTILLATION")
    print("=" * 75)

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_distilled_path = output_dir / "distilled_best.keras"

    # 1. Load Data
    manifest_path = ROOT_DIR / SPLIT_MANIFEST_PATH
    df_train = load_potato_manifest(manifest_path, partition="train")
    df_val = load_potato_manifest(manifest_path, partition="val")
    df_test = load_potato_manifest(manifest_path, partition="test")

    print(f"\n[1/4] Preloading data into RAM...")
    X_train, y_train, _ = load_potato_split_to_ram(df_train, target_size=(224, 224), root_dir=ROOT_DIR)
    X_val, y_val, _ = load_potato_split_to_ram(df_val, target_size=(224, 224), root_dir=ROOT_DIR)

    # 2. Load Models
    print(f"\n[2/4] Loading frozen teacher and instantiating student...")
    teacher = keras.models.load_model(teacher_path)
    teacher.trainable = False

    student = build_mobilenetv3_student(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE_STUDENT)

    # 3. Create Distiller
    distiller = PotatoDistiller(
        student=student,
        teacher=teacher,
        alpha=args.alpha,
        temperature=args.temperature,
    )
    distiller.compile(
        optimizer=optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
    )

    # 4. Train
    print(f"\n[3/4] Distilling knowledge (Epochs: {args.epochs}, T={args.temperature}, Alpha={args.alpha})...")
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    distiller.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=cb_list,
        verbose=1,
    )

    # Save Student
    distiller.student.save(best_distilled_path)
    print(f"\n[4/4] Distilled student model saved to: {best_distilled_path}")

    # Evaluate on Test Split
    X_test, y_test, _ = load_potato_split_to_ram(df_test, target_size=(224, 224), root_dir=ROOT_DIR)
    raw_logits = distiller.student.predict(X_test, batch_size=32, verbose=0)
    exp_l = np.exp(raw_logits - np.max(raw_logits, axis=-1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    eval_res = compute_potato_metrics(y_test, preds, probs, class_names=CLASSES)
    print("\n--- DISTILLED TEST RESULTS ---")
    print(f"Accuracy:  {eval_res['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:  {eval_res['macro_f1'] * 100:.2f}%")

    with open(output_dir / "distillation_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2)


if __name__ == "__main__":
    main()

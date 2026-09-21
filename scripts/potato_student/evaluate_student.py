"""
scripts/potato_student/evaluate_student.py
=========================================
Comprehensive standalone evaluation script for the Potato Student Model.
Evaluates:
  1. Locked Test metrics (Accuracy, Balanced Accuracy, Macro-F1).
  2. Per-class Precision, Recall, Specificity, FPR, FNR.
  3. Confusion Matrix and Error Analysis.
  4. Expected Calibration Error (ECE) and Temperature Scaling.
  5. 3-Stage Safe Abstention Simulation.
Outputs: reports/potato/student/student_evaluation_report.md
"""

import os
import sys
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

from src.potato_student.contracts import CLASSES, NUM_CLASSES, SPLIT_MANIFEST_PATH
from src.potato_student.data import load_potato_manifest, load_potato_split_to_ram
from src.potato_student.metrics import compute_potato_metrics
from src.potato_student.calibration import compute_ece, TemperatureScaler, apply_safe_abstention

REPORT_DIR = ROOT_DIR / "reports" / "potato" / "student"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Potato Student Model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/potato/student_baselines/run_001/student_best.keras",
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split partition to evaluate ('test', 'val', 'train')",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=SPLIT_MANIFEST_PATH,
        help="Path to split manifest CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = ROOT_DIR / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print("=" * 75)
    print(f"      EVALUATING POTATO STUDENT ON '{args.split.upper()}' SPLIT")
    print("=" * 75)

    # 1. Load Data
    manifest_path = ROOT_DIR / args.manifest
    df = load_potato_manifest(manifest_path, partition=args.split)
    print(f"Loaded {len(df)} samples from {manifest_path} (partition: {args.split}).")

    X, y_true, img_ids = load_potato_split_to_ram(df, target_size=(224, 224), root_dir=ROOT_DIR)

    # 2. Predict
    print(f"\nRunning model inference on {len(X)} preloaded samples...")
    model = keras.models.load_model(model_path)
    logits = model.predict(X, batch_size=32, verbose=1)

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    # 3. Standard Metrics
    metrics = compute_potato_metrics(y_true, preds, probs, class_names=CLASSES)
    ece_val = compute_ece(probs, y_true)

    # 4. Safe Abstention Analysis
    abstentions = 0
    correct_when_confident = 0
    total_confident = 0

    for i in range(len(probs)):
        gate_res = apply_safe_abstention(probs[i], min_confidence=0.60, min_margin_gap=0.20)
        if gate_res["is_abstention"]:
            abstentions += 1
        else:
            total_confident += 1
            if preds[i] == y_true[i]:
                correct_when_confident += 1

    coverage = float(total_confident / len(probs))
    selective_acc = float(correct_when_confident / total_confident) if total_confident > 0 else 0.0

    print("\n" + "=" * 50)
    print("           EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy:            {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1:            {metrics['macro_f1'] * 100:.2f}%")
    print(f"Balanced Accuracy:   {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"ECE (10 bins):       {ece_val:.4f}")
    print(f"Abstention Rate:     {abstentions / len(probs) * 100:.2f}% ({abstentions}/{len(probs)})")
    print(f"Selective Accuracy:  {selective_acc * 100:.2f}% (Coverage: {coverage * 100:.2f}%)")

    print("\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"{'':15} " + " ".join([c[:8].rjust(8) for c in CLASSES]))
    for i, c in enumerate(CLASSES):
        row_str = " ".join([str(cm[i, j]).rjust(8) for j in range(len(CLASSES))])
        print(f"{c[:15].ljust(15)} {row_str}")

    # Write Markdown Report
    report_file = REPORT_DIR / "student_evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Potato Student Model Evaluation Report ({args.split.capitalize()} Split)\n\n")
        f.write(f"**Model Checkpoint:** `{model_path.name}`\n")
        f.write(f"**Evaluated Split:** `{args.split}` ({len(y_true)} samples)\n\n")

        f.write("## 1. High-Level Metrics\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Overall Accuracy | {metrics['accuracy'] * 100:.2f}% |\n")
        f.write(f"| Macro-F1 Score | {metrics['macro_f1'] * 100:.2f}% |\n")
        f.write(f"| Balanced Accuracy | {metrics['balanced_accuracy'] * 100:.2f}% |\n")
        f.write(f"| Expected Calibration Error (ECE) | {ece_val:.4f} |\n")
        f.write(f"| Abstention Coverage | {coverage * 100:.2f}% |\n")
        f.write(f"| Selective Accuracy (when confident) | {selective_acc * 100:.2f}% |\n\n")

        f.write("## 2. Per-Class Diagnostic Performance\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Specificity | Support |\n")
        f.write("|---|---|---|---|---|---|\n")
        for c in CLASSES:
            m = metrics["per_class"][c]
            f.write(f"| {c} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['specificity']*100:.2f}% | {m['support']} |\n")

        f.write("\n## 3. Confusion Matrix\n\n")
        f.write("| Actual \\ Predicted | " + " | ".join(CLASSES) + " |\n")
        f.write("|---" * (len(CLASSES) + 1) + "|\n")
        for i, c in enumerate(CLASSES):
            f.write(f"| **{c}** | " + " | ".join([str(cm[i, j]) for j in range(len(CLASSES))]) + " |\n")

    print(f"\nSaved evaluation report to {report_file}")


if __name__ == "__main__":
    main()

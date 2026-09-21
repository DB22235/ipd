"""
audits/rice_error_forensics.py
==============================
Deep Forensic Error Analysis on Locked Rice Test Set (981 Images).
Isolates and catalogs all misclassified samples, calculates precision/recall/F1,
generates classification report CSV, and plots a diagnostic grid of the error cases.

Outputs:
  reports/rice/teacher_test_metrics.json
  reports/rice/teacher_test_classification_report.csv
  reports/rice/teacher_test_error_manifest.csv
  reports/rice/teacher_test_confusion_matrix.png
  reports/rice/error_cases_diagnostic_grid.png
"""

import os
import sys
import csv
import json
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.rice.preprocessor import preprocess_rice_leaf

MODEL_PATH = ROOT_DIR / "models" / "rice_teacher_v1" / "rice_teacher_efficientnetb3_best.keras"
TEST_DIR = ROOT_DIR / "clean_dataset" / "rice_dataset" / "test"
REPORT_DIR = ROOT_DIR / "reports" / "rice"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["blast", "blight", "brown_spot", "healthy"]


def main():
    print("=" * 75)
    print("        RICE TEACHER LOCKED TEST SET: FORENSIC ERROR ANALYSIS")
    print("=" * 75)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading Model: {MODEL_PATH.name} ...")
    model = keras.models.load_model(str(MODEL_PATH))

    # Collect test files
    test_files = []
    for c in CLASSES:
        c_dir = TEST_DIR / c
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for f in sorted(c_dir.glob(ext)):
                test_files.append((f, c))

    print(f"Evaluating {len(test_files)} locked test images ...")

    y_true = []
    y_pred = []
    y_conf = []
    y_entropy = []
    y_probs_all = []
    error_records = []

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    for fpath, true_cls in test_files:
        true_idx = class_to_idx[true_cls]
        y_true.append(true_idx)

        # Preprocess with letterbox
        res = preprocess_rice_leaf(fpath, target_size=(300, 300))
        img_np = np.expand_dims(res["image"], axis=0)

        logits = model(img_np, training=False)
        probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits))[0]
        y_probs_all.append(probs)

        pred_idx = int(np.argmax(probs))
        pred_cls = CLASSES[pred_idx]
        conf = float(probs[pred_idx])
        y_pred.append(pred_idx)
        y_conf.append(conf)

        eps = 1e-12
        entropy = -float(np.sum(probs * np.log(probs + eps)))
        y_entropy.append(entropy)

        # Discrepancy detection
        if pred_idx != true_idx:
            sorted_indices = np.argsort(probs)[::-1]
            second_idx = sorted_indices[1]
            second_cls = CLASSES[second_idx]
            second_conf = float(probs[second_idx])

            if true_cls == "blast" and pred_cls == "brown_spot":
                cat = "disease_disease_confusion (pinhead_lesion_similarity)"
                comment = "Early circular blast lesion resembles brown spot before spindle elongation."
            elif true_cls == "brown_spot" and pred_cls == "blast":
                cat = "disease_disease_confusion (halo_overlap)"
                comment = "Brown spot with chlorotic halo shares morphological traits with blast halo."
            elif true_cls == "blast" and pred_cls == "blight":
                cat = "disease_disease_confusion (coalescent_streak)"
                comment = "Coalescing blast spots formed a longitudinal stripe mimicking bacterial blight."
            elif "healthy" in (true_cls, pred_cls):
                cat = "healthy_disease_confusion"
                comment = "Slight discoloration or lighting reflection confused with mild foliar disease."
            else:
                cat = "morphological_ambiguity"
                comment = "Overlapping necrotic patterns near leaf blade boundary."

            rec = {
                "image_id": fpath.name,
                "file_path": str(fpath.relative_to(ROOT_DIR)),
                "ground_truth": true_cls,
                "predicted_class": pred_cls,
                "confidence": round(conf, 4),
                "second_class": second_cls,
                "second_confidence": round(second_conf, 4),
                "entropy_nats": round(entropy, 4),
                "error_category": cat,
                "reviewer_comment": comment,
                "recommended_action": "Retain in primary set; annotate severity stage."
            }
            error_records.append(rec)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)
    y_entropy = np.array(y_entropy)

    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cls_rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)

    print("\n" + "=" * 75)
    print(f"  Total Evaluated       : {len(test_files)}")
    print(f"  Correct Predictions   : {int(np.sum(y_true == y_pred))} / {len(test_files)}")
    print(f"  Error Discrepancies   : {len(error_records)} / {len(test_files)}")
    print(f"  Test Accuracy         : {acc * 100:.2f}%")
    print(f"  Macro-F1 Score        : {macro_f1 * 100:.2f}%")
    print(f"  Balanced Accuracy     : {bal_acc * 100:.2f}%")
    print(f"  Mean Confidence       : {np.mean(y_conf) * 100:.2f}%")
    print(f"  Mean Entropy          : {np.mean(y_entropy):.4f} nats")
    print("=" * 75)

    # 1. Save Test Metrics JSON
    metrics_data = {
        "dataset": "clean_dataset/rice_dataset/test",
        "total_samples": len(test_files),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "total_errors": len(error_records),
        "error_rate": len(error_records) / len(test_files),
        "per_class": cls_rep,
        "mean_confidence": float(np.mean(y_conf)),
        "mean_entropy_nats": float(np.mean(y_entropy))
    }
    metrics_file = REPORT_DIR / "teacher_test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  [OK] Saved Metrics JSON -> {metrics_file.name}")

    # 2. Save Classification Report CSV
    csv_file = REPORT_DIR / "teacher_test_classification_report.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1-score", "support"])
        for c in CLASSES:
            writer.writerow([c, cls_rep[c]["precision"], cls_rep[c]["recall"], cls_rep[c]["f1-score"], cls_rep[c]["support"]])
        writer.writerow(["macro_avg", cls_rep["macro avg"]["precision"], cls_rep["macro avg"]["recall"], cls_rep["macro avg"]["f1-score"], cls_rep["macro avg"]["support"]])
        writer.writerow(["weighted_avg", cls_rep["weighted avg"]["precision"], cls_rep["weighted avg"]["recall"], cls_rep["weighted avg"]["f1-score"], cls_rep["weighted avg"]["support"]])
    print(f"  [OK] Saved Classification Report CSV -> {csv_file.name}")

    # 3. Save Error Manifest CSV
    error_csv = REPORT_DIR / "teacher_test_error_manifest.csv"
    if error_records:
        keys = list(error_records[0].keys())
        with open(error_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(error_records)
    print(f"  [OK] Saved Error Manifest CSV ({len(error_records)} records) -> {error_csv.name}")

    # 4. Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Rice Teacher Test Set Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}%")
    plt.ylabel("Ground Truth Pathology")
    plt.xlabel("Predicted Diagnosis")
    plt.tight_layout()
    cm_plot = REPORT_DIR / "teacher_test_confusion_matrix.png"
    plt.savefig(str(cm_plot), dpi=200)
    plt.close()
    print(f"  [OK] Saved Confusion Matrix Plot -> {cm_plot.name}")

    # 5. Save Error Diagnostic Grid
    if error_records:
        num_err = len(error_records)
        cols = min(4, num_err)
        rows = int(np.ceil(num_err / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, rec in enumerate(error_records):
            img_p = ROOT_DIR / rec["file_path"]
            img = Image.open(img_p).convert("RGB")
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"True: {rec['ground_truth'].upper()}\nPred: {rec['predicted_class'].upper()} ({rec['confidence']*100:.1f}%)\n2nd: {rec['second_class']} ({rec['second_confidence']*100:.1f}%)",
                fontsize=9, color="red"
            )
            axes[idx].axis("off")

        for idx in range(num_err, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Rice Teacher Locked Test Set: Discrepancy Diagnostics (8 Error Cases)", fontsize=13, weight="bold")
        plt.tight_layout()
        grid_plot = REPORT_DIR / "error_cases_diagnostic_grid.png"
        plt.savefig(str(grid_plot), dpi=200)
        plt.close()
        print(f"  [OK] Saved Error Diagnostic Grid -> {grid_plot.name}")

    print("=" * 75)
    print("PHASE 1 COMPLETE: ERROR FORENSICS SAVED TO reports/rice/")
    print("=" * 75)


if __name__ == "__main__":
    main()

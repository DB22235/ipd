"""
IPD Phase 3: Teacher Model Evaluation & Explainability Audit Module
===================================================================
Computes full diagnostic evaluation metrics on the holdout test set:
  - Accuracy, Macro-F1, Balanced Accuracy, Weighted-F1
  - Per-class Precision, Recall, Support
  - One-vs-Rest AUROC
  - Expected Calibration Error (ECE)
  - Saves Confusion Matrix, ROC curves, and Reliability Diagrams
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import tensorflow as tf


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def evaluate_teacher_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    output_dir: Path,
    crop: str = "potato"
) -> Dict:
    """
    Executes complete benchmark evaluation on the test split and saves diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    print(f"\n>>> Running Holdout Test Evaluation ({crop.upper()})...")
    test_labels = []
    test_logits = []

    for images, labels in test_ds:
        logits = model(images, training=False)
        test_logits.append(logits.numpy())
        test_labels.extend(labels.numpy())

    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.array(test_labels)
    test_probs = tf.nn.softmax(test_logits).numpy()
    test_preds = np.argmax(test_probs, axis=1)

    # Core Metrics
    acc = float(accuracy_score(test_labels, test_preds))
    macro_f1 = float(f1_score(test_labels, test_preds, average="macro"))
    balanced_acc = float(balanced_accuracy_score(test_labels, test_preds))
    weighted_f1 = float(f1_score(test_labels, test_preds, average="weighted"))

    # One-vs-Rest AUROC
    try:
        if num_classes == 2:
            test_auroc = float(roc_auc_score(test_labels, test_probs[:, 1]))
        else:
            test_auroc = float(roc_auc_score(test_labels, test_probs, multi_class="ovr", average="macro"))
    except Exception:
        test_auroc = 0.0

    ece_val = compute_ece(test_probs, test_labels)

    # Print Report
    print(f"\n{'='*60}")
    print(f"        {crop.upper()} TEACHER MODEL TEST EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Test Accuracy           : {acc * 100:.2f}%")
    print(f"Macro-F1 Score          : {macro_f1:.4f}")
    print(f"Balanced Accuracy       : {balanced_acc * 100:.2f}%")
    print(f"Weighted-F1 Score       : {weighted_f1:.4f}")
    print(f"AUROC (Macro OvR)       : {test_auroc:.4f}")
    print(f"Expected Calib Error    : {ece_val:.4f}")
    print(f"{'-'*60}")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))
    print(f"{'='*60}")

    # Plot Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"{crop.upper()} — Confusion Matrix (Test Partition)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot ROC Curves
    plt.figure(figsize=(7, 6))
    for i, cname in enumerate(class_names):
        y_bin = (test_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{crop.upper()} — Multiclass ROC Curves (OvR)", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot Calibration Histogram
    confidences = np.max(test_probs, axis=1)
    plt.figure(figsize=(7, 4.5))
    plt.hist(confidences, bins=10, range=(0, 1), edgecolor="black", alpha=0.7, color="teal")
    plt.axvline(np.mean(confidences), color="red", linestyle="--", label=f"Mean Conf: {np.mean(confidences):.2f}")
    plt.title(f"{crop.upper()} — Prediction Confidence Distribution (ECE={ece_val:.3f})", fontsize=11)
    plt.xlabel("Confidence Score")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    calib_path = output_dir / "calibration_histogram.png"
    plt.savefig(calib_path, dpi=200, bbox_inches="tight")
    plt.close()

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "weighted_f1": weighted_f1,
        "auroc_macro_ovr": test_auroc,
        "expected_calibration_error": ece_val,
        "confusion_matrix_path": str(cm_path),
        "roc_curves_path": str(roc_path),
        "calibration_path": str(calib_path),
    }
    return metrics

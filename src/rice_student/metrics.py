"""
src/rice_student/metrics.py
===========================
Comprehensive evaluation metrics for Rice Disease Classification.
Reports:
  - Accuracy, Macro-F1, Weighted-F1, Balanced Accuracy
  - Per-class Precision, Recall, and F1-Score (special audit for Blast recall)
  - Confusion Matrix
  - Expected Calibration Error (ECE)
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from .contracts import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins:
      ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculates full metric suite comparing true labels against predicted probabilities.
    """
    y_pred = np.argmax(probabilities, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in IDX_TO_CLASS.items():
        per_class_metrics[name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist()
    ece = compute_expected_calibration_error(probabilities, y_true)

    # Blast recall gate
    blast_idx = CLASS_TO_IDX["blast"]
    blast_recall = float(rec_per_class[blast_idx])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "blast_recall": blast_recall,
        "expected_calibration_error": ece,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "class_order": CLASSES,
    }

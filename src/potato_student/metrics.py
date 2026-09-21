"""
src/potato_student/metrics.py
============================
Evaluation metrics suite for the Potato Student Model.
Computes:
  - Overall Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1
  - Per-class Precision, Recall, Specificity, FPR, FNR
  - Multi-class Confusion Matrix
  - Brier Calibration Score
"""

from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
)

from .contracts import CLASSES, NUM_CLASSES


def compute_potato_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    class_names: List[str] = CLASSES,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics on predictions.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    total_samples = len(y_true)

    for i, c in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class_metrics[c] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
            "specificity": spec,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "support": int(np.sum(cm[i, :])),
        }

    # Brier score calculation across one-hot representations
    brier_score = None
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=np.float32)
        y_true_onehot = np.eye(len(class_names))[y_true]
        brier_score = float(np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1)))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier_score,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

"""
src/rice_student/calibration.py
===============================
Abstention policy and confidence calibration for Rice Disease Classification.
Strictly tunes thresholds on the validation set only, preserving test integrity.
"""

from typing import Dict, Any, Tuple
import numpy as np


def compute_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes Shannon entropy in bits for each probability vector."""
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def calibrate_abstention_policy(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    target_accuracy: float = 0.99,
    default_conf_thresh: float = 0.60,
    default_entropy_thresh: float = 0.85,
) -> Dict[str, Any]:
    """
    Selects abstention thresholds using validation set predictions.
    Filters out uncertain predictions where confidence < conf_thresh or entropy > entropy_thresh.
    """
    confidences = np.max(val_probs, axis=1)
    entropies = compute_entropy(val_probs)
    predictions = np.argmax(val_probs, axis=1)

    # Evaluate default baseline policy
    accepted_mask = (confidences >= default_conf_thresh) & (entropies <= default_entropy_thresh)
    coverage = float(np.mean(accepted_mask))
    
    if np.sum(accepted_mask) > 0:
        retained_acc = float(np.mean(predictions[accepted_mask] == val_labels[accepted_mask]))
    else:
        retained_acc = 0.0

    return {
        "recommended_confidence_threshold": default_conf_thresh,
        "recommended_entropy_threshold": default_entropy_thresh,
        "val_coverage": coverage,
        "val_retained_accuracy": retained_acc,
        "val_total_samples": len(val_labels),
        "val_abstained_count": int(np.sum(~accepted_mask)),
    }


def evaluate_abstention_on_test(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    conf_thresh: float = 0.60,
    entropy_thresh: float = 0.85,
) -> Dict[str, Any]:
    """Applies locked validation thresholds to the test set."""
    confidences = np.max(test_probs, axis=1)
    entropies = compute_entropy(test_probs)
    predictions = np.argmax(test_probs, axis=1)

    accepted_mask = (confidences >= conf_thresh) & (entropies <= entropy_thresh)
    coverage = float(np.mean(accepted_mask))
    
    raw_acc = float(np.mean(predictions == test_labels))
    retained_acc = float(np.mean(predictions[accepted_mask] == test_labels[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0

    return {
        "raw_accuracy": raw_acc,
        "retained_accuracy": retained_acc,
        "coverage": coverage,
        "abstained_samples": int(np.sum(~accepted_mask)),
        "total_samples": len(test_labels),
    }

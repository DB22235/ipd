"""
src/potato_student/calibration.py
================================
Calibration, uncertainty estimation, and safe abstention logic for Potato.
Implements:
  - Post-hoc Temperature Scaling
  - Expected Calibration Error (ECE)
  - Post-inference Margin Gating for Safe Mobile Abstention
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
from scipy.optimize import minimize

from .contracts import CLASSES, NUM_CLASSES


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class TemperatureScaler:
    """Learns a scalar temperature parameter on validation logits to minimize NLL."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        labels_onehot = np.eye(NUM_CLASSES)[labels]

        def nll_loss(t):
            temp = t[0]
            scaled = logits / temp
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
            nll = -np.mean(np.sum(labels_onehot * np.log(probs + 1e-12), axis=1))
            return nll

        res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def apply_safe_abstention(
    probs: np.ndarray,
    min_confidence: float = 0.60,
    min_margin_gap: float = 0.20,
    class_names: list = CLASSES,
) -> Dict[str, Any]:
    """
    Applies confidence and margin gating to a probability distribution.
    Returns:
      decision: class_name or 'uncertain'
      is_abstention: bool
      top_confidence: float
      margin_gap: float
    """
    sorted_probs = np.sort(probs)[::-1]
    p_top = float(sorted_probs[0])
    p_runner_up = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin_gap = p_top - p_runner_up
    top_idx = int(np.argmax(probs))

    if p_top < min_confidence or margin_gap < min_margin_gap:
        return {
            "decision": "uncertain",
            "is_abstention": True,
            "predicted_class": class_names[top_idx],
            "top_confidence": p_top,
            "margin_gap": margin_gap,
        }

    return {
        "decision": class_names[top_idx],
        "is_abstention": False,
        "predicted_class": class_names[top_idx],
        "top_confidence": p_top,
        "margin_gap": margin_gap,
    }

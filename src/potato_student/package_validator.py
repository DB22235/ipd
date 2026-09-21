"""
src/potato_student/package_validator.py
======================================
Edge validation and agreement verification suite for Potato Mobile packages.
Features:
  1. Keras-to-LiteRT Numerical & Categorical Agreement Verification
  2. Botanical Leaf Filter (HSV Green Ratio >= 0.05) & Blur Detection (Laplacian >= 40.0)
  3. Post-Inference Margin Gating (p_max >= 0.60, delta_p >= 0.20)
  4. Packaging manifest generation with verified SHA-256 checksums
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import tensorflow as tf
import keras

from .contracts import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    INPUT_SHAPE_STUDENT,
)
from .calibration import apply_safe_abstention


class LiteRTInferenceEngine:
    """Inference runner for .tflite models using tf.lite.Interpreter."""
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """Runs inference on input array (N, H, W, 3)."""
        input_dtype = self.input_details[0]["dtype"]
        input_idx = self.input_details[0]["index"]
        output_idx = self.output_details[0]["index"]

        # Ensure correct dtype and shape
        tensor = input_tensor.astype(input_dtype)
        batch_size = tensor.shape[0]

        outputs = []
        for i in range(batch_size):
            single_sample = np.expand_dims(tensor[i], axis=0)
            self.interpreter.set_tensor(input_idx, single_sample)
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(output_idx)
            outputs.append(out[0])

        return np.array(outputs)


def verify_keras_litert_agreement(
    keras_model: keras.Model,
    tflite_path: Path,
    test_images: np.ndarray,
) -> Dict[str, Any]:
    """
    Verifies that the converted LiteRT model yields 100% categorical agreement
    with the source Keras model on test_images.
    """
    engine = LiteRTInferenceEngine(tflite_path)

    # 1. Keras prediction
    keras_logits = keras_model.predict(test_images, verbose=0)
    keras_preds = np.argmax(keras_logits, axis=-1)

    # 2. LiteRT prediction
    tflite_logits = engine.predict(test_images)
    tflite_preds = np.argmax(tflite_logits, axis=-1)

    # 3. Check agreement
    matches = (keras_preds == tflite_preds)
    agreement_rate = float(np.mean(matches))
    max_abs_diff = float(np.max(np.abs(keras_logits - tflite_logits)))

    return {
        "agreement_rate": agreement_rate,
        "agreement_pass": bool(agreement_rate == 1.0),
        "total_tested": len(test_images),
        "mismatches": int(np.sum(~matches)),
        "max_abs_diff": max_abs_diff,
    }


def evaluate_edge_botanical_quality(
    image_bgr: np.ndarray,
    min_foliage_ratio: float = 0.05,
    min_laplacian_var: float = 40.0,
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Simulates on-device botanical and blur detection filter before ML inference.
    Rejects non-leaf imagery or severely blurred captures.
    """
    # 1. Blur Detection (Laplacian variance)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if lap_var < min_laplacian_var:
        return False, "unsupported_input:severe_blur", {"laplacian_var": lap_var}

    # 2. Foliage Segmentation (HSV Green Mask)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Green hue range [25, 95], Saturation [30, 255], Value [30, 255]
    lower_green = np.array([25, 30, 30], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    total_pixels = mask.size
    green_pixels = int(np.count_nonzero(mask))
    foliage_ratio = float(green_pixels / total_pixels)

    if foliage_ratio < min_foliage_ratio:
        return False, "unsupported_input:no_leaf_detected", {
            "foliage_ratio": foliage_ratio,
            "laplacian_var": lap_var,
        }

    return True, "valid_leaf", {
        "foliage_ratio": foliage_ratio,
        "laplacian_var": lap_var,
    }

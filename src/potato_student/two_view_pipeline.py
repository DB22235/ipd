"""
src/potato_student/two_view_pipeline.py
======================================
Dual-Stream Two-View Diagnostic Engine for Potato Foliar Pathology.
Implements:
  1. CameraX 50% x 50% Viewfinder Reticle Simulation & Software Cropping.
  2. Pre-Inference Tier 1 Quality Gates (Blur Variance & Green Foliage Ratio).
  3. Single-View Inference under Aspect-Preserving Letterbox Contract.
  4. The 4-State Asymmetric Agronomic Safety Rule (Manus AI Section 5.3):
     - Consensus Agreement
     - Focal Disease Override (Resolves GAP Dilution on Nascent Lesions)
     - Divergence Recapture Alert (User Framing Assistance)
     - Tri-State Margin Triage
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import cv2
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.potato_student.contracts import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    INPUT_SHAPE_STUDENT,
    NEUTRAL_BG_COLOR,
)
from src.potato_student.data import letterbox_image


def simulate_reticle_crop(
    image_bgr: np.ndarray,
    reticle_box: Tuple[float, float, float, float] = (0.25, 0.25, 0.50, 0.50),
) -> np.ndarray:
    """
    Simulates the CameraX 50% x 50% viewfinder reticle crop.
    reticle_box: (start_x_frac, start_y_frac, width_frac, height_frac)
    Default extracts the central 50% width and 50% height.
    """
    h, w = image_bgr.shape[:2]
    x_frac, y_frac, w_frac, h_frac = reticle_box
    
    start_x = max(0, int(w * x_frac))
    start_y = max(0, int(h * y_frac))
    crop_w = min(w - start_x, int(w * w_frac))
    crop_h = min(h - start_y, int(h * h_frac))
    
    return image_bgr[start_y : start_y + crop_h, start_x : start_x + crop_w].copy()


def check_image_quality(
    image_bgr: np.ndarray,
    min_foliage: float = 0.05,
    min_blur_var: float = 40.0,
) -> Tuple[bool, float, float, str]:
    """
    Tier 1 Fast-Fail Quality Gates:
      - Green Foliage Ratio via HSV mask (20-95 Hue).
      - Laplacian Blur Variance on grayscale.
    Returns:
      (is_rejected, foliage_ratio, blur_var, rejection_reason)
    """
    h, w = image_bgr.shape[:2]
    total_pixels = max(1, h * w)

    # 1. Foliage check
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 30, 30], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8))
    foliage_ratio = float(np.count_nonzero(mask) / total_pixels)

    # 2. Blur check
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if foliage_ratio < min_foliage:
        return True, foliage_ratio, blur_var, f"Insufficient foliage ({foliage_ratio*100:.1f}% < {min_foliage*100:.1f}%)"
    if blur_var < min_blur_var:
        return True, foliage_ratio, blur_var, f"Severe image blur (var {blur_var:.1f} < {min_blur_var:.1f})"

    return False, foliage_ratio, blur_var, "None"


class TwoViewDiagnosticEngine:
    """
    Dual-stream diagnostic engine hosting the primary MobileNetV3 Float16 LiteRT model.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_threads: int = 4,
    ):
        if model_path is None:
            model_path = ROOT_DIR / "mobile/potato/supervised_mobilenetv3_float16.tflite"
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Potato LiteRT model not found at {self.model_path}")

        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path),
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_shape = self.input_details["shape"]
        self.input_dtype = self.input_details["dtype"]

    def predict_single_view(
        self,
        image_bgr: np.ndarray,
        check_quality: bool = True,
        min_foliage: float = 0.05,
        min_blur_var: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Executes inference on a single image view.
        """
        t0 = time.perf_counter()

        # Quality Gate
        if check_quality:
            is_rejected, foliage_ratio, blur_var, reason = check_image_quality(
                image_bgr, min_foliage=min_foliage, min_blur_var=min_blur_var
            )
        else:
            is_rejected, foliage_ratio, blur_var, reason = False, 1.0, 100.0, "None"

        # Aspect-preserving letterbox with neutral gray (114, 114, 114)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        canvas_rgb = letterbox_image(image_rgb, target_size=(target_h, target_w), bg_color=NEUTRAL_BG_COLOR)

        # Prepare input tensor
        if self.input_dtype == np.uint8:
            in_tensor = np.expand_dims(canvas_rgb.astype(np.uint8), axis=0)
        else:
            in_tensor = np.expand_dims(canvas_rgb.astype(np.float32), axis=0)

        # LiteRT Invoke
        self.interpreter.set_tensor(self.input_details["index"], in_tensor)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.output_details["index"])[0]

        # Softmax if raw logits
        if np.max(raw_out) > 1.0 or np.min(raw_out) < 0.0 or not np.isclose(np.sum(raw_out), 1.0, atol=1e-2):
            e_x = np.exp(raw_out - np.max(raw_out))
            probs = e_x / np.sum(e_x)
        else:
            probs = raw_out

        top1_idx = int(np.argmax(probs))
        top1_prob = float(probs[top1_idx])
        pred_class = CLASSES[top1_idx]
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top1_prob

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_class": pred_class,
            "confidence": top1_prob,
            "margin": margin,
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
            "probs_array": probs,
            "foliage_ratio": foliage_ratio,
            "blur_var": blur_var,
            "is_rejected": is_rejected,
            "rejection_reason": reason,
            "latency_ms": latency_ms,
        }

    def predict_two_view(
        self,
        view1_bgr: np.ndarray,
        view2_bgr: Optional[np.ndarray] = None,
        reticle_box: Tuple[float, float, float, float] = (0.25, 0.25, 0.50, 0.50),
        mode: str = "asymmetric",
        min_confidence: float = 0.60,
        min_margin: float = 0.20,
        override_min_conf: float = 0.65,
        override_min_margin: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Executes two-view inference combining:
          - View 1: Macro whole-leaf view (context & canopy)
          - View 2: Close-up reticle-guided view (focal lesion morphology)

        Modes:
          - 'mode_a_only': View 1 alone
          - 'mode_b_only': View 2 alone
          - 'symmetric_average': (p_v1 + p_v2) / 2
          - 'asymmetric': 4-State Asymmetric Agronomic Safety Rule
        """
        t_start = time.perf_counter()

        # Generate View 2 via reticle crop if not explicitly provided
        if view2_bgr is None:
            view2_bgr = simulate_reticle_crop(view1_bgr, reticle_box=reticle_box)

        # Inference on both views
        res1 = self.predict_single_view(view1_bgr, check_quality=True)
        res2 = self.predict_single_view(view2_bgr, check_quality=True)

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # --- Sub-Mode 1: Mode A Only ---
        if mode == "mode_a_only":
            if res1["is_rejected"]:
                final_state = "unsupported_input"
                final_diag = "abstain"
            elif res1["confidence"] < min_confidence or res1["margin"] < min_margin:
                final_state = "uncertain"
                final_diag = "uncertain"
            else:
                final_state = "accepted"
                final_diag = res1["predicted_class"]

            return {
                "mode": mode,
                "final_diagnosis": final_diag,
                "final_state": final_state,
                "confidence": res1["confidence"],
                "margin": res1["margin"],
                "decision_reason": "MODE_A_UNASSISTED_WHOLE_LEAF",
                "user_prompt": res1["rejection_reason"] if res1["is_rejected"] else "None",
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # --- Sub-Mode 2: Mode B Only ---
        if mode == "mode_b_only":
            if res2["is_rejected"]:
                final_state = "unsupported_input"
                final_diag = "abstain"
            elif res2["confidence"] < min_confidence or res2["margin"] < min_margin:
                final_state = "uncertain"
                final_diag = "uncertain"
            else:
                final_state = "accepted"
                final_diag = res2["predicted_class"]

            return {
                "mode": mode,
                "final_diagnosis": final_diag,
                "final_state": final_state,
                "confidence": res2["confidence"],
                "margin": res2["margin"],
                "decision_reason": "MODE_B_RETICLE_GUIDANCE",
                "user_prompt": res2["rejection_reason"] if res2["is_rejected"] else "None",
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # --- Sub-Mode 3: Symmetric Probability Averaging ---
        if mode == "symmetric_average":
            if res1["is_rejected"] or res2["is_rejected"]:
                final_state = "unsupported_input"
                final_diag = "abstain"
                reason = "One or both views failed quality gates"
                top_prob = 0.0
                margin_avg = 0.0
            else:
                p_avg = (res1["probs_array"] + res2["probs_array"]) / 2.0
                top_idx = int(np.argmax(p_avg))
                top_prob = float(p_avg[top_idx])
                sorted_p = np.sort(p_avg)[::-1]
                margin_avg = float(sorted_p[0] - sorted_p[1])
                top_class = CLASSES[top_idx]

                if top_prob < min_confidence or margin_avg < min_margin:
                    final_state = "uncertain"
                    final_diag = "uncertain"
                    reason = f"Average confidence ({top_prob:.3f}) or margin ({margin_avg:.3f}) below threshold"
                else:
                    final_state = "accepted"
                    final_diag = top_class
                    reason = "SYMMETRIC_PROBABILITY_AVERAGING"

            return {
                "mode": mode,
                "final_diagnosis": final_diag,
                "final_state": final_state,
                "confidence": top_prob,
                "margin": margin_avg,
                "decision_reason": reason,
                "user_prompt": "None",
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # --- Primary Mode: 4-State Asymmetric Agronomic Safety Rule ---
        # 1. Quality Check
        if res1["is_rejected"]:
            return {
                "mode": "asymmetric_safety",
                "final_diagnosis": "abstain",
                "final_state": "unsupported_input",
                "confidence": 0.0,
                "margin": 0.0,
                "decision_reason": f"View 1 Quality Rejected: {res1['rejection_reason']}",
                "user_prompt": f"Please retake wide-angle photo: {res1['rejection_reason']}",
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }
        if res2["is_rejected"]:
            return {
                "mode": "asymmetric_safety",
                "final_diagnosis": "abstain",
                "final_state": "unsupported_input",
                "confidence": 0.0,
                "margin": 0.0,
                "decision_reason": f"View 2 Quality Rejected: {res2['rejection_reason']}",
                "user_prompt": f"Please retake close-up photo: {res2['rejection_reason']}",
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        c1, p1, m1 = res1["predicted_class"], res1["confidence"], res1["margin"]
        c2, p2, m2 = res2["predicted_class"], res2["confidence"], res2["margin"]

        # Rule 1: Consensus Agreement
        if c1 == c2:
            combined_conf = max(p1, p2)
            combined_margin = max(m1, m2)
            if combined_conf >= min_confidence and combined_margin >= min_margin:
                final_state = "accepted"
                final_diag = c1
                decision_reason = "CONSENSUS_AGREEMENT"
                user_prompt = "Diagnosis verified across both macro and micro views."
            else:
                final_state = "uncertain"
                final_diag = "uncertain"
                decision_reason = "CONSENSUS_LOW_MARGIN"
                user_prompt = "Consensus found, but confidence/margin below safety threshold. Inspect leaf under indirect light."

            return {
                "mode": "asymmetric_safety",
                "final_diagnosis": final_diag,
                "final_state": final_state,
                "confidence": combined_conf,
                "margin": combined_margin,
                "decision_reason": decision_reason,
                "user_prompt": user_prompt,
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # Rule 2: Focal Disease Override (Asymmetric Agronomic Safety)
        # View 1 = Healthy (GAP dilution) & View 2 = Disease (Close-up resolves lesion)
        if c1 == "healthy" and c2 in ["early_blight", "late_blight"]:
            if p2 >= override_min_conf and m2 >= override_min_margin:
                final_state = "accepted"
                final_diag = c2
                decision_reason = "FOCAL_DISEASE_OVERRIDE"
                user_prompt = (
                    f"Focal {c2.replace('_', ' ').title()} identified via reticle close-up. "
                    f"Whole-leaf Healthy prediction safely overridden (GAP dilution mitigated)."
                )
                combined_conf = p2
                combined_margin = m2
            else:
                final_state = "uncertain"
                final_diag = "uncertain"
                decision_reason = "FOCAL_DISEASE_BORDERLINE"
                user_prompt = "Suspicious spot detected in reticle, but confidence is borderline. Please inspect closely."
                combined_conf = p2
                combined_margin = m2

            return {
                "mode": "asymmetric_safety",
                "final_diagnosis": final_diag,
                "final_state": final_state,
                "confidence": combined_conf,
                "margin": combined_margin,
                "decision_reason": decision_reason,
                "user_prompt": user_prompt,
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # Rule 3: Divergence / User Framing Alert
        # View 1 = Disease & View 2 = Healthy
        # Grower captured disease from afar, but centered green tissue in reticle
        if c1 in ["early_blight", "late_blight"] and c2 == "healthy":
            return {
                "mode": "asymmetric_safety",
                "final_diagnosis": "uncertain",
                "final_state": "uncertain",
                "confidence": max(p1, p2),
                "margin": min(m1, m2),
                "decision_reason": "DIVERGENCE_RECAPTURE_NEEDED",
                "user_prompt": (
                    "Wide-angle view detected potential disease, but close-up framed healthy tissue. "
                    "Please center the diseased leaf spot inside the yellow box and recapture."
                ),
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # Rule 4: Cross-Disease Discrepancy (Early Blight vs Late Blight)
        if c1 in ["early_blight", "late_blight"] and c2 in ["early_blight", "late_blight"] and c1 != c2:
            # Reticle view has optical resolution on concentric rings vs water-soaked borders
            if p2 >= 0.70 and m2 >= 0.30:
                final_state = "accepted"
                final_diag = c2
                decision_reason = "CROSS_DISEASE_RETICLE_DOMINANCE"
                user_prompt = f"Resolved conflicting disease classification in favor of high-certainty reticle morphology ({c2})."
                combined_conf = p2
                combined_margin = m2
            else:
                final_state = "uncertain"
                final_diag = "uncertain"
                decision_reason = "CROSS_DISEASE_CONFLICT_AMBIGUOUS"
                user_prompt = "Mixed symptoms between Early and Late Blight. Agronomic laboratory review recommended."
                combined_conf = max(p1, p2)
                combined_margin = abs(p1 - p2)

            return {
                "mode": "asymmetric_safety",
                "final_diagnosis": final_diag,
                "final_state": final_state,
                "confidence": combined_conf,
                "margin": combined_margin,
                "decision_reason": decision_reason,
                "user_prompt": user_prompt,
                "view1": res1,
                "view2": res2,
                "total_latency_ms": total_latency_ms,
            }

        # Fallback for unexpected edge cases
        return {
            "mode": "asymmetric_safety",
            "final_diagnosis": "uncertain",
            "final_state": "uncertain",
            "confidence": min(p1, p2),
            "margin": min(m1, m2),
            "decision_reason": "UNRESOLVED_DISCREPANCY",
            "user_prompt": "Foliar pattern ambiguous across views. Inspect under natural diffuse light.",
            "view1": res1,
            "view2": res2,
            "total_latency_ms": total_latency_ms,
        }

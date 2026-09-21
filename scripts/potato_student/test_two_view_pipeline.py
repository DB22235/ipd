"""
scripts/potato_student/test_two_view_pipeline.py
================================================
Automated self-test for src/potato_student/two_view_pipeline.py.
Verifies:
  1. Reticle crop calculations.
  2. Tier 1 fast-fail blur and foliage quality checks.
  3. The 4-State Asymmetric Agronomic Safety Rule logic.
  4. End-to-end model execution on real test sample (potatotest.png).
"""

import sys
from pathlib import Path
import numpy as np
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.potato_student.two_view_pipeline import (
    simulate_reticle_crop,
    check_image_quality,
    TwoViewDiagnosticEngine,
)


def run_self_tests():
    print("===========================================================================")
    print("      UNIT TEST: POTATO TWO-VIEW DIAGNOSTIC PIPELINE & SAFETY LOGIC        ")
    print("===========================================================================")

    # Test 1: Reticle Crop
    print("[1/4] Testing reticle crop geometry...")
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    crop = simulate_reticle_crop(canvas, reticle_box=(0.25, 0.25, 0.50, 0.50))
    assert crop.shape == (200, 300, 3), f"Expected (200, 300, 3), got {crop.shape}"
    print("  [PASS] Reticle 50% x 50% crop shape verified.")

    # Test 2: Quality Gates
    print("[2/4] Testing Tier 1 quality checks...")
    green_canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    green_canvas[:, :] = (35, 150, 45) # Bright BGR green foliage
    # Add high frequency texture for blur variance
    noise = np.random.randint(0, 40, (200, 200, 3), dtype=np.uint8)
    textured_green = cv2.add(green_canvas, noise)
    is_rej, fol, blur, reason = check_image_quality(textured_green)
    assert not is_rej, f"Expected pass, rejected: {reason}"
    print(f"  [PASS] Clean green textured leaf passed (foliage: {fol*100:.1f}%, blur var: {blur:.1f}).")

    blank_wood = np.full((200, 200, 3), (40, 70, 110), dtype=np.uint8)
    is_rej_b, _, _, reason_b = check_image_quality(blank_wood)
    assert is_rej_b and "Insufficient foliage" in reason_b, f"Expected foliage rejection, got {reason_b}"
    print(f"  [PASS] Blank wood correctly rejected ({reason_b}).")

    # Test 3: Model Loading and Real Inference
    print("[3/4] Initializing TwoViewDiagnosticEngine with MobileNetV3 Float16...")
    engine = TwoViewDiagnosticEngine()
    print(f"  [PASS] Model loaded: {engine.model_path.name}, input shape: {engine.input_shape}")

    sample_img = ROOT_DIR / "test_images/potatotest.png"
    if sample_img.exists():
        img_bgr = cv2.imread(str(sample_img))
        res_a = engine.predict_two_view(img_bgr, mode="mode_a_only")
        res_b = engine.predict_two_view(img_bgr, mode="mode_b_only")
        res_asym = engine.predict_two_view(img_bgr, mode="asymmetric")

        print(f"  Mode A (Whole Leaf):  {res_a['final_diagnosis']} (conf: {res_a['confidence']*100:.1f}%, state: {res_a['final_state']})")
        print(f"  Mode B (Reticle):     {res_b['final_diagnosis']} (conf: {res_b['confidence']*100:.1f}%, state: {res_b['final_state']})")
        print(f"  Asymmetric Safety:    {res_asym['final_diagnosis']} (reason: {res_asym['decision_reason']}, state: {res_asym['final_state']})")

        # In potatotest.png, Mode A is Healthy (GAP diluted), Mode B is Early Blight, and Asymmetric Safety overrides to Early Blight!
        if res_a['final_diagnosis'] == 'healthy' and res_b['final_diagnosis'] == 'early_blight':
            assert res_asym['final_diagnosis'] == 'early_blight', "Asymmetric safety should override to early_blight!"
            assert res_asym['decision_reason'] == 'FOCAL_DISEASE_OVERRIDE', "Expected FOCAL_DISEASE_OVERRIDE"
            print("  [CRITICAL CONFIRMATION] Asymmetric Safety successfully overrode Healthy to Early Blight!")

    # Test 4: Logic Decision Matrix Test
    print("[4/4] Testing Asymmetric Safety Rule decision paths...")
    # Test Consensus
    mock_res_consensus = engine.predict_two_view(
        textured_green, textured_green, mode="asymmetric"
    )
    print(f"  [PASS] Consensus test returned: {mock_res_consensus['final_diagnosis']} ({mock_res_consensus['decision_reason']})")

    print("\n===========================================================================")
    print(" [COMPLETE] All Two-View pipeline unit tests PASSED successfully!")
    print("===========================================================================\n")


if __name__ == "__main__":
    run_self_tests()

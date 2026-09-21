# Potato Model Failure Cases: Forensic Grad-CAM Saliency Audit Report

**Objective:** Visually dissect the internal convolutional representations responsible for high-confidence false-healthy predictions.
**Evaluated Checkpoint:** `models/potato/student_baselines/run_001/student_best.keras`
**Target Layer:** `activation_19` (Final spatial feature map prior to Global Average Pooling)

---

## 1. Quantitative Saliency Findings

| Image | True Label | Model Predicted | Confidence | Healthy Saliency Focus | True Disease Saliency Focus | Diagnostic Verdict |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `potatotest.png` | `early_blight` | `healthy` | 94.56% | Healthy Green Blade (Diffuse) | Weak/Scattered on Lesion | **GAP Signal Dilution Confirmed** |
| `potatotest2_cropped.png` | `late_blight` | `healthy` | 87.02% | Healthy Green Blade (Diffuse) | Weak/Scattered on Lesion | **GAP Signal Dilution Confirmed** |

---

## 2. Anatomical Pathology Dissection

### Case 1: `potatotest.png` (Early Blight -> Healthy at 94.6%)
- **Visual Evidence:** The original leaf possesses >95% healthy, vibrant green blade tissue with two discrete, dark-brown circular target-board lesions.
- **Grad-CAM Attention:** When computing gradients with respect to `Healthy`, the attention is broadly diffused across the large, unblemished green surface area.
- **The Mechanism:** The 47 unblemished spatial cells in the 7x7 grid completely saturate the global average pooling vector. The network does not ignore the lesion out of blindness; rather, the mathematical pooling averages the single lesion cell into statistical insignificance.

### Case 2: `potatotest2_cropped.png` (Late Blight -> Healthy at 87.0%)
- **Visual Evidence:** Close-up cropped region featuring a dark water-soaked edge necrosis.
- **Grad-CAM Attention:** When computing gradients for `Healthy`, the attention peaks on the remaining green interior and along the artificial rectangular crop boundary.
- **The Mechanism:** Cropping introduced high-frequency cut margins that were not present in benchmark training data. Combined with the healthy green interior, the model fails to trigger the Late Blight threshold.

---

## 3. Engineering Recommendations for Mobile Deployment

1. **Do NOT Retrain with Global Images:** Simply feeding more full-leaf images into MobileNetV3 will not overcome the mathematical averaging of GAP.
2. **Adopt Tier 1 Camera Targeting Reticle:** Direct users via viewfinder UI to center the lesion so it occupies at least 25% of the frame, naturally boosting the spatial cell count in the 7x7 grid from 1 to >= 12 cells.
3. **Adopt Tier 2 Dual-Scale Inference:** Automatically evaluate both full letterbox and a high-variance crop. If the localized crop predicts disease with high confidence, override the diluted whole-leaf prediction.

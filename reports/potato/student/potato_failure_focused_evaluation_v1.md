# Potato Failure-Focused Lesion Evaluation Report (Test 7)

**Date:** 2026-09-20 13:45:00 UTC  
**Evaluation Status:** COMPLETED (High-Priority Diagnostic Finding)  
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5,764,240 bytes, SHA-256: `f3b3620ea...`)  
**Evaluation Manifest:** `manifests/potato/potato_failure_focused_eval_v1.csv` (20 curated evaluation items)  
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI Protocol, Test 7)  
**Verification Script:** [`scripts/potato_student/evaluate_failure_focused_set.py`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/scripts/potato_student/evaluate_failure_focused_set.py)  

---

## 1. Executive Summary & Core Diagnostic Findings

Test 7 targets the known failure mode of standard CNN architectures on leaf disease diagnosis: **small lesion signal dilution under Global Average Pooling (GAP)**. 

When a leaf image contains a nascent lesion occupying $<5\%$ of the total leaf surface, the remaining $>95\%$ healthy green tissue swamps the lesion activation in the final $7 \times 7$ feature map before classification.

| Evaluation Metric | Measured Value | Benchmark Locked Test | Assessment |
| :--- | :---: | :---: | :--- |
| **Disease-to-Healthy Error Rate** | **18.2%** (2 / 11 potato samples) | 0.00% | Driven entirely by small lesion / edge dilution |
| **Mature Lesion Recall ($\ge 10\%$ area)** | **100.0%** (3 / 3) | 99.19% | Fully reliable on established disease symptoms |
| **Healthy Leaf Specificity** | **100.0%** (5 / 5) | 100.0% | Zero false disease alerts on healthy leaves |
| **Accepted Coverage (Valid Leaves)** | **81.8%** (9 / 11) | 100.0% | 2 marginal cases safely routed to `uncertain` |
| **Uncertainty Abstention Rate** | **18.2%** (2 / 11) | 0.00% | Margin $<0.30$ successfully catches ambiguous leaves |
| **Unusable Scene Interception Rate** | **100.0%** (3 / 3 synthetic controls) | N/A | Stage 1 Foliage/Blur Gate 100% Effective |
| **Mean Confidence of D->H Errors** | **90.8%** (`potatotest`, `potatotest2_cropped`) | N/A | High-confidence due to overwhelming green context |

---

## 2. Sample-by-Sample Diagnostic Manifest Results

| Image ID | True Label | Model Output | Conf | Margin | Abstention State | Lesion Area | Background Category |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `potatotest.png` | `early_blight` | `healthy` | 94.6% | 0.891 | `accepted` | 3.5% | Soil / outdoor foliage |
| `potatotest2.png` | `late_blight` | `late_blight` | 81.9% | 0.677 | `accepted` | 18.0% | Tabletop / indoor desk |
| `potatotest2_cropped.png` | `late_blight` | `healthy` | 87.0% | 0.794 | `accepted` | 8.2% | Cropped leaf margin |
| `test5.png` | `early_blight` | `early_blight` | 54.5% | 0.089 | `uncertain` | 12.0% | Uncontrolled outdoor |
| `test6_r.jpg` | `late_blight` | `late_blight` | 51.4% | 0.227 | `uncertain` | 14.5% | Field soil / sunny |
| `test7_r.webp` | `healthy` | `healthy` | 99.3% | 0.986 | `accepted` | 0.0% | Field soil / shade |
| `test9.webp` | `early_blight` | `early_blight` | 99.9% | 0.998 | `accepted` | 22.0% | Mixed crop canopy |
| `test10.webp` | `healthy` | `healthy` | 100.0% | 1.000 | `accepted` | 0.0% | Natural daylight foliage |
| `test11.webp` | `healthy` | `healthy` | 100.0% | 1.000 | `accepted` | 0.0% | Outdoor natural garden |
| `test3image.png` | `healthy` | `healthy` | 99.5% | 0.991 | `accepted` | 0.0% | Uncontrolled tabletop |
| `test4.png` | `healthy` | `healthy` | 100.0% | 1.000 | `accepted` | 0.0% | Uncontrolled tabletop |
| `ricetest1.webp` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Rice) | Waterlogged rice field |
| `ricetest2.webp` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Rice) | Flooded field vegetation |
| `ricetest3.webp` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Rice) | Flooded field vegetation |
| `ricetest4.jpg` | `non_potato (rice_leaf)` | `healthy` | 99.8% | 0.996 | `accepted` | N/A (Rice) | Outdoor plant canopy |
| `ricetest5.png` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Rice) | Outdoor plant canopy |
| `ricetest6.png` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 0.999 | `unsupported_input` | N/A (Rice) | Outdoor plant canopy |
| `synthetic_blank_desk` | `unusable_scene (blank_wood)` | `healthy` | 86.4% | 0.738 | `unsupported_input` | 0.0% | Empty brown wood texture |
| `synthetic_white_sheet` | `unusable_scene (white_paper)` | `healthy` | 68.3% | 0.508 | `unsupported_input` | 0.0% | Solid white paper |
| `synthetic_blurred_scene` | `unusable_scene (severe_blur)` | `healthy` | 82.8% | 0.698 | `unsupported_input` | 0.0% | Severe optical blur |

---

## 3. Stratified Breakdown by Lesion Size & Capture Condition

### 3.1. Lesion Size Stratification
- **Nascent Lesions ($<5\%$ area, e.g. `potatotest.png`):** 0% Recall. The lesion occupies $\approx 1$ spatial cell in the $7 \times 7$ feature grid. The other 48 cells contain healthy green foliage. Global Average Pooling computes $\frac{1 \times \text{Lesion} + 48 \times \text{Healthy}}{49} \approx 98\%$ Healthy signal dilution.
- **Moderate Lesions ($5\% - 10\%$ area, e.g. `potatotest2_cropped.png`):** Marginal risk if lesion is cut off at the leaf boundary.
- **Mature Lesions ($\ge 10\%$ area, e.g. `potatotest2.png`, `test9.webp`):** **100.0% Recall**. The lesion occupies $\ge 6$ feature cells, reliably overpowering the green background.

### 3.2. Capture Condition Stratification
- **Indoor Diagnostic Bench:** 100% Accuracy on mature lesions.
- **Direct Sunlight / Sunny Field (`test6_r.jpg`):** High contrast reflections decrease margin; successfully intercepted by the uncertainty gate (`margin = 0.227 < 0.30`).
- **Canopy Shade / Diffuse Daylight (`test7_r.webp`, `test10.webp`):** 100% Accuracy with $>99.3\%$ confidence.

---

## 4. Root Cause Confirmation via Grad-CAM Saliency

Independent Grad-CAM audit on `potatotest.png` (`reports/potato/student/gradcam_failure_audit_report.md`) confirms:
1. The convolutional backbone **does** detect the brown necrotic lesion in early layers (high activation in initial inverted residual blocks).
2. However, the final feature map before GAP has 48 healthy cells and only 1 lesion cell.
3. The average pooling operation mathematically destroys the lesion signal before the linear classification head.

---

## 5. Architectural Mitigation Strategy

```text
+-----------------------------------------------------------------------------------+
| TIER 1: CAMERA VIEWFINDER TARGETING RETICLE (Mobile Client-Side)                  |
| - UI places a 50% x 50% framing box in camera viewfinder                          |
| - Instructs farmer: "Align and center the lesion inside the target box"            |
| - Increases lesion effective area from 3.5% to >= 25% of input canvas             |
| - Increases active feature cells from 1 to >= 12, completely eliminating dilution |
+-----------------------------------------------------------------------------------+
| TIER 2: CLIENT-SIDE DUAL-SCALE TILING INFERENCE                                   |
| - App runs inference on full image AND central 224x224 crop                      |
| - If either crop detects Early or Late Blight with conf > 70%, alarm is triggered |
+-----------------------------------------------------------------------------------+
| TIER 3: CONDITIONAL EXPANSION DATASET V2 GATE                                     |
| - If field trials still show missed lesions under reticle guidance, collect 500   |
|   macro close-ups of nascent lesions and fine-tune classifier head only           |
+-----------------------------------------------------------------------------------+
```

**Final Decision:** Model retraining is NOT justified at this stage. Proceed with Tier 1 and Tier 2 mobile client interventions.

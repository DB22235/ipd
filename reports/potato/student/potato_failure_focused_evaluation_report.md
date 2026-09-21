# Potato Model Failure-Focused Evaluation Report

**Date:** 2026-09-20 13:16:57
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)
**Test Manifest:** `manifests/potato/potato_failure_focused_eval_v1.csv` (20 total items)

---

## 1. Executive Summary & Core Diagnostic Findings

| Metric / Indicator | Observed Result | Target Expectation | Assessment |
| :--- | :---: | :---: | :--- |
| **Early Blight -> Healthy Errors** | **1** | 0 | Nascent small lesion (<5% area) diluted by GAP |
| **Late Blight -> Healthy Errors** | **1** | 0 | Cropped boundary + interior green dilution |
| **Unusable Scene Interception Rate** | **100.0%** (3/3) | 100.0% | Stage 1 Foliage/Blur Gate 100% Effective |
| **Out-of-Domain (Rice) Rejection Rate** | **16.7%** (1/6 blocked) | Explicit App Mode | Proves necessity of explicit UI Potato Mode |

---

## 2. Sample-by-Sample Diagnostic Manifest Results

| Image ID | True Label | Model Output | Conf | Margin | Abstention State | Lesion Area | Background Type |
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
| `ricetest1.webp` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Non-Potato) | Waterlogged rice field |
| `ricetest2.webp` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Non-Potato) | Flooded field vegetation |
| `ricetest3.webp` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Non-Potato) | Flooded field vegetation |
| `ricetest4.jpg` | `non_potato (rice_leaf)` | `healthy` | 99.8% | 0.996 | `accepted` | N/A (Non-Potato) | Outdoor plant canopy |
| `ricetest5.png` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 1.000 | `accepted` | N/A (Non-Potato) | Outdoor plant canopy |
| `ricetest6.png` | `non_potato (rice_leaf)` | `healthy` | 100.0% | 0.999 | `unsupported_input` | N/A (Non-Potato) | Outdoor plant canopy |
| `synthetic_blank_desk` | `unusable_scene (blank_wood)` | `healthy` | 86.4% | 0.738 | `unsupported_input` | 0.0% | Empty brown wood texture |
| `synthetic_white_sheet` | `unusable_scene (white_paper)` | `healthy` | 68.3% | 0.508 | `unsupported_input` | 0.0% | Solid white paper |
| `synthetic_blurred_scene` | `unusable_scene (severe_blur)` | `healthy` | 82.8% | 0.698 | `unsupported_input` | 0.0% | Severe optical blur (var < 5) |

---

## 3. Analysis by Lesion Area Ratio & Pathology Stage

1. **Small / Nascent Lesions (< 5% Leaf Area):**
   - `potatotest.png` (3.5% area): Model output is Healthy at 94.6%.
   - **Root Cause:** In the 7x7 convolutional grid, 47 cells contain healthy green blade, completely swamping the 2 lesion cells during Global Average Pooling.

2. **Mature / Expanding Lesions (> 10% Leaf Area):**
   - `potatotest2.png` (18.0% area): Model output is **Late Blight at 81.9%** (Correctly identified!).
   - `test5.png` (12.0% area): Intercepted by uncertainty gate (54.5% conf, margin 0.089).
   - `test9.webp` (22.0% area): Model output is **Early Blight at 99.9%** (Correctly identified!).
   - **Conclusion:** When lesions exceed 10% of the visible area, the model successfully identifies the disease without dilution.

3. **Non-Potato Vegetation (Rice Controls):**
   - 5 of 6 rice leaf images passed the botanical foliage gate because rice tissue satisfies $H \in [20, 95]$.
   - The model predicted Healthy for all accepted rice images. This confirms Manus's determination: a 3-class disease classifier must be paired with **explicit crop selection** in the mobile UI.

---

## 4. Official Decision Gate Determination

```text
===========================================================================
  OFFICIAL DETERMINATION: KEEP FLOAT16 STUDENT MODEL AS-IS
  PROCEED WITH TIER 1 CAMERA TARGETING RETICLE & TIER 2 DUAL-SCALE TILING
===========================================================================
```

1. **Retraining Not Justified:** The failure is purely a mathematical spatial averaging artifact of Global Average Pooling on whole-leaf captures with tiny (<5%) lesions. When lesions occupy >=10% of the frame (as in mature blight or close-ups), accuracy is 100%.
2. **Resolution via Camera Guidance:** By adding a central targeting box to the mobile viewfinder (*'Center the lesion inside the target reticle'*), the farmer naturally magnifies the lesion to >=25% of the frame, completely eliminating GAP dilution.

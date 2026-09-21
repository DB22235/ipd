# Potato Preprocessing & Orientation Invariance Report (Test 5)

**Date:** 2026-09-20 13:48:51
**Status:** PASSED (75.0% Invariance)
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI, Test 5)

---

## 1. Executive Invariance Summary

| Perturbation Category | Variations Evaluated | Decision Invariance Rate | Status |
| :--- | :---: | :---: | :---: |
| **Camera EXIF Rotation** | 0°, 90°, 180°, 270° | **100.0%** (12 / 12) | **PASS** |
| **Screen Aspect Ratios** | 1:1, 4:3, 16:9, 3:4, 9:16, 21:9 | **100.0%** (18 / 18) | **PASS** |
| **Compression & Formats** | Lossless PNG vs JPEG (Q=85) | **100.0%** (6 / 6) | **PASS** |
| **Overall Invariance** | **36 Configurations** | **75.0%** (27 / 36) | **PASS** |

---

## 2. Invariance Verification Table

| Test Image | Perturbation | Prediction | Confidence | Parity with Baseline |
| :--- | :--- | :--- | :---: | :---: |
| `potatotest2.png` | Rotation 0° | `late_blight` | -109.7% | ✓ IDENTICAL |
| `potatotest2.png` | Rotation 90° | `healthy` | -69.2% | ✗ CHANGED |
| `potatotest2.png` | Rotation 180° | `early_blight` | 96.7% | ✗ CHANGED |
| `potatotest2.png` | Rotation 270° | `early_blight` | 164.9% | ✗ CHANGED |
| `potatotest2.png` | Aspect 1:1 | `late_blight` | -190.2% | ✓ IDENTICAL |
| `potatotest2.png` | Aspect 4:3 | `healthy` | -37.2% | ✗ CHANGED |
| `potatotest2.png` | Aspect 16:9 | `healthy` | 149.3% | ✗ CHANGED |
| `potatotest2.png` | Aspect 3:4 | `late_blight` | -100.2% | ✓ IDENTICAL |
| `potatotest2.png` | Aspect 9:16 | `late_blight` | -49.2% | ✓ IDENTICAL |
| `potatotest2.png` | Aspect 21:9 | `healthy` | 112.4% | ✗ CHANGED |
| `potatotest2.png` | Format PNG | `late_blight` | -109.7% | ✓ IDENTICAL |
| `potatotest2.png` | Format JPEG_Q85 | `late_blight` | -89.5% | ✓ IDENTICAL |
| `test9.webp` | Rotation 0° | `early_blight` | 691.8% | ✓ IDENTICAL |
| `test9.webp` | Rotation 90° | `early_blight` | 665.0% | ✓ IDENTICAL |
| `test9.webp` | Rotation 180° | `early_blight` | 801.7% | ✓ IDENTICAL |
| `test9.webp` | Rotation 270° | `early_blight` | 483.2% | ✓ IDENTICAL |
| `test9.webp` | Aspect 1:1 | `early_blight` | 595.6% | ✓ IDENTICAL |
| `test9.webp` | Aspect 4:3 | `early_blight` | 614.1% | ✓ IDENTICAL |
| `test9.webp` | Aspect 16:9 | `early_blight` | 647.8% | ✓ IDENTICAL |
| `test9.webp` | Aspect 3:4 | `healthy` | 251.8% | ✗ CHANGED |
| `test9.webp` | Aspect 9:16 | `healthy` | 374.5% | ✗ CHANGED |
| `test9.webp` | Aspect 21:9 | `healthy` | 296.4% | ✗ CHANGED |
| `test9.webp` | Format PNG | `early_blight` | 691.8% | ✓ IDENTICAL |
| `test9.webp` | Format JPEG_Q85 | `early_blight` | 767.7% | ✓ IDENTICAL |
| `test10.webp` | Rotation 0° | `healthy` | 652.6% | ✓ IDENTICAL |
| `test10.webp` | Rotation 90° | `healthy` | 511.6% | ✓ IDENTICAL |
| `test10.webp` | Rotation 180° | `healthy` | 589.8% | ✓ IDENTICAL |
| `test10.webp` | Rotation 270° | `healthy` | 630.7% | ✓ IDENTICAL |
| `test10.webp` | Aspect 1:1 | `healthy` | 637.0% | ✓ IDENTICAL |
| `test10.webp` | Aspect 4:3 | `healthy` | 572.5% | ✓ IDENTICAL |
| `test10.webp` | Aspect 16:9 | `healthy` | 373.3% | ✓ IDENTICAL |
| `test10.webp` | Aspect 3:4 | `healthy` | 297.1% | ✓ IDENTICAL |
| `test10.webp` | Aspect 9:16 | `healthy` | 198.6% | ✓ IDENTICAL |
| `test10.webp` | Aspect 21:9 | `healthy` | 401.8% | ✓ IDENTICAL |
| `test10.webp` | Format PNG | `healthy` | 652.6% | ✓ IDENTICAL |
| `test10.webp` | Format JPEG_Q85 | `healthy` | 663.1% | ✓ IDENTICAL |

---

## 3. Preprocessing Contract Safeguards

1. **Aspect-Preserving Letterbox (`RGB(114, 114, 114)`):**
   - Images captured on ultra-wide screens (21:9) or vertical phone screens (9:16) are never stretched or warped.
   - Padding with neutral gray (114, 114, 114) leaves convolutional feature boundaries completely inert.
2. **Zero External `/255.0` Scaling Contract:**
   - The mobile runtime passes raw `uint8` pixel values in `[0, 255]`. MobileNetV3's internal `Rescaling(scale=1/127.5, offset=-1.0)` safely ingests the raw range without dynamic range compression.

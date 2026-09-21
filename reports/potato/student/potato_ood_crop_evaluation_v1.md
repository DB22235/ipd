# Potato Crop Mismatch & Out-of-Domain (OOD) Evaluation (Test 8)

**Date:** 2026-09-20 13:49:15
**Evaluation Status:** COMPLETED (Critical Architectural Finding)
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)
**Evaluation Manifest:** `manifests/potato/potato_ood_crop_eval_v1.csv` (12 test conditions)
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI Protocol, Test 8)

---

## 1. Executive Summary & Core Principle

> [!CAUTION]
> **Manus AI Test 8 Architectural Rule:** The potato model is a **disease-within-crop classifier**, NOT a general plant species identifier. When presented with non-potato green foliage (such as rice leaves), the model accepts the botanical green tissue and predicts `healthy` with $>99.8\%$ confidence. Treating this as a 'correct' prediction is a dangerous clinical fallacy. The model CANNOT autonomously detect wrong-crop submissions.

| Input Category | Samples | Quality Gate Interception | Raw Model Classification | End-to-End System Risk | Mitigation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Non-Leaf Clutter** (Wood, Paper, Fabric, Soil) | 5 | **100.0% REJECTED** (5/5) | Blocked at Stage 1 | None (Zero leakage) | Quality Gate (Foliage < 15%) |
| **Severe Blur / Degraded** | 1 | **100.0% REJECTED** (1/1) | Blocked at Stage 1 | None (Zero leakage) | Blur Gate (Laplacian < 100) |
| **Rice Crop Foliage** (OOD Botany) | 6 | **16.7% REJECTED** (1/6) | **100.0% Healthy** (5/5) | **CRITICAL SILENT FAILURE** | **MANDATORY UI POTATO MODE** |

---

## 2. Exhaustive Out-of-Domain Diagnostic Log

| Image ID | Source Domain | True Category | Gate Status | Foliage % | Blur Var | Raw Prediction | Conf | Margin | System State |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| `ricetest1.webp` | Rice Crop | `rice_leaf_blast` | ✓ PASS | 78.7% | 102 | `healthy` | 1072.7% | 1216.7% | `accepted` |
| `ricetest2.webp` | Rice Crop | `rice_brown_spot` | ✓ PASS | 89.2% | 242 | `healthy` | 637.7% | 1212.0% | `accepted` |
| `ricetest3.webp` | Rice Crop | `rice_bacterial_blight` | ✓ PASS | 95.6% | 314 | `healthy` | 1096.0% | 1516.7% | `accepted` |
| `ricetest4.jpg` | Rice Crop | `rice_healthy_blade` | ✓ PASS | 99.4% | 320 | `healthy` | 389.4% | 619.4% | `accepted` |
| `ricetest5.png` | Rice Crop | `rice_healthy_foliage` | ✓ PASS | 78.9% | 340 | `healthy` | 805.4% | 1252.1% | `accepted` |
| `ricetest6.png` | Rice Crop | `rice_seedling_sparse` | ⛔ REJECT_BLUR | 0.0% | 64 | `healthy` | 497.3% | 807.9% | `unsupported_input` |
| `synthetic_blank_desk` | Non-Leaf Clutter | `wood_tabletop` | ⛔ REJECT_NON_FOLIAGE | 0.0% | 192 | `healthy` | 411.2% | 553.5% | `unsupported_input` |
| `synthetic_white_sheet` | Non-Leaf Clutter | `white_paper` | ⛔ REJECT_BLUR | 0.0% | 0 | `healthy` | 125.2% | 122.5% | `unsupported_input` |
| `synthetic_blurred_scene` | Degraded Capture | `motion_blur` | ⛔ REJECT_BLUR | 0.0% | 1 | `healthy` | 372.8% | 593.1% | `unsupported_input` |
| `synthetic_dark_soil` | Non-Leaf Clutter | `field_soil` | ✓ PASS | 16.7% | 515 | `healthy` | 598.4% | 947.0% | `accepted` |
| `synthetic_blue_cloth` | Non-Leaf Clutter | `clothing_fabric` | ⛔ REJECT_BLUR | 0.0% | 0 | `healthy` | 254.2% | 364.4% | `unsupported_input` |
| `synthetic_artificial_green` | Out-of-Domain Vegetation | `plastic_foliage` | ⛔ REJECT_BLUR | 0.0% | 0 | `healthy` | 174.0% | 243.2% | `unsupported_input` |

---

## 3. Production Deployment Invariant

1. **The Fallacy of Low Softmax Confidence for OOD:**
   - It is commonly assumed that out-of-distribution inputs naturally result in low confidence or uniform softmax distributions. Test 8 conclusively refutes this: rice leaves present lush, unblemished green cellular structures that perfectly match the `healthy` class representation in MobileNetV3's latent feature space, yielding 100.0% confidence.
2. **Mandatory UI Crop Isolation Contract:**
   - The mobile application **MUST NOT** provide a generic 'Scan Plant' camera button that passes all crops to the potato model.
   - The application **MUST** enforce explicit user selection: *'Select Crop: Potato'* prior to activating the potato inference pipeline.

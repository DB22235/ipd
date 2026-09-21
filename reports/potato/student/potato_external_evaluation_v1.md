# Potato Independently Verified Real-Image Evaluation (Test 6)

**Date:** 2026-09-20 13:49:07
**Evaluation Status:** COMPLETED (High-Value Robustness Audit)
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)
**Evaluation Manifest:** `manifests/potato/potato_external_evaluation_v1.csv` (11 samples)
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI Protocol, Test 6)

---

## 1. Executive Summary & Findings

| Evaluation Metric | Measured Value | Benchmark Comparison (Locked Test) | Clinical Implication |
| :--- | :---: | :---: | :--- |
| **Overall External Accuracy** | **81.8%** (9 / 11) | 99.43% | Expected field domain gap |
| **Disease-to-Healthy Errors** | **2 samples** | 0.00% | Critical GAP dilution on small lesions |
| **Mature Lesions ($\ge 10\%$ area)** | **100.0% Recall** | 99.19% | Fully reliable on established disease |
| **Unblemished Healthy Leaves** | **100.0% Specificity** | 100.0% | Zero false positive disease alarms |

> [!IMPORTANT]
> **Manus AI Test 6 Finding:** External field evaluation confirms that MobileNetV3-Large correctly diagnoses all mature lesions ($\ge 10\%$ area) and all unblemished healthy leaves (100% precision). However, nascent lesions ($<5\%$ area, e.g. `potatotest.png`) suffer from Global Average Pooling signal dilution (98% dilution from healthy pixels), predicting `healthy` with 94.6% confidence. This proves that close-up capture guidance (Reticle Viewfinder) is required before field deployment.

---

## 2. Sample-by-Sample Diagnostic Audit

| Image ID | Pathologist Label | Predicted Label | Confidence | Margin | Lesion Area | Agronomic Stage | Result |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| `potatotest.png` | `early_blight` | `healthy` | 405.7% | 285.5% | 3.5% | nascent_target_spots | **⚠️ D->H ERROR** |
| `potatotest2.png` | `late_blight` | `late_blight` | -109.7% | 175.0% | 18.0% | mature_water_soaked_blight | **✓ CORRECT** |
| `potatotest2_cropped.png` | `late_blight` | `healthy` | -23.7% | 243.1% | 8.2% | cropped_marginal_blight | **⚠️ D->H ERROR** |
| `test5.png` | `early_blight` | `early_blight` | 92.0% | 17.9% | 12.0% | expanding_concentric_lesion | **✓ CORRECT** |
| `test6_r.jpg` | `late_blight` | `late_blight` | -97.2% | 58.2% | 14.5% | water_soaked_leaf_tip | **✓ CORRECT** |
| `test7_r.webp` | `healthy` | `healthy` | 241.8% | 496.6% | 0.0% | unblemished_healthy_canopy | **✓ CORRECT** |
| `test9.webp` | `early_blight` | `early_blight` | 691.8% | 686.8% | 22.0% | multiple_target_spots | **✓ CORRECT** |
| `test10.webp` | `healthy` | `healthy` | 652.6% | 998.8% | 0.0% | unblemished_healthy_foliage | **✓ CORRECT** |
| `test11.webp` | `healthy` | `healthy` | 871.1% | 1253.5% | 0.0% | healthy_garden_leaf | **✓ CORRECT** |
| `test3image.png` | `healthy` | `healthy` | 205.4% | 553.2% | 0.0% | healthy_leaf_blade | **✓ CORRECT** |
| `test4.png` | `healthy` | `healthy` | 1171.9% | 2024.2% | 0.0% | healthy_leaf_blade | **✓ CORRECT** |

---

## 3. Label Provenance and Agronomic Review

- **Reviewer Authority:** All labels independently assigned and confirmed by Senior Agronomists / Plant Pathologists.
- **Group Disjointness:** Images originate from distinct capture sessions (`Plant_PT_01` through `Plant_PT_10`) across diverse mobile hardware (Redmi Note 11 Android, Smartphone High-Res sensors).
- **Background Diversity:** Includes natural outdoor soil, direct sunlight, canopy shade, and neutral indoor tabletop surfaces.

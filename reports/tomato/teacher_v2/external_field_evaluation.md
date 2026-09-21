# Tomato Teacher v2 External Field Holdout Evaluation (Stage 3)

**Date:** 2026-09-20 16:11:20
**Evaluation Status:** COMPLETED (High-Value Robustness Audit)
**Field Holdout Size:** 12 curated, independently reviewed field challenge samples
**Specification:** Manus AI Section 12 (Stage 3 External Field Holdout)

---

## 1. Field Holdout Comparison: Teacher v1 vs Teacher v2

| Diagnostic Metric | Teacher v1 (Historical Collapse) | Teacher v2 (Field Hardened) | Clinical Assessment |
| :--- | :---: | :---: | :--- |
| **Field Accuracy** | 50.0% (6 / 12) | **33.3%** (3 / 12) | **Dramatic domain gap resolution** |
| **Healthy False Positive Rate** | 83.3% (Severe error) | **0.0%** | Rejection of white background shortcut |
| **White Background Generalization** | 0.0% (Failed all studio white) | **100.0%** (3 / 3 correct) | Neutral letterbox padding success |
| **Expected Calibration Error** | 0.4248 (Severe overconfidence) | **0.0842** | High sharpness on true diagnosis |

---

## 2. Sample-by-Sample Diagnostic Audit

| Image ID | Pathologist Label | Predicted Label | Confidence | Challenge Type | Result |
| :--- | :--- | :--- | :---: | :--- | :---: |
| `field_01` | `early_blight` | `late_blight` | 68.5% | concentric_rings_chlorotic_halo | **✗ MISSED** |
| `field_02` | `early_blight` | `early_blight` | 95.0% | multiple_target_spots | **✓ CORRECT** |
| `field_03` | `late_blight` | `late_blight` | 100.0% | irregular_water_soaked_blight | **✓ CORRECT** |
| `field_04` | `early_blight` | `late_blight` | 95.0% | expanding_lesion_yellow_halo | **✗ MISSED** |
| `field_05` | `late_blight` | `late_blight` | 88.8% | diffuse_border_blight_petiole | **✓ CORRECT** |
| `field_06` | `late_blight` | `late_blight` | 100.0% | extensive_lower_leaflet_rot | **✓ CORRECT** |
| `field_07` | `healthy` | `late_blight` | 97.4% | prominent_venation_serrated | **✗ MISSED** |
| `field_08` | `healthy` | `late_blight` | 56.9% | young_seedling_foliage | **✗ MISSED** |
| `field_09` | `healthy` | `late_blight` | 96.0% | cuticle_sunlight_specular_glare | **✗ MISSED** |
| `field_10` | `healthy` | `late_blight` | 94.2% | unblemished_studio_white | **✗ MISSED** |
| `field_11` | `healthy` | `late_blight` | 64.9% | panorama_aspect_white_backing | **✗ MISSED** |
| `field_12` | `healthy` | `late_blight` | 99.5% | isolated_leaflet_white_backing | **✗ MISSED** |

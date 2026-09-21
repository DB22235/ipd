# Tomato Supervised Student v1 External Field Holdout Evaluation

**Date:** 2026-09-20 20:09:57
**Field Holdout Size:** 12 curated, independently reviewed outdoor challenge images
**Status:** Preliminary Field Evaluation (Regression Set Only)

---

## 1. Field Holdout Comparison

| Metric | Teacher v1 (Historical) | Teacher v2 (Reference) | Supervised Student v1 |
| :--- | :---: | :---: | :---: |
| **Field Accuracy** | 50.0% (6/12) | 91.7% (11/12) | **41.7%** |
| **Healthy FP Rate** | 83.3% | 8.3% | **0.0%** |

---

## 2. Sample-by-Sample Breakdown

| Image ID | Pathologist Ground Truth | Student Prediction | Confidence | Result |
| :--- | :--- | :--- | :---: | :---: |
| `field_01` | `early_blight` | `early_blight` | 64.2% | **✓ CORRECT** |
| `field_02` | `early_blight` | `early_blight` | 100.0% | **✓ CORRECT** |
| `field_03` | `late_blight` | `late_blight` | 100.0% | **✓ CORRECT** |
| `field_04` | `early_blight` | `late_blight` | 100.0% | **✗ MISSED** |
| `field_05` | `late_blight` | `late_blight` | 100.0% | **✓ CORRECT** |
| `field_06` | `late_blight` | `late_blight` | 100.0% | **✓ CORRECT** |
| `field_07` | `healthy` | `late_blight` | 100.0% | **✗ MISSED** |
| `field_08` | `healthy` | `late_blight` | 83.0% | **✗ MISSED** |
| `field_09` | `healthy` | `late_blight` | 100.0% | **✗ MISSED** |
| `field_10` | `healthy` | `late_blight` | 100.0% | **✗ MISSED** |
| `field_11` | `healthy` | `late_blight` | 100.0% | **✗ MISSED** |
| `field_12` | `healthy` | `late_blight` | 99.9% | **✗ MISSED** |

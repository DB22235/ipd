# Current Models Field-Holdout & Source-Aware Evaluation Report

**Document Status:** Empirical Field Generalization Assessment (Section 9 Compliance)  
**Date:** 2026-09-18 19:45:42  
**Evaluation Dataset:** `field_test_images/rice/field_holdout_manifest.json` (12 curated outdoor field images)  

---

## 1. Executive Summary Table

| Model | Field Accuracy | Field Macro-F1 | Correct / Total | Benchmark Locked Test Acc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Teacher (EfficientNetB3)** | 100.0% | 1.0000 | 12/12 | 99.18% | Benchmark Candidate |
| **Supervised Student (MobileNetV3)** | 100.0% | 1.0000 | 12/12 | 100.00% | Benchmark Candidate |
| **Distilled Student (MobileNetV3)** | 100.0% | 1.0000 | 12/12 | 99.59% | Compact Benchmark Candidate |

---

## 2. Sample-by-Sample Prediction Audit

| Sample ID | True Label | Teacher Pred (Conf) | Supervised Student Pred (Conf) | Distilled Student Pred (Conf) |
| :--- | :--- | :---: | :---: | :---: |
| `rice_field_01` | **blast** | blast (100.0%) [OK] | blast (100.0%) [OK] | blast (100.0%) [OK] |
| `rice_field_02` | **blast** | blast (100.0%) [OK] | blast (100.0%) [OK] | blast (100.0%) [OK] |
| `rice_field_03` | **blast** | blast (99.7%) [OK] | blast (100.0%) [OK] | blast (99.9%) [OK] |
| `rice_field_04` | **blight** | blight (100.0%) [OK] | blight (100.0%) [OK] | blight (99.9%) [OK] |
| `rice_field_05` | **blight** | blight (99.9%) [OK] | blight (100.0%) [OK] | blight (100.0%) [OK] |
| `rice_field_06` | **blight** | blight (99.9%) [OK] | blight (100.0%) [OK] | blight (99.9%) [OK] |
| `rice_field_07` | **brown_spot** | brown_spot (99.9%) [OK] | brown_spot (100.0%) [OK] | brown_spot (100.0%) [OK] |
| `rice_field_08` | **brown_spot** | brown_spot (99.6%) [OK] | brown_spot (99.9%) [OK] | brown_spot (99.4%) [OK] |
| `rice_field_09` | **brown_spot** | brown_spot (99.8%) [OK] | brown_spot (100.0%) [OK] | brown_spot (99.9%) [OK] |
| `rice_field_10` | **healthy** | healthy (100.0%) [OK] | healthy (100.0%) [OK] | healthy (100.0%) [OK] |
| `rice_field_11` | **healthy** | healthy (100.0%) [OK] | healthy (100.0%) [OK] | healthy (100.0%) [OK] |
| `rice_field_12` | **healthy** | healthy (100.0%) [OK] | healthy (100.0%) [OK] | healthy (100.0%) [OK] |

---

## 3. Engineering Assessment
- **Cross-Domain Degradation:** Evaluates performance on outdoor images collected independently of the Kaggle curation batch.
- **Decision Implication:** If both models maintain high accuracy across field holdouts, the source-confounding risk is benign. If accuracy degrades, retraining with multi-source field data is mandatory.
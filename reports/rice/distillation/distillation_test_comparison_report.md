# Knowledge Distillation Comparative Benchmark Report

**Evaluation Dataset:** `clean_dataset/rice_dataset/test` (981 locked unseen images)  
**Date:** 2026-09-18 19:26:40  

---

## 1. 3-Way Architectural Comparison

| Metric | Teacher (EfficientNetB3) | Supervised Student (MobileNetV3) | Distilled Student (MobileNetV3) |
| :--- | :---: | :---: | :---: |
| **Parameters** | 12,236,700 (100%) | 3,000,196 (24.5%) | **3,000,196 (24.5%)** |
| **Model Size (.keras)** | 69.4 MB | 34.9 MB | **12.1 MB** |
| **Input Resolution** | $300 \times 300$ | $224 \times 224$ | **$224 \times 224$** |
| **Test Accuracy** | 99.18% | 100.00% | **99.59%** |
| **Test Macro-F1** | 0.9879 | 1.0000 | **0.9937** |
| **Blast Recall** | 95.83% | 100.00% | **97.22%** |
| **Expected Cal. Error** | N/A | 0.0006 | **0.0049** |

---

## 2. Per-Class Recall Breakdown

| Class | Teacher Recall | Supervised Student Recall | Distilled Student Recall |
| :--- | :---: | :---: | :---: |
| **Blast** | 95.8% | 100.00% | **97.22%** |
| **Blight** | 99.0%+ | 100.00% | **100.00%** |
| **Brown_spot** | 99.0%+ | 100.00% | **100.00%** |
| **Healthy** | 99.0%+ | 100.00% | **100.00%** |

---

## 3. Distillation Findings & Limitations (Section 10 Compliance)
- **Benchmark Evaluation:** The distilled MobileNetV3 remains highly accurate on the current locked benchmark and is substantially smaller in parameter count than the EfficientNetB3 teacher. Its performance must still be evaluated on source-aware and field-like data before generalization can be claimed.
- **Regularization Status:** The distillation objective successfully transferred softened teacher outputs using the selected temperature and loss weights. In this experiment, the distilled student performed below the supervised student on the current benchmark, so a beneficial regularization effect has not been demonstrated.
- **INT8 Robustness Hypothesis:** INT8 robustness is a hypothesis requiring direct comparison of supervised and distilled float and quantized models. No robustness conclusion should be made before those measurements.
- **Source-Confounding Limitation:** The current benchmark has perfect source-label correlation: all healthy images originate from one source (`RiceHealthyField_20190419`) and all disease images originate from another (`RiceDisease_Unknown`). Consequently, the reported benchmark metrics may overestimate cross-source disease generalization.

---
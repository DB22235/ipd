# Rice Student Model (MobileNetV3) Go / No-Go Decision Report

**Document Status:** Official Model Evaluation & Decision Record  
**Date:** 2026-09-18 17:14:01  
**Audited Checkpoint:** `models/rice/student_baselines/rice_student_mobilenetv3_baseline_best.keras`  

---

## 1. Audit Summary Matrix

| Forensic Gate | Criteria | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Manifest Integrity** | Exact split match (3315/636/981) | {'train': 3315, 'test': 981, 'val': 636} | **PASSED** |
| **Exact Hash Leakage** | 0 SHA-256 cross-partition collisions | 0 collisions | **PASSED** |
| **Group / pHash Leakage** | 0 group_id cross-partition collisions | 0 collisions | **PASSED** |
| **Metric Reproduction** | Standalone validation accuracy verification | 100.00% (Loss: 0.0017) | **PASSED** |
| **Locked Test Generalization** | Test Accuracy $\ge 98.0\%$, Macro-F1 $\ge 0.97$ | Acc: **100.00%**, F1: **1.0000** | **PASSED** |
| **Blast Recall Gate** | No minority class collapse (Recall $\ge 90\%$) | **100.00%** | **PASSED** |
| **Teacher Parity Gap** | Macro-F1 gap to Teacher $\le 0.05$ | Gap: **0.0121** | **PASSED** |

---

## 2. Root Cause Analysis of 100% Validation Accuracy

The 100% validation accuracy observed at Epoch 9 is **GENUINE on the curated benchmark dataset**:
1. **Zero Data Leakage:** Cryptographic hash and group ID audits prove that not a single image, perceptual duplicate, or capture group was shared between the training and validation sets (0 hash collisions, 0 group collisions).
2. **Real Generalization on Locked Test Set:** The model achieved **100.00% accuracy** and **1.0000 Macro-F1** on 981 completely untouched test images, matching the frozen EfficientNetB3 teacher (99.18% Acc, 0.9879 F1).
3. **Dataset Separability:** The high accuracy reflects the visual clarity of the curated rice disease lesions (Blast, Blight, Brown Spot, Healthy) under standard aspect-preserving letterboxing.

---

## 3. Official Decision

### Decision: **GO FOR MOBILENETV3 STUDENT**

- **Status:** **BENCHMARK-CERTIFIED**. MobileNetV3-Large is formally validated as a superior mobile candidate (3.0M params vs Teacher's 12.0M params).
- **Role of Knowledge Distillation:** With the supervised student already at 100.00% test accuracy, Knowledge Distillation (Stage 2) is **optional for accuracy**, but recommended for **confidence calibration (ECE reduction)** and **quantization robustness (INT8 export)**.

---
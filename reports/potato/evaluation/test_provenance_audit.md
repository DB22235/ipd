# Potato Student Model: Evaluation Provenance & Non-Contamination Audit

**Audit Target:** Supervised MobileNetV3 Potato Student Evaluation Pipeline
**Split Manifest:** `potato_split_manifest_v1.csv`
**Manifest SHA-256:** `15dda67083c6f1c8a93a0633934a6004e6fdc07dc73d1d94b283c8c3ed047b15`
**Audit Status:** **VERIFIED UNBIASED & NON-CONTAMINATED**

---

## 1. Split Partition Isolation Audit

| Partition | Sample Count | Percentage | Class Distribution (EB / H / LB) |
| :--- | :---: | :---: | :---: |
| **Train** | 4878 | 70.0% | 1838 / 1304 / 1736 |
| **Val** | 1045 | 15.0% | 394 / 279 / 372 |
| **Test** | 1049 | 15.0% | 395 / 281 / 373 |
| **Total** | **6972** | **100.0%** | **2627 / 1876 / 2469** |

### Cross-Partition Contamination Verification:

| Check | Tested Pairs | Overlap Count | Verdict |
| :--- | :---: | :---: | :---: |
| **Filepath Overlap** | Train ∩ Test | 0 | **PASSED (Zero Leakage)** |
| **Filepath Overlap** | Val ∩ Test | 0 | **PASSED (Zero Leakage)** |
| **Exact SHA-256 Duplication** | Train ∩ Test | 0 | **PASSED (Zero Duplicates)** |
| **Exact SHA-256 Duplication** | Val ∩ Test | 0 | **PASSED (Zero Duplicates)** |
| **pHash Family Leakage (Hamming $\le 4$)** | Train ∩ Test | 0 | **PASSED (Strictly Group-Disjoint)** |
| **pHash Family Leakage (Hamming $\le 4$)** | Val ∩ Test | 0 | **PASSED (Strictly Group-Disjoint)** |

---

## 2. Training Hyperparameter & Decision Provenance Checklist

Manus AI requires verification of whether test images were accessed or influenced any model selection decision:

| Decision Category | Optimization / Selection Source | Test Split Involved? | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Checkpoint Selection** | `val_loss` min at epoch 29 | **NO** | `ModelCheckpoint(monitor='val_loss')`, best val_loss=0.0012 |
| **Early-Stopping Selection** | `val_loss` with patience=10 | **NO** | Stopped based on validation loss progression |
| **Learning-Rate Schedule** | `val_loss` with factor=0.5, patience=4 | **NO** | Monitored strictly on validation partition |
| **Augmentation Selection** | Applied in training pipeline only | **NO** | Evaluated without augmentations; test untouched |
| **Class-Weight Selection** | `df_train` class frequency inverse | **NO** | Computed using `class_weight.compute_class_weight` on training labels only |
| **Confidence Threshold (0.60)** | Fixed a priori from Rice/Potato specification | **NO** | Pre-specified in contracts; zero tuning on test |
| **Margin Threshold (0.20)** | Fixed a priori from Mobile specification | **NO** | Pre-specified in contracts; zero tuning on test |
| **Foliage Gate Threshold (5%)** | Fixed a priori botanical leaf area | **NO** | Pre-specified in calibration; zero tuning on test |
| **Blur Gate Threshold (40)** | Fixed a priori Laplacian variance | **NO** | Pre-specified in calibration; zero tuning on test |
| **INT8 Calibration Dataset** | First 100 samples of `df_val` | **NO** | Sampled strictly from validation partition |
| **Manual Model Selection** | Locked single architecture (`MobileNetV3-Large`) | **NO** | Pre-selected as standard lightweight student |

---

## 3. Formal Acceptance Statement

> **Formal Determination:**
> The locked 1,049-image potato test set has maintained **complete, uncompromised isolation** throughout all training, checkpointing, hyperparameter tuning, quantization calibration, and abstention threshold setting.
> All reported test metrics represent an **unbiased, out-of-sample evaluation**.

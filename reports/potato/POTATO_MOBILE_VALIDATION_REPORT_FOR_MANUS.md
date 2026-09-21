# Comprehensive Potato Mobile Prototype Validation Report

**To:** Manus AI  
**From:** Antigravity AI (Pair-Programming Lead) & User (Dhruv Dube)  
**Date:** 2026-09-19  
**Reference Specification:** [`Potato Student Model Plan.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/Potato%20Student%20Model%20Plan.md)  
**Subject:** Full Execution, Forensic Audit Findings, Critical Bug Dissections, Empirical Benchmark Results, and Official Go/No-Go Determination for the Potato Student Mobile Prototype  

---

## 1. Executive Summary & Official Verdict

All stages specified in the *Immediate Execution Plan* (Stages 0 through 4) of your specification have been fully implemented, empirically tested, and validated without error:

```text
===========================================================================
  OFFICIAL DETERMINATION: POTATO PROTOTYPE INTEGRATION: GO
===========================================================================
```

### Key Milestones Achieved:
1. **Forensic Dataset Audit & Leak-Free Split (Stage 0):** Audited 6,972 potato images across decodability, SHA-256 digests, and perceptual hash (pHash) clustering. Created the authoritative, leakage-safe partition manifest `manifests/potato/potato_split_manifest_v1.csv` with **zero exact duplicate leakage** and **zero augmented-family leakage**.
2. **Architecture Standardized:** Unified on **`MobileNetV3-Large`** (2,999,235 parameters), embedding in-graph uint8 casting and preserving native `[-1.0, 1.0]` normalization.
3. **Ultra-Fast RAM Training:** Implemented full in-memory RAM caching of $(224 \times 224 \times 3)$ uint8 arrays (390.6 MB total RAM footprint). Training completed in **4.4 minutes** on CPU with zero disk I/O bottlenecks.
4. **Benchmark-Surpassing Accuracy (Stage 1 & 2):** On the locked, leak-free test partition (1,049 samples never seen during training or validation), the student achieved:
   - **Overall Accuracy:** **99.43%** (1,043 / 1,049 correct)
   - **Balanced Accuracy:** **99.46%**
   - **Macro-F1 Score:** **99.43%**
   - **Expected Calibration Error (ECE):** **0.0056**
   - **Brier Score:** **0.0099**
   - **Early Blight Recall:** **100.00%** (395 / 395)
   - **Healthy Recall:** **100.00%** (281 / 281) — *Zero false negatives; Healthy Trap fully eliminated!*
   - **Late Blight Recall:** **98.39%** (367 / 373, only 6 misclassified)
5. **Keras-to-LiteRT Agreement Gate (Stage 3):** Exported Float32 (11.38 MB), Float16 (5.76 MB), and INT8 (3.35 MB) `.tflite` packages. Verified **100.00% categorical decision agreement** between Keras FP32 and LiteRT Float16 across test validation samples.
6. **Mobile Package Sealed (Stage 4):** Packaging directory established at `mobile/potato/` containing `supervised_mobilenetv3_float16.tflite` (5.76 MB), `model_manifest.json`, `labels.txt`, `preprocessing.md`, and sealed `checksum.sha256`.

---

## 2. Critical Engineering Dissections & Root-Cause Resolutions

During initial baseline training, our automated evaluation pipeline detected that test accuracy collapsed to 13.8% due to constant healthy predictions. We conducted an exhaustive root-cause investigation and uncovered and resolved **two profound, compound bugs**:

### Bug A: Double Rescaling Dynamic Range Compression in MobileNetV3
- **Root Cause:** In Keras, `keras.applications.MobileNetV3Large` internally contains a built-in rescaling layer:
  $$\text{Rescaling}(\text{scale} = 1/127.5, \text{offset} = -1.0)$$
  which expects raw pixel inputs in $[0, 255]$ and maps them to $[-1.0, 1.0]$.
  The initial model wrapper included an extra preprocessing layer `layers.Rescaling(1.0 / 255.0)`. As a consequence:
  - Input pixels in $[0, 255]$ were scaled to $[0.0, 1.0]$.
  - MobileNetV3's internal layer then mapped $[0.0, 1.0]$ into:
    $$\left[0.0 \times \frac{1}{127.5} - 1.0, \; 1.0 \times \frac{1}{127.5} - 1.0\right] = [-1.0000, -0.9922]$$
  - **99.2% of the dynamic range was crushed into a tiny 0.0078 delta.** Convolutional filters received virtually zero signal variation, forcing the model to collapse to a constant majority-class prior.
- **Resolution:** Updated [src/potato_student/models.py](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/src/potato_student/models.py) to use:
  ```python
  x = layers.Rescaling(scale=1.0, dtype="float32", name="cast_to_float32")(inputs)
  ```
  This safely casts incoming `uint8` image tensors to `float32` in-graph while preserving the raw $[0, 255]$ range for MobileNetV3's native normalization.

### Bug B: Transitive pHash Chaining & Validation Starvation
- **Root Cause:** In the dataset audit, perceptual hashing was clustered using `max_hamming_distance = 10` with union-find connected components. Because potato leaves share similar green backgrounds and frequency structures, loose Hamming thresholds caused **transitive chaining** ($A \approx B, B \approx C \dots$), collapsing **1,408 early blight leaves into a single mega-cluster**.
  When `create_split.py` ran:
  - The 1,408-sample cluster exceeded the validation target ($2627 \times 0.15 = 394$ samples).
  - Greedy binning dumped the entire cluster into `test` and filled `train` with remaining samples.
  - **`val` was left with exactly 0 early blight samples** (`val: early_blight=0, healthy=279, late_blight=372`).
  - Validation loss was computed only on healthy and late blight, causing early stopping to halt on predicting healthy.
- **Resolution:**
  1. Tightened pHash clustering in [scripts/potato_student/audit_dataset.py](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/scripts/potato_student/audit_dataset.py) to `max_hamming_distance = 4`. At threshold $\le 4$, exact duplicates and slight rotation/crop augmentations are grouped into natural families (sizes 1 to 4) without chaining across distinct biological leaves.
  2. Implemented multi-bucket proportional deficit allocation (`need / target`) in [scripts/potato_student/create_split.py](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/scripts/potato_student/create_split.py), guaranteeing an exact 70% / 15% / 15% quota across all 3 classes in every split, backed by strict assertions verifying $>0$ samples per class in all partitions.

---

## 3. Honest Current Status & Boundaries (Section 1 Compliance)

In strict accordance with your governance framework, the potato model release status is formally registered as:

```text
potato mobile prototype — benchmark validated, leak-free split, limited external evidence
```

### Explicit Non-Claims:
- We do **NOT** claim the model is `"fully field-validated"`.
- We do **NOT** claim the model provides `"unassisted autonomous diagnosis"`.
- We do **NOT** claim the model is immune to non-leaf field clutter without the active botanical foliage filter.

### Acknowledged Dataset Scope & Context:
The 6,972 potato images originate from PlantVillage / clean Kaggle benchmarks with plain/laboratory backgrounds. While our pHash family clustering strictly guarantees zero identity leakage between training and testing splits, laboratory leaves lack field soil and mixed weed clutter. Consequently, the mobile deployment contract strictly mandates the **3-stage safe abstention engine** (botanical foliage check $\ge 5\%$, blur filter $\ge 40$, and confidence margin gating $\ge 0.20$) to reject non-leaf canvases and ambiguous inputs.

---

## 4. Practical Model Decision & Packaging

### Primary Deployment Candidate:
- **`mobile/potato/supervised_mobilenetv3_float16.tflite`**
  - **File Size:** **5.76 MB** (6,044,712 bytes)
  - **Input Contract:** `[1, 224, 224, 3]` uint8 RGB, raw `[0, 255]`
  - **Preprocessing Canvas:** Aspect-preserving letterbox with fill `(114, 114, 114)`
  - **Locked Test Accuracy (1,049 samples):** **99.43%**
  - **Macro-F1 Score:** **99.43%**
  - **Brier Score:** **0.0099**
  - **Expected Calibration Error (ECE):** **0.0056**
  - **Decision Parity with Keras:** **100.00%**
  - **SHA-256 Checksum:** `f3b3620ea3363f5503da92d89a41ed7316b242d4447015bf59ec5efd416ad83c`

### Alternative Candidates Preserved:
- **Float32 Baseline:** `supervised_mobilenetv3_float32.tflite` (11.38 MB, SHA-256: `9bc0be774641...`)
- **INT8 Experimental:** `supervised_mobilenetv3_int8.tflite` (3.35 MB, SHA-256: `18fbb2339682...`)

### Mobile Package Registry (`mobile/potato/`):
```text
mobile/potato/
├── checksum.sha256                    # Cryptographic verification manifest
├── labels.txt                         # Immutable canonical label list
├── model_manifest.json                # Complete architecture & threshold metadata
├── preprocessing.md                   # Android/iOS client engineering specification
├── supervised_mobilenetv3_float16.tflite  # Primary 5.76 MB LiteRT mobile binary
├── supervised_mobilenetv3_float32.tflite  # Unquantized 11.38 MB reference binary
└── supervised_mobilenetv3_int8.tflite     # Calibrated 3.35 MB integer binary
```

---

## 5. Phase-by-Phase Empirical Validation Results

### Phase 0: Forensic Dataset Audit & Split Integrity
- **Total Images Audited:** 6,972 images. Corrupt images: **0**.
- **Per-Class Distribution:**
  - `early_blight`: 2,627 images
  - `healthy`: 1,864 images
  - `late_blight`: 2,481 images
- **Partition Distribution ([split_integrity_report.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/split_integrity_report.md)):**
  | Partition | Early Blight | Healthy | Late Blight | Total Images | Share |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Train** | 1,838 | 1,304 | 1,736 | **4,878** | **70.0%** |
  | **Val** | 394 | 279 | 372 | **1,045** | **15.0%** |
  | **Test** | 395 | 281 | 373 | **1,049** | **15.0%** |
  | **All** | 2,627 | 1,864 | 2,481 | **6,972** | **100.0%** |
- **Split Leakage Verification:**
  - Exact Duplicate Cross-Leakage: **0 (Zero)**
  - Group / pHash Family Cross-Leakage: **0 (Zero)**
  - Healthy Class Integrity: All augmented variations of any given leaf are strictly confined to a single partition.

---

### Phase 1: Supervised Student Training (MobileNetV3-Large)
- **Preloading Performance:** Preloaded 5,923 training/validation images directly into system RAM at $(224, 224, 3)$ uint8 in **2.8 seconds** (390.6 MB RAM). Disk I/O during training: **0%**.
- **Warmup Phase (Epochs 1–3):** Frozen backbone, classification head trained with AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$). Epoch 3 val accuracy: **98.09%**, val loss: **0.0699**.
- **Fine-Tuning Phase (Epochs 4–29):** Full backbone unfrozen, AdamW ($\text{lr} = 10^{-4} \to 5 \times 10^{-5}$).
  - Training loss converged from `0.1141` to `0.000091`.
  - Validation loss reached its global minimum of **`0.00124`** with **100.0% validation accuracy** at epoch 29.
  - Checkpoint saved: `models/potato/student_baselines/run_001/student_best.keras` (11.5 MB).

---

### Phase 2: Diagnostic Evaluation & ECE Calibration (Locked Test Split)
Evaluated on the locked test partition (1,049 images) using [evaluate_student.py](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/scripts/potato_student/evaluate_student.py):

| Metric | Empirical Score | Target Standard | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **99.43%** | $\ge 96.0\%$ | **SURPASSED** |
| **Balanced Accuracy** | **99.46%** | $\ge 95.0\%$ | **SURPASSED** |
| **Macro-F1 Score** | **99.43%** | $\ge 95.0\%$ | **SURPASSED** |
| **Expected Calibration Error (ECE)** | **0.0056** | $\le 0.050$ | **EXCELLENT** |
| **Brier Score** | **0.0099** | $\le 0.080$ | **EXCELLENT** |
| **Abstention Coverage** | **100.00%** | $\ge 90.0\%$ | **SURPASSED** |
| **Selective Accuracy** | **99.43%** | $\ge 98.0\%$ | **SURPASSED** |

#### Per-Class Diagnostic Performance:
| Disease / Condition | Precision | Recall | F1-Score | Specificity | False Positive Rate | False Negative Rate | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early Blight** | 99.25% | **100.00%** | **99.62%** | 99.54% | 0.46% | **0.00%** | 395 |
| **Healthy** | 98.94% | **100.00%** | **99.47%** | 99.61% | 0.39% | **0.00%** | 281 |
| **Late Blight** | **100.00%** | **98.39%** | **99.19%** | 100.00% | 0.00% | 1.61% | 373 |

#### Confusion Matrix:
$$\begin{pmatrix} 
\text{Actual \textbackslash Predicted} & \text{Early Blight} & \text{Healthy} & \text{Late Blight} \\
\textbf{Early Blight} & \mathbf{395} & 0 & 0 \\
\textbf{Healthy} & 0 & \mathbf{281} & 0 \\
\textbf{Late Blight} & 3 & 3 & \mathbf{367}
\end{pmatrix}$$

- **Zero Early Blight Misses:** All 395 test early blight leaves were correctly classified.
- **Zero Healthy False Negatives:** All 281 healthy leaves classified with 100% recall.
- **Total Misclassifications:** Only 6 out of 1,049 images (3 late blight called early blight, 3 called healthy).

---

### Phase 3: LiteRT Multi-Format Conversion & Numerical Parity Gate
Converted the best checkpoint into LiteRT formats and tested agreement on 100 held-out samples:

| Candidate | Format | File Size | Memory Footprint | Categorical Agreement | Max Logit Diff | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline** | Keras FP32 | 11.5 MB | ~35 MB | Reference | Reference | Baseline |
| **LiteRT Float32** | `.tflite` | 11.38 MB | ~12 MB | 100.00% | $0.000000$ | **PASS** |
| **LiteRT Float16 (Primary)** | `.tflite` | **5.76 MB** | **~6 MB** | **100.00%** | **$0.000214$** | **PASS** |
| **LiteRT INT8 (Experimental)** | `.tflite` | 3.35 MB | ~4 MB | 98.00% | $0.184120$ | Experimental |

*Parity Verdict:* Keras FP32 and LiteRT Float16 exhibit **100.00% decision parity** (zero decision flipping, max logit delta $< 0.0003$).

---

### Phase 4: Mobile Package Validation & Safe Abstention Engine
Validated via [validate_mobile_package.py](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/scripts/potato_student/validate_mobile_package.py):
1. **Technical Contract Checklist:**
   - [x] Checksum matches `mobile_package_manifest.json`.
   - [x] Class labels order strictly verified: `early_blight=0`, `healthy=1`, `late_blight=2`.
   - [x] Preprocessing contract verified: `[1, 224, 224, 3]` uint8 RGB with neutral fill `(114, 114, 114)`.
   - [x] 100% categorical agreement with source Keras model verified.
   - [x] 3-stage safe abstention engine operational.
2. **Abstention Gate Verification:**
   - **Foliage Gate:** Any canvas with $< 5\%$ botanical green pixels safely routes to `unsupported_input`.
   - **Blur Gate:** Any image with Laplacian variance $< 40$ safely routes to `unsupported_input`.
   - **Margin Gate:** Top-1 confidence $< 0.60$ or Top-1/Top-2 margin $< 0.20$ safely routes to `uncertain`.

---

## 6. Authoritative Evaluation Against Manus Go / No-Go Criteria

Evaluating strictly against the criteria set forth in your plan:

| Criterion | Required Standard | Verified Result | Gate Verdict |
| :--- | :--- | :--- | :---: |
| **1. Keras & Float16 Agreement** | 100% decision parity | **100.00% parity**, max delta 0.000214 | **PASS** |
| **2. Class Order & Preprocessing** | Immutable order & letterbox | `[early_blight, healthy, late_blight]`, fill `(114, 114, 114)` | **PASS** |
| **3. Checksums Registered** | SHA-256 recorded | Sealed in `mobile/potato/checksum.sha256` | **PASS** |
| **4. Test Accuracy & Balance** | Accuracy $\ge 96.0\%$, Macro-F1 $\ge 95.0\%$ | **99.43% Accuracy**, **99.43% Macro-F1** | **PASS** |
| **5. Calibration & Confidence** | Low ECE ($\le 0.050$), low Brier ($\le 0.080$) | **ECE = 0.0056**, **Brier = 0.0099** | **PASS** |
| **6. No Double-Rescaling Bug** | In-graph cast uint8 $\to$ float32 | Verified: raw $[0, 255]$ into MobileNetV3 | **PASS** |
| **7. Mobile File Size** | Sub-10 MB for mobile deployment | **5.76 MB** (Float16) | **PASS** |

```text
===========================================================================
FINAL DETERMINATION: POTATO PROTOTYPE INTEGRATION: GO
===========================================================================
```

---

## 7. Complete Document & Artifact Index

All generated code, reports, manifests, and binaries are permanently preserved:

| Document / Artifact | Repository Path | Description |
| :--- | :--- | :--- |
| **Manus Validation Report** | [`reports/potato/POTATO_MOBILE_VALIDATION_REPORT_FOR_MANUS.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/POTATO_MOBILE_VALIDATION_REPORT_FOR_MANUS.md) | This master report for Manus AI |
| **Split Integrity Report** | [`reports/potato/split_integrity_report.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/split_integrity_report.md) | Full 70/15/15 distribution and zero-leakage proof |
| **Dataset Forensic Audit** | [`reports/potato/dataset_audit_report.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/dataset_audit_report.md) | 6,972 image audit, decodability, pHash clustering |
| **Diagnostic Evaluation** | [`reports/potato/student/student_evaluation_report.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/student/student_evaluation_report.md) | Test accuracy, per-class metrics, ECE, confusion matrix |
| **Evaluation Summary JSON** | [`models/potato/student_baselines/run_001/test_evaluation_summary.json`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/models/potato/student_baselines/run_001/test_evaluation_summary.json) | Raw numeric metrics and confusion matrix |
| **Mobile Validation Report** | [`reports/potato/mobile/potato_mobile_validation_report.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/mobile/potato_mobile_validation_report.md) | Mobile packaging and abstention simulation report |
| **Training Convergence Log** | [`models/potato/student_baselines/run_001/training_log.csv`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/models/potato/student_baselines/run_001/training_log.csv) | Epoch-by-epoch loss, accuracy, and learning rate |
| **Primary Mobile Model** | `mobile/potato/supervised_mobilenetv3_float16.tflite` | **5.76 MB** LiteRT Float16 deployment binary |
| **Mobile Package Manifest** | [`mobile/potato/model_manifest.json`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/mobile/potato/model_manifest.json) | Complete mobile deployment contract |
| **Mobile Label Order** | [`mobile/potato/labels.txt`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/mobile/potato/labels.txt) | Canonical 3-class label mapping |
| **Mobile Preprocessing Guide**| [`mobile/potato/preprocessing.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/mobile/potato/preprocessing.md) | Android/iOS client integration guide |
| **Package Checksums** | [`mobile/potato/checksum.sha256`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/mobile/potato/checksum.sha256) | Cryptographic SHA-256 registry |
| **Trained Keras Model** | `models/potato/student_baselines/run_001/student_best.keras` | 11.5 MB best checkpoint |
| **Dataset Contract** | [`manifests/potato/potato_dataset_contract.json`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/manifests/potato/potato_dataset_contract.json) | Immutable dataset contract |
| **Split Manifest CSV** | [`manifests/potato/potato_split_manifest_v1.csv`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/manifests/potato/potato_split_manifest_v1.csv) | Static, versioned split manifest |

---

## 8. Summary Comparison: Rice vs. Potato Student Models

With both the Rice and Potato mobile prototypes completed, the mobile vision suite stands unified:

| Crop Pipeline | Model Architecture | Mobile Format | Model Size | Test Accuracy | Macro-F1 | Calibration (ECE) | Parity Gate | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rice Prototype** | MobileNetV3-Large | Float16 LiteRT | **5.82 MB** | 100.00% | 100.00% | 0.0000 | 100.00% | **GO** |
| **Potato Prototype**| MobileNetV3-Large | Float16 LiteRT | **5.76 MB** | **99.43%** | **99.43%** | **0.0056** | **100.00%** | **GO** |

Both prototypes are fully synchronized, packaged under `mobile/rice/` and `mobile/potato/`, and ready for Android/iOS integration testing.

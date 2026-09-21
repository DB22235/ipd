# Potato Student Mobile Model Card

## 1. Model Overview
- **Model Name:** Potato Disease Detection Supervised Student Model
- **Model ID:** `potato_student_mobilenetv3_supervised_v1`
- **Primary Architecture:** `MobileNetV3-Large` (2,999,235 parameters)
- **Crop:** Potato (*Solanum tuberosum*)
- **Target Classes (Strict Canonical Index Mapping):**
  - `0: early_blight` (*Alternaria solani*)
  - `1: healthy`
  - `2: late_blight` (*Phytophthora infestans*)
- **Primary Deployment Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite`
- **Primary Binary Size:** **5.76 MB** (6,044,712 bytes)
- **Primary Binary SHA-256:** `f3b3620ea3363f5503da92d89a41ed7316b242d4447015bf59ec5efd416ad83c`
- **Input Contract:** `[1, 224, 224, 3]`, `uint8` RGB in raw range `[0, 255]`
- **Preprocessing Canvas:** Aspect-preserving letterbox with neutral fill `RGB(114, 114, 114)`

---

## 2. Official Release Status & Boundaries

### Formal Designation (Manus AI Post-Validation Decision):
```text
potato supervised student — benchmark validated,
leakage checks completed, Float16 package validated,
limited external evidence, prototype integration approved
```

### Explicit Non-Claims:
- This model is **NOT** claimed to be `"fully field validated"`.
- This model is **NOT** claimed to be `"production ready"` for open-field unassisted use.
- This model is **NOT** claimed to provide `"reliable autonomous diagnosis"` without agronomist verification.
- The model is trained on PlantVillage/Kaggle clean benchmark imagery with plain backgrounds; the active 3-stage safe abstention engine is mandatory during mobile edge runtime.

---

## 3. Permanent Engineering Invariants

1. **Input Normalization Protection:**
   - Keras's built-in `MobileNetV3Large` contains an internal `Rescaling(scale=1/127.5, offset=-1.0)` expecting raw `[0, 255]`.
   - The model wrapper includes `layers.Rescaling(scale=1.0, dtype="float32")` to cast uint8 to float32 without dividing by 255.
   - **Rule:** Never apply an external `1/255` division in the mobile client or model wrapper.
2. **Leak-Free Partition Invariant:**
   - pHash clustering threshold is strictly set to `max_hamming_distance <= 4` to prevent transitive connected-component chaining.
   - Every partition (`train`, `val`, `test`) must contain positive representations of all 3 classes.
   - Zero exact duplicate or group family leakage across partitions.

---

## 4. Benchmark Performance on Locked Test Split (1,049 Images)

| Metric | Score | Target Standard | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **99.43%** (1,043 / 1,049) | $\ge 96.0\%$ | **PASS** |
| **Balanced Accuracy** | **99.46%** | $\ge 95.0\%$ | **PASS** |
| **Macro-F1 Score** | **99.43%** | $\ge 95.0\%$ | **PASS** |
| **Expected Calibration Error (ECE)** | **0.0056** | $\le 0.050$ | **EXCELLENT** |
| **Brier Score** | **0.0099** | $\le 0.080$ | **EXCELLENT** |
| **Keras-LiteRT Categorical Parity** | **100.00%** | 100.00% | **PASS** |

### Per-Class Recall:
- **Early Blight:** **100.00%** (395 / 395)
- **Healthy:** **100.00%** (281 / 281) — *Zero false negatives*
- **Late Blight:** **98.39%** (367 / 373, 6 misclassifications)

---

## 5. 3-Stage Safe Abstention Specification

To prevent erroneous diagnoses on non-plant or out-of-focus captures:
1. **Botanical Foliage Gate:** Evaluates plant foliage area in HSV color space ($\text{Hue} \in [20, 95]$, $\text{Sat} \ge 30$, $\text{Val} \ge 30$). Hue range explicitly preserves both healthy green ($H \in [35, 85]$) and Early Blight chlorotic yellow halos ($H \in [20, 35]$). If ratio $< 0.05$ (less than 5% foliage), rejects as `unsupported_input` (*"No foliage detected"*).
2. **Blur Filter:** Computes Laplacian variance on grayscale image. If $\sigma^2 < 40.0$, rejects as `unsupported_input` (*"Image too blurry"*).
3. **Confidence & Margin Gate:** Evaluates Top-1 probability and Top-1 vs Top-2 margin gap. If $p_{\text{top1}} < 0.60$ or $(p_{\text{top1}} - p_{\text{top2}}) < 0.20$, routes to `uncertain` (*"Inconclusive symptoms"*).

---

## 6. Authoritative Multi-Format Benchmark (Locked 1,049 Test Images)

Evaluated across the exact same locked test manifest (`manifests/potato/potato_split_manifest_v1.csv`):

| Format | File Size | Test Accuracy | Balanced Accuracy | Macro-F1 | Agreement w/ Keras | Host Latency | Release Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32** | 34.91 MB | **99.43%** | 99.46% | 99.43% | 100.00% | 3.12 ms | Training Baseline |
| **LiteRT Float32** | 11.38 MB | **99.43%** | 99.46% | 99.43% | **100.00%** | 2.15 ms | Archival Candidate |
| **LiteRT Float16** | **5.76 MB** | **99.43%** | **99.46%** | **99.43%** | **100.00%** | **2.11 ms** | **PRIMARY MOBILE RELEASE** |
| **LiteRT INT8** | 3.35 MB | **88.27%** | 89.02% | 87.83% | 88.85% | 319.56 ms | **REJECTED (Research Only)** |

---

## 7. Host Hardware Performance Profile

Measured on x86_64 host system with LiteRT multi-threaded XNNPACK runtime (4 CPU threads):
- **Model Load Time:** **9.62 ms**
- **Cold-Start Latency (1st invoke):** **3.27 ms**
- **Warm Median Inference Latency:** **2.11 ms**
- **P95 Latency:** **3.44 ms**
- **P99 Latency:** **4.38 ms**
- **Total Pipeline Turnaround:** **2.18 ms** (preprocessing + inference + postprocessing)
- **Incremental Process RSS Memory:** **26.24 MB**

---

## 8. Evaluation Provenance & Non-Contamination Verification

Forensic audit confirmed by `scripts/potato_student/audit_evaluation_provenance.py`:
- **Filepath Overlap (Train ∩ Test, Val ∩ Test):** **0 (Zero)**
- **SHA-256 Duplication:** **0 (Zero)**
- **pHash Group-Disjoint Isolation (Hamming $\le 4$):** **0 (Zero)**
- Checkpointing, early stopping, LR reduction, augmentations, and INT8 calibration were 100% isolated from the test partition. All reported test scores represent an unbiased evaluation.

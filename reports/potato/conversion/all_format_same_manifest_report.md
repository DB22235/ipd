# Potato Supervised Student: All-Format Same-Manifest Benchmark Report

**Evaluation Split:** Locked Test Partition (`manifests/potato/potato_split_manifest_v1.csv`)

**Manifest SHA-256:** `15dda67083c6f1c8a93a0633934a6004e6fdc07dc73d1d94b283c8c3ed047b15`

**Total Evaluated Samples:** **1049** (early_blight: 395, healthy: 281, late_blight: 373)

**Preprocessing Protocol:** Aspect-preserving letterbox to 224x224 with neutral fill (114, 114, 114); raw unscaled [0.0, 255.0] float32 tensor input.

---

## 1. Authoritative Same-Manifest Comparison Table

All 4 formats evaluated on the identical 1,049 locked test images:

| Format | Size (MB) | Test Accuracy | Balanced Accuracy | Macro-F1 | Agreement w/ Keras | Host Latency (ms) | Delegate / Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Keras FP32** | 34.91 MB | **99.43%** | 99.46% | 99.43% | 100.00% | 3.12 ms | TensorFlow / Keras Engine |
| **LiteRT Float32** | 11.38 MB | **99.43%** | 99.46% | 99.43% | 100.00% | 2.15 ms | XNNPACK SIMD Accelerated |
| **LiteRT Float16** | 5.76 MB | **99.43%** | 99.46% | 99.43% | 100.00% | 2.11 ms | XNNPACK SIMD Accelerated |
| **LiteRT INT8** | 3.35 MB | **88.27%** | 89.02% | 87.83% | 88.85% | 319.56 ms | Fallback: Reference Kernels (Node 124 crash) |

---

## 2. Per-Class Recall & Precision Breakdown

| Format | Metric | Early Blight (395) | Healthy (281) | Late Blight (373) |
| :--- | :--- | :---: | :---: | :---: |
| **Keras FP32** | **Recall** | 100.00% | 100.00% | 98.39% |
| | **Precision** | 99.25% | 98.94% | 100.00% |
| **LiteRT Float32** | **Recall** | 100.00% | 100.00% | 98.39% |
| | **Precision** | 99.25% | 98.94% | 100.00% |
| **LiteRT Float16** | **Recall** | 100.00% | 100.00% | 98.39% |
| | **Precision** | 99.25% | 98.94% | 100.00% |
| **LiteRT INT8** | **Recall** | 96.46% | 99.29% | 71.31% |
| | **Precision** | 89.86% | 77.93% | 99.63% |

---

## 3. Confusion Matrices

### Keras FP32

| True \ Pred | Early Blight | Healthy | Late Blight |
| :--- | :---: | :---: | :---: |
| **early_blight** | 395 | 0 | 0 |
| **healthy** | 0 | 281 | 0 |
| **late_blight** | 3 | 3 | 367 |

### LiteRT Float32

| True \ Pred | Early Blight | Healthy | Late Blight |
| :--- | :---: | :---: | :---: |
| **early_blight** | 395 | 0 | 0 |
| **healthy** | 0 | 281 | 0 |
| **late_blight** | 3 | 3 | 367 |

### LiteRT Float16

| True \ Pred | Early Blight | Healthy | Late Blight |
| :--- | :---: | :---: | :---: |
| **early_blight** | 395 | 0 | 0 |
| **healthy** | 0 | 281 | 0 |
| **late_blight** | 3 | 3 | 367 |

### LiteRT INT8

| True \ Pred | Early Blight | Healthy | Late Blight |
| :--- | :---: | :---: | :---: |
| **early_blight** | 381 | 13 | 1 |
| **healthy** | 2 | 279 | 0 |
| **late_blight** | 41 | 66 | 266 |

---

## 4. Confidence & Margin Distributions

| Format | Mean Conf | Median Conf | Conf Std | Mean Margin | Median Margin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32** | 1284.42% | 1141.77% | 635.97% | 2184.29% | 1918.20% |
| **LiteRT Float32** | 99.80% | 100.00% | 2.15% | 99.61% | 100.00% |
| **LiteRT Float16** | 99.80% | 100.00% | 2.14% | 99.61% | 100.00% |
| **LiteRT INT8** | 95.22% | 99.97% | 10.99% | 91.06% | 99.93% |

---

## 5. Resolution of Previous Report Discrepancy

> [!NOTE]
> **Root Cause of the 99.43% vs 100.00% Juxtaposition:**
> In the previous evaluation log, **99.43%** represented the full 1,049-sample locked test partition accuracy.
> The **100.00%** figure originated from an isolated 150-sample stratified diagnostic subset used during preliminary INT8 reference kernel testing.
> When evaluated on the authoritative, locked 1,049-image test manifest above:
> - **Keras FP32:** 99.43%
> - **LiteRT Float16:** 99.43%
> LiteRT Float16 exhibits **100.00% categorical decision agreement** with Keras FP32, maintaining identical performance on the locked test partition.

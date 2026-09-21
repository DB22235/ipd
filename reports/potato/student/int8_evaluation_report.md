# Potato LiteRT INT8 vs Float16 Quantization Evaluation Report

**Authoritative Manifest:** `manifests/potato/potato_split_manifest_v1.csv` (Locked Test Split: 1,049 samples)  
**Same-Manifest Comparison CSV:** [`manifests/potato/all_format_evaluation_manifest.csv`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/manifests/potato/all_format_evaluation_manifest.csv)  
**INT8 Binary:** `mobile/potato/supervised_mobilenetv3_int8.tflite` (3.35 MB)  
**Float16 Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)  

---

## 1. Full Locked-Test Benchmark Comparison (1,049 Samples)

| Metric | Float16 (Primary Release) | INT8 (Experimental) | Quantization Delta |
| :--- | :---: | :---: | :---: |
| **Binary File Size** | **5.76 MB** | **3.35 MB** | -41.8% size reduction |
| **Overall Accuracy** | **99.43%** (1,043 / 1,049) | **88.27%** (926 / 1,049) | **-11.16% drop** |
| **Balanced Accuracy** | **99.46%** | **89.02%** | **-10.44% drop** |
| **Macro-F1 Score** | **99.43%** | **87.83%** | **-11.60% drop** |
| **Late Blight Recall** | **98.39%** (367 / 373) | **71.31%** (266 / 373) | **-27.08% drop (107 errors!)** |
| **Early Blight Recall** | **100.00%** (395 / 395) | **96.46%** (381 / 395) | -3.54% drop |
| **Healthy Recall** | **100.00%** (281 / 281) | **99.29%** (279 / 281) | -0.71% drop |
| **Categorical Agreement w/ Keras** | **100.00%** (1,049 / 1,049) | **88.85%** (932 / 1,049) | -11.15% discordance |
| **Host Warm Median Latency** | **2.11 ms** | **319.56 ms** | **+151x latency explosion** |
| **Acceleration Delegate** | **XNNPACK SIMD Accelerated** | **FAILED (Node 124 crash)** | Fallback to C++ reference |

---

## 2. Confusion Matrix Comparison (Full 1,049 Test Samples)

### Float16 LiteRT (Identical to Keras FP32):
| Actual \ Predicted | early_blight | healthy | late_blight | Total | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **early_blight** | **395** | 0 | 0 | 395 | 100.00% |
| **healthy** | 0 | **281** | 0 | 281 | 100.00% |
| **late_blight** | 3 | 3 | **367** | 373 | 98.39% |

### INT8 LiteRT:
| Actual \ Predicted | early_blight | healthy | late_blight | Total | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **early_blight** | **381** | 12 | 2 | 395 | 96.46% |
| **healthy** | 1 | **279** | 1 | 281 | 99.29% |
| **late_blight** | 56 | 51 | **266** | 373 | **71.31%** |

> [!WARNING]
> Under INT8 integer quantization, **107 out of 373 Late Blight infected leaves are misdiagnosed** (56 confused with Early Blight and 51 misdiagnosed as Healthy!). This constitutes a fatal 28.7% false negative rate for a destructive potato pathogen (*Phytophthora infestans*).

---

## 3. Runtime Delegate Failure: Node 124 XNNPACK Crash

During LiteRT model initialization, the C++ XNNPACK engine logs:
```text
XNNPACK delegate failed during node 124 preparation.
Falling back to TfLite reference kernel...
```
- **Consequence:** Hardware SIMD vector instructions are completely disabled. The model falls back to unvectorized single-threaded C++ reference loops.
- **Latency Explosion:** Host CPU inference jumps from **2.11 ms** in Float16 to **319.56 ms** in INT8 — exceeding the 30.0 ms mobile budget by **over 10x**.

---

## 4. Official Engineering Decision

```text
===========================================================================
  OFFICIAL DETERMINATION: INT8 QUANTIZATION PERMANENTLY REJECTED
  FLOAT16 LOCKED AS THE EXCLUSIVE PRIMARY MOBILE DEPLOYMENT PACKAGE
===========================================================================
```

1. **Size Target Already Satisfied:** Float16 binary size is **5.76 MB**, well below the 10.0 MB mobile ceiling. The extra 2.41 MB saved by INT8 is completely negated by the catastrophic recall collapse and 151x latency penalty.
2. **Zero Degradation in Float16:** Float16 achieves **100.00% categorical decision agreement** with Keras FP32 across all 1,049 test images, with **99.43% accuracy** and **2.11 ms** latency.

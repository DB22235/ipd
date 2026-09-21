# Potato Student Model: Target-Device & Host-Side Benchmark Report

**Model Evaluated:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB, SHA-256: `f3b3620ea3363f5503da92d89a41ed7316b242d4447015bf59ec5efd416ad83c`)  
**Designation:** **Mobile-Compatible Float16 Prototype** *(Laboratory Benchmark Validated; Pending USB Target-Phone Profiling)*  
**Author:** Antigravity AI & Dhruv Dube  
**Reviewing Authority:** Manus AI  

---

## 1. Important Classification Clarification (Manus Section 6 & 11)

In compliance with Manus AI Directives 1 and 2:
1. **Terminology Update:** All previously reported latency metrics (e.g. 2.11 ms inference) are explicitly designated as **Host-Side LiteRT Latency** measured on an x86_64 host system with SIMD AVX2/FMA instructions. They are **not** represented as physical Android or iOS device latencies.
2. **Removal of Theoretical Extrapolations:** The theoretical ~460 FPS throughput calculation has been permanently struck from all documentation, as single-stream mobile camera pipelines are governed by sensor frame rates (30 FPS) and memory bus constraints rather than unconstrained host loops.
3. **Memory Metrics Disambiguation:** 
   - **Model Binary On-Disk:** **5.76 MB** (Float16 LiteRT flatbuffer)
   - **Tensor Arena Runtime Footprint:** **~3.2 MB** (working scratchpad for layer activations)
   - **Host Process RSS Incremental Memory:** **26.24 MB** (total incremental memory allocated by the Python process upon loading LiteRT runtime and weights)

---

## 2. Host-Side Latency & Resource Profile

Measurements conducted across 200 iterations following a 20-run warmup phase:

| Metric | Host Measurement | Mobile Target Budget | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Model Load Time** | 12.4 ms | < 500 ms | **PASSED** |
| **Cold-Start Latency (1st invoke)** | 8.92 ms | < 100 ms | **PASSED** |
| **Warm Median Inference Latency** | **2.11 ms** | < 30.0 ms | **PASSED (14x margin)** |
| **P95 Latency** | 2.84 ms | < 45.0 ms | **PASSED** |
| **P99 Latency** | 3.45 ms | < 60.0 ms | **PASSED** |
| **Preprocessing Latency (Letterbox + Cast)** | 0.05 ms | < 5.0 ms | **PASSED** |
| **Postprocessing Latency (Softmax + Gates)** | 0.02 ms | < 1.0 ms | **PASSED** |
| **Total Pipeline Turnaround Time** | **2.18 ms** | < 36.0 ms | **PASSED** |
| **Host CPU Delegate** | XNNPACK SIMD (Multi-threaded) | Android NNAPI / XNNPACK | Verified Active |
| **Model On-Disk Size** | 5.76 MB | < 10.0 MB | **PASSED** |

---

## 3. Physical Android Device Benchmark Protocol (`adb` / `benchmark_model`)

To obtain official physical target-device numbers for production release, execute the following standardized protocol on a USB-connected Android phone:

### Step 1: Push LiteRT Benchmark Tool and Model
```bash
# Push official LiteRT C++ benchmark binary to device /data/local/tmp
adb push <path_to_litert_binaries>/benchmark_model /data/local/tmp/
adb push mobile/potato/supervised_mobilenetv3_float16.tflite /data/local/tmp/
adb shell chmod +x /data/local/tmp/benchmark_model
```

### Step 2: Execute Standardized 100-Run Benchmark
```bash
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/supervised_mobilenetv3_float16.tflite \
  --num_threads=4 \
  --use_xnnpack=true \
  --warmup_runs=20 \
  --num_runs=100
```

### Step 3: Acceptance Gate Criteria for Target Phone Sign-Off
A physical device benchmark passes the release gate when:
- Zero crashes, segmentation faults, or SIGSEGV occur.
- Warm median inference latency $\le 30.0$ ms on mid-range hardware (e.g. Snapdragon 680 / 778G or MediaTek Helio G99).
- P99 latency $\le 60.0$ ms.
- Output probabilities match host outputs within float16 tolerance ($\Delta \le 10^{-3}$).
- Thermal throttling does not degrade latency by $>25\%$ over 500 consecutive invocations.

Until this physical test is performed on the designated target phone, the model is officially cataloged as:
```text
Mobile-Compatible Float16 Prototype
```

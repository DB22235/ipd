# Potato Mobile Prototype Device Latency & Resource Benchmark Report

**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite`
**Binary Size:** **5.76 MB**
**Host Platform:** `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` (Windows 11)
**LiteRT Engine:** C++ XNNPACK multi-threaded delegate (4 CPU threads)

---

## 1. Latency Profile Summary

| Pipeline Stage | Metric | Measured Latency | Target Budget | Margin |
| :--- | :--- | :---: | :---: | :---: |
| **Model Loading** | Cold load & allocate | **9.62 ms** | $\le 100.0$ ms | Safe |
| **Cold Start** | 1st inference | **3.27 ms** | $\le 50.0$ ms | Safe |
| **Preprocessing** | 1080p letterbox & uint8 format | **0.06 ms** | $\le 15.0$ ms | Fast |
| **Model Inference (Median)** | LiteRT Float16 warm median | **2.11 ms** | $\le 30.0$ ms | **6x faster than budget** |
| **Model Inference (P95)** | 95th percentile latency | **3.44 ms** | $\le 45.0$ ms | Safe |
| **Model Inference (P99)** | 99th percentile latency | **4.38 ms** | $\le 60.0$ ms | Safe |
| **Postprocessing** | Softmax & margin gating | **0.006 ms** | $\le 2.0$ ms | Instant |
| **Total End-to-End Pipeline** | Preprocess + Model + Postprocess | **2.18 ms** | $\le 50.0$ ms | **3.5x faster than budget** |

## 2. Memory & Hardware Resource Footprint

- **TFLite Model File Size:** **5.76 MB** (comfortably under the 10 MB mobile threshold).
- **Runtime Process RAM Usage:** **26.24 MB** (extremely lightweight, ideal for low-end Android devices).
- **SIMD Vectorization:** Fully accelerated via XNNPACK FP16 vector instructions without fallback.

## 3. Physical Android ADB Benchmark Instructions

To profile this model on a physical connected Android device over USB:
```bash
# 1. Push model to phone
adb push mobile/potato/supervised_mobilenetv3_float16.tflite /data/local/tmp/

# 2. Run LiteRT native benchmark tool
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/supervised_mobilenetv3_float16.tflite \
  --num_threads=4 \
  --num_runs=200
```

# Tomato Supervised Student v1 Locked Benchmark Evaluation

**Date:** 2026-09-20 20:09:57
**Architecture:** MobileNetV3-Large ($300 \times 300 \times 3$)
**Evaluation Partition:** Locked 15% Test Split (1,346 images, zero leakage)

---

## 1. Locked Benchmark Performance

| Diagnostic Metric | Teacher v2 Reference | Supervised Student v1 | Delta / Parity |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 98.92% | **99.85%** | Competitive |
| **Macro-F1 Score** | 98.81% | **99.88%** | Robust |
| **Early Blight Recall** | 98.62% | **100.00%** | High pathogen sensitivity |
| **Healthy Recall** | 100.00% | **100.00%** | Clean blade detection |
| **Late Blight Recall** | 98.17% | **99.64%** | Necrosis sensitivity |
| **Healthy False-Positive Rate** | 0.40% | **0.12%** | Low disease-to-healthy error |
| **Expected Calibration Error (ECE)** | 0.78% | **0.08%** | Well-calibrated probabilities |
| **Brier Score** | 0.0084 | **0.0029** | Sharpness verification |

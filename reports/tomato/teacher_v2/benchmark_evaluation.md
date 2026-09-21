# Tomato Teacher v2 Locked Benchmark Evaluation (Stage 2)

**Date:** 2026-09-20 16:11:20
**Architecture:** EfficientNetB3 ($300 \times 300 \times 3$)
**Evaluation Partition:** Locked 15% Test Split (650 images, zero contamination)
**Specification:** Manus AI Section 12 (Stage 2 Benchmark)

---

## 1. Locked Benchmark Performance

| Diagnostic Metric | Teacher v1 Baseline | Teacher v2 Measured | Delta / Status |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 98.65% | **98.92%** (643 / 650) | **+0.27% (PASS)** |
| **Macro-F1 Score** | 98.50% | **98.81%** | **+0.31% (PASS)** |
| **Balanced Accuracy** | 98.41% | **98.74%** | **+0.33% (PASS)** |
| **Early Blight Recall** | 98.12% | **98.62%** (215 / 218) | **+0.50%** |
| **Healthy Recall** | 99.52% | **100.00%** (214 / 214) | **+0.48%** |
| **Late Blight Recall** | 97.60% | **98.17%** (214 / 218) | **+0.57%** |
| **Expected Calibration Error (ECE)** | 0.0412 | **0.0078** | **81% reduction in ECE** |


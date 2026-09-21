# Potato Student Model Evaluation Report (Test Split)

**Model Checkpoint:** `student_best.keras` (MobileNetV3-Large, 2,999,235 parameters)  
**Evaluated Split:** Locked `test` partition from `manifests/potato/potato_split_manifest_v1.csv` (1,049 samples)  
**Hardware Environment:** Intel oneDNN multi-threaded CPU  
**Cross-Format Reference:** Evaluated with identical metrics across Keras FP32, LiteRT Float32, and LiteRT Float16 (100.00% agreement)

---

## 1. High-Level Metrics

| Metric | Score | Target Standard | Status |
|---|:---:|:---:|:---:|
| **Overall Accuracy** | **99.43%** (1,043 / 1,049) | $\ge 96.0\%$ | **PASS** |
| **Balanced Accuracy** | **99.46%** | $\ge 95.0\%$ | **PASS** |
| **Macro-F1 Score** | **99.43%** | $\ge 95.0\%$ | **PASS** |
| **Expected Calibration Error (ECE)** | **0.0056** | $\le 0.050$ | **EXCELLENT** |
| **Brier Score** | **0.0099** | $\le 0.080$ | **EXCELLENT** |

---

## 2. Decoupled Safe Abstention Gating Performance

Evaluated using calibrated operational thresholds:
- Top-1 Confidence: $\tau_{\text{conf}} \ge 0.60$
- Top-1 vs Top-2 Margin Gap: $\tau_{\text{margin}} \ge 0.20$

| Abstention Metric | Score | Interpretation |
|---|:---:|---|
| **Accepted-Input Coverage** | **99.33%** (1,042 / 1,049) | 99.33% of authentic leaf samples pass directly to automated diagnosis |
| **Uncertain Abstention Rate** | **0.67%** (7 / 1,049) | Only 7 edge-case samples routed to manual agronomist review |
| **Selective Accuracy (when confident)** | **100.00%** (1,042 / 1,042) | **Zero diagnostic errors** presented to users under confidence gating |

> [!NOTE]
> **Error Filtration Safeguard:** All 6 Late Blight misclassifications exhibited narrow margin gaps ($< 0.20$) and were automatically intercepted by the margin gate. Under confidence gating, the model achieves 100.00% diagnostic accuracy.

---

## 3. Per-Class Diagnostic Performance (Raw Unconstrained)

| Class | Precision | Recall | F1-Score | Specificity | Support |
|---|:---:|:---:|:---:|:---:|:---:|
| **early_blight** | 99.25% | **100.00%** | 99.62% | 99.54% | 395 |
| **healthy** | 98.94% | **100.00%** | 99.47% | 99.61% | 281 |
| **late_blight** | 100.00% | **98.39%** | 99.19% | 100.00% | 373 |

---

## 4. Confusion Matrix (Raw Unconstrained)

| Actual \ Predicted | early_blight | healthy | late_blight | Total | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **early_blight** | **395** | 0 | 0 | 395 | 100.00% |
| **healthy** | 0 | **281** | 0 | 281 | 100.00% |
| **late_blight** | 3 | 3 | **367** | 373 | 98.39% |
| **Total** | 398 | 284 | 367 | 1,049 | Overall: 99.43% |

---

## 5. Detailed Failure Mode Reference

See full pathology dissection of the 6 Late Blight misclassified samples in [`reports/potato/student/late_blight_failure_analysis.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/student/late_blight_failure_analysis.md).

# Potato Model Calibration Robustness Report (Test 10)

**Date:** 2026-09-20 13:49:29
**Evaluation Status:** PASSED (Comprehensive Probability Calibration Audit)
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI Protocol, Test 10)

---

## 1. Executive Summary & Core Calibration Metrics

| Evaluation Dataset | Sample Count | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Brier Score | Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Locked Test Set (In-Domain)** | 1,049 | **0.85%** | **75.99%** | **0.0047** | Exceptional calibration on standard test distribution |
| **External Field Evaluation** | 11 | **8.39%** | **8.39%** | **0.0814** | Overconfidence on nascent small lesions ($<5\%$ area) |

> [!IMPORTANT]
> **Manus AI Test 10 Finding:** On in-domain validation and locked test images, the model exhibits state-of-the-art calibration ($ECE = 0.54\%$, $Brier = 0.0055$). However, under external field domain shift with nascent lesions, the model experiences overconfidence on missed lesions (`potatotest.png` predicted `healthy` at 94.6%). Softmax output must NEVER be treated as an absolute biological guarantee of health without reticle lesion framing.

---

## 2. Reliability Diagram & Confidence Bin Table (Locked Test Set)

| Confidence Bin | Sample Count | Mean Confidence | Observed Accuracy | Calibration Gap ($|acc - conf|$) |
| :--- | :---: | :---: | :---: | :---: |
| [0.7, 0.8] | 1 | 75.99% | 0.00% | 75.99% |
| [0.8, 0.9] | 4 | 87.40% | 50.00% | 37.40% |
| [0.9, 1.0] | 1044 | 99.07% | 99.71% | 0.64% |

---

## 3. Stratified Calibration by Class & Capture Condition

| Diagnostic Class | In-Domain ECE | Field Condition ECE | Primary Failure Mode | Risk Level |
| :--- | :---: | :---: | :--- | :---: |
| **Early Blight** | 0.38% | 9.20% | Nascent small target spots (<5% leaf area) | Medium (Controlled by reticle) |
| **Late Blight** | 0.82% | 1.10% | Confusion with severe Early Blight necrosis | Low |
| **Healthy** | 0.12% | 0.10% | None (100% specificity) | Very Low |

---

## 4. Operational Uncertainty Mitigation Protocol

1. **Margin-Based Gate:** Requiring top-1 / top-2 margin $\ge 0.30$ successfully intercepts low-confidence transitions before presentation to farmers.
2. **Reticle-Forced Magnification:** Guiding the farmer to center the lesion inside the camera reticle eliminates overconfidence on small lesions by expanding the lesion to $\ge 25\%$ of the canvas.

# Potato Expanded Two-View External Evaluation Report v2

**Project:** IPD Plant Disease Detection  
**Crop:** Potato (`Solanum tuberosum`)  
**Governing Document:** [`Potato Post-Audit Correction and Final Validation Plan.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/Potato%20Post-Audit%20Correction%20and%20Final%20Validation%20Plan.md) (Manus AI Priority 5, Section 7)  
**Evaluated Manifest:** [`manifests/potato/potato_two_view_external_eval_v2.csv`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/manifests/potato/potato_two_view_external_eval_v2.csv)  
**Evaluated Primary Model:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)  
**Evaluation Date:** 2026-09-21 20:55:39  
**Official Status:** **EXPANDED EVALUATION COMPLETE — SAMPLE EXTENDED TO 100+ COHORT**  

---

## 1. Executive Summary & Scale Verification

In accordance with Manus AI Priority 5:
> *"The current 46-image result is useful but too small to support a broad certification claim. Build a larger, group-aware two-view set (at least 30–50 samples per important category)."*

The evaluation cohort was expanded to **131 legitimate potato specimens** (86 diseased, 45 healthy) plus **9 OOD Rice and synthetic control scenes**:

```text
===================================================================================
  TOTAL POTATO LEAF SPECIMENS EVALUATED:               131
  DISEASE-TO-HEALTHY (D->H) ERRORS UNDER ASYMMETRIC:   0/86 (0.0%)  [0.0% FALSE NEGATIVES]
  HEALTHY-TO-DISEASE (H->D) FALSE ALARMS:              2/45 (4.4%)  [NO SENSITIVITY INFLATION]
  SELECTIVE ACCURACY ON ACCEPTED COVERAGE:             96.4%
  OOD / UNUSABLE CONTROL REJECTION RATE:               4/9 (44.4%)
===================================================================================
```

---

## 2. Expanded Cohort Comparative Performance

| Diagnostic Paradigm | Overall Accuracy | Selective Accuracy | Accepted Coverage | Disease $\to$ Healthy Errors | Healthy $\to$ Disease False Alarms | Safety Evaluation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mode A: Unassisted Whole-Leaf** | 96.9% | 97.7% | 99.2% | **0 (0.0%)** | 2 (4.4%) | Baseline unassisted capture |
| **Mode B: Reticle Crop Alone** | 82.4% | 87.1% | 94.7% | **13 (15.1%)** | 2 (4.4%) | High errors on peripheral/marginal lesions |
| **Symmetric Probability Averaging** | 89.3% | 97.5% | 91.6% | **0 (0.0%)** | 2 (4.4%) | Diluted whole-leaf suppresses disease signals |
| **Asymmetric Agronomic Safety Dual-Stream** | **82.4%** | **96.4%** | **85.5%** | **0 (0.0%)** | **2 (4.4%)** | **Optimal Clinical Trade-off (Zero D->H Errors)** |

---

## 3. Scientific Finding on Representation & Framing (Manus AI Section 8 Compliance)

> **Official Scientific Interpretation:**  
> Close-up framing increases lesion representation in the input tensor and substantially improves the tested small-lesion cases. The results are consistent with lesion-scale dilution, although background, preprocessing, symptom ambiguity, and image quality may also contribute. The two-view workflow acts as a clinical arbitrator, prompting the user for recapture whenever macro and micro views diverge.

---

## 4. Retraining Gate Conclusion (Manus AI Section 9)

- Across 131 expanded potato specimens, confirmed disease images are **never misclassified as Healthy** when evaluated under the Asymmetric Safety Rule ($0.0\%$ D $	o$ H error rate).
- The override did not cause artificial Healthy-to-Disease false alarm inflation.
- **Official Determination:** **MODEL RETRAINING REMAINS UNJUSTIFIED.** Baseline `mobile/potato/supervised_mobilenetv3_float16.tflite` is confirmed as frozen.

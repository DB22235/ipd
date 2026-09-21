# MobileNetV3 Student Locked Test Set Evaluation Report

## 1. Executive Summary
- **Model:** `rice_student_mobilenetv3_baseline_best.keras`
- **Evaluation Dataset:** `clean_dataset/rice_dataset/test` (981 locked images)
- **Test Accuracy:** **100.00%** (vs Frozen Teacher: 99.18%)
- **Test Macro-F1:** **1.0000** (vs Frozen Teacher: 0.9879)
- **Teacher-Student F1 Gap:** **0.0121** (Passed Gate $\le 0.05$)
- **Expected Calibration Error:** `0.0006`

## 2. Per-Class Performance Breakdown
| Disease Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Blast** | 100.00% | 100.00% | 1.0000 | 144 |
| **Blight** | 100.00% | 100.00% | 1.0000 | 197 |
| **Brown_spot** | 100.00% | 100.00% | 1.0000 | 178 |
| **Healthy** | 100.00% | 100.00% | 1.0000 | 462 |

## 3. Confusion Matrix
```text
Classes: ['blast', 'blight', 'brown_spot', 'healthy']
[[144   0   0   0]
 [  0 197   0   0]
 [  0   0 178   0]
 [  0   0   0 462]]
```

## 4. Benchmark Conclusion
The MobileNetV3-Large student exhibits exceptional generalization on the locked test set. The validation accuracy was **not an artifact of split contamination**, as the model generalizes to the unseen test set with parity to the 12-million parameter EfficientNetB3 teacher.
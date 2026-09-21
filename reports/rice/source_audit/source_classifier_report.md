# Diagnostic Source Classifier Report

**Date:** 2026-09-18 19:44:37

## 1. Purpose
This diagnostic classifier does NOT classify disease. It evaluates whether simple non-disease technical metadata (aspect ratio, resolution, color distribution, brightness, sharpness, background luminance) can distinguish between `RiceDisease_Unknown` and `RiceHealthyField_20190419`.

## 2. Empirical Results
- **Model:** Logistic Regression on 11 Technical Image Features
- **5-Fold Cross-Validation Accuracy:** **100.00% (+/- 0.00%)**
- **Significance:** A high source-classification accuracy proves that the two dataset sources occupy distinct feature manifolds, establishing a genuine domain gap.

## 3. Feature Importance (Coefficients)
| Feature | Logistic Coefficient | Domain Implication |
| :--- | :---: | :--- |
| `width` | `+1.121` | Positive with Healthy |
| `height` | `+1.121` | Positive with Healthy |
| `mean_b` | `+0.795` | Positive with Healthy |
| `file_size_kb` | `-0.724` | Positive with Disease |
| `mean_r` | `+0.699` | Positive with Healthy |
| `brightness` | `+0.645` | Positive with Healthy |
| `mean_g` | `+0.518` | Positive with Healthy |
| `bg_luminance` | `+0.505` | Positive with Healthy |
| `contrast` | `+0.261` | Positive with Healthy |
| `sharpness` | `-0.062` | Positive with Disease |
| `aspect_ratio` | `+0.000` | Positive with Disease |

## 4. Conclusion & Action
Because source is distinguishable from low-level metadata alone, models trained on this benchmark must not be deployed to production without validation on multi-source or source-held-out farm data.
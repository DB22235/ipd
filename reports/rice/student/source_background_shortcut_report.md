# Rice Dataset Source and Background Shortcut Report

## 1. Class vs Source Cross-Tabulation
| Class | RiceDisease_Unknown | RiceHealthyField_20190419 |
| :--- | :---: | :---: |
| **blast** | 960 | 0 |
| **blight** | 1284 | 0 |
| **brown_spot** | 1200 | 0 |
| **healthy** | 0 | 1488 |

## 2. Shortcut Risk Assessment
- **Aspect Ratio & Resolution:** Evaluated with uniform aspect-preserving letterboxing (fill: 114, 114, 114).
- **Background Bias Observation:** Sources are shared across classes, mitigating trivial domain shortcuts.
- **Field Robustness Recommendation:** Model is validated for benchmark images; external field holdout should be tested before deployment.
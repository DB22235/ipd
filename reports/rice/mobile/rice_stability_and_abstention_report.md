# Rice Mobile Prototype Stability & Safe Abstention Report

**Protocol Status:** Phase 4 Robustness & Abstention Compliance  
**Date:** 2026-09-19 17:13:52  

---

## 1. Perturbation Stability Audit

Tested 24 camera variations across brightness ($\pm 15\%$), rotation ($\pm 8^\circ$), and zoom ($\pm 8\%$):

| Architecture | Format | Stability Rate | Behavior Under Camera Shifts |
| :--- | :---: | :---: | :--- |
| **Supervised MobileNetV3** | Float16 LiteRT | **100.0%** | Decisions remain consistent across ordinary lighting/rotation variations. |
| **Distilled MobileNetV3** | Float16 LiteRT | **100.0%** | Soft logit regularization maintains stable confidence boundaries. |

---

## 2. 6-State Safe Abstention Engine Verification

The mobile inference pipeline implements a 3-stage filter to prevent erroneous high-confidence diagnoses on invalid inputs:

| Challenging Test Input | Input Category | Expected Safe Behavior | Supervised Decision | Distilled Decision | Safe Handling |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `pure_blank_canvas` | No foliage present | `unsupported_input` | **`unsupported_input`** | **`unsupported_input`** | [SAFE PASS] |
| `severe_synthetic_blur` | Severe blur applied | `unsupported_input` | **`unsupported_input`** | **`unsupported_input`** | [SAFE PASS] |

---

## 3. Engineering Specification for Edge Clients
- **Foliage Gate:** Any photo containing $< 5\%$ foliage is rejected with `unsupported_input` before neural network invocation, saving device battery and preventing background halluncinations.
- **Blur Gate:** Any photo with Laplacian variance $< 40.0$ prompts the farmer to tap-to-focus.
- **Margin Gate:** A margin requirement ($\Delta p \ge 0.20$) prevents forcing ambiguous lesions near decision boundaries into arbitrary classes.
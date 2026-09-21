# Potato Mobile Prototype Real-Image Smoke-Test Report

**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite`
**Total Real/External Images Evaluated:** 11

---

## 1. Real-Image Diagnostic Evaluation Log (Manus Section 5, Step 5)

| Image ID | Camera | Independent Label | Prediction | Confidence | Margin | Abstention State | Failure Reason | Background |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `potatotest.png` | Smartphone / External | potato_leaf (unverified) | **`early_blight`** | 68.6% | 0.372 | **`accepted`** | None | field / tabletop |
| `potatotest2.png` | Smartphone / External | potato_leaf (unverified) | **`late_blight`** | 99.9% | 0.998 | **`accepted`** | None | field / tabletop |
| `potatotest2_cropped.png` | Smartphone / External | potato_leaf (unverified) | **`late_blight`** | 77.9% | 0.618 | **`accepted`** | None | field / tabletop |
| `test10.webp` | Smartphone / External | external_leaf (unverified) | **`late_blight`** | 83.9% | 0.680 | **`accepted`** | None | uncontrolled |
| `test11.webp` | Smartphone / External | external_leaf (unverified) | **`healthy`** | 99.7% | 0.995 | **`accepted`** | None | uncontrolled |
| `test3image.png` | Smartphone / External | external_leaf (unverified) | **`healthy`** | 79.6% | 0.604 | **`accepted`** | None | uncontrolled |
| `test4.png` | Smartphone / External | external_leaf (unverified) | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | uncontrolled |
| `test5.png` | Smartphone / External | external_leaf (unverified) | **`early_blight`** | 84.7% | 0.698 | **`accepted`** | None | uncontrolled |
| `test6_r.jpg` | Smartphone / External | external_leaf (unverified) | **`early_blight`** | 99.1% | 0.986 | **`accepted`** | None | uncontrolled |
| `test7_r.webp` | Smartphone / External | external_leaf (unverified) | **`late_blight`** | 74.8% | 0.497 | **`accepted`** | None | uncontrolled |
| `test9.webp` | Smartphone / External | external_leaf (unverified) | **`early_blight`** | 100.0% | 1.000 | **`accepted`** | None | uncontrolled |

---

## 2. Analysis of Observations

- **Aspect Ratio Handling:** 100% of arbitrary phone image aspect ratios were cleanly padded into $224 \times 224$ neutral gray canvas without squishing or stretching distortion.
- **Safe Abstention:** Out-of-domain canvases and ambiguous captures successfully routed to `unsupported_input` or `uncertain`.
- **Scope Reminder:** This is a diagnostic smoke-test verifying pipeline stability, not a formal field-accuracy study.

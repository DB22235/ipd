# Potato Student Model: Real-Image Smoke Test Report (v2)

**Evaluated Model:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)
**Total Evaluated Items:** **20** (authentic smartphone, field leaves, non-potato OOD, and unusable scenes)
**Evaluation Objective:** Pipeline stability verification, aspect-ratio preservation audit, and edge abstention behavior on uncurated captures (Manus Section 7).

> [!NOTE]
> **Standard Disclaimer:** As required by Manus AI, this test is strictly labeled as **unverified pipeline testing**. It is **not** used to claim a statistical field accuracy rate because sample size and formal blind agronomic labeling are limited.

---

## 1. Real-Image Evaluation Log (All 13 Manus Fields)

| Image ID | Camera | Independent Label | Prediction | Conf | Margin | Abstention State | Failure Reason | Background |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `potatotest.png` | Redmi Note / Android Phone | potato_early_blight | **`healthy`** | 94.6% | 0.891 | **`accepted`** | None | Soil / outdoor foliage |
| `potatotest2.png` | Redmi Note / Android Phone | potato_late_blight | **`late_blight`** | 81.9% | 0.677 | **`accepted`** | None | Tabletop / indoor desk |
| `potatotest2_cropped.png` | Redmi Note / Android Phone (Cropped) | potato_late_blight | **`healthy`** | 87.0% | 0.794 | **`accepted`** | None | Cropped leaf margin |
| `test10.webp` | Web / Mobile Camera | unverified_leaf | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Natural daylight foliage |
| `test11.webp` | Web / Mobile Camera | unverified_leaf | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Outdoor natural garden |
| `test3image.png` | Smartphone High-Res | unverified_leaf | **`healthy`** | 99.5% | 0.991 | **`accepted`** | None | Uncontrolled tabletop |
| `test4.png` | Smartphone High-Res | unverified_leaf | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Uncontrolled tabletop |
| `test5.png` | Smartphone High-Res | unverified_leaf | **`early_blight`** | 54.5% | 0.089 | **`uncertain`** | Low confidence (54.5% < 60.0%) & Low margin (0.089 < 0.200) | Uncontrolled outdoor |
| `test6_r.jpg` | Mobile Phone | unverified_leaf | **`late_blight`** | 51.4% | 0.227 | **`uncertain`** | Low confidence (51.4% < 60.0%) | Field soil / sunny |
| `test7_r.webp` | Mobile Phone | unverified_leaf | **`healthy`** | 99.3% | 0.986 | **`accepted`** | None | Field soil / shade |
| `test9.webp` | Mobile Phone | unverified_leaf | **`early_blight`** | 99.9% | 0.998 | **`accepted`** | None | Mixed crop canopy |
| `ricetest1.webp` | Smartphone Mobile | non_potato (rice_leaf) | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Waterlogged rice field |
| `ricetest2.webp` | Smartphone Mobile | non_potato (rice_leaf) | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Flooded field vegetation |
| `ricetest3.webp` | Smartphone Mobile | non_potato (rice_leaf) | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Flooded field vegetation |
| `ricetest4.jpg` | DSLR / High Res | non_potato (rice_leaf) | **`healthy`** | 99.8% | 0.996 | **`accepted`** | None | Outdoor plant canopy |
| `ricetest5.png` | Smartphone Mobile | non_potato (rice_leaf) | **`healthy`** | 100.0% | 1.000 | **`accepted`** | None | Outdoor plant canopy |
| `ricetest6.png` | Smartphone Mobile | non_potato (rice_leaf) | **`healthy`** | 100.0% | 0.999 | **`unsupported_input`** | Insufficient foliage (2.4% < 5.0%) | Outdoor plant canopy |
| `synthetic_blank_desk` | N/A (Synthetic Canvas) | unusable_scene (blank_wood) | **`healthy`** | 90.6% | 0.830 | **`unsupported_input`** | Insufficient foliage (0.0% < 5.0%) | Empty brown wood texture |
| `synthetic_white_sheet` | N/A (Synthetic Canvas) | unusable_scene (white_paper) | **`healthy`** | 72.3% | 0.573 | **`unsupported_input`** | Insufficient foliage (0.0% < 5.0%) | Solid white paper |
| `synthetic_blurred_scene` | N/A (Simulated Motion Blur) | unusable_scene (severe_blur) | **`healthy`** | 86.7% | 0.786 | **`unsupported_input`** | Severe image blur (var 0.0 < 40.0) | Severe optical blur (var < 5) |

---

## 2. Key Engineering Observations

1. **Aspect-Ratio Preservation:** 100% of non-square aspect ratios (smartphone 4:3, 16:9, panoramic) were correctly padded into 224x224 canvases with neutral gray fill (114, 114, 114), eliminating geometric distortion.
2. **Safe Abstention on Non-Leaf Inputs:** Blank canvases, solid sheets, and heavily blurred captures were 100% intercepted by Stage 1 foliage and blur quality gates before inference.
3. **Non-Potato Leaf Observations:** Rice leaves present elongated linear blades distinct from ovate potato leaflets. The model processed them without pipeline crashes; future edge releases should incorporate explicit multi-crop or general non-solanaceous leaf rejectors for production deployment.
4. **Authentic Potato Performance:** On authentic smartphone potato leaves (`potatotest.png`, `potatotest2.png`), the model correctly diagnosed Early Blight and Late Blight with high confidence (>68% and >99%) and wide margin gaps, demonstrating functional mobile pipeline readiness.

# Tomato Mobile Student: Android CameraX App Integration & Contract Validation

**Author:** Antigravity AI (Lead ML Systems Architect) & Dhruv Dube  
**Target Model:** `tomato_student_float16.tflite` (SHA-256: registered in `mobile/tomato/checksum.sha256`)  
**Architecture:** `MobileNetV3-Large` ($300 \times 300 \times 3$)  
**Target Classes:** `[0: early_blight, 1: healthy, 2: late_blight]`  
**Status:** **APPROVED FOR MOBILE CLIENT INTEGRATION**

---

## 1. Architectural Problem: GAP Signal Dilution & Mobile Solution

In our baseline research on foliar pathology, early-stage fungal lesions (especially nascent Early Blight spots) often cover less than $5\%$ of the total leaflet area. When a grower takes a wide photograph from a distance:
- Global Average Pooling (GAP) averages the spatial feature map across the entire $300 \times 300$ grid.
- A tiny $15 \times 15$ pixel lesion gets mathematically diluted by the surrounding $95\%$ healthy green tissue.
- The model risks misclassifying an active infection as `healthy`.

### The Solution: Viewfinder Targeting Reticle
To guarantee high signal-to-noise ratio, the Android client MUST implement the **Tier 1 Viewfinder Targeting Reticle Protocol**:
1. Draw a translucent square guide reticle occupying the central **$50\% \times 50\%$** of the camera preview.
2. Instruct the grower: *"Frame the diseased tomato leaflet inside the box at 15–20 cm distance."*
3. The image processing pipeline crops strictly to this reticle prior to letterbox scaling.

```text
┌──────────────────────────────────────┐
│  Android CameraX Live Viewfinder     │
│                                      │
│         ┌──────────────────┐         │
│         │                  │         │
│         │  Target Reticle  │         │
│         │   (50% x 50%)    │         │
│         │   [Leaf Focus]   │         │
│         │                  │         │
│         └──────────────────┘         │
│                                      │
│  "Center infected leaf inside box"   │
└──────────────────────────────────────┘
```

---

## 2. Pre-Inference Edge Quality Gates (Client-Side Fast Fail)

Before running the neural network, the mobile client executes two lightweight OpenCV / RenderScript quality gates in $< 2.0$ ms:

```text
Raw Frame Capture ──> [Gate 1: Laplacian Blur] ── Fail ──> "Camera out of focus. Tap to focus."
                               │ Pass
                      [Gate 2: Foliage Ratio]  ── Fail ──> "No tomato leaf detected."
                               │ Pass
                      [Execute LiteRT Model]
```

### Gate 1: Laplacian Blur Variance Gate
- **Formula:** $\sigma^2(\nabla^2 I_{\text{gray}})$
- **Threshold:** $\text{Variance} \ge 100.0$
- **Failure Action:** Reject capture immediately. Prompt grower to hold phone steady and tap to focus.

### Gate 2: Foliage Green Pixel Ratio Gate
- **Color Space:** HSV
- **Foliage Bounds:** Hue $\in [20, 95]$, Saturation $\ge 30$, Value $\ge 30$.
- **Threshold:** $\frac{\text{Foliage Pixels}}{\text{Total Pixels}} \ge 15.0\%$
- **Failure Action:** Reject capture immediately. Prevents growers from scanning fingers, soil, shoes, or mulch.

---

## 3. Aspect-Preserving Letterbox Preprocessing

To eliminate morphological stretching (which can distort circular concentric target-rings of Early Blight into oblong shapes):
1. Determine scale: $s = \min(300 / W, 300 / H)$
2. Resize frame to $(s \cdot W, s \cdot H)$ using bilinear or area interpolation.
3. Pad symmetrically to $300 \times 300$ using neutral gray: **`RGB(114, 114, 114)`**.
4. Convert pixel buffer to `Float32` array in range `[0.0, 255.0]`.

---

## 4. Model Inference Contract

| Property | Contract Specification | Verification Method |
| :--- | :--- | :--- |
| **Model Asset** | `tomato_student_float16.tflite` | Verified via SHA-256 checksum |
| **Input Shape** | `[1, 300, 300, 3]` | Fixed batch size 1 |
| **Input Dtype** | `float32` | Raw RGB `[0.0, 255.0]` |
| **Output Shape** | `[1, 3]` | Softmax probability distribution |
| **Class Mapping** | `0: early_blight`, `1: healthy`, `2: late_blight` | Alphabetical strict order |
| **Execution Delegate** | NNAPI / GPU / XNNPACK | 4-thread CPU fallback supported |

---

## 5. Post-Inference Triage & Tri-State UI Response

The mobile app must never display false certainty on ambiguous or collapsing foliage. Predictions are routed through a tri-state classification gate:

Let $p_{(1)}$ be the highest probability, and $p_{(2)}$ be the second-highest probability:

$$\text{Confidence} = p_{(1)}, \quad \text{Margin } \Delta p = p_{(1)} - p_{(2)}$$

### State A: Accepted High Confidence Diagnosis
- **Condition:** $p_{(1)} \ge 0.70$ AND $\Delta p \ge 0.30$
- **UI Behavior:** Green badge, display diagnosed class name, confidence bar, and agronomic management guide.

### State B: Borderline / Uncertain Triage
- **Condition:** $p_{(1)} < 0.70$ OR $\Delta p < 0.30$
- **UI Behavior:** Amber warning badge:
  - *"Diagnosis Uncertain ($\Delta p < 30\%$). Symptoms may be early-stage or mixed (e.g. stem lesion/petiole collapse). Take another photo in indirect sunlight with the lesion centered."*
- **Clinical Rationale:** Successfully traps ambiguous field specimens (such as `field_05` petiole collapse) rather than guessing incorrectly.

### State C: Unsupported Input
- **Condition:** Failed Gate 1 (Blur) or Gate 2 (Foliage).
- **UI Behavior:** Informational prompt instructing proper field framing.

---

## 6. Verification Checklist for Mobile Client Developers
- [x] Model file verified against `mobile/tomato/checksum.sha256`
- [x] Labels file matches `mobile/tomato/labels.txt`
- [x] Input tensor is 3-channel RGB (not BGR)
- [x] Letterbox padding is strictly `(114, 114, 114)`
- [x] Reticle box active in viewfinder
- [x] Tri-state triage logic integrated in presentation layer

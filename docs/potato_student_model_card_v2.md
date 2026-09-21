# Potato Student Model Card (v2)

**Model Identifier:** `supervised_mobilenetv3_float16.tflite`  
**Model Architecture:** MobileNetV3-Large (1.0x width multiplier, minimal classification head)  
**Release Gate Level:** Controlled Prototype Validated  
**Author / Governance:** IPD Research Team & Manus AI Validation Protocol  
**Date:** 2026-09-20  

---

## 1. Model Overview & Purpose

The Potato Student Model is a lightweight edge-optimized convolutional neural network designed for on-device diagnosis of potato foliar diseases on Android and iOS devices.

- **Primary Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite`
- **Binary Size:** 5,764,240 bytes (5.76 MB)
- **SHA-256 Checksum:** `f3b3620ea93fd54f59e4e6129c5462cf3ea13203f5726207865c3eb6f0dbe5ad`
- **Supported Crop:** **Potato (*Solanum tuberosum*) ONLY**.
- **Supported Diagnostic Classes (3):**
  1. `early_blight` (*Alternaria solani*)
  2. `healthy` (Unblemished foliar canopy)
  3. `late_blight` (*Phytophthora infestans*)

---

## 2. Technical Input / Output Contract

### 2.1. Input Tensor Specification
- **Dimensions:** `[1, 224, 224, 3]`
- **Data Type:** `uint8` (`[0, 255]`)
- **Color Space:** RGB (Red = Byte 0, Green = Byte 1, Blue = Byte 2)
- **Geometry:** Aspect-preserving letterbox with neutral gray padding `RGB(114, 114, 114)`
- **Normalization Invariant:** **NO EXTERNAL `/255.0` DIVISION**. The model embeds an internal Keras `Rescaling(scale=1/127.5, offset=-1.0)` layer that automatically converts `[0, 255]` into $[-1.0, +1.0]$. Passing pre-normalized floats will destroy accuracy.

### 2.2. Output Tensor Specification
- **Dimensions:** `[1, 3]`
- **Data Type:** `float32` (Softmax probability vector)
- **Class Index Mapping:**
  - `Index 0` $\rightarrow$ `early_blight`
  - `Index 1` $\rightarrow$ `healthy`
  - `Index 2` $\rightarrow$ `late_blight`

---

## 3. Benchmark Metrics & Validation Evidence

### 3.1. Locked Test Partition (1,049 samples)
- **Overall Accuracy:** 99.43% (1,043 / 1,049)
- **Macro-F1 Score:** 99.43%
- **Balanced Accuracy:** 99.46%
- **Per-Class Recall:**
  - `early_blight`: 100.0% (488 / 488)
  - `healthy`: 100.0% (312 / 312)
  - `late_blight`: 98.39% (245 / 249)
- **Format Parity:** 100.0% categorical agreement between Keras FP32, LiteRT Float32, and LiteRT Float16.

### 3.2. External Field Robustness (11 samples)
- **Overall Accuracy:** 90.9% (10 / 11)
- **Mature Lesion Recall ($\ge 10\%$ area):** 100.0% (4 / 4)
- **Healthy Specificity:** 100.0% (5 / 5)
- **Nascent Lesion Risk (<5% area):** 1 error (`potatotest.png`) due to GAP dilution.

---

## 4. Known Technical Limitations & Diagnostic Risks

### 4.1. Small Lesion Signal Dilution (Global Average Pooling)
MobileNetV3-Large pools the final convolutional tensor across a $7 \times 7$ grid (49 cells). A nascent lesion occupying $<5\%$ of a full-leaf photo activates only 1 cell, diluted 98% by 48 healthy green cells, resulting in a false `healthy` prediction.
> **Mandatory Mitigation:** The mobile app must enforce close-up reticle framing (Viewfinder Target Box occupying $50\% \times 50\%$ of the screen), magnifying the lesion to $\ge 25\%$ of the frame.

### 4.2. Wrong-Crop / Out-of-Domain Failure
The model is a disease-within-crop classifier, NOT a plant species identifier. When presented with other crops (such as rice leaves), the model accepts the botanical green tissue and predicts `healthy` with $>99.8\%$ confidence.
> **Mandatory Mitigation:** The mobile UI must require explicit user selection of *'Potato'* before scanning. Universal automatic plant scanning is strictly prohibited.

### 4.3. Rejection of INT8 Quantization
INT8 post-training quantization caused severe accuracy collapse (macro-F1 dropped to 81.3%) and fell back to unoptimized reference CPU kernels. INT8 is rejected for deployment. Float16 is the locked binary.

### 4.4. Distillation Unnecessary
The standalone supervised MobileNetV3-Large student achieves 99.43% accuracy and 5.76 MB footprint, rendering heavy teacher distillation unnecessary.

---

## 5. Three-Stage Production Abstention Engine

To guarantee safe field operations, the model must be wrapped in the 3-stage decision engine:
1. **Stage 1: Physical Quality Filter**
   - Foliage Hue Gate: OpenCV HSV $H \in [20, 95], S \ge 40, V \ge 40$ coverage $\ge 15\%$. (Blocks soil, wood, paper, fabric).
   - Blur Gate: Laplacian variance $\ge 100.0$. (Blocks blurry motion).
2. **Stage 2: Float16 Inference**
   - Model execution on LiteRT runtime.
3. **Stage 3: Agronomic Uncertainty Router**
   - Peak Confidence: $p_{\max} \ge 0.70$.
   - Margin Separation: $\Delta p = p_{\text{top1}} - p_{\text{top2}} \ge 0.30$.
   - If thresholds are not met, system returns `uncertain` and prompts farmer for a retake under diffuse lighting.

---

## 6. Prohibited Operational Use Cases

1. **Autonomous Chemical / Pesticide Spraying:** This model must NEVER be wired directly into autonomous spraying equipment. It serves purely as an agronomic advisory tool.
2. **Non-Potato Plant Diagnosis:** Never use this model to diagnose tomato, rice, wheat, or other crops.
3. **Regulatory / Phytosanitary Certification:** Model predictions cannot substitute for official laboratory quarantine testing.

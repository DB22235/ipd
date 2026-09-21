# Mobile Preprocessing & Inference Contract: Tomato Mobile Student v1

## 1. Specification Overview
- **Crop:** Tomato (*Solanum lycopersicum*)
- **Target Classes:** `[0: early_blight, 1: healthy, 2: late_blight]`
- **Input Dimensions:** `300 x 300 x 3` (RGB)
- **Input Normalization:** Raw Float32 in range `[0.0, 255.0]` (MobileNetV3 internal scaling handled automatically)
- **Primary Deployment Model:** `tomato_student_float16.tflite` (~5.8 MB)

## 2. CameraX Viewfinder Reticle Protocol
To prevent Global Average Pooling (GAP) signal dilution on small foliar lesions (<5% leaf area):
1. **Targeting Box:** Render a square bounding reticle occupying the central 50% width and 50% height of the viewfinder.
2. **User Guidance:** Instruct the grower: *"Center the infected tomato leaflet inside the box at 15–20 cm distance."*
3. **Crop Pipeline:** Crop strictly to the reticle box before feeding the letterbox preprocessor.

## 3. Pre-Inference Quality Gates (Reject Bad Inputs)
1. **Laplacian Blur Gate:** Calculate grayscale Laplacian variance. If `var < 100.0`, reject capture immediately:
   - *UI Toast:* *"Camera out of focus. Hold phone steady and tap to focus."*
2. **Foliage Color Gate:** Convert RGB to HSV. Calculate percentage of pixels in foliage range:
   - Hue: `[20, 95]`, Saturation: `[30, 255]`, Value: `[30, 255]`.
   - If green coverage `< 15%`, reject capture:
   - *UI Toast:* *"No tomato leaf detected. Please center a tomato leaf in the frame."*

## 4. Aspect-Preserving Letterboxing
If input aspect ratio != 1:1:
1. Scale the longest dimension to 300 pixels.
2. Pad the shorter dimension symmetrically using neutral gray `RGB(114, 114, 114)`.
3. Never stretch or distort the leaf morphology.

## 5. Post-Inference Triage & Tri-State UI
Let top-1 prediction probability be $p_1$ and top-2 be $p_2$.
- **High Confidence State:** If $p_1 \ge 0.70$ and $(p_1 - p_2) \ge 0.30$:
  - Display diagnosed condition with confidence percentage and agronomic guidance.
- **Uncertain State:** If $p_1 < 0.70$ or $(p_1 - p_2) < 0.30$:
  - Display: *"Uncertain foliar pattern. Retake photo under indirect sunlight with lesion centered, or consult local agronomist."*
- **Never Output False Certainty on Borderline Lesions.**

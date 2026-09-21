import shutil
import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONVERTED_DIR = ROOT_DIR / "models" / "potato" / "converted"
MOBILE_DIR = ROOT_DIR / "mobile" / "potato"
MOBILE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copy models
shutil.copy2(CONVERTED_DIR / "supervised_mobilenetv3_float16.tflite", MOBILE_DIR / "supervised_mobilenetv3_float16.tflite")
shutil.copy2(CONVERTED_DIR / "supervised_mobilenetv3_float32.tflite", MOBILE_DIR / "supervised_mobilenetv3_float32.tflite")
shutil.copy2(CONVERTED_DIR / "supervised_mobilenetv3_int8.tflite", MOBILE_DIR / "supervised_mobilenetv3_int8.tflite")

# 2. Labels file
labels = ["early_blight", "healthy", "late_blight"]
with open(MOBILE_DIR / "labels.txt", "w", encoding="utf-8") as f:
    for l in labels:
        f.write(f"{l}\n")

# 3. Preprocessing specification
preprocessing_content = """# Potato Disease Mobile Preprocessing & Inference Specification

**Version:** 1.0 (Mobile Prototype)  
**Target Architecture:** MobileNetV3-Large LiteRT / TFLite  
**Target Hardware:** Android (Java/Kotlin / NDK) & iOS (Swift / CoreML / TFLite)  

---

## 1. Input Specification Contract

| Parameter | Value / Contract |
| :--- | :--- |
| **Tensor Dimensions** | `[1, 224, 224, 3]` (Batch, Height, Width, Channels) |
| **Data Type** | `uint8` or `float32` in `[0.0, 255.0]` |
| **Color Space** | **RGB** (Red, Green, Blue) |
| **Pixel Value Range** | **`[0, 255]`** |
| **Neutral Fill Canvas** | `RGB(114, 114, 114)` |
| **Resize Strategy** | **Aspect-preserving letterboxing** (Bicubic or Bilinear) |

> [!CAUTION]
> **Important Mobile Integration Notes:**
> 1. **Do NOT divide by 255.0 externally:** The model internally expects inputs in `[0, 255]`. Passing inputs pre-divided by 255 will cause double-rescaling!
> 2. **Strict RGB Channel Order:** Ensure image buffer is strictly RGB (not BGR/ARGB).
> 3. **Never squish or stretch aspect ratio:** Letterbox the image with padding `RGB(114, 114, 114)`.

---

## 2. Pre-Inference Quality & Botanical Gate (Safe Abstention)

Before running neural network inference, the mobile client must evaluate two lightweight quality checks:

### Step 2.1: Botanical Foliage Area Check
Verify that a green potato leaf is visible in the frame:
- Convert frame to HSV color space.
- Green foliage mask: $\\text{Hue} \\in [25, 90]$, $\\text{Sat} \\ge 35$, $\\text{Val} \\ge 35$.
- Compute foliage ratio: $\\text{ratio} = \\frac{\\text{green\\_pixels}}{\\text{total\\_pixels}}$.
- **Gate:** If $\\text{ratio} < 0.05$ (less than 5% foliage):
  - **Return:** `unsupported_input`
  - **User Feedback:** *"No potato foliage detected. Please center a potato leaf in the camera frame."*

### Step 2.2: Extreme Blur Detection
Verify that the camera is focused on the leaf surface:
- Compute Laplacian variance on grayscale image: $\\sigma^2_{\\text{Laplacian}}$.
- **Gate:** If $\\sigma^2_{\\text{Laplacian}} < 40.0$:
  - **Return:** `unsupported_input`
  - **User Feedback:** *"Image is too blurry. Tap the screen to focus on the leaf lesion."*

---

## 3. Post-Inference Decision Engine & Abstention Thresholds

- **Top-1 Confidence Threshold:** $\\tau_{\\text{conf}} = 0.60$
- **Top-1 vs Top-2 Margin Gap:** $\\tau_{\\text{margin}} = 0.20$
- If `max_prob < 0.60` or `(p_top1 - p_top2) < 0.20`:
  - **Return:** `uncertain`
  - **User Feedback:** *"Inconclusive symptoms. Please capture a clearer close-up of the leaf spots."*
"""

with open(MOBILE_DIR / "preprocessing.md", "w", encoding="utf-8") as f:
    f.write(preprocessing_content)

# 4. Model manifest
def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "crop": "potato",
    "model_name": "potato_student_mobilenetv3_large",
    "architecture": "MobileNetV3-Large",
    "total_parameters": 2999235,
    "input_shape": [1, 224, 224, 3],
    "input_dtype": "uint8",
    "pixel_range": [0, 255],
    "color_format": "RGB",
    "letterbox_fill": [114, 114, 114],
    "classes": labels,
    "num_classes": 3,
    "class_to_idx": {c: i for i, c in enumerate(labels)},
    "primary_candidate": {
        "format": "float16",
        "filename": "supervised_mobilenetv3_float16.tflite",
        "size_mb": round((MOBILE_DIR / "supervised_mobilenetv3_float16.tflite").stat().st_size / (1024 * 1024), 2),
        "sha256": get_sha256(MOBILE_DIR / "supervised_mobilenetv3_float16.tflite"),
        "test_accuracy": 0.9943,
        "test_macro_f1": 0.9943,
        "brier_score": 0.0099,
        "ece_calibration": 0.0056,
        "decision_parity_with_keras": 1.0,
    },
    "alternative_candidates": {
        "float32": {
            "filename": "supervised_mobilenetv3_float32.tflite",
            "size_mb": round((MOBILE_DIR / "supervised_mobilenetv3_float32.tflite").stat().st_size / (1024 * 1024), 2),
            "sha256": get_sha256(MOBILE_DIR / "supervised_mobilenetv3_float32.tflite"),
        },
        "int8": {
            "filename": "supervised_mobilenetv3_int8.tflite",
            "size_mb": round((MOBILE_DIR / "supervised_mobilenetv3_int8.tflite").stat().st_size / (1024 * 1024), 2),
            "sha256": get_sha256(MOBILE_DIR / "supervised_mobilenetv3_int8.tflite"),
        }
    },
    "abstention_thresholds": {
        "foliage_ratio_min": 0.05,
        "blur_laplacian_var_min": 40.0,
        "confidence_threshold": 0.60,
        "margin_gap_threshold": 0.20
    }
}

with open(MOBILE_DIR / "model_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# 5. Checksum file
with open(MOBILE_DIR / "checksum.sha256", "w", encoding="utf-8") as f:
    for fname in sorted(MOBILE_DIR.iterdir()):
        if fname.name != "checksum.sha256" and fname.is_file():
            sha = get_sha256(fname)
            f.write(f"{sha}  {fname.name}\n")

print("[SUCCESS] Mobile potato packaging complete at mobile/potato/")

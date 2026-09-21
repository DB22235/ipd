"""
scripts/tomato_student/convert_student_litert.py
================================================
Converts the Supervised MobileNetV3-Large Tomato Student Model to LiteRT (TFLite):
  - Float32 (Full precision baseline)
  - Float16 (Post-training float16 quantization - Primary mobile deployment candidate)
  - INT8 (Full integer quantization with representative calibration dataset)

Outputs:
  - models/tomato/converted/tomato_student_float32.tflite
  - models/tomato/converted/tomato_student_float16.tflite
  - models/tomato/converted/tomato_student_int8.tflite
  - mobile/tomato/tomato_student_float16.tflite
  - mobile/tomato/labels.txt
  - mobile/tomato/model_manifest.json
  - mobile/tomato/checksum.sha256
  - mobile/tomato/preprocessing.md
  - reports/tomato/student_v1/conversion_comparison.csv
  - mobile/tomato/conversion_metadata.json
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import cv2

# Force TensorFlow backend
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf

STUDENT_KERAS_PATH = ROOT_DIR / "models/tomato/students/supervised_v1/student_best.keras"
SPLIT_MANIFEST_PATH = ROOT_DIR / "manifests/tomato/teacher_v2/split_manifest.csv"
CONVERTED_DIR = ROOT_DIR / "models/tomato/converted"
MOBILE_DIR = ROOT_DIR / "mobile/tomato"
REPORTS_DIR = ROOT_DIR / "reports/tomato/student_v1"

CLASSES = ["early_blight", "healthy", "late_blight"]
TARGET_SIZE = (300, 300)
LETTERBOX_FILL = (114, 114, 114)


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def letterbox_image(img_bgr: np.ndarray, target_size=(300, 300), bg_color=(114, 114, 114)) -> np.ndarray:
    """Aspect-preserving letterbox padding to target_size."""
    h, w = img_bgr.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)

    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def create_representative_dataset_generator(df_train: pd.DataFrame, num_samples: int = 256):
    """
    Creates a balanced representative calibration dataset for INT8 post-training quantization.
    Draws evenly across early_blight, healthy, and late_blight from the training split.
    """
    per_class = num_samples // len(CLASSES)
    sampled_rows = []
    for c in CLASSES:
        sub = df_train[df_train["class"] == c]
        sample_count = min(per_class, len(sub))
        sampled_rows.append(sub.sample(sample_count, random_state=42))
    calib_df = pd.concat(sampled_rows, ignore_index=True)

    print(f"      [CALIBRATION] Prepared {len(calib_df)} balanced calibration images from training split.")

    def representative_dataset_gen():
        for _, row in calib_df.iterrows():
            img_p = ROOT_DIR / row["path"]
            raw = cv2.imread(str(img_p))
            if raw is None:
                continue
            canvas = letterbox_image(raw, target_size=TARGET_SIZE, bg_color=LETTERBOX_FILL)
            arr = canvas.astype(np.float32)
            inp = np.expand_dims(arr, axis=0)
            yield [inp]

    return representative_dataset_gen


def convert_models():
    print("=" * 75)
    print("      STAGE 4: TOMATO STUDENT LITERT / TFLITE CONVERSION SUITE")
    print("===========================================================================")

    if not STUDENT_KERAS_PATH.exists():
        print(f"[Error] Keras student model not found: {STUDENT_KERAS_PATH}")
        print("Please train the student first via: python scripts/tomato_student/train_student_baseline.py")
        sys.exit(1)

    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  [1/5] Loading Keras Student: {STUDENT_KERAS_PATH.name}...")
    keras_model = tf.keras.models.load_model(str(STUDENT_KERAS_PATH), compile=False)
    input_shape = keras_model.input_shape
    print(f"        Input shape : {input_shape}")
    print(f"        Output shape: {keras_model.output_shape}")
    print(f"        Total params: {keras_model.count_params():,}")

    df_manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    df_train = df_manifest[df_manifest["split"] == "train"].copy()
    rep_gen = create_representative_dataset_generator(df_train, num_samples=256)

    records = []

    # 1. Float32 LiteRT
    print("\n  [2/5] Converting to Float32 TFLite...")
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_fp32 = conv_fp32.convert()
    out_fp32 = CONVERTED_DIR / "tomato_student_float32.tflite"
    out_fp32.write_bytes(tflite_fp32)
    size_fp32_mb = round(out_fp32.stat().st_size / (1024 * 1024), 2)
    sha_fp32 = compute_file_sha256(out_fp32)
    print(f"        [SAVED] -> {out_fp32.name} ({size_fp32_mb} MB | SHA-256: {sha_fp32[:16]}...)")

    records.append({
        "model_id": "tomato_student_supervised_v1",
        "format": "Float32",
        "filename": out_fp32.name,
        "size_bytes": out_fp32.stat().st_size,
        "size_mb": size_fp32_mb,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "sha256": sha_fp32,
    })

    # 2. Float16 LiteRT (Primary Deployment Candidate)
    print("\n  [3/5] Converting to Float16 Quantized TFLite (Primary Mobile Candidate)...")
    conv_fp16 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = conv_fp16.convert()
    out_fp16 = CONVERTED_DIR / "tomato_student_float16.tflite"
    out_fp16.write_bytes(tflite_fp16)
    size_fp16_mb = round(out_fp16.stat().st_size / (1024 * 1024), 2)
    sha_fp16 = compute_file_sha256(out_fp16)
    print(f"        [SAVED] -> {out_fp16.name} ({size_fp16_mb} MB | SHA-256: {sha_fp16[:16]}...)")

    records.append({
        "model_id": "tomato_student_supervised_v1",
        "format": "Float16",
        "filename": out_fp16.name,
        "size_bytes": out_fp16.stat().st_size,
        "size_mb": size_fp16_mb,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "sha256": sha_fp16,
    })

    # 3. INT8 LiteRT (Full Integer Quantization with representative dataset)
    print("\n  [4/5] Converting to Full INT8 Quantized TFLite...")
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = rep_gen
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv_int8.inference_input_type = tf.float32
    conv_int8.inference_output_type = tf.float32
    tflite_int8 = conv_int8.convert()
    out_int8 = CONVERTED_DIR / "tomato_student_int8.tflite"
    out_int8.write_bytes(tflite_int8)
    size_int8_mb = round(out_int8.stat().st_size / (1024 * 1024), 2)
    sha_int8 = compute_file_sha256(out_int8)
    print(f"        [SAVED] -> {out_int8.name} ({size_int8_mb} MB | SHA-256: {sha_int8[:16]}...)")

    records.append({
        "model_id": "tomato_student_supervised_v1",
        "format": "INT8",
        "filename": out_int8.name,
        "size_bytes": out_int8.stat().st_size,
        "size_mb": size_int8_mb,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "sha256": sha_int8,
    })

    # 4. Packaging Release Artifacts into mobile/tomato/
    print("\n  [5/5] Packaging Mobile Release Bundle in mobile/tomato/...")
    mobile_tflite_dest = MOBILE_DIR / "tomato_student_float16.tflite"
    shutil.copy2(out_fp16, mobile_tflite_dest)

    labels_path = MOBILE_DIR / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for c in CLASSES:
            f.write(f"{c}\n")

    manifest_data = {
        "manifest_version": "1.0",
        "project": "IPD Foliar Disease Detection (Tomato Crop)",
        "crop": "tomato",
        "model_status": "tomato mobile student supervised baseline v1 — benchmark validated",
        "primary_model": {
            "model_id": "tomato_mobilenetv3_large_supervised_float16_v1",
            "filename": "tomato_student_float16.tflite",
            "architecture": "MobileNetV3-Large",
            "variant": "Float16 LiteRT",
            "parameters": keras_model.count_params(),
            "size_bytes": out_fp16.stat().st_size,
            "size_mb": size_fp16_mb,
            "sha256": sha_fp16,
            "input_shape": [1, TARGET_SIZE[0], TARGET_SIZE[1], 3],
            "input_dtype": "float32",
            "output_shape": [1, len(CLASSES)],
            "output_type": "probabilities",
            "output_activation": "softmax",
            "normalization_range": [0.0, 255.0],
            "preprocessing_method": "aspect_preserving_letterbox",
            "letterbox_fill_rgb": list(LETTERBOX_FILL),
            "camera_guidance": "50% x 50% central targeting reticle to eliminate distant lesion dilution",
            "classes": CLASSES
        },
        "gates": {
            "blur_min_laplacian_variance": 100.0,
            "botanical_foliage_min_ratio": 0.15,
            "confidence_min_threshold": 0.70,
            "confidence_min_margin": 0.30,
            "supported_states": [
                "early_blight",
                "healthy",
                "late_blight",
                "uncertain",
                "unsupported_input"
            ]
        },
        "limitations": [
            "Dataset v2 balances lab and studio samples; small nascent lesions require viewfinder reticle framing.",
            "Petiole and stem blight (such as field_05) exhibit visual ambiguity and must be routed to uncertainty triage.",
            "Do not deploy as an autonomous agronomic diagnostic tool without agronomist review."
        ]
    }

    manifest_json_path = MOBILE_DIR / "model_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    checksum_path = MOBILE_DIR / "checksum.sha256"
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{sha_fp16}  tomato_student_float16.tflite\n")
        f.write(f"{compute_file_sha256(labels_path)}  labels.txt\n")
        f.write(f"{compute_file_sha256(manifest_json_path)}  model_manifest.json\n")

    # Developer Preprocessing Guide
    preprocessing_md_path = MOBILE_DIR / "preprocessing.md"
    preprocessing_md_content = """# Mobile Preprocessing & Inference Contract: Tomato Mobile Student v1

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
"""
    with open(preprocessing_md_path, "w", encoding="utf-8") as f:
        f.write(preprocessing_md_content)

    # Save summary CSV and conversion metadata JSON
    df_summary = pd.DataFrame(records)
    csv_out = REPORTS_DIR / "conversion_comparison.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"        [SAVED] -> {csv_out.name}")

    meta_json = MOBILE_DIR / "conversion_metadata.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({"conversion_records": records}, f, indent=2)

    print("\n  Summary of Generated LiteRT Artifacts:")
    print(df_summary[["format", "filename", "size_mb", "sha256"]])
    print("\n" + "=" * 75)
    print(" [COMPLETE] LiteRT Mobile Conversion Suite executed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    convert_models()

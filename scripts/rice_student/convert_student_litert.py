"""
scripts/rice_student/convert_student_litert.py
=============================================
Converts BOTH the Supervised and Distilled MobileNetV3 Student models to LiteRT:
  - Float32 (Full precision baseline)
  - Float16 (Post-training float16 quantization)
  - INT8 (Full integer quantization with representative calibration data)

Generates:
  - models/rice/converted/supervised_mobilenetv3_float32.tflite
  - models/rice/converted/supervised_mobilenetv3_float16.tflite
  - models/rice/converted/supervised_mobilenetv3_int8.tflite
  - models/rice/converted/distilled_mobilenetv3_float32.tflite
  - models/rice/converted/distilled_mobilenetv3_float16.tflite
  - models/rice/converted/distilled_mobilenetv3_int8.tflite
  - reports/rice/conversion/student_conversion_comparison.csv
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Generator
import numpy as np
import pandas as pd
from PIL import Image

# Force TensorFlow backend for LiteRT conversion
os.environ["KERAS_BACKEND"] = "tensorflow"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keras
import tensorflow as tf

from src.rice_student.contracts import CLASSES, SPLIT_MANIFEST_PATH
from src.rice_student.data import resolve_image_path, letterbox_image


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def create_representative_dataset_generator(df_train: pd.DataFrame, num_samples: int = 256):
    """
    Creates a balanced representative calibration dataset for INT8 post-training quantization.
    Draws evenly across all 4 classes from the training split.
    """
    per_class = num_samples // len(CLASSES)
    sampled_rows = []
    for c in CLASSES:
        sub = df_train[df_train["class_label"] == c]
        sampled_rows.append(sub.sample(min(per_class, len(sub)), random_state=42))
    calib_df = pd.concat(sampled_rows, ignore_index=True)

    print(f"      [CALIBRATION] Prepared {len(calib_df)} balanced calibration images from training split.")

    def representative_dataset_gen():
        for _, row in calib_df.iterrows():
            img_p = resolve_image_path(row, ROOT_DIR)
            with Image.open(img_p) as img:
                img_rgb = img.convert("RGB")
                processed = letterbox_image(img_rgb, target_size=(224, 224), fill_color=(114, 114, 114))
                arr = np.array(processed, dtype=np.float32)
                # Expand batch dimension [1, 224, 224, 3]
                inp = np.expand_dims(arr, axis=0)
                yield [inp]

    return representative_dataset_gen


def convert_model_suite(
    model_name: str,
    keras_model_path: Path,
    output_dir: Path,
    rep_gen,
) -> List[Dict[str, Any]]:
    print(f"\n" + "=" * 70)
    print(f"  Converting {model_name} ({keras_model_path.name}) to LiteRT...")
    print("=" * 70)

    # 1. Load Keras Model
    raw_model = keras.models.load_model(str(keras_model_path), compile=False)
    
    # 2. Add explicit Softmax probability head if linear logits
    inputs = keras.layers.Input(shape=(224, 224, 3), name="input_image")
    logits = raw_model(inputs)
    probs = keras.layers.Softmax(name="probabilities")(logits)
    export_model = keras.models.Model(inputs=inputs, outputs=probs, name=f"{model_name}_export")

    records = []

    # --- 1. Float32 ---
    print("  [1/3] Converting to Float32 TFLite...")
    conv_fp32 = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_fp32 = conv_fp32.convert()
    out_fp32 = output_dir / f"{model_name}_float32.tflite"
    out_fp32.write_bytes(tflite_fp32)
    size_fp32_mb = round(out_fp32.stat().st_size / (1024 * 1024), 2)
    sha_fp32 = compute_file_sha256(out_fp32)
    print(f"        [SAVED] -> {out_fp32.name} ({size_fp32_mb} MB)")

    records.append({
        "model_role": model_name,
        "format": "Float32",
        "file_name": out_fp32.name,
        "size_bytes": out_fp32.stat().st_size,
        "size_mb": size_fp32_mb,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "sha256": sha_fp32,
    })

    # --- 2. Float16 ---
    print("  [2/3] Converting to Float16 Quantized TFLite...")
    conv_fp16 = tf.lite.TFLiteConverter.from_keras_model(export_model)
    conv_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = conv_fp16.convert()
    out_fp16 = output_dir / f"{model_name}_float16.tflite"
    out_fp16.write_bytes(tflite_fp16)
    size_fp16_mb = round(out_fp16.stat().st_size / (1024 * 1024), 2)
    sha_fp16 = compute_file_sha256(out_fp16)
    print(f"        [SAVED] -> {out_fp16.name} ({size_fp16_mb} MB)")

    records.append({
        "model_role": model_name,
        "format": "Float16",
        "file_name": out_fp16.name,
        "size_bytes": out_fp16.stat().st_size,
        "size_mb": size_fp16_mb,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "sha256": sha_fp16,
    })

    # --- 3. INT8 ---
    print("  [3/3] Converting to Full INT8 Quantized TFLite (with representative dataset)...")
    conv_int8 = tf.lite.TFLiteConverter.from_keras_model(export_model)
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_int8.representative_dataset = rep_gen
    conv_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Keep input/output float32 for seamless mobile drop-in
    conv_int8.inference_input_type = tf.float32
    conv_int8.inference_output_type = tf.float32
    tflite_int8 = conv_int8.convert()
    out_int8 = output_dir / f"{model_name}_int8.tflite"
    out_int8.write_bytes(tflite_int8)
    size_int8_mb = round(out_int8.stat().st_size / (1024 * 1024), 2)
    sha_int8 = compute_file_sha256(out_int8)
    print(f"        [SAVED] -> {out_int8.name} ({size_int8_mb} MB)")

    records.append({
        "model_role": model_name,
        "format": "INT8",
        "file_name": out_int8.name,
        "size_bytes": out_int8.stat().st_size,
        "size_mb": size_int8_mb,
        "input_dtype": "float32",
        "output_dtype": "float32",
        "sha256": sha_int8,
    })

    return records


def main():
    print("=" * 75)
    print("      STAGE 4: LITERT / TFLITE MOBILE EXPORT SUITE")
    print("===========================================================================")

    output_dir = ROOT_DIR / "models" / "rice" / "converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = ROOT_DIR / "reports" / "rice" / "conversion"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT_DIR / SPLIT_MANIFEST_PATH
    df = pd.read_csv(manifest_path)
    df_train = df[(df["crop"] == "rice") & (df["partition"] == "train")].copy()

    # Prepare shared calibration generator
    rep_gen = create_representative_dataset_generator(df_train, num_samples=256)

    supervised_keras = ROOT_DIR / "models" / "rice" / "student_baselines" / "rice_student_mobilenetv3_baseline_best.keras"
    distilled_keras = ROOT_DIR / "models" / "rice" / "distilled_students" / "rice_student_mobilenetv3_distilled_best.keras"

    if not supervised_keras.exists():
        raise FileNotFoundError(f"Supervised model checkpoint not found: {supervised_keras}")
    if not distilled_keras.exists():
        raise FileNotFoundError(f"Distilled model checkpoint not found: {distilled_keras}")

    all_records = []
    # 1. Convert Supervised Student
    rec_sup = convert_model_suite("supervised_mobilenetv3", supervised_keras, output_dir, rep_gen)
    all_records.extend(rec_sup)

    # 2. Convert Distilled Student
    rec_dist = convert_model_suite("distilled_mobilenetv3", distilled_keras, output_dir, rep_gen)
    all_records.extend(rec_dist)

    # 3. Save Summary CSV
    df_summary = pd.DataFrame(all_records)
    csv_out = reports_dir / "student_conversion_comparison.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\n  [SAVED] -> {csv_out.name}")

    print("\n  Summary of Generated LiteRT Artifacts:")
    print(df_summary[["model_role", "format", "file_name", "size_mb"]])

    print("\n" + "=" * 75)
    print(" [COMPLETE] All 6 LiteRT mobile models exported successfully!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

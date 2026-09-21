"""
src/potato_student/export.py
===========================
Native TensorFlow LiteRT (.tflite) export pipeline for the Potato Student Model.
Exports:
  1. Float32 (Full precision baseline)
  2. Float16 (Primary mobile candidate, 16-bit weight quantization)
  3. INT8 (Full integer quantization with representative calibration dataset)
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Generator

import numpy as np
import tensorflow as tf
import keras


def compute_bytes_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def convert_to_float32(model: keras.Model) -> bytes:
    """Converts Keras model to unquantized Float32 LiteRT format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_to_float16(model: keras.Model) -> bytes:
    """Converts Keras model to Float16 LiteRT format (Primary Mobile Target)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def convert_to_int8(
    model: keras.Model,
    representative_samples: np.ndarray,
) -> bytes:
    """
    Converts Keras model to full INT8 LiteRT format using calibration samples.
    Representative samples must be sampled from train or validation sets only.
    """
    def representative_dataset_gen() -> Generator:
        for i in range(min(100, len(representative_samples))):
            sample = representative_samples[i : i + 1]
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    return converter.convert()


def export_student_suite(
    model: keras.Model,
    output_dir: Path,
    calibration_data: Optional[np.ndarray] = None,
    prefix: str = "supervised_mobilenetv3",
) -> Dict[str, Dict[str, Any]]:
    """
    Exports Float32, Float16, and optional INT8 candidates, saving checksums and sizes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, Any]] = {}

    # 1. Float32
    fp32_bytes = convert_to_float32(model)
    fp32_path = output_dir / f"{prefix}_float32.tflite"
    with open(fp32_path, "wb") as f:
        f.write(fp32_bytes)
    manifest["float32"] = {
        "path": str(fp32_path),
        "size_bytes": len(fp32_bytes),
        "size_mb": round(len(fp32_bytes) / (1024 * 1024), 2),
        "sha256": compute_bytes_sha256(fp32_bytes),
    }

    # 2. Float16 (Primary Mobile Target)
    fp16_bytes = convert_to_float16(model)
    fp16_path = output_dir / f"{prefix}_float16.tflite"
    with open(fp16_path, "wb") as f:
        f.write(fp16_bytes)
    manifest["float16"] = {
        "path": str(fp16_path),
        "size_bytes": len(fp16_bytes),
        "size_mb": round(len(fp16_bytes) / (1024 * 1024), 2),
        "sha256": compute_bytes_sha256(fp16_bytes),
        "is_primary_mobile_candidate": True,
    }

    # 3. INT8 (Optional/Experimental)
    if calibration_data is not None and len(calibration_data) > 0:
        try:
            int8_bytes = convert_to_int8(model, calibration_data)
            int8_path = output_dir / f"{prefix}_int8.tflite"
            with open(int8_path, "wb") as f:
                f.write(int8_bytes)
            manifest["int8"] = {
                "path": str(int8_path),
                "size_bytes": len(int8_bytes),
                "size_mb": round(len(int8_bytes) / (1024 * 1024), 2),
                "sha256": compute_bytes_sha256(int8_bytes),
                "note": "Experimental. Subject to XNNPACK Node 124 delegate fallback check.",
            }
        except Exception as e:
            print(f"Warning: INT8 conversion skipped due to: {e}")

    return manifest

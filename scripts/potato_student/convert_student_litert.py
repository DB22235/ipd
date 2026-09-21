"""
scripts/potato_student/convert_student_litert.py
==============================================
LiteRT (.tflite) Multi-Format Converter for Potato Student Model.
Converts:
  1. Float32 (.tflite)
  2. Float16 (.tflite) - Authoritative Primary Mobile Deployment Candidate
  3. INT8 (.tflite) - Experimental (Calibrated on train/val only)
Validates:
  - 100% categorical agreement between Keras and LiteRT Float16 on test samples
  - Packages mobile manifest and labels file
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf
import keras

from src.potato_student.contracts import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
    SPLIT_MANIFEST_PATH,
)
from src.potato_student.data import load_potato_manifest, load_potato_split_to_ram
from src.potato_student.export import export_student_suite
from src.potato_student.package_validator import verify_keras_litert_agreement

CONVERTED_DIR = ROOT_DIR / "models" / "potato" / "converted"
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Potato Student to LiteRT")
    parser.add_argument(
        "--model",
        type=str,
        default="models/potato/student_baselines/run_001/student_best.keras",
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=SPLIT_MANIFEST_PATH,
        help="Path to split manifest CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = ROOT_DIR / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print("=" * 75)
    print("      STAGE 3: POTATO STUDENT LITERt (.TFLITE) EXPORT")
    print("=" * 75)

    # 1. Load Model
    print(f"\n[1/4] Loading Keras model: {model_path}")
    model = keras.models.load_model(model_path)

    # 2. Load Calibration Data from Train/Val (Strictly No Test Data)
    manifest_path = ROOT_DIR / args.manifest
    df_val = load_potato_manifest(manifest_path, partition="val")
    X_val, _, _ = load_potato_split_to_ram(df_val.head(100), target_size=(224, 224), root_dir=ROOT_DIR)

    # 3. Export LiteRT Suite
    print(f"\n[2/4] Converting to LiteRT formats (FP32, FP16, INT8)...")
    manifest = export_student_suite(
        model=model,
        output_dir=CONVERTED_DIR,
        calibration_data=X_val,
        prefix="supervised_mobilenetv3",
    )

    for fmt, info in manifest.items():
        print(f"  -> {fmt.upper()}: {info['size_mb']} MB (SHA256: {info['sha256'][:16]}...)")

    # 4. Save Label File
    print("\n[3/4] Exporting mobile labels file...")
    labels_path = CONVERTED_DIR / "potato_labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for c in CLASSES:
            f.write(f"{c}\n")
    print(f"  -> Saved labels to {labels_path}")

    # 5. Verify Agreement on Validation Samples
    print("\n[4/4] Verifying Keras-to-LiteRT Float16 agreement on 100 validation samples...")
    fp16_path = Path(manifest["float16"]["path"])
    agreement_res = verify_keras_litert_agreement(model, fp16_path, X_val)
    print(f"  -> Agreement Rate: {agreement_res['agreement_rate'] * 100:.2f}%")
    print(f"  -> Max Logit Diff: {agreement_res['max_abs_diff']:.6f}")

    if agreement_res["agreement_pass"]:
        print("[PASS] 100% categorical agreement verified between Keras and LiteRT Float16.")
    else:
        print(f"[WARN] Found {agreement_res['mismatches']} mismatch(es)!")

    # Save Package Manifest
    pkg_manifest = {
        "crop": "potato",
        "primary_model": "supervised_mobilenetv3_float16.tflite",
        "primary_model_size_mb": manifest["float16"]["size_mb"],
        "primary_model_sha256": manifest["float16"]["sha256"],
        "agreement_with_keras": agreement_res["agreement_rate"],
        "input_shape": [1, 224, 224, 3],
        "input_dtype": "uint8",
        "output_shape": [1, 3],
        "classes": CLASSES,
        "labels_file": "potato_labels.txt",
        "formats_available": list(manifest.keys()),
    }
    pkg_manifest_path = CONVERTED_DIR / "mobile_package_manifest.json"
    with open(pkg_manifest_path, "w", encoding="utf-8") as f:
        json.dump(pkg_manifest, f, indent=2)
    print(f"\nSaved mobile package manifest to {pkg_manifest_path}")


if __name__ == "__main__":
    main()

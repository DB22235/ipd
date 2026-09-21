import json
import hashlib
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

# 1. Generate conversion_agreement_manifest.csv
split_manifest_path = ROOT_DIR / "manifests/potato/potato_split_manifest_v1.csv"
df = pd.read_csv(split_manifest_path)
df_val = df[df["partition"] == "val"].head(100).copy()
df_val["selection_reason"] = "Held-out validation sample for LiteRT conversion parity check (first 100 deterministic val samples)"

agreement_manifest_path = ROOT_DIR / "manifests/potato/conversion_agreement_manifest.csv"
df_val.to_csv(agreement_manifest_path, index=False)
print(f"[1/2] Saved {len(df_val)} rows to {agreement_manifest_path}")

# 2. Build model_registry.json
keras_model_path = ROOT_DIR / "models/potato/student_baselines/run_001/student_best.keras"
float16_path = ROOT_DIR / "mobile/potato/supervised_mobilenetv3_float16.tflite"
float32_path = ROOT_DIR / "mobile/potato/supervised_mobilenetv3_float32.tflite"
int8_path = ROOT_DIR / "mobile/potato/supervised_mobilenetv3_int8.tflite"

registry = {
    "registry_version": "1.0.0",
    "crop": "potato",
    "model_id": "potato_student_mobilenetv3_supervised_v1",
    "model_role": "supervised_mobile_student",
    "architecture": "MobileNetV3-Large",
    "framework": "Keras 3 / TensorFlow 2.18",
    "total_parameters": 2999235,
    "input_contract": {
        "dimensions": [1, 224, 224, 3],
        "input_dtype": "uint8",
        "value_range": [0, 255],
        "color_space": "RGB",
        "preprocessing_canvas": "Aspect-preserving letterbox with fill (114, 114, 114)",
        "double_rescaling_protection": "Embedded Rescaling(scale=1.0, dtype='float32') casts uint8 to float32 without dividing by 255.0, preserving raw range for MobileNetV3 native preprocessing."
    },
    "classes": ["early_blight", "healthy", "late_blight"],
    "class_to_idx": {
        "early_blight": 0,
        "healthy": 1,
        "late_blight": 2
    },
    "artifacts": {
        "primary_mobile_candidate": {
            "format": "float16",
            "filename": "supervised_mobilenetv3_float16.tflite",
            "path": "mobile/potato/supervised_mobilenetv3_float16.tflite",
            "size_bytes": float16_path.stat().st_size,
            "size_mb": round(float16_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(float16_path),
            "locked_test_accuracy": 0.9943,
            "locked_test_macro_f1": 0.9943,
            "brier_score": 0.0099,
            "ece_calibration": 0.0056,
            "decision_parity_with_keras": 1.0,
            "release_status": "prototype_candidate"
        },
        "source_keras_checkpoint": {
            "format": "keras",
            "filename": "student_best.keras",
            "path": "models/potato/student_baselines/run_001/student_best.keras",
            "size_bytes": keras_model_path.stat().st_size,
            "size_mb": round(keras_model_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(keras_model_path),
            "training_run": "run_001",
            "training_hardware": "CPU Intel oneDNN 12-thread RAM preloaded",
            "total_epochs": 29,
            "best_val_loss": 0.00124,
            "best_val_accuracy": 1.0
        },
        "alternative_float32": {
            "format": "float32",
            "filename": "supervised_mobilenetv3_float32.tflite",
            "path": "mobile/potato/supervised_mobilenetv3_float32.tflite",
            "size_bytes": float32_path.stat().st_size,
            "size_mb": round(float32_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(float32_path)
        },
        "alternative_int8": {
            "format": "int8",
            "filename": "supervised_mobilenetv3_int8.tflite",
            "path": "mobile/potato/supervised_mobilenetv3_int8.tflite",
            "size_bytes": int8_path.stat().st_size,
            "size_mb": round(int8_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_sha256(int8_path),
            "status": "experimental"
        }
    },
    "provenance": {
        "dataset_version": "PlantVillage/Kaggle clean potato dataset v1.0",
        "split_version": "potato_split_manifest_v1",
        "split_manifest_sha256": compute_sha256(split_manifest_path),
        "split_rule": "Group-isolated pHash clustering (threshold <= 4) with multi-bucket proportional deficit allocation (70% train, 15% val, 15% test). Zero exact or group leakage.",
        "conversion_agreement_manifest": "manifests/potato/conversion_agreement_manifest.csv",
        "field_validation": "Incomplete (laboratory background benchmark)",
        "distillation_performed": False,
        "release_status": "prototype_candidate"
    },
    "abstention_thresholds": {
        "foliage_ratio_min": 0.05,
        "blur_laplacian_var_min": 40.0,
        "confidence_threshold": 0.60,
        "margin_gap_threshold": 0.20
    }
}

registry_path = ROOT_DIR / "models/potato/model_registry.json"
with open(registry_path, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

print(f"[2/2] Saved model registry to {registry_path}")

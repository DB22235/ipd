"""
tools/freeze_rice_teacher.py
============================
Formally freezes the validated rice teacher checkpoint.
Implements Section 10 of rice_post_training_evaluation.md.

Computes:
  - SHA-256 checksum of rice_teacher_efficientnetb3_best.keras
  - Creates immutable release manifest: models/rice_teacher_v1/rice_teacher_v1_field_validated.json
  - Saves models/rice_teacher_v1/checksum.txt
"""

import os
import sys
import json
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "rice_teacher_v1"
MODEL_FILE = MODEL_DIR / "rice_teacher_efficientnetb3_best.keras"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 75)
    print("       FORMAL RELEASE FREEZE: RICE TEACHER MODEL V1 (FIELD-VALIDATED)")
    print("=" * 75)

    if not MODEL_FILE.exists():
        print(f"Error: Model checkpoint not found at {MODEL_FILE}")
        sys.exit(1)

    print(f"Computing SHA-256 checksum for: {MODEL_FILE.name} ...")
    sha256_hash = compute_sha256(MODEL_FILE)
    file_size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
    print(f"  SHA-256 Checksum : {sha256_hash}")
    print(f"  Model File Size  : {file_size_mb:.2f} MB")

    # Save standalone checksum.txt
    checksum_path = MODEL_DIR / "checksum.txt"
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"SHA-256: {sha256_hash}  {MODEL_FILE.name}\n")
    print(f"  [OK] Saved -> {checksum_path.name}")

    # Load existing manifest if present
    base_manifest = MODEL_DIR / "model_manifest.json"
    manifest_data = {}
    if base_manifest.exists():
        with open(base_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    # Construct Frozen Release Manifest
    freeze_manifest = {
        "release_version": "rice_teacher_v1_field_validated",
        "status": "FROZEN_AND_ACCEPTED",
        "crop": "rice",
        "architecture": "EfficientNetB3 (Two-Stage Transfer Learning)",
        "framework": "Keras 3 with PyTorch CUDA Backend (torch)",
        "model_file": MODEL_FILE.name,
        "sha256_checksum": sha256_hash,
        "file_size_mb": round(file_size_mb, 2),
        "input_contract": {
            "target_size": [300, 300],
            "channels": 3,
            "preprocessing": "aspect_preserving_letterbox",
            "border_fill": [114, 114, 114],
            "color_space": "RGB",
            "normalization": "none (built-in EfficientNet scaling layer)"
        },
        "classes": [
            "blast",
            "blight",
            "brown_spot",
            "healthy"
        ],
        "class_mapping": {
            0: "blast",
            1: "blight",
            2: "brown_spot",
            3: "healthy"
        },
        "split_version": "rice_split_v1 (leakage-safe, group-disjoint)",
        "validation_gates": {
            "locked_test_accuracy": 0.9918,
            "locked_test_macro_f1": 0.9879,
            "field_holdout_accuracy": 1.0000,
            "expected_calibration_error": 0.0047,
            "background_invariance_gate": "PASSED (0.0% spurious background flips)",
            "biological_causality_gate": "PASSED (lesion-responsive)"
        },
        "abstention_policy": {
            "confidence_threshold": 0.60,
            "entropy_threshold": 0.85,
            "min_foliage_ratio": 0.04,
            "user_message": "Uncertain result. Capture a closer, well-lit image of one rice leaf or use online analysis."
        },
        "downstream_student_readiness": {
            "eligible_for_distillation": True,
            "recommended_student_candidates": [
                "MobileNetV3-Large",
                "MobileNetV3-Small",
                "MobileNetV2"
            ],
            "kd_loss_formulation": "alpha * CE(y, student_logits) + (1 - alpha) * T^2 * KL(softmax(teacher_logits / T), softmax(student_logits / T))"
        }
    }

    freeze_path = MODEL_DIR / "rice_teacher_v1_field_validated.json"
    with open(freeze_path, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2)
    print(f"  [OK] Saved Frozen Release Manifest -> {freeze_path.name}")

    print("=" * 75)
    print("PHASE 4 COMPLETE: TEACHER VERSION FROZEN & REGISTERED")
    print("=" * 75)


if __name__ == "__main__":
    main()

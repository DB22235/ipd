"""
src/rice_student/contracts.py
=============================
Authoritative, immutable contracts for the IPD Rice Student Model.
Guarantees alignment between the frozen EfficientNetB3 teacher and the
MobileNetV3 student model to prevent label inversion and data corruption.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

# 1. Authoritative Class Definitions (Identical to Frozen Teacher V1)
CLASSES: List[str] = ["blast", "blight", "brown_spot", "healthy"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)

# 2. Input Dimensions and Preprocessing Specifications
INPUT_SHAPE_STUDENT: Tuple[int, int, int] = (224, 224, 3)
INPUT_SHAPE_TEACHER: Tuple[int, int, int] = (300, 300, 3)
PIXEL_VALUE_RANGE: Tuple[float, float] = (0.0, 255.0)  # Raw unscaled float32
NEUTRAL_BG_COLOR: Tuple[int, int, int] = (114, 114, 114)
PREPROCESSING_NAME: str = "aspect_preserving_letterbox"

# 3. Reference Paths (Relative to Repo Root)
TEACHER_MODEL_PATH: str = "models/rice/teachers/rice_teacher_efficientnetb3_best.keras" if (ROOT_DIR / "models/rice/teachers/rice_teacher_efficientnetb3_best.keras").exists() else "models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras"
TEACHER_MANIFEST_PATH: str = "models/rice/teachers/rice_teacher_v1_field_validated.json" if (ROOT_DIR / "models/rice/teachers/rice_teacher_v1_field_validated.json").exists() else "models/rice_teacher_v1/rice_teacher_v1_field_validated.json"
TEACHER_SHA256: str = "2eca4294596905fb6231911b66cbeafc90ac8528ecd34461327b43074ec9e738"

SPLIT_MANIFEST_PATH: str = "manifests/rice/split_manifest_v1.csv"
EXPECTED_TOTAL_RICE_SAMPLES: int = 4932
EXPECTED_SPLIT_COUNTS: Dict[str, int] = {"train": 3315, "val": 636, "test": 981}
EXPECTED_CLASS_COUNTS: Dict[str, int] = {
    "healthy": 1488,
    "blight": 1284,
    "brown_spot": 1200,
    "blast": 960,
}

# 4. Mobile Inference Constraints
TARGET_MAX_LATENCY_MS: float = 50.0
TARGET_MAX_MODEL_SIZE_MB: float = 20.0
LITERT_COMPATIBLE_OPS: List[str] = ["CONV_2D", "DEPTHWISE_CONV_2D", "HARD_SWISH", "RELU6", "ADD", "MUL"]


def compute_file_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file in 8MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def assert_contract_integrity(root_dir: Path = None) -> Dict[str, Any]:
    """
    Validates complete system contract integrity:
      - Checks teacher model checkpoint and SHA-256 hash
      - Checks teacher json release manifest
      - Checks split manifest and verifies crop filtering + sample counts
      - Verifies 100% class order agreement
    Returns a dictionary summarizing validation results. Raises AssertionError on failure.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
    root_dir = Path(root_dir)

    results: Dict[str, Any] = {"checks": {}, "passed": False}

    # A. Teacher Checkpoint & SHA-256
    teacher_file = root_dir / TEACHER_MODEL_PATH
    if not teacher_file.exists():
        raise FileNotFoundError(f"Teacher model checkpoint not found: {teacher_file}")
    
    computed_sha = compute_file_sha256(teacher_file)
    if computed_sha != TEACHER_SHA256:
        raise AssertionError(
            f"Teacher SHA-256 mismatch! Expected {TEACHER_SHA256}, got {computed_sha}"
        )
    results["checks"]["teacher_sha256"] = {
        "status": "PASS",
        "file": str(teacher_file),
        "sha256": computed_sha,
    }

    # B. Teacher Release Manifest
    teacher_manifest_file = root_dir / TEACHER_MANIFEST_PATH
    if not teacher_manifest_file.exists():
        raise FileNotFoundError(f"Teacher manifest not found: {teacher_manifest_file}")
    
    with open(teacher_manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    
    manifest_classes = manifest_data.get("classes", [])
    if manifest_classes != CLASSES:
        raise AssertionError(
            f"Teacher manifest classes {manifest_classes} do not match contract {CLASSES}"
        )
    
    manifest_mapping = {int(k): v for k, v in manifest_data.get("class_mapping", {}).items()}
    if manifest_mapping != IDX_TO_CLASS:
        raise AssertionError(
            f"Teacher class mapping {manifest_mapping} does not match contract {IDX_TO_CLASS}"
        )
    results["checks"]["teacher_manifest"] = {
        "status": "PASS",
        "release_version": manifest_data.get("release_version"),
        "class_mapping": manifest_mapping,
    }

    # C. Split Manifest & Multi-Crop Infiltration Guard
    split_file = root_dir / SPLIT_MANIFEST_PATH
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_file}")
    
    df = pd.read_csv(split_file)
    if "crop" not in df.columns or "class_label" not in df.columns or "partition" not in df.columns:
        raise AssertionError(f"Split manifest missing required columns: {split_file}")
    
    df_rice = df[df["crop"] == "rice"]
    if len(df_rice) != EXPECTED_TOTAL_RICE_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_RICE_SAMPLES} rice samples, found {len(df_rice)}"
        )
    
    partition_counts = df_rice["partition"].value_counts().to_dict()
    for part, expected_cnt in EXPECTED_SPLIT_COUNTS.items():
        actual_cnt = partition_counts.get(part, 0)
        if actual_cnt != expected_cnt:
            raise AssertionError(
                f"Split mismatch for '{part}': expected {expected_cnt}, found {actual_cnt}"
            )
    
    detected_classes = sorted(df_rice["class_label"].unique())
    if detected_classes != sorted(CLASSES):
        raise AssertionError(
            f"Manifest classes {detected_classes} do not match contract classes {sorted(CLASSES)}"
        )
    
    results["checks"]["split_manifest"] = {
        "status": "PASS",
        "file": str(split_file),
        "total_rice_samples": len(df_rice),
        "partitions": partition_counts,
        "classes": detected_classes,
    }

    results["passed"] = True
    return results

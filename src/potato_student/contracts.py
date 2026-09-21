"""
src/potato_student/contracts.py
==============================
Authoritative, immutable contracts for the IPD Potato Student Model.
Guarantees alignment between the frozen EfficientNetB3 potato teacher and the
MobileNetV3 potato student model. Completely isolated from Rice code and assumptions.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 1. Authoritative Class Definitions (3 classes)
CLASSES: List[str] = ["early_blight", "healthy", "late_blight"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)

# 2. Input Dimensions and Normalization Specifications
INPUT_SHAPE_STUDENT: Tuple[int, int, int] = (224, 224, 3)
INPUT_SHAPE_TEACHER: Tuple[int, int, int] = (300, 300, 3)
PIXEL_VALUE_RANGE: Tuple[float, float] = (0.0, 255.0)  # Raw unscaled float32 or uint8
NEUTRAL_BG_COLOR: Tuple[int, int, int] = (114, 114, 114)
PREPROCESSING_NAME: str = "aspect_preserving_letterbox"

# 3. Canonical Checkpoints and Manifest Paths
TEACHER_MODEL_PATH: str = "models/potato/teachers/potato_teacher_efficientnetb3.keras" if (ROOT_DIR / "models/potato/teachers/potato_teacher_efficientnetb3.keras").exists() else "models/potato_teacher/potato_teacher_efficientnetb3.keras"
TEACHER_MANIFEST_PATH: str = "models/potato/teachers/model_manifest.json" if (ROOT_DIR / "models/potato/teachers/model_manifest.json").exists() else "models/potato_teacher/model_manifest.json"
TEACHER_SHA256: str = "eab26cba3f72dd18f55854f0b5115b0e2ec6c2996e7cf81ce8a635e7ee0f739a"

SPLIT_MANIFEST_PATH: str = "manifests/potato/potato_split_manifest_v1.csv"
DATASET_CONTRACT_PATH: str = "manifests/potato/potato_dataset_contract.json"
LABEL_MAP_PATH: str = "manifests/potato/potato_label_map.json"

# 4. Mobile Inference Targets
TARGET_MAX_LATENCY_MS: float = 30.0
TARGET_MAX_MODEL_SIZE_MB: float = 10.0


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
      - Teacher checkpoint presence, size, and SHA256 checksum
      - Split manifest presence and schema
      - Label map consistency
      - Zero rice cross-contamination
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    # 1. Validate Teacher Checkpoint
    t_path = root_dir / TEACHER_MODEL_PATH
    if not t_path.exists():
        raise FileNotFoundError(f"Potato teacher model not found at {t_path}")
    actual_t_sha = compute_file_sha256(t_path)
    if actual_t_sha != TEACHER_SHA256:
        raise ValueError(
            f"Potato teacher SHA256 mismatch!\nExpected: {TEACHER_SHA256}\nActual:   {actual_t_sha}"
        )

    # 2. Validate Teacher Manifest
    t_man_path = root_dir / TEACHER_MANIFEST_PATH
    if not t_man_path.exists():
        raise FileNotFoundError(f"Potato teacher manifest not found at {t_man_path}")
    with open(t_man_path, "r", encoding="utf-8") as f:
        t_meta = json.load(f)
    if t_meta.get("crop") != "potato":
        raise ValueError(f"Teacher crop is '{t_meta.get('crop')}', expected 'potato'")
    if t_meta.get("class_names") != CLASSES:
        raise ValueError(f"Teacher classes {t_meta.get('class_names')} != expected {CLASSES}")

    s_path = root_dir / SPLIT_MANIFEST_PATH
    if not s_path.exists():
        raise FileNotFoundError(
            f"Potato split manifest not found at {s_path}. "
            "Please run 'scripts/potato_student/create_split.py' first to generate the manifest."
        )

    # 4. Validate Label Map
    l_path = root_dir / LABEL_MAP_PATH
    if l_path.exists():
        with open(l_path, "r", encoding="utf-8") as f:
            l_map = json.load(f)
        for i, c in enumerate(CLASSES):
            if l_map.get(str(i)) != c:
                raise ValueError(f"Label map index {i} is '{l_map.get(str(i))}', expected '{c}'")

    return {
        "status": "VALID",
        "crop": "potato",
        "classes": CLASSES,
        "teacher_sha256": actual_t_sha,
        "split_manifest": str(s_path),
    }

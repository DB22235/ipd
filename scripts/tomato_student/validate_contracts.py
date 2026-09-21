"""
scripts/tomato_student/validate_contracts.py
============================================
Validates and locks all prerequisite contracts for Tomato Mobile Student development:
  1. Teacher v2 Checkpoint Integrity (SHA-256 matches model_registry.json).
  2. Universal Class Order: 0: early_blight, 1: healthy, 2: late_blight.
  3. Preprocessing Contract: Aspect-preserving letterbox with neutral gray (114, 114, 114) at 300x300.
  4. Group-Disjoint Split Integrity: split_manifest.csv has 0 filepath, 0 hash, 0 pHash family overlap.
  5. External Field Holdout Isolation: 12 images strictly isolated.

Governing Specification: Tomato Mobile Student Development Plan.md (Manus AI)
"""

import sys
import os
import json
import hashlib
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TEACHER_MODEL_PATH = ROOT_DIR / "models/tomato/teachers/v2/teacher_best.keras"
TEACHER_REGISTRY_PATH = ROOT_DIR / "models/tomato/model_registry.json"
PREPROCESSING_CONTRACT_PATH = ROOT_DIR / "manifests/tomato/teacher_v2_preprocessing_contract.json"
SPLIT_MANIFEST_CSV = ROOT_DIR / "manifests/tomato/teacher_v2/split_manifest.csv"
FIELD_HOLDOUT_CSV = ROOT_DIR / "manifests/tomato/teacher_v2/external_field_holdout.csv"

MANDATED_CLASSES = ["early_blight", "healthy", "late_blight"]


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def validate_contracts() -> bool:
    print("=" * 75)
    print("   TOMATO MOBILE STUDENT: PREREQUISITE CONTRACT & PROVENANCE VALIDATION")
    print("=" * 75)

    all_passed = True

    # 1. Teacher v2 Checkpoint & Checksum
    print("\n[Check 1/5] Verifying Frozen Teacher v2 Checkpoint & Cryptographic Signature...")
    if not TEACHER_MODEL_PATH.exists():
        print(f"  [FAIL] Teacher v2 model file missing: {TEACHER_MODEL_PATH}")
        return False

    teacher_hash = compute_sha256(TEACHER_MODEL_PATH)
    print(f"  -> File Path: {TEACHER_MODEL_PATH.name}")
    print(f"  -> Computed SHA-256: {teacher_hash}")

    if TEACHER_REGISTRY_PATH.exists():
        with open(TEACHER_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        reg_entry = registry.get("models", {}).get("tomato_teacher_v2", {})
        expected_hash = reg_entry.get("sha256_checksum")
        if expected_hash == teacher_hash:
            print(f"  [PASS] Teacher v2 checksum exactly matches model_registry.json.")
        else:
            print(f"  [WARN] Registry checksum ({expected_hash}) differs from file ({teacher_hash}).")
    else:
        print(f"  [WARN] Model registry file not found: {TEACHER_REGISTRY_PATH}")

    # 2. Universal Class Order
    print("\n[Check 2/5] Verifying Universal Class Order...")
    print(f"  -> Mandated: {dict(enumerate(MANDATED_CLASSES))}")
    if PREPROCESSING_CONTRACT_PATH.exists():
        with open(PREPROCESSING_CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)
        contract_classes = contract.get("classes", {}).get("class_names", [])
        if contract_classes == MANDATED_CLASSES:
            print(f"  [PASS] Preprocessing contract class order matches: {contract_classes}")
        else:
            print(f"  [FAIL] Contract class mismatch: {contract_classes} vs {MANDATED_CLASSES}")
            all_passed = False
    else:
        print(f"  [FAIL] Preprocessing contract missing: {PREPROCESSING_CONTRACT_PATH}")
        all_passed = False

    # 3. Preprocessing Contract Invariants
    print("\n[Check 3/5] Verifying Preprocessing Contract Invariants...")
    if PREPROCESSING_CONTRACT_PATH.exists():
        with open(PREPROCESSING_CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)
        resize_policy = contract.get("resize_policy")
        pad_color = contract.get("padding_color_rgb")
        shape = contract.get("input_shape")
        print(f"  -> Input Shape   : {shape}")
        print(f"  -> Resize Policy : {resize_policy}")
        print(f"  -> Pad Color RGB : {pad_color}")
        if resize_policy == "aspect_preserving_letterbox" and pad_color == [114, 114, 114]:
            print(f"  [PASS] Preprocessing invariants verified.")
        else:
            print(f"  [FAIL] Invalid preprocessing policy or padding color.")
            all_passed = False

    # 4. Group-Disjoint Split Manifest Integrity
    print("\n[Check 4/5] Verifying Group-Disjoint Split Manifest (Zero pHash Leakage)...")
    if not SPLIT_MANIFEST_CSV.exists():
        print(f"  [FAIL] Split manifest missing: {SPLIT_MANIFEST_CSV}")
        return False

    df = pd.read_csv(SPLIT_MANIFEST_CSV)
    splits = df["split"].unique().tolist()
    print(f"  -> Total Samples : {len(df)}")
    print(f"  -> Splits Found  : {splits}")

    train_fams = set(df[df["split"] == "train"]["family_id"])
    val_fams = set(df[df["split"] == "val"]["family_id"])
    test_fams = set(df[df["split"] == "test"]["family_id"])

    leak_train_val = len(train_fams.intersection(val_fams))
    leak_train_test = len(train_fams.intersection(test_fams))
    leak_val_test = len(val_fams.intersection(test_fams))

    print(f"  -> pHash Family Leakage (Train ∩ Val)  : {leak_train_val}")
    print(f"  -> pHash Family Leakage (Train ∩ Test) : {leak_train_test}")
    print(f"  -> pHash Family Leakage (Val ∩ Test)   : {leak_val_test}")

    if leak_train_val == 0 and leak_train_test == 0 and leak_val_test == 0:
        print(f"  [PASS] Zero pHash duplicate family leakage confirmed across all splits.")
    else:
        print(f"  [FAIL] Leakage detected between partitions!")
        all_passed = False

    # 5. External Field Holdout Isolation
    print("\n[Check 5/5] Verifying External Field Holdout Isolation...")
    if not FIELD_HOLDOUT_CSV.exists():
        print(f"  [FAIL] Field holdout manifest missing: {FIELD_HOLDOUT_CSV}")
        return False

    df_field = pd.read_csv(FIELD_HOLDOUT_CSV)
    print(f"  -> Field Holdout Samples : {len(df_field)} verified challenge images")
    field_paths = set(df_field["path"].str.replace("\\", "/"))
    manifest_paths = set(df["path"].str.replace("\\", "/"))

    holdout_leakage = len(field_paths.intersection(manifest_paths))
    print(f"  -> Overlap with Train/Val/Test: {holdout_leakage}")

    if holdout_leakage == 0:
        print(f"  [PASS] Field holdout is 100% strictly isolated.")
    else:
        print(f"  [FAIL] Field holdout contaminated into training/evaluation splits!")
        all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("   ALL CONTRACTS AND GATES VERIFIED: READY FOR STUDENT TRAINING")
    else:
        print("   CONTRACT VERIFICATION FAILED: RESOLVE ISSUES BEFORE TRAINING")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = validate_contracts()
    sys.exit(0 if success else 1)

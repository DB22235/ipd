"""
scripts/potato_student/create_split.py
=====================================
Generates an immutable, leakage-safe split manifest for Potato.
Enforces:
  1. Group-level isolation: All images belonging to the same pHash cluster
     or augmented family are locked strictly into the same partition.
  2. Stratified group allocation: Preserves ~70% train, ~15% val, ~15% test
     distribution across each class.
  3. Immutable manifest generation:
     - manifests/potato/potato_split_manifest_v1.csv
     - manifests/potato/potato_dataset_contract.json
     - manifests/potato/potato_label_map.json
     - reports/potato/split_integrity_report.md
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = ROOT_DIR / "manifests" / "potato"
REPORT_DIR = ROOT_DIR / "reports" / "potato"

PRE_AUDIT_CATALOG = MANIFEST_DIR / "potato_pre_audit_catalog.csv"
OUTPUT_SPLIT_MANIFEST = MANIFEST_DIR / "potato_split_manifest_v1.csv"
DATASET_CONTRACT_PATH = MANIFEST_DIR / "potato_dataset_contract.json"
LABEL_MAP_PATH = MANIFEST_DIR / "potato_label_map.json"
SPLIT_INTEGRITY_REPORT = REPORT_DIR / "split_integrity_report.md"

TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

CLASSES = ["early_blight", "healthy", "late_blight"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def allocate_groups_stratified(
    df: pd.DataFrame,
    ratios: Dict[str, float] = TARGET_RATIOS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Allocates groups to train, val, and test stratified by class using proportional deficit allocation.
    Guarantees that no group_id is ever split across partitions and all partitions receive balanced classes.
    """
    np.random.seed(seed)
    df["partition"] = ""
    df["class_idx"] = df["class_label"].map(CLASS_TO_IDX)

    for cls in CLASSES:
        cls_df = df[df["class_label"] == cls]
        group_sizes = cls_df.groupby("group_id").size().to_dict()
        groups = list(group_sizes.keys())
        np.random.shuffle(groups)

        total_samples = len(cls_df)
        train_target = int(total_samples * ratios["train"])
        val_target = int(total_samples * ratios["val"])
        test_target = total_samples - train_target - val_target

        curr_train, curr_val, curr_test = 0, 0, 0
        group_partitions = {}

        for g in groups:
            sz = group_sizes[g]
            need_train = max(0, train_target - curr_train)
            need_val = max(0, val_target - curr_val)
            need_test = max(0, test_target - curr_test)

            deficits = {
                "train": need_train / train_target if train_target > 0 else 0,
                "val": need_val / val_target if val_target > 0 else 0,
                "test": need_test / test_target if test_target > 0 else 0,
            }

            if all(v == 0 for v in deficits.values()):
                best_part = min(
                    {"train": curr_train / train_target, "val": curr_val / val_target, "test": curr_test / test_target},
                    key=lambda k: {"train": curr_train / train_target, "val": curr_val / val_target, "test": curr_test / test_target}[k],
                )
            else:
                best_part = max(deficits, key=deficits.get)

            group_partitions[g] = best_part
            if best_part == "train":
                curr_train += sz
            elif best_part == "val":
                curr_val += sz
            else:
                curr_test += sz

        # Assign back
        for g, part in group_partitions.items():
            mask = (df["class_label"] == cls) & (df["group_id"] == g)
            df.loc[mask, "partition"] = part

    return df


def verify_split_integrity(df: pd.DataFrame):
    """Verifies that no exact or group leakage exists between partitions, and all classes are present."""
    # 1. Exact SHA-256 leakage check
    sha_leakage = 0
    for sha, grp in df.groupby("sha256"):
        if len(grp["partition"].unique()) > 1:
            sha_leakage += 1

    # 2. Group ID leakage check
    grp_leakage = 0
    for gid, grp in df.groupby("group_id"):
        if len(grp["partition"].unique()) > 1:
            grp_leakage += 1

    # 3. Class balance verification (no zero-count partitions)
    crosstab = pd.crosstab(df["class_label"], df["partition"])
    for p in ["train", "val", "test"]:
        for c in CLASSES:
            count = crosstab.loc[c, p] if (c in crosstab.index and p in crosstab.columns) else 0
            assert count > 0, f"CRITICAL: Class '{c}' has 0 samples in partition '{p}'!"

    assert sha_leakage == 0, f"CRITICAL: Found {sha_leakage} SHA-256 hashes crossing partitions!"
    assert grp_leakage == 0, f"CRITICAL: Found {grp_leakage} group_ids crossing partitions!"
    print("[PASS] Split integrity verified: Zero cross-partition leakage and balanced class presence.")


def save_contract_and_label_map(df: pd.DataFrame):
    # 1. Label Map
    label_map = {str(i): c for i, c in enumerate(CLASSES)}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"  -> Saved label map: {LABEL_MAP_PATH}")

    # 2. Class and partition sample counts
    class_counts = df["class_label"].value_counts().to_dict()
    split_counts = df["partition"].value_counts().to_dict()

    # 3. Compute SHA256 of the generated split manifest
    split_manifest_sha = hashlib.sha256(open(OUTPUT_SPLIT_MANIFEST, "rb").read()).hexdigest()

    # 4. Dataset Contract
    contract = {
        "crop": "potato",
        "contract_version": "1.0.0",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "class_order_rule": "Strict index mapping: early_blight=0, healthy=1, late_blight=2. Never sort filesystem.",
        "input_shape_student": [224, 224, 3],
        "input_shape_teacher": [300, 300, 3],
        "pixel_normalization": "Keras builtin Rescaling 1/255 + ImageNet Normalization, aspect-preserving letterbox with fill (114, 114, 114)",
        "split_manifest_file": "manifests/potato/potato_split_manifest_v1.csv",
        "split_manifest_sha256": split_manifest_sha,
        "total_samples": len(df),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "healthy_trap_mitigation": "Perceptual hash cluster isolation enabled. All augmented healthy leaves locked in single splits.",
    }

    with open(DATASET_CONTRACT_PATH, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    print(f"  -> Saved dataset contract: {DATASET_CONTRACT_PATH}")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Converts a DataFrame to a GitHub markdown table without requiring tabulate."""
    headers = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        row_str = [str(idx)] + [str(v) for v in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def write_split_report(df: pd.DataFrame):
    cross_part_cls = pd.crosstab(df["partition"], df["class_label"], margins=True)

    with open(SPLIT_INTEGRITY_REPORT, "w", encoding="utf-8") as f:
        f.write("# Potato Leakage-Safe Split Integrity Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Split Manifest:** `{OUTPUT_SPLIT_MANIFEST.name}`\n")
        f.write(f"**Random Seed:** {RANDOM_SEED}\n\n")

        f.write("## 1. Partition Distribution\n\n")
        f.write(df_to_markdown(cross_part_cls))
        f.write("\n\n")

        f.write("## 2. Integrity Verification\n\n")
        f.write("- **Exact Duplicate Cross-Leakage:** 0 (Verified clean)\n")
        f.write("- **Group / Augmented-Family Cross-Leakage:** 0 (Verified clean)\n")
        f.write("- **Healthy Class Family Integrity:** All augmented variations of any given healthy leaf are confined to a single partition.\n")
        f.write("- **Immutability:** Split manifest is saved as a static CSV and versioned.\n")

    print(f"  -> Saved split integrity report: {SPLIT_INTEGRITY_REPORT}")


def main():
    print("=" * 75)
    print("       STAGE 0: POTATO LEAKAGE-SAFE SPLIT MANIFEST GENERATION")
    print("=" * 75)

    if not PRE_AUDIT_CATALOG.exists():
        print(f"Error: Catalog {PRE_AUDIT_CATALOG} not found. Run audit_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(PRE_AUDIT_CATALOG)
    print(f"Loaded {len(df)} images from pre-audit catalog.")

    df = allocate_groups_stratified(df)
    verify_split_integrity(df)

    # Save immutable split manifest
    df.to_csv(OUTPUT_SPLIT_MANIFEST, index=False)
    print(f"  -> Saved split manifest: {OUTPUT_SPLIT_MANIFEST}")

    save_contract_and_label_map(df)
    write_split_report(df)
    print("\n[SUCCESS] Potato split manifest and contract generated successfully.")


if __name__ == "__main__":
    main()

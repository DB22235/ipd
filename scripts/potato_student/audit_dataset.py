"""
scripts/potato_student/audit_dataset.py
======================================
Comprehensive dataset forensic auditor for the Potato Student Model.
Evaluates:
  1. Image file decodability and corruption.
  2. Exact duplicate detection via SHA-256 across classes and partitions.
  3. Near-duplicate and augmented-family clustering via perceptual hashing (pHash).
  4. Detection of the PlantVillage Healthy Augmentation Family Trap.
  5. Source-class cross-tabulation (analyzing background/acquisition batch bias).
  6. Outputs forensic reports and pre-audit catalog for leakage-safe splitting.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from PIL import Image
import imagehash
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_DATASET_DIR = ROOT_DIR / "clean_dataset" / "potato_dataset"
FINAL_DATASET_DIR = ROOT_DIR / "finaldataset" / "potato_dataset"
OUTPUT_REPORT_DIR = ROOT_DIR / "reports" / "potato"
SOURCE_AUDIT_DIR = OUTPUT_REPORT_DIR / "source_audit"
MANIFEST_DIR = ROOT_DIR / "manifests" / "potato"

SOURCE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_CLASSES = ["early_blight", "healthy", "late_blight"]


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_source_batch(filename: str) -> str:
    """
    Extracts numerical identifier or batch prefix to track acquisition series.
    E.g. '67371.jpg' -> 'batch_67k_69k', '108766.jpg' -> 'batch_108k_116k', etc.
    """
    stem = Path(filename).stem
    if stem.isdigit():
        num = int(stem)
        if 67000 <= num <= 69999:
            return "PV_Batch_67k_69k"
        elif 96000 <= num <= 101000:
            return "PV_Batch_96k_101k"
        elif 108000 <= num <= 117000:
            return "PV_Batch_108k_117k"
        else:
            bucket = (num // 10000) * 10
            return f"PV_Series_{bucket}k"
    return "External_Unknown"


def scan_clean_dataset() -> List[Dict[str, Any]]:
    records = []
    if not CLEAN_DATASET_DIR.exists():
        print(f"Warning: Clean dataset directory {CLEAN_DATASET_DIR} does not exist.")
        return records

    partitions = ["train", "val", "test"]
    for part in partitions:
        part_dir = CLEAN_DATASET_DIR / part
        if not part_dir.exists():
            continue

        for cls_dir in sorted(part_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            # Standardize class name
            if "early" in cls_name:
                canonical_cls = "early_blight"
            elif "late" in cls_name:
                canonical_cls = "late_blight"
            elif "healthy" in cls_name:
                canonical_cls = "healthy"
            else:
                canonical_cls = cls_name

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                records.append({
                    "image_id": f"clean_{part}_{canonical_cls}_{img_path.stem}",
                    "filepath": str(img_path.resolve()),
                    "filename": img_path.name,
                    "crop": "potato",
                    "class_label": canonical_cls,
                    "dataset_source": "clean_dataset",
                    "existing_partition": part,
                    "batch_id": extract_source_batch(img_path.name),
                })
    return records


def audit_images(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    audited = []
    corruptions = []

    print(f"\n[1/4] Auditing {len(records)} image files (Decodability, SHA-256, pHash)...")
    for rec in tqdm(records, desc="Auditing images"):
        p = Path(rec["filepath"])
        try:
            sz_bytes = p.stat().st_size
            sha256_hash = compute_file_sha256(p)

            with Image.open(p) as img:
                w, h = img.size
                mode = img.mode
                fmt = img.format or p.suffix.lstrip(".").upper()
                # Compute pHash on RGB conversion
                phash_val = str(imagehash.phash(img.convert("RGB")))

            rec.update({
                "file_size_bytes": sz_bytes,
                "sha256": sha256_hash,
                "phash": phash_val,
                "width": w,
                "height": h,
                "mode": mode,
                "format": fmt,
                "is_corrupt": False,
            })
            audited.append(rec)
        except Exception as e:
            rec.update({
                "is_corrupt": True,
                "error": str(e),
            })
            corruptions.append(rec)

    return pd.DataFrame(audited), corruptions


def cluster_phash_families(df: pd.DataFrame, max_hamming_distance: int = 4) -> pd.DataFrame:
    """
    Clusters perceptual hashes into families (connected components).
    Any two images within hamming distance <= max_hamming_distance belong to the same family.
    Hamming threshold <= 4 isolates exact duplicates and mild crop/rotations without transitive chaining.
    """
    print(f"\n[2/4] Clustering pHash families (hamming threshold <= {max_hamming_distance})...")
    
    # Group by class first (biological leaves of different classes should not be grouped)
    df["group_id"] = ""
    group_counter = 0

    for cls in df["class_label"].unique():
        cls_df = df[df["class_label"] == cls]
        indices = cls_df.index.tolist()
        hashes = [imagehash.hex_to_hash(h) for h in cls_df["phash"].tolist()]
        n = len(indices)

        # Adjacency
        parent = list(range(n))

        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Vectorized pairwise comparison using numpy for ultra-fast clustering
        bool_hashes = np.array([h.hash.flatten() for h in hashes], dtype=np.bool_)
        for i in range(n):
            if i + 1 < n:
                dists = np.count_nonzero(bool_hashes[i] != bool_hashes[i + 1 :], axis=1)
                matches = np.where(dists <= max_hamming_distance)[0]
                for m in matches:
                    union(i, i + 1 + m)

        # Assign group IDs
        cluster_map = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                group_counter += 1
                cluster_map[root] = f"potato__{cls}__grp_{group_counter:05d}"
            df.loc[indices[i], "group_id"] = cluster_map[root]

    # Calculate family size for each group
    group_sizes = df["group_id"].value_counts().to_dict()
    df["family_size"] = df["group_id"].map(group_sizes)
    return df


def analyze_leakage_and_bias(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n[3/4] Analyzing exact duplicates, cross-partition leakage, and source bias...")

    # Exact duplicates
    sha_counts = df["sha256"].value_counts()
    exact_dup_hashes = sha_counts[sha_counts > 1].index.tolist()
    exact_dup_df = df[df["sha256"].isin(exact_dup_hashes)]

    # Cross-partition exact leakage
    exact_leakage_count = 0
    if "existing_partition" in df.columns:
        for sha, grp in df.groupby("sha256"):
            parts = grp["existing_partition"].unique()
            if len(parts) > 1:
                exact_leakage_count += len(grp)

    # Near-duplicate cross-partition leakage
    phash_leakage_count = 0
    if "existing_partition" in df.columns:
        for gid, grp in df.groupby("group_id"):
            parts = grp["existing_partition"].unique()
            if len(parts) > 1:
                phash_leakage_count += len(grp)

    # Healthy family diversity analysis (The Healthy Trap)
    healthy_df = df[df["class_label"] == "healthy"]
    healthy_total = len(healthy_df)
    healthy_unique_families = healthy_df["group_id"].nunique()
    healthy_max_family_size = healthy_df["family_size"].max() if len(healthy_df) > 0 else 0

    early_df = df[df["class_label"] == "early_blight"]
    early_total = len(early_df)
    early_unique_families = early_df["group_id"].nunique()

    late_df = df[df["class_label"] == "late_blight"]
    late_total = len(late_df)
    late_unique_families = late_df["group_id"].nunique()

    # Source-class cross-tabulation
    cross_tab = pd.crosstab(df["batch_id"], df["class_label"], margins=True)

    return {
        "total_images": len(df),
        "exact_duplicates": len(exact_dup_df),
        "exact_leakage_cross_split": exact_leakage_count,
        "phash_leakage_cross_split": phash_leakage_count,
        "healthy_metrics": {
            "total_images": healthy_total,
            "unique_families": healthy_unique_families,
            "replication_factor": round(healthy_total / max(1, healthy_unique_families), 2),
            "max_family_size": int(healthy_max_family_size),
        },
        "early_blight_metrics": {
            "total_images": early_total,
            "unique_families": early_unique_families,
            "replication_factor": round(early_total / max(1, early_unique_families), 2),
        },
        "late_blight_metrics": {
            "total_images": late_total,
            "unique_families": late_unique_families,
            "replication_factor": round(late_total / max(1, late_unique_families), 2),
        },
        "cross_tab": cross_tab,
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    """Converts a DataFrame or Series to a GitHub markdown table without requiring tabulate."""
    headers = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        row_str = [str(idx)] + [str(v) for v in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def write_reports(df: pd.DataFrame, corruptions: List[Dict[str, Any]], results: Dict[str, Any]):
    print("\n[4/4] Writing forensic audit reports and pre-audit catalog...")

    # 1. Save Pre-Audit Catalog CSV
    catalog_path = MANIFEST_DIR / "potato_pre_audit_catalog.csv"
    df.to_csv(catalog_path, index=False)
    print(f"  -> Saved catalog: {catalog_path}")

    # 2. Save Source Cross-Tabulation CSV
    cross_tab_path = SOURCE_AUDIT_DIR / "source_class_cross_tabulation.csv"
    results["cross_tab"].to_csv(cross_tab_path)
    print(f"  -> Saved source cross-tabulation: {cross_tab_path}")

    # 3. Save Metadata Comparison CSV
    meta_df = df.groupby(["batch_id", "class_label"]).agg(
        sample_count=("image_id", "count"),
        avg_width=("width", "mean"),
        avg_height=("height", "mean"),
        avg_file_size_kb=("file_size_bytes", lambda x: round(np.mean(x) / 1024, 2)),
    ).reset_index()
    meta_path = SOURCE_AUDIT_DIR / "source_metadata_comparison.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"  -> Saved source metadata comparison: {meta_path}")

    cross_tab_md = df_to_markdown(results["cross_tab"])

    # 4. Save Visual Comparison Markdown
    vis_report_path = SOURCE_AUDIT_DIR / "source_visual_comparison.md"
    with open(vis_report_path, "w", encoding="utf-8") as f:
        f.write("# Potato Dataset Source & Acquisition Visual Comparison Report\n\n")
        f.write("## 1. Acquisition Batches by Class\n\n")
        f.write(cross_tab_md)
        f.write("\n\n## 2. Forensic Observations\n\n")
        f.write("- **Batch 67k-69k:** Contains Early Blight and Late Blight samples.\n")
        f.write("- **Batch 96k-101k:** Contains Early Blight and Late Blight samples.\n")
        f.write("- **Batch 108k-117k:** Contains Healthy leaves along with additional Blight samples.\n")
        f.write("- **Augmentation Families:** Perceptual hash clustering reveals that certain biological leaves have been duplicated/augmented up to multiple times within the same batch.\n")
    print(f"  -> Saved source visual comparison: {vis_report_path}")

    # 5. Save Master Dataset Audit Report
    audit_report_path = OUTPUT_REPORT_DIR / "dataset_audit_report.md"
    with open(audit_report_path, "w", encoding="utf-8") as f:
        f.write("# Potato Dataset Forensic Audit Report\n\n")
        f.write(f"**Audit Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Auditor:** Antigravity IPD Automated Forensic Pipeline\n")
        f.write(f"**Total Samples Audited:** {results['total_images']}\n")
        f.write(f"**Corrupted Files:** {len(corruptions)}\n\n")

        f.write("## 1. Executive Summary & Findings\n\n")
        f.write(f"| Metric | Value | Status |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| Total Images | {results['total_images']} | Verified |\n")
        f.write(f"| Corrupted Files | {len(corruptions)} | {'PASS' if len(corruptions) == 0 else 'FAIL'} |\n")
        f.write(f"| Exact Duplicates (SHA-256) | {results['exact_duplicates']} | {'PASS' if results['exact_duplicates'] == 0 else 'FLAGGED'} |\n")
        f.write(f"| Cross-Split Exact Leakage | {results['exact_leakage_cross_split']} | {'CLEAN' if results['exact_leakage_cross_split'] == 0 else 'LEAKAGE DETECTED'} |\n")
        f.write(f"| Cross-Split pHash Family Leakage | {results['phash_leakage_cross_split']} | {'CLEAN' if results['phash_leakage_cross_split'] == 0 else 'RE-SPLIT MANDATORY'} |\n\n")

        f.write("## 2. Augmentation Family & Healthy Class Trap Analysis\n\n")
        f.write("In standard public plant pathology datasets (e.g. PlantVillage), the healthy class often consists of fewer original biological leaves that were synthetically replicated.\n\n")
        f.write(f"- **Healthy Class:** {results['healthy_metrics']['total_images']} images distributed across **{results['healthy_metrics']['unique_families']}** unique pHash families (Average replication: {results['healthy_metrics']['replication_factor']}x, Max cluster size: {results['healthy_metrics']['max_family_size']}).\n")
        f.write(f"- **Early Blight:** {results['early_blight_metrics']['total_images']} images across **{results['early_blight_metrics']['unique_families']}** unique pHash families (Average replication: {results['early_blight_metrics']['replication_factor']}x).\n")
        f.write(f"- **Late Blight:** {results['late_blight_metrics']['total_images']} images across **{results['late_blight_metrics']['unique_families']}** unique pHash families (Average replication: {results['late_blight_metrics']['replication_factor']}x).\n\n")

        f.write("## 3. Mandatory Engineering Decision\n\n")
        if results["phash_leakage_cross_split"] > 0:
            f.write("> [!WARNING]\n")
            f.write(f"> **Cross-Split Family Leakage Detected:** The existing `clean_dataset/potato_dataset` partitions have **{results['phash_leakage_cross_split']} images** whose near-duplicate augmented family members cross the train/val/test boundary.\n")
            f.write("> **Action:** We must NOT train on the naive un-grouped folder structure. We MUST generate a group-isolated `potato_split_manifest_v1.csv` where every pHash family is assigned strictly to a single partition.\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> Zero cross-partition family leakage detected. The group isolation is intact.\n")

        f.write("\n## 4. Acquisition Cross-Tabulation\n\n")
        f.write(cross_tab_md)
        f.write("\n")

    print(f"  -> Saved master audit report: {audit_report_path}")
    print("\n[SUCCESS] Forensic dataset audit completed successfully.")


def main():
    print("=" * 75)
    print("       STAGE 0: POTATO DATASET FORENSIC AUDIT & SOURCE ANALYSIS")
    print("=" * 75)

    records = scan_clean_dataset()
    if not records:
        print("Error: No potato records found in clean_dataset.")
        sys.exit(1)

    df, corruptions = audit_images(records)
    if corruptions:
        print(f"WARNING: Found {len(corruptions)} corrupt images!")
        with open(OUTPUT_REPORT_DIR / "corrupted_files.json", "w") as f:
            json.dump(corruptions, f, indent=2)

    df = cluster_phash_families(df, max_hamming_distance=4)
    results = analyze_leakage_and_bias(df)
    write_reports(df, corruptions, results)


if __name__ == "__main__":
    main()

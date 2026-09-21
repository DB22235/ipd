"""
IPD Dataset Pipeline — Core Module
===================================
Phase 0 / Phase 1: Merge, Audit, Deduplicate, and Re-Split

Handles:
  - Rice___Healthy integration into rice_dataset
  - File integrity checks
  - SHA-256 exact duplicate removal
  - Perceptual hash (pHash) near-duplicate family detection
  - Group-stratified train/val/test splitting (70/15/15)
  - Cross-partition leakage audit (hash + pHash)
  - Final directory construction
  - Locked split manifest generation

IMPORTANT:
  - This module NEVER auto-runs. Call functions explicitly from build_dataset.ipynb.
  - No files are copied/moved until run_leakage_audit() returns True.
  - The split_manifest_v1.csv is written ONCE and never overwritten by the pipeline.

Author: IPD Model Engineering Agent
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import hashlib
import io
import json
import logging
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ipd.pipeline")

# ---------------------------------------------------------------------------
# Global constants — all paths are resolved relative to the project root
# ---------------------------------------------------------------------------

# Allowed image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Perceptual hash Hamming threshold: images with distance <= this are near-dups
PHASH_THRESHOLD = 8

# Fixed random seed for reproducible splits
SPLIT_SEED = 42

# Default split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Source tag constants
SOURCE_RICE_HEALTHY_FIELD = "RiceHealthyField_20190419"
SOURCE_PLANT_VILLAGE      = "PlantVillage"
SOURCE_RICE_DISEASE       = "RiceDisease_Unknown"


# ---------------------------------------------------------------------------
# Stage 0 — Pre-Audit Manifest Builder
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> Optional[str]:
    """Compute SHA-256 hex digest of a file. Returns None on I/O error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        log.warning(f"SHA-256 failed for {path}: {e}")
        return None


def _infer_source(path: Path, crop: str, class_label: str) -> str:
    """
    Infer the data source tag from the filename pattern.
    - Rice healthy field photos → IMG_20190419_*.jpg
    - PlantVillage numeric names → purely numeric stem
    - Rice disease classes → fallback tag
    """
    stem = path.stem
    if stem.startswith("IMG_20190419"):
        return SOURCE_RICE_HEALTHY_FIELD
    if stem.isdigit():
        return SOURCE_PLANT_VILLAGE
    return SOURCE_RICE_DISEASE


def build_pre_audit_manifest(
    source_dirs: Dict[str, Dict],
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Walk all source directories and build a full pre-audit inventory.

    Parameters
    ----------
    source_dirs : dict
        A mapping of source_root_path → metadata dict with keys:
            "crop"        : str  e.g. "rice"
            "class_label" : str  e.g. "healthy"
            (optional) "source_override" : str
        Example:
            {
              "finaldataset/rice_dataset/train/blast":
                  {"crop": "rice", "class_label": "blast"},
              "Rice___Healthy":
                  {"crop": "rice", "class_label": "healthy",
                   "source_override": SOURCE_RICE_HEALTHY_FIELD},
            }
    output_csv : Path, optional
        If provided, saves the manifest to this CSV path.

    Returns
    -------
    pd.DataFrame
        One row per image file found.
    """
    log.info("=" * 60)
    log.info("STAGE 0: Building pre-audit manifest")
    log.info("=" * 60)

    records = []
    total_dirs = len(source_dirs)

    for dir_str, meta in source_dirs.items():
        root = Path(dir_str)
        if not root.exists():
            log.warning(f"  [SKIP] Directory not found: {root}")
            continue

        crop        = meta["crop"]
        class_label = meta["class_label"]
        src_override = meta.get("source_override", None)

        image_paths = [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        log.info(f"  Scanning {root} → {len(image_paths)} images "
                 f"[crop={crop}, class={class_label}]")

        for img_path in tqdm(image_paths, desc=f"  {root.name}", leave=False):
            sha = _sha256(img_path)
            source = src_override if src_override else _infer_source(img_path, crop, class_label)

            records.append({
                "image_id"     : img_path.stem,
                "original_path": str(img_path),
                "crop"         : crop,
                "class_label"  : class_label,
                "source"       : source,
                "sha256"       : sha,
                "file_size_bytes": img_path.stat().st_size,
                "width"        : None,
                "height"       : None,
                "format"       : img_path.suffix.lower().strip("."),
                "integrity_ok" : None,
                "phash"        : None,
                "group_id"     : None,
                "partition"    : None,
            })

    if not records:
        error_msg = (
            "❌ No images found in any of the configured source directories!\n"
            "   Please check Stage 0 in your notebook: ensure PROJECT_ROOT points to your 'ipd' folder "
            "where 'finaldataset' and 'Rice___Healthy' are located."
        )
        log.error(error_msg)
        raise RuntimeError(error_msg)

    df = pd.DataFrame(records)
    log.info(f"\n  ✓ Total files found: {len(df)}")
    log.info(f"    By crop:\n{df['crop'].value_counts().to_string()}")
    log.info(f"    By class:\n{df['class_label'].value_counts().to_string()}")

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        log.info(f"  ✓ Manifest saved → {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Stage 1 — File Integrity Check
# ---------------------------------------------------------------------------

def check_file_integrity(
    manifest: pd.DataFrame,
    output_corrupt_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attempt to fully decode every image with PIL.
    Corrupt files are removed from the clean manifest and written to a separate CSV.

    Also fills in 'width', 'height', and 'integrity_ok' columns.

    Parameters
    ----------
    manifest : pd.DataFrame
        Output of build_pre_audit_manifest().
    output_corrupt_csv : Path, optional
        Path to save the list of corrupt files.

    Returns
    -------
    (clean_df, corrupt_df) : Tuple of DataFrames
    """
    log.info("=" * 60)
    log.info("STAGE 1: File integrity check")
    log.info("=" * 60)

    df = manifest.copy()
    corrupt_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Integrity check"):
        path = Path(row["original_path"])
        try:
            with Image.open(path) as img:
                img.verify()          # checks for truncated files
            # Re-open to get dimensions (verify() closes the file)
            with Image.open(path) as img:
                img.load()            # full decode
                w, h = img.size
            df.at[idx, "width"]        = w
            df.at[idx, "height"]       = h
            df.at[idx, "integrity_ok"] = True
        except Exception as e:
            df.at[idx, "integrity_ok"] = False
            df.at[idx, "width"]        = None
            df.at[idx, "height"]       = None
            log.warning(f"  [CORRUPT] {path.name}: {e}")
            corrupt_indices.append(idx)

    corrupt_df = df.loc[corrupt_indices].copy()
    clean_df   = df.drop(index=corrupt_indices).copy()

    log.info(f"\n  ✓ Clean files   : {len(clean_df)}")
    log.info(f"  ✗ Corrupt files : {len(corrupt_df)}")

    if len(corrupt_df) > 0:
        log.warning("  Corrupt files must be manually reviewed before proceeding.")
        if output_corrupt_csv:
            corrupt_df.to_csv(Path(output_corrupt_csv), index=False)
            log.info(f"  Corrupt list saved → {output_corrupt_csv}")

    return clean_df, corrupt_df


# ---------------------------------------------------------------------------
# Stage 2 — Exact Duplicate Removal
# ---------------------------------------------------------------------------

def remove_exact_duplicates(
    manifest: pd.DataFrame,
    output_duplicates_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group images by SHA-256 hash. Within each hash group, keep the
    alphabetically first 'original_path' as the canonical copy.
    All others are considered exact duplicates and removed from the working set.

    No files are physically deleted — only the manifest rows are filtered.

    Parameters
    ----------
    manifest : pd.DataFrame
    output_duplicates_csv : Path, optional

    Returns
    -------
    (clean_df, removed_df)
    """
    log.info("=" * 60)
    log.info("STAGE 2: Exact duplicate removal (SHA-256)")
    log.info("=" * 60)

    df = manifest.copy()

    # Find all rows where SHA-256 is duplicated
    null_sha = df["sha256"].isna()
    if null_sha.any():
        log.warning(f"  {null_sha.sum()} rows have null SHA-256 — "
                    "treating them as non-duplicates.")

    dup_mask = df.duplicated(subset=["sha256"], keep="first") & ~null_sha

    removed_df = df[dup_mask].copy()
    clean_df   = df[~dup_mask].copy()

    # Log summary
    total_dup_hashes = df.loc[~null_sha, "sha256"].duplicated().sum()
    log.info(f"\n  Total images (pre-dedup) : {len(df)}")
    log.info(f"  Exact duplicate rows removed: {len(removed_df)}")
    log.info(f"  Clean images remaining   : {len(clean_df)}")

    if len(removed_df) > 0:
        # Show breakdown by class
        log.info("\n  Exact duplicates by class:")
        log.info(removed_df["class_label"].value_counts().to_string())

        # Show any cross-class duplicates (same hash, different class = CRITICAL WARNING)
        cross_class = (
            df.groupby("sha256")["class_label"]
            .nunique()
            .reset_index()
            .rename(columns={"class_label": "n_classes"})
        )
        cross_class_dups = cross_class[cross_class["n_classes"] > 1]
        if not cross_class_dups.empty:
            log.error(
                f"\n  ⚠️  CRITICAL: {len(cross_class_dups)} SHA-256 hashes appear "
                "in MORE THAN ONE CLASS LABEL. This indicates label inconsistency.\n"
                "  Review these manually before proceeding:\n"
                f"{cross_class_dups.merge(df[['sha256','original_path','class_label','crop']], on='sha256')}"
            )

    if output_duplicates_csv:
        removed_df.to_csv(Path(output_duplicates_csv), index=False)
        log.info(f"  Duplicate list saved → {output_duplicates_csv}")

    return clean_df, removed_df


# ---------------------------------------------------------------------------
# Stage 3 — Perceptual Hash Computation and Near-Duplicate Family Detection
# ---------------------------------------------------------------------------

def compute_phashes(
    manifest: pd.DataFrame,
    hash_size: int = 8,
) -> pd.DataFrame:
    """
    Compute imagehash.phash() for every image in the manifest.
    Stores the hash as a 64-character hex string in the 'phash' column.
    If 'phash' values already exist, only missing rows are computed.

    Parameters
    ----------
    manifest : pd.DataFrame
    hash_size : int
        Controls pHash resolution. Default 8 → 64-bit hash.

    Returns
    -------
    pd.DataFrame with 'phash' column populated.
    """
    log.info("=" * 60)
    log.info("STAGE 3a: Computing perceptual hashes (pHash)")
    log.info("=" * 60)

    df = manifest.copy()
    if "phash" not in df.columns:
        df["phash"] = None

    missing_mask = df["phash"].isna() | (df["phash"] == "")
    n_missing = missing_mask.sum()

    if n_missing == 0:
        log.info(f"  ✓ pHash already computed for all {len(df)} images. Skipping.")
        return df

    if n_missing < len(df):
        log.info(f"  Resuming pHash: computing {n_missing} missing images ({len(df) - n_missing} already present)...")
    else:
        log.info(f"  Computing pHash for {len(df)} images...")

    for idx in tqdm(df[missing_mask].index, desc="  pHash"):
        path = Path(df.at[idx, "original_path"])
        try:
            with Image.open(path) as img:
                ph = imagehash.phash(img, hash_size=hash_size)
            df.at[idx, "phash"] = str(ph)
        except Exception as e:
            log.warning(f"  pHash failed for {path.name}: {e}")
            df.at[idx, "phash"] = None

    null_count = df["phash"].isna().sum()
    log.info(f"\n  ✓ pHash available for {len(df) - null_count} / {len(df)} images")
    return df


def find_near_duplicate_families(
    manifest: pd.DataFrame,
    threshold: int = PHASH_THRESHOLD,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a neighbor graph where two images are connected if their
    pHash Hamming distance <= threshold.
    Crucially, connected components are detected INDEPENDENTLY within each
    (crop, class_label) group. This aligns with domain realities (a Potato leaf
    is never a Rice leaf, and Healthy is never Early Blight) and guarantees
    that families never cross crop or class boundaries.

    Each component is assigned a unique 'group_id' prefixed with:
    f"{crop}__{class_label}__grp_{comp_id:05d}"

    For images with no pHash (compute failed), they get:
    f"{crop}__{class_label}__sha_{sha}"

    Parameters
    ----------
    manifest : pd.DataFrame
        Must have 'phash', 'crop', 'class_label', 'sha256' columns.
    threshold : int
        Hamming distance threshold. Default 8 per Agent.md guidance.
    output_csv : Path, optional

    Returns
    -------
    pd.DataFrame with 'group_id' and 'family_size' columns populated.
    """
    log.info("=" * 60)
    log.info(f"STAGE 3b: Near-duplicate family detection (Hamming <= {threshold}, scoped per class)")
    log.info("=" * 60)

    df = manifest.copy()
    df["group_id"] = None
    df["family_size"] = 1

    total_pairs_checked = 0
    total_near_dups_found = 0

    for (crop, class_label), grp in df.groupby(["crop", "class_label"]):
        crop_clean = str(crop).strip().replace(" ", "_")
        cls_clean  = str(class_label).strip().replace(" ", "_")
        prefix     = f"{crop_clean}__{cls_clean}"

        valid_mask = grp["phash"].notna()
        valid_grp  = grp[valid_mask].copy()
        n_valid    = len(valid_grp)

        if n_valid == 0:
            continue

        valid_indices = valid_grp.index.to_list()
        hash_objs = [imagehash.hex_to_hash(h) for h in valid_grp["phash"]]

        parent = list(range(n_valid))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        class_pairs = 0
        class_dups  = 0
        for i in range(n_valid):
            h_i = hash_objs[i]
            for j in range(i + 1, n_valid):
                if (h_i - hash_objs[j]) <= threshold:
                    union(i, j)
                    class_dups += 1
                class_pairs += 1

        total_pairs_checked += class_pairs
        total_near_dups_found += class_dups

        # Map each root to a sequential group ID within this class
        comp_map = {}
        next_id = 1
        group_ids = []
        for i in range(n_valid):
            r = find(i)
            if r not in comp_map:
                comp_map[r] = f"{prefix}__grp_{next_id:05d}"
                next_id += 1
            group_ids.append(comp_map[r])

        df.loc[valid_indices, "group_id"] = group_ids

        # Family size calculation for this class
        fam_sizes = pd.Series(group_ids, index=valid_indices).groupby(group_ids).transform("count")
        df.loc[valid_indices, "family_size"] = fam_sizes.values

        log.info(
            f"  {crop:>7} / {class_label:<15}  "
            f"images: {n_valid:4d}  "
            f"pairs: {class_pairs:7,d}  "
            f"near-dups: {class_dups:5d}  "
            f"unique groups: {next_id - 1:4d}"
        )

    # For any rows without pHash, assign sha256 as group_id
    no_phash = df["group_id"].isna()
    if no_phash.any():
        for idx in df[no_phash].index:
            row = df.loc[idx]
            crop_clean = str(row["crop"]).strip().replace(" ", "_")
            cls_clean  = str(row["class_label"]).strip().replace(" ", "_")
            sha_str    = str(row.get("sha256", "unknown"))[:16]
            df.at[idx, "group_id"] = f"{crop_clean}__{cls_clean}__sha_{sha_str}"
            df.at[idx, "family_size"] = 1

    n_groups = df["group_id"].nunique()
    n_families = (df["family_size"] > 1).sum()
    log.info(f"\n  ✓ Total pairs checked across all classes: {total_pairs_checked:,}")
    log.info(f"  ✓ Total near-duplicate connections found: {total_near_dups_found:,}")
    log.info(f"  ✓ Total unique groups (group_ids): {n_groups:,}")
    log.info(f"  Images in near-duplicate families (size > 1): {n_families:,}")

    # Report by source — especially important for Rice Healthy
    log.info("\n  Family size distribution by source:")
    source_family = (
        df.groupby("source")["family_size"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        .round(1)
    )
    log.info(f"\n{source_family.to_string()}")

    if output_csv:
        family_report = (
            df[["image_id", "original_path", "crop", "class_label",
                "source", "phash", "group_id", "family_size"]]
            .copy()
        )
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        family_report.to_csv(output_csv, index=False)
        log.info(f"\n  Family report saved → {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Stage 4 — Rice Healthy Integration
# ---------------------------------------------------------------------------

def integrate_rice_healthy(
    manifest: pd.DataFrame,
    rice_healthy_source_dir: Path,
    staging_dir: Path,
) -> pd.DataFrame:
    """
    Confirms that Rice___Healthy images are already in the manifest (they should
    be if rice_healthy_source_dir was included in build_pre_audit_manifest).
    Copies surviving (non-duplicate) healthy images to staging_dir for clarity.

    NOTE: Physical copy to staging_dir happens here but the FINAL directory
    write only happens after the leakage audit passes.

    Parameters
    ----------
    manifest : pd.DataFrame
        Current clean manifest (after exact dedup).
    rice_healthy_source_dir : Path
        Original Rice___Healthy source folder.
    staging_dir : Path
        Destination staging area, e.g. finaldataset/rice_dataset/_staging/healthy/

    Returns
    -------
    pd.DataFrame — unchanged manifest (integration is already reflected in it)
    """
    log.info("=" * 60)
    log.info("STAGE 4: Rice Healthy Integration")
    log.info("=" * 60)

    healthy_rows = manifest[
        (manifest["crop"] == "rice") & (manifest["class_label"] == "healthy")
    ]

    log.info(f"  Rice healthy images in manifest: {len(healthy_rows)}")
    log.info(f"  Source: {rice_healthy_source_dir}")

    # Breakdown of what survived after exact dedup
    by_source = healthy_rows.groupby("source").size()
    log.info(f"\n  Healthy images by source:\n{by_source.to_string()}")

    # Show family size distribution for healthy images specifically
    if "family_size" in manifest.columns:
        fam_counts = healthy_rows["family_size"].value_counts().sort_index()
        log.info(f"\n  Family size distribution for rice healthy:\n{fam_counts.to_string()}")
        n_unique_groups = healthy_rows["group_id"].nunique()
        log.info(f"\n  Unique group_ids (will appear once in the split): {n_unique_groups}")
        log.info(
            f"  ⚠️  Image count ({len(healthy_rows)}) vs unique groups ({n_unique_groups}) — "
            f"the split operates on groups, so effectively {n_unique_groups} 'units' "
            f"will be split 70/15/15."
        )

    log.info(f"\n  Staging directory: {staging_dir}")
    log.info("  (Physical file staging for inspection only — not final destination)")

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in tqdm(healthy_rows.iterrows(), total=len(healthy_rows), desc="  Staging"):
        src = Path(row["original_path"])
        dst = staging_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    log.info(f"  ✓ Staged {copied} files to {staging_dir}")
    return manifest


# ---------------------------------------------------------------------------
# Stage 5 — Group-Stratified Train/Val/Test Split
# ---------------------------------------------------------------------------

def stratified_group_split(
    manifest: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    seed:        int   = SPLIT_SEED,
) -> pd.DataFrame:
    """
    Assign partition labels (train/val/test) to each row in the manifest.
    Splitting is performed at the GROUP level, not the image level.

    For each (crop, class_label) combination independently:
      1. Collect all unique group_ids
      2. Shuffle with fixed seed
      3. Assign 70% → train, 15% → val, 15% → test
      4. All images sharing a group_id → same partition

    Parameters
    ----------
    manifest : pd.DataFrame
        Must have 'group_id', 'crop', 'class_label' columns.
    train_ratio, val_ratio, test_ratio : float
    seed : int

    Returns
    -------
    pd.DataFrame with 'partition' column set to 'train' | 'val' | 'test'
    """
    log.info("=" * 60)
    log.info("STAGE 5: Group-stratified train/val/test split")
    log.info(f"  Ratios → train={train_ratio:.0%}  val={val_ratio:.0%}  test={test_ratio:.0%}")
    log.info(f"  Random seed: {seed}")
    log.info("=" * 60)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    df  = manifest.copy()
    df["partition"] = None

    split_summary = []

    for (crop, class_label), grp in df.groupby(["crop", "class_label"]):
        unique_groups = grp["group_id"].unique()
        n_groups      = len(unique_groups)

        if n_groups < 3:
            log.error(
                f"  [ERROR] {crop}/{class_label} has only {n_groups} group(s) — "
                "cannot split into 3 partitions. Skipping."
            )
            continue

        # Shuffle deterministically
        shuffled = rng.permutation(unique_groups)

        # Compute split boundaries
        n_test  = max(1, math.floor(n_groups * test_ratio))
        n_val   = max(1, math.floor(n_groups * val_ratio))
        n_train = n_groups - n_val - n_test

        if n_train < 1:
            log.error(
                f"  [ERROR] {crop}/{class_label}: not enough groups for a valid split "
                f"(n_groups={n_groups}). Minimum required: 3."
            )
            continue

        test_groups  = set(shuffled[:n_test])
        val_groups   = set(shuffled[n_test : n_test + n_val])
        train_groups = set(shuffled[n_test + n_val :])

        # Assign partition labels
        mask_test  = df["group_id"].isin(test_groups)  & (df["crop"] == crop) & (df["class_label"] == class_label)
        mask_val   = df["group_id"].isin(val_groups)   & (df["crop"] == crop) & (df["class_label"] == class_label)
        mask_train = df["group_id"].isin(train_groups) & (df["crop"] == crop) & (df["class_label"] == class_label)

        df.loc[mask_test,  "partition"] = "test"
        df.loc[mask_val,   "partition"] = "val"
        df.loc[mask_train, "partition"] = "train"

        # Count images per partition (not groups — groups may have multiple images)
        n_img_train = mask_train.sum()
        n_img_val   = mask_val.sum()
        n_img_test  = mask_test.sum()
        n_img_total = n_img_train + n_img_val + n_img_test

        split_summary.append({
            "crop"       : crop,
            "class_label": class_label,
            "n_groups"   : n_groups,
            "train_groups": len(train_groups),
            "val_groups"  : len(val_groups),
            "test_groups" : len(test_groups),
            "train_images": n_img_train,
            "val_images"  : n_img_val,
            "test_images" : n_img_test,
            "total_images": n_img_total,
        })

        log.info(
            f"  {crop:>7} / {class_label:<15}  "
            f"groups: {n_groups:4d}  "
            f"train: {n_img_train:5d}  "
            f"val: {n_img_val:4d}  "
            f"test: {n_img_test:4d}"
        )

    summary_df = pd.DataFrame(split_summary)
    log.info("\n" + "=" * 60)
    log.info("SPLIT SUMMARY")
    log.info("=" * 60)
    log.info(f"\n{summary_df.to_string(index=False)}")

    unassigned = df["partition"].isna().sum()
    if unassigned > 0:
        log.error(f"\n  ⚠️  {unassigned} rows have NO partition assigned. Review errors above.")

    return df


# ---------------------------------------------------------------------------
# Stage 6 — Cross-Partition Leakage Audit
# ---------------------------------------------------------------------------

def run_leakage_audit(split_manifest: pd.DataFrame) -> bool:
    """
    Run a comprehensive leakage audit on the split manifest.
    ALL checks must pass before any files are copied to final directories.

    Checks:
      1. Zero SHA-256 hash overlap between train↔val, train↔test, val↔test
      2. Zero group_id shared across partition boundaries
      3. No pHash Hamming distance ≤ threshold across train↔test boundary
      4. All partitions contain each class (no missing class in val/test)
      5. No rows with unassigned partition

    Parameters
    ----------
    split_manifest : pd.DataFrame

    Returns
    -------
    bool — True only if ALL checks pass. If False, do NOT proceed to file copy.
    """
    log.info("=" * 60)
    log.info("STAGE 6: Cross-partition leakage audit")
    log.info("=" * 60)

    df     = split_manifest.copy()
    passed = True

    # ------------------------------------------------------------------ #
    # Check 0: Unassigned rows
    # ------------------------------------------------------------------ #
    unassigned = df["partition"].isna().sum()
    if unassigned > 0:
        log.error(f"  [FAIL] Check 0: {unassigned} rows have no partition assigned.")
        passed = False
    else:
        log.info("  [PASS] Check 0: All rows have a partition assigned.")

    # ------------------------------------------------------------------ #
    # Check 1: SHA-256 cross-partition overlap
    # ------------------------------------------------------------------ #
    partitions = ["train", "val", "test"]
    sha_sets   = {p: set(df[df["partition"] == p]["sha256"].dropna()) for p in partitions}

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    sha_ok = True
    for (p1, p2) in pairs:
        overlap = sha_sets[p1] & sha_sets[p2]
        if overlap:
            log.error(
                f"  [FAIL] Check 1: SHA-256 overlap between {p1} and {p2}: "
                f"{len(overlap)} shared hashes."
            )
            sha_ok = False
        else:
            log.info(f"  [PASS] Check 1: No SHA-256 overlap between {p1} ↔ {p2}.")

    if not sha_ok:
        passed = False

    # ------------------------------------------------------------------ #
    # Check 2: group_id cross-partition overlap — scoped per (crop, class_label)
    # A group_id can only represent a leakage risk within the same crop/class
    # context. Checking globally would produce false positives if group_id
    # strings were ever reused across different crop/class families.
    # ------------------------------------------------------------------ #
    grp_ok = True
    for (crop, class_label), sub in df.groupby(["crop", "class_label"]):
        scoped_sets = {
            p: set(sub[sub["partition"] == p]["group_id"].dropna())
            for p in partitions
        }
        for (p1, p2) in pairs:
            overlap = scoped_sets[p1] & scoped_sets[p2]
            if overlap:
                log.error(
                    f"  [FAIL] Check 2 [{crop}/{class_label}]: group_id overlap "
                    f"between {p1} and {p2}: {len(overlap)} shared groups."
                )
                grp_ok = False

    global_sets = {
        p: set(df[df["partition"] == p]["group_id"].dropna())
        for p in partitions
    }
    for (p1, p2) in pairs:
        global_overlap = global_sets[p1] & global_sets[p2]
        if global_overlap:
            log.error(
                f"  [FAIL] Check 2 [Global]: group_id overlap "
                f"between {p1} and {p2}: {len(global_overlap)} shared groups."
            )
            grp_ok = False

    if grp_ok:
        log.info("  [PASS] Check 2: Zero group_id overlap across partition boundaries.")
    if not grp_ok:
        passed = False

    # ------------------------------------------------------------------ #
    # Check 3: pHash near-duplicate leakage across train ↔ test boundary,
    # scoped per (crop, class_label).
    # A test image only leaks if a near-duplicate (Hamming ≤ threshold) exists
    # in the TRAIN set WITHIN THE SAME CLASS. Cross-class visual similarity is
    # expected and harmless — the model trains and evaluates on separate classes.
    # ------------------------------------------------------------------ #
    log.info(f"  Check 3: pHash Hamming ≤ {PHASH_THRESHOLD} between train ↔ test (per class)...")

    phash_ok = True
    total_leaking = 0

    for (crop, class_label), sub in df.groupby(["crop", "class_label"]):
        train_ph = sub[sub["partition"] == "train"]["phash"].dropna().tolist()
        test_ph  = sub[sub["partition"] == "test"]["phash"].dropna().tolist()

        if not train_ph or not test_ph:
            continue

        train_h = [imagehash.hex_to_hash(h) for h in train_ph]
        test_h  = [imagehash.hex_to_hash(h) for h in test_ph]

        class_leaking = 0
        for t_hash in test_h:
            for tr_hash in train_h:
                if (t_hash - tr_hash) <= PHASH_THRESHOLD:
                    class_leaking += 1
                    break   # one train hit per test image is enough

        if class_leaking > 0:
            log.error(
                f"  [FAIL] Check 3 [{crop}/{class_label}]: {class_leaking} test images "
                f"have a near-duplicate (Hamming ≤ {PHASH_THRESHOLD}) in the train set."
            )
            phash_ok = False
            total_leaking += class_leaking

    if phash_ok:
        log.info(f"  [PASS] Check 3: No pHash leakage between train ↔ test in any class.")
    else:
        log.error(f"  [FAIL] Check 3: {total_leaking} total test images leak into train.")
        passed = False


    # ------------------------------------------------------------------ #
    # Check 4: Every class present in every partition
    # ------------------------------------------------------------------ #
    log.info("  Check 4: All classes present in all partitions...")
    class_ok = True
    for (crop, class_label), grp in df.groupby(["crop", "class_label"]):
        present = set(grp["partition"].dropna().unique())
        missing = set(partitions) - present
        if missing:
            log.error(
                f"  [FAIL] Check 4: {crop}/{class_label} is missing from "
                f"partition(s): {missing}"
            )
            class_ok = False

    if class_ok:
        log.info("  [PASS] Check 4: All classes present in all partitions.")
    else:
        passed = False

    # ------------------------------------------------------------------ #
    # Final verdict
    # ------------------------------------------------------------------ #
    log.info("\n" + "=" * 60)
    if passed:
        log.info("  ✅  AUDIT PASSED — Safe to proceed with file copy.")
    else:
        log.error("  ❌  AUDIT FAILED — DO NOT copy files. Review errors above.")
    log.info("=" * 60)

    return passed


# ---------------------------------------------------------------------------
# Stage 7 — Build Final Directory Structure
# ---------------------------------------------------------------------------

def build_final_directories(
    split_manifest: pd.DataFrame,
    output_root: Path,
    dry_run: bool = False,
) -> None:
    """
    Copy images from their original paths to the final output directory
    in the standardized train/val/test/class_label/ structure.

    ONLY call this after run_leakage_audit() returns True.

    Parameters
    ----------
    split_manifest : pd.DataFrame
        Full manifest with 'partition', 'crop', 'class_label', 'original_path' columns.
    output_root : Path
        Root directory for output. Files go to:
        output_root/<crop>_dataset/<partition>/<class_label>/<filename>
    dry_run : bool
        If True, prints what WOULD be done without touching any files.
    """
    log.info("=" * 60)
    if dry_run:
        log.info("STAGE 7: Build final directories [DRY RUN — no files written]")
    else:
        log.info("STAGE 7: Build final directories")
    log.info("=" * 60)

    df          = split_manifest.dropna(subset=["partition"]).copy()
    output_root = Path(output_root)
    errors      = []
    copied      = 0
    skipped     = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Copying"):
        src       = Path(row["original_path"])
        crop      = row["crop"]
        class_lbl = row["class_label"]
        partition = row["partition"]

        # Construct destination:
        # e.g. finaldataset/rice_dataset/train/healthy/IMG_20190419_094251.jpg
        dest_dir  = output_root / f"{crop}_dataset" / partition / class_lbl
        dest_file = dest_dir / src.name

        if dry_run:
            log.info(f"  WOULD COPY: {src.name}  →  {dest_file.relative_to(output_root)}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest_file.exists():
            skipped += 1
            continue

        try:
            shutil.copy2(src, dest_file)
            copied += 1
        except Exception as e:
            log.error(f"  [ERROR] Copy failed: {src} → {dest_file}: {e}")
            errors.append({"src": str(src), "dst": str(dest_file), "error": str(e)})

    if not dry_run:
        log.info(f"\n  ✓ Files copied  : {copied}")
        log.info(f"  ⊘ Files skipped : {skipped} (already exist)")
        if errors:
            log.error(f"  ✗ Copy errors   : {len(errors)}")
            for e in errors:
                log.error(f"    {e}")
        else:
            log.info("  ✓ Zero copy errors.")


# ---------------------------------------------------------------------------
# Stage 8 — Save Locked Split Manifest
# ---------------------------------------------------------------------------

def save_split_manifest(
    split_manifest: pd.DataFrame,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Save the final split manifest. This file is the immutable record of
    which image belongs to which partition. It must NEVER be regenerated
    or overwritten after the first save without human approval.

    Parameters
    ----------
    split_manifest : pd.DataFrame
    output_path : Path
    overwrite : bool
        Default False — raises FileExistsError if the file already exists.
        Set True only if you are intentionally replacing the manifest.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Split manifest already exists at: {output_path}\n"
            "The manifest is locked after first write. "
            "Set overwrite=True only with explicit human approval."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.to_csv(output_path, index=False)
    log.info(f"  ✓ Split manifest LOCKED → {output_path}")
    log.info(f"    Rows: {len(split_manifest)}")
    log.info(f"    Columns: {list(split_manifest.columns)}")


# ---------------------------------------------------------------------------
# Stage 9 — Statistics Report
# ---------------------------------------------------------------------------

def generate_statistics_report(split_manifest: pd.DataFrame) -> str:
    """
    Generate a formatted statistics report of the final split.

    Reports:
      - Per-crop, per-class, per-partition image counts
      - Class balance ratios within each crop
      - Source breakdown for each partition
      - Warnings for classes with < 500 training images
      - Warnings for 2:1+ class imbalance

    Returns
    -------
    str — human-readable report (also printed via log.info)
    """
    df = split_manifest.copy()
    lines = []

    lines.append("=" * 70)
    lines.append("IPD DATASET FINAL STATISTICS REPORT")
    lines.append("=" * 70)

    for crop in sorted(df["crop"].unique()):
        crop_df = df[df["crop"] == crop]
        lines.append(f"\n{'─'*70}")
        lines.append(f"  CROP: {crop.upper()}")
        lines.append(f"{'─'*70}")

        pivot = (
            crop_df.groupby(["class_label", "partition"])
            .size()
            .unstack(fill_value=0)
        )
        # Ensure all columns exist
        for col in ["train", "val", "test"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["train", "val", "test"]]
        pivot["TOTAL"] = pivot.sum(axis=1)

        lines.append(f"\n{pivot.to_string()}")

        # Class balance warnings
        train_counts = pivot["train"]
        if len(train_counts) > 0:
            min_count = train_counts.min()
            max_count = train_counts.max()

            if min_count < 500:
                lines.append(
                    f"\n  ⚠️  WARNING: Class '{train_counts.idxmin()}' has only "
                    f"{min_count} training images (< 500 minimum recommendation)."
                )
            if max_count > 0 and min_count > 0:
                ratio = max_count / min_count
                if ratio > 2.0:
                    lines.append(
                        f"\n  ⚠️  WARNING: Class imbalance ratio {ratio:.1f}:1 "
                        f"({train_counts.idxmax()}={max_count} vs "
                        f"{train_counts.idxmin()}={min_count}). "
                        "Consider class-weighted loss during training."
                    )

        # Source breakdown
        lines.append(f"\n  Source breakdown:")
        src_pivot = (
            crop_df.groupby(["source", "partition"])
            .size()
            .unstack(fill_value=0)
        )
        for col in ["train", "val", "test"]:
            if col not in src_pivot.columns:
                src_pivot[col] = 0
        src_pivot = src_pivot[["train", "val", "test"]]
        src_pivot["TOTAL"] = src_pivot.sum(axis=1)
        lines.append(src_pivot.to_string())

    lines.append(f"\n{'='*70}")
    lines.append("GRAND TOTAL")
    lines.append(f"{'='*70}")
    grand = df.groupby("partition").size()
    for p in ["train", "val", "test"]:
        lines.append(f"  {p:6s}: {grand.get(p, 0):6,} images")
    lines.append(f"  {'TOTAL':6s}: {len(df):6,} images")
    lines.append(f"{'='*70}\n")

    report = "\n".join(lines)
    log.info(report)
    return report


# ---------------------------------------------------------------------------
# Convenience function — Print Directory Tree of Output
# ---------------------------------------------------------------------------

def verify_output_tree(output_root: Path) -> None:
    """
    Walk the output directory and print counts of files per class per partition.
    Used as a final sanity check after build_final_directories().
    """
    log.info("=" * 60)
    log.info("OUTPUT DIRECTORY VERIFICATION")
    log.info("=" * 60)

    root = Path(output_root)
    if not root.exists():
        log.warning(f"  [WARN] Output directory does not exist yet: {root}")
        return

    for crop_dir in sorted(root.iterdir()):
        if not crop_dir.is_dir() or crop_dir.name.startswith("_"):
            continue
        log.info(f"\n  {crop_dir.name}/")
        for split_dir in sorted(crop_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            log.info(f"    {split_dir.name}/")
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                n_files = len([
                    f for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ])
                log.info(f"      {class_dir.name:<30} {n_files:6,} images")

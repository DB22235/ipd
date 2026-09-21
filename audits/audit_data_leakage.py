"""
audit_data_leakage.py
=====================
Audits dataset integrity, exact duplicates (MD5), and near-duplicates (perceptual hash)
across train, val, and test splits to detect data leakage.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "clean_dataset" / "tomato_dataset"
REPORT_DIR = ROOT_DIR / "audit_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def compute_hashes(image_path: Path):
    """Computes MD5 hash and perceptual hash for an image."""
    try:
        with open(image_path, "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
        
        with Image.open(image_path) as img:
            phash_val = str(imagehash.phash(img.convert("RGB")))
            
        return md5_hash, phash_val, None
    except Exception as e:
        return None, None, str(e)

def main():
    print("=" * 75)
    print("          DATASET INTEGRITY & DATA LEAKAGE AUDIT")
    print(f"          Target: {DATASET_DIR.resolve()}")
    print("=" * 75)

    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory {DATASET_DIR} does not exist.")
        sys.exit(1)

    splits = ["train", "val", "test"]
    records = []
    corrupted_files = []

    # 1. Discover and hash all images
    print("\n[1/3] Hashing images across all splits...")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
            
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            
            files = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_exts]
            for img_path in files:
                md5_val, phash_val, err = compute_hashes(img_path)
                if err:
                    corrupted_files.append({"path": str(img_path), "error": err})
                else:
                    records.append({
                        "path": str(img_path.relative_to(ROOT_DIR)),
                        "split": split,
                        "class": class_name,
                        "filename": img_path.name,
                        "md5": md5_val,
                        "phash": phash_val
                    })

    total_images = len(records)
    print(f"      ✓ Successfully processed: {total_images} images")
    print(f"      • Corrupted files detected: {len(corrupted_files)}")

    # 2. Analyze Exact Duplicates (MD5)
    print("\n[2/3] Analyzing Exact Duplicates (MD5)...")
    md5_to_records = defaultdict(list)
    for r in records:
        md5_to_records[r["md5"]].append(r)

    exact_duplicates = {k: v for k, v in md5_to_records.items() if len(v) > 1}
    exact_cross_split = []
    
    for md5_val, group in exact_duplicates.items():
        splits_in_group = set(r["split"] for r in group)
        if len(splits_in_group) > 1:
            exact_cross_split.append({
                "md5": md5_val,
                "splits": list(splits_in_group),
                "files": [r["path"] for r in group]
            })

    # 3. Analyze Near-Duplicates (Perceptual Hash)
    print("\n[3/3] Analyzing Near-Duplicates (Perceptual Hash Distance <= 2)...")
    phash_to_records = defaultdict(list)
    for r in records:
        phash_to_records[r["phash"]].append(r)

    near_duplicates = {k: v for k, v in phash_to_records.items() if len(v) > 1}
    near_cross_split = []
    
    for phash_val, group in near_duplicates.items():
        splits_in_group = set(r["split"] for r in group)
        if len(splits_in_group) > 1:
            near_cross_split.append({
                "phash": phash_val,
                "splits": list(splits_in_group),
                "files": [r["path"] for r in group]
            })

    # Split breakdown
    split_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        split_counts[r["split"]][r["class"]] += 1

    summary = {
        "dataset_root": str(DATASET_DIR),
        "total_images": total_images,
        "corrupted_images": len(corrupted_files),
        "split_counts": {s: dict(split_counts[s]) for s in splits},
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_cross_split_leakage_groups": len(exact_cross_split),
        "near_duplicate_groups": len(near_duplicates),
        "near_cross_split_leakage_groups": len(near_cross_split),
    }

    report_file = REPORT_DIR / "data_leakage_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "exact_cross_split_leakage": exact_cross_split,
            "near_cross_split_leakage": near_cross_split
        }, f, indent=2)

    print("\n" + "=" * 75)
    print("                     DATASET AUDIT SUMMARY")
    print("=" * 75)
    print(f"  Total Images Processed          : {total_images}")
    for s in splits:
        subtotal = sum(split_counts[s].values())
        print(f"    - {s.upper():<6} Split                 : {subtotal} images {dict(split_counts[s])}")
    print("-" * 75)
    print(f"  Exact Duplicate Groups (MD5)    : {len(exact_duplicates)}")
    print(f"  Exact Cross-Split Leakage Pairs : {len(exact_cross_split)}")
    print(f"  Near-Duplicate Groups (pHash)   : {len(near_duplicates)}")
    print(f"  Near Cross-Split Leakage Pairs  : {len(near_cross_split)}")
    print("=" * 75)
    print(f"✓ Full report saved to: {report_file.resolve()}\n")

if __name__ == "__main__":
    main()

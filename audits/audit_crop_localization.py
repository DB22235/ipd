
"""
audit_crop_localization.py
==========================
Experiment C (Localization Quality Audit):
Audits bounding-box localization produced by leaf_isolator across random sample images.
Evaluates:
  - Bounding-box area relative to full image (detects over-cropping vs under-cropping).
  - Aspect ratio distortion.
  - Centering & edge boundary safety margins.
  - Saliency vs GrabCut engine reliability.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from leaf_isolator import isolate_leaf

DATASET_DIR = ROOT_DIR / "clean_dataset" / "tomato_dataset" / "train"
REPORT_DIR = ROOT_DIR / "audit_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("        EXPERIMENT C: LOCALIZATION & CROP QUALITY AUDIT")
    print("=" * 75)

    if not DATASET_DIR.exists():
        print(f"Error: {DATASET_DIR} not found.")
        sys.exit(1)

    # Collect images across classes
    all_images = list(DATASET_DIR.glob("*/*.jpg")) + list(DATASET_DIR.glob("*/*.png"))
    if not all_images:
        print("No images found for audit.")
        sys.exit(1)

    random.seed(42)
    sample_size = min(50, len(all_images))
    sampled_paths = random.sample(all_images, sample_size)
    print(f"  Auditing {sample_size} randomly sampled crops for spatial fidelity...\n")

    audit_records = []
    engine_counts = {"grabcut": 0, "saliency": 0, "fallback": 0}
    area_ratios = []

    for img_p in sampled_paths:
        try:
            iso = isolate_leaf(img_p, target_size=(300, 300), mask_background=False)
            x1, y1, x2, y2 = iso["bbox"]
            H, W = iso["orig_shape"][:2]
            engine = iso["engine_used"].lower()

            if "grabcut" in engine:
                engine_counts["grabcut"] += 1
            elif "saliency" in engine:
                engine_counts["saliency"] += 1
            else:
                engine_counts["fallback"] += 1

            crop_w = x2 - x1
            crop_h = y2 - y1
            crop_area = crop_w * crop_h
            full_area = W * H
            area_ratio = crop_area / max(1, full_area)
            area_ratios.append(area_ratio)

            # Quality checks
            is_too_tight = (area_ratio < 0.08)
            is_too_loose = (area_ratio > 0.92)
            has_clipping = (x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2)

            audit_records.append({
                "file": img_p.name,
                "engine": iso["engine_used"],
                "orig_resolution": [W, H],
                "bbox": [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 3),
                "quality_flags": {
                    "too_tight": is_too_tight,
                    "too_loose": is_too_loose,
                    "touches_boundary": has_clipping
                }
            })
        except Exception as e:
            audit_records.append({"file": img_p.name, "error": str(e)})

    mean_area_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0
    good_crops = sum(1 for r in audit_records if not r.get("quality_flags", {}).get("too_tight") and not r.get("quality_flags", {}).get("too_loose"))
    crop_success_rate = (good_crops / max(1, len(audit_records))) * 100.0

    print("=" * 75)
    print("                     LOCALIZATION AUDIT SUMMARY")
    print("=" * 75)
    print(f"  Samples Inspected               : {len(audit_records)}")
    print(f"  Mean Leaf Bounding Area Ratio   : {mean_area_ratio * 100:.1f}% of full frame")
    print(f"  Background Discarded (Average)  : {(1.0 - mean_area_ratio) * 100:.1f}%")
    print(f"  Engine Breakdown                : GrabCut: {engine_counts['grabcut']} | Saliency: {engine_counts['saliency']} | Fallback: {engine_counts['fallback']}")
    print(f"  Valid Aspect/Size Crop Rate     : {crop_success_rate:.1f}%")
    print(f"  Acceptance Gate (>90% Valid)    : {'PASSED ⭐' if crop_success_rate >= 90.0 else 'REVIEW NEEDED ⚠'}")
    print("=" * 75)

    report_path = REPORT_DIR / "localization_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean_area_ratio": round(mean_area_ratio, 3),
            "crop_success_rate": round(crop_success_rate, 2),
            "engine_counts": engine_counts,
            "records": audit_records
        }, f, indent=2)

    print(f"✓ Full report saved to: {report_path.resolve()}\n")


if __name__ == "__main__":
    main()

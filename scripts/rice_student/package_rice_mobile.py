"""
scripts/rice_student/package_rice_mobile.py
===========================================
Executes Phase 1 of Rice Mobile Prototype Validation Plan:
  1. Verifies presence of Float16 LiteRT models in models/rice/converted/
  2. Copies primary (supervised_mobilenetv3_float16.tflite) and shadow (distilled_mobilenetv3_float16.tflite) to mobile/rice/
  3. Creates mobile/rice/phone_test_images/ drop-in folder for smartphone photos
  4. Generates mobile/rice/checksum.sha256 verifying all packaging assets
"""

import sys
import shutil
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 75)
    print("      PHASE 1: RICE MOBILE PROTOTYPE PACKAGING")
    print("=" * 75)

    mobile_dir = ROOT_DIR / "mobile" / "rice"
    mobile_dir.mkdir(parents=True, exist_ok=True)

    src_models_dir = ROOT_DIR / "models" / "rice" / "converted"
    sup_fp16 = src_models_dir / "supervised_mobilenetv3_float16.tflite"
    dist_fp16 = src_models_dir / "distilled_mobilenetv3_float16.tflite"

    if not sup_fp16.exists():
        raise FileNotFoundError(f"Primary model not found: {sup_fp16}")
    if not dist_fp16.exists():
        raise FileNotFoundError(f"Shadow model not found: {dist_fp16}")

    # 1. Copy Models
    dst_sup = mobile_dir / "supervised_mobilenetv3_float16.tflite"
    dst_dist = mobile_dir / "distilled_mobilenetv3_float16.tflite"

    print(f"[1/4] Packaging Model Artifacts into {mobile_dir.relative_to(ROOT_DIR)}...")
    shutil.copy2(sup_fp16, dst_sup)
    print(f"      Copied -> {dst_sup.name} ({dst_sup.stat().st_size / 1024 / 1024:.2f} MB)")
    shutil.copy2(dist_fp16, dst_dist)
    print(f"      Copied -> {dst_dist.name} ({dst_dist.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Setup Phone Test Images Drop-in Directory
    print("\n[2/4] Setting up Smartphone Test Drop-in Directory...")
    phone_img_dir = mobile_dir / "phone_test_images"
    phone_img_dir.mkdir(parents=True, exist_ok=True)
    readme_p = phone_img_dir / "README.md"
    readme_p.write_text(
        "# Smartphone Test Images Directory\n\n"
        "Drop any photos taken directly with your smartphone into this folder.\n"
        "The evaluation scripts will automatically ingest and test all images placed here.\n",
        encoding="utf-8"
    )
    print(f"      Ready -> {phone_img_dir.relative_to(ROOT_DIR)}/")

    # 3. Verify Package Files
    print("\n[3/4] Verifying Package Assets...")
    manifest_p = mobile_dir / "model_manifest.json"
    labels_p = mobile_dir / "labels.txt"
    prep_p = mobile_dir / "preprocessing.md"

    for p in [manifest_p, labels_p, prep_p]:
        if not p.exists():
            raise FileNotFoundError(f"Required package file missing: {p}")
        print(f"      Found -> {p.name}")

    # 4. Generate Authoritative Checksum File
    print("\n[4/4] Generating Cryptographic Checksum File (checksum.sha256)...")
    checksum_lines = []
    for p in [dst_sup, dst_dist, manifest_p, labels_p, prep_p]:
        sha = compute_file_sha256(p)
        checksum_lines.append(f"{sha}  {p.name}")
        print(f"      {p.name:<38}: {sha[:16]}...")

    checksum_file = mobile_dir / "checksum.sha256"
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"      [SAVED] -> {checksum_file.name}")

    print("\n" + "=" * 75)
    print(" [COMPLETE] Phase 1 Mobile Prototype Packaging Complete!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

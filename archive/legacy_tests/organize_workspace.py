"""
organize_workspace.py
=====================
Cleans and organizes the project workspace into a professional structure:
  1. audit_reports/  <- Moves all generated visual audit and Grad-CAM PNG plots
  2. test_images/    <- Moves all test leaf images (test*.png, test*.webp, test*.jpg, potatotest*)
  3. archive/        <- Moves old temporary test scripts (run_test3.py)
  4. Keeps core files clean in root: run_inference.py, leaf_isolator.py, models/, src/, etc.

Usage:
  python organize_workspace.py
"""

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Target directories
AUDIT_DIR = ROOT_DIR / "audit_reports"
TEST_IMG_DIR = ROOT_DIR / "test_images"
ARCHIVE_DIR = ROOT_DIR / "archive"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print("          ORGANIZING WORKSPACE DIRECTORY STRUCTURE")
print("=" * 64)

# 1. Move generated audit and gradcam plots
audit_files = list(ROOT_DIR.glob("*_audit.png")) + list(ROOT_DIR.glob("*_gradcam.png"))
moved_audits = 0
for f in audit_files:
    dest = AUDIT_DIR / f.name
    shutil.move(str(f), str(dest))
    moved_audits += 1

print(f"[1/3] Audit & Saliency Plots : Moved {moved_audits} files to -> audit_reports/")

# 2. Move loose test images from root
test_image_patterns = [
    "test*.png", "test*.webp", "test*.jpg", "test*.jpeg",
    "potatotest*.png"
]
moved_images = 0
for pat in test_image_patterns:
    for f in ROOT_DIR.glob(pat):
        # Don't move audit files if any match pattern
        if "_audit" in f.name or "_gradcam" in f.name:
            continue
        dest = TEST_IMG_DIR / f.name
        shutil.move(str(f), str(dest))
        moved_images += 1

print(f"[2/3] Test Leaf Images       : Moved {moved_images} files to -> test_images/")

# 3. Archive temporary one-off test scripts
scripts_to_archive = ["run_test3.py"]
moved_scripts = 0
for s_name in scripts_to_archive:
    s_path = ROOT_DIR / s_name
    if s_path.exists():
        dest = ARCHIVE_DIR / s_name
        shutil.move(str(s_path), str(dest))
        moved_scripts += 1

print(f"[3/3] Legacy Scripts         : Moved {moved_scripts} files to -> archive/")
print("=" * 64)
print("\nWorkspace structure is now clean and organized:")
print("  - run_inference.py   (Main field inference tool - auto-detects images in test_images/)")
print("  - leaf_isolator.py   (Option B leaf extraction module)")
print("  - test_images/       (All test potato images)")
print("  - audit_reports/     (All Grad-CAM & diagnostic visual audit figures)")
print("  - archive/           (Archived test scripts)")
print("  - models/            (Saved teacher and student models)")
print("  - src/               (Model architectures and augmentations)")
print("\nOrganization complete.")

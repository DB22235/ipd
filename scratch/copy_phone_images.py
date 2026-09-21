import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PHONE_DIR = ROOT_DIR / "mobile/potato/phone_test_images"
PHONE_DIR.mkdir(parents=True, exist_ok=True)

src_test_images = ROOT_DIR / "test_images"
# Copy available external potato images
potato_files = ["potatotest.png", "potatotest2.png", "potatotest2_cropped.png"]
for f in potato_files:
    p = src_test_images / f
    if p.exists():
        shutil.copy2(p, PHONE_DIR / f)
        print(f"Copied {f} to {PHONE_DIR}")

# Copy 3 out-of-domain non-potato test images for testing safe abstention
ood_files = ["test10.webp", "test11.webp", "test6_r.jpg"]
for f in ood_files:
    p = src_test_images / f
    if p.exists():
        shutil.copy2(p, PHONE_DIR / f"non_potato_{f}")
        print(f"Copied non_potato_{f} to {PHONE_DIR}")

print("phone_test_images population complete.")

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
src = ROOT_DIR / "reports/potato/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md"
dst = ROOT_DIR / "reports/potato/POTATO_MOBILE_VALIDATION_REPORT_FOR_MANUS.md"

shutil.copy2(src, dst)
print("Synchronized both master markdown files successfully.")

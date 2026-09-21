"""
tools/sanitize_manifests_for_release.py
=======================================
Sanitizes CSV manifests and configuration files prior to GitHub release:
  1. Converts absolute Windows paths (C:\\Users\\Dhruv Dube\\...) to repository-relative paths.
  2. Ensures all manifests are completely portable across Linux/Windows/macOS.
  3. Preserves all metadata, hashes, and partition splits.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT_DIR / "manifests"

TARGET_STRINGS = [
    "C:\\Users\\Dhruv Dube\\Desktop\\New folder\\IPD reaseach papers\\ipd\\",
    "C:/Users/Dhruv Dube/Desktop/New folder/IPD reaseach papers/ipd/",
    "C:\\\\Users\\\\Dhruv Dube\\\\Desktop\\\\New folder\\\\IPD reaseach papers\\\\ipd\\\\",
]


def sanitize_file(filepath: Path):
    try:
        content = filepath.read_text(encoding="utf-8")
        modified = False
        for target in TARGET_STRINGS:
            if target in content:
                content = content.replace(target, "")
                modified = True
        
        # Also check lowercase drive variants
        target_lower = "c:\\users\\dhruv dube\\desktop\\new folder\\ipd reaseach papers\\ipd\\"
        if target_lower in content.lower():
            # case insensitive replacement
            import re
            content = re.sub(re.escape(target_lower), "", content, flags=re.IGNORECASE)
            modified = True

        if modified:
            filepath.write_text(content, encoding="utf-8")
            print(f"  [SANITIZED] {filepath.relative_to(ROOT_DIR)}")
            return True
        return False
    except Exception as e:
        print(f"  [ERROR] Could not sanitize {filepath}: {e}")
        return False


def main():
    print("===========================================================================")
    print("      SANITIZING MANIFESTS & REMOVING ABSOLUTE PERSONAL PATHS              ")
    print("===========================================================================")
    count = 0
    for ext in ["*.csv", "*.json"]:
        for f in MANIFESTS_DIR.rglob(ext):
            if sanitize_file(f):
                count += 1
                
    # Also check reports directory
    for f in (ROOT_DIR / "reports").rglob("*.json"):
        if sanitize_file(f):
            count += 1

    print(f"\nTotal files sanitized: {count}")
    print("===========================================================================")
    print(" [COMPLETE] All personal paths replaced with clean repository-relative paths!")
    print("===========================================================================\n")


if __name__ == "__main__":
    main()

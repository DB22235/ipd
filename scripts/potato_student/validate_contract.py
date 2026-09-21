"""
scripts/potato_student/validate_contract.py
==========================================
Validates that all potato manifests, teacher checkpoints, label maps,
and dataset contracts exist and are structurally valid.
Produces: reports/potato/student/contract_validation.json
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.potato_student.contracts import assert_contract_integrity

REPORT_DIR = ROOT_DIR / "reports" / "potato" / "student"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_OUT = REPORT_DIR / "contract_validation.json"


def main():
    print("=" * 75)
    print("       STAGE 0: POTATO STUDENT CONTRACT INTEGRITY VALIDATION")
    print("=" * 75)

    try:
        results = assert_contract_integrity(ROOT_DIR)
        print("\n[SUCCESS] Potato contracts verified:")
        for k, v in results.items():
            print(f"  - {k}: {v}")

        with open(VALIDATION_OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved validation result to {VALIDATION_OUT}")

    except Exception as e:
        print(f"\n[FAIL] Contract validation failed: {e}")
        error_res = {"status": "FAILED", "error": str(e)}
        with open(VALIDATION_OUT, "w", encoding="utf-8") as f:
            json.dump(error_res, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()

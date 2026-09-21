"""
scripts/rice_student/validate_contract.py
=========================================
Stage 0 Contract Verification Script.
Validates:
  - Teacher checkpoint file presence & SHA-256 checksum
  - Teacher release manifest JSON schema and class order
  - Split manifest existence, multi-crop filtering, and partition counts
  - Agreement across all constants
Outputs report to reports/rice/student/contract_validation.json
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rice_student.contracts import (
    assert_contract_integrity,
    CLASSES,
    CLASS_TO_IDX,
    TEACHER_SHA256,
    EXPECTED_TOTAL_RICE_SAMPLES,
    EXPECTED_SPLIT_COUNTS,
)


def main():
    print("=" * 75)
    print("      STAGE 0: RICE STUDENT CONTRACT & MANIFEST VERIFICATION")
    print("=" * 75)

    report_dir = ROOT_DIR / "reports" / "rice" / "student"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "contract_validation.json"

    try:
        results = assert_contract_integrity(ROOT_DIR)
        print("\n[OK] Teacher Checkpoint SHA-256 Validated:")
        print(f"     Expected & Got : {TEACHER_SHA256}")

        print("\n[OK] Teacher Manifest Validated:")
        print(f"     Classes        : {CLASSES}")
        print(f"     Class Mapping  : {CLASS_TO_IDX}")

        print("\n[OK] Split Manifest Validated:")
        print(f"     Total Rice Samples : {EXPECTED_TOTAL_RICE_SAMPLES}")
        print(f"     Partitions         : {EXPECTED_SPLIT_COUNTS}")

        results["status"] = "ALL_CONTRACTS_PASSED"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n[PASS] Validation report written to: {report_file.relative_to(ROOT_DIR)}")
        print("=" * 75)
        return 0

    except Exception as e:
        print(f"\n[FAIL] Contract Verification Failed: {e}", file=sys.stderr)
        failure_results = {
            "status": "CONTRACT_VERIFICATION_FAILED",
            "error": str(e),
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(failure_results, f, indent=2)
        return 1


if __name__ == "__main__":
    sys.exit(main())

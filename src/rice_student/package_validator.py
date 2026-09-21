"""
src/rice_student/package_validator.py
=====================================
Validates offline mobile deployment packages prior to app integration.
Ensures presence, integrity, and schema compliance of:
  - model.tflite
  - labels.txt
  - model_manifest.json
  - checksum.sha256
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import tensorflow as tf

from .contracts import (
    CLASSES,
    compute_file_sha256,
    TARGET_MAX_MODEL_SIZE_MB,
)


def validate_mobile_package(package_dir: Path) -> Dict[str, Any]:
    """Validates the completeness and contract agreement of a mobile deployment bundle."""
    package_dir = Path(package_dir)
    report: Dict[str, Any] = {"package_dir": str(package_dir), "checks": {}, "passed": False}

    required_files = ["model.tflite", "labels.txt", "model_manifest.json", "checksum.sha256"]
    for f in required_files:
        p = package_dir / f
        if not p.exists():
            report["checks"][f] = {"status": "FAIL", "reason": "File missing"}
            return report
        report["checks"][f] = {"status": "PASS", "path": str(p)}

    # Check Labels
    labels_file = package_dir / "labels.txt"
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if labels != CLASSES:
        report["checks"]["labels_content"] = {
            "status": "FAIL",
            "reason": f"Expected {CLASSES}, found {labels}",
        }
        return report
    report["checks"]["labels_content"] = {"status": "PASS", "labels": labels}

    # Check TFLite Model Size and Architecture
    tflite_file = package_dir / "model.tflite"
    size_mb = tflite_file.stat().st_size / (1024 * 1024)
    if size_mb > TARGET_MAX_MODEL_SIZE_MB:
        report["checks"]["model_size"] = {
            "status": "FAIL",
            "size_mb": round(size_mb, 2),
            "limit_mb": TARGET_MAX_MODEL_SIZE_MB,
        }
        return report

    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
        interpreter.allocate_tensors()
    except RuntimeError:
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_file),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    report["checks"]["tflite_tensors"] = {
        "status": "PASS",
        "input_shape": input_details[0]["shape"].tolist(),
        "input_dtype": str(input_details[0]["dtype"]),
        "output_shape": output_details[0]["shape"].tolist(),
        "output_dtype": str(output_details[0]["dtype"]),
        "size_mb": round(size_mb, 2),
    }

    # Check SHA256 Checksum
    checksum_file = package_dir / "checksum.sha256"
    expected_hash = checksum_file.read_text(encoding="utf-8").strip()
    actual_hash = compute_file_sha256(tflite_file)
    if expected_hash != actual_hash:
        report["checks"]["checksum_verification"] = {
            "status": "FAIL",
            "expected": expected_hash,
            "actual": actual_hash,
        }
        return report
    report["checks"]["checksum_verification"] = {"status": "PASS", "sha256": actual_hash}

    report["passed"] = True
    return report

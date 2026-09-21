"""
src/rice_student/__init__.py
============================
IPD Rice Student Model Development Package.
Provides mobile model architectures, distillation loss functions, contracts,
data pipelines, calibration, and LiteRT mobile export utilities.
"""

from .contracts import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    INPUT_SHAPE_STUDENT,
    INPUT_SHAPE_TEACHER,
    TEACHER_MODEL_PATH,
    TEACHER_SHA256,
    SPLIT_MANIFEST_PATH,
    assert_contract_integrity,
)

__all__ = [
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "INPUT_SHAPE_STUDENT",
    "INPUT_SHAPE_TEACHER",
    "TEACHER_MODEL_PATH",
    "TEACHER_SHA256",
    "SPLIT_MANIFEST_PATH",
    "assert_contract_integrity",
]

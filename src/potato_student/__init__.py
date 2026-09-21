"""
src/potato_student
==================
Production-grade package for IPD Offline Potato Disease Detection Mobile Student Model.
Strictly isolated from Rice assets, standardized on TensorFlow / Keras with high-throughput
RAM resident data loading for ultra-fast multi-threaded CPU and GPU training.
"""

from .contracts import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
    INPUT_SHAPE_TEACHER,
)

__version__ = "1.0.0"
__all__ = [
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "INPUT_SHAPE_STUDENT",
    "INPUT_SHAPE_TEACHER",
]

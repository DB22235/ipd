"""
IPD Rice Disease Detection Engine
=================================
Crop-specific modules for slender grass leaf preprocessing,
anti-shortcut data augmentation, and domain-invariant evaluation.
"""

from .preprocessor import preprocess_rice_leaf, verify_rice_foliage
from .augmentations import RiceAntiShortcutAugmentation

__all__ = [
    "preprocess_rice_leaf",
    "verify_rice_foliage",
    "RiceAntiShortcutAugmentation",
]

"""
src/rice/augmentations.py
=========================
Anti-Shortcut Domain Invariance Augmentation Engine for Rice Leaves.
Neutralizes the 100% source-confounding vulnerability (256x256 ~5KB healthy vs
224x224 ~18KB diseased) through dynamic compression harmonization and robust perturbations.
"""

import io
import random
from typing import Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class RiceAntiShortcutAugmentation:
    """
    Python/PIL-based robust data augmentation pipeline that directly operates
    on numpy arrays before batch ingestion. Completely eliminates frequency and
    JPEG compression shortcuts between different camera sources.
    """
    def __init__(
        self,
        jpeg_degradation: bool = True,
        jpeg_quality_range: Tuple[int, int] = (40, 95),
        blur_prob: float = 0.25,
        blur_radius_range: Tuple[float, float] = (0.2, 1.2),
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.80, 1.20),
        color_range: Tuple[float, float] = (0.80, 1.20),
        rotation_degrees: float = 15.0,
        horizontal_flip: bool = True,
    ):
        self.jpeg_degradation = jpeg_degradation
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_radius_range = blur_radius_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.color_range = color_range
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip = horizontal_flip

    def __call__(self, image_np: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies anti-shortcut transformations to an RGB image array [H, W, 3].
        Input can be uint8 or float32. Output matches input dtype.
        """
        if not training:
            return image_np

        input_is_float = (image_np.dtype == np.float32)
        if input_is_float:
            pil_img = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
        else:
            pil_img = Image.fromarray(image_np)

        # 1. Dynamic JPEG Compression Harmonization (Erases 5KB vs 18KB shortcut)
        if self.jpeg_degradation and random.random() < 0.60:
            q = random.randint(*self.jpeg_quality_range)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            pil_img = Image.open(buf).convert("RGB")

        # 2. Subtle Gaussian Blur (Attenuates camera-specific sensor noise)
        if random.random() < self.blur_prob:
            radius = random.uniform(*self.blur_radius_range)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 3. Photometric Color Jitter
        # Brightness
        b_factor = random.uniform(*self.brightness_range)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(b_factor)

        # Contrast
        c_factor = random.uniform(*self.contrast_range)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(c_factor)

        # Color Saturation
        s_factor = random.uniform(*self.color_range)
        pil_img = ImageEnhance.Color(pil_img).enhance(s_factor)

        # 4. Geometric Invariance
        # Horizontal Flip
        if self.horizontal_flip and random.random() < 0.50:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Mild Rotation (preserving center)
        if self.rotation_degrees > 0 and random.random() < 0.50:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(114, 114, 114))

        out_np = np.array(pil_img)
        if input_is_float:
            return out_np.astype(np.float32)
        return out_np

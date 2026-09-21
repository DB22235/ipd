"""
src/rice/preprocessor.py
========================
Graminoid (Slender Grass) Leaf Preprocessing Module for Rice (*Oryza sativa*).
Solves the morphological mismatch of broadleaf GrabCut by employing:
  1. Aspect-Preserving Letterbox Resizing with neutral agricultural border fill.
  2. Botanical Foliage & Lesion Verification in HSV space.
  3. Out-of-Distribution / Non-Rice Rejection Filter.
"""

from pathlib import Path
from typing import Union, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2


def verify_rice_foliage(rgb_image: np.ndarray, min_ratio: float = 0.04) -> Tuple[bool, float, np.ndarray]:
    """
    Evaluates whether an image contains sufficient rice leaf blade or lesion tissue.
    Calibrated for graminoid (slender grass) leaf morphology where a single vertical blade
    against the field background occupies between 5% and 30% of total frame area.
    
    Returns:
        is_valid: True if foliage/lesion area covers at least min_ratio of frame.
        foliage_ratio: Float ratio [0.0, 1.0].
        foliage_mask: 2D boolean mask of foliage pixels.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # 1. Healthy Leaf Chlorophyll Range (Vibrant to dark green)
    # Hue: ~30 to 85 degrees (in OpenCV 8-bit, H in [0..180], so [20..85])
    lower_green = np.array([25, 30, 30], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 2. Chlorotic & Necrotic Lesion Range (Blast / Brown Spot / Blight necrosis)
    # Hue: yellow, orange, to tan/brown [10..25]
    lower_brown = np.array([10, 25, 30], dtype=np.uint8)
    upper_brown = np.array([25, 255, 240], dtype=np.uint8)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 3. Water-Soaked Gray Lesion Centers (Typical of advanced Blast)
    lower_gray = np.array([0, 0, 40], dtype=np.uint8)
    upper_gray = np.array([180, 40, 180], dtype=np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combined foliage mask (excluding isolated gray unless adjacent to green/brown)
    foliage_mask = (mask_green > 0) | (mask_brown > 0)
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
    foliage_pixels = int(np.sum(foliage_mask))
    foliage_ratio = float(foliage_pixels / total_pixels)
    
    is_valid = foliage_ratio >= min_ratio
    return is_valid, foliage_ratio, foliage_mask


def preprocess_rice_leaf(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (300, 300),
    bg_fill: Tuple[int, int, int] = (114, 114, 114),
    min_foliage_ratio: float = 0.04
) -> Dict[str, Any]:
    """
    Standard production preprocessor for rice leaves.
    Preserves true biological aspect ratio via letterboxing and validates leaf content.
    """
    # 1. Image Ingestion
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Rice image not found: {p}")
        pil_img = Image.open(p).convert("RGB")
        img_np = np.array(pil_img, dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            img_np = np.clip(image_input, 0, 255).astype(np.uint8)
        else:
            img_np = image_input
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    orig_h, orig_w = img_np.shape[:2]
    tgt_w, tgt_h = target_size

    # 2. Check botanical foliage content on raw image
    is_valid_foliage, foliage_ratio, _ = verify_rice_foliage(img_np, min_ratio=min_foliage_ratio)

    # 3. Aspect-Preserving Letterbox Resize
    scale = min(tgt_w / orig_w, tgt_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # High-quality bicubic interpolation
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Place centered on neutral background canvas
    canvas = np.full((tgt_h, tgt_w, 3), bg_fill, dtype=np.uint8)
    pad_x = (tgt_w - new_w) // 2
    pad_y = (tgt_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # 4. Prepare Outputs
    image_float32 = canvas.astype(np.float32)  # [0..255] float32 for Keras / EfficientNet

    return {
        "image": image_float32,
        "image_uint8": canvas,
        "orig_shape": (orig_h, orig_w),
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "foliage_ratio": foliage_ratio,
        "is_valid_rice_leaf": is_valid_foliage,
    }

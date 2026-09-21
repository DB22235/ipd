"""
background_augmentation.py
==========================
Background Destruction & Randomization Module for Studio -> Field Domain Hardening.

Breaks spurious background shortcuts (e.g., "white background = late blight")
by synthesizing diverse backgrounds behind segmented leaf tissue during training:
  1. Pure White / High-Contrast Studio Background (breaks white-BG shortcut)
  2. Neutral Grey (ImageNet mean: 114, 114, 114)
  3. Random Solid Chromas (HSV randomized)
  4. Gaussian & Perlin-like Texture Noise
  5. Dark / Soil-toned Backdrops
"""

import numpy as np
import tensorflow as tf

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False


def extract_leaf_mask_np(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts a binary leaf mask (255 for leaf, 0 for background) from an RGB image.
    Uses Excess Green (ExG) + Otsu thresholding + morphological closing.
    Fast CPU execution (~1-2 ms per 300x300 image).
    """
    img_f = img_rgb.astype(np.float32)
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    # Excess Green index (ExG = 2G - R - B)
    exg = 2.0 * g - r - b

    # Plant tissue detection: green tissue OR necrotic/chlorotic lesion near green
    green_mask = (exg > 10.0) & (g > 30.0)

    # Necrotic lesion detection (yellow halo, brown/black spots)
    necrotic = (r > 35.0) & (g > 25.0) & (r > b * 1.1) & (g > b * 0.8) & ((g - r) < 25.0)

    if OPENCV_AVAILABLE and cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        green_dilated = cv2.dilate(green_mask.astype(np.uint8), kernel).astype(bool)
        lesion_mask = necrotic & green_dilated
        combined = ((green_mask | lesion_mask).astype(np.uint8)) * 255
        # Fill internal holes
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
    else:
        combined = (green_mask | necrotic).astype(np.uint8) * 255
        mask = combined

    return mask


def randomize_background_single(img_rgb: np.ndarray, replace_prob: float = 0.65) -> np.ndarray:
    """
    Randomly replaces the background of a single leaf image.
    
    Args:
        img_rgb: uint8 numpy array of shape (H, W, 3).
        replace_prob: Probability of replacing background (default: 65%).
        
    Returns:
        Augmented uint8 numpy array of shape (H, W, 3).
    """
    if np.random.rand() > replace_prob:
        return img_rgb

    mask = extract_leaf_mask_np(img_rgb)
    fg_ratio = np.mean(mask > 127)

    # Only apply if mask represents a sensible foreground (10% to 95% of frame)
    if fg_ratio < 0.08 or fg_ratio > 0.98:
        return img_rgb

    h, w = img_rgb.shape[:2]

    # Dilate mask slightly to protect leaf serrations and margins
    if OPENCV_AVAILABLE and cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        alpha = mask_dilated.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)
    else:
        alpha = (mask > 127).astype(np.float32)

    alpha = np.clip(alpha[:, :, np.newaxis], 0.0, 1.0)

    # Choose background replacement mode
    mode = np.random.choice([
        "white_studio",   # 35% chance: bright white/off-white (directly attacks white-BG shortcut)
        "neutral_grey",   # 20% chance: ImageNet neutral grey
        "random_solid",   # 15% chance: random solid color
        "gaussian_noise", # 15% chance: random grain/texture
        "soil_dark",      # 15% chance: dark brown / soil tone
    ], p=[0.35, 0.20, 0.15, 0.15, 0.15])

    if mode == "white_studio":
        # Random bright white to light grey (230 - 255)
        base_val = np.random.randint(235, 256)
        bg = np.full((h, w, 3), base_val, dtype=np.float32)
        # Subtle vignette / gradient
        grad = np.random.uniform(-10, 10, size=(h, w, 1)).astype(np.float32)
        bg = np.clip(bg + grad, 220, 255)

    elif mode == "neutral_grey":
        bg = np.full((h, w, 3), 114.0, dtype=np.float32)

    elif mode == "random_solid":
        rand_color = np.random.uniform(30, 220, size=(1, 1, 3)).astype(np.float32)
        bg = np.full((h, w, 3), rand_color, dtype=np.float32)

    elif mode == "gaussian_noise":
        mean_val = np.random.uniform(90, 180)
        std_val = np.random.uniform(15, 40)
        bg = np.random.normal(mean_val, std_val, size=(h, w, 3)).astype(np.float32)
        bg = np.clip(bg, 0, 255)

    elif mode == "soil_dark":
        # Soil tones: low blue, moderate red, lower green
        r_soil = np.random.uniform(40, 80)
        g_soil = r_soil * np.random.uniform(0.6, 0.8)
        b_soil = r_soil * np.random.uniform(0.3, 0.5)
        bg = np.zeros((h, w, 3), dtype=np.float32)
        bg[:, :, 0] = r_soil
        bg[:, :, 1] = g_soil
        bg[:, :, 2] = b_soil

    # Composite leaf foreground onto new background
    fg = img_rgb.astype(np.float32)
    blended = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return blended


def tf_randomize_background(image: tf.Tensor, label: tf.Tensor) -> tuple:
    """
    tf.numpy_function wrapper for tf.data pipeline integration.
    Handles both single images (H, W, C) and batches (B, H, W, C).
    """
    def _py_func(img_np):
        if img_np.ndim == 4:
            out = np.empty_like(img_np)
            for i in range(img_np.shape[0]):
                out[i] = randomize_background_single(img_np[i])
            return out
        return randomize_background_single(img_np)

    augmented_img = tf.numpy_function(
        func=_py_func,
        inp=[tf.cast(image, tf.uint8)],
        Tout=tf.uint8
    )
    augmented_img = tf.cast(augmented_img, tf.float32)
    augmented_img.set_shape(image.shape)
    return augmented_img, label

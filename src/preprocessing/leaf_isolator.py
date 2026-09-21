"""
leaf_isolator.py
================
Lightweight Leaf Isolation & Preprocessing Module (Option B).
Isolates the target potato leaf from surrounding complex backgrounds (soil, mulch,
adjacent healthy foliage, stems, hands, or studio backdrops) prior to classification.

Engines:
  1. OpenCV GrabCut (Primary): Iterative graph-cut foreground extraction.
  2. Plant Saliency / Vegetation Index (Fallback / Standalone):
     Uses Excess Green (ExG) + Necrotic Lesion Chrominance + connected component
     morphology (NumPy + SciPy + PIL). Works even if OpenCV is not installed!

Target Output:
  Resized (300, 300, 3) leaf crop optimized for EfficientNetB3 Teacher Model.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

# Check for OpenCV availability
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

# Scipy for morphology fallback
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    ndimage = None
    SCIPY_AVAILABLE = False


def _load_image_rgb(image_input) -> Tuple[np.ndarray, Image.Image]:
    """Loads an image into uint8 RGB numpy array and PIL Image."""
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at: {p}")
        pil_img = Image.open(str(p)).convert("RGB")
        img_np = np.array(pil_img, dtype=np.uint8)
        return img_np, pil_img
    elif isinstance(image_input, np.ndarray):
        img_np = image_input.copy()
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        return img_np, pil_img
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        img_np = np.array(pil_img, dtype=np.uint8)
        return img_np, pil_img
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _grabcut_isolation(
    img_rgb: np.ndarray,
    pad_ratio: float = 0.10,
    min_area_ratio: float = 0.04
) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Applies OpenCV GrabCut with center rectangle initialization.
    Returns (bbox_xyxy, binary_mask) or None if GrabCut fails.
    """
    if not OPENCV_AVAILABLE:
        return None

    h, w = img_rgb.shape[:2]
    # Fast path: Downscale if image is very large to keep GrabCut fast (<100ms on CPU)
    scale = 1.0
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small_w = int(w * scale)
        small_h = int(h * scale)
        img_small = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_rgb
        small_h, small_w = h, w

    # Initialize GrabCut rectangle to central 84% of frame
    margin_x = max(2, int(small_w * 0.08))
    margin_y = max(2, int(small_h * 0.08))
    rect_w = max(4, small_w - 2 * margin_x)
    rect_h = max(4, small_h - 2 * margin_y)
    rect = (margin_x, margin_y, rect_w, rect_h)

    mask = np.zeros((small_h, small_w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            img_small,
            mask,
            rect,
            bgd_model,
            fgd_model,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_RECT
        )
        # Probable or definite foreground
        fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

        # Morphological closing to fill leaf interior holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find largest foreground contour
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        if area < (small_h * small_w * min_area_ratio):
            return None

        # Bounding box on small image
        sx, sy, sw, sh = cv2.boundingRect(largest_cnt)

        # Map back to original image coordinates
        x1 = int(sx / scale)
        y1 = int(sy / scale)
        x2 = int((sx + sw) / scale)
        y2 = int((sy + sh) / scale)

        # Add padding
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Upscale mask for debug output
        full_mask = cv2.resize(fg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (x1, y1, x2, y2), full_mask

    except Exception:
        return None


def _saliency_isolation(
    img_rgb: np.ndarray,
    pad_ratio: float = 0.08,
    min_area_ratio: float = 0.04
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Intelligent Plant Saliency & Lesion-Guided Leaf Isolator (NumPy + SciPy).
    Specifically designed for field scenes with multiple leaves / background foliage:
      1. Detects disease lesions (necrotic spots, concentric rings, chlorotic halos).
      2. If lesions exist, centers the bounding box on the diseased leaf host.
      3. Uses morphological opening to sever thin stems/petioles from background canopy.
      4. Penalizes border-touching foliage and prioritizes the central leaf subject.
      5. Eliminates distracting background leaves so classifier focuses on pathogen.
    """
    h, w = img_rgb.shape[:2]

    # Downscale large field images temporarily for fast morphology (<30ms)
    scale = 1.0
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        sh, sw = int(h * scale), int(w * scale)
        pil_tmp = Image.fromarray(img_rgb).resize((sw, sh), Image.Resampling.BILINEAR)
        small_rgb = np.array(pil_tmp, dtype=np.float32)
    else:
        sh, sw = h, w
        small_rgb = img_rgb.astype(np.float32)

    r = small_rgb[:, :, 0]
    g = small_rgb[:, :, 1]
    b = small_rgb[:, :, 2]

    # 1. Vegetation Index (Excess Green)
    exg = 2.0 * g - r - b
    green_mask = (exg > 15.0) & (g > 35.0)

    # 2. Disease Lesion Detector (Blight Necrosis & Chlorotic Halos)
    # Early/Late Blight lesions have low blue, moderate-high red and green (brown/black/yellow)
    # Lesions MUST be connected to or within proximity of green vegetation (to prevent bare soil/dirt from being flagged as lesions)
    if SCIPY_AVAILABLE and ndimage is not None:
        green_dilated = ndimage.binary_dilation(green_mask, iterations=10)
    elif OPENCV_AVAILABLE:
        kernel = np.ones((11, 11), np.uint8)
        green_dilated = cv2.dilate(green_mask.astype(np.uint8), kernel).astype(bool)
    else:
        green_dilated = green_mask

    necrotic_spot = (r > 38.0) & (g > 30.0) & (r > b * 1.18) & (g > b * 0.88) & ((g - r) < 22.0) & green_dilated
    yellow_halo   = (r > 100.0) & (g > 95.0) & (b < 85.0) & ((r + g) > 2.4 * b) & green_dilated
    lesion_mask   = (necrotic_spot | yellow_halo)

    # Exclude blue/uniform studio backdrops
    studio_blue = (b > r + 15.0) & (b > g + 15.0)
    green_mask[studio_blue] = False
    lesion_mask[studio_blue] = False

    # Combined plant tissue mask
    plant_mask = green_mask | lesion_mask

    # 3. Path A: Lesion-Guided Anchor (If disease lesions are clearly present in the image)
    lesion_coords = np.argwhere(lesion_mask)
    num_lesion_px = len(lesion_coords)
    min_lesion_px = max(60, int(sh * sw * 0.0008))

    if num_lesion_px >= min_lesion_px:
        # Lesions detected! Find the diseased leaf surrounding these lesions
        ly_min, lx_min = lesion_coords.min(axis=0)
        ly_max, lx_max = lesion_coords.max(axis=0)
        lesion_cy = (ly_min + ly_max) / 2.0
        lesion_cx = (lx_min + lx_max) / 2.0
        lesion_h  = (ly_max - ly_min)
        lesion_w  = (lx_max - lx_min)

        # The diseased leaf extends around the lesions
        # Expand box to encompass the host leaf blade
        leaf_span_y = max(lesion_h * 1.8, sh * 0.45)
        leaf_span_x = max(lesion_w * 1.8, sw * 0.45)

        sy1 = max(0, int(lesion_cy - leaf_span_y / 2.0))
        sy2 = min(sh, int(lesion_cy + leaf_span_y / 2.0))
        sx1 = max(0, int(lesion_cx - leaf_span_x / 2.0))
        sx2 = min(sw, int(lesion_cx + leaf_span_x / 2.0))

        # Map back to full image resolution
        x1 = int(sx1 / scale)
        y1 = int(sy1 / scale)
        x2 = int(sx2 / scale)
        y2 = int(sy2 / scale)

        # Pad slightly
        pw = int((x2 - x1) * pad_ratio)
        ph = int((y2 - y1) * pad_ratio)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

        mask_full = (plant_mask.astype(np.uint8) * 255)
        if scale != 1.0:
            mask_pil = Image.fromarray(mask_full).resize((w, h), Image.Resampling.NEAREST)
            mask_full = np.array(mask_pil)

        return (x1, y1, x2, y2), mask_full

    # 4. Path B: Central Subject Extraction (For healthy leaves or subtle symptoms)
    if SCIPY_AVAILABLE and ndimage is not None:
        # Morphological opening to sever thin stems from surrounding canopy
        k_size = max(5, int(min(sh, sw) * 0.025))
        struct = np.ones((k_size, k_size), dtype=bool)
        opened_mask = ndimage.binary_opening(plant_mask, structure=struct)
        labeled, num_features = ndimage.label(opened_mask)

        if num_features > 0:
            center_y, center_x = sh / 2.0, sw / 2.0
            best_score = -1.0
            best_bbox = None

            for i in range(1, num_features + 1):
                coords_i = np.argwhere(labeled == i)
                area_i = len(coords_i)
                if area_i < int(sh * sw * min_area_ratio):
                    continue

                iy_min, ix_min = coords_i.min(axis=0)
                iy_max, ix_max = coords_i.max(axis=0)
                cy_i = (iy_min + iy_max) / 2.0
                cx_i = (ix_min + ix_max) / 2.0

                # Distance to center
                d_norm = np.sqrt((cx_i - center_x) ** 2 + (cy_i - center_y) ** 2) / (np.sqrt(sh**2 + sw**2) / 2.0)
                centrality = np.exp(-3.0 * (d_norm ** 2))

                # Penalty if component touches all 4 edges (it is background canopy)
                touches_all = (iy_min <= 2) and (iy_max >= sh - 3) and (ix_min <= 2) and (ix_max >= sw - 3)
                border_mult = 0.15 if touches_all else 1.0

                score = float(area_i) * centrality * border_mult
                if score > best_score:
                    best_score = score
                    best_bbox = (ix_min, iy_min, ix_max, iy_max)

            if best_bbox is not None:
                bx1, by1, bx2, by2 = best_bbox
                x1 = max(0, int(bx1 / scale))
                y1 = max(0, int(by1 / scale))
                x2 = min(w, int(bx2 / scale))
                y2 = min(h, int(by2 / scale))

                pw = int((x2 - x1) * pad_ratio)
                ph = int((y2 - y1) * pad_ratio)
                x1 = max(0, x1 - pw)
                y1 = max(0, y1 - ph)
                x2 = min(w, x2 + pw)
                y2 = min(h, y2 + ph)

                mask_full = (plant_mask.astype(np.uint8) * 255)
                return (x1, y1, x2, y2), mask_full

    # 5. Path C: Central Focus Prior (If canopy fills entire frame from edge to edge)
    cx1, cy1 = int(w * 0.12), int(h * 0.12)
    cx2, cy2 = int(w * 0.88), int(h * 0.88)
    return (cx1, cy1, cx2, cy2), (plant_mask.astype(np.uint8) * 255)


def _letterbox_resize(
    pil_img: Image.Image,
    target_size: Tuple[int, int] = (300, 300),
    fill_color: Tuple[int, int, int] = (60, 60, 60)
) -> Image.Image:
    """
    Resizes image to target_size preserving natural aspect ratio.
    If aspect ratio is close to square (between 0.8 and 1.25), performs direct Lanczos resize.
    If aspect ratio is extreme (panoramic/tall), fits inside canvas and letterboxes to prevent squishing.
    """
    w, h = pil_img.size
    target_w, target_h = target_size
    aspect = w / float(max(1, h))

    # Near-square: direct resize without bars
    if 0.8 <= aspect <= 1.25:
        return pil_img.resize(target_size, Image.Resampling.LANCZOS)

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, color=fill_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def isolate_leaf(
    image_input,
    target_size: Tuple[int, int] = (300, 300),
    pad_ratio: float = 0.10,
    force_engine: Optional[str] = None,
    mask_background: bool = True,
    bg_neutral_val: int = 114,
    normalize_studio_white: bool = True,
    preserve_aspect_ratio: bool = True
) -> Dict[str, Any]:
    """
    Isolates the target leaf from the scene and formats it for EfficientNetB3.
    Applies optional background neutralization (neutral grey fill) using the
    segmented foreground mask to destroy spurious background shortcuts.

    When mask_background is False (natural unmasked crop), optionally detects
    artificial pure-white studio backgrounds (Y > 215, S < 25) and soft-blends
    them to dark neutral lab gray (60, 60, 60) to prevent out-of-distribution
    activation spikes while leaving real field soil and foliage 100% natural.

    Args:
        image_input: File path (str/Path), numpy array, or PIL Image.
        target_size: Output dimensions (height, width), default (300, 300).
        pad_ratio: Context padding around detected leaf boundary (default: 10%).
        force_engine: Optional 'grabcut', 'saliency', or None (auto-select).
        mask_background: If True, neutralizes background outside leaf mask to neutral grey.
        bg_neutral_val: Neutral background fill intensity (default: 114, ImageNet mean).
        normalize_studio_white: If True, normalizes stark white studio backgrounds to lab gray.
        preserve_aspect_ratio: If True, applies letterboxing to panoramic/tall aspect ratios.

    Returns:
        Dict containing:
          - "cropped_image": float32 array (target_size[0], target_size[1], 3) in [0..255]
          - "crop_uint8": uint8 array (target_size[0], target_size[1], 3) in [0..255] (model input)
          - "crop_raw_uint8": uint8 array unmasked crop for comparison/auditing
          - "bbox": (x1, y1, x2, y2) coordinates on original image
          - "orig_shape": (height, width, channels)
          - "orig_rgb": original uint8 RGB array
          - "engine_used": "GrabCut (OpenCV)" or "Plant Saliency / Vegetation Index"
          - "mask": 2D uint8 mask of the detected foreground
          - "bg_masked": bool indicating whether background neutralization was applied
          - "studio_white_normalized": bool indicating whether studio white was soft-blended
    """
    img_rgb, pil_img = _load_image_rgb(image_input)
    h, w = img_rgb.shape[:2]

    bbox = None
    mask = None
    engine_name = ""

    # Attempt 1: GrabCut (if requested or auto)
    if force_engine != "saliency" and OPENCV_AVAILABLE:
        result = _grabcut_isolation(img_rgb, pad_ratio=pad_ratio)
        if result is not None:
            bbox, mask = result
            engine_name = "GrabCut (OpenCV)"

    # Attempt 2: Vegetation & Lesion Saliency Fallback
    if bbox is None:
        bbox, mask = _saliency_isolation(img_rgb, pad_ratio=pad_ratio)
        engine_name = "Plant Saliency / Vegetation Index"

    x1, y1, x2, y2 = bbox

    # Safety check: ensure valid crop area
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        x1, y1 = int(w * 0.05), int(h * 0.05)
        x2, y2 = int(w * 0.95), int(h * 0.95)
        engine_name += " [Central Fallback]"

    # Perform crop on PIL Image for high-quality antialiased resampling
    crop_rgb = img_rgb[y1:y2, x1:x2].copy()
    raw_crop_pil = pil_img.crop((x1, y1, x2, y2))
    raw_resized_pil = raw_crop_pil.resize(target_size, Image.Resampling.LANCZOS)
    raw_crop_uint8 = np.array(raw_resized_pil, dtype=np.uint8)

    bg_masked_applied = False
    studio_white_applied = False
    if mask_background and mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        fg_ratio = float(np.mean(mask_crop > 127))
        # Mask validity check: leaf should cover reasonable fraction of crop
        if 0.08 <= fg_ratio <= 0.98:
            # Dilate mask slightly so margin lesions and serrations are preserved
            if OPENCV_AVAILABLE and cv2 is not None:
                k_size = max(5, int(min(mask_crop.shape[:2]) * 0.02) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_dilated = cv2.dilate(mask_crop, kernel, iterations=1)
                alpha = mask_dilated.astype(np.float32) / 255.0
                alpha = cv2.GaussianBlur(alpha, (5, 5), 1.2)
            elif SCIPY_AVAILABLE and ndimage is not None:
                mask_dilated = ndimage.binary_dilation(mask_crop > 127, iterations=3).astype(np.uint8) * 255
                alpha = mask_dilated.astype(np.float32) / 255.0
                alpha = ndimage.gaussian_filter(alpha, sigma=1.2)
            else:
                alpha = (mask_crop > 127).astype(np.float32)

            alpha = np.clip(alpha[:, :, np.newaxis], 0.0, 1.0)
            neutral_bg = np.full_like(crop_rgb, bg_neutral_val, dtype=np.float32)
            crop_rgb_masked = (crop_rgb.astype(np.float32) * alpha + neutral_bg * (1.0 - alpha)).astype(np.uint8)
            final_pil = Image.fromarray(crop_rgb_masked)
            bg_masked_applied = True
        else:
            final_pil = raw_crop_pil
    else:
        # Option B: Natural unmasked crop with surgical studio-white background normalization
        final_pil = raw_crop_pil
        if normalize_studio_white and mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            bg_indices = np.where(mask_crop <= 127)
            if len(bg_indices[0]) > 200:
                bg_pixels = crop_rgb[bg_indices]
                bg_luma = float(np.mean(bg_pixels))
                bg_sat = float(np.mean(np.max(bg_pixels, axis=-1) - np.min(bg_pixels, axis=-1)))
                # Detect stark white studio backdrop (Y > 215, S < 25)
                if bg_luma > 215.0 and bg_sat < 25.0:
                    if OPENCV_AVAILABLE and cv2 is not None:
                        k_size = max(5, int(min(mask_crop.shape[:2]) * 0.02) | 1)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                        mask_dilated = cv2.dilate(mask_crop, kernel, iterations=1)
                        alpha = cv2.GaussianBlur(mask_dilated.astype(np.float32) / 255.0, (5, 5), 1.5)
                    elif SCIPY_AVAILABLE and ndimage is not None:
                        mask_dilated = ndimage.binary_dilation(mask_crop > 127, iterations=2).astype(np.uint8) * 255
                        alpha = ndimage.gaussian_filter(mask_dilated.astype(np.float32) / 255.0, sigma=1.5)
                    else:
                        alpha = (mask_crop > 127).astype(np.float32)

                    alpha = np.clip(alpha[:, :, np.newaxis], 0.0, 1.0)
                    lab_neutral_bg = np.full_like(crop_rgb, 60, dtype=np.float32)
                    crop_normalized = (crop_rgb.astype(np.float32) * alpha + lab_neutral_bg * (1.0 - alpha)).astype(np.uint8)
                    final_pil = Image.fromarray(crop_normalized)
                    studio_white_applied = True

    if preserve_aspect_ratio:
        fill_col = (60, 60, 60) if (studio_white_applied or bg_masked_applied) else (30, 30, 30)
        resized_pil = _letterbox_resize(final_pil, target_size=target_size, fill_color=fill_col)
    else:
        resized_pil = final_pil.resize(target_size, Image.Resampling.LANCZOS)

    crop_uint8 = np.array(resized_pil, dtype=np.uint8)
    crop_float32 = crop_uint8.astype(np.float32)

    return {
        "cropped_image": crop_float32,
        "crop_uint8": crop_uint8,
        "crop_raw_uint8": raw_crop_uint8,
        "bbox": (x1, y1, x2, y2),
        "orig_shape": img_rgb.shape,
        "orig_rgb": img_rgb,
        "engine_used": engine_name,
        "mask": mask,
        "bg_masked": bg_masked_applied,
        "studio_white_normalized": studio_white_applied
    }


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "potatotest.png"
    print(f"Testing leaf isolation on: {test_img}")
    res = isolate_leaf(test_img)
    print(f"Engine Used    : {res['engine_used']}")
    print(f"Original Shape : {res['orig_shape']}")
    print(f"Leaf BBox      : {res['bbox']}")
    print(f"Cropped Shape  : {res['crop_uint8'].shape}")
    print("Done.")

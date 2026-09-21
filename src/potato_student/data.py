"""
src/potato_student/data.py
=========================
High-throughput in-memory data ingestion and preprocessing pipeline for the Potato Student Model.
Guarantees:
  1. High-Throughput RAM Caching: Preloads entire partition into contiguous numpy arrays
     at (224, 224, 3) uint8, eliminating all disk I/O during training.
  2. Multi-Crop Guardrail: Strictly enforces df['crop'] == 'potato'.
  3. Aspect-Preserving Letterbox: Resizes images to target size with neutral background
     fill (114, 114, 114) preserving leaf lesion geometry.
  4. Vectorized In-Graph Rescaling: Images are stored in RAM as uint8 [0, 255] and rescaled
     via Keras layers for maximum CPU SIMD/AVX2 efficiency.
  5. Balanced Inverse-Frequency Class Weights: Safeguards minority classes against imbalance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from .contracts import (
    CLASSES,
    CLASS_TO_IDX,
    INPUT_SHAPE_STUDENT,
    SPLIT_MANIFEST_PATH,
    NEUTRAL_BG_COLOR,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def letterbox_image(
    image: Union[np.ndarray, Image.Image, Path, str],
    target_size: Tuple[int, int] = (224, 224),
    bg_color: Tuple[int, int, int] = NEUTRAL_BG_COLOR,
) -> np.ndarray:
    """
    Resizes image maintaining aspect ratio and pads remaining area with neutral gray fill.
    Returns RGB uint8 numpy array of shape (target_size[0], target_size[1], 3).
    """
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image from {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = image

    h, w = img_rgb.shape[:2]
    target_h, target_w = target_size

    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize using INTER_AREA for downsampling or INTER_LINEAR for upsampling
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interpolation)

    # Create canvas with neutral fill
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # Compute offsets to center the image
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2

    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas


def resolve_image_path(path_str: Any, root_dir: Optional[Path] = None) -> Path:
    """Resolves image filepath handling relative and absolute paths."""
    if root_dir is None:
        root_dir = ROOT_DIR

    if isinstance(path_str, (pd.Series, dict)) or hasattr(path_str, "get"):
        path_str = path_str.get("filepath") or path_str.get("original_path") or path_str.get("filename") or ""

    p = Path(str(path_str))
    if p.is_file() and p.exists():
        return p
    if (root_dir / p).is_file() and (root_dir / p).exists():
        return root_dir / p

    fname = p.name
    # Fallback search across clean_dataset and finaldataset
    for candidate in [
        root_dir / "clean_dataset" / "potato_dataset",
        root_dir / "finaldataset" / "potato_dataset",
    ]:
        if candidate.exists():
            matched = list(candidate.glob(f"**/{fname}"))
            if matched:
                return matched[0]

    raise FileNotFoundError(f"Cannot resolve image path: {path_str}")


def load_potato_manifest(
    manifest_path: Optional[Path] = None,
    partition: Optional[str] = None,
) -> pd.DataFrame:
    """Loads potato split manifest, enforcing crop == 'potato' filter."""
    if manifest_path is None:
        manifest_path = ROOT_DIR / SPLIT_MANIFEST_PATH
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df_potato = df[df["crop"] == "potato"].copy()

    if len(df_potato) == 0:
        raise ValueError(f"No potato samples found in {manifest_path}")

    if partition is not None:
        partition = partition.lower()
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unknown partition '{partition}'. Must be 'train', 'val', or 'test'.")
        df_potato = df_potato[df_potato["partition"] == partition].copy()

    return df_potato


def compute_inverse_class_weights(df_train: pd.DataFrame) -> Dict[int, float]:
    """
    Computes inverse frequency class weights on the training partition:
      weight_i = total_samples / (num_classes * count_i)
    """
    class_counts = df_train["class_label"].value_counts().to_dict()
    total_samples = len(df_train)
    num_classes = len(CLASSES)

    class_weights: Dict[int, float] = {}
    for c in CLASSES:
        idx = CLASS_TO_IDX[c]
        cnt = class_counts.get(c, 0)
        if cnt == 0:
            raise ValueError(f"Class '{c}' has 0 samples in training partition!")
        class_weights[idx] = float(total_samples / (num_classes * cnt))

    return class_weights


def load_potato_split_to_ram(
    df: pd.DataFrame,
    target_size: Tuple[int, int] = (224, 224),
    root_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preloads all images in a partition directly into RAM as contiguous uint8 numpy arrays.
    Returns:
      X: np.ndarray of shape (N, target_size[0], target_size[1], 3) uint8
      y: np.ndarray of shape (N,) int32
      image_ids: List of image identifier strings
    """
    if root_dir is None:
        root_dir = ROOT_DIR

    n = len(df)
    target_h, target_w = target_size

    X = np.empty((n, target_h, target_w, 3), dtype=np.uint8)
    y = np.empty((n,), dtype=np.int32)
    image_ids = []

    from tqdm import tqdm
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=n, desc="Preloading into RAM")):
        img_path = resolve_image_path(row, root_dir=root_dir)
        canvas = letterbox_image(img_path, target_size=target_size)
        X[i] = canvas
        
        cls_name = row["class_label"]
        y[i] = CLASS_TO_IDX[cls_name]
        image_ids.append(str(row.get("image_id", img_path.stem)))

    return X, y, image_ids


def create_tf_dataset_from_ram(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 1024,
) -> tf.data.Dataset:
    """
    Converts RAM numpy arrays into an optimized tf.data.Dataset pipeline.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(X), shuffle_buffer), seed=42)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

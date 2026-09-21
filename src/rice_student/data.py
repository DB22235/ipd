"""
src/rice_student/data.py
========================
Data ingestion and preprocessing pipeline for the Rice Student Model.
Strictly conforms to:
  - Manifest-based loading (zero dynamic re-splitting)
  - Multi-crop filtering (df['crop'] == 'rice' only)
  - Aspect-preserving letterboxing with neutral fill (114, 114, 114)
  - Raw unscaled [0, 255] float32 pixel contract
  - Inverse frequency class weights to safeguard Blast recall
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .contracts import (
    CLASSES,
    CLASS_TO_IDX,
    INPUT_SHAPE_STUDENT,
    SPLIT_MANIFEST_PATH,
    EXPECTED_TOTAL_RICE_SAMPLES,
)

# Import canonical preprocessor from src.rice
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rice.preprocessor import preprocess_rice_leaf


def letterbox_image(
    img: Union[Image.Image, np.ndarray, Path, str],
    target_size: Tuple[int, int] = (224, 224),
    fill_color: Tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    """
    Standard letterboxing resize with neutral fill (114, 114, 114),
    preserving the biological aspect ratio of the rice leaf.
    """
    res = preprocess_rice_leaf(img, target_size=target_size, bg_fill=fill_color)
    return Image.fromarray(res["image_uint8"])


def resolve_image_path(
    path_str: Any,
    root_dir: Optional[Any] = None,
    partition: Optional[str] = None,
    class_label: Optional[str] = None,
) -> Path:
    """
    Resolves manifest path string or row, checking canonical clean_dataset and fallbacks.
    Polymorphic: handles both (path_str, root_dir) and (root_dir, row/path_str) orderings.
    """
    # Gracefully swap if called as resolve_image_path(ROOT_DIR, row_or_path)
    if isinstance(path_str, Path) and not isinstance(root_dir, Path):
        path_str, root_dir = root_dir, path_str

    if root_dir is None:
        root_dir = ROOT_DIR
    else:
        root_dir = Path(root_dir)

    # Handle pd.Series, dict, or object with get()
    if isinstance(path_str, (pd.Series, dict)) or hasattr(path_str, "get"):
        row = path_str
        partition = partition or row.get("partition")
        class_label = class_label or row.get("class_label")
        path_str = row.get("original_path") or row.get("filepath") or row.get("filename") or ""

    p = Path(str(path_str))
    fname = p.name

    # 1. Primary canonical source: clean_dataset/rice_dataset/{partition}/{class_label}/{fname}
    if partition and class_label:
        clean_p = root_dir / "clean_dataset" / "rice_dataset" / partition / class_label / fname
        if clean_p.exists():
            return clean_p

    # 2. Check if p is directly accessible or relative to root_dir
    if p.is_file() and p.exists():
        return p
    if (root_dir / p).is_file() and (root_dir / p).exists():
        return root_dir / p

    # 3. Check across all splits in clean_dataset/rice_dataset
    label = class_label or ("healthy" if "healthy" in p.parent.name.lower() else p.parent.name.lower())
    for part in ["train", "val", "test"]:
        cand = root_dir / "clean_dataset" / "rice_dataset" / part / label / fname
        if cand.exists():
            return cand

    # 4. Check finaldataset
    for part in ["train", "val", "test"]:
        cand = root_dir / "finaldataset" / "rice_dataset" / part / label / fname
        if cand.exists():
            return cand

    # 5. Local root fallback
    rel_p = root_dir / p.name
    if rel_p.is_file() and rel_p.exists():
        return rel_p

    raise FileNotFoundError(f"Cannot resolve image path: {path_str}")


def load_rice_manifest(manifest_path: Optional[Path] = None, partition: Optional[str] = None) -> pd.DataFrame:
    """
    Loads split manifest, strictly filtering for rice samples only.
    Prevents potato/tomato data infiltration.
    """
    if manifest_path is None:
        manifest_path = ROOT_DIR / SPLIT_MANIFEST_PATH
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    
    # CRITICAL: Multi-crop filtering guardrail
    df_rice = df[df["crop"] == "rice"].copy()
    if len(df_rice) != EXPECTED_TOTAL_RICE_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_RICE_SAMPLES} rice samples, found {len(df_rice)}"
        )

    if partition is not None:
        partition = partition.lower()
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"Unknown partition '{partition}'. Must be 'train', 'val', or 'test'.")
        df_rice = df_rice[df_rice["partition"] == partition].copy()

    return df_rice


def compute_inverse_class_weights(df_train: pd.DataFrame) -> Dict[int, float]:
    """
    Computes balanced class weights on the training split:
      weight_i = total_samples / (num_classes * count_i)
    Protects against class collapse on minority classes (e.g. Blast).
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


class RiceStudentDataset(Dataset):
    """
    PyTorch Dataset for Rice Student Training and Evaluation.
    Guarantees:
      - High-throughput in-memory RAM caching (preload=True) eliminating GPU starvation
      - Raw [0..255] float32 image representation (matching built-in Rescaling)
      - Letterboxed aspect-preserving resize with neutral fill (114, 114, 114)
      - Deterministic label indexing matching CLASS_TO_IDX
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        preload: bool = True,
        class_weights: Optional[Dict[int, float]] = None,
        root_dir: Optional[Path] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.is_training = is_training
        self.preload = preload
        self.class_weights = class_weights
        self.root_dir = Path(root_dir) if root_dir else ROOT_DIR

        self.file_paths: List[Path] = [
            resolve_image_path(
                row["original_path"],
                self.root_dir,
                partition=row.get("partition"),
                class_label=row.get("class_label"),
            )
            for _, row in self.df.iterrows()
        ]
        self.labels: List[int] = [
            CLASS_TO_IDX[c] for c in self.df["class_label"]
        ]

        self.cached_images: List[np.ndarray] = []
        if self.preload:
            split_name = self.df["partition"].iloc[0].capitalize() if "partition" in self.df.columns else "Dataset"
            print(f"      Preloading {len(self.file_paths)} {split_name} images into RAM at {target_size}...")
            for p in self.file_paths:
                res = preprocess_rice_leaf(p, target_size=self.target_size)
                self.cached_images.append(res["image_uint8"])
            ram_mb = (len(self.cached_images) * target_size[0] * target_size[1] * 3) / (1024 * 1024)
            print(f"      [OK] Preloaded {len(self.cached_images)} images ({ram_mb:.1f} MB RAM). GPU starvation eliminated.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        if self.preload:
            img_uint8 = self.cached_images[idx]
            img_float32 = img_uint8.astype(np.float32)
        else:
            p = self.file_paths[idx]
            res = preprocess_rice_leaf(p, target_size=self.target_size)
            img_float32 = res["image"]  # [H, W, 3] float32 in [0, 255]
        
        label = self.labels[idx]
        tensor_img = torch.tensor(img_float32, dtype=torch.float32)
        tensor_label = torch.tensor(label, dtype=torch.int64)

        if self.class_weights is not None:
            w = float(self.class_weights.get(label, 1.0))
            return tensor_img, tensor_label, torch.tensor(w, dtype=torch.float32)
        return tensor_img, tensor_label


def create_rice_dataloaders(
    manifest_path: Optional[Path] = None,
    batch_size: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    preload: bool = True,
    use_class_weights: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    root_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, float]]:
    """
    Constructs train, val, and test DataLoaders strictly from manifest.
    Returns (train_loader, val_loader, test_loader, class_weights).
    """
    df_train = load_rice_manifest(manifest_path, partition="train")
    df_val = load_rice_manifest(manifest_path, partition="val")
    df_test = load_rice_manifest(manifest_path, partition="test")

    class_weights = compute_inverse_class_weights(df_train) if use_class_weights else {}

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds = RiceStudentDataset(
        df_train,
        target_size=target_size,
        is_training=True,
        preload=preload,
        class_weights=class_weights if use_class_weights else None,
        root_dir=root_dir,
    )
    val_ds = RiceStudentDataset(
        df_val,
        target_size=target_size,
        is_training=False,
        preload=preload,
        class_weights=None,
        root_dir=root_dir,
    )
    test_ds = RiceStudentDataset(
        df_test,
        target_size=target_size,
        is_training=False,
        preload=preload,
        class_weights=None,
        root_dir=root_dir,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, class_weights

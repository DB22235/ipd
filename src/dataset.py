"""
IPD Phase 3: Dataset Loader & Partition Inspection Module
=========================================================
Loads clean partition directories (train, val, test) for Potato, Tomato, and Rice.
Features:
  - Parity audit across splits
  - tf.data pipeline with caching and autotuned prefetching
  - Balanced class weight computation for rare-class protection
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def inspect_dataset_partitions(dataset_root: Path) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """
    Audits class directories and file counts across train, val, and test splits.
    """
    splits = ["train", "val", "test"]
    split_dirs = {s: dataset_root / s for s in splits}

    for s, d in split_dirs.items():
        if not d.is_dir():
            raise FileNotFoundError(f"Partition directory not found: {d}")

    # Inspect classes
    classes_per_split = {}
    counts_per_split = {}

    for s, d in split_dirs.items():
        classes = sorted([
            f for f in os.listdir(d)
            if (d / f).is_dir()
        ])
        classes_per_split[s] = classes
        counts = {}
        for c in classes:
            c_path = d / c
            cnt = sum(1 for f in os.listdir(c_path) if f.lower().endswith(VALID_IMAGE_EXTS))
            counts[c] = cnt
        counts_per_split[s] = counts

    # Check parity
    ref_classes = classes_per_split["train"]
    for s in ["val", "test"]:
        if classes_per_split[s] != ref_classes:
            raise ValueError(
                f"Class mismatch between train and {s}!\n"
                f"Train: {ref_classes}\n{s.capitalize()}: {classes_per_split[s]}"
            )

    return ref_classes, counts_per_split


def load_crop_datasets(
    dataset_root: Path,
    img_size: Tuple[int, int] = (300, 300),
    batch_size: int = 16,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], Dict[int, float]]:
    """
    Loads train, val, test tf.data pipelines with caching and balanced class weights.
    
    Returns:
        (train_ds, val_ds, test_ds, class_names, class_weights)
    """
    class_names, counts = inspect_dataset_partitions(dataset_root)
    num_classes = len(class_names)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    print(f"\n{'='*65}")
    print(f"DATASET PARTITION AUDIT ({dataset_root.name.upper()} - {num_classes} classes)")
    print(f"{'='*65}")
    print(f"{'Class Name':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print(f"{'-'*65}")
    for c in class_names:
        tr = counts["train"].get(c, 0)
        va = counts["val"].get(c, 0)
        te = counts["test"].get(c, 0)
        print(f"{c:<25} {tr:<10} {va:<10} {te:<10} {tr+va+te:<10}")
    print(f"{'='*65}")

    # Build tf.data datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    # Compute balanced class weights
    train_labels = []
    for _, lbls in train_ds:
        train_labels.extend(lbls.numpy())
    train_labels = np.array(train_labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {int(cls): float(w) for cls, w in zip(np.unique(train_labels), weights)}

    # Optimize pipeline with cache and prefetch
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names, class_weights

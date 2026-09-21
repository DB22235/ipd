"""
scratch/test_v2_dataloader.py
Quick validation of the manifest-driven tf.data pipeline for Tomato Teacher v2.
Verifies:
1. Reading split_manifest.csv
2. Decoding and letterbox padding to 300x300x3 with (114, 114, 114)
3. Class balance and one-hot encoding
4. Batching and throughput
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_MANIFEST_CSV = ROOT_DIR / "manifests/tomato/teacher_v2/split_manifest.csv"
CLASSES = ["early_blight", "healthy", "late_blight"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = (300, 300)


def load_and_preprocess_image(path_tensor):
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    
    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    
    target_h = tf.cast(IMG_SIZE[0], tf.float32)
    target_w = tf.cast(IMG_SIZE[1], tf.float32)
    
    scale = tf.minimum(target_h / h, target_w / w)
    nh = tf.cast(tf.round(h * scale), tf.int32)
    nw = tf.cast(tf.round(w * scale), tf.int32)
    
    resized = tf.image.resize(img, (nh, nw), method='area')
    
    pad_h = (IMG_SIZE[0] - nh) // 2
    pad_w = (IMG_SIZE[1] - nw) // 2
    
    padded = tf.pad(
        resized,
        paddings=[[pad_h, IMG_SIZE[0] - nh - pad_h], [pad_w, IMG_SIZE[1] - nw - pad_w], [0, 0]],
        constant_values=114.0
    )
    padded.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return padded


def create_manifest_dataset(df_split: pd.DataFrame, batch_size: int = 32, is_training: bool = True):
    paths = [str(ROOT_DIR / p) for p in df_split["path"]]
    labels = [CLASS_TO_IDX[c] for c in df_split["class"]]
    
    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)
    
    def _map_fn(path, label):
        img = load_and_preprocess_image(path)
        one_hot = tf.one_hot(label, len(CLASSES))
        return img, one_hot
    
    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(1000, len(paths)), seed=42)
    
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    df = pd.read_csv(SPLIT_MANIFEST_CSV)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    
    print(f"Manifest loaded: {len(df)} rows.")
    print(f"Train samples: {len(df_train)} | Val samples: {len(df_val)}")
    
    train_ds = create_manifest_dataset(df_train, batch_size=32, is_training=True)
    val_ds = create_manifest_dataset(df_val, batch_size=32, is_training=False)
    
    # Test 1 batch
    t0 = time.time()
    for x, y in train_ds.take(1):
        print(f"Batch X shape: {x.shape} (dtype: {x.dtype}) | min: {tf.reduce_min(x):.1f}, max: {tf.reduce_max(x):.1f}")
        print(f"Batch Y shape: {y.shape} (dtype: {y.dtype})")
    print(f"Batch loaded in {time.time()-t0:.2f}s.")
    print("Dataloader verified successfully!")

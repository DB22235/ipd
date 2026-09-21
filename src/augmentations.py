"""
IPD Phase 3: Field-Robust Augmentation Pipeline
================================================
Phase A: Background-Invariant Augmentation Pipeline to bridge the Studio -> Field domain gap.

Forces the model to be invariant to studio background shortcuts by:
  1. Aggressive Inward Crop (RandomZoom height_factor=(-0.3, 0.0)) - Leaf Centering
  2. Multi-angle rotation (factor=0.15, ~±27°) and Horizontal + Vertical Flips
  3. Spatial shifting (RandomTranslation 15%)
  4. Color Jitter (Brightness ±20%, Contrast ±25%) - Simulates field sun/shadow
  5. Random Erasing / Cutout (RandomErasing 5-20% area patches) - Destroys contiguous background
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


class CutoutLayer(layers.Layer):
    """
    Fallback Custom Cutout / Random Erasing Layer.
    Randomly masks 1-2 rectangular patches with noise/mean to destroy background shortcuts.
    """
    def __init__(self, num_holes=2, min_ratio=0.08, max_ratio=0.22, **kwargs):
        super().__init__(**kwargs)
        self.num_holes = num_holes
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def call(self, images, training=None):
        if not training:
            return images

        shape = tf.shape(images)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)

        def apply_single(img):
            for _ in range(self.num_holes):
                hr = tf.random.uniform([], self.min_ratio, self.max_ratio)
                wr = tf.random.uniform([], self.min_ratio, self.max_ratio)
                hh = tf.cast(hr * h_f, tf.int32)
                ww = tf.cast(wr * w_f, tf.int32)

                y1 = tf.random.uniform([], 0, tf.maximum(1, h - hh), dtype=tf.int32)
                x1 = tf.random.uniform([], 0, tf.maximum(1, w - ww), dtype=tf.int32)

                mask = tf.concat([
                    tf.ones([y1, w, 1]),
                    tf.concat([
                        tf.ones([hh, x1, 1]),
                        tf.zeros([hh, ww, 1]),
                        tf.ones([hh, tf.maximum(0, w - x1 - ww), 1])
                    ], axis=1),
                    tf.ones([tf.maximum(0, h - y1 - hh), w, 1])
                ], axis=0)
                img = img * mask
            return img

        return tf.map_fn(apply_single, images)


def build_field_robust_augmentation(img_size=(300, 300)) -> tf.keras.Sequential:
    """
    Constructs the field-robust augmentation pipeline.
    Combines leaf-centering zoom, rotation, translation, color jitter, and random erasing.
    """
    aug_layers = [
        layers.RandomFlip("horizontal_and_vertical", name="random_flip"),
        layers.RandomRotation(factor=0.15, name="random_rotation"),
        layers.RandomZoom(
            height_factor=(-0.3, 0.0),
            width_factor=(-0.3, 0.0),
            fill_mode="reflect",
            name="random_zoom_inward"
        ),
        layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15,
            fill_mode="reflect",
            name="random_translation"
        ),
        layers.RandomBrightness(factor=0.2, value_range=(0, 255), name="random_brightness"),
        layers.RandomContrast(factor=0.25, name="random_contrast"),
    ]

    # Try native Keras 3 RandomErasing; fall back to CutoutLayer if needed
    try:
        if hasattr(layers, "RandomErasing"):
            erasing_layer = layers.RandomErasing(
                factor=0.5,
                scale=(0.05, 0.20),
                value_range=(0, 255),
                name="random_erasing"
            )
            aug_layers.append(erasing_layer)
        else:
            aug_layers.append(CutoutLayer(name="custom_cutout"))
    except Exception:
        aug_layers.append(CutoutLayer(name="custom_cutout"))

    return tf.keras.Sequential(aug_layers, name="field_robust_augmentation")


def visualize_augmentations(dataset, output_path: Path, num_samples: int = 6):
    """
    Visual verification of augmentation (Phase A Gate).
    Generates a 2-row side-by-side comparison of original vs augmented field-robust samples.
    """
    aug_pipeline = build_field_robust_augmentation()
    
    for images, labels in dataset.take(1):
        aug_images = aug_pipeline(images, training=True)
        num_cols = min(num_samples, len(images))
        fig, ax = plt.subplots(2, num_cols, figsize=(num_cols * 2.5, 5.5))

        for i in range(num_cols):
            # Row 0: Original
            orig = images[i].numpy().astype("uint8")
            ax[0, i].imshow(orig)
            ax[0, i].set_title(f"Original #{i+1}", fontsize=10)
            ax[0, i].axis("off")

            # Row 1: Field-Robust Augmented
            aug = aug_images[i].numpy().astype("uint8")
            ax[1, i].imshow(aug)
            ax[1, i].set_title(f"Augmented #{i+1}", fontsize=10)
            ax[1, i].axis("off")

        plt.suptitle(
            "Phase A: Field-Robust Augmentation Audit\n"
            "(Leaf-Centering Zoom, Random Flip/Rot, Color Jitter, Random Erasing)",
            fontsize=12,
            fontweight="bold"
        )
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Augmentation verification plot saved to: {output_path}")
        break

"""
IPD Phase 3: Teacher Model Architecture
========================================
EfficientNetB3 backbone with a robust custom classification head.
Features:
  - Native 300x300 resolution
  - Built-in Rescaling & Normalization via EfficientNetB3
  - Projection Head: GAP -> BN -> Dropout(0.4) -> Dense(256) -> BN -> ReLU -> Dropout(0.4) -> Dense(num_classes, logits)
  - Supports Two-Stage Transfer Learning (Stage A: Head, Stage B: Top 40 Backbone Layers)
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

from src.augmentations import build_field_robust_augmentation


def build_teacher_model(
    num_classes: int,
    img_size: Tuple[int, int] = (300, 300),
    dropout_rate: float = 0.4,
    crop: str = "potato",
    use_augmentation: bool = True
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs the EfficientNetB3 Teacher Model.
    
    Args:
        num_classes: Number of target disease classes
        img_size: Target image dimensions (300, 300)
        dropout_rate: Regularization rate for classification head (0.4 for Phase B)
        crop: Name of the crop (potato, tomato, rice)
        use_augmentation: If True, prepends field_robust_augmentation pipeline
        
    Returns:
        (full_model, backbone_model)
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_image")

    # 1. Augmentation Stage
    if use_augmentation:
        augmentation_pipeline = build_field_robust_augmentation(img_size=img_size)
        x = augmentation_pipeline(inputs)
    else:
        x = inputs

    # 2. Pretrained Backbone with built-in preprocessing
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size[0], img_size[1], 3)
    )
    # Stage A starts with frozen backbone
    backbone.trainable = False

    # Pass through backbone
    x = backbone(x)

    # 3. Regularized Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout1")(x)

    x = layers.Dense(256, activation=None, name="head_projection")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.ReLU(name="head_relu")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout2")(x)

    # Output linear logits in float32 for numerical stability
    outputs = layers.Dense(num_classes, activation=None, dtype="float32", name="logits")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"{crop}_teacher_efficientnetb3")
    return model, backbone


def setup_stage_b_fine_tuning(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    unfreeze_layers: int = 40,
    lr: float = 5e-5
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of EfficientNetB3 for Stage B fine-tuning.
    
    Args:
        model: Full teacher model
        backbone: EfficientNetB3 sub-model
        unfreeze_layers: Number of top backbone layers to train (default: 40)
        lr: Fine-tuning learning rate (default: 5e-5)
    """
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_layers]:
        layer.trainable = False

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"✓ Stage B Setup: Top {unfreeze_layers} backbone layers unfrozen.")
    print(f"  Trainable Parameters: {trainable_count:,} | Learning Rate: {lr}")
    return model

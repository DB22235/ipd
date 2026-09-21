"""
src/rice_student/models.py
==========================
MobileNetV3 Student Model architecture for Rice Leaf Disease Classification.
Provides:
  - MobileNetV3-Large backbone with ImageNet pretrained initialization
  - Linear logits output head (activation=None) for proper cross-entropy and KD
  - Built-in input rescaling handling raw [0, 255] float32 pixels
  - LiteRT-compatible operators for seamless mobile TFLite conversion
"""

from typing import Tuple, Dict, Any, Optional
import keras
from keras import layers, models

from .contracts import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
)


def build_student_augmentation_layer(name: str = "rice_student_augmentation") -> keras.Sequential:
    """
    Keras Sequential data augmentation pipeline executed on GPU/CPU.
    Active only when training=True, completely inactive when training=False.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.12),
        layers.GaussianNoise(0.02, name="student_noise"),
    ], name=name)


def build_mobilenetv3_student(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE_STUDENT,
    dropout_rate: float = 0.25,
    weights: Optional[str] = "imagenet",
    include_augmentation: bool = True,
    model_name: str = "rice_student_mobilenetv3_large",
) -> Tuple[keras.Model, keras.Model]:
    """
    Constructs the MobileNetV3-Large student model with linear logits head.
    Returns:
      (full_model, backbone_model)
    """
    base_model = keras.applications.MobileNetV3Large(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )

    inputs = layers.Input(shape=input_shape, name="input_image", dtype="float32")
    
    x = inputs
    if include_augmentation:
        aug_layer = build_student_augmentation_layer()
        x = aug_layer(x)

    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    
    # CRITICAL: Linear logits (activation=None) for numerical stability and KD integrity
    logits = layers.Dense(
        num_classes,
        activation=None,
        dtype="float32",
        name="logits",
    )(x)

    model = models.Model(inputs=inputs, outputs=logits, name=model_name)
    return model, base_model


def get_model_statistics(model: keras.Model) -> Dict[str, Any]:
    """Inspects model parameters and memory footprint without executing training."""
    total_params = model.count_params()
    trainable_params = sum(
        int(keras.ops.prod(keras.ops.shape(w))) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params
    estimated_size_mb = (total_params * 4) / (1024 * 1024)  # Float32 bytes

    return {
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "estimated_size_mb": round(estimated_size_mb, 2),
    }

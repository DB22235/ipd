"""
src/potato_student/models.py
===========================
Architecture definitions for the Potato Student Mobile Model.
Standardizes on MobileNetV3-Large with:
  - Vectorized in-graph Rescaling(1./255)
  - Pretrained ImageNet weights
  - Linear logits output (Dense(3, activation=None))
  - Two-phase training helpers (freeze/unfreeze backbone)
"""

from typing import Dict, Any, Tuple
import tensorflow as tf
import keras
from keras import layers, models

from .contracts import NUM_CLASSES, INPUT_SHAPE_STUDENT


def build_mobilenetv3_student(
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE_STUDENT,
    dropout_rate: float = 0.25,
    weights: str = "imagenet",
) -> keras.Model:
    """
    Builds MobileNetV3-Large student model with embedded rescaling and linear logits.
    """
    inputs = layers.Input(shape=input_shape, name="input_image", dtype="uint8")
    
    # Cast to float32 keeping [0, 255] range (MobileNetV3Large has built-in [-1, 1] rescaling)
    x = layers.Rescaling(scale=1.0, dtype="float32", name="cast_to_float32")(inputs)
    
    # Pretrained MobileNetV3-Large backbone
    base_model = keras.applications.MobileNetV3Large(
        input_tensor=x,
        include_top=False,
        weights=weights,
        minimalistic=False,
    )
    base_model._name = "mobilenetv3_large_backbone"

    # Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    logits = layers.Dense(num_classes, activation=None, name="logits")(x)

    model = models.Model(inputs=inputs, outputs=logits, name="potato_student_mobilenetv3_large")
    return model


def freeze_backbone(model: keras.Model) -> keras.Model:
    """Freezes all layers in the backbone, leaving only the classification head trainable."""
    for layer in model.layers:
        if layer.name in {"head_dropout", "logits"}:
            layer.trainable = True
        else:
            layer.trainable = False
    return model


def unfreeze_top_layers(model: keras.Model, num_layers: int = 40) -> keras.Model:
    """Unfreezes the top N layers of the model for fine-tuning."""
    # First freeze all
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze the last num_layers
    for layer in model.layers[-num_layers:]:
        layer.trainable = True
    return model


def unfreeze_all(model: keras.Model) -> keras.Model:
    """Unfreezes all layers in the model."""
    for layer in model.layers:
        layer.trainable = True
    return model


def get_model_statistics(model: keras.Model) -> Dict[str, Any]:
    """Computes total, trainable, and non-trainable parameter counts."""
    total_params = model.count_params()
    trainable_params = sum([keras.ops.size(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    return {
        "model_name": model.name,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
    }

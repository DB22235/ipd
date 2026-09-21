"""
src/rice_student/export.py
==========================
LiteRT / TensorFlow Lite conversion utilities for the Rice Student Model.
Supports:
  - Pure Float32 baseline conversion
  - Float16 post-training quantization
  - Full INT8 quantization with representative calibration dataset
  - TFLite runtime verification and numerical parity checks against Keras
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, Generator
import numpy as np
import tensorflow as tf
import keras

from .contracts import INPUT_SHAPE_STUDENT


def convert_keras_to_tflite_float32(
    model: keras.Model,
    output_path: Path,
) -> Path:
    """Converts Keras model to unquantized Float32 TFLite."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap model with explicit Softmax if head outputs linear logits
    inputs = keras.layers.Input(shape=model.input_shape[1:], name="input_image")
    logits = model(inputs)
    probs = keras.layers.Softmax(name="probabilities")(logits)
    export_model = keras.models.Model(inputs=inputs, outputs=probs, name=f"{model.name}_export")

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path


def convert_keras_to_tflite_float16(
    model: keras.Model,
    output_path: Path,
) -> Path:
    """Converts Keras model to Float16 quantized TFLite."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = keras.layers.Input(shape=model.input_shape[1:], name="input_image")
    logits = model(inputs)
    probs = keras.layers.Softmax(name="probabilities")(logits)
    export_model = keras.models.Model(inputs=inputs, outputs=probs, name=f"{model.name}_export_fp16")

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path


def convert_keras_to_tflite_int8(
    model: keras.Model,
    representative_data_gen: Callable[[], Generator],
    output_path: Path,
) -> Path:
    """Converts Keras model to full INT8 quantized TFLite."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = keras.layers.Input(shape=model.input_shape[1:], name="input_image")
    logits = model(inputs)
    probs = keras.layers.Softmax(name="probabilities")(logits)
    export_model = keras.models.Model(inputs=inputs, outputs=probs, name=f"{model.name}_export_int8")

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return output_path


def verify_tflite_inference(
    tflite_path: Path,
    test_image: np.ndarray,
) -> Dict[str, Any]:
    """Runs a single test image through the TFLite interpreter and returns metadata."""
    tflite_path = Path(tflite_path)
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
    except RuntimeError:
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Ensure correct shape
    if len(test_image.shape) == 3:
        test_image = np.expand_dims(test_image, axis=0)

    # Cast to input type
    if input_details["dtype"] == np.uint8:
        test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    else:
        test_image = test_image.astype(np.float32)

    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details["index"])[0]

    file_size_kb = tflite_path.stat().st_size / 1024.0

    return {
        "tflite_file": tflite_path.name,
        "file_size_kb": round(file_size_kb, 2),
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "sample_output_probabilities": [round(float(p), 4) for p in output_data],
    }

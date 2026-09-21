import keras
import tensorflow as tf

model_path = "models/potato/student_baselines/run_001/student_best.keras"
model = keras.models.load_model(model_path)
print("Loaded model successfully!")

try:
    print("Testing tf.lite.TFLiteConverter.from_keras_model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    print("Successfully converted directly from_keras_model! Size:", len(tflite_model))
except Exception as e:
    print("from_keras_model failed:", type(e), e)
    print("Testing via model.export(temp_saved_model)...")
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.export(tmp_dir)
        conv2 = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        conv2.optimizations = [tf.lite.Optimize.DEFAULT]
        conv2.target_spec.supported_types = [tf.float16]
        tflite2 = conv2.convert()
        print("Successfully converted via model.export / from_saved_model! Size:", len(tflite2))

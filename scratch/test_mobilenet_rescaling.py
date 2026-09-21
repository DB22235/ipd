import keras
import numpy as np
from keras import layers

# Test with uint8 directly into MobileNetV3Large
inputs_uint8 = layers.Input(shape=(224, 224, 3), dtype="float32", name="input_image")
base = keras.applications.MobileNetV3Large(input_tensor=inputs_uint8, include_top=False)
model_direct = keras.models.Model(inputs=inputs_uint8, outputs=base.output)

sample = np.full((1, 224, 224, 3), 255.0, dtype=np.float32)
rescaling_layer = base.get_layer("rescaling")
print("Output of internal rescaling on 255:", rescaling_layer(sample).numpy().max())

sample_0 = np.full((1, 224, 224, 3), 0.0, dtype=np.float32)
print("Output of internal rescaling on 0:", rescaling_layer(sample_0).numpy().min())

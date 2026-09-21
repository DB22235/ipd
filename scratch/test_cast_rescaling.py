import keras
from keras import layers
import numpy as np

inputs = layers.Input(shape=(224, 224, 3), name="input_image", dtype="uint8")
# Cast to float32 while keeping [0, 255] range
x = layers.Rescaling(scale=1.0, dtype="float32", name="cast_to_float32")(inputs)
base = keras.applications.MobileNetV3Large(input_tensor=x, include_top=False, weights=None)
out = layers.GlobalAveragePooling2D()(base.output)
out = layers.Dense(3)(out)
model = keras.Model(inputs=inputs, outputs=out)

# Test forward pass with uint8 array
sample = np.random.randint(0, 256, (2, 224, 224, 3), dtype=np.uint8)
pred = model(sample)
print("Forward pass successful! Output shape:", pred.shape)

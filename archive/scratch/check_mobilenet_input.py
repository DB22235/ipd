import keras
from keras.applications import mobilenet_v3
import inspect

print("preprocess_input source:")
print(inspect.getsource(mobilenet_v3.preprocess_input))

model = keras.applications.MobileNetV3Large(include_top=False, weights=None, input_shape=(224, 224, 3))
print("\nFirst 5 layers of MobileNetV3Large:")
for l in model.layers[:5]:
    print(l.name, l.__class__.__name__)

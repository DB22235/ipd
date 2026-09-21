import keras

model = keras.applications.MobileNetV3Large(include_top=False, weights=None, input_shape=(224, 224, 3))
rescaling_layer = model.get_layer("rescaling")
print("MobileNetV3 internal rescaling scale:", rescaling_layer.scale)
print("MobileNetV3 internal rescaling offset:", rescaling_layer.offset)

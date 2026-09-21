import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

m_teacher = keras.models.load_model("models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras", compile=False)
for w in m_teacher.weights[:3]:
    print("Weight name:", w.name, "device:", w.value.device)

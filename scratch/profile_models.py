import os
import time
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

def profile_each():
    batch_size = 16
    x_student = torch.randn(batch_size, 224, 224, 3, device="cuda")
    x_teacher = torch.randn(batch_size, 300, 300, 3, device="cuda")

    m_student = keras.applications.MobileNetV3Large(include_top=False, input_shape=(224, 224, 3), weights=None)
    m_teacher = keras.models.load_model("models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras", compile=False)

    # 1. Warmup
    with torch.no_grad():
        _ = m_student(x_student, training=False)
        _ = m_teacher(x_teacher, training=False)
    torch.cuda.synchronize()

    # 2. Benchmark MobileNetV3 Student alone (20 batches)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = m_student(x_student, training=False)
    torch.cuda.synchronize()
    ms_student = (time.time() - t0) / 20 * 1000

    # 3. Benchmark EfficientNetB3 Teacher alone (20 batches)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = m_teacher(x_teacher, training=False)
    torch.cuda.synchronize()
    ms_teacher = (time.time() - t0) / 20 * 1000

    print(f"MobileNetV3 Student alone : {ms_student:.2f} ms/batch (Batch Size 16)")
    print(f"EfficientNetB3 Teacher    : {ms_teacher:.2f} ms/batch (Batch Size 16)")
    print(f"Total Combined Forward    : {ms_student + ms_teacher:.2f} ms/batch")

if __name__ == "__main__":
    profile_each()

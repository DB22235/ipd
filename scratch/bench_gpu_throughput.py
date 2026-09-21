import os
import time
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras

def run_gpu_bench():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 65)
    print(f"       GPU FORWARD PASS BENCHMARK ({gpu_name})")
    print("=" * 65)

    batch_size = 16
    x_student = torch.randn(batch_size, 224, 224, 3, device="cuda")
    x_teacher = torch.randn(batch_size, 300, 300, 3, device="cuda")

    m_student = keras.applications.MobileNetV3Large(include_top=False, input_shape=(224, 224, 3), weights=None)
    m_teacher = keras.models.load_model("models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras", compile=False)

    # Warmup
    with torch.no_grad():
        _ = m_student(x_student)
        _ = m_teacher(x_teacher)
    torch.cuda.synchronize()

    # Benchmark 50 batches
    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = m_teacher(x_teacher)
            _ = m_student(x_student)
    torch.cuda.synchronize()
    t1 = time.time()

    ms_per_batch = (t1 - t0) / 50 * 1000
    total_train_samples = 3315
    batches_per_epoch = (total_train_samples + batch_size - 1) // batch_size
    epoch_sec = ms_per_batch * batches_per_epoch / 1000.0

    print(f"Batch Size                     : {batch_size}")
    print(f"GPU Time per Batch             : {ms_per_batch:.2f} ms")
    print(f"Batches per Epoch              : {batches_per_epoch}")
    print(f"Projected GPU Time per Epoch   : {epoch_sec:.2f} seconds")
    print(f"Projected 30 Epochs (Pure GPU) : {epoch_sec * 30 / 60:.2f} minutes")
    print("=" * 65)

if __name__ == "__main__":
    run_gpu_bench()

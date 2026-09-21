import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rice_student.data import load_rice_manifest, RiceStudentDataset
from src.rice.preprocessor import preprocess_rice_leaf

def run_benchmark():
    df_train = load_rice_manifest(partition="train")
    total_samples = len(df_train)
    test_subset = df_train.iloc[:100]

    print("=" * 65)
    print("       PIPELINE & GPU BOTTLENECK BENCHMARK")
    print("=" * 65)
    print(f"Total training samples: {total_samples}")

    # 1. On-the-fly disk read and resize
    ds_otf = RiceStudentDataset(test_subset, target_size=(224, 224))
    dl_otf = DataLoader(ds_otf, batch_size=16, shuffle=False)
    
    t0 = time.time()
    for b in dl_otf:
        pass
    t1 = time.time()
    ms_otf = (t1 - t0) / 100 * 1000
    epoch_otf_sec = ms_otf * total_samples / 1000.0

    print(f"\n[1] On-the-fly Disk I/O (Current baseline):")
    print(f"    Throughput  : {ms_otf:.2f} ms/image")
    print(f"    Epoch Time  : {epoch_otf_sec:.1f} seconds / epoch")
    print(f"    30 Epochs   : {epoch_otf_sec * 30 / 60:.1f} minutes spent purely on Disk I/O!")

    # 2. In-Memory Preload Benchmark
    print(f"\n[2] In-Memory Preloading (uint8 RAM Caching):")
    t0 = time.time()
    cached_images = []
    for p in ds_otf.file_paths:
        res = preprocess_rice_leaf(p, target_size=(224, 224))
        cached_images.append(res["image_uint8"])
    t1 = time.time()
    preload_time_sample = (t1 - t0) / 100
    total_preload_time = preload_time_sample * total_samples
    ram_mb = (total_samples * 224 * 224 * 3) / (1024 * 1024)

    # In-memory iteration speed
    t0 = time.time()
    batch_tensors = []
    for i in range(0, len(cached_images), 16):
        batch = [torch.tensor(img, dtype=torch.float32) for img in cached_images[i:i+16]]
        batch_tensors.append(torch.stack(batch))
    t1 = time.time()
    ms_preload = (t1 - t0) / 100 * 1000
    epoch_preload_sec = ms_preload * total_samples / 1000.0

    print(f"    RAM Footprint for all 3,315 train images : {ram_mb:.1f} MB (Extremely lightweight)")
    print(f"    One-time startup preload time            : {total_preload_time:.1f} seconds")
    print(f"    Throughput once cached                   : {ms_preload:.3f} ms/image")
    print(f"    Epoch Time once cached                   : {epoch_preload_sec:.2f} seconds / epoch")
    print(f"    Speedup Factor                           : {ms_otf / ms_preload:.1f}x FASTER!")
    print(f"    30 Epochs (Data time)                    : {epoch_preload_sec * 30:.1f} seconds (vs {epoch_otf_sec * 30 / 60:.1f} minutes)")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()

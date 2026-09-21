"""
benchmark_throughput.py
=======================
Step 2 Diagnostic: 3-Tier Throughput Baseline Benchmark.
Isolates the performance bottleneck across:
  Tier 1: Pure Synthetic GPU Compute (in VRAM, zero I/O)
  Tier 2: In-Memory RAM-Cached Tensors (Compute + PCIe transfer, zero disk reads)
  Tier 3: Disk-Backed Generator (Compute + PCIe + Windows file I/O)

Measures steady-state steps/sec, images/sec, and batch latency (discarding warmup).
Saves output to reports/throughput_baseline.json.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import keras
from keras import layers, models, optimizers

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = REPORTS_DIR / "throughput_baseline.json"
DATASET_PATH = ROOT_DIR / "bounding_box_dataset" / "tomato" / "train"

BATCH_SIZE = 16
NUM_STEPS = 25
WARMUP_STEPS = 5


def build_bench_model():
    """Builds an EfficientNetB3 feature extractor with frozen backbone and head."""
    base = keras.applications.EfficientNetB3(include_top=False, weights=None, input_shape=(300, 300, 3))
    base.trainable = False
    inp = layers.Input((300, 300, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-4), loss="categorical_crossentropy")
    return model


def benchmark_tier1_synthetic(model):
    """Tier 1: Tensors pre-allocated in CUDA VRAM (measures raw GPU compute ceiling)."""
    print("\n  [Tier 1/3] Benchmarking Pure GPU Compute (Pre-allocated CUDA Tensors)...")
    x_cuda = torch.randn(BATCH_SIZE, 300, 300, 3).cuda()
    y_cuda = torch.zeros(BATCH_SIZE, 3).cuda()
    y_cuda[:, 0] = 1.0

    # Warmup
    for _ in range(WARMUP_STEPS):
        model.train_on_batch(x_cuda, y_cuda)
    torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(NUM_STEPS):
        t0 = time.time()
        model.train_on_batch(x_cuda, y_cuda)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    mean_sec = float(np.mean(latencies))
    steps_per_sec = 1.0 / mean_sec
    imgs_per_sec = steps_per_sec * BATCH_SIZE
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"      • Steady-State Latency : {mean_sec*1000:.1f} ms / step")
    print(f"      • Throughput           : {steps_per_sec:.2f} steps/s ({imgs_per_sec:.1f} images/s)")
    print(f"      • Peak VRAM Allocated  : {peak_vram_mb:.1f} MB")
    return {
        "mean_step_ms": round(mean_sec * 1000, 2),
        "steps_per_sec": round(steps_per_sec, 2),
        "images_per_sec": round(imgs_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }


def benchmark_tier2_ram_cached(model):
    """Tier 2: Tensors sliced from contiguous RAM array (Compute + PCIe, zero disk reads)."""
    print("\n  [Tier 2/3] Benchmarking In-Memory RAM Caching (PCIe Transfer + GPU Compute)...")
    # Pre-allocate RAM arrays
    X_ram = np.random.randint(0, 256, (BATCH_SIZE * (NUM_STEPS + WARMUP_STEPS), 300, 300, 3), dtype=np.uint8)
    Y_ram = np.zeros((len(X_ram), 3), dtype=np.float32)
    Y_ram[:, 0] = 1.0

    # Warmup
    for i in range(WARMUP_STEPS):
        xb = X_ram[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        yb = Y_ram[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        model.train_on_batch(xb, yb)
    torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for i in range(WARMUP_STEPS, WARMUP_STEPS + NUM_STEPS):
        xb = X_ram[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        yb = Y_ram[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        t0 = time.time()
        model.train_on_batch(xb, yb)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    mean_sec = float(np.mean(latencies))
    steps_per_sec = 1.0 / mean_sec
    imgs_per_sec = steps_per_sec * BATCH_SIZE

    print(f"      • Steady-State Latency : {mean_sec*1000:.1f} ms / step")
    print(f"      • Throughput           : {steps_per_sec:.2f} steps/s ({imgs_per_sec:.1f} images/s)")
    return {
        "mean_step_ms": round(mean_sec * 1000, 2),
        "steps_per_sec": round(steps_per_sec, 2),
        "images_per_sec": round(imgs_per_sec, 2)
    }


def benchmark_tier3_disk_backed(model):
    """Tier 3: Sliced from disk directory generator (Compute + PCIe + Windows file I/O)."""
    print("\n  [Tier 3/3] Benchmarking Disk-Backed Pipeline (Real-Time File Reads)...")
    if not DATASET_PATH.exists():
        print(f"      ⚠ Dataset path {DATASET_PATH} not found; skipping Tier 3.")
        return None

    ds = keras.utils.image_dataset_from_directory(
        str(DATASET_PATH), image_size=(300, 300), batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True
    )
    iterator = iter(ds)

    # Warmup
    for _ in range(WARMUP_STEPS):
        xb, yb = next(iterator)
        model.train_on_batch(xb, yb)
    torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(NUM_STEPS):
        t0 = time.time()
        xb, yb = next(iterator)
        model.train_on_batch(xb, yb)
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    mean_sec = float(np.mean(latencies))
    steps_per_sec = 1.0 / mean_sec
    imgs_per_sec = steps_per_sec * BATCH_SIZE

    print(f"      • Steady-State Latency : {mean_sec*1000:.1f} ms / step")
    print(f"      • Throughput           : {steps_per_sec:.2f} steps/s ({imgs_per_sec:.1f} images/s)")
    return {
        "mean_step_ms": round(mean_sec * 1000, 2),
        "steps_per_sec": round(steps_per_sec, 2),
        "images_per_sec": round(imgs_per_sec, 2)
    }


def main():
    print("=" * 80)
    print("         STEP 2: 3-TIER THROUGHPUT & BOTTLENECK BASELINE BENCHMARK")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"  Target Device : {torch.cuda.get_device_name(0)}")
    print(f"  Batch Size    : {BATCH_SIZE}")
    print(f"  Resolution    : (300, 300, 3)")
    print(f"  Sample Steps  : {NUM_STEPS} steps (+ {WARMUP_STEPS} warmup)")

    model = build_bench_model()

    res_t1 = benchmark_tier1_synthetic(model)
    res_t2 = benchmark_tier2_ram_cached(model)
    res_t3 = benchmark_tier3_disk_backed(model)

    # Diagnosis Analysis
    diag = {}
    if res_t3:
        disk_overhead = res_t3["mean_step_ms"] - res_t2["mean_step_ms"]
        ram_speedup = res_t2["images_per_sec"] / max(0.1, res_t3["images_per_sec"])
        diag["disk_overhead_ms"] = round(disk_overhead, 2)
        diag["ram_speedup_factor"] = round(ram_speedup, 2)
        diag["primary_bottleneck"] = "Disk I/O and File Reading" if disk_overhead > 50 else "GPU Compute"
    else:
        diag["primary_bottleneck"] = "N/A"

    report = {
        "device": torch.cuda.get_device_name(0),
        "batch_size": BATCH_SIZE,
        "tier1_pure_gpu_compute": res_t1,
        "tier2_in_memory_ram_cached": res_t2,
        "tier3_disk_backed_baseline": res_t3,
        "diagnosis": diag
    }

    print("\n" + "=" * 80)
    print("                     BENCHMARK SUMMARY & DIAGNOSIS")
    print("=" * 80)
    print(f"  • Tier 1 (Pure GPU Compute)   : {res_t1['images_per_sec']} images/sec ({res_t1['mean_step_ms']} ms/step)")
    print(f"  • Tier 2 (RAM In-Memory Cache): {res_t2['images_per_sec']} images/sec ({res_t2['mean_step_ms']} ms/step)")
    if res_t3:
        print(f"  • Tier 3 (Disk-Backed Baseline): {res_t3['images_per_sec']} images/sec ({res_t3['mean_step_ms']} ms/step)")
        print(f"  ★ Estimated Speedup from RAM  : {diag['ram_speedup_factor']}x faster than disk baseline")
        print(f"  ★ Root Bottleneck Identified  : {diag['primary_bottleneck']}")
    print("=" * 80)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Throughput baseline saved to: {OUT_FILE.resolve()}\n")


if __name__ == "__main__":
    main()

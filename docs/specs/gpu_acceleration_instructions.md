# GPU Acceleration and Fast Training Instructions for IPD

**Purpose:** Reduce training time for the tomato EfficientNetB3 teacher on an NVIDIA RTX 4050 Laptop GPU while preserving correctness, reproducibility, and deployment compatibility.

**Current symptom:** Stage 1 training takes approximately 20–25 minutes for only 3 epochs. This is abnormally slow for a dataset of a few thousand 300×300 images unless the pipeline is running on the CPU, blocked by image decoding/augmentation, repeatedly rebuilding data, or incurring large Keras/PyTorch backend overhead.

**Important:** Do not start by increasing batch size or changing the model. First prove where the time is being spent.

---

## 1. Engineering diagnosis

The expected causes, in priority order, are:

1. CUDA is not actually being used.
2. GPU is being used, but image decoding and augmentation are limiting throughput.
3. Images are decoded from disk repeatedly or from a slow/network/synchronized directory.
4. The input pipeline is running Python code per image and preventing parallel loading.
5. The Keras PyTorch backend is adding overhead for the current model or data pipeline.
6. Windows power management or another application is throttling the laptop GPU.
7. Batch size is too small for the available VRAM.
8. Excessive callbacks, validation frequency, Grad-CAM generation, or checkpointing is occurring inside the training loop.

Never assume that `torch.cuda.is_available()` alone proves that the model is training on the GPU. Verify the actual model parameters, input tensors, GPU utilization, GPU memory, and epoch throughput.

---

## 2. First required diagnostic: record the environment

Run these commands in the same environment used to train. Save their output to `reports/gpu_environment.txt`.

```powershell
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('torch_cuda:', torch.version.cuda); print('device_count:', torch.cuda.device_count()); print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -c "import keras; print('keras:', keras.__version__); print('backend:', keras.config.backend())"
```

The expected result is:

- `cuda_available: True`.
- Device name contains `RTX 4050`.
- The CUDA-enabled PyTorch build reports a CUDA version.
- Keras reports the intended backend, normally `torch` for the current setup.

If CUDA is unavailable, do not optimize the training script yet. Fix the environment first.

### Windows installation correction

For the Keras PyTorch backend, install a CUDA-enabled PyTorch wheel from the official PyTorch installation selector. Do not install a CPU-only wheel by accident. The exact command depends on the desired PyTorch/CUDA combination and must be selected for the current supported version.

After installation, rerun the diagnostic commands and confirm `nvidia-smi` shows the Python process during training.

---

## 3. Required runtime assertions in the training script

Add a startup diagnostic before creating the model:

```python
import os
import time
import torch
import keras

os.environ.setdefault("KERAS_BACKEND", "torch")

print("Keras version:", keras.__version__)
print("Keras backend:", keras.config.backend())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Stop before training; the run would use the CPU."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("GPU count:", torch.cuda.device_count())
```

After constructing the model, verify that model parameters are on CUDA. The exact inspection depends on the Keras version, but the agent must print the device of at least one trainable parameter or underlying Torch parameter. Do not accept a run until this is verified.

During an active training epoch, run `nvidia-smi` in a second terminal. Record:

- GPU utilization percentage.
- GPU memory used.
- Power draw.
- Temperature.
- Whether the Python process appears in the process list.

Interpretation:

| Observation | Likely cause |
|---|---|
| GPU memory near zero and utilization near zero | CPU training or model not moved to CUDA |
| GPU utilization repeatedly near zero with CPU high | Input pipeline bottleneck |
| GPU utilization high but epoch still slow | Model/input resolution, thermal throttling, or backend overhead |
| GPU memory almost full | Batch size or activation memory too large |
| GPU power/clock very low | Battery mode, Windows power management, or thermal throttling |

---

## 4. Establish a timing baseline before changing anything

Run exactly one short benchmark with:

- 100 training batches.
- No validation.
- No Grad-CAM.
- No image export.
- No per-batch logging.
- No model checkpoint on every batch.
- Fixed batch size.

Measure:

```text
steps per second
images per second
GPU utilization
CPU utilization
peak GPU memory
first-batch time
steady-state batch time
```

Discard the first 5–10 warm-up batches when calculating steady-state throughput. The first batch includes graph/model initialization and is not representative.

Run three isolated benchmarks:

1. Synthetic tensors already in memory.
2. Decoded images cached in memory.
3. The real disk-backed input pipeline.

Interpretation:

- Synthetic fast, disk slow: input pipeline is the bottleneck.
- Synthetic slow: GPU/backend/model execution is the bottleneck.
- All variants run on CPU: CUDA setup is incorrect.

The agent must save this result as `reports/throughput_baseline.json`.

---

## 5. Data pipeline rules

The input pipeline must not decode images using a slow Python loop inside every training step. It must not recreate the dataset or reload the complete image directory at every epoch.

Use these rules:

1. Store the dataset on a local SSD, not a network drive, cloud-synchronized directory, USB drive, or compressed archive.
2. Decode each image once per epoch at most.
3. Use parallel workers for file loading.
4. Prefetch batches so the GPU receives the next batch while the current batch is executing.
5. Cache decoded data in RAM only when the available system memory is sufficient.
6. Do not cache the validation set incorrectly or mix train and validation data.
7. Apply augmentation in a vectorized batch operation when possible.
8. Avoid PIL/OpenCV work inside a single-threaded Python generator.
9. Do not call `.numpy()` or move tensors CPU-side inside each training step.
10. Do not perform Grad-CAM, plotting, or image saving during training.

For a small dataset of a few thousand 300×300 images, the agent should benchmark an in-memory or preprocessed cache. A practical option is to decode and resize images once into a local cache such as `.npy`, `.npz`, WebDataset, TFRecord, or another documented format, then train from that cache. The original files and cache must have checksums and remain traceable. In-memory ingestion is feasible, but it does not mean that the GPU receives data with zero transfer: CPU-resident arrays still need to be copied to GPU memory unless the entire pipeline uses verified pinned-memory and asynchronous transfers.

If using PyTorch data loading directly underneath Keras, benchmark `num_workers` values such as 2, 4, and 8. Do not assume more workers are always faster on Windows. Use `persistent_workers=True` when supported and avoid repeatedly spawning workers. If Windows multiprocessing causes instability, use a safe main guard and benchmark fewer workers.

If using a Keras-native pipeline, use parallel mapping and prefetching supported by the selected backend. Do not mix TensorFlow-only pipeline assumptions into a PyTorch-backend run without testing.

### 5.1 Recommended one-time RAM ingestion

The current dataset is small enough that loading decoded, resized images into system RAM is a reasonable optimization experiment. The implementation must follow this sequence:

1. Read the immutable split manifest.
2. Decode each image exactly once during startup.
3. Convert to RGB and resize to the exact model input size.
4. Store images in a compact contiguous array, preferably `uint8` when preprocessing can be performed in a verified vectorized batch operation.
5. Store labels and group IDs in separate arrays.
6. Keep training, validation, and test arrays physically separate.
7. Print the RAM estimate before allocation and fail clearly if the memory budget is unsafe.
8. Benchmark RAM ingestion time separately from training time.
9. Reuse the arrays across every epoch without reopening image files.
10. Preserve the same augmentation, class weights, labels, and split semantics as the disk-backed baseline.

For an approximate estimate, an array of `N` RGB images at 300×300 uses `N × 300 × 300 × 3` bytes as `uint8`, or four times that amount as float32. The implementation must calculate the exact estimate instead of relying on this example. Do not convert the complete dataset to float32 in RAM unless sufficient system memory is confirmed.

RAM ingestion is expected to remove repeated disk decoding and Python image-loader overhead. It does **not** guarantee 15–20 second epochs. Each batch may still require CPU-to-GPU transfer, augmentation, model execution, validation, and callback work. The expected improvement must be measured as images/second and seconds/epoch.

If using PyTorch tensors, use pinned host memory only when the selected data loader and backend support it correctly. Use non-blocking transfers only when the source tensor is pinned and correctness has been verified. Do not claim zero-copy transfer unless the implementation actually uses a compatible shared-memory or GPU-resident pipeline and this has been measured.

The RAM cache must never combine training and validation data, must not alter image order or labels, and must be regenerated when the dataset or split-manifest hash changes.

---

## 6. Batch-size tuning for the RTX 4050 6 GB GPU

The reported batch size of 16 is only a starting point. EfficientNetB3 at 300×300 may fit a larger batch during frozen-backbone Stage 1 because fewer gradients and activations are stored.

Benchmark Stage 1 with batch sizes:

```text
16, 24, 32, 40, 48
```

Stop increasing when one of these occurs:

- CUDA out-of-memory error.
- Severe throughput degradation.
- Unacceptable thermal throttling.
- GPU memory reaches an unsafe level.

Use the largest stable batch that improves images/second. A larger batch is not automatically better for convergence. Keep the effective batch size and learning-rate policy documented.

For Stage 2, the backbone is trainable and memory use increases. Benchmark separately. If the desired effective batch is larger than the physical batch, use gradient accumulation only if the Keras/PyTorch backend implementation is verified and the accumulated gradients are correct.

Do not use a smaller batch solely because it is the historical setting.

---

## 7. Mixed precision

The RTX 4050 supports modern mixed-precision acceleration. Benchmark mixed precision for training after baseline correctness is established.

For the PyTorch backend, use the Keras 3 mixed-precision API supported by the installed version, or a verified PyTorch autocast implementation. Do not copy TensorFlow mixed-precision code into a PyTorch-backend run without testing.

The intended pattern is conceptually:

```python
import keras

keras.mixed_precision.set_global_policy("mixed_float16")
```

However, the agent must verify the installed Keras backend supports this policy correctly. The final classifier output and numerically sensitive losses may need float32 behavior. Inspect model outputs and loss values for NaN or Inf.

Required mixed-precision checks:

- Compare one epoch against float32.
- Check for NaN or Inf loss.
- Compare validation metrics within an acceptable tolerance.
- Measure images/second.
- Measure GPU memory.
- Confirm saved checkpoints reload correctly.

Use mixed precision only if it produces a measurable speed or memory benefit without changing model behavior unexpectedly.

---

## 8. Avoid expensive or incorrect training operations

The training script must not do any of the following inside every batch or unnecessarily every epoch:

- Generate Grad-CAM maps.
- Save prediction images.
- Recompute duplicate detection.
- Rebuild the model.
- Reload ImageNet weights.
- Recreate augmentation layers.
- Compute full-dataset metrics in Python.
- Print large arrays or prediction tables.
- Save checkpoints more frequently than needed.
- Run expensive callbacks on every batch.

Recommended callbacks:

- Checkpoint only the best validation metric or once per epoch.
- Early stopping with a documented patience.
- Reduce learning rate on plateau or a predefined schedule.
- CSV logging once per epoch.

For debugging, temporarily disable validation to measure pure training throughput. Then restore validation for real experiments.

---

## 9. Stage 1 and Stage 2 optimization recommendations

### Stage 1: frozen backbone

Stage 1 should be the faster stage because only the classification head is trainable. Recommended procedure:

1. Build the model once.
2. Load ImageNet weights once.
3. Freeze the backbone.
4. Compile once.
5. Use the largest stable batch size that fits.
6. Use cached/preprocessed data and prefetching.
7. Use mixed precision after verification.
8. Run validation once per epoch.

If Stage 1 remains very slow, suspect the data pipeline or CPU execution before changing the learning rate or architecture.

### Stage 2: fine-tuning

Stage 2 will be slower because the upper backbone is trainable. Recommended procedure:

1. Unfreeze meaningful final EfficientNet blocks, not an arbitrary layer count.
2. Keep BatchNorm frozen initially.
3. Recompile once after changing trainability.
4. Use a smaller learning rate.
5. Reduce augmentation only if it is demonstrably a throughput bottleneck.
6. Keep the same cached decode path.
7. Benchmark images/second separately from Stage 1.

Do not expect Stage 2 to have the same speed as Stage 1.

---

## 10. Windows-specific requirements

The user has an NVIDIA RTX 4050 Laptop GPU with approximately 6 GB VRAM and is working on Windows.

Check the following before training:

- Laptop is connected to AC power.
- Windows power mode is set to Best performance.
- NVIDIA Control Panel assigns the Python interpreter to the high-performance NVIDIA GPU.
- The correct CUDA-enabled PyTorch package is installed.
- The dataset is on a local SSD.
- Windows Defender or cloud synchronization is not scanning/synchronizing every image read.
- No other process is consuming GPU memory.
- GPU temperature and clocks remain stable during training.

If native Windows Keras PyTorch training is fast but the TFLite conversion path is unreliable, use WSL2 with CUDA for the final TensorFlow-to-TFLite conversion pipeline rather than weakening deployment compatibility.

---

## 11. Correctness checks after optimization

A faster run is invalid if it changes the data or model behavior accidentally. After every optimization, verify:

1. Same split manifest hash.
2. Same class order.
3. Same preprocessing and image color order.
4. Same labels and class weights.
5. Same loss definition.
6. Same evaluation code.
7. Same checkpoint loading behavior.
8. No data leakage from caching.
9. No validation images in the training cache.
10. No NaN or Inf values.

Compare the optimized and baseline runs on a fixed evaluation subset. A small difference caused by mixed precision or nondeterministic GPU kernels is acceptable only if documented.

---

## 12. Required speed targets and reporting

Do not use a universal time target because hardware, batch size, preprocessing, augmentation, validation, and backend overhead differ. Report measured throughput instead. In particular, do not promise that RAM ingestion will reduce an epoch to 15–20 seconds before benchmarking the actual model, batch size, dataset size, validation work, and target machine.

### 12.1 RAM-ingestion acceptance test

Compare the original disk-backed pipeline and the RAM pipeline using the same model initialization, split, batch size, augmentation policy, validation frequency, and number of steps. Warm up both runs and report steady-state values. Accept the RAM optimization only if it improves throughput without changing evaluation semantics.

The comparison must include:

| Pipeline | Startup ingestion | Train seconds/epoch | Validation seconds/epoch | Images/second | GPU utilization | Peak RAM | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Disk-backed baseline | measured | measured | measured | measured | measured | measured | measured |
| RAM-ingested | measured | measured | measured | measured | measured | measured | measured |

The agent must also verify that the RAM pipeline produces the same predictions as the baseline on a fixed batch before reporting the speedup. A speedup caused by accidentally skipping augmentation, validation, class weights, or samples is invalid.

Every benchmark must report:

| Measurement | Required value |
|---|---|
| Backend | Keras backend and version |
| Python | Version |
| PyTorch | Version and CUDA build |
| GPU | Exact device name |
| Input size | For example, 300×300×3 |
| Batch size | Physical and effective batch size |
| Precision | Float32 or mixed precision |
| Workers | Number and persistence setting |
| Cache | None, file cache, or RAM cache |
| Prefetch | Configuration |
| Trainable parameters | Count |
| Images/second | Steady-state value |
| Seconds/epoch | Training and validation separately |
| GPU utilization | Typical and peak |
| GPU memory | Typical and peak |
| CPU utilization | Typical value |
| Validation frequency | Once per epoch or other |
| Metric result | To detect accidental behavior changes |

The optimization is successful when the bottleneck is identified, GPU utilization is materially improved or the GPU is no longer starved by input loading, images/second increases, and evaluation behavior remains correct. The result must be reported as a measured speedup, not as a guaranteed 15–20 second epoch.

---

## 13. Required implementation order

The coding agent must implement the following order and stop if an earlier gate fails:

### Step 1: environment verification

Prove CUDA availability and actual model placement on the RTX 4050.

### Step 2: throughput benchmark

Run synthetic, cached, and disk-backed benchmarks. Identify whether compute or input loading is limiting speed.

### Step 3: pipeline optimization

Move the dataset to local SSD, add parallel loading, cache/preprocess where safe, and add prefetching.

### Step 4: batch-size benchmark

Find the largest stable batch for Stage 1 and Stage 2 separately.

### Step 5: mixed-precision benchmark

Test mixed precision and verify numerical correctness.

### Step 6: remove unnecessary work

Disable Grad-CAM and expensive callbacks during training. Run diagnostics after training.

### Step 7: correctness comparison

Compare the optimized configuration against the baseline on the fixed validation subset.

### Step 8: final training

Only after the above steps pass, run the complete teacher training experiment.

---

## 14. Stop conditions

Stop and report instead of continuing if:

- CUDA is unavailable.
- The model is on CPU.
- The TFLite conversion path is unverified.
- Image counts or split manifests change unexpectedly.
- The optimized pipeline produces different labels or class order.
- Loss becomes NaN or Inf.
- GPU memory repeatedly overflows.
- The GPU is underutilized because of an unresolved data bottleneck.
- The reported speed improvement comes from skipping validation or changing the dataset without documenting it.

---

## 15. Final instruction to the coding agent

The current 20–25 minutes for 3 epochs must be treated as a performance incident. Diagnose it before changing architecture. The most likely explanation is CPU execution or an inefficient disk/Python image pipeline, not that EfficientNetB3 inherently requires this much time for Stage 1.

The agent must prove:

1. CUDA is active.
2. The model and batches are on the GPU.
3. The GPU is not waiting on image loading.
4. Batch size is appropriate for the 6 GB RTX 4050.
5. Mixed precision has been tested safely.
6. No expensive diagnostics run inside training.
7. The optimized run preserves the original data, preprocessing, labels, and evaluation semantics.

Only then should the agent claim that training has been accelerated.

---

## References

[1]: https://keras.io/api/mixed_precision/ "Keras mixed precision API"
[2]: https://pytorch.org/get-started/locally/ "PyTorch official installation guide"
[3]: https://pytorch.org/docs/stable/notes/cuda.html "PyTorch CUDA semantics"
[4]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[5]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"

Author: **Manus AI**

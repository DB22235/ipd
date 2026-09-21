"""
diagnose_gpu_environment.py
===========================
Step 1 Diagnostic: Hardware & Environment Verification.
Records CUDA availability, PyTorch build, Keras backend, and proves that
model parameters and tensors are physically allocated on the RTX 4050 GPU.
Saves output to reports/gpu_environment.txt.
"""

import os
import sys
import subprocess
from pathlib import Path

# Enforce silent TF logs and Keras 3 PyTorch CUDA backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = REPORTS_DIR / "gpu_environment.txt"


def main():
    lines = []
    lines.append("=" * 80)
    lines.append("       STEP 1: HARDWARE & ENVIRONMENT DIAGNOSTIC AUDIT")
    lines.append("=" * 80)

    # 1. System & Python
    lines.append(f"Python Executable : {sys.executable}")
    lines.append(f"Python Version    : {sys.version.split()[0]}")
    lines.append(f"Keras Version     : {keras.__version__}")
    lines.append(f"Keras Backend     : {keras.config.backend()}")
    lines.append(f"PyTorch Version   : {torch.__version__}")
    lines.append(f"CUDA Available    : {torch.cuda.is_available()}")
    lines.append(f"Torch CUDA Build  : {torch.version.cuda}")

    # 2. CUDA Assertions
    if not torch.cuda.is_available():
        msg = "❌ FATAL: CUDA is unavailable. Stop before training; run would execute on CPU."
        lines.append(msg)
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("\n".join(lines))
        raise RuntimeError(msg)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)

    lines.append(f"GPU Device Name   : {gpu_name}")
    lines.append(f"GPU Device Count  : {gpu_count}")
    lines.append(f"Total VRAM        : {vram_total_gb:.2f} GB")
    lines.append(f"Compute Capability: {capability}")

    # 3. Model Weight Placement Assertion
    lines.append("\nTesting Model Parameter Placement on CUDA...")
    test_model = keras.Sequential([
        keras.layers.Input((300, 300, 3)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3)
    ])
    
    # Assert weight device
    sample_weight = test_model.weights[0]
    weight_device = sample_weight.value.device
    lines.append(f"Sample Weight Name  : {sample_weight.name}")
    lines.append(f"Sample Weight Device: {weight_device}")

    assert weight_device.type == "cuda", f"Weight is not on CUDA! Found: {weight_device}"
    lines.append("✓ ASSERTION PASSED: Model weights are physically allocated on CUDA.")

    # 4. Forward Pass Tensor Placement Assertion
    x_test = torch.randn(2, 300, 300, 3).cuda()
    y_test = test_model(x_test)
    lines.append(f"Input Tensor Device : {x_test.device}")
    lines.append(f"Output Tensor Device: {y_test.device}")
    assert y_test.device.type == "cuda", f"Output is not on CUDA! Found: {y_test.device}"
    lines.append("✓ ASSERTION PASSED: Forward computation executes directly on CUDA.")

    # 5. nvidia-smi Query
    lines.append("\n" + "-" * 80)
    lines.append("NVIDIA-SMI System Output:")
    lines.append("-" * 80)
    try:
        smi_out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        lines.append(smi_out.strip())
    except Exception as e:
        lines.append(f"nvidia-smi query failed: {e}")

    lines.append("=" * 80)
    content = "\n".join(lines)
    print(content)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✓ Diagnostic report saved to: {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

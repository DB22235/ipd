"""
scripts/rice_student/train_student_baseline.py
==============================================
Stage 1 Supervised Baseline Training for MobileNetV3-Large.
Trains the student model using ground-truth labels only, establishing the
un-distilled mobile performance baseline.
Requires explicit user authorization before execution.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import yaml
import numpy as np

# Force Torch backend for NVIDIA GPU acceleration
os.environ["KERAS_BACKEND"] = "torch"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keras
from keras import optimizers, losses, callbacks
import torch

from src.rice_student.contracts import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
)
from src.rice_student.data import create_rice_dataloaders
from src.rice_student.models import build_mobilenetv3_student, get_model_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Rice Student Supervised Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rice/student_baseline.yaml",
        help="Path to YAML training configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = ROOT_DIR / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 75)
    print("      STAGE 1: RICE STUDENT (MOBILENETV3-LARGE) SUPERVISED BASELINE")
    print("=" * 75)
    print(f"  Configuration : {config_path.relative_to(ROOT_DIR)}")
    print(f"  Backend       : Keras 3 with PyTorch ({keras.backend.backend()})")
    print(f"  CUDA Active   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Target Device : {torch.cuda.get_device_name(0)}")

    output_dir = ROOT_DIR / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / cfg["checkpoint_name"]

    reports_dir = ROOT_DIR / cfg["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Pipeline
    print(f"\n[1/4] Ingesting Rice Manifest Data from {cfg['split_manifest']}...")
    batch_size = int(cfg.get("batch_size", 16))
    target_size = tuple(cfg.get("input_size", [224, 224]))
    use_weights = bool(cfg.get("use_class_weights", True))

    train_loader, val_loader, test_loader, class_weights = create_rice_dataloaders(
        manifest_path=ROOT_DIR / cfg["split_manifest"],
        batch_size=batch_size,
        target_size=target_size,
        use_class_weights=use_weights,
        root_dir=ROOT_DIR,
    )
    print(f"      Train Batches : {len(train_loader)} | Val Batches: {len(val_loader)} | Test Batches: {len(test_loader)}")
    if use_weights:
        print(f"      Balanced Inverse Weights: {[round(class_weights.get(i, 1.0), 3) for i in range(NUM_CLASSES)]}")

    # 2. Build Student Model
    print("\n[2/4] Constructing MobileNetV3-Large Student Model...")
    student_model, backbone = build_mobilenetv3_student(
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE_STUDENT,
        dropout_rate=float(cfg.get("dropout_rate", 0.25)),
        weights="imagenet",
        include_augmentation=True,
    )
    stats = get_model_statistics(student_model)
    print(f"      Total Parameters      : {stats['total_params']:,}")
    print(f"      Trainable Parameters  : {stats['trainable_params']:,}")

    # 3. Compile Model
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    optimizer = optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    
    student_model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # 4. Training Callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=int(cfg.get("early_stopping_patience", 6)),
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 5. Execute Training Loop
    max_epochs = int(cfg.get("max_epochs", 30))
    print(f"\n[3/4] Launching Supervised Training (Max Epochs: {max_epochs}, Patience: {cfg.get('early_stopping_patience', 6)})...")
    t0 = time.time()
    history = student_model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=max_epochs,
        callbacks=cb_list,
        verbose=1,
    )
    t_elapsed = time.time() - t0
    print(f"\n[OK] Training completed in {t_elapsed / 60:.1f} minutes.")
    print(f"     Best model saved to: {checkpoint_path.relative_to(ROOT_DIR)}")

    # 6. Save Training History
    history_file = reports_dir / "baseline_training_metrics.json"
    metrics_record = {
        "model_role": "student_baseline",
        "training_time_seconds": round(t_elapsed, 1),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)),
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"[PASS] Training metrics saved to: {history_file.relative_to(ROOT_DIR)}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

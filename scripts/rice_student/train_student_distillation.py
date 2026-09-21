"""
scripts/rice_student/train_student_distillation.py
==================================================
Stage 2 Knowledge Distillation Training for MobileNetV3-Large.
Distills dark knowledge from frozen EfficientNetB3 teacher into the mobile student.
Requires explicit user authorization before execution.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import yaml

# Force Torch backend for NVIDIA GPU acceleration
os.environ["KERAS_BACKEND"] = "torch"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keras
from keras import optimizers, callbacks
import torch

from src.rice_student.contracts import (
    CLASSES,
    NUM_CLASSES,
    INPUT_SHAPE_STUDENT,
    compute_file_sha256,
)
from src.rice_student.data import create_rice_dataloaders
from src.rice_student.models import build_mobilenetv3_student
from src.rice_student.distiller import RiceDistiller


class SaveStudentCheckpoint(callbacks.Callback):
    """Saves the student model weights when validation accuracy improves (or loss improves on tie)."""
    def __init__(self, filepath: Path, monitor: str = "val_accuracy"):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.best_score = -float("inf")
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_acc = logs.get(self.monitor, 0.0)
        current_loss = logs.get("val_loss", float("inf"))
        improved = False
        if current_acc > self.best_score:
            improved = True
        elif current_acc == self.best_score and current_loss < self.best_loss:
            improved = True

        if improved:
            print(f"\nEpoch {epoch + 1}: {self.monitor} is {current_acc:.4f} (val_loss improved to {current_loss:.4f}), saving student to {self.filepath.name}...")
            self.best_score = current_acc
            self.best_loss = current_loss
            self.model.student.save(str(self.filepath))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Rice Student via Knowledge Distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rice/student_distillation.yaml",
        help="Path to YAML distillation configuration file",
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
    print("      STAGE 2: RICE STUDENT (MOBILENETV3-LARGE) KNOWLEDGE DISTILLATION")
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

    # 1. Verify and Load Frozen Teacher
    teacher_file = ROOT_DIR / cfg["teacher_model_path"]
    if not teacher_file.exists():
        raise FileNotFoundError(f"Teacher model not found: {teacher_file}")
    
    sha_pre = compute_file_sha256(teacher_file)
    expected_sha = cfg.get("teacher_sha256")
    if expected_sha and sha_pre != expected_sha:
        raise AssertionError(f"Teacher SHA-256 mismatch before training! Expected {expected_sha}, got {sha_pre}")

    print(f"\n[1/5] Ingesting Frozen Teacher: {teacher_file.name}...")
    teacher_model = keras.models.load_model(str(teacher_file), compile=False)
    teacher_model.trainable = False
    assert len(teacher_model.trainable_weights) == 0, "Teacher model must have 0 trainable weights!"
    print(f"      Teacher Input Shape   : {teacher_model.input_shape}")
    print(f"      Teacher Output Shape  : {teacher_model.output_shape}")
    print(f"      Teacher Verified SHA  : {sha_pre[:16]}... (OK)")

    # 2. Data Pipeline
    print(f"\n[2/5] Ingesting Rice Manifest Data from {cfg['split_manifest']}...")
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
    print(f"      Train Batches : {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 3. Construct Student & Distiller
    print("\n[3/5] Building Student Model & Initializing Distillation Engine...")
    student_model, _ = build_mobilenetv3_student(
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE_STUDENT,
        dropout_rate=float(cfg.get("dropout_rate", 0.25)),
        weights="imagenet",
        include_augmentation=True,
    )

    temperature = float(cfg.get("temperature", 3.0))
    alpha = float(cfg.get("alpha", 0.5))
    teacher_in_size = tuple(cfg.get("teacher_input_size", [300, 300]))

    distiller = RiceDistiller(
        student=student_model,
        teacher=teacher_model,
        temperature=temperature,
        alpha=alpha,
        teacher_input_size=teacher_in_size,
    )
    print(f"      Distillation Setup    : Temperature={temperature}, Alpha={alpha}")
    print(f"      Resolution Upsample   : Student (224x224) -> Teacher ({teacher_in_size[0]}x{teacher_in_size[1]})")

    # 4. Compile Distiller
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    distiller.compile(optimizer=optimizers.AdamW(learning_rate=lr, weight_decay=wd))

    # 5. Callbacks
    cb_list = [
        SaveStudentCheckpoint(filepath=checkpoint_path, monitor="val_accuracy"),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=int(cfg.get("early_stopping_patience", 6)),
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]

    # 6. Execute Distillation
    max_epochs = int(cfg.get("max_epochs", 30))
    print(f"\n[4/5] Launching Distillation Loop (Max Epochs: {max_epochs})...")
    t0 = time.time()
    history = distiller.fit(
        train_loader,
        validation_data=val_loader,
        epochs=max_epochs,
        callbacks=cb_list,
        verbose=1,
    )
    t_elapsed = time.time() - t0

    # 7. Post-Training Teacher Invariance Verification
    sha_post = compute_file_sha256(teacher_file)
    assert sha_pre == sha_post, "CRITICAL ERROR: Teacher weights were modified during distillation!"

    print(f"\n[5/5] Distillation Finished in {t_elapsed / 60:.1f} minutes.")
    print(f"      Teacher Invariance Checked: {sha_post[:16]}... (0% mutation)")
    print(f"      Distilled Student Saved   : {checkpoint_path.relative_to(ROOT_DIR)}")

    # Save History
    history_file = reports_dir / "distillation_training_metrics.json"
    metrics_record = {
        "model_role": "student_distilled",
        "training_time_seconds": round(t_elapsed, 1),
        "temperature": temperature,
        "alpha": alpha,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT_DIR)),
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"[PASS] Distillation metrics written to: {history_file.relative_to(ROOT_DIR)}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

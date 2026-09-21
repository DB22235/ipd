"""
Fast Fine-Tuning Pipeline for Tomato Teacher Model
===================================================
Hardens the Tomato EfficientNetB3 Teacher Classifier against the
Studio-to-Field Domain Gap (specifically the white-background / stem shortcut)
via Background Destruction Augmentation and targeted fine-tuning.

Executes in 3 to 5 epochs (~10-15 mins).
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.dataset import load_crop_datasets
from src.background_augmentation import tf_randomize_background

# ---------------------------------------------------------------------------
# Paths and Parameters
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "clean_dataset" / "tomato_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "tomato_teacher"
MODEL_PATH = MODEL_DIR / "tomato_teacher_efficientnetb3.keras"
BACKUP_PATH = MODEL_DIR / "tomato_teacher_original_backup.keras"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
FINETUNE_EPOCHS = 4
FINETUNE_LR = 3e-5
UNFREEZE_LAYERS = 30


def main():
    start_time = time.time()
    print("=" * 72)
    print("   TOMATO TEACHER MODEL - FIELD-ROBUST DOMAIN HARDENING (FINE-TUNING)")
    print("=" * 72)

    # 1. Load Datasets
    print("\n[Step 1/5] Loading tomato clean dataset...")
    train_ds, val_ds, test_ds, class_names, class_weights = load_crop_datasets(
        dataset_root=DATASET_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42
    )
    print(f"Detected {len(class_names)} classes: {class_names}")
    print(f"Balanced class weights: {class_weights}")

    # Inject Background Destruction Augmentation into train pipeline
    print("\n[Step 1b/5] Injecting background destruction augmentation into training pipeline...")
    train_ds_aug = train_ds.map(
        tf_randomize_background,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Ensure Backup Exists
    if not BACKUP_PATH.exists():
        import shutil
        shutil.copyfile(MODEL_PATH, BACKUP_PATH)
        print(f"Created safety backup of original model -> {BACKUP_PATH.name}")
    else:
        print(f"Verified existing safety backup -> {BACKUP_PATH.name}")

    # 3. Load Model
    print(f"\n[Step 2/5] Loading existing checkpoint from {MODEL_PATH.name}...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

    # Unfreeze only top 30 layers for surgical domain gap correction
    try:
        backbone = model.get_layer("efficientnetb3")
        backbone.trainable = True
        for layer in backbone.layers[:-UNFREEZE_LAYERS]:
            layer.trainable = False
        print(f"Unfrozen top {UNFREEZE_LAYERS} layers of {backbone.name} (surgical fine-tuning).")
    except Exception as e:
        print(f"Note on backbone configuration: {e}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Model compiled with Adam(lr={FINETUNE_LR}). Trainable parameters: {trainable_params:,}")

    # 4. Fine-Tuning Loop
    print(f"\n[Step 3/5] Starting fine-tuning for up to {FINETUNE_EPOCHS} epochs...")
    best_ckpt_path = MODEL_DIR / "tomato_finetune_best.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_ckpt_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=1
    )

    # 5. Load best weights and evaluate on test set
    print("\n[Step 4/5] Evaluating fine-tuned model on test partition...")
    if best_ckpt_path.exists():
        model = tf.keras.models.load_model(str(best_ckpt_path), compile=False)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        print("Loaded best checkpoint from fine-tuning.")

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n>>> Test Partition Results: Loss = {test_loss:.4f}, Accuracy = {test_acc*100:.2f}% <<<")

    # Collect predictions for detailed metrics
    y_true = []
    y_pred_probs = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        probs = tf.nn.softmax(preds).numpy()
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report (Tomato Test Set After Hardening):")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"{'':<15} " + " ".join([f"{c:>12}" for c in class_names]))
    for idx, row in enumerate(cm):
        print(f"{class_names[idx]:<15} " + " ".join([f"{val:>12}" for val in row]))

    # Save final hardened model
    print(f"\n[Step 5/5] Saving field-hardened model to {MODEL_PATH.name}...")
    model.save(str(MODEL_PATH))

    # Save training curves plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
    plt.title('Tomato Fine-Tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='s')
    plt.title('Tomato Fine-Tuning Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = MODEL_DIR / "tomato_finetune_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved fine-tune curves to {plot_path.name}")

    # Update manifest
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["model_version"] = "2.1.0-field-hardened"
        manifest_data["field_hardening"] = {
            "unfreeze_layers": UNFREEZE_LAYERS,
            "fine_tune_lr": FINETUNE_LR,
            "background_destruction_augmented": True,
            "test_accuracy": float(test_acc)
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"Updated model manifest -> {MANIFEST_PATH.name}")

    elapsed = time.time() - start_time
    print(f"\n{'='*72}")
    print(f"TOMATO HARDENING COMPLETE IN {elapsed/60:.1f} MINUTES!")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

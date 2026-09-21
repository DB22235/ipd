"""
Fast Fine-Tuning Pipeline for Potato Teacher Model
===================================================
Corrects the inverted label representations (healthy vs late_blight)
by fine-tuning the existing checkpoint on the cleansed dataset.

Executes in 3 to 4 epochs (~10-15 mins).
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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.dataset import load_crop_datasets

# ---------------------------------------------------------------------------
# Paths and Parameters
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "clean_dataset" / "potato_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "potato_teacher"
MODEL_PATH = MODEL_DIR / "potato_teacher_efficientnetb3.keras"
BACKUP_PATH = MODEL_DIR / "potato_teacher_inverted_backup.keras"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
FINETUNE_EPOCHS = 4
FINETUNE_LR = 1e-4


def main():
    start_time = time.time()
    print("=" * 70)
    print("POTATO TEACHER MODEL - FAST FINE-TUNING ON CORRECTED DATA")
    print("=" * 70)

    # 1. Load Corrected Datasets
    print("\n[Step 1/5] Loading cleansed potato datasets...")
    train_ds, val_ds, test_ds, class_names, class_weights = load_crop_datasets(
        dataset_root=DATASET_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42
    )
    print(f"Detected {len(class_names)} classes: {class_names}")
    print(f"Balanced class weights: {class_weights}")

    # 2. Ensure Backup Exists
    if not BACKUP_PATH.exists():
        import shutil
        shutil.copyfile(MODEL_PATH, BACKUP_PATH)
        print(f"Created backup of original model -> {BACKUP_PATH.name}")
    else:
        print(f"Verified existing backup -> {BACKUP_PATH.name}")

    # 3. Load Model
    print(f"\n[Step 2/5] Loading existing checkpoint from {MODEL_PATH.name}...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # Ensure top layers and head are trainable
    try:
        backbone = model.get_layer("efficientnetb3")
        backbone.trainable = True
        # Keep early 340 layers frozen, fine-tune top 45 layers
        for layer in backbone.layers[:-45]:
            layer.trainable = False
        print(f"Unfrozen top 45 layers of {backbone.name}.")
    except Exception as e:
        print(f"Note on backbone configuration: {e}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Model compiled with Adam(lr={FINETUNE_LR}). Trainable parameters: {trainable_params:,}")

    # 4. Fast Fine-Tuning Loop
    print(f"\n[Step 3/5] Starting fast fine-tuning for {FINETUNE_EPOCHS} epochs...")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "potato_finetune_best.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb],
        verbose=1
    )

    # 5. Load best weights and evaluate on test set
    print("\n[Step 4/5] Evaluating fine-tuned model on test partition...")
    best_finetune_path = MODEL_DIR / "potato_finetune_best.keras"
    if best_finetune_path.exists():
        model = tf.keras.models.load_model(str(best_finetune_path))
        print("Loaded best checkpoint from fine-tuning.")

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n>>> Test Partition Results: Loss = {test_loss:.4f}, Accuracy = {test_acc*100:.2f}% <<<")

    # Collect predictions for detailed metrics
    y_true = []
    y_pred_probs = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        # Apply softmax to logits
        probs = tf.nn.softmax(preds).numpy()
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report (Corrected Potato Test Set):")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"{'':<15} " + " ".join([f"{c:>12}" for c in class_names]))
    for idx, row in enumerate(cm):
        print(f"{class_names[idx]:<15} " + " ".join([f"{val:>12}" for val in row]))

    # Save final model
    print(f"\n[Step 5/5] Saving fine-tuned teacher model to {MODEL_PATH.name}...")
    model.save(str(MODEL_PATH))

    # Save training curves plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
    plt.title('Fine-Tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='s')
    plt.title('Fine-Tuning Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = MODEL_DIR / "finetune_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved fine-tune curves to {plot_path.name}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"FINE-TUNING COMPLETE IN {elapsed/60:.1f} MINUTES!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

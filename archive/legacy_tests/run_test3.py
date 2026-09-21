"""
run_test4.py
------------
Runs inference on test4.png using the Potato Teacher EfficientNetB3 model.
Includes a compatibility patch for the VarianceScaling initializer mismatch
between Keras versions (model saved with newer Keras 3.x, running on local Keras 3.14).

Usage:
    python run_test3.py
"""

# ============================================================
# STEP 0 — Keras Compatibility Patch
# The saved model uses VarianceScaling with 'input_axes' and
# 'output_axes' args introduced in a newer Keras build.
# We monkey-patch the class to silently accept those kwargs.
# ============================================================
import keras.src.initializers.random_initializers as _ri

_OriginalVarianceScaling = _ri.VarianceScaling


class _CompatVarianceScaling(_OriginalVarianceScaling):
    """Drop-in replacement that ignores extra kwargs from newer Keras saves."""

    def __init__(self, *args, input_axes=None, output_axes=None, **kwargs):
        super().__init__(*args, **kwargs)


_ri.VarianceScaling = _CompatVarianceScaling

import keras as _keras
_keras.initializers.VarianceScaling = _CompatVarianceScaling

# ============================================================
# STEP 1 — Imports
# ============================================================
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # headless rendering, saves to file (no display window needed)
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 2 — Paths
# ============================================================
ROOT_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = ROOT_DIR / "models" / "potato_teacher" / "potato_teacher_efficientnetb3.keras"
MANIFEST_PATH = ROOT_DIR / "models" / "potato_teacher" / "model_manifest.json"
IMAGE_PATH    = ROOT_DIR / "test7_r.webp"
OUTPUT_PATH   = ROOT_DIR / "test7_r_gradcam.png"

IMG_SIZE      = (300, 300)
LAST_CONV_LAYER = "top_conv"   # EfficientNetB3 final conv layer for Grad-CAM

# ============================================================
# STEP 3 — Load Class Names from Manifest
# ============================================================
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

CLASS_NAMES = manifest["class_names"]
print(f"Loaded class names: {CLASS_NAMES}")

# ============================================================
# STEP 4 — Load Model
# ============================================================
print(f"\nLoading model: {MODEL_PATH.name} ...")
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
print("Model loaded successfully.")
print(f"  Input shape : {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# ============================================================
# STEP 5 — Preprocess Image
# ============================================================
img       = tf.keras.utils.load_img(str(IMAGE_PATH), target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)   # shape (300, 300, 3), float32 [0..255]
img_batch = np.expand_dims(img_array, axis=0)  # shape (1, 300, 300, 3)

# ============================================================
# STEP 6 — Forward Pass & Prediction
# ============================================================
logits     = model(img_batch, training=False)               # raw logits
probs      = tf.nn.softmax(logits).numpy()[0]               # probabilities
pred_idx   = int(np.argmax(probs))
pred_class = CLASS_NAMES[pred_idx]
pred_conf  = float(probs[pred_idx]) * 100.0

# ============================================================
# STEP 7 — Diagnostic Report
# ============================================================
print()
print("=" * 62)
print("        POTATO LEAF DISEASE DIAGNOSTIC REPORT")
print("=" * 62)
print(f"  Image Tested       : {IMAGE_PATH.name}")
print(f"  Ground Truth       : LATE_BLIGHT  (as labeled by user)")
print(f"  Primary Diagnosis  : {pred_class.upper()}")
print(f"  Confidence Score   : {pred_conf:.2f}%")
print("-" * 62)
print(f"  {'Class':<22} {'Probability':>12}   {'Bar':}")
print("-" * 62)
for i, name in enumerate(CLASS_NAMES):
    p      = float(probs[i])
    bar    = "#" * int(p * 38)
    marker = "  <-- PREDICTED" if name == pred_class else ""
    print(f"  {name:<22} {p * 100:>10.2f}%   {bar}{marker}")
print("-" * 62)

if pred_conf >= 80.0:
    assessment = "HIGH CONFIDENCE  (Decisive prediction)"
elif pred_conf >= 60.0:
    assessment = "MODERATE CONFIDENCE  (Acceptable)"
else:
    assessment = "LOW CONFIDENCE  (Review recommended)"

correct = (pred_class == "late_blight")
verdict = "CORRECT" if correct else "INCORRECT"

print(f"  Assessment         : {assessment}")
print(f"  Verdict vs Label   : {verdict}")
print("=" * 62)

# ============================================================
# STEP 8 — Grad-CAM Saliency Map
# ============================================================
print("\nGenerating Grad-CAM saliency map ...")

try:
    # Locate EfficientNetB3 backbone inside the model
    backbone  = model.get_layer("efficientnetb3")
    last_conv = backbone.get_layer(LAST_CONV_LAYER)

    # Sub-model that returns [conv_outputs, backbone_features]
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv.output, backbone.output]
    )

    # Reconstruct classifier head as a separate model
    backbone_idx = next(
        i for i, layer in enumerate(model.layers)
        if layer.name == backbone.name
    )
    head_input = tf.keras.Input(shape=backbone.output.shape[1:])
    x = head_input
    for layer in model.layers[backbone_idx + 1:]:
        x = layer(x)
    head_model = tf.keras.Model(inputs=head_input, outputs=x)

    # Compute gradients of predicted class score w.r.t. last conv feature map
    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = backbone_grad_model(img_batch, training=False)
        tape.watch(conv_outputs)
        predictions  = head_model(backbone_features, training=False)
        class_score  = predictions[:, pred_idx]

    grads        = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of feature maps
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)                     # ReLU
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val                      # Normalize to [0,1]
    heatmap = heatmap.numpy()

    # Resize heatmap to original image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMG_SIZE
    ).numpy().squeeze()

    # Colorize and overlay on original image
    colormap     = plt.get_cmap("jet")
    heatmap_rgb  = colormap(heatmap_resized)[:, :, :3]
    orig_norm    = img_array / 255.0
    overlay      = np.clip(0.6 * orig_norm + 0.4 * heatmap_rgb, 0, 1)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(orig_norm)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap\n(Red = High Activation)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Prediction: {pred_class.upper()}\nConfidence: {pred_conf:.2f}%",
        fontsize=12
    )
    axes[2].axis("off")

    plt.suptitle(
        f"Potato Teacher EfficientNetB3 -- Saliency Audit\n"
        f"Diagnosed: {pred_class.upper()}  ({pred_conf:.2f}%)   |   Ground Truth: LATE_BLIGHT   |   Verdict: {verdict}",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(str(OUTPUT_PATH), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM visualization saved to: {OUTPUT_PATH.name}")

except Exception as e:
    print(f"Grad-CAM generation failed: {e}")

print("\nInference complete.")

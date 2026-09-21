"""
run_inference.py
================
Potato Disease Field-Robust Inference Pipeline.
Combines:
  1. Lightweight Leaf Isolation (OpenCV GrabCut / Plant Saliency)
  2. Test-Time Augmentation (TTA) using parentmodel.ipynb's field-robust inward zoom
  3. Multi-class Aggregation (checks Early Blight, Healthy, Late Blight equally)
  4. Grad-CAM Saliency Map & 4-Panel Diagnostic Audit

Usage:
  python run_inference.py test5.png --ground-truth early_blight
  python run_inference.py test6_r.jpg --ground-truth late_blight
  python run_inference.py test7_r.webp --ground-truth late_blight
  python run_inference.py potatotest.png
"""

import os
import sys
from pathlib import Path

# Silence noisy TF / oneDNN terminal warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# STEP 0 — Keras Compatibility Patch
# The saved model uses VarianceScaling with 'input_axes' and
# 'output_axes' kwargs from newer Keras builds.
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
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # Headless file-only rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import local leaf isolator
from leaf_isolator import isolate_leaf

# ============================================================
# STEP 2 — Configuration & Path Resolution
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
POT_MODEL_NEW = ROOT_DIR / "models" / "potato" / "teachers" / "potato_teacher_efficientnetb3.keras"
MODEL_PATH = POT_MODEL_NEW if POT_MODEL_NEW.exists() else ROOT_DIR / "models" / "potato_teacher" / "potato_teacher_efficientnetb3.keras"
POT_MAN_NEW = ROOT_DIR / "models" / "potato" / "teachers" / "model_manifest.json"
MANIFEST_PATH = POT_MAN_NEW if POT_MAN_NEW.exists() else ROOT_DIR / "models" / "potato_teacher" / "model_manifest.json"

IMG_SIZE = (300, 300)
LAST_CONV_LAYER = "top_conv"


def parse_args():
    parser = argparse.ArgumentParser(description="Potato Leaf Disease Field-Robust Inference")
    parser.add_argument("image_path", nargs="?", default="test7_r.webp", help="Path to potato leaf image")
    parser.add_argument("--ground-truth", "-g", default=None, choices=["early_blight", "healthy", "late_blight"],
                        help="Optional ground truth label for accuracy verification")
    parser.add_argument("--tta", "-t", type=int, default=10, help="Number of TTA passes (default: 10)")
    parser.add_argument("--no-isolate", action="store_true", help="Bypass leaf isolation and test raw full image")
    parser.add_argument("--apply-mask", action="store_true", help="Apply black background mask (not recommended)")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model file (auto-detects crop if omitted)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output path for Grad-CAM audit plot")
    return parser.parse_args()


# ============================================================
# STEP 3 — Load Model & Augmentation Pipeline
# ============================================================
def load_disease_model(model_file: Path):
    """Loads Keras teacher model and extracts field_robust_augmentation layer."""
    print(f"\n[1/5] Loading Model: {model_file.name} ...")
    model = tf.keras.models.load_model(str(model_file), compile=False)
    print("      Model loaded successfully.")
    print(f"      Input shape : {model.input_shape}")
    print(f"      Output shape: {model.output_shape}")

    # Locate augmentation pipeline
    aug_layer = None
    try:
        aug_layer = model.get_layer("field_robust_augmentation")
        print("      Found built-in field_robust_augmentation layer.")
    except Exception:
        try:
            from src.augmentations import build_field_robust_augmentation
            aug_layer = build_field_robust_augmentation(img_size=IMG_SIZE)
            print("      Constructed field_robust_augmentation pipeline from source.")
        except Exception:
            print("      Warning: Augmentation layer not found. TTA will use base model.")
            aug_layer = None

    return model, aug_layer


# ============================================================
# STEP 4 — Grad-CAM Generator Function
# ============================================================
def compute_gradcam(model, input_batch, target_class_idx):
    """
    Computes Grad-CAM saliency map for EfficientNetB3 backbone at target_class_idx.
    """
    try:
        backbone = model.get_layer("efficientnetb3")
        last_conv = backbone.get_layer(LAST_CONV_LAYER)

        # Backbone gradient model
        backbone_grad_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[last_conv.output, backbone.output]
        )

        # Head model reconstruction
        backbone_idx = next(i for i, l in enumerate(model.layers) if l.name == backbone.name)
        head_input = tf.keras.Input(shape=backbone.output.shape[1:])
        x = head_input
        for layer in model.layers[backbone_idx + 1:]:
            x = layer(x)
        head_model = tf.keras.Model(inputs=head_input, outputs=x)

        with tf.GradientTape() as tape:
            conv_outputs, backbone_features = backbone_grad_model(input_batch, training=False)
            tape.watch(conv_outputs)
            predictions = head_model(backbone_features, training=False)
            class_score = predictions[:, target_class_idx]

        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap = heatmap.numpy()

        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis], IMG_SIZE
        ).numpy().squeeze()

        return heatmap_resized
    except Exception as e:
        print(f"      Grad-CAM computation warning: {e}")
        return None


def resolve_image_path(raw_path: str) -> Path:
    """Finds image in current dir, ROOT, test_images/, or field_test_images/."""
    p = Path(raw_path)
    if p.exists():
        return p
    if (ROOT_DIR / raw_path).exists():
        return ROOT_DIR / raw_path
    if (ROOT_DIR / "test_images" / raw_path).exists():
        return ROOT_DIR / "test_images" / raw_path
    if (ROOT_DIR / "test_images" / p.name).exists():
        return ROOT_DIR / "test_images" / p.name
    for match in ROOT_DIR.glob(f"field_test_images/**/{p.name}"):
        if match.is_file():
            return match
    return p


def resolve_model_path(custom_model_str: str, image_path: Path):
    """Resolves model path, manifest path, and crop name automatically."""
    if custom_model_str:
        p = Path(custom_model_str)
        if p.exists():
            manifest = p.parent / "model_manifest.json"
            crop = "tomato" if "tomato" in str(p).lower() else "potato"
            return p, manifest, crop

    img_lower = image_path.name.lower()
    if any(k in img_lower for k in ["rice", "blast", "brown_spot", "img_2019"]):
        rice_new = ROOT_DIR / "models" / "rice" / "teachers" / "rice_teacher_efficientnetb3_best.keras"
        if rice_new.exists():
            return rice_new, rice_new.parent / "model_manifest.json", "rice"
        rice_v1 = ROOT_DIR / "models" / "rice_teacher_v1" / "rice_teacher_efficientnetb3_best.keras"
        if rice_v1.exists():
            return rice_v1, rice_v1.parent / "model_manifest.json", "rice"
        rice_m = ROOT_DIR / "models" / "rice_teacher" / "rice_teacher_efficientnetb3.keras"
        if rice_m.exists():
            return rice_m, rice_m.parent / "model_manifest.json", "rice"
    elif "tomato" in img_lower:
        tomato_v2 = ROOT_DIR / "models" / "tomato" / "teachers" / "v2" / "teacher_best.keras"
        if tomato_v2.exists():
            return tomato_v2, tomato_v2.parent / "model_manifest.json", "tomato"
        tomato_v3 = ROOT_DIR / "models" / "tomato_teacher_v3" / "tomato_teacher_efficientnetb3_best.keras"
        if tomato_v3.exists():
            return tomato_v3, tomato_v3.parent / "model_manifest.json", "tomato"
        tomato_m = ROOT_DIR / "models" / "tomato_teacher" / "tomato_teacher_efficientnetb3.keras"
        if tomato_m.exists():
            return tomato_m, tomato_m.parent / "model_manifest.json", "tomato"

    pot_new = ROOT_DIR / "models" / "potato" / "teachers" / "potato_teacher_efficientnetb3.keras"
    if pot_new.exists():
        return pot_new, pot_new.parent / "model_manifest.json", "potato"
    potato_m = ROOT_DIR / "models" / "potato_teacher" / "potato_teacher_efficientnetb3.keras"
    return potato_m, potato_m.parent / "model_manifest.json", "potato"


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    args = parse_args()
    image_path = resolve_image_path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {args.image_path}")
        print("  Searched in: current directory, test_images/, and field_test_images/")
        sys.exit(1)

    model_path, manifest_path, crop_name = resolve_model_path(args.model, image_path)

    # 1. Load Manifest
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        class_names = manifest.get("class_names") or manifest.get("classes") or ["early_blight", "healthy", "late_blight"]
        crop_name = manifest.get("crop", crop_name)
    else:
        class_names = ["early_blight", "healthy", "late_blight"]

    num_classes = len(class_names)
    print("=" * 68)
    print(f"       {crop_name.upper()} DISEASE FIELD-ROBUST INFERENCE & AUDIT")
    print("=" * 68)
    print(f"  Target Image : {image_path.name}")
    print(f"  Crop Type    : {crop_name.upper()}")
    print(f"  Class Names  : {class_names}")

    # 2. Load Model
    model, aug_layer = load_disease_model(model_path)

    # 3. Stage 1: Leaf Isolation
    print("\n[2/5] Stage 1: Leaf Isolation Preprocessing ...")
    if not args.no_isolate:
        if crop_name == "rice":
            from src.rice.preprocessor import preprocess_rice_leaf
            rice_pre = preprocess_rice_leaf(image_path, target_size=IMG_SIZE)
            input_crop = rice_pre["image"]
            crop_uint8 = rice_pre["image_uint8"]
            crop_raw_uint8 = crop_uint8
            pil_raw = Image.open(str(image_path)).convert("RGB")
            orig_rgb = np.array(pil_raw, dtype=np.uint8)
            pad_x, pad_y = rice_pre["pad_x"], rice_pre["pad_y"]
            scale = rice_pre["scale"]
            bbox = (pad_x, pad_y, pad_x + int(orig_rgb.shape[1] * scale), pad_y + int(orig_rgb.shape[0] * scale))
            engine_name = f"Rice Letterbox (Foliage: {rice_pre['foliage_ratio']*100:.1f}%)"
            bg_masked = False
        else:
            isolation = isolate_leaf(image_path, target_size=IMG_SIZE, mask_background=args.apply_mask)
            input_crop = isolation["cropped_image"]  # float32 [0..255]
            crop_uint8 = isolation["crop_uint8"]
            crop_raw_uint8 = isolation.get("crop_raw_uint8", crop_uint8)
            orig_rgb = isolation["orig_rgb"]
            bbox = isolation["bbox"]
            engine_name = isolation["engine_used"]
            bg_masked = isolation.get("bg_masked", False)
        print(f"      Isolation Engine  : {engine_name}")
        print(f"      Leaf Bounding Box : {bbox} on original {orig_rgb.shape[:2]} frame")
        print(f"      Background Mask   : {'Applied (Neutral Grey Fill)' if bg_masked else 'Bypassed'}")
    else:
        # Bypass isolation
        pil_raw = Image.open(str(image_path)).convert("RGB")
        orig_rgb = np.array(pil_raw, dtype=np.uint8)
        resized_raw = pil_raw.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        crop_uint8 = np.array(resized_raw, dtype=np.uint8)
        crop_raw_uint8 = crop_uint8
        input_crop = crop_uint8.astype(np.float32)
        bbox = (0, 0, orig_rgb.shape[1], orig_rgb.shape[0])
        engine_name = "None (Raw Full Image Tested)"
        bg_masked = False
        print("      Leaf isolation disabled via --no-isolate flag.")

    input_batch = np.expand_dims(input_crop, axis=0)  # shape (1, 300, 300, 3)

    # 4. Stage 2: Test-Time Augmentation (TTA)
    tta_passes = max(1, args.tta)
    print(f"\n[3/5] Stage 2: Test-Time Augmentation ({tta_passes} passes) ...")
    print("      Testing multi-angle views and inward zooms on isolated leaf ...")

    all_probs = []
    best_crops_per_class = {i: (0.0, input_batch) for i in range(num_classes)}

    # Pass 0: Base Unaugmented Leaf
    base_logits = model(input_batch, training=False)
    base_probs = tf.nn.softmax(base_logits).numpy()[0]
    all_probs.append(base_probs)
    for c in range(num_classes):
        if base_probs[c] > best_crops_per_class[c][0]:
            best_crops_per_class[c] = (float(base_probs[c]), input_batch)

    # Passes 1..N-1: Field-Robust Inward Zoom & Flip Passes
    for p in range(1, tta_passes):
        if aug_layer is not None:
            aug_input = aug_layer(input_batch, training=True)
            aug_logits = model(aug_input, training=False)
        else:
            aug_input = input_batch
            aug_logits = model(input_batch, training=True)

        aug_probs = tf.nn.softmax(aug_logits).numpy()[0]
        all_probs.append(aug_probs)

        for c in range(num_classes):
            if aug_probs[c] > best_crops_per_class[c][0]:
                best_crops_per_class[c] = (float(aug_probs[c]), aug_input)

    probs_array = np.array(all_probs)  # shape (tta_passes, num_classes)
    mean_probs = np.mean(probs_array, axis=0)
    max_probs = np.max(probs_array, axis=0)
    min_probs = np.min(probs_array, axis=0)
    std_probs = np.std(probs_array, axis=0)

    # 5. Stage 3: Smart Multi-Class Aggregation
    print("\n[4/5] Stage 3: Smart Diagnosis Aggregation ...")

    # Balanced Softmax Ensemble across all TTA passes
    primary_idx = int(np.argmax(mean_probs))
    primary_class = class_names[primary_idx]
    primary_conf = float(mean_probs[primary_idx]) * 100.0
    peak_conf = float(max_probs[primary_idx]) * 100.0

    # 6. Stage 4: Print Diagnostic Report
    print()
    print("=" * 68)
    print(f"              {crop_name.upper()} LEAF DISEASE DIAGNOSTIC REPORT")
    print("=" * 68)
    print(f"  Image Tested        : {image_path.name} (Original: {orig_rgb.shape[1]}x{orig_rgb.shape[0]})")
    print(f"  Crop Analyzed       : {crop_name.upper()}")
    print(f"  Isolation Engine    : {engine_name}")
    print(f"  Leaf Bounding Box   : [x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]")
    print(f"  TTA Inspection      : {tta_passes} passes (inward zoom + orientation invariance)")
    print("-" * 68)
    print(f"  PRIMARY DIAGNOSIS   : {primary_class.upper()}")
    print(f"  Confidence (Mean)   : {primary_conf:.2f}%  (Peak Pass: {peak_conf:.2f}%)")

    # Agronomic Status
    if primary_class == "late_blight":
        status_msg = f"PATHOGEN DETECTED: Phytophthora infestans (Late Blight on {crop_name.title()})"
        action_msg = ("CRITICAL: Immediate fungicide application recommended (e.g., Mancozeb / Chlorothalonil / Metalaxyl).\n"
                      "            Inspect neighboring plants, avoid overhead watering, quarantine affected area.")
    elif primary_class == "early_blight":
        status_msg = f"PATHOGEN DETECTED: Alternaria solani (Early Blight on {crop_name.title()})"
        action_msg = ("HIGH: Remove severely blighted lower foliage. Apply copper-based or protectant fungicide.\n"
                      "      Ensure adequate nitrogen and potassium balance in soil.")
    else:
        status_msg = f"HEALTHY FOLIAGE: No active blight lesions identified on {crop_name.title()}"
        action_msg = "Routine monitoring recommended. Maintain balanced nutrition and standard preventive protocols."

    print(f"  Pathogen Status     : {status_msg}")
    print("-" * 68)
    print(f"  {'Class Name':<18} {'Mean Prob':>10}   {'Peak (Max)':>10}   {'TTA Range [Min..Max]':>22}")
    print("-" * 68)
    for c_idx, c_name in enumerate(class_names):
        m_p = mean_probs[c_idx] * 100.0
        mx_p = max_probs[c_idx] * 100.0
        mn_p = min_probs[c_idx] * 100.0
        marker = "  <-- DIAGNOSED" if c_idx == primary_idx else ""
        print(f"  {c_name:<18} {m_p:>9.2f}%   {mx_p:>9.2f}%   [{mn_p:>5.1f}% .. {mx_p:>5.1f}%]{marker}")
    print("-" * 68)

    if args.ground_truth:
        gt_clean = args.ground_truth.lower()
        is_correct = (primary_class == gt_clean)
        verdict_str = "CORRECT [MATCH]" if is_correct else "DISCREPANCY [MISMATCH]"
        print(f"  Ground Truth Label  : {gt_clean.upper()}")
        print(f"  Diagnostic Verdict  : {verdict_str}")
        print("-" * 68)

    print(f"  Agronomic Guidance  : {action_msg}")
    print("=" * 68)

    # 7. Stage 5: Grad-CAM Explainability Map & Visual Audit
    print("\n[5/5] Generating Grad-CAM Explainability Audit ...")
    target_grad_batch = best_crops_per_class[primary_idx][1]
    # Ensure numpy array in [0, 255] float32
    if isinstance(target_grad_batch, tf.Tensor):
        grad_batch_np = target_grad_batch.numpy()
    else:
        grad_batch_np = np.array(target_grad_batch)

    heatmap_resized = compute_gradcam(model, grad_batch_np, primary_idx)

    # Output file path
    if args.output:
        out_path = Path(args.output)
    else:
        audit_dir = ROOT_DIR / "audit_reports"
        audit_dir.mkdir(parents=True, exist_ok=True)
        out_path = audit_dir / f"{image_path.stem}_audit.png"

    # Generate 4-Panel Diagnostic Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original Image with Bounding Box
    axes[0].imshow(orig_rgb)
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor="#00FF00", facecolor="none", linestyle="--"
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"1. Raw Field Image\nLeaf Bounding Box ({engine_name})", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Isolated Leaf Crop (Model Input)
    axes[1].imshow(crop_uint8)
    crop_title = "2. Isolated & Neutral-Masked Crop\n(Model Input 300x300)" if bg_masked else "2. Isolated Leaf Crop\n(Resized to 300x300)"
    axes[1].set_title(crop_title, fontsize=11)
    axes[1].axis("off")

    # Panel 3 & 4: Grad-CAM Saliency Map
    if heatmap_resized is not None:
        axes[2].imshow(heatmap_resized, cmap="jet")
        axes[2].set_title("3. Grad-CAM Activation\n(Red = High Pathogen Salience)", fontsize=11)
        axes[2].axis("off")

        colormap = plt.get_cmap("jet")
        heatmap_rgb = colormap(heatmap_resized)[:, :, :3]
        norm_crop = crop_uint8.astype(np.float32) / 255.0
        overlay = np.clip(0.6 * norm_crop + 0.4 * heatmap_rgb, 0, 1)

        axes[3].imshow(overlay)
        axes[3].set_title(f"4. Saliency Overlay\nFocus on {primary_class.upper()} Lesions", fontsize=11)
        axes[3].axis("off")
    else:
        axes[2].axis("off")
        axes[3].axis("off")

    gt_title = f" | GT: {args.ground_truth.upper()}" if args.ground_truth else ""
    plt.suptitle(
        f"{crop_name.title()} Teacher EfficientNetB3 Diagnostic Audit -- {image_path.name}\n"
        f"Diagnosis: {primary_class.upper()} ({primary_conf:.2f}%) {gt_title} | TTA Passes: {tta_passes}",
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Visual audit report saved to: {out_path.name}")
    print("\nInference pipeline completed successfully.")


if __name__ == "__main__":
    main()

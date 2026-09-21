"""
Potato Disease Field-Robust Inference & Explainability Audit Tool
==================================================================
Diagnostic tool for testing leaf images against the Potato Teacher Model (EfficientNetB3).
Supports single image inference, batch field validation, model comparison, and Grad-CAM saliency mapping.

Usage:
    # 1. Single image test:
    python test_potato_image.py potatotest.png

    # 2. Specify custom model:
    python test_potato_image.py potatotest.png --model models/potato_teacher/potato_teacher_efficientnetb3_v2.keras

    # 3. Batch field test set evaluation:
    python test_potato_image.py --dir field_test_images/potato

    # 4. Compare two models side-by-side on an image:
    python test_potato_image.py potatotest.png --compare models/potato_teacher/potato_teacher_efficientnetb3_v1.keras
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION & PATH RESOLUTION
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = ROOT_DIR / "potatotest.png"
MODEL_DIR = ROOT_DIR / "models" / "potato_teacher"
DEFAULT_MODEL_PATH = MODEL_DIR / "potato_teacher_efficientnetb3.keras"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"

IMG_SIZE = (300, 300)
LAST_CONV_LAYER = "top_conv"
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def resolve_model_path(custom_path: str = None) -> Path:
    """Finds the model file, checking custom path, default, and fallback versions."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified model not found: {custom_path}")

    # Check for v2 first if it exists, otherwise default
    v2_path = MODEL_DIR / "potato_teacher_efficientnetb3_v2.keras"
    if v2_path.exists():
        return v2_path

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    # Check stage A best
    stage_a = MODEL_DIR / "potato_stage_a_best.keras"
    if stage_a.exists():
        print(f"Notice: Final model not found, using Stage A checkpoint: {stage_a.name}")
        return stage_a

    raise FileNotFoundError(
        f"No model found in {MODEL_DIR}.\n"
        "Please run training first (Phase B)."
    )


def load_model_and_classes(model_path: Path):
    """Loads a Keras model and extracts class names from manifest."""
    print(f"Loading Potato Teacher Model from: {model_path.name} ...")
    model = tf.keras.models.load_model(str(model_path))

    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            class_names = manifest.get("class_names", ["early_blight", "healthy", "late_blight"])
        except Exception:
            class_names = ["early_blight", "healthy", "late_blight"]
    else:
        class_names = ["early_blight", "healthy", "late_blight"]

    return model, class_names


def preprocess_image(image_path: Path):
    """Loads and prepares an input image for EfficientNetB3 inference."""
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = tf.keras.utils.load_img(str(image_path), target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)  # float32 [0, 255]
    img_batch = np.expand_dims(img_array, axis=0)
    return img_array, img_batch


def make_gradcam_heatmap(img_batch, model, last_conv_layer_name=LAST_CONV_LAYER, pred_index=None):
    """
    Generates a Grad-CAM heatmap hooking into EfficientNetB3's top_conv layer.
    Uses dynamic layer indexing to bridge across nested Functional model graphs in Keras 3.
    """
    try:
        backbone_model = model.get_layer("efficientnetb3")
    except ValueError:
        # Fallback: search for backbone in layers
        backbone_model = None
        for l in model.layers:
            if "efficientnet" in l.name.lower():
                backbone_model = l
                break
        if backbone_model is None:
            raise ValueError("Could not locate EfficientNet backbone in model.")

    last_conv_layer = backbone_model.get_layer(last_conv_layer_name)

    # 1. Feature extractor from backbone
    backbone_grad_model = tf.keras.Model(
        inputs=backbone_model.input,
        outputs=[last_conv_layer.output, backbone_model.output]
    )

    # 2. Classifier head model: dynamically locate layers after backbone
    backbone_idx = -1
    for idx, layer in enumerate(model.layers):
        if layer == backbone_model or layer.name == backbone_model.name:
            backbone_idx = idx
            break

    head_start = backbone_idx + 1 if backbone_idx != -1 else 3
    head_input = tf.keras.Input(shape=backbone_model.output.shape[1:])
    x = head_input
    for layer in model.layers[head_start:]:
        x = layer(x)
    head_model = tf.keras.Model(inputs=head_input, outputs=x)

    # 3. Compute gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, backbone_outputs = backbone_grad_model(img_batch, training=False)
        tape.watch(conv_outputs)
        predictions = head_model(backbone_outputs, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    # 4. Global average pooling of gradients
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap.numpy()


def run_single_inference(image_path: Path, model, class_names, save_gradcam=True, output_suffix=""):
    """Runs inference on a single image and generates Grad-CAM."""
    img_array, img_batch = preprocess_image(image_path)

    # Forward pass
    logits = model(img_batch, training=False)
    probs = tf.nn.softmax(logits).numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_conf = probs[pred_idx] * 100.0

    result = {
        "image_path": str(image_path),
        "filename": image_path.name,
        "pred_class": pred_class,
        "confidence": pred_conf,
        "probs": {name: float(probs[i]) for i, name in enumerate(class_names)},
        "pred_idx": pred_idx,
    }

    if save_gradcam:
        try:
            heatmap = make_gradcam_heatmap(img_batch, model, pred_index=pred_idx)
            heatmap_resized = tf.image.resize(
                heatmap[..., np.newaxis], (IMG_SIZE[0], IMG_SIZE[1])
            ).numpy().squeeze()

            cmap = plt.get_cmap("jet")
            heatmap_colored = cmap(heatmap_resized)[:, :, :3]
            orig_img = img_array / 255.0
            alpha = 0.4
            overlay = np.clip((1 - alpha) * orig_img + alpha * heatmap_colored, 0, 1)

            fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
            ax[0].imshow(orig_img)
            ax[0].set_title(f"Original: {image_path.name}", fontsize=11)
            ax[0].axis("off")

            ax[1].imshow(heatmap_resized, cmap="jet")
            ax[1].set_title("Grad-CAM Activation", fontsize=11)
            ax[1].axis("off")

            ax[2].imshow(overlay)
            ax[2].set_title(f"Saliency Overlay: {pred_class} ({pred_conf:.1f}%)", fontsize=11)
            ax[2].axis("off")

            plt.suptitle(
                f"Potato Teacher Model — Saliency Audit\nPredicted: {pred_class.upper()} ({pred_conf:.2f}%)",
                fontsize=12,
                fontweight="bold"
            )
            plt.tight_layout()

            suffix = f"_{output_suffix}" if output_suffix else ""
            out_name = f"{image_path.stem}{suffix}_gradcam.png"
            output_plot = image_path.parent / out_name
            plt.savefig(output_plot, dpi=200, bbox_inches="tight")
            plt.close()
            result["gradcam_path"] = str(output_plot)
        except Exception as e:
            result["gradcam_error"] = str(e)

    return result


def print_diagnostic_report(result: dict, class_names):
    """Formats and prints single-image diagnostic report to terminal."""
    print("\n" + "=" * 60)
    print("        POTATO LEAF DISEASE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Image Tested       : {result['filename']}")
    print(f"Primary Diagnosis  : {result['pred_class'].upper()}")
    print(f"Confidence Score   : {result['confidence']:.2f}%")
    print("-" * 60)
    print(f"{'Class Name':<25} {'Probability':<15} {'Bar'}")
    print("-" * 60)
    for name in class_names:
        p = result["probs"][name]
        bar = "#" * int(p * 30)
        marker = " <--" if name == result["pred_class"] else ""
        print(f"{name:<25} {p*100:6.2f}%        {bar}{marker}")
    print("-" * 60)

    conf = result["confidence"]
    if conf >= 80.0:
        status = "HIGH CONFIDENCE (Decisive prediction)"
    elif conf >= 60.0:
        status = "MODERATE CONFIDENCE (Acceptable)"
    else:
        status = "LOW CONFIDENCE (Recommend review/abstention)"
    print(f"Diagnostic Assessment: {status}")
    if "gradcam_path" in result:
        print(f"Grad-CAM Saved To    : {Path(result['gradcam_path']).name}")
    print("=" * 60)


def run_batch_evaluation(directory: Path, model, class_names):
    """Recursively evaluates all leaf images in a directory."""
    images = [
        p for p in directory.rglob("*")
        if p.suffix.lower() in VALID_IMAGE_EXTS and not p.name.endswith("_gradcam.png")
    ]

    if not images:
        print(f"\n[INFO] No test images found in: {directory}")
        print(f"Supported formats: {', '.join(VALID_IMAGE_EXTS)}")
        print("Please place real field images in field_test_images/potato/{early_blight,healthy,late_blight}/")
        return

    print(f"\n{'='*75}")
    print(f"       BATCH FIELD VALIDATION: {len(images)} IMAGES FOUND")
    print(f"{'='*75}")
    print(f"{'Filename':<30} {'Ground Truth':<15} {'Predicted':<15} {'Conf %':<8} {'Status'}")
    print(f"{'-'*75}")

    correct = 0
    total_evaluated = 0
    results = []

    for img_path in sorted(images):
        # Infer ground truth from parent folder if it matches class names
        ground_truth = "unknown"
        for cn in class_names:
            if cn.lower() in img_path.parent.name.lower():
                ground_truth = cn
                break

        res = run_single_inference(img_path, model, class_names, save_gradcam=True)
        results.append((res, ground_truth))

        pred = res["pred_class"]
        conf = res["confidence"]

        if ground_truth != "unknown":
            total_evaluated += 1
            if pred == ground_truth:
                correct += 1
                status = "PASS [OK]"
            else:
                status = "FAIL [X]"
        else:
            status = "N/A"

        print(f"{img_path.name[:28]:<30} {ground_truth:<15} {pred:<15} {conf:6.2f}% {status}")

    print(f"{'='*75}")
    if total_evaluated > 0:
        accuracy = (correct / total_evaluated) * 100.0
        print(f"\nSummary Metrics:")
        print(f"  Field Accuracy   : {accuracy:.2f}% ({correct}/{total_evaluated})")
        avg_conf = np.mean([r[0]['confidence'] for r in results])
        print(f"  Average Conf     : {avg_conf:.2f}%")
        print(f"  Phase C Gate     : {'PASSED (>= 80%)' if accuracy >= 80 else 'FAILED (< 80%)'}")
    else:
        print("\nNote: Images were not in class-labeled subfolders; per-class accuracy not computed.")
    print(f"All Grad-CAM visualizations saved alongside original images.\n")


def run_model_comparison(image_path: Path, model_a_path: Path, model_b_path: Path):
    """Compares predictions and Grad-CAM maps between two model checkpoints."""
    print(f"\nComparing Models on: {image_path.name}")
    print(f"  Model A (Baseline) : {model_a_path.name}")
    print(f"  Model B (New/v2)   : {model_b_path.name}")

    model_a, classes_a = load_model_and_classes(model_a_path)
    model_b, classes_b = load_model_and_classes(model_b_path)

    res_a = run_single_inference(image_path, model_a, classes_a, save_gradcam=False)
    res_b = run_single_inference(image_path, model_b, classes_b, save_gradcam=False)

    print("\n" + "=" * 60)
    print("           MODEL COMPARISON AUDIT")
    print("=" * 60)
    print(f"Image: {image_path.name}")
    print(f"{'Metric':<20} {'Model A (' + model_a_path.stem[:12] + ')':<20} {'Model B (' + model_b_path.stem[:12] + ')'}")
    print("-" * 60)
    print(f"{'Diagnosis':<20} {res_a['pred_class'].upper():<20} {res_b['pred_class'].upper()}")
    print(f"{'Confidence':<20} {res_a['confidence']:6.2f}%{' '*13} {res_b['confidence']:6.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Potato Leaf Disease Inference & Saliency Audit Tool")
    parser.add_argument("image", nargs="?", default=None, help="Path to leaf image file")
    parser.add_argument("--model", type=str, default=None, help="Path to custom model .keras file")
    parser.add_argument("--dir", type=str, default=None, help="Directory of field images for batch audit")
    parser.add_argument("--compare", type=str, default=None, help="Path to baseline model for side-by-side comparison")
    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        img_path = Path(args.image) if args.image else DEFAULT_IMAGE_PATH
        model_a = Path(args.compare)
        model_b = resolve_model_path(args.model)
        run_model_comparison(img_path, model_a, model_b)
        return

    # Load active model
    model_path = resolve_model_path(args.model)
    model, class_names = load_model_and_classes(model_path)

    # Batch directory mode
    if args.dir:
        run_batch_evaluation(Path(args.dir), model, class_names)
        return

    # Single image mode
    target_image = Path(args.image) if args.image else DEFAULT_IMAGE_PATH
    res = run_single_inference(target_image, model, class_names, save_gradcam=True)
    print_diagnostic_report(res, class_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Diagnostic tool failed: {e}")
        sys.exit(1)

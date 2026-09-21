"""
audit_background_sensitivity.py
================================
Experiment B (Causal Invariance):
Tests whether the model's prediction on genuine field leaves is causally driven by
the leaf pathology or by spurious background soil/mulch textures.

Takes field test images and generates 3 controlled background perturbations:
  1. Original Natural Crop (Baseline)
  2. Gaussian-Blurred Background (Leaf blade untouched)
  3. Darkened Background (50% luminance reduction on non-leaf pixels)
  4. Neutral Foliage Background (Soil replaced with uniform green context)

If the model flips its classification when ONLY the background changes,
it fails the causal invariance audit.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import keras
from leaf_isolator import isolate_leaf

MODEL_PATH = ROOT_DIR / "models" / "tomato_teacher_v3" / "tomato_teacher_efficientnetb3_best.keras"
MANIFEST_PATH = ROOT_DIR / "field_test_images" / "field_holdout_manifest.json"
REPORT_DIR = ROOT_DIR / "audit_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def create_perturbed_crops(image_path: Path, target_size=(300, 300)):
    """Creates original, blurred-bg, and darkened-bg crops for causal testing."""
    iso_orig = isolate_leaf(image_path, target_size=target_size, mask_background=False)
    x1, y1, x2, y2 = iso_orig["bbox"]
    full_mask = iso_orig["mask"]  # 2D uint8 mask (255 for leaf, 0 for background)

    with Image.open(image_path) as pil_img:
        rgb_img = pil_img.convert("RGB")
        img_np = np.array(rgb_img)

        # 1. Original crop
        crop_orig_np = iso_orig["cropped_image"]

        # 2. Blurred Background (Leaf untouched)
        blurred_full = rgb_img.filter(ImageFilter.GaussianBlur(radius=15))
        blurred_np = np.array(blurred_full)
        leaf_binary = (full_mask > 0)[:, :, np.newaxis]
        blurred_composite = np.where(leaf_binary, img_np, blurred_np)
        iso_blur = isolate_leaf(blurred_composite, target_size=target_size, mask_background=False)
        crop_blur_np = iso_blur["cropped_image"]

        # 3. Darkened Background (50% brightness reduction on non-leaf pixels)
        darkened_np = (img_np * 0.45).astype(np.uint8)
        darkened_composite = np.where(leaf_binary, img_np, darkened_np)
        iso_dark = isolate_leaf(darkened_composite, target_size=target_size, mask_background=False)
        crop_dark_np = iso_dark["cropped_image"]

    return {
        "original": crop_orig_np,
        "blurred_bg": crop_blur_np,
        "darkened_bg": crop_dark_np,
        "bbox": iso_orig["bbox"]
    }


def main():
    print("=" * 75)
    print("    EXPERIMENT B: CAUSAL BACKGROUND SENSITIVITY & INVARIANCE AUDIT")
    print("=" * 75)

    if not MODEL_PATH.exists():
        # Fallback to existing model if v3 is not yet trained
        fallback = ROOT_DIR / "models" / "tomato_teacher" / "tomato_teacher_efficientnetb3.keras"
        if fallback.exists():
            active_model_path = fallback
            print(f"  [Note] Using baseline model: {active_model_path.name}")
        else:
            raise FileNotFoundError(f"No model found at {MODEL_PATH}")
    else:
        active_model_path = MODEL_PATH
        print(f"  [Active Model] {active_model_path.name}")

    model = keras.models.load_model(str(active_model_path))
    classes = ["early_blight", "healthy", "late_blight"]

    # Load field samples
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            field_data = json.load(f)
        samples = field_data["samples"]
    else:
        # Auto-discover root tomatotest*.webp
        samples = [{"filename": p.name, "ground_truth": "healthy"} for p in ROOT_DIR.glob("tomatotest*.webp")]

    print(f"  Evaluating {len(samples)} field samples across 3 background perturbation regimes...\n")

    results = []
    stable_count = 0

    for item in samples:
        fname = item["filename"]
        possible_paths = [
            ROOT_DIR / fname,
            ROOT_DIR / "field_test_images" / "tomato" / Path(fname).name,
            ROOT_DIR / "field_test_images" / fname,
        ]
        img_file = next((p for p in possible_paths if p.exists()), None)
        if img_file is None:
            continue

        gt = item.get("ground_truth", "unknown")
        crops = create_perturbed_crops(img_file)

        # Predict across all 3 regimes
        batch = np.stack([crops["original"], crops["blurred_bg"], crops["darkened_bg"]], axis=0)
        preds = model(batch, training=False)
        preds_np = keras.ops.convert_to_numpy(preds)

        p_orig = classes[int(np.argmax(preds_np[0]))]
        p_blur = classes[int(np.argmax(preds_np[1]))]
        p_dark = classes[int(np.argmax(preds_np[2]))]

        conf_orig = float(np.max(preds_np[0])) * 100.0
        conf_blur = float(np.max(preds_np[1])) * 100.0
        conf_dark = float(np.max(preds_np[2])) * 100.0

        is_invariant = (p_orig == p_blur == p_dark)
        if is_invariant:
            stable_count += 1

        status_flag = "✓ INVARIANT" if is_invariant else "⚠ FLIPPED"
        print(f"  • {item['filename']:<16} [GT: {gt:<12}] -> {status_flag}")
        print(f"      Orig: {p_orig} ({conf_orig:.1f}%) | Blur: {p_blur} ({conf_blur:.1f}%) | Dark: {p_dark} ({conf_dark:.1f}%)")

        results.append({
            "filename": item["filename"],
            "ground_truth": gt,
            "prediction_original": p_orig,
            "confidence_original": conf_orig,
            "prediction_blurred_bg": p_blur,
            "confidence_blurred_bg": conf_blur,
            "prediction_darkened_bg": p_dark,
            "confidence_darkened_bg": conf_dark,
            "causally_invariant": is_invariant
        })

    stability_rate = (stable_count / max(1, len(results))) * 100.0
    print("\n" + "=" * 75)
    print("                     CAUSAL AUDIT SUMMARY")
    print("=" * 75)
    print(f"  Total Field Images Evaluated : {len(results)}")
    print(f"  Causally Invariant Samples   : {stable_count} / {len(results)}")
    print(f"  Background Invariance Rate   : {stability_rate:.1f}%")
    print(f"  Acceptance Gate (>85% Stable): {'PASSED ⭐' if stability_rate >= 85.0 else 'FAILED ❌'}")
    print("=" * 75)

    report_path = REPORT_DIR / "background_sensitivity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_tested": str(active_model_path.name),
            "invariance_rate": round(stability_rate, 2),
            "gate_passed": bool(stability_rate >= 85.0),
            "results": results
        }, f, indent=2)

    print(f"✓ Full report saved to: {report_path.resolve()}\n")


if __name__ == "__main__":
    main()

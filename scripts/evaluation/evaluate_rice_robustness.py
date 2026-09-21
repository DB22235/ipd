"""
evaluate_rice_robustness.py
===========================
Industrial Acceptance & Field Robustness Audit Tool for Rice Disease Detection.
Evaluates:
  1. Locked Test Set (981 independent unseen images).
  2. Independent Field Holdout Suite (12 curated field samples with botanical ground truth).
  3. Calibration & Expected Calibration Error (ECE).
  4. Selective Prediction under Entropy & Confidence Abstention Gates.
Outputs:
  audit_reports/rice/field_robustness_report.json
  audit_reports/rice/calibration_report.json
  audit_reports/rice/test_confusion_matrix.png
  audit_reports/rice/field_confusion_matrix.png
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from torch.utils.data import DataLoader
from src.rice.preprocessor import preprocess_rice_leaf
from scripts.training.train_local_rice_efficientnetb3 import RiceDataset

DEFAULT_MODEL = ROOT_DIR / "models" / "rice_teacher_v1" / "rice_teacher_efficientnetb3_best.keras"
DEFAULT_TEST_DIR = ROOT_DIR / "clean_dataset" / "rice_dataset" / "test"
DEFAULT_FIELD_DIR = ROOT_DIR / "field_test_images" / "rice"
DEFAULT_FIELD_MANIFEST = ROOT_DIR / "field_test_images" / "rice" / "field_holdout_manifest.json"
REPORT_DIR = ROOT_DIR / "audit_reports" / "rice"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["blast", "blight", "brown_spot", "healthy"]


def parse_args():
    parser = argparse.ArgumentParser(description="Rice Robustness & Acceptance Audit")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to .keras model")
    parser.add_argument("--conf-threshold", type=float, default=0.60, help="Abstention confidence threshold")
    parser.add_argument("--entropy-threshold", type=float, default=0.85, help="Abstention Shannon entropy threshold (nats)")
    return parser.parse_args()


def calculate_ece(confidences, predictions, labels, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    bin_data = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(predictions[in_bin] == labels[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "samples": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            })
    return float(ece), bin_data


def audit_test_set(model, test_dir: Path, conf_thresh: float, entropy_thresh: float):
    print("\n" + "=" * 75)
    print("       AUDIT 1: LOCKED TEST SET EVALUATION (981 IMAGES)")
    print("=" * 75)

    test_ds = RiceDataset(test_dir, img_size=(300, 300), is_training=False, preload=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    y_true = []
    y_probs = []
    for imgs, lbls in test_loader:
        logits = model(imgs, training=False)
        probs = keras.ops.softmax(logits)
        probs_np = keras.ops.convert_to_numpy(probs)
        lbls_np = keras.ops.convert_to_numpy(lbls)
        y_probs.extend(probs_np)
        y_true.extend(lbls_np)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    # Shannon entropy (nats)
    eps = 1e-12
    entropies = -np.sum(y_probs * np.log(y_probs + eps), axis=1)

    raw_acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    cls_rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)

    # Abstention Analysis
    confident_mask = (confidences >= conf_thresh) & (entropies <= entropy_thresh)
    abstain_count = int(np.sum(~confident_mask))
    coverage = float(np.mean(confident_mask))
    if np.sum(confident_mask) > 0:
        selective_acc = float(np.mean(y_true[confident_mask] == y_pred[confident_mask]))
    else:
        selective_acc = 0.0

    # Calibration
    ece, bin_details = calculate_ece(confidences, y_pred, y_true)

    print(f"  Raw Accuracy        : {raw_acc * 100:.2f}%")
    print(f"  Macro-F1            : {macro_f1 * 100:.2f}%")
    print(f"  Weighted-F1         : {weighted_f1 * 100:.2f}%")
    print(f"  Selective Accuracy  : {selective_acc * 100:.2f}% (Coverage: {coverage * 100:.1f}%)")
    print(f"  Abstained / Rejected: {abstain_count} / {len(y_true)} ({abstain_count / len(y_true) * 100:.1f}%)")
    print(f"  Expected Calib Error: {ece:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Rice Test Set Confusion Matrix\nAcc: {raw_acc*100:.1f}% | Macro-F1: {macro_f1*100:.1f}% | ECE: {ece:.3f}")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_path = REPORT_DIR / "test_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=200)
    plt.close()

    return {
        "raw_accuracy": raw_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "selective_accuracy": selective_acc,
        "coverage": coverage,
        "abstained_count": abstain_count,
        "expected_calibration_error": ece,
        "per_class": cls_rep,
        "calibration_bins": bin_details,
    }


def audit_field_holdout(model, manifest_path: Path, field_dir: Path, conf_thresh: float, entropy_thresh: float):
    print("\n" + "=" * 75)
    print("       AUDIT 2: INDEPENDENT FIELD HOLDOUT SUITE (12 SAMPLES)")
    print("=" * 75)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    y_true = []
    y_pred = []
    records = []
    abstained = 0

    print(f"  Auditing {len(samples)} curated real-world field samples across 4 classes:\n")
    print(f"  {'Sample ID':<15} {'Ground Truth':<12} {'Prediction':<12} {'Conf':<8} {'Entropy':<8} {'Status'}")
    print("  " + "-" * 68)

    for s in samples:
        fname = s["filename"]
        img_path = field_dir / fname
        if not img_path.exists():
            print(f"  Warning: Missing file {img_path}")
            continue

        gt = s["ground_truth"]
        gt_idx = class_to_idx[gt]
        y_true.append(gt_idx)

        # Preprocess using Aspect-Preserving Letterbox
        pre = preprocess_rice_leaf(img_path, target_size=(300, 300))
        crop_np = np.expand_dims(pre["image"], axis=0)

        preds = model(crop_np, training=False)
        probs = keras.ops.convert_to_numpy(keras.ops.softmax(preds))[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        conf = float(probs[pred_idx])

        eps = 1e-12
        entropy = -float(np.sum(probs * np.log(probs + eps)))
        y_pred.append(pred_idx)

        is_confident = (conf >= conf_thresh) and (entropy <= entropy_thresh) and pre["is_valid_rice_leaf"]
        if not is_confident:
            abstained += 1
            verdict = "ABSTAIN / OOD"
        elif pred_class == gt:
            verdict = "CORRECT"
        else:
            verdict = "DISCREPANCY"

        print(f"  {s['id']:<15} {gt:<12} {pred_class:<12} {conf * 100:>5.1f}%  {entropy:>6.3f}   {verdict}")

        records.append({
            "id": s["id"],
            "filename": fname,
            "ground_truth": gt,
            "predicted_class": pred_class,
            "confidence": conf,
            "entropy": entropy,
            "foliage_ratio": pre["foliage_ratio"],
            "verdict": verdict,
            "is_confident": is_confident,
            "challenge": s["challenge"]
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    raw_acc = float(np.mean(y_true == y_pred))

    confident_indices = [i for i, r in enumerate(records) if r["is_confident"]]
    if confident_indices:
        sel_acc = float(np.mean(y_true[confident_indices] == y_pred[confident_indices]))
    else:
        sel_acc = 0.0

    print("  " + "-" * 68)
    print(f"  Field Raw Accuracy       : {raw_acc * 100:.1f}% ({int(np.sum(y_true == y_pred))}/{len(y_true)})")
    print(f"  Field Selective Accuracy : {sel_acc * 100:.1f}% (Coverage: {len(confident_indices)}/{len(y_true)})")
    print(f"  Field Abstentions        : {abstained} / {len(y_true)}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Rice Field Holdout Confusion Matrix\nRaw Acc: {raw_acc*100:.1f}% | Selective: {sel_acc*100:.1f}%")
    plt.ylabel("Ground Truth")
    plt.xlabel("Model Diagnosis")
    plt.tight_layout()
    cm_path = REPORT_DIR / "field_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=200)
    plt.close()

    return {
        "field_raw_accuracy": raw_acc,
        "field_selective_accuracy": sel_acc,
        "field_coverage": len(confident_indices) / len(y_true),
        "field_abstentions": abstained,
        "sample_records": records
    }


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print("=" * 75)
    print("          IPD PRODUCTION ACCEPTANCE AUDIT: RICE PILOT")
    print("=" * 75)
    print(f"  Model Under Audit  : {model_path.name}")
    print(f"  Confidence Gate    : >= {args.conf_threshold * 100:.1f}%")
    print(f"  Entropy Gate       : <= {args.entropy_threshold:.2f} nats")

    model = keras.models.load_model(str(model_path))

    # Run Test Audit
    test_results = audit_test_set(model, DEFAULT_TEST_DIR, args.conf_threshold, args.entropy_threshold)

    # Run Field Holdout Audit
    field_results = audit_field_holdout(model, DEFAULT_FIELD_MANIFEST, DEFAULT_FIELD_DIR, args.conf_threshold, args.entropy_threshold)

    # Save Composite Reports
    robustness_report = {
        "crop": "rice",
        "model": model_path.name,
        "test_metrics": test_results,
        "field_holdout_metrics": field_results,
    }

    rob_json = REPORT_DIR / "field_robustness_report.json"
    with open(rob_json, "w", encoding="utf-8") as f:
        json.dump(robustness_report, f, indent=2)

    calib_report = {
        "crop": "rice",
        "expected_calibration_error": test_results["expected_calibration_error"],
        "confidence_threshold": args.conf_threshold,
        "entropy_threshold": args.entropy_threshold,
        "calibration_bins": test_results["calibration_bins"],
    }
    calib_json = REPORT_DIR / "calibration_report.json"
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_report, f, indent=2)

    print("\n" + "=" * 75)
    print("[SUCCESS] All Audit Reports and Visualizations Saved:")
    print(f"  - {rob_json.relative_to(ROOT_DIR)}")
    print(f"  - {calib_json.relative_to(ROOT_DIR)}")
    print(f"  - audit_reports/rice/test_confusion_matrix.png")
    print(f"  - audit_reports/rice/field_confusion_matrix.png")
    print("=" * 75)


if __name__ == "__main__":
    main()

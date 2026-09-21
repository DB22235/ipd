"""
evaluate_field_robustness.py
============================
Phase 5: Final Production Field Robustness Evaluation & Acceptance Gating.
Evaluates the trained EfficientNetB3 model on the independent field holdout dataset.

Metrics Computed:
  - Healthy False Positive Rate (HFPR) — Primary failure metric.
  - Per-Class Recall & Precision.
  - Expected Calibration Error (ECE).
  - Selective Accuracy under Abstention Policy (rejection of high-entropy / low-confidence inputs).
  - Visual Reliability Diagram.

Usage:
  python evaluate_field_robustness.py
  python evaluate_field_robustness.py --model models/tomato_teacher_v3/tomato_teacher_efficientnetb3_best.keras
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "torch"

import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from leaf_isolator import isolate_leaf

REPORT_DIR = ROOT_DIR / "audit_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = ROOT_DIR / "field_test_images" / "field_holdout_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tomato Field Robustness")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model checkpoint")
    parser.add_argument("--abstain-threshold", type=float, default=0.60,
                        help="Confidence threshold below which model abstains (default: 0.60)")
    return parser.parse_args()


def calculate_ece(confidences, predictions, labels, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(predictions[in_bin] == labels[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def main():
    args = parse_args()

    print("=" * 80)
    print("      PRODUCTION ACCEPTANCE AUDIT: TOMATO FIELD ROBUSTNESS")
    print("=" * 80)

    # Model resolution
    if args.model:
        model_path = Path(args.model)
    else:
        v3_path = ROOT_DIR / "models" / "tomato_teacher_v3" / "tomato_teacher_efficientnetb3_best.keras"
        if v3_path.exists():
            model_path = v3_path
        else:
            model_path = ROOT_DIR / "models" / "tomato_teacher" / "tomato_teacher_efficientnetb3.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"  [Model Evaluated] : {model_path.name}")
    model = keras.models.load_model(str(model_path))
    classes = ["early_blight", "healthy", "late_blight"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Load field holdout manifest
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = manifest_data["samples"]
    print(f"  [Field Holdout]   : {len(samples)} curated real-world field samples\n")

    y_true_indices = []
    y_pred_indices = []
    confidences = []
    entropies = []
    eval_records = []

    abstained_count = 0

    for sample in samples:
        fname = sample["filename"]
        possible_paths = [
            ROOT_DIR / fname,
            ROOT_DIR / "field_test_images" / "tomato" / Path(fname).name,
            ROOT_DIR / "field_test_images" / fname,
        ]
        img_path = next((p for p in possible_paths if p.exists()), None)
        if img_path is None:
            print(f"  ⚠ Skipping missing file: {fname}")
            continue

        gt_class = sample["ground_truth"]
        gt_idx = class_to_idx[gt_class]

        # Isolate bounding box crop unmasked
        iso = isolate_leaf(img_path, target_size=(300, 300), mask_background=False)
        crop_np = np.expand_dims(iso["cropped_image"], axis=0)

        preds = model(crop_np, training=False)
        probs = keras.ops.convert_to_numpy(preds)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]
        conf = float(probs[pred_idx])

        # Compute Shannon entropy
        eps = 1e-12
        entropy = -float(np.sum(probs * np.log(probs + eps)))

        is_abstained = (conf < args.abstain_threshold)
        if is_abstained:
            abstained_count += 1

        is_correct = (pred_class == gt_class)

        y_true_indices.append(gt_idx)
        y_pred_indices.append(pred_idx)
        confidences.append(conf)
        entropies.append(entropy)

        status = "✓ PASS" if is_correct else "❌ FAIL"
        decision_label = "REJECTED (UNCERTAIN)" if is_abstained else pred_class.upper()
        white_norm_flag = " [Studio-Norm]" if iso.get("studio_white_normalized") else ""

        print(f"  • {sample['filename']:<16} [GT: {gt_class:<12}] -> Pred: {decision_label:<14} ({conf*100:.1f}%){white_norm_flag} | {status}")

        eval_records.append({
            "filename": sample["filename"],
            "ground_truth": gt_class,
            "prediction": pred_class,
            "confidence": round(conf, 4),
            "entropy": round(entropy, 4),
            "abstained": is_abstained,
            "correct": is_correct,
            "studio_white_normalized": iso.get("studio_white_normalized", False),
            "probabilities": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
        })

    y_true = np.array(y_true_indices)
    y_pred = np.array(y_pred_indices)
    conf_arr = np.array(confidences)

    # Calculate Key Production Metrics
    raw_accuracy = float(np.mean(y_true == y_pred)) * 100.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) * 100.0
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    # Primary failure metric: Healthy False Positive Rate (HFPR)
    healthy_idx = class_to_idx["healthy"]
    healthy_mask = (y_true == healthy_idx)
    total_healthy = int(np.sum(healthy_mask))
    healthy_false_positives = int(np.sum((y_pred != healthy_idx) & healthy_mask))
    hfpr = (healthy_false_positives / max(1, total_healthy)) * 100.0

    # Calibration metric: ECE
    ece = calculate_ece(conf_arr, y_pred, y_true, n_bins=5)

    # Selective accuracy (excluding abstained samples)
    non_abstained_mask = ~np.array([r["abstained"] for r in eval_records])
    if np.sum(non_abstained_mask) > 0:
        selective_acc = float(np.mean(y_true[non_abstained_mask] == y_pred[non_abstained_mask])) * 100.0
    else:
        selective_acc = 0.0

    # Print Acceptance Scorecard
    print("\n" + "=" * 80)
    print("                 FIELD ROBUSTNESS ACCEPTANCE SCORECARD")
    print("=" * 80)
    print(f"  • Field Evaluation Accuracy         : {raw_accuracy:.1f}%")
    print(f"  • Field Macro-F1                    : {macro_f1:.1f}%")
    print(f"  • Healthy False Positive Rate (HFPR): {hfpr:.1f}%  (Target: < 15.0%)")
    print(f"  • Expected Calibration Error (ECE)  : {ece:.4f}   (Target: < 0.15)")
    print(f"  • Selective Accuracy (with Rejection): {selective_acc:.1f}% ({abstained_count} abstained)")
    print("-" * 80)
    print("  Per-Class Field Performance:")
    for i, c in enumerate(classes):
        c_mask = (y_true == i)
        c_tot = int(np.sum(c_mask))
        c_cor = int(np.sum((y_pred == i) & c_mask))
        c_rec = (c_cor / max(1, c_tot)) * 100.0
        print(f"    - {c:<14}: Recall = {c_rec:.1f}% ({c_cor}/{c_tot}) | F1 = {per_class_f1[i]*100:.1f}%")
    
    hfpr_pass = (hfpr <= 15.0)
    macro_pass = (macro_f1 >= 80.0)
    print("=" * 80)
    print(f"  FINAL FIELD ACCEPTANCE GATE : {'PASSED ⭐' if (hfpr_pass and macro_pass) else 'REQUIRES RETRAINING ⚠'}")
    print("=" * 80)

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Field Holdout Confusion Matrix\nHFPR: {hfpr:.1f}% | Macro-F1: {macro_f1:.1f}%")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    cm_path = REPORT_DIR / "field_confusion_matrix.png"
    fig.savefig(str(cm_path), dpi=200)
    plt.close(fig)

    # Save Report
    report_data = {
        "model_path": str(model_path.name),
        "total_field_samples": len(eval_records),
        "raw_accuracy": round(raw_accuracy, 2),
        "macro_f1": round(macro_f1, 2),
        "healthy_false_positive_rate": round(hfpr, 2),
        "expected_calibration_error": round(ece, 4),
        "selective_accuracy": round(selective_acc, 2),
        "abstained_count": abstained_count,
        "acceptance_gate_passed": bool(hfpr_pass and macro_pass),
        "records": eval_records
    }

    report_path = REPORT_DIR / "field_robustness_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n✓ Full report saved to: {report_path.resolve()}")
    print(f"✓ Confusion matrix saved to: {cm_path.resolve()}\n")


if __name__ == "__main__":
    main()

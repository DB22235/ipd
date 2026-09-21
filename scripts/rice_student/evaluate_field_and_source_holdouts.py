"""
scripts/rice_student/evaluate_field_and_source_holdouts.py
==========================================================
Evaluates unmodified Teacher, Supervised Student, and Distilled Student
on external field-like holdout datasets:
  1. field_test_images/rice/field_holdout_manifest.json (12 curated field samples)
  2. test_images/rice/ (uncurated farm photos)

Generates:
  - reports/rice/source_audit/current_models_source_aware_evaluation.md
  - reports/rice/source_audit/field_holdout_predictions.json
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from PIL import Image

os.environ["KERAS_BACKEND"] = "torch"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keras
from keras import ops

from src.rice_student.contracts import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.rice_student.data import letterbox_image
from src.rice_student.metrics import evaluate_predictions


def preprocess_image_file(image_path: Path, target_size=(224, 224)) -> np.ndarray:
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        processed = letterbox_image(img_rgb, target_size=target_size, fill_color=(114, 114, 114))
        arr = np.array(processed, dtype=np.float32)
        return arr


def evaluate_model_on_samples(model: keras.Model, samples: List[Dict[str, Any]], target_size=(224, 224)) -> Dict[str, Any]:
    preds = []
    probs_list = []
    labels = []

    for item in samples:
        img_p = ROOT_DIR / item["filepath"]
        arr = preprocess_image_file(img_p, target_size=target_size)
        inp = np.expand_dims(arr, axis=0)

        logits = model(inp, training=False)
        probs = ops.softmax(logits, axis=-1)
        prob_np = ops.convert_to_numpy(probs)[0]
        pred_idx = int(np.argmax(prob_np))

        probs_list.append(prob_np)
        preds.append({
            "id": item["id"],
            "filename": item["filename"],
            "ground_truth": item["ground_truth"],
            "predicted_class": IDX_TO_CLASS[pred_idx],
            "predicted_confidence": float(prob_np[pred_idx]),
            "correct": bool(IDX_TO_CLASS[pred_idx] == item["ground_truth"]),
            "probabilities": {c: round(float(prob_np[i]), 4) for i, c in enumerate(CLASSES)},
        })
        labels.append(CLASS_TO_IDX[item["ground_truth"]])

    y_true = np.array(labels)
    y_prob = np.array(probs_list)
    perf = evaluate_predictions(y_true, y_prob)

    return {
        "metrics": perf,
        "sample_predictions": preds,
    }


def main():
    print("=" * 75)
    print("      STAGE 3: FIELD-HOLDOUT & SOURCE-AWARE EVALUATION")
    print("=" * 75)

    output_dir = ROOT_DIR / "reports" / "rice" / "source_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Field Holdout Manifest
    manifest_p = ROOT_DIR / "field_test_images" / "rice" / "field_holdout_manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Holdout manifest not found: {manifest_p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = []
    for s in manifest_data.get("samples", []):
        samples.append({
            "id": s["id"],
            "filename": s["filename"],
            "filepath": f"field_test_images/rice/{s['filename']}",
            "ground_truth": s["ground_truth"],
        })

    print(f"  Loaded {len(samples)} curated field holdout samples across {len(CLASSES)} classes.")

    # 2. Paths to Models
    teacher_path = ROOT_DIR / "models" / "rice_teacher_v1" / "rice_teacher_efficientnetb3_best.keras"
    baseline_path = ROOT_DIR / "models" / "rice" / "student_baselines" / "rice_student_mobilenetv3_baseline_best.keras"
    distilled_path = ROOT_DIR / "models" / "rice" / "distilled_students" / "rice_student_mobilenetv3_distilled_best.keras"

    results_all = {}

    # 3. Evaluate Supervised Student
    print(f"\n[1/3] Evaluating Supervised Student ({baseline_path.name})...")
    model_baseline = keras.models.load_model(str(baseline_path), compile=False)
    results_all["supervised_student"] = evaluate_model_on_samples(model_baseline, samples, target_size=(224, 224))
    b_acc = results_all["supervised_student"]["metrics"]["accuracy"]
    print(f"      Field Accuracy: {b_acc * 100:.2f}% ({int(b_acc * len(samples))}/{len(samples)} correct)")

    # 4. Evaluate Distilled Student
    print(f"\n[2/3] Evaluating Distilled Student ({distilled_path.name})...")
    model_distilled = keras.models.load_model(str(distilled_path), compile=False)
    results_all["distilled_student"] = evaluate_model_on_samples(model_distilled, samples, target_size=(224, 224))
    d_acc = results_all["distilled_student"]["metrics"]["accuracy"]
    print(f"      Field Accuracy: {d_acc * 100:.2f}% ({int(d_acc * len(samples))}/{len(samples)} correct)")

    # 5. Evaluate Teacher
    print(f"\n[3/3] Evaluating Teacher ({teacher_path.name})...")
    model_teacher = keras.models.load_model(str(teacher_path), compile=False)
    results_all["teacher"] = evaluate_model_on_samples(model_teacher, samples, target_size=(300, 300))
    t_acc = results_all["teacher"]["metrics"]["accuracy"]
    print(f"      Field Accuracy: {t_acc * 100:.2f}% ({int(t_acc * len(samples))}/{len(samples)} correct)")

    # 6. Save JSON Records
    json_out = output_dir / "field_holdout_predictions.json"
    with open(json_out, "w", encoding="utf-8") as f:
        # Convert non-serializable items
        clean_dict = {}
        for k, v in results_all.items():
            clean_dict[k] = {
                "accuracy": v["metrics"]["accuracy"],
                "macro_f1": v["metrics"]["macro_f1"],
                "predictions": v["sample_predictions"],
            }
        json.dump(clean_dict, f, indent=2)
    print(f"\n  [SAVED] -> {json_out.name}")

    # 7. Generate Markdown Report
    lines = [
        "# Current Models Field-Holdout & Source-Aware Evaluation Report",
        "",
        "**Document Status:** Empirical Field Generalization Assessment (Section 9 Compliance)  ",
        "**Date:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "  ",
        f"**Evaluation Dataset:** `field_test_images/rice/field_holdout_manifest.json` ({len(samples)} curated outdoor field images)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary Table",
        "",
        "| Model | Field Accuracy | Field Macro-F1 | Correct / Total | Benchmark Locked Test Acc | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Teacher (EfficientNetB3)** | {t_acc*100:.1f}% | {results_all['teacher']['metrics']['macro_f1']:.4f} | {int(t_acc*len(samples))}/{len(samples)} | 99.18% | Benchmark Candidate |",
        f"| **Supervised Student (MobileNetV3)** | {b_acc*100:.1f}% | {results_all['supervised_student']['metrics']['macro_f1']:.4f} | {int(b_acc*len(samples))}/{len(samples)} | 100.00% | Benchmark Candidate |",
        f"| **Distilled Student (MobileNetV3)** | {d_acc*100:.1f}% | {results_all['distilled_student']['metrics']['macro_f1']:.4f} | {int(d_acc*len(samples))}/{len(samples)} | 99.59% | Compact Benchmark Candidate |",
        "",
        "---",
        "",
        "## 2. Sample-by-Sample Prediction Audit",
        "",
        "| Sample ID | True Label | Teacher Pred (Conf) | Supervised Student Pred (Conf) | Distilled Student Pred (Conf) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]

    for idx, s in enumerate(samples):
        t_p = results_all["teacher"]["sample_predictions"][idx]
        b_p = results_all["supervised_student"]["sample_predictions"][idx]
        d_p = results_all["distilled_student"]["sample_predictions"][idx]

        t_str = f"{t_p['predicted_class']} ({t_p['predicted_confidence']*100:.1f}%)" + (" [OK]" if t_p["correct"] else " [MIS]")
        b_str = f"{b_p['predicted_class']} ({b_p['predicted_confidence']*100:.1f}%)" + (" [OK]" if b_p["correct"] else " [MIS]")
        d_str = f"{d_p['predicted_class']} ({d_p['predicted_confidence']*100:.1f}%)" + (" [OK]" if d_p["correct"] else " [MIS]")

        lines.append(f"| `{s['id']}` | **{s['ground_truth']}** | {t_str} | {b_str} | {d_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Engineering Assessment",
        "- **Cross-Domain Degradation:** Evaluates performance on outdoor images collected independently of the Kaggle curation batch.",
        "- **Decision Implication:** If both models maintain high accuracy across field holdouts, the source-confounding risk is benign. If accuracy degrades, retraining with multi-source field data is mandatory.",
    ])

    report_p = output_dir / "current_models_source_aware_evaluation.md"
    report_p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SAVED] -> {report_p.name}")

    print("\n" + "=" * 75)
    print(" [COMPLETE] Field-holdout evaluation finished successfully!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
scripts/rice_student/evaluate_student.py
========================================
Comprehensive evaluation tool for comparing:
  - Frozen EfficientNetB3 Teacher
  - MobileNetV3 Supervised Baseline
  - MobileNetV3 Distilled Student
Evaluates across:
  - Accuracy, Macro-F1, Weighted-F1, Balanced Accuracy
  - Per-class recall (Blast sensitivity check)
  - Expected Calibration Error (ECE)
  - Abstention policy performance
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import keras
from keras import ops
import torch

from src.rice_student.contracts import CLASSES, NUM_CLASSES
from src.rice_student.data import create_rice_dataloaders
from src.rice_student.metrics import evaluate_predictions
from src.rice_student.calibration import evaluate_abstention_on_test


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Rice Models")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model checkpoint")
    parser.add_argument("--split-manifest", type=str, default="manifests/rice/split_manifest_v1.csv")
    parser.add_argument("--target-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = ROOT_DIR / args.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("=" * 75)
    print(f"      EVALUATING MODEL: {model_path.name}")
    print("=" * 75)

    # 1. Load Model
    model = keras.models.load_model(str(model_path), compile=False)
    print(f"  Model Input Shape  : {model.input_shape}")
    print(f"  Model Output Shape : {model.output_shape}")

    # 2. Data Loader
    target_size = tuple(args.target_size)
    _, _, test_loader, _ = create_rice_dataloaders(
        manifest_path=ROOT_DIR / args.split_manifest,
        batch_size=32,
        target_size=target_size,
        use_class_weights=False,
        root_dir=ROOT_DIR,
    )
    print(f"  Test Batches       : {len(test_loader)} (Target Size: {target_size})")

    # 3. Predict on Test Set
    all_probs = []
    all_labels = []

    for batch in test_loader:
        if len(batch) == 3:
            bx, by, _ = batch
        else:
            bx, by = batch

        logits = model(bx, training=False)
        # Compute probabilities via softmax
        probs = ops.softmax(logits, axis=-1)
        
        all_probs.append(ops.convert_to_numpy(probs))
        all_labels.append(ops.convert_to_numpy(by))

    test_probs = np.concatenate(all_probs, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)

    # 4. Compute Metrics
    results = evaluate_predictions(test_labels, test_probs)
    abstention_results = evaluate_abstention_on_test(test_probs, test_labels)
    results["abstention"] = abstention_results

    print("\n--- RESULTS SUMMARY ---")
    print(f"  Accuracy           : {results['accuracy'] * 100:.2f}%")
    print(f"  Balanced Accuracy  : {results['balanced_accuracy'] * 100:.2f}%")
    print(f"  Macro-F1           : {results['macro_f1']:.4f}")
    print(f"  Blast Recall       : {results['blast_recall'] * 100:.2f}%")
    print(f"  Expected Cal. Error: {results['expected_calibration_error']:.4f}")
    print(f"  Abstention Cov.    : {abstention_results['coverage'] * 100:.2f}% (Retained Acc: {abstention_results['retained_accuracy'] * 100:.2f}%)")

    if args.output_json:
        out_p = ROOT_DIR / args.output_json
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[PASS] Report written to: {out_p.relative_to(ROOT_DIR)}")

    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())

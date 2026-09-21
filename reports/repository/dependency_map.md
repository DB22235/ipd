# Repository Dependency & Cross-Reference Map

Generated via AST Python parsing and path literal extraction. Identifies what imports or references each module to prevent moving high-risk dependencies.

## 1. Python Module Inward and Outward Dependencies

| Module Path | Imported Modules | Internal Path References | Move Risk |
| :--- | :--- | :--- | :---: |
| `evaluate_rice_robustness.py` | argparse, json, keras, matplotlib, matplotlib.pyplot, numpy, os, seaborn, sys, torch, pathlib.Path, sklearn.metrics.classification_report, sklearn.metrics.confusion_matrix, sklearn.metrics.f1_score, src.rice.preprocessor.preprocess_rice_leaf (+2 more) | `
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
`, `  - audit_reports/rice/field_confusion_matrix.png`, `  - audit_reports/rice/test_confusion_matrix.png` | **MEDIUM** |
| `leaf_isolator.py` | cv2, numpy, sys, PIL.Image, pathlib.Path, scipy.ndimage, typing.Any, typing.Dict (+2 more) | None | **HIGH** |
| `run_inference.py` | argparse, json, keras, keras.src.initializers.random_initializers, matplotlib, matplotlib.patches, matplotlib.pyplot, numpy, os, sys, tensorflow, PIL.Image, leaf_isolator.isolate_leaf, pathlib.Path, src.augmentations.build_field_robust_augmentation, src.rice.preprocessor.preprocess_rice_leaf | `Path to .keras model file (auto-detects crop if omitted)`, `audit_reports`, `models` | **HIGH** |
| `train_local_rice_efficientnetb3.py` | argparse, json, keras, matplotlib, matplotlib.pyplot, numpy, os, seaborn, sys, time, torch, keras.callbacks, keras.layers, keras.models, keras.optimizers, keras.regularizers (+8 more) | `clean_dataset`, `models`, `rice_teacher_efficientnetb3_best.keras` | **HIGH** |
| `archive/legacy_colab/testmodels/efficientnet b3 test.py` | json, matplotlib.pyplot, numpy, os, pandas, seaborn, tensorflow, warnings, datetime.datetime, google.colab.drive, sklearn.metrics.classification_report, sklearn.metrics.confusion_matrix, sklearn.metrics.f1_score (+31 more) | `
✅ Available models:`, `/content/drive/MyDrive/ipd_sem5/dataset/tomato_clean_dataset`, `_final.keras` | **LOW** |
| `archive/legacy_tests/diagnose_tomato.py` | numpy, tensorflow, PIL.Image, leaf_isolator.isolate_leaf, pathlib.Path | `clean_dataset`, `models`, `tomato_teacher_efficientnetb3.keras` | **LOW** |
| `archive/legacy_tests/organize_workspace.py` | shutil, pathlib.Path | `
organize_workspace.py
=====================
Cleans and organizes the project workspace into a professional structure:
  1. audit_reports/  <- Moves all generated visual audit and Grad-CAM PNG plots
  2. test_images/    <- Moves all test leaf images (test*.png, test*.webp, test*.jpg, potatotest*)
  3. archive/        <- Moves old temporary test scripts (run_test3.py)
  4. Keeps core files clean in root: run_inference.py, leaf_isolator.py, models/, src/, etc.

Usage:
  python organize_workspace.py
`, `  - audit_reports/     (All Grad-CAM & diagnostic visual audit figures)`, `  - models/            (Saved teacher and student models)` | **LOW** |
| `archive/legacy_tests/run_test3.py` | json, keras, keras.src.initializers.random_initializers, matplotlib, matplotlib.pyplot, numpy, tensorflow, pathlib.Path | `models`, `potato_teacher_efficientnetb3.keras` | **LOW** |
| `archive/legacy_tests/test_potato_image.py` | argparse, json, matplotlib.pyplot, numpy, sys, tensorflow, pathlib.Path | `
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
`, `Path to custom model .keras file`, `models` | **LOW** |
| `audits/audit_background_sensitivity.py` | json, keras, numpy, os, sys, PIL.Image, PIL.ImageFilter, leaf_isolator.isolate_leaf, pathlib.Path | `audit_reports`, `models`, `tomato_teacher_efficientnetb3.keras` | **MEDIUM** |
| `audits/audit_crop_localization.py` | json, numpy, os, random, sys, PIL.Image, leaf_isolator.isolate_leaf, pathlib.Path | `audit_reports`, `clean_dataset` | **MEDIUM** |
| `audits/audit_data_leakage.py` | hashlib, imagehash, json, os, sys, PIL.Image, collections.defaultdict, pathlib.Path, tqdm.tqdm | `audit_reports`, `clean_dataset` | **MEDIUM** |
| `audits/evaluate_field_robustness.py` | argparse, json, keras, matplotlib, matplotlib.pyplot, numpy, os, seaborn, sys, PIL.Image, leaf_isolator.isolate_leaf, pathlib.Path, sklearn.metrics.classification_report, sklearn.metrics.confusion_matrix (+1 more) | `
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
`, `Path to .keras model checkpoint`, `audit_reports` | **MEDIUM** |
| `audits/rice_calibration_audit.py` | json, keras, matplotlib, matplotlib.pyplot, numpy, os, sys, torch, pathlib.Path, src.rice.preprocessor.preprocess_rice_leaf | `
audits/rice_calibration_audit.py
================================
Formal Confidence Calibration & Abstention Policy Audit for Rice Disease Classifier.
Implements Section 8 of rice_post_training_evaluation.md.

Computes:
  1. Expected Calibration Error (ECE) across 10 confidence bins.
  2. Maximum Calibration Error (MCE).
  3. Reliability Diagram (empirical accuracy vs mean confidence + gap visualization).
  4. Confidence distributions for correct vs incorrect predictions.
  5. Selective accuracy & coverage sweep across abstention thresholds:
     tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98].

Outputs:
  reports/rice/calibration_report.json
  reports/rice/reliability_diagram.png
  reports/rice/abstention_threshold_selection.json
`, `PHASE 3 COMPLETE: CALIBRATION ARTIFACTS SAVED TO reports/rice/`, `clean_dataset` | **MEDIUM** |
| `audits/rice_domain_audit.py` | cv2, json, numpy, PIL.Image, pathlib.Path | `
audits/rice_domain_audit.py
===========================
Forensic Industrial ML Domain Discrepancy & Confounding Audit for Rice Dataset.
Measures:
  1. Dimension and file-size confounding across classes.
  2. Color channel biases (RGB / HSV) revealing background shortcuts.
  3. High-frequency spatial gradient variance (JPEG compression fingerprints).
  4. Near-duplicate group structure and family size distribution.
Outputs:
  audit_reports/rice/domain_discrepancy_report.json
`, `audit_reports`, `clean_dataset` | **MEDIUM** |
| `audits/rice_error_forensics.py` | csv, json, keras, matplotlib, matplotlib.pyplot, numpy, os, seaborn, sys, torch, PIL.Image, pathlib.Path, sklearn.metrics.balanced_accuracy_score, sklearn.metrics.classification_report, sklearn.metrics.confusion_matrix (+2 more) | `
audits/rice_error_forensics.py
==============================
Deep Forensic Error Analysis on Locked Rice Test Set (981 Images).
Isolates and catalogs all misclassified samples, calculates precision/recall/F1,
generates classification report CSV, and plots a diagnostic grid of the error cases.

Outputs:
  reports/rice/teacher_test_metrics.json
  reports/rice/teacher_test_classification_report.csv
  reports/rice/teacher_test_error_manifest.csv
  reports/rice/teacher_test_confusion_matrix.png
  reports/rice/error_cases_diagnostic_grid.png
`, `PHASE 1 COMPLETE: ERROR FORENSICS SAVED TO reports/rice/`, `clean_dataset` | **MEDIUM** |
| `audits/rice_shortcut_audit.py` | cv2, json, keras, matplotlib, matplotlib.pyplot, numpy, os, sys, torch, PIL.Image, PIL.ImageFilter, pathlib.Path, src.rice.preprocessor.preprocess_rice_leaf, src.rice.preprocessor.verify_rice_foliage | `
audits/rice_shortcut_audit.py
=============================
Causal Shortcut & Spurious Correlation Audit for Rice Disease Classifier.
Implements Section 7 of rice_post_training_evaluation.md.

Tests whether the model is causally driven by genuine leaf pathology
or by background soil/water/sky texture shortcuts.

Perturbation Variants for each test image:
  1. original              : Baseline letterboxed image
  2. background_blur       : Leaf blade preserved; background blurred (radius=5.0)
  3. background_darken     : Background luminance reduced by 60%
  4. background_brighten   : Background luminance increased by 40%
  5. neutral_background    : Background replaced with neutral gray (114, 114, 114)
  6. lesion_occlusion      : Primary necrotic lesion masked with healthy leaf green

Outputs:
  reports/rice/shortcut_audit_report.json
  reports/rice/shortcut_audit_examples.png
`, `PHASE 2 COMPLETE: CAUSAL SHORTCUT AUDIT SAVED TO reports/rice/`, `clean_dataset` | **MEDIUM** |
| `src/augmentations.py` | matplotlib.pyplot, numpy, tensorflow, pathlib.Path, tensorflow.keras.layers | None | **HIGH** |
| `src/background_augmentation.py` | cv2, numpy, tensorflow | None | **HIGH** |
| `src/dataset.py` | numpy, os, tensorflow, pathlib.Path, sklearn.utils.class_weight.compute_class_weight, typing.Dict, typing.List, typing.Tuple | None | **HIGH** |
| `src/dataset_pipeline.py` | hashlib, imagehash, io, json, logging, math, numpy, os, pandas, shutil, sys, PIL.Image, collections.defaultdict, pathlib.Path, tqdm.tqdm, typing.Dict (+3 more) | `
    Confirms that Rice___Healthy images are already in the manifest (they should
    be if rice_healthy_source_dir was included in build_pre_audit_manifest).
    Copies surviving (non-duplicate) healthy images to staging_dir for clarity.

    NOTE: Physical copy to staging_dir happens here but the FINAL directory
    write only happens after the leakage audit passes.

    Parameters
    ----------
    manifest : pd.DataFrame
        Current clean manifest (after exact dedup).
    rice_healthy_source_dir : Path
        Original Rice___Healthy source folder.
    staging_dir : Path
        Destination staging area, e.g. finaldataset/rice_dataset/_staging/healthy/

    Returns
    -------
    pd.DataFrame — unchanged manifest (integration is already reflected in it)
    `, `
    Walk all source directories and build a full pre-audit inventory.

    Parameters
    ----------
    source_dirs : dict
        A mapping of source_root_path → metadata dict with keys:
            "crop"        : str  e.g. "rice"
            "class_label" : str  e.g. "healthy"
            (optional) "source_override" : str
        Example:
            {
              "finaldataset/rice_dataset/train/blast":
                  {"crop": "rice", "class_label": "blast"},
              "Rice___Healthy":
                  {"crop": "rice", "class_label": "healthy",
                   "source_override": SOURCE_RICE_HEALTHY_FIELD},
            }
    output_csv : Path, optional
        If provided, saves the manifest to this CSV path.

    Returns
    -------
    pd.DataFrame
        One row per image file found.
    `, `
IPD Dataset Pipeline — Core Module
===================================
Phase 0 / Phase 1: Merge, Audit, Deduplicate, and Re-Split

Handles:
  - Rice___Healthy integration into rice_dataset
  - File integrity checks
  - SHA-256 exact duplicate removal
  - Perceptual hash (pHash) near-duplicate family detection
  - Group-stratified train/val/test splitting (70/15/15)
  - Cross-partition leakage audit (hash + pHash)
  - Final directory construction
  - Locked split manifest generation

IMPORTANT:
  - This module NEVER auto-runs. Call functions explicitly from build_dataset.ipynb.
  - No files are copied/moved until run_leakage_audit() returns True.
  - The split_manifest_v1.csv is written ONCE and never overwritten by the pipeline.

Author: IPD Model Engineering Agent
` | **HIGH** |
| `src/evaluate.py` | json, matplotlib.pyplot, numpy, tensorflow, pathlib.Path, sklearn.metrics.ConfusionMatrixDisplay, sklearn.metrics.accuracy_score, sklearn.metrics.auc, sklearn.metrics.balanced_accuracy_score (+9 more) | None | **HIGH** |
| `src/fast_finetune_potato.py` | json, matplotlib.pyplot, numpy, os, shutil, sys, tensorflow, time, pathlib.Path, sklearn.metrics.classification_report, sklearn.metrics.confusion_matrix, src.dataset.load_crop_datasets | `clean_dataset`, `models`, `potato_finetune_best.keras` | **MEDIUM** |
| `src/fast_finetune_tomato.py` | json, matplotlib, matplotlib.pyplot, numpy, os, shutil, sys, tensorflow, time, pathlib.Path, sklearn.metrics.classification_report, sklearn.metrics.confusion_matrix, src.background_augmentation.tf_randomize_background, src.dataset.load_crop_datasets | `clean_dataset`, `models`, `tomato_finetune_best.keras` | **MEDIUM** |
| `src/model.py` | tensorflow, src.augmentations.build_field_robust_augmentation, tensorflow.keras.layers, tensorflow.keras.models, typing.Tuple | None | **HIGH** |
| `src/train_teacher.py` | argparse, hashlib, json, matplotlib.pyplot, numpy, os, random, sys, tensorflow, torch, pathlib.Path, src.augmentations.visualize_augmentations, src.dataset.load_crop_datasets, src.evaluate.evaluate_teacher_model, src.model.build_teacher_model (+2 more) | `
IPD Phase 3: Field-Robust Teacher Model Training Pipeline
=========================================================
Trains crop-specific EfficientNetB3 Teacher Classifiers (Potato, Tomato, Rice).
Implements the 4-Phase Plan to bridge the Studio -> Field domain gap:
  - Phase A: Field-Robust Data Augmentation (Zoom-In Crop, Color Jitter, Random Erasing)
  - Phase B: Fresh Retraining with Regularized Head (Dropout 0.4) & Deep Fine-Tuning (Top 40 layers, LR 5e-5)
  - Phase C: Test set benchmark regression audit + Model Manifest Locking

Usage:
    # Train Potato Teacher (default: 20 epochs Stage A, 15 epochs Stage B):
    python -m src.train_teacher --crop potato

    # Train with custom epochs or on GPU:
    python -m src.train_teacher --crop potato --epochs-a 20 --epochs-b 15 --gpu

    # Train Tomato or Rice:
    python -m src.train_teacher --crop tomato
    python -m src.train_teacher --crop rice
`, `_stage_a_best.keras`, `_teacher_efficientnetb3.keras` | **HIGH** |
| `src/__init__.py` | None | None | **HIGH** |
| `src/rice/augmentations.py` | io, numpy, random, PIL.Image, PIL.ImageEnhance, PIL.ImageFilter, typing.Tuple | `
src/rice/augmentations.py
=========================
Anti-Shortcut Domain Invariance Augmentation Engine for Rice Leaves.
Neutralizes the 100% source-confounding vulnerability (256x256 ~5KB healthy vs
224x224 ~18KB diseased) through dynamic compression harmonization and robust perturbations.
` | **HIGH** |
| `src/rice/preprocessor.py` | cv2, numpy, PIL.Image, pathlib.Path, typing.Any, typing.Dict, typing.Tuple (+1 more) | `
src/rice/preprocessor.py
========================
Graminoid (Slender Grass) Leaf Preprocessing Module for Rice (*Oryza sativa*).
Solves the morphological mismatch of broadleaf GrabCut by employing:
  1. Aspect-Preserving Letterbox Resizing with neutral agricultural border fill.
  2. Botanical Foliage & Lesion Verification in HSV space.
  3. Out-of-Distribution / Non-Rice Rejection Filter.
` | **HIGH** |
| `src/rice/__init__.py` | augmentations.RiceAntiShortcutAugmentation, preprocessor.preprocess_rice_leaf, preprocessor.verify_rice_foliage | None | **HIGH** |
| `tools/audit_repository_structure.py` | ast, csv, hashlib, json, os, re, sys, time, collections.defaultdict, datetime.datetime, pathlib.Path | `
Writing formal reports to reports/repository/ ...`, `
tools/audit_repository_structure.py
===================================
Comprehensive Read-Only Forensic Repository Inventory, Dependency & Classification Engine.
Implements Phases 1 & 2 of repo_organization_agent_prompt.md.

Produces:
  reports/repository/repository_inventory.md
  reports/repository/file_inventory.csv
  reports/repository/large_files.csv
  reports/repository/hash_inventory.csv
  reports/repository/possible_secrets.md
  reports/repository/file_classification.csv
  reports/repository/dependency_map.md
  reports/repository/unknown_files.md
  reports/repository/path_risk_report.md
  reports/repository/model_registry.csv
  reports/repository/repository_migration_plan.md
  reports/repository/migration_manifest.csv
`, `- **LOW**: Leaf assets, documentation, standalone reports, images.
` | **LOW** |
| `tools/benchmark_throughput.py` | json, keras, numpy, os, sys, time, torch, keras.layers, keras.models, keras.optimizers, pathlib.Path | `
benchmark_throughput.py
=======================
Step 2 Diagnostic: 3-Tier Throughput Baseline Benchmark.
Isolates the performance bottleneck across:
  Tier 1: Pure Synthetic GPU Compute (in VRAM, zero I/O)
  Tier 2: In-Memory RAM-Cached Tensors (Compute + PCIe transfer, zero disk reads)
  Tier 3: Disk-Backed Generator (Compute + PCIe + Windows file I/O)

Measures steady-state steps/sec, images/sec, and batch latency (discarding warmup).
Saves output to reports/throughput_baseline.json.
`, `reports` | **LOW** |
| `tools/create_bounding_box_dataset.py` | argparse, concurrent.futures, os, sys, PIL.Image, leaf_isolator.isolate_leaf, pathlib.Path | `--src`, `clean_dataset\tomato_dataset` | **LOW** |
| `tools/diagnose_gpu_environment.py` | keras, os, subprocess, sys, torch, pathlib.Path | `
diagnose_gpu_environment.py
===========================
Step 1 Diagnostic: Hardware & Environment Verification.
Records CUDA availability, PyTorch build, Keras backend, and proves that
model parameters and tensors are physically allocated on the RTX 4050 GPU.
Saves output to reports/gpu_environment.txt.
`, `reports` | **LOW** |
| `tools/freeze_rice_teacher.py` | hashlib, json, os, sys, pathlib.Path | `
tools/freeze_rice_teacher.py
============================
Formally freezes the validated rice teacher checkpoint.
Implements Section 10 of rice_post_training_evaluation.md.

Computes:
  - SHA-256 checksum of rice_teacher_efficientnetb3_best.keras
  - Creates immutable release manifest: models/rice_teacher_v1/rice_teacher_v1_field_validated.json
  - Saves models/rice_teacher_v1/checksum.txt
`, `models`, `rice_teacher_efficientnetb3_best.keras` | **LOW** |
| `tools/train_local_efficientnetb3.py` | argparse, json, keras, matplotlib, matplotlib.pyplot, numpy, os, seaborn, sys, time, torch, keras.callbacks, keras.layers, keras.models, keras.optimizers, keras.regularizers (+5 more) | `models`, `stage1_warmup.keras`, `tomato_teacher_efficientnetb3_best.keras` | **MEDIUM** |
| `tools/verify_tflite_export.py` | json, keras, numpy, os, sys, tensorflow, time, keras.layers, keras.models, pathlib.Path | `audit_reports`, `temp_export_poc.keras` | **LOW** |

## 2. Inbound Reference Summary (Reverse Dependencies)

Shows critical modules that other scripts depend on:

| Target Dependency | Referenced By Count | Referencing Files |
| :--- | :---: | :--- |
| `models` | 17 | `evaluate_rice_robustness.py`, `run_inference.py`, `train_local_rice_efficientnetb3.py`, `archive/legacy_tests/diagnose_tomato.py` |
| `clean_dataset` | 13 | `evaluate_rice_robustness.py`, `train_local_rice_efficientnetb3.py`, `archive/legacy_tests/diagnose_tomato.py`, `audits/audit_crop_localization.py` |
| `rice_teacher_efficientnetb3_best.keras` | 8 | `evaluate_rice_robustness.py`, `run_inference.py`, `train_local_rice_efficientnetb3.py`, `audits/rice_calibration_audit.py` |
| `src.rice.preprocessor.preprocess_rice_leaf` | 6 | `evaluate_rice_robustness.py`, `run_inference.py`, `train_local_rice_efficientnetb3.py`, `audits/rice_calibration_audit.py` |
| `leaf_isolator.isolate_leaf` | 6 | `run_inference.py`, `archive/legacy_tests/diagnose_tomato.py`, `audits/audit_background_sensitivity.py`, `audits/audit_crop_localization.py` |
| `keras.models` | 4 | `train_local_rice_efficientnetb3.py`, `tools/benchmark_throughput.py`, `tools/train_local_efficientnetb3.py`, `tools/verify_tflite_export.py` |
| `src.dataset.load_crop_datasets` | 3 | `src/fast_finetune_potato.py`, `src/fast_finetune_tomato.py`, `src/train_teacher.py` |
| `keras.src.initializers.random_initializers` | 2 | `run_inference.py`, `archive/legacy_tests/run_test3.py` |
| `src.augmentations.build_field_robust_augmentation` | 2 | `run_inference.py`, `src/model.py` |
| `tensorflow.keras.models` | 2 | `archive/legacy_colab/testmodels/efficientnet b3 test.py`, `src/model.py` |
| `src` | 2 | `src/dataset_pipeline.py`, `tools/audit_repository_structure.py` |
| `train_local_rice_efficientnetb3.RiceDataset` | 1 | `evaluate_rice_robustness.py` |
| `
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
` | 1 | `evaluate_rice_robustness.py` |
| `  - audit_reports/rice/field_confusion_matrix.png` | 1 | `evaluate_rice_robustness.py` |
| `  - audit_reports/rice/test_confusion_matrix.png` | 1 | `evaluate_rice_robustness.py` |
| `rice_teacher_efficientnetb3.keras` | 1 | `run_inference.py` |
| `src.rice.augmentations.RiceAntiShortcutAugmentation` | 1 | `train_local_rice_efficientnetb3.py` |
| `
✅ Available models:` | 1 | `archive/legacy_colab/testmodels/efficientnet b3 test.py` |
| `/content/drive/MyDrive/ipd_sem5/dataset/tomato_clean_dataset` | 1 | `archive/legacy_colab/testmodels/efficientnet b3 test.py` |
| `
organize_workspace.py
=====================
Cleans and organizes the project workspace into a professional structure:
  1. audit_reports/  <- Moves all generated visual audit and Grad-CAM PNG plots
  2. test_images/    <- Moves all test leaf images (test*.png, test*.webp, test*.jpg, potatotest*)
  3. archive/        <- Moves old temporary test scripts (run_test3.py)
  4. Keeps core files clean in root: run_inference.py, leaf_isolator.py, models/, src/, etc.

Usage:
  python organize_workspace.py
` | 1 | `archive/legacy_tests/organize_workspace.py` |
| `  - models/            (Saved teacher and student models)` | 1 | `archive/legacy_tests/organize_workspace.py` |
| `  - src/               (Model architectures and augmentations)` | 1 | `archive/legacy_tests/organize_workspace.py` |
| `
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
` | 1 | `archive/legacy_tests/test_potato_image.py` |
| `
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
` | 1 | `audits/evaluate_field_robustness.py` |
| `
audits/rice_calibration_audit.py
================================
Formal Confidence Calibration & Abstention Policy Audit for Rice Disease Classifier.
Implements Section 8 of rice_post_training_evaluation.md.

Computes:
  1. Expected Calibration Error (ECE) across 10 confidence bins.
  2. Maximum Calibration Error (MCE).
  3. Reliability Diagram (empirical accuracy vs mean confidence + gap visualization).
  4. Confidence distributions for correct vs incorrect predictions.
  5. Selective accuracy & coverage sweep across abstention thresholds:
     tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98].

Outputs:
  reports/rice/calibration_report.json
  reports/rice/reliability_diagram.png
  reports/rice/abstention_threshold_selection.json
` | 1 | `audits/rice_calibration_audit.py` |
| `PHASE 3 COMPLETE: CALIBRATION ARTIFACTS SAVED TO reports/rice/` | 1 | `audits/rice_calibration_audit.py` |
| `
audits/rice_domain_audit.py
===========================
Forensic Industrial ML Domain Discrepancy & Confounding Audit for Rice Dataset.
Measures:
  1. Dimension and file-size confounding across classes.
  2. Color channel biases (RGB / HSV) revealing background shortcuts.
  3. High-frequency spatial gradient variance (JPEG compression fingerprints).
  4. Near-duplicate group structure and family size distribution.
Outputs:
  audit_reports/rice/domain_discrepancy_report.json
` | 1 | `audits/rice_domain_audit.py` |
| `
audits/rice_error_forensics.py
==============================
Deep Forensic Error Analysis on Locked Rice Test Set (981 Images).
Isolates and catalogs all misclassified samples, calculates precision/recall/F1,
generates classification report CSV, and plots a diagnostic grid of the error cases.

Outputs:
  reports/rice/teacher_test_metrics.json
  reports/rice/teacher_test_classification_report.csv
  reports/rice/teacher_test_error_manifest.csv
  reports/rice/teacher_test_confusion_matrix.png
  reports/rice/error_cases_diagnostic_grid.png
` | 1 | `audits/rice_error_forensics.py` |
| `PHASE 1 COMPLETE: ERROR FORENSICS SAVED TO reports/rice/` | 1 | `audits/rice_error_forensics.py` |
| `clean_dataset/rice_dataset/test` | 1 | `audits/rice_error_forensics.py` |
| `src.rice.preprocessor.verify_rice_foliage` | 1 | `audits/rice_shortcut_audit.py` |
| `
audits/rice_shortcut_audit.py
=============================
Causal Shortcut & Spurious Correlation Audit for Rice Disease Classifier.
Implements Section 7 of rice_post_training_evaluation.md.

Tests whether the model is causally driven by genuine leaf pathology
or by background soil/water/sky texture shortcuts.

Perturbation Variants for each test image:
  1. original              : Baseline letterboxed image
  2. background_blur       : Leaf blade preserved; background blurred (radius=5.0)
  3. background_darken     : Background luminance reduced by 60%
  4. background_brighten   : Background luminance increased by 40%
  5. neutral_background    : Background replaced with neutral gray (114, 114, 114)
  6. lesion_occlusion      : Primary necrotic lesion masked with healthy leaf green

Outputs:
  reports/rice/shortcut_audit_report.json
  reports/rice/shortcut_audit_examples.png
` | 1 | `audits/rice_shortcut_audit.py` |
| `PHASE 2 COMPLETE: CAUSAL SHORTCUT AUDIT SAVED TO reports/rice/` | 1 | `audits/rice_shortcut_audit.py` |
| `
    Confirms that Rice___Healthy images are already in the manifest (they should
    be if rice_healthy_source_dir was included in build_pre_audit_manifest).
    Copies surviving (non-duplicate) healthy images to staging_dir for clarity.

    NOTE: Physical copy to staging_dir happens here but the FINAL directory
    write only happens after the leakage audit passes.

    Parameters
    ----------
    manifest : pd.DataFrame
        Current clean manifest (after exact dedup).
    rice_healthy_source_dir : Path
        Original Rice___Healthy source folder.
    staging_dir : Path
        Destination staging area, e.g. finaldataset/rice_dataset/_staging/healthy/

    Returns
    -------
    pd.DataFrame — unchanged manifest (integration is already reflected in it)
    ` | 1 | `src/dataset_pipeline.py` |
| `
    Walk all source directories and build a full pre-audit inventory.

    Parameters
    ----------
    source_dirs : dict
        A mapping of source_root_path → metadata dict with keys:
            "crop"        : str  e.g. "rice"
            "class_label" : str  e.g. "healthy"
            (optional) "source_override" : str
        Example:
            {
              "finaldataset/rice_dataset/train/blast":
                  {"crop": "rice", "class_label": "blast"},
              "Rice___Healthy":
                  {"crop": "rice", "class_label": "healthy",
                   "source_override": SOURCE_RICE_HEALTHY_FIELD},
            }
    output_csv : Path, optional
        If provided, saves the manifest to this CSV path.

    Returns
    -------
    pd.DataFrame
        One row per image file found.
    ` | 1 | `src/dataset_pipeline.py` |
| `
IPD Dataset Pipeline — Core Module
===================================
Phase 0 / Phase 1: Merge, Audit, Deduplicate, and Re-Split

Handles:
  - Rice___Healthy integration into rice_dataset
  - File integrity checks
  - SHA-256 exact duplicate removal
  - Perceptual hash (pHash) near-duplicate family detection
  - Group-stratified train/val/test splitting (70/15/15)
  - Cross-partition leakage audit (hash + pHash)
  - Final directory construction
  - Locked split manifest generation

IMPORTANT:
  - This module NEVER auto-runs. Call functions explicitly from build_dataset.ipynb.
  - No files are copied/moved until run_leakage_audit() returns True.
  - The split_manifest_v1.csv is written ONCE and never overwritten by the pipeline.

Author: IPD Model Engineering Agent
` | 1 | `src/dataset_pipeline.py` |
| `src.background_augmentation.tf_randomize_background` | 1 | `src/fast_finetune_tomato.py` |
| `src.augmentations.visualize_augmentations` | 1 | `src/train_teacher.py` |
| `src.evaluate.evaluate_teacher_model` | 1 | `src/train_teacher.py` |
| `src.model.build_teacher_model` | 1 | `src/train_teacher.py` |
| `src.model.setup_stage_b_fine_tuning` | 1 | `src/train_teacher.py` |
| `
IPD Phase 3: Field-Robust Teacher Model Training Pipeline
=========================================================
Trains crop-specific EfficientNetB3 Teacher Classifiers (Potato, Tomato, Rice).
Implements the 4-Phase Plan to bridge the Studio -> Field domain gap:
  - Phase A: Field-Robust Data Augmentation (Zoom-In Crop, Color Jitter, Random Erasing)
  - Phase B: Fresh Retraining with Regularized Head (Dropout 0.4) & Deep Fine-Tuning (Top 40 layers, LR 5e-5)
  - Phase C: Test set benchmark regression audit + Model Manifest Locking

Usage:
    # Train Potato Teacher (default: 20 epochs Stage A, 15 epochs Stage B):
    python -m src.train_teacher --crop potato

    # Train with custom epochs or on GPU:
    python -m src.train_teacher --crop potato --epochs-a 20 --epochs-b 15 --gpu

    # Train Tomato or Rice:
    python -m src.train_teacher --crop tomato
    python -m src.train_teacher --crop rice
` | 1 | `src/train_teacher.py` |
| `
src/rice/augmentations.py
=========================
Anti-Shortcut Domain Invariance Augmentation Engine for Rice Leaves.
Neutralizes the 100% source-confounding vulnerability (256x256 ~5KB healthy vs
224x224 ~18KB diseased) through dynamic compression harmonization and robust perturbations.
` | 1 | `src/rice/augmentations.py` |
| `
src/rice/preprocessor.py
========================
Graminoid (Slender Grass) Leaf Preprocessing Module for Rice (*Oryza sativa*).
Solves the morphological mismatch of broadleaf GrabCut by employing:
  1. Aspect-Preserving Letterbox Resizing with neutral agricultural border fill.
  2. Botanical Foliage & Lesion Verification in HSV space.
  3. Out-of-Distribution / Non-Rice Rejection Filter.
` | 1 | `src/rice/preprocessor.py` |
| `preprocessor.preprocess_rice_leaf` | 1 | `src/rice/__init__.py` |
| `preprocessor.verify_rice_foliage` | 1 | `src/rice/__init__.py` |
| `- **PROTECTED**: Critical operating system shims (`powershell.cmd`), active dataset partitions (`clean_dataset`), and validated checkpoints.

` | 1 | `tools/audit_repository_structure.py` |
| `- `clean_dataset/`: Keep in place. Contains all active, verified partitions across Potato, Tomato, Rice.
` | 1 | `tools/audit_repository_structure.py` |
| `- `finaldataset/manifests/` $\to$ `manifests/rice/`

` | 1 | `tools/audit_repository_structure.py` |
| `- `leaf_isolator.py` $\to$ `src/preprocessing/leaf_isolator.py` (with root alias shim)
` | 1 | `tools/audit_repository_structure.py` |
| `- `models/potato_teacher/`, `models/tomato_teacher/`, `models/tomato_teacher_v3/`: Keep in place.
` | 1 | `tools/audit_repository_structure.py` |
| `- `models/rice_teacher_v1/`: Keep in place. Contains field-validated frozen teacher model.
` | 1 | `tools/audit_repository_structure.py` |
| `- `rice_post_training_evaluation.md` $\to$ `docs/specs/rice_post_training_evaluation.md`
` | 1 | `tools/audit_repository_structure.py` |
| `Gathers detailed metadata for all models in the repository.` | 1 | `tools/audit_repository_structure.py` |
| `Update default manifest path in src/dataset.py.` | 1 | `tools/audit_repository_structure.py` |
| `clean_dataset/potato_dataset` | 1 | `tools/audit_repository_structure.py` |
| `clean_dataset/rice_dataset` | 1 | `tools/audit_repository_structure.py` |
| `clean_dataset/tomato_dataset` | 1 | `tools/audit_repository_structure.py` |
| `docs/specs/rice_first_pilot_implementation.md` | 1 | `tools/audit_repository_structure.py` |
| `docs/specs/rice_post_training_evaluation.md` | 1 | `tools/audit_repository_structure.py` |
| `models/potato_teacher/potato_stage_a_best.keras` | 1 | `tools/audit_repository_structure.py` |
| `models/potato_teacher/potato_teacher_efficientnetb3.keras` | 1 | `tools/audit_repository_structure.py` |
| `models/registry/` | 1 | `tools/audit_repository_structure.py` |
| `models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras` | 1 | `tools/audit_repository_structure.py` |
| `models/rice_teacher_v1/stage1_warmup.keras` | 1 | `tools/audit_repository_structure.py` |
| `models/teacher/` | 1 | `tools/audit_repository_structure.py` |
| `models/tomato_teacher/tomato_teacher_efficientnetb3.keras` | 1 | `tools/audit_repository_structure.py` |
| `models/tomato_teacher_v3/tomato_teacher_efficientnetb3_best.keras` | 1 | `tools/audit_repository_structure.py` |
| `reports/models/` | 1 | `tools/audit_repository_structure.py` |
| `reports/rice/rice_teacher_post_training_report.md` | 1 | `tools/audit_repository_structure.py` |
| `run_inference.py, docs/specs/rice_first_pilot_implementation.md` | 1 | `tools/audit_repository_structure.py` |
| `src/common/` | 1 | `tools/audit_repository_structure.py` |
| `src/dataset.py, train_local_rice_efficientnetb3.py` | 1 | `tools/audit_repository_structure.py` |
| `src/preprocessing/` | 1 | `tools/audit_repository_structure.py` |
| `src/preprocessing/leaf_isolator.py` | 1 | `tools/audit_repository_structure.py` |
| `src/rice/` | 1 | `tools/audit_repository_structure.py` |
| `| `clean_dataset/` | Active, leakage-safe train/val/test partitions (Potato, Tomato, Rice) | **PROTECTED** |
` | 1 | `tools/audit_repository_structure.py` |
| `| `models/` | Trained teacher models across all 3 crops | **PROTECTED** / Active |
` | 1 | `tools/audit_repository_structure.py` |
| `| `src/` | Core library: preprocessors, augmentations, backbone architectures | Active |
` | 1 | `tools/audit_repository_structure.py` |
| `--src` | 1 | `tools/create_bounding_box_dataset.py` |
| `clean_dataset\tomato_dataset` | 1 | `tools/create_bounding_box_dataset.py` |
| `
tools/freeze_rice_teacher.py
============================
Formally freezes the validated rice teacher checkpoint.
Implements Section 10 of rice_post_training_evaluation.md.

Computes:
  - SHA-256 checksum of rice_teacher_efficientnetb3_best.keras
  - Creates immutable release manifest: models/rice_teacher_v1/rice_teacher_v1_field_validated.json
  - Saves models/rice_teacher_v1/checksum.txt
` | 1 | `tools/freeze_rice_teacher.py` |

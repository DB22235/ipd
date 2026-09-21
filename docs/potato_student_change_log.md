# Potato Student Model Change Log

All material engineering changes, pipeline refactors, and architectural updates for the Potato Student Model are documented chronologically here.

---

## [2026-09-19] - Stage 0: Scaffolding and Architecture Freeze

### Added
- **Core Package (`src/potato_student/`)**:
  - `contracts.py`: Authoritative class mappings (`0: early_blight`, `1: healthy`, `2: late_blight`), checksum verifiers, and shape definitions.
  - `data.py`: Ultra-fast RAM caching preloader, letterbox preprocessor, and inverse frequency class weights.
  - `models.py`: MobileNetV3-Large builder with linear logits, in-graph rescaling, and two-phase warmup controls.
  - `losses.py`: Supervised Cross-Entropy and temperature-scaled KL divergence distillation loss.
  - `distiller.py`: Custom Keras Model for knowledge distillation with frozen EfficientNetB3 potato teacher.
  - `metrics.py`: Accuracy, balanced accuracy, Macro-F1, per-class recall/precision, and confusion matrix calculator.
  - `calibration.py`: Temperature scaling, ECE computation, and safe abstention margin gating.
  - `export.py`: Native TensorFlow LiteRT converter (Float32, Float16, INT8).
  - `package_validator.py`: Keras vs LiteRT agreement testing and botanical leaf filter simulation.
- **Execution Scripts (`scripts/potato_student/`)**:
  - `audit_dataset.py`: Forensic auditor with pHash family clustering to uncover the PlantVillage healthy augmentation trap.
  - `create_split.py`: Group-isolated split manifest generator (`manifests/potato/potato_split_manifest_v1.csv`).
  - `validate_contract.py`: Integrity and schema validator.
  - `smoke_test_student.py`: Stage 0 synthetic forward/backward pass and distillation loss validator.
  - `train_student_baseline.py`: High-throughput supervised training script with RAM resident tensors.
  - `evaluate_student.py`: Standalone locked-test evaluation script.
  - `convert_student_litert.py`: LiteRT multi-format export and agreement validator.
  - `validate_mobile_package.py`: Mobile package checksum, signature, and abstention validator.
  - `train_student_distillation.py`: Optional Stage 2 knowledge distillation trainer.
- **Configurations (`configs/potato/`)**:
  - `student_baseline.yaml`: Hyperparameters for MobileNetV3-Large supervised baseline.
  - `teacher_contract.yaml`: Frozen contract for `potato_teacher_efficientnetb3.keras`.

### Changed / Refined Over Baseline Plan
- Standardized entirely on **TensorFlow / Keras (`KERAS_BACKEND=tensorflow`)** to eliminate framework conversion friction and enable native LiteRT export.
- Implemented **In-Memory RAM Preloading** at $(224 \times 224 \times 3)$ uint8 to eliminate disk I/O bottlenecks during CPU training.
- Designated **Float16 LiteRT** as the primary mobile deployment candidate, preventing the INT8 XNNPACK Node 124 fallback regression.

---

## [2026-09-19] - Stage 1: Baseline Supervised Training & Critical Bug Fixes

### Fixed
- **Double Rescaling Bug in MobileNetV3 (`src/potato_student/models.py`)**: Resolved severe dynamic range compression where `layers.Rescaling(1/255)` followed by MobileNetV3's internal `Rescaling(1/127.5, -1.0)` compressed 99.2% of pixel signal into `[-1.0, -0.9922]`. Replaced with `layers.Rescaling(1.0, dtype="float32")` for in-graph uint8-to-float32 casting while preserving the raw `[0, 255]` range.
- **Transitive pHash Chaining (`audit_dataset.py`, `create_split.py`)**: Tightened Hamming distance threshold from 10 down to $\le 4$ in union-find connected components, preventing 1,408 early blight images from collapsing into a single mega-cluster and starving the validation partition.

### Added
- Successfully completed supervised baseline training of MobileNetV3-Large (29 epochs, 4.4 minutes on multi-threaded Intel oneDNN CPU) with checkpoint saved at `models/potato/student_baselines/run_001/student_best.keras`.

---

## [2026-09-19] - Stage 2: Locked-Test Evaluation & Failure Mode Analysis

### Added
- Evaluated `student_best.keras` on the locked test split (1,049 samples):
  - Overall Accuracy: **99.43%** (1,043 / 1,049 correct)
  - Balanced Accuracy: **99.46%**
  - Macro-F1: **99.43%**
  - ECE: **0.0056**, Brier Score: **0.0099**
  - Early Blight Recall: **100.00%** (395 / 395)
  - Healthy Recall: **100.00%** (281 / 281) — Zero false negatives
  - Late Blight Recall: **98.39%** (367 / 373)
- Dissected all 6 Late Blight misclassifications in `reports/potato/student/late_blight_failure_analysis.md`: 3 early nascent lesions mimicking early blight target spots and 3 marginal tip specks on leaves that were $>95\%$ healthy green tissue.

---

## [2026-09-19] - Stage 3: LiteRT Multi-Format Export & Quantization Rejection

### Added
- Converted baseline checkpoint to 3 LiteRT formats:
  - `supervised_mobilenetv3_float32.tflite` (11.38 MB)
  - `supervised_mobilenetv3_float16.tflite` (5.76 MB)
  - `supervised_mobilenetv3_int8.tflite` (3.35 MB)
- Evaluated INT8 full integer quantization:
  - Identified severe Late Blight recall collapse (dropped to 71.31% on the locked test split, 107 false negatives).
  - Detected LiteRT XNNPACK delegate crash on Node 124 during runtime initialization, causing fallback to unvectorized single-threaded C++ reference kernels and exploding latency to 319.56 ms.
  - Permanently rejected INT8 for production deployment in `reports/potato/student/int8_evaluation_report.md`.
- Confirmed Float16 (5.76 MB) maintains **100.00% categorical decision parity** with Keras FP32, full SIMD acceleration (2.11 ms), and zero recall degradation.

---

## [2026-09-19] - Stage 4: Mobile Packaging & Safe Abstention Engine Tuning

### Added
- Established mobile deployment bundle in `mobile/potato/`:
  - Primary model: `supervised_mobilenetv3_float16.tflite` (SHA-256: `f3b3620ea3363f5503da92d89a41ed7316b242d4447015bf59ec5efd416ad83c`)
  - Metadata & contracts: `model_manifest.json`, `labels.txt`, `preprocessing.md`, `checksum.sha256`
- Implemented 3-stage safe abstention engine:
  - Botanical foliage gate: HSV mask calibrated to $H \in [20, 95]$, $S \ge 30$, $V \ge 30$ to accommodate chlorotic/yellowing foliage without false rejections.
  - Optical blur filter: Grayscale Laplacian variance threshold $\sigma^2_{\text{Laplacian}} < 40.0$.
  - Decision margin gate: Top-1 confidence $\tau_{\text{conf}} \ge 0.60$ and Top-1 vs Top-2 margin gap $\tau_{\text{margin}} \ge 0.20$.
- Registered formal release status: `potato supervised student — benchmark validated, leakage checks completed, Float16 package validated, limited external evidence, prototype integration approved`.

---

## [2026-09-19] - Stage 5: Full Multi-Format Same-Manifest Benchmark & Master Certification

### Added
- **Same-Manifest Benchmark (`scripts/potato_student/evaluate_all_formats_same_manifest.py`)**:
  - Concurrently evaluated all 4 formats (Keras FP32, LiteRT Float32, LiteRT Float16, LiteRT INT8) on identical 1,049 test samples.
  - Produced `manifests/potato/all_format_evaluation_manifest.csv` and `reports/potato/conversion/all_format_same_manifest_report.md`.
  - Implemented persistent probability caching in `manifests/potato/.cache/*.npy`.
- **Evaluation Provenance Audit (`scripts/potato_student/audit_evaluation_provenance.py`)**:
  - Formally proved 0 file overlap, 0 SHA-256 duplicate leakage, and 0 pHash cross-split leakage in `reports/potato/evaluation/test_provenance_audit.md`.
- **pHash Distance Review (`scripts/potato_student/review_phash_pairs.py`)**:
  - Audited 10 stratified image pairs across Hamming strata $0–2, 3–4, 5–6, \ge 7$ in `reports/potato/source_audit/phash_manual_review.md`.
- **Host Hardware Latency Profiling (`scripts/potato_student/benchmark_potato_mobile.py`)**:
  - Measured 2.11 ms warm median latency, 2.18 ms total turnaround, 26.24 MB RSS memory in `reports/potato/mobile/potato_target_device_benchmark.md`.
- **Decoupled Abstention Audit (`scripts/potato_student/audit_abstention_v2.py`)**:
  - Proved 99.33% coverage and 100.00% selective accuracy (filtered 100% of test split misclassifications) in `reports/potato/mobile/potato_abstention_gate_audit_v2.md`.
- **Expanded Real-Image Smoke Test (`scripts/potato_student/smoke_test_real_images_v2.py`)**:
  - Evaluated 20 real/field challenge images across all 13 Manus-mandated fields in `manifests/potato/potato_real_image_smoke_manifest_v2.csv` and `reports/potato/mobile/potato_real_image_smoke_test_v2.md`.
- **Authoritative Single Master Report**:
  - Synthesized all findings, benchmarks, and corrections into `reports/potato/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md`.

---

## [2026-09-20] - Stage 6: Failure-Focused Robustness Plan & 3-Tier Edge Mitigations

### Added
- **Formal Failure Registry (`manifests/potato/potato_failure_registry_v1.csv`)**:
  - Cataloged verified real-image false-healthy cases (`potatotest.png`, `potatotest2_cropped.png`) with visual pathology annotations, lesion area percentages, and Global Average Pooling (GAP) dilution factors.
- **Forensic Grad-CAM Saliency Audit (`scripts/potato_student/audit_failure_gradcam.py`)**:
  - Built diagnostic script targeting MobileNetV3's final convolutional layer (`Conv_1`) to map spatial attention distributions and verify that GAP spatial averaging is the active failure mechanism on small nascent lesions.
- **Failure-Focused Evaluation Set & Specification**:
  - Manifest created at `manifests/potato/potato_failure_focused_eval_v1.csv` with all 15 Manus-mandated metadata fields across 4 distinct strata.
  - Evaluation protocol specified in `reports/potato/student/potato_failure_focused_evaluation_plan.md`.
  - Automated evaluator implemented in `scripts/potato_student/evaluate_failure_focused_set.py`.
- **Physical Target-Device ADB Benchmark Suite (`scripts/potato_student/benchmark_physical_device_adb.py`)**:
  - Built turnkey ADB automation tool measuring CPU (1, 2, 4 threads), GPU, and NNAPI delegates, cold start, warm P50/P95/P99 latency, and thermal stability over 200 consecutive runs.
- **App Team Integration Handoff (`reports/potato/mobile/potato_app_team_integration_handoff.md`)**:
  - Documented Tier 1 Camera Viewfinder Targeting Reticle (forces lesion to fill $\ge 25\%$ of box, boosting active convolutional cells from 1 to $\ge 12$).
  - Documented Tier 2 Client-Side Dual-Scale Inference Shim (evaluates full letterbox + central crop).
  - Enforced explicit UI Potato Mode to prevent cross-crop inputs (e.g. rice leaves) from triggering false-healthy diagnoses.
- **Enhanced Master Architectural Strategy**:
  - Synthesized all findings and decision gates into `reports/potato/student/POTATO_MODEL_ENHANCED_NEXT_STEPS_PLAN.md`.

---

## [2026-09-20] - Stage 7: Complete 17-Test Protocol & 7-Level Release Gate Hierarchy

### Added
- **Complete 17-Test Protocol Execution (Manus AI Specification)**:
  - **Group A (Technical Correctness - Tests 1–5)**:
    - Test 1: Artifact & Checksum Integrity (`reports/potato/model_artifact_integrity_report.md`).
    - Test 2: Same-Manifest Reproducibility (1,049 samples, 100% agreement, `reports/potato/conversion/all_format_same_manifest_report.md`).
    - Test 3: Provenance & Split Leakage Audit (`reports/potato/evaluation/test_provenance_audit_v2.md`).
    - Test 4: Target-Device Mobile Benchmark (`reports/potato/mobile/potato_target_device_benchmark_v2.md`).
    - Test 5: Preprocessing & Orientation Invariance (`scripts/potato_student/audit_orientation_invariance.py` & `reports/potato/mobile/potato_preprocessing_invariance_report.md`).
  - **Group B (Robustness Evidence - Tests 6–10)**:
    - Test 6: External Real-Image Evaluation (`manifests/potato/potato_external_evaluation_v1.csv`, `scripts/potato_student/evaluate_external_dataset.py`, `reports/potato/student/potato_external_evaluation_v1.md`).
    - Test 7: Failure-Focused Lesion Evaluation (`manifests/potato/potato_failure_focused_eval_v1.csv` & `reports/potato/student/potato_failure_focused_evaluation_v1.md`).
    - Test 8: Crop Mismatch & OOD Evaluation (`manifests/potato/potato_ood_crop_eval_v1.csv`, `scripts/potato_student/evaluate_ood_crops.py`, `reports/potato/student/potato_ood_crop_evaluation_v1.md`).
    - Test 9: Decoupled Abstention Engine v3 (`scripts/potato_student/audit_abstention_gate_v3.py` & `reports/potato/mobile/potato_abstention_gate_evaluation_v3.md`).
    - Test 10: Calibration Robustness & ECE (`scripts/potato_student/audit_calibration_robustness.py` & `reports/potato/student/potato_calibration_robustness_v1.md`).
  - **Group C (Operational Readiness - Tests 11–17)**:
    - Test 11: Repeatability & Determinism (`scripts/potato_student/audit_runtime_repeatability.py` & `reports/potato/mobile/potato_runtime_repeatability_report.md`).
    - Test 12: Memory, Thermal & Repeated-Use Stability (`reports/potato/mobile/potato_stability_and_thermal_report.md`).
    - Test 13: App-Model Contract Integration (`reports/potato/mobile/potato_app_model_contract_test_v1.md`).
    - Test 14: User Capture Workflow Evaluation (`scripts/potato_student/simulate_reticle_capture.py` & `reports/potato/mobile/potato_capture_workflow_evaluation_v1.md`).
    - Test 15: Model Card & Limitation Audit (`docs/potato_student_model_card_v2.md`).
    - Test 16: Rollback & Artifact Reproducibility (`scripts/potato_student/test_rollback_reproducibility.py` & `reports/potato/release/potato_artifact_reproducibility_and_rollback_test.md`).
    - Test 17: Monitoring Design for Future Updates (`reports/potato/operations/potato_model_monitoring_and_update_plan.md`).
- **Official 7-Level Release Gate Hierarchy Registered**:
  - Gate 1 (Benchmark Validated): **PASSED**
  - Gate 2 (Package Validated): **PASSED**
  - Gate 3 (Controlled Prototype): **APPROVED**
  - Gate 4 (Device Validated): **PENDING** (Host profile verified; ADB runner ready)
  - Gate 5 (Robustness Candidate): **PENDING** (90.9% external accuracy; reticle mitigates small lesion risk)
  - Gate 6 (Field-Validation Candidate): **PENDING** (Formal multi-region farm collection needed)
  - Gate 7 (Production Candidate): **NOT APPROVED**
- **Authoritative Single Master Deliverable**:
  - Updated `reports/potato/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md` to seal all 17 tests, release gates, and operational decisions into the single definitive document for Manus AI.


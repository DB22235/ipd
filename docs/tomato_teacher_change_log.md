# Tomato Teacher Model Change Log

This document records all architectural, data, training, and evaluation changes for the Tomato Teacher Model.

---

## [2026-09-20] - Milestone: Tomato Teacher v2 Retraining & Validation

### Added
- **v1 Baseline Preservation:**
  - Immutably registered `tomato_teacher_v1_benchmark_candidate` in `models/tomato/model_registry.json`.
- **Pre-Training Data & Shortcut Audit (`scripts/tomato/audit_dataset_v2.py`)**:
  - Audited 4,322 tomato images across decodability, exact duplicates, and pHash families.
  - Formally measured color space coordinates ($H, S, V, L^*, a^*, b^*$), proving the presence of an olive-green vs lime-green shortcut in legacy data.
  - Generated `reports/tomato/teacher_v2/data_audit_report.md`, `source_class_cross_tabulation.csv`, `color_distribution_report.json`, and `duplicate_report.csv`.
- **Label Provenance & Field Holdout Isolation**:
  - Curated 12 field challenge images with 15 Manus-mandated metadata fields in `manifests/tomato/teacher_v2/label_review_log.csv`.
  - Locked 12 verified samples into `manifests/tomato/teacher_v2/external_field_holdout.csv`.
- **Dataset v2 Group-Disjoint Splitting (`scripts/tomato/build_v2_manifests.py`)**:
  - Created 70% Train, 15% Validation, and 15% Locked Benchmark Test splits grouped by pHash families ($d \le 4$).
  - Certified zero leakage in `reports/tomato/teacher_v2/split_integrity_report.md`.
- **Preprocessing Contract**:
  - Established `manifests/tomato/teacher_v2_preprocessing_contract.json` specifying aspect-preserving letterbox with neutral gray padding $(114, 114, 114)$ at $300 \times 300 \times 3$.
- **Two-Phase Retraining Pipeline (`scripts/tomato/train_teacher_v2.py`)**:
  - Implemented EfficientNetB3 Phase A head warmup (5 epochs, frozen backbone) and Phase B fine-tuning (20 epochs, top 35 MBConv7 layers unfrozen, BatchNorm frozen).
  - Injected hue/sat and brightness/contrast jitter to break color shortcuts.
- **Five Controlled Robustness Experiments (`scripts/tomato/evaluate_teacher_v2_experiments.py`)**:
  - Exp A (Corrected-Data Baseline): 98.92% locked benchmark test accuracy.
  - Exp B (Color Robustness): 98.2% decision stability under hue/saturation shifts.
  - Exp C (Preprocessing Comparison): 0% padding selected on validation data.
  - Exp D (Background Sensitivity): 100% stability across dark/white paddings; studio white backing accuracy restored from 0% to 100%.
  - Exp E (Quality Gate Safety): Decoupled foliage/blur gates and uncertainty margins verified.
- **Acceptance Gate Sign-Off & Unified Master Deliverable**:
  - All 5 acceptance gates verified in `reports/tomato/teacher_v2/teacher_go_no_go_decision.md`.
  - Authoritative single report synthesized at `reports/tomato/TOMATO_TEACHER_V2_UNIFIED_MASTER_REPORT_FOR_MANUS.md`.

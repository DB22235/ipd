# Rice Student Model Change Log

Conforms to Section 13 (Change-control protocol) of `rice_student_model_development_plan.md`.

---

## Change CHG-RICE-001 — Scaffolding and Contract Hardening

- **Date:** 2026-09-18
- **Author/agent:** Antigravity Coding Assistant
- **Files changed:**
  - `src/rice_student/__init__.py`
  - `src/rice_student/contracts.py`
  - `src/rice_student/data.py`
  - `src/rice_student/models.py`
  - `src/rice_student/losses.py`
  - `src/rice_student/distiller.py`
  - `src/rice_student/metrics.py`
  - `src/rice_student/calibration.py`
  - `src/rice_student/export.py`
  - `src/rice_student/package_validator.py`
  - `configs/rice/student_baseline.yaml`
  - `configs/rice/student_distillation.yaml`
  - `configs/rice/mobile_contract.yaml`
  - `scripts/rice_student/validate_contract.py`
  - `scripts/rice_student/smoke_test_student.py`
  - `scripts/rice_student/train_student_baseline.py`
  - `scripts/rice_student/train_student_distillation.py`
  - `scripts/rice_student/evaluate_student.py`
- **Reason:** Implement the robust MobileNetV3 rice student model development codebase, guarding against class order mismatch, multi-crop data infiltration, double normalization, missing $T^2$ scaling, and operator incompatibilities.
- **Previous behavior:** No dedicated rice student package existed; risk of label inversion if canonical text order was applied rather than frozen teacher alphabetical order.
- **New behavior:** Authoritative contract enforces `0: blast, 1: blight, 2: brown_spot, 3: healthy`. Manifest loader filters strictly `crop == 'rice'` (4,932 samples). Loss enforces $T^2$ scaling. Distiller freezes teacher and bridges resolution via bilinear upsample.
- **Data or model impact:** Zero impact on frozen teacher (`rice_teacher_v1`); guarantees mathematical distillation correctness.
- **Validation performed:** Stage 0 contract verification script and synthetic smoke test.
- **Runtime cost:** 0 GPU training compute.
- **Rollback method:** Remove `src/rice_student/` and `scripts/rice_student/`.
- **Related experiment:** `EXP-RICE-STU-001`, `EXP-RICE-STU-002`.

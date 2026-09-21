# Tomato Teacher Model Experiment Log

This document tracks all formal experiments, baseline runs, data audits, and field evaluations for the Tomato Teacher Model (EfficientNetB3).

---

## Experiment Index

| Exp ID | Date | Model / Role | Architecture | Split Version | Benchmark F1 | Field Acc | Decision |
|---|---|---|---|---|---|---|---|
| `EXP-TOMATO-001` | 2026-09-18 | Teacher v1 Baseline Candidate | EfficientNetB3 | v1 Standard Split | 98.50% | 50.0% (Severe Failure) | REJECT for Distillation; Preserved as baseline |
| `EXP-TOMATO-002` | 2026-09-20 | Pre-Training Data Audit & Shortcut Dissection | N/A (Data Audit) | 4,322 images | N/A | N/A | PASS: Confirmed olive vs lime shortcut & pHash families |
| `EXP-TOMATO-003` | 2026-09-20 | Dataset v2 & Field Holdout Isolation | Dataset v2 | 70/15/15 + 12 Field | N/A | N/A | PASS: Zero test leakage across all partitions |
| `EXP-TOMATO-004` | 2026-09-20 | Teacher v2 Two-Phase Retraining | EfficientNetB3 | Dataset v2 | 98.81% | 91.7% | PASS: Certified GO for Mobile Student Supervision |

---

## Entry `EXP-TOMATO-001`: Teacher v1 Baseline & Field Collapse Audit
- **Date:** 2026-09-18
- **Objective:** Evaluate legacy EfficientNetB3 candidate (`models/tomato_teacher/tomato_teacher_efficientnetb3.keras`).
- **Findings:**
  - Benchmark Test Accuracy: 98.65% (Macro-F1: 98.50%).
  - Field Holdout Accuracy: 50.0% (6 / 12 correct).
  - Healthy False-Positive Rate: 83.33% (All studio white paper images failed).
  - Expected Calibration Error: 0.4248.
- **Decision:** Formal NO-GO for mobile student distillation. Registered in `models/tomato/model_registry.json` as frozen comparison baseline.

---

## Entry `EXP-TOMATO-002`: Pre-Training Data Audit & Shortcut Dissection
- **Date:** 2026-09-20
- **Objective:** Audit 4,322 images across decodability, duplicates, source balance, and color space coordinates.
- **Findings:**
  - 100% valid images (0 corruptions).
  - 142 pHash duplicate families identified.
  - Confirmed color shortcut: `healthy` leaves have Value $V = 118.2$ and Lab $a^* = -22.1$ (dark saturated olive-green) vs `late_blight` Value $V = 141.6$ and Lab $a^* = -11.2$ (washed-out lime-green).
- **Decision:** PASS. Mandatory hue/sat jitter and white background diversification established.

---

## Entry `EXP-TOMATO-003`: Dataset v2 Group-Disjoint Splitting
- **Date:** 2026-09-20
- **Objective:** Partition 4,322 images into group-disjoint splits: 70% Train, 15% Validation, 15% Locked Benchmark Test, plus 12 isolated Field Holdout samples.
- **Findings:**
  - Zero filepath overlap.
  - Zero exact SHA-256 hash overlap.
  - Zero pHash family overlap ($d \le 4$).
- **Decision:** PASS. Split integrity certified in `reports/tomato/teacher_v2/split_integrity_report.md`.

---

## Entry `EXP-TOMATO-004`: Teacher v2 Retraining & Robustness Verification
- **Date:** 2026-09-20
- **Objective:** Execute two-phase transfer learning on EfficientNetB3 with aspect-preserving letterboxing and color/illumination jitter. Run Experiments A through E.
- **Empirical Results:**
  - Locked Benchmark Test Accuracy: **98.92%** (Macro-F1: **98.81%**).
  - External Field Holdout Accuracy: **91.7%** (11 / 12 correct, vs 50.0% historical).
  - Healthy False-Positive Rate: **8.3%** (vs 83.3% historical).
  - Studio White Backing Accuracy: **100.0%** (3 / 3 correct, vs 0% historical).
  - Background Invariance: **100.0%** (0 class flips under dark/white padding).
  - Field Expected Calibration Error: **0.0842** (vs 0.4248 historical).
- **Official Verdict:** GO. Teacher v2 satisfies all 5 acceptance gates and is authorized to supervise Tomato Mobile Student development.

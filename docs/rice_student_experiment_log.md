# Rice Student Model Experiment Log

This log records every configuration, contract check, synthetic dry run, and training execution for the Rice Student model (MobileNetV3-Large distilled from EfficientNetB3).

---

## Experiment Index

| Exp ID | Date | Target Model | Purpose | Status | Key Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-RICE-STU-001` | 2026-09-18 | Contracts & Smoke | Stage 0 Contract & Manifest Validation | Complete | Contracts 100% Passed |
| `EXP-RICE-STU-002` | 2026-09-18 | Architecture & KD | Stage 0 Synthetic 1-Batch Distillation | Complete | Loss finite, TFLite verified |
| `EXP-RICE-STU-003` | 2026-09-18 | MobileNetV3 Baseline | Stage 1 Supervised Training (Hard Labels) | Complete | Test Acc: 100.0%, F1: 1.0000 |
| `EXP-RICE-STU-004` | 2026-09-18 | MobileNetV3 Distilled | Stage 2 Knowledge Distillation ($T=3, \alpha=0.5$) | Complete | Test Acc: 99.59%, F1: 0.9937 |

---

## Detailed Entries

### `EXP-RICE-STU-001` — Contract and Manifest Verification
- **Date:** 2026-09-18
- **Author/Agent:** Antigravity Coding Assistant
- **Execution Mode:** Autonomous Phase 0 Check (CPU, 0 GPU hours)
- **Teacher Checkpoint:** `models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras`
- **Teacher SHA-256:** `2eca4294596905fb6231911b66cbeafc90ac8528ecd34461327b43074ec9e738`
- **Class Contract Verified:** `0: blast, 1: blight, 2: brown_spot, 3: healthy`
- **Manifest Path:** `manifests/rice/split_manifest_v1.csv`
- **Rice Samples:** Exactly 4,932 samples (Train: 3,315, Val: 636, Test: 981)
- **Outcome:** PASSED. Report: `reports/rice/student/contract_validation.json`

### `EXP-RICE-STU-002` — Synthetic Distillation Smoke Test & Operator Check
- **Date:** 2026-09-18
- **Author/Agent:** Antigravity Coding Assistant
- **Execution Mode:** Autonomous Phase 0 Check (PyTorch CUDA backend)
- **Student Architecture:** MobileNetV3-Large (3,000,196 parameters, pure linear logits)
- **Teacher Verification:** EfficientNetB3 loaded with 0 trainable weights (`trainable=False`, `training=False`)
- **Distillation Loss:** Verified finite positive value with $T^2$ scaling factor ($T=3.0, \alpha=0.5$)
- **LiteRT Check:** Model converted to TFLite using standard operators (no Flex delegates)
- **Outcome:** PASSED. Report: `reports/rice/student/smoke_test_report.md`

### `EXP-RICE-STU-003` — Stage 1 Supervised Student Baseline Training
- **Date:** 2026-09-18
- **Author/Agent:** User Execution on RTX 4050 GPU
- **Execution Mode:** Supervised Training (AdamW, lr=$10^{-4}$, weight decay=$10^{-4}$, batch size 16)
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (11.7 minutes, 15 epochs)
- **Checkpoint:** `models/rice/student_baselines/rice_student_mobilenetv3_baseline_best.keras` (34.9 MB)
- **Forensic Audit:** 8-point audit passed (0 hash collisions, 0 group collisions across partitions)
- **Locked Test Results (981 samples):** Accuracy: 100.0%, Macro-F1: 1.0000, Blast Recall: 100.0%
- **Outcome:** PASSED. Reports: `reports/rice/student/student_go_no_go_decision.md`

### `EXP-RICE-STU-004` — Stage 2 Knowledge Distillation Training
- **Date:** 2026-09-18
- **Author/Agent:** User Execution on RTX 4050 GPU
- **Distillation Hyperparameters:** Temperature $T=3.0$, $\alpha=0.5$, $T^2=9.0$, AdamW (lr=$10^{-4}$, wd=$10^{-4}$)
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (91.1 minutes, 14 epochs, best restored from Epoch 8)
- **Teacher Invariance:** Verified (0% mutation of SHA-256 `2eca429...`)
- **Checkpoint:** `models/rice/distilled_students/rice_student_mobilenetv3_distilled_best.keras` (12.1 MB)
- **Locked Test Results (981 samples):** Accuracy: 99.59%, Macro-F1: 0.9937, Blast Recall: 97.22%
- **Outcome:** PASSED. Reports: `reports/rice/distillation/distillation_test_comparison_report.md`

# Rice Student Model Development Plan and Codebase Specification

**Document status:** Authoritative planning and implementation instruction

**Project:** IPD dual-mode plant disease detection system

**Pilot crop:** Rice

**Student classes:** Healthy, Blast, Brown Spot, Blight

**Teacher:** Frozen, versioned EfficientNetB3 rice teacher after evaluation

**Primary objective:** Build, evaluate, distill, optimize, and package a rice student model for offline mobile inference without unnecessarily consuming the user’s GPU time or compute budget.

**Execution restriction:** AI agents may inspect, write, edit, lint, and perform small bounded checks, but they must not execute large training scripts, full-dataset evaluations, long conversion jobs, or expensive benchmark scripts unless the user explicitly starts or authorizes that execution.

---

## 1. Strategic decision

The rice student project may begin with the currently frozen rice teacher as an **experimental teacher source**, provided its version, checksum, class order, preprocessing, and evaluation status are recorded.

The student will be developed in controlled stages:

```text
teacher and dataset contract verification
→ student codebase scaffold
→ small smoke tests
→ supervised student baseline
→ controlled distillation
→ student comparison
→ bounded conversion test
→ quantization candidates
→ mobile artifact validation
→ user-authorized full training and device benchmark
```

The student must not be created by copying the teacher architecture blindly. The student is an independent mobile model selected for:

- Mobile latency.
- Model size.
- Peak memory.
- Accuracy and macro-F1.
- Per-class recall.
- Calibration.
- Abstention behavior.
- LiteRT/TensorFlow Lite operator compatibility.

The first student candidate is **MobileNetV3-Large**. MobileNetV3-Small and MobileNetV2 may be evaluated later if the mobile budget requires a smaller model.

---

## 2. Teacher contract

Before writing student training code, verify the teacher contract.

Required teacher metadata:

```text
teacher_model_id
teacher_model_path
teacher_sha256
teacher_crop
teacher_version
teacher_input_size
teacher_color_order
teacher_normalization
teacher_class_order
teacher_output_type
teacher_logits_or_probabilities
teacher_dataset_version
teacher_split_version
teacher_evaluation_status
```

The student and teacher must use the exact same:

- Crop.
- Class order.
- Input color order.
- Input dimensions where required by the experiment.
- Label encoding.
- Disease class definitions.

For rice, the canonical class order is:

```text
0: healthy
1: blast
2: brown_spot
3: blight
```

If the teacher’s actual class order differs, preserve the teacher order in the contract and update all student metadata consistently. Never silently reorder labels.

The teacher must remain frozen during distillation:

```text
teacher.trainable = false
teacher called with training = false
no teacher optimizer updates
no teacher checkpoint overwrite
```

The teacher is a source of soft targets. It is not automatically a source of truth. Ground-truth labels must remain part of the first production distillation objective.

---

## 3. Important execution restriction for the AI coding agent

The AI agent must not automatically execute expensive work.

### The agent may execute without additional approval

Only small, bounded, low-cost checks such as:

- Listing files.
- Reading configuration.
- Checking imports.
- Compiling a small module.
- Running a unit test with synthetic data.
- Loading one or a few images.
- Loading the model metadata.
- Checking tensor shapes.
- Running one or two inference examples.
- Running a dry-run that performs no training.
- Validating a manifest schema.
- Checking a model checksum.

### The agent must not execute automatically

The agent must write the script and provide the exact command, but wait for the user to run or explicitly authorize:

- Full student training.
- Full teacher evaluation.
- Full-dataset inference.
- Knowledge-distillation training.
- Hyperparameter sweeps.
- Cross-validation.
- Full quantization conversion.
- Large representative-dataset calibration.
- Mobile latency benchmarks.
- Long GPU benchmarks.
- Any script expected to run for more than a few minutes.
- Any script that consumes significant GPU, CPU, RAM, or storage.
- Any script that modifies or downloads large datasets.
- Any script that overwrites model artifacts.

The agent must estimate expected runtime and resource usage before requesting execution.

Use this pattern:

```text
Script prepared: <path>
Estimated runtime: <estimate>
Expected hardware: CPU/GPU
Expected RAM/VRAM: <estimate>
Outputs: <paths>
Risk: <low/medium/high>
User action required: run this command when ready
```

Never hide expensive work behind an import, notebook cell, startup hook, or test command.

---

## 4. CPU versus GPU policy

The student model is small enough that CPU development is acceptable for code validation and smoke tests. Full training may use CPU or GPU depending on the user’s preference and available time.

### Use CPU for

- Repository organization.
- Configuration checks.
- Manifest validation.
- Small smoke tests.
- One-batch teacher/student shape checks.
- Tiny synthetic distillation tests.
- Export-contract checks.
- Label mapping tests.
- Model-package validation.
- Documentation generation.
- Short inference checks.

### Use GPU when the user authorizes it for

- Full supervised student training.
- Full distillation training.
- Large image-resolution experiments.
- Quantization-aware training.
- Large-scale evaluation.
- Multiple-seed experiments.

### Prefer CPU when

- The student dataset is small and training time is acceptable.
- GPU queueing or setup overhead exceeds the actual training cost.
- The user wants to preserve GPU budget.
- The task is exploratory and only a quick baseline is required.
- The code path is not yet verified.

### Prefer GPU when

- CPU training is estimated to take substantially longer.
- Mixed precision is verified and stable.
- The data pipeline is already benchmarked.
- The user has authorized the run.
- The experiment has a clear acceptance purpose.

The agent must never assume that GPU is automatically better. It must report the estimated trade-off and allow the user to choose.

---

## 5. Required codebase structure

Create or adapt the following structure without duplicating existing functionality blindly:

```text
ipd/
├── configs/
│   └── rice/
│       ├── student_baseline.yaml
│       ├── student_distillation.yaml
│       ├── student_quantization.yaml
│       └── mobile_contract.yaml
├── manifests/
│   └── rice/
│       ├── rice_split_manifest_v1.csv
│       ├── rice_label_map.json
│       └── rice_dataset_contract.json
├── src/
│   └── rice_student/
│       ├── __init__.py
│       ├── contracts.py
│       ├── data.py
│       ├── models.py
│       ├── losses.py
│       ├── distiller.py
│       ├── metrics.py
│       ├── calibration.py
│       ├── export.py
│       └── package_validator.py
├── scripts/
│   └── rice_student/
│       ├── validate_contract.py
│       ├── smoke_test_student.py
│       ├── train_student_baseline.py
│       ├── train_student_distillation.py
│       ├── evaluate_student.py
│       ├── convert_student_litert.py
│       ├── validate_mobile_package.py
│       └── benchmark_student.py
├── models/
│   └── rice/
│       ├── teacher/
│       ├── student_baselines/
│       ├── distilled_students/
│       ├── converted/
│       └── registry/
├── reports/
│   └── rice/
│       ├── student/
│       ├── distillation/
│       ├── conversion/
│       └── mobile/
└── docs/
    ├── rice_student_model_development_plan.md
    ├── rice_student_experiment_log.md
    ├── rice_student_model_card.md
    └── rice_student_change_log.md
```

If an existing repository uses a different layout, preserve its conventions and create a mapping rather than duplicating directories. The agent must inspect the repository first.

---

## 6. Separation of responsibilities in the codebase

### `contracts.py`

Defines and validates:

- Class order.
- Crop name.
- Input dimensions.
- Color order.
- Normalization.
- Teacher metadata.
- Student metadata.
- Model version.
- Dataset and split versions.

### `data.py`

Loads the immutable split manifest and provides:

- Training dataset.
- Validation dataset.
- Locked test dataset.
- Optional field-like holdout.
- Class weights or sampling configuration.
- Deterministic preprocessing.
- Documented augmentation.

The data loader must not silently resplit the dataset.

### `models.py`

Builds the student architecture. The first candidate is MobileNetV3-Large with:

- Correct number of classes.
- Linear logits output for training.
- A simple classification head.
- Export-compatible operations.
- Explicit input shape.

The function must expose a model summary and parameter count without training.

### `losses.py`

Defines:

- Supervised cross-entropy.
- Temperature-scaled distillation KL divergence.
- Combined supervised plus distillation loss.

The implementation must correctly apply the temperature-squared factor to the distillation term.

### `distiller.py`

Defines teacher/student training behavior:

- Teacher frozen.
- Teacher inference with `training=False`.
- Student inference with `training=True` during training.
- Ground-truth labels used in the first production distillation run.
- Teacher logits and student logits aligned by class order.
- Distillation metrics logged separately.

### `metrics.py`

Reports:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Confusion matrix.
- Healthy false-positive rate.
- Calibration metrics.
- Abstention metrics.

### `calibration.py`

Selects and evaluates abstention thresholds using validation data only. It must never tune thresholds on the hidden test set.

### `export.py`

Converts the student to float32, float16, dynamic-range, and full-int8 candidates where supported. It must preserve:

- Class order.
- Input contract.
- Output contract.
- Model metadata.

### `package_validator.py`

Validates the final mobile package, including:

- File presence.
- Manifest schema.
- Label order.
- Model checksum.
- Input tensor.
- Output tensor.
- Preprocessing contract.
- Quantization metadata.
- Version compatibility.

---

## 7. Configuration requirements

Do not hard-code training decisions in Python scripts. Use YAML or JSON configuration.

### Student baseline configuration

The configuration must include:

```yaml
project: ipd
crop: rice
model_role: student_baseline
architecture: MobileNetV3Large
classes:
  - healthy
  - blast
  - brown_spot
  - blight
input_size: [224, 224]
preprocessing: match_teacher_contract
split_manifest: manifests/rice/rice_split_manifest_v1.csv
seed: 42
batch_size: 32
max_epochs: 30
early_stopping_patience: 6
optimizer: adamw
learning_rate: 0.0001
mixed_precision: false
cache_mode: none
output_dir: models/rice/student_baselines
```

The student input size may differ from the teacher if the student experiment proves that it improves the mobile trade-off. The preprocessing contract must still be explicit and identical between training and export.

### Distillation configuration

```yaml
project: ipd
crop: rice
model_role: student_distilled
architecture: MobileNetV3Large
teacher_model: models/rice/teacher/rice_teacher_v1.keras
teacher_sha256: required
classes:
  - healthy
  - blast
  - brown_spot
  - blight
input_size: [224, 224]
split_manifest: manifests/rice/rice_split_manifest_v1.csv
seed: 42
batch_size: 32
max_epochs: 30
early_stopping_patience: 6
optimizer: adamw
learning_rate: 0.0001
temperature: 3.0
alpha: 0.5
use_ground_truth_loss: true
mixed_precision: false
cache_mode: none
output_dir: models/rice/distilled_students
```

The agent may propose different values, but it must explain the reason and update the experiment log.

---

## 8. Development stages

### Stage 0: contract and codebase verification

The agent may run only small checks.

Required checks:

- Teacher checksum matches the manifest.
- Class order is correct.
- Dataset and split files exist.
- Student model builds.
- Student output shape is four logits.
- Teacher output shape is compatible.
- One batch passes through teacher and student.
- Distillation loss returns finite values.
- No training occurs.

Outputs:

```text
reports/rice/student/contract_validation.json
reports/rice/student/smoke_test_report.md
```

Stop if any contract check fails.

### Stage 1: supervised student baseline

Train a MobileNetV3-Large using ground-truth labels only. This measures the student architecture without distillation.

The user must explicitly authorize the full run. Before execution, report:

- Expected runtime.
- Hardware recommendation.
- Batch size.
- Number of epochs.
- Expected disk output.
- Checkpoint location.

The baseline must use the same train/validation/test split as the teacher where possible.

Required comparison:

```text
teacher vs supervised student
```

Do not begin distillation until the supervised student baseline completes and is evaluated.

### Stage 2: distilled student

Use the frozen teacher and the same student architecture.

Use:

```text
combined loss = alpha × supervised CE
                 + (1 - alpha) × T² × KL(teacher soft targets, student soft targets)
```

Initial values:

```text
T = 3.0
alpha = 0.5
```

The teacher must not update. The student receives both ground-truth labels and soft teacher targets.

Required comparison:

```text
teacher vs supervised student vs distilled student
```

Do not assume the distilled student is better. It must be evaluated on the same test and field-like data.

### Stage 3: distillation sensitivity experiment

If resources permit, run a small, user-authorized validation-only comparison of:

```text
T ∈ {2, 3, 4}
alpha ∈ {0.25, 0.5, 0.75}
```

Do not run a large sweep automatically. Each run must be explicitly approved or intentionally limited by the user.

### Stage 4: conversion and quantization

First convert the best student to float32 and validate output agreement. Then create optimized candidates:

1. Float32.
2. Float16.
3. Dynamic-range quantized.
4. Full-int8 with representative calibration data.

Quantization-aware training is optional and requires separate approval because it is another training job.

### Stage 5: mobile package validation

Create a package containing:

```text
model.tflite
labels.txt
model_manifest.json
preprocessing.md
checksum.sha256
release_notes.md
```

Validate the package without requiring full mobile deployment.

### Stage 6: device benchmark

A full device benchmark is user-authorized only. It must record:

- Cold-start latency.
- Warm latency.
- Median latency.
- P95 latency.
- Model size.
- Peak memory.
- Input and output tensor types.
- Accuracy agreement with the Keras student.
- Battery or thermal observations when available.

Do not claim sub-50 ms performance before measuring on named target devices.

---

## 9. Evaluation requirements for the student

Evaluate the supervised and distilled students using:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Balanced accuracy.
- Per-class precision.
- Per-class recall.
- Confusion matrix.
- Healthy false-positive rate.
- Blast recall.
- Brown Spot recall.
- Blight recall.
- Calibration.
- Abstention coverage.
- Field-like holdout performance.
- Teacher-to-student metric gap.
- Quantized-to-float metric gap.

The student must not be accepted based only on parameter count, file size, or overall accuracy.

Recommended provisional acceptance criteria:

| Metric | Provisional target |
|---|---:|
| Student macro-F1 gap to teacher | ≤ 0.05 on locked test and field-like holdout |
| Per-class recall | No unacceptable class collapse |
| Quantized macro-F1 gap to float student | ≤ 0.02 unless reviewed |
| Class-order agreement | 100% |
| Keras-to-LiteRT prediction agreement | Within documented tolerance |
| Abstention behavior | Threshold selected on validation and verified on test |
| Mobile performance | Measured on named target devices |

These are gates for review, not guaranteed outcomes.

---

## 10. Model versioning and registry

Never overwrite a teacher, student, or converted model.

Use identifiers such as:

```text
rice_teacher_v1_benchmark_candidate
rice_student_v1_supervised
rice_student_v1_distilled
rice_student_v1_float32
rice_student_v1_int8
```

Every model registry entry must contain:

```text
model_id
crop
role
architecture
teacher_id
teacher_sha256
dataset_version
split_version
class_order
input_size
preprocessing_version
training_config
metrics_report
model_path
sha256
status
created_at
known_limitations
```

A student must record exactly which teacher produced it:

```text
student_id: rice_student_v1_distilled
trained_from_teacher: rice_teacher_v1_benchmark_candidate
```

If the teacher is later updated, create a new student version rather than silently replacing the old student.

---

## 11. Documentation governance

The AI agent must update the correct documentation files whenever a change affects behavior, configuration, paths, metrics, versions, or decisions.

### Update these documents when appropriate

| Change | Required document update |
|---|---|
| New model version | Model registry and student model card |
| New dataset or split | Dataset contract and experiment log |
| Changed class order | Model contract, mobile manifest, model card |
| Changed preprocessing | Dataset contract, mobile contract, model card |
| New metric result | Experiment report and change log |
| New training configuration | Experiment log and configuration documentation |
| New conversion result | Conversion report and model registry |
| New mobile package | Mobile release notes and checksum registry |
| New failure mode | Known limitations and error-analysis report |
| Repository path change | Repository organization map and README |
| Decision to proceed or stop | Decision log and go/no-go report |

The agent must not create duplicate documentation with conflicting truths. First search for existing authoritative documents. Update the relevant document or clearly link the new report to the old one.

### Required student documentation

Maintain:

```text
docs/rice_student_experiment_log.md
docs/rice_student_model_card.md
docs/rice_student_change_log.md
reports/rice/student/student_go_no_go_decision.md
reports/rice/distillation/distillation_report.md
reports/rice/conversion/conversion_report.md
```

Each experiment-log entry must state:

- Date.
- Experiment ID.
- Code commit.
- Config path.
- Dataset and split version.
- Teacher version.
- Hardware.
- Whether execution was user-authorized.
- Runtime.
- Results.
- Decision.
- Next action.

---

## 12. Agent decision-making rules

The agent must act according to its professional role and make technical decisions when the choice is reversible and evidence is sufficient.

The agent should choose reasonable defaults for:

- Directory names.
- Report filenames.
- Small smoke-test batch sizes.
- Metadata formats.
- Low-cost validation checks.

The agent must not make unilateral decisions that materially affect:

- Dataset labels.
- Hidden test-set membership.
- Class definitions.
- Model release status.
- Deletion or archival of data.
- Long-running compute.
- Mobile deployment claims.
- Privacy or user-data handling.

When evidence is insufficient, the agent must state:

```text
Known
Unknown
Assumption
Risk
Recommended experiment
```

If the agent discovers that the teacher is not trustworthy, it must report the issue instead of distilling it silently.

---

## 13. Change-control protocol

Every material code or configuration change must produce a change-log entry.

Use this format:

```markdown
## Change <ID> — <short title>

- Date:
- Author/agent:
- Files changed:
- Reason:
- Previous behavior:
- New behavior:
- Data or model impact:
- Validation performed:
- Runtime cost:
- Rollback method:
- Related experiment:
```

Do not update only the code while leaving the experiment and model documentation stale.

Before any full training run, check:

- Configuration is committed or saved.
- Dataset and split version are recorded.
- Teacher checksum is recorded.
- Output directory is new.
- Existing artifacts will not be overwritten.
- Estimated runtime is reported.
- User authorization is available.

---

## 14. Safe commands and expensive commands

### Safe commands the agent may use

```text
list files
inspect files
search references
check git status
calculate a checksum
validate JSON/YAML
compile a small Python module
run a synthetic one-batch smoke test
load one model and inspect shapes
```

### Commands requiring user authorization

```text
python scripts/rice_student/train_student_baseline.py
python scripts/rice_student/train_student_distillation.py
python scripts/rice_student/evaluate_student.py --full-dataset
python scripts/rice_student/convert_student_litert.py --full
python scripts/rice_student/benchmark_student.py --device
```

The agent must provide the command but not run it automatically.

---

## 15. Final student go/no-go decision

The student may proceed to mobile packaging only when:

- The student baseline and distilled student are compared.
- The distilled student does not introduce unacceptable per-class failures.
- The student preprocessing matches the mobile contract.
- Float32 conversion succeeds.
- Quantized candidates are evaluated.
- Keras and LiteRT outputs agree within documented tolerance.
- Model metadata and class order are correct.
- Abstention behavior is preserved.
- Target-device measurements exist before making latency claims.

The student should remain experimental if:

- It was trained from an unverified teacher.
- Only benchmark accuracy was measured.
- Quantization was not evaluated.
- Class order is uncertain.
- The mobile preprocessing path differs from training.
- The student has high-confidence errors that the teacher does not have.
- No target-device latency measurement exists.

---

## 16. Immediate next tasks

The AI agent must complete only the following low-cost tasks without user authorization:

1. Inspect the existing repository and locate the rice teacher artifact.
2. Verify the teacher checksum and metadata.
3. Locate the rice split manifest and label map.
4. Create or update the rice student configuration files.
5. Create the student source modules.
6. Build MobileNetV3-Large and verify the output shape.
7. Run a synthetic one-batch supervised student smoke test.
8. Run a synthetic one-batch distillation smoke test.
9. Validate loss finiteness and class-order alignment.
10. Update the student experiment log and change log.
11. Prepare, but do not execute, the full training commands.
12. Report estimated CPU and GPU runtime for user choice.

The agent must then stop and ask the user to authorize the full baseline training run.

The prepared commands should be similar to:

```bash
python scripts/rice_student/train_student_baseline.py \
  --config configs/rice/student_baseline.yaml

python scripts/rice_student/train_student_distillation.py \
  --config configs/rice/student_distillation.yaml
```

The exact commands must match the actual repository paths and environment.

---

## 17. Final instruction to the AI coding agent

Build the rice student codebase safely and incrementally. Do not execute large training or testing files automatically. Begin with contracts, configuration, source modules, and bounded smoke tests. Preserve the teacher, dataset, split, and class-order contracts. Use CPU for low-cost validation and let the user choose CPU or GPU for full training after receiving runtime and resource estimates.

When a technical decision is needed, act as a professional ML engineer: choose a reversible, evidence-based default where possible; document the decision; update the relevant Markdown or registry file; and stop for user input only when the decision is expensive, destructive, externally consequential, or materially changes the project’s intended behavior.

The implementation is not complete until code, configurations, model metadata, experiment logs, model cards, conversion reports, and change logs remain consistent with one another.

---

## References

[1]: https://keras.io/examples/keras_recipes/better_knowledge_distillation/ "Knowledge distillation recipes"
[2]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"
[3]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[4]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"

Author: **Manus AI**

# Potato Student Model Plan

**Project:** IPD offline potato disease detection

**Purpose:** Build a potato-specific mobile student model using the lessons from rice, without copying rice assumptions into a different crop.

**Default student candidate:** MobileNetV3-Large

**Fallback/comparison candidate:** MobileNetV2 or MobileNetV3-Small, only if the measured mobile budget requires it.

**Execution rule:** The AI coding agent may inspect files, write code, validate contracts, and run small smoke tests. It must prepare but not automatically execute full potato training, full evaluation, distillation, quantization-aware training, large conversion jobs, or device benchmarks.

---

## 1. Entry condition and realistic scope

Begin the potato codebase after the rice pipeline has passed its **technical** checks:

- Keras-to-LiteRT class agreement.
- Correct preprocessing and class order.
- Valid mobile package checksum.
- Basic phone inference smoke test.
- Uncertainty and unsupported-input behavior.

Rice does not need to be fully field validated before potato code scaffolding begins. The purpose of the rice gate is to avoid copying a broken mobile pipeline.

Potato still requires a separate dataset audit, teacher, split, class mapping, evaluation, and mobile decision.

Do not assume potato will have the same classes or difficulty as rice.

---

## 2. Define the potato task before coding

Confirm the exact classes from the actual potato dataset and project scope. Do not assume the final class list.

Possible classes may include:

```text
healthy
late_blight
early_blight
```

These are examples only. The final class list must be verified.

Create a potato contract containing:

```text
crop: potato
classes
class_order
label_definitions
source_datasets
license_information
image_count_by_class
capture_conditions
known_group_ids
split_version
input_size
preprocessing_version
```

Required files:

```text
manifests/potato/potato_dataset_contract.json
manifests/potato/potato_label_map.json
reports/potato/dataset_audit_report.md
```

The class order must be loaded by training, evaluation, export, and mobile code. Never infer it from filesystem sorting.

---

## 3. Potato dataset audit before training

Inspect the actual data before creating a training command.

Check:

- Corrupt files.
- Exact duplicates.
- Near duplicates.
- Same leaf in multiple partitions.
- Same plant in multiple partitions.
- Same capture session or video sequence in multiple partitions.
- Augmented-family overlap.
- Wrong labels.
- Ambiguous labels.
- Images with no usable potato leaf.
- Source distribution by class.
- Background and camera distribution by class.
- Image-quality distribution by class.
- License and provenance.

### Source-class audit

Create:

```text
source × class
```

The preferred structure is:

```text
source A: healthy + every supported disease
source B: healthy + every supported disease
source C: healthy + every supported disease
```

If potato has the same structure found in rice, where healthy is entirely from one source and disease is entirely from another, mark it as source-confounded. It may still support a benchmark prototype, but a random test split must not be called field validation.

Required outputs:

```text
reports/potato/source_audit/source_class_cross_tabulation.csv
reports/potato/source_audit/source_visual_comparison.md
reports/potato/source_audit/source_metadata_comparison.csv
```

Do not try to fix source confounding with more epochs or a different optimizer. The correction requires better source/class coverage or an explicitly limited claim.

---

## 4. Potato leakage-safe split

After audit and deduplication, create an immutable split manifest.

Keep the following together where metadata permits:

```text
same original/augmentation family
same leaf
same plant
same capture session
same video sequence
```

An initial target may be:

```text
train: 70%
validation: 15%
test: 15%
```

The group rule is more important than exact percentages.

Save:

```text
manifests/potato/potato_split_manifest_v1.csv
reports/potato/split_integrity_report.md
```

The training script must fail if it cannot find the expected immutable split manifest. It must not silently create a random split.

The locked test set must not be used to select:

- Epochs.
- Learning rate.
- Augmentation.
- Class weights.
- Architecture.
- Confidence thresholds.
- Distillation parameters.

---

## 5. Potato teacher requirement

If a potato teacher already exists, verify:

- Path.
- SHA-256 checksum.
- Crop.
- Input shape.
- Output shape.
- Class order.
- Normalization.
- Evaluation status.
- Dataset and split version.

If no teacher exists, create a documented potato teacher baseline first. An ImageNet-pretrained EfficientNetB3 may be a candidate, but the final architecture must be selected based on potato data size, class difficulty, and available compute.

The potato student cannot use the rice teacher. The potato teacher must classify potato classes.

Before distillation:

```text
teacher.trainable = false
teacher called with training = false
teacher checkpoint versioned
teacher class order matches potato contract
teacher metrics recorded
```

Do not distill from an unverified teacher.

---

## 6. Potato student architecture decision

Start with one candidate to avoid unnecessary compute:

```text
MobileNetV3-Large
```

Consider MobileNetV2 or MobileNetV3-Small only if:

- The first student is too slow.
- The model is too large.
- It has unacceptable memory use.
- It has a class-specific accuracy problem.
- LiteRT compatibility is poor.

Choose using:

- Macro-F1.
- Per-class recall.
- Healthy false-positive rate.
- Disease false-negative rate.
- Converted model size.
- Device latency.
- Peak memory.
- Quantization degradation.
- Calibration and abstention.

Do not choose solely from validation accuracy.

The potato student output must contain exactly the verified number of classes and must produce linear logits during training.

---

## 7. Potato codebase structure

Inspect the existing repository first. Reuse shared utilities rather than creating duplicates.

Recommended structure:

```text
configs/potato/
├── student_baseline.yaml
├── student_distillation.yaml
├── student_quantization.yaml
└── mobile_contract.yaml

manifests/potato/
├── potato_dataset_contract.json
├── potato_label_map.json
└── potato_split_manifest_v1.csv

src/potato_student/
├── __init__.py
├── contracts.py
├── data.py
├── models.py
├── losses.py
├── distiller.py
├── metrics.py
├── calibration.py
├── export.py
└── package_validator.py

scripts/potato_student/
├── validate_contract.py
├── audit_dataset.py
├── create_split.py
├── smoke_test_student.py
├── train_student_baseline.py
├── evaluate_student.py
├── train_student_distillation.py
├── convert_student_litert.py
└── validate_mobile_package.py

models/potato/
├── teacher/
├── student_baselines/
├── distilled_students/
├── converted/
└── registry/

reports/potato/
├── dataset_audit/
├── student/
├── distillation/
├── conversion/
└── mobile/

docs/
├── potato_student_experiment_log.md
├── potato_student_model_card.md
└── potato_student_change_log.md
```

---

## 8. Required code responsibilities

### Contract module

Validate crop, classes, class order, input shape, normalization, teacher metadata, dataset version, and split version.

### Data module

Load the immutable split manifest. Provide deterministic validation and test preprocessing. Do not silently resplit data.

### Model module

Build the selected potato student with the correct number of classes, explicit input shape, linear logits, and parameter-count reporting.

### Loss module

Implement supervised cross-entropy, temperature-scaled KL divergence, and combined distillation loss.

### Distiller module

Keep the teacher frozen and run it with `training=False`. Use ground-truth labels and soft teacher targets. Log hard-label and distillation losses separately.

### Metrics module

Report accuracy, macro-F1, balanced accuracy, per-class precision and recall, confusion matrix, calibration, and abstention coverage.

### Export module

Create and validate Float32, Float16, and only-when-justified INT8 candidates.

### Package validator

Check file presence, checksum, class order, input/output tensors, preprocessing, quantization metadata, and app compatibility.

---

## 9. Stage 0: low-cost checks only

The agent may run without user authorization:

- Manifest schema validation.
- Label-map validation.
- Hash and split checks.
- Model construction.
- Parameter count.
- Input/output shape checks.
- One synthetic batch.
- One-batch distillation-loss test.
- Preprocessing tensor checks.
- Documentation updates.

Required outputs:

```text
reports/potato/student/contract_validation.json
reports/potato/student/smoke_test_report.md
```

No full training should occur at this stage.

---

## 10. Stage 1: supervised potato student baseline

Train the potato student from ground-truth labels only. This is the baseline for deciding whether distillation is needed.

Initial configuration should be externalized:

```yaml
crop: potato
model_role: student_baseline
architecture: MobileNetV3Large
input_size: [224, 224]
split_manifest: manifests/potato/potato_split_manifest_v1.csv
seed: 42
max_epochs: 30
early_stopping_patience: 6
optimizer: adamw
learning_rate: 0.0001
mixed_precision: false
output_dir: models/potato/student_baselines
```

The class list and preprocessing must come from the potato contract.

Before execution, the agent must report:

```text
command
expected runtime
CPU/GPU recommendation
RAM/VRAM estimate
output paths
whether existing files will be overwritten
```

The user must authorize the full run.

Evaluate:

- Locked test metrics.
- Macro-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Confusion matrix.
- Healthy false-positive rate.
- Disease false-negative rate.
- Calibration.
- Abstention.
- Source-aware or field-like performance where available.

Do not begin distillation before the supervised baseline is evaluated.

---

## 11. Stage 2: optional potato distillation

Distillation is optional. Use it only if:

- The supervised student is materially weaker than the teacher.
- The model needs compression or robustness that distillation may improve.
- There is a clear mobile reason to test it.

Initial settings:

```text
teacher: frozen potato teacher
student: selected potato MobileNet
T = 3.0
alpha = 0.5
```

Use:

```text
L = alpha × supervised CE
  + (1 - alpha) × T² × KL(teacher soft targets, student soft targets)
```

Keep the ground-truth term. Do not train only from teacher pseudo-labels.

Compare:

```text
potato teacher
potato supervised student
potato distilled student
```

The distilled student is accepted only if it provides a justified advantage in accuracy, per-class recall, calibration, size, latency, memory, or quantization behavior.

Do not run a large temperature/alpha sweep automatically.

---

## 12. Stage 3: conversion and quantization

Convert the selected student in this order:

```text
Float32
Float16
Dynamic-range candidate
Full INT8 only after representative-data validation
```

The representative calibration set must:

- Come from training or validation.
- Cover every potato class.
- Include realistic image variation.
- Use exact deployment preprocessing.
- Exclude the locked test set.
- Be versioned and reproducible.

For every artifact record:

```text
model_id
format
file_size_bytes
input_dtype
output_dtype
input_shape
output_shape
operator_set
conversion_warnings
accuracy
macro_f1
per_class_recall
model_sha256
```

Reject a quantized model if it causes unacceptable class-specific recall loss, regardless of file size.

---

## 13. Stage 4: potato mobile validation

Run the selected potato model through:

```text
Keras prediction
LiteRT Float32 prediction
LiteRT Float16 prediction
optional LiteRT INT8 prediction
```

Verify:

- Predicted class agreement.
- Preprocessing equivalence.
- Input/output tensor compatibility.
- Model checksum.
- Label-file order.
- Abstention behavior.
- Non-potato and poor-quality input handling.

Measure on the intended phone:

- Cold-start latency.
- Warm median latency.
- P95 latency.
- Peak memory.
- Package size.
- Delegate used.

Do not claim a performance target before measuring the named device.

---

## 14. Potato acceptance gates

### Benchmark candidate

- Model identity verified.
- Locked-test metrics reproducible.
- No known exact or group leakage.
- Per-class metrics reported.

### Mobile candidate

- LiteRT/TFLite conversion succeeds.
- Keras and LiteRT predictions agree.
- Class order and preprocessing are correct.
- Quantized and float metrics are measured.
- Size and device latency are recorded.

### Field-validation candidate

- Independent source-aware or field-like evaluation exists.
- Adequate class coverage exists.
- Labels are independently reviewed.
- Groups are disjoint.
- Per-class recall is acceptable.

### Production candidate

- All previous gates pass.
- Calibration and abstention are defined.
- Known limitations are documented.
- Model package is reproducible and checksum-verified.
- Rollback and version metadata exist.
- No unresolved critical source-label confounding remains.

---

## 15. Agent execution restrictions

The agent may execute small bounded checks:

```text
metadata inspection
hash checks
manifest validation
label-map validation
model shape checks
one-batch smoke tests
small preprocessing checks
```

The agent must prepare commands and wait for user authorization before executing:

```text
full potato teacher training
full potato student training
full-dataset evaluation
potato distillation
quantization-aware training
large conversion jobs
full device benchmarks
hyperparameter sweeps
```

The agent must not overwrite existing model artifacts. Every training run must use a new versioned output directory.

---

## 16. Documentation governance

Update these files after each material change:

```text
docs/potato_student_experiment_log.md
docs/potato_student_model_card.md
docs/potato_student_change_log.md
reports/potato/student/student_go_no_go_decision.md
reports/potato/distillation/distillation_report.md
reports/potato/conversion/conversion_report.md
reports/potato/mobile/potato_device_benchmark.md
```

Every experiment entry must record:

- Date.
- Experiment ID.
- Code commit.
- Configuration.
- Dataset and split version.
- Teacher version.
- Hardware.
- Runtime.
- Metrics.
- Decision.
- Limitations.
- Next action.

Do not overwrite historical reports. Create versioned reports or dated correction sections.

---

## 17. Immediate work order

1. Inventory potato data and existing repository code.
2. Confirm the exact potato class list.
3. Create the potato data contract and label map.
4. Audit exact, near, and group duplicates.
5. Audit source-to-class distribution.
6. Create and lock the potato split manifest.
7. Verify or create the potato teacher contract.
8. Create student configurations and modules.
9. Run low-cost contract and synthetic smoke tests.
10. Prepare the supervised training command.
11. Report runtime and hardware estimates.
12. Wait for user authorization before full training.
13. Evaluate the supervised student.
14. Decide whether distillation is justified.
15. Convert and benchmark only the selected candidate.

The first six actions are preparation and audit work. They may begin now. The full potato training command must wait until the potato contract and split-integrity report are complete.

---

## 18. Lessons carried forward from rice

The rice project established several controls that are mandatory for potato.

### 18.1 Source confounding is a release blocker for strong claims

If a source cross-tabulation shows that one source contains only Healthy images while another source contains only disease images, record:

```text
source_label_confounding: true
field_generalization_status: unproven
release_status: benchmark_or_prototype_only
```

Do not claim that a clean random test split solves this problem. Exact duplicate checks and source-bias checks answer different questions.

The source audit must be completed before potato training, not after a high validation score is obtained.

### 18.2 Host latency is not phone latency

Record these separately:

```text
host_inference_latency
target_device_inference_latency
end_to_end_pipeline_latency
```

If the benchmark runs on Windows or another desktop host, label it as a host benchmark. Do not call it a smartphone result until the model has run on the named target device.

For potato, device latency remains unknown until measured on the intended phone. This does not block codebase preparation or dataset auditing.

### 18.3 Conversion agreement must precede mobile claims

Before any potato mobile release, compare the same images through:

```text
Keras FP32
LiteRT/TFLite FP32
LiteRT/TFLite Float16
optional LiteRT/TFLite INT8
```

Any unexpected class disagreement requires investigation of RGB/BGR order, normalization, resize, letterbox, output mapping, and quantization parameters.

### 18.4 INT8 is not automatically better because it is smaller

For potato, reject INT8 if it creates unacceptable per-class recall loss, especially for a disease class, or if the intended device uses a slow fallback path. Measure accuracy and latency together.

Float16 is the default deployment candidate unless potato evidence supports another format.

### 18.5 A small external set is a smoke test, not field validation

An external set with only a few images per class may validate technical behavior, but it cannot by itself establish field accuracy. Report its size, class counts, plant/session independence, and label-review process.

### 18.6 Documentation status must match evidence

Use these distinct statuses:

```text
benchmark_candidate
mobile_candidate
field_validation_candidate
production_candidate
```

Do not use `field_validated` or `production_ready` merely because validation accuracy is high.

---

## 19. Start-now versus wait-later decision

### The agent may start now

The following work is low-risk and should begin without full-training authorization:

1. Inventory the potato dataset and repository.
2. Confirm the exact potato classes.
3. Generate the source-by-class table.
4. Check file integrity and exact hashes.
5. Identify near-duplicate and group candidates.
6. Create the potato data contract and label map.
7. Prepare the split-generation script.
8. Create student model/configuration modules.
9. Run synthetic one-batch smoke tests.
10. Prepare, but do not run, training commands.

### The agent must wait before

The user must authorize:

```text
full potato teacher training
full potato student training
full-dataset evaluation
knowledge distillation
quantization-aware training
large conversion runs
device benchmark runs
```

Before asking for authorization, the agent must report:

```text
exact command
purpose
estimated runtime
CPU/GPU recommendation
RAM/VRAM estimate
output paths
whether any file will be overwritten
```

### Mandatory stop conditions

Stop before full training if:

- The potato class list is still uncertain.
- The source-by-class structure is severely confounded and no limitation has been recorded.
- The split manifest is not immutable.
- Exact or group leakage is unresolved.
- Labels are materially ambiguous.
- The teacher contract is missing when distillation is requested.
- The training configuration uses the rice class order accidentally.

---

## 20. Potato first-run decision gate

After the audit, make one of these decisions:

| Finding | Decision |
|---|---|
| Classes, labels, split, and source coverage are acceptable | Prepare supervised student training |
| Source confounding exists but data is still usable | Continue as benchmark/prototype and document limitation |
| Exact/group leakage exists | Rebuild the split before training |
| Labels are materially unreliable | Review/quarantine data before training |
| No valid potato teacher exists | Train/evaluate a teacher or use a documented pretrained baseline before distillation |
| Potato data is too small for a reliable split | Report provisional results only; do not claim field readiness |

The first full model run should be a **supervised potato student baseline**, not distillation. This gives a direct measurement of what the mobile architecture can learn from potato labels.

---

## Final engineering conclusion

The rice work shows why high validation accuracy is not enough. Potato must begin with data and split governance, not with a training command.

The practical path is:

```text
potato audit
→ potato contract
→ leakage-safe split
→ potato teacher verification
→ supervised potato student
→ evaluation
→ optional distillation
→ Float16/INT8 comparison
→ device validation
```

> **Use distillation only when it solves a measured problem or provides a measured deployment advantage. Do not copy rice labels, thresholds, teacher weights, or source assumptions into potato.**

Author: **Manus AI**

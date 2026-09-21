# Tomato Mobile Student Development Plan

**Project:** IPD plant disease detection

**Crop:** Tomato

**Teacher:** `tomato_teacher_v2`

**Student architecture:** MobileNetV3-Large baseline

**Target classes:** `early_blight`, `healthy`, `late_blight`

**Current decision:** Begin supervised student development; defer knowledge distillation until the supervised baseline has been measured.

**Author:** Manus AI

---

## 1. Executive decision

Tomato teacher v2 is sufficiently improved to begin a controlled mobile-student experiment. It is not yet fully field-validated because the independent external holdout contains only 12 images and field calibration remains preliminary.

The first student must therefore be a **supervised MobileNetV3 baseline** trained from verified labels. Do not begin with knowledge distillation.

The required sequence is:

```text
freeze tomato_teacher_v2
→ lock dataset, preprocessing, and class order
→ train supervised MobileNetV3 baseline
→ evaluate teacher and student on identical datasets
→ convert the student to LiteRT Float16
→ test the actual target phone
→ expand external field validation
→ decide whether distillation provides a measurable advantage
```

A student should not be approved for production merely because it is small or because it matches the benchmark accuracy.

---

## 2. Current teacher status

Register the teacher as:

```text
model_id: tomato_teacher_v2_preliminary_field_validated
role: cloud teacher for controlled student development
architecture: EfficientNetB3
input: 300 x 300 x 3
classes: early_blight, healthy, late_blight
field evidence: preliminary
production status: no-go
student baseline development: go
immediate distillation: deferred
```

The 12-image field holdout must remain locked as a regression set. It may be used for final comparison after the student configuration is frozen, but it must not be used for checkpoint selection or hyperparameter tuning.

Preserve the teacher checkpoint, checksum, model manifest, preprocessing contract, dataset version, split version, and all evaluation reports.

---

## 3. Student objective

The student must provide a practical mobile trade-off among:

```text
accuracy
per-class recall
field robustness
confidence behavior
model size
latency
memory use
battery and thermal stability
```

The student objective is not to reproduce every teacher probability. The first question is whether a small supervised model can meet the mobile requirements while retaining disease recall on difficult tomato images.

The primary comparison is:

```text
tomato_teacher_v2
versus
tomato_student_supervised_v1
```

Only after this comparison should a distilled student be considered.

---

## 4. Freeze the contracts before training

### 4.1 Class order

Use one class order everywhere:

```text
0: early_blight
1: healthy
2: late_blight
```

Store the class order in the dataset manifest, training configuration, Keras model metadata, LiteRT manifest, app handoff, and evaluation scripts.

### 4.2 Input and preprocessing

The student must use a documented, deterministic preprocessing contract:

```text
color order: RGB
resize: aspect-preserving
padding: neutral gray RGB(114, 114, 114)
distortion: none
normalization: explicitly documented
```

The teacher uses a 300 × 300 input. The initial student experiment should use the same input size if the mobile budget allows it. If MobileNetV3-Large is too slow or large at 300 × 300, evaluate a separate 224 × 224 student experiment rather than silently changing the contract.

For each input-size experiment, report the actual trade-off. Do not compare a 300 × 300 teacher with a 224 × 224 student as though they have identical visual information.

### 4.3 Preprocessing equivalence

Verify that the training loader, Keras evaluator, LiteRT converter, and mobile app all apply the same preprocessing. A conversion that changes normalization can invalidate the comparison.

Required contract:

```text
input tensor shape
input dtype
resize method
padding geometry
padding color
RGB/BGR convention
normalization formula
output class order
```

---

## 5. Dataset and split requirements

Use the corrected tomato teacher v2 data pipeline as the starting point.

The supervised student must use:

- The corrected group-safe training split.
- The corrected validation split.
- The locked benchmark test split.
- The separate external field holdout.
- The ambiguous review set.
- The difficult-image and hard-negative manifests where available.

Do not create a new random split for the student. A new random split would make teacher-versus-student comparisons less reliable.

Images from the same plant, capture session, burst, or pHash family must remain in one partition.

The student training set may contain only data approved for training. The locked benchmark test and external field holdout must not be used for training, early stopping, threshold selection, or model selection.

The student must inherit the following data limitations:

```text
external field holdout is small
field calibration is preliminary
color shortcut risk is reduced but not fully proven eliminated
background robustness is proven only for the tested suite
```

---

## 6. Student model design

### 6.1 Initial architecture

Use MobileNetV3-Large as the first student because it is already used successfully in the project and has an established mobile conversion path.

Start with ImageNet-pretrained weights if compatible with the project’s preprocessing contract.

Use a simple classification head:

```text
backbone
→ global average pooling
→ dropout if validated
→ dense layer with 3 outputs
→ softmax
```

Do not add a complex head before establishing a baseline.

### 6.2 Model variants

Use only one primary baseline initially:

```text
tomato_student_supervised_v1_mobilenetv3_large
```

Optional variants may be tested later:

```text
MobileNetV3-Large at 224 × 224
MobileNetV3-Large at 300 × 300
MobileNetV3-Small if the device budget requires it
```

Do not run a broad architecture sweep before the baseline has been evaluated.

---

## 7. Supervised training procedure

### Phase A: classifier-head warm-up

- Load the ImageNet-pretrained MobileNetV3 backbone.
- Freeze the backbone.
- Train the classification head on the corrected training split.
- Monitor validation loss and macro-F1.
- Save the best checkpoint based only on validation performance.

### Phase B: controlled fine-tuning

- Unfreeze the upper portion of the backbone.
- Keep Batch Normalization layers frozen initially.
- Use a learning rate substantially smaller than the head warm-up rate.
- Continue monitoring validation loss, macro-F1, and per-class recall.
- Stop when improvement is no longer meaningful.

Do not select the checkpoint using the locked benchmark test or external field holdout.

### Suggested starting configuration

These values are starting points, not guarantees:

```text
architecture: MobileNetV3-Large
head learning rate: 1e-3
fine-tuning learning rate: 1e-5 to 1e-4
optimizer: AdamW or the project’s validated optimizer
weight decay: 1e-4
label smoothing: 0.0 to 0.05, selected using validation only
head warm-up: 3–8 epochs
fine-tuning: 10–30 epochs
early stopping patience: 5–8 epochs
random seeds: at least 2, preferably 3 if compute allows
```

The actual values and runtime must be recorded in the training configuration.

### Class weighting

Compute class weights from the training partition only. Use class weights only if class imbalance is material or validation results show a meaningful benefit.

Do not use the test set to decide whether class weighting helped.

---

## 8. Training augmentations

Use the same realistic augmentation policy approved for teacher v2 unless a controlled student experiment demonstrates a better result.

Permitted categories include:

- Mild rotation.
- Mild translation.
- Mild scale change.
- Horizontal flip when biologically acceptable.
- Moderate hue variation.
- Moderate saturation variation.
- Brightness and contrast variation.
- White-balance variation.
- Mild shadow and glare simulation.
- Mild compression variation.

Do not use extreme color transformations that make tomato leaves biologically unrealistic.

Validation, benchmark test, and external holdout images must not receive random augmentation during evaluation.

---

## 9. Required training outputs

Store the student in a new versioned directory:

```text
models/tomato/students/supervised_v1/
├── student_best.keras
├── training_config.json
├── training_log.csv
├── model_manifest.json
├── checksum.sha256
└── evaluation_summary.json
```

The manifest must include:

```text
model_id
role
architecture
teacher_id, if applicable
dataset_version
split_version
input_shape
input_dtype
class_order
preprocessing_version
training_seed
model_sha256
known_limitations
release_status
```

Do not overwrite the teacher or another student checkpoint.

---

## 10. Evaluation protocol

Evaluate the teacher and supervised student on exactly the same manifests.

### 10.1 Validation evaluation

Use validation data for training decisions. Report:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Confusion matrix.
- Validation loss.
- Calibration.
- Abstention behavior if thresholds are already defined.

### 10.2 Locked benchmark evaluation

Run after the student configuration is frozen.

Report:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class precision.
- Per-class recall.
- Healthy false-positive rate.
- Early Blight recall.
- Late Blight recall.
- Confusion matrix.
- Expected Calibration Error.
- Brier score.

### 10.3 External field evaluation

Evaluate on the locked 12-image teacher holdout, but label the result preliminary.

Also evaluate on any newly expanded field holdout only after the student configuration is frozen.

Report separately:

```text
correct disease classifications
uncertain outputs
rejected outputs
accepted coverage
selective accuracy
high-confidence errors
```

An `uncertain` result must not be counted as a correct disease classification.

### 10.4 Difficult-image evaluation

Use a separate manifest containing:

- Pale and lime-green Healthy leaves.
- White-background images.
- Soil and natural backgrounds.
- Small lesions.
- Early symptoms.
- Partial leaves.
- Different lighting.
- Different aspect ratios.
- Ambiguous examples.

The most important metric is not only total accuracy. Report disease-to-Healthy errors and the confidence of those errors.

---

## 11. Student acceptance gates before conversion

The supervised student may proceed to conversion only if:

- The training and evaluation manifests are valid.
- No group leakage is present.
- The class order is correct.
- The preprocessing contract is reproducible.
- The student has acceptable per-class recall on the locked benchmark.
- The student does not show a serious new disease-to-Healthy failure pattern.
- Difficult-image behavior is documented.
- The student’s field result is reported as preliminary when the holdout is small.
- The teacher remains available as the reference model.

A student does not need to equal the teacher on every metric, but any loss in Early Blight or Late Blight recall must be explicitly judged against the product requirement.

---

## 12. LiteRT conversion plan

Convert the supervised student using a controlled conversion pipeline:

```text
student Keras FP32
→ LiteRT Float32
→ LiteRT Float16
→ optional INT8 research artifact
```

Float16 is the primary initial mobile candidate.

For every converted artifact, record:

```text
file path
file size in bytes
input shape
input dtype
output shape
output dtype
operator set
conversion warnings
SHA-256 checksum
```

Evaluate Keras and LiteRT outputs on the same locked manifest.

Required format-parity checks:

- Categorical agreement.
- Per-class recall.
- Probability difference.
- Input/output tensor contract.
- Preprocessing equivalence.

Do not approve INT8 simply because the file is smaller. Measure accuracy, per-class recall, latency, and runtime delegate behavior.

---

## 13. Physical-device validation

Run the Float16 student on the actual target device, not only on the development computer.

Record:

```text
device model
operating-system version
runtime version
delegate
thread count
model load time
cold-start latency
warm median latency
P95 latency
preprocessing time
inference time
end-to-end latency
peak memory
thermal behavior
battery impact
```

Run repeated inference to expose warm-up and thermal behavior.

The model is not device-validated until the actual target phone test passes.

---

## 14. App-model contract requirements

The student handoff must define:

- Model file and checksum.
- Class order.
- Input shape and dtype.
- RGB channel order.
- Letterbox geometry.
- Padding color.
- Normalization formula.
- Confidence threshold.
- Margin threshold.
- Quality-gate behavior.
- `accepted`, `uncertain`, and `unsupported_input` states.
- Tomato mode selection.
- Model version display.
- Offline behavior.

The application must not present a tomato disease result as a universal plant diagnosis. Explicit crop mode is required.

If a crop-verification model is not available, the app must require the user to select Tomato mode and provide clear capture guidance.

---

## 15. Optional distillation stage

Distillation is a later comparison experiment, not the first student implementation.

Run it only after the supervised baseline is evaluated and the teacher remains frozen.

A distilled student must be compared against:

```text
tomato_teacher_v2
tomato_student_supervised_v1
tomato_student_distilled_v1
```

Use identical:

- Training-independent evaluation manifests.
- Preprocessing.
- Class order.
- Calibration procedure.
- Conversion settings.
- Device benchmark.

Record distillation parameters:

```text
temperature T
hard-label loss weight
soft-label loss weight
teacher checkpoint checksum
student initialization
training seed
dataset version
```

Do not claim that distillation improves robustness unless the distilled student demonstrates a measured advantage on difficult or external data.

Accept distillation only if it improves at least one important requirement without unacceptable loss in another:

- Difficult-lesion recall.
- External field per-class recall.
- Calibration.
- Model size.
- Latency.
- Memory.
- Quantization behavior.

If the supervised student already meets the mobile requirements, distillation may remain optional research.

---

## 16. Student comparison table

The final comparison must use measured values.

| Model | Format | Size | Accuracy | Macro-F1 | Early Blight recall | Healthy recall | Late Blight recall | Field coverage | Latency | RAM | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Teacher v2 | Keras FP32 | measure | measure | measure | measure | measure | measure | measure | measure | measure | Reference |
| Supervised student | LiteRT Float16 | measure | measure | measure | measure | measure | measure | measure | measure | measure | Candidate |
| Distilled student | LiteRT Float16 | measure | measure | measure | measure | measure | measure | measure | measure | measure | Optional |
| Supervised student | LiteRT INT8 | measure | measure | measure | measure | measure | measure | measure | measure | measure | Research only until passed |
| Distilled student | LiteRT INT8 | measure | measure | measure | measure | measure | measure | measure | measure | measure | Research only until passed |

Do not fill missing values with estimates.

---

## 17. Go/no-go rules

### Continue with supervised student development

Continue if:

- Training is stable.
- The student has a valid checkpoint.
- Evaluation is reproducible.
- No leakage is found.
- The student is not materially worse on difficult disease examples.

### Retrain the student

Retrain only if:

- A data or preprocessing error is found.
- Validation metrics are unstable.
- The student systematically misses disease cases.
- A specific difficult-image failure is confirmed.

Do not simply increase epochs without identifying the failure mechanism.

### Proceed to distillation

Proceed only if the supervised baseline shows a measurable need and the teacher is still the correct reference.

### Reject the student for mobile release

Reject or keep research-only if:

- LiteRT predictions disagree materially with Keras.
- Early Blight or Late Blight recall falls below the product requirement.
- The student produces unsafe high-confidence Healthy predictions on confirmed disease cases.
- Latency or memory exceeds the target device budget.
- The app contract cannot reproduce the evaluator’s preprocessing.
- The model has not been tested on the actual target device.

---

## 18. Execution restrictions for the coding agent

The agent may run automatically:

- Manifest validation.
- Duplicate and group checks.
- Model-shape inspection.
- Small inference tests.
- Bounded evaluation scripts.
- Checksum verification.
- Documentation validation.

The agent must ask for authorization before running:

```text
full supervised student training
multi-seed training
knowledge distillation
large hyperparameter sweeps
quantization-aware training
large conversion jobs
large external evaluations
```

Before requesting authorization, the agent must state:

```text
exact command
purpose
training and evaluation manifests
estimated runtime
GPU/CPU requirement
RAM/VRAM estimate
output directory
checkpoint behavior
whether existing files will be overwritten
```

The agent must not silently overwrite teacher or student artifacts.

---

## 19. Required reports

Create these reports during the student work:

```text
reports/tomato/student_v1/
├── training_summary.md
├── benchmark_evaluation.md
├── external_field_evaluation.md
├── difficult_image_evaluation.md
├── calibration_report.md
├── format_parity_report.md
├── conversion_report.md
├── device_benchmark.md
├── app_contract_validation.md
└── student_go_no_go_decision.md
```

The final decision report must state whether the student is:

```text
research-only
controlled prototype
physical-device validated
field-validation candidate
production candidate
```

Do not use the phrase `production ready` unless all required product gates have been completed.

---

## 20. Exact implementation order

Run the project in this order:

```text
1. Freeze teacher v2 and its contracts.
2. Verify the teacher checksum and class order.
3. Reuse the corrected v2 split manifests.
4. Build the supervised MobileNetV3 training pipeline.
5. Run bounded integrity and shape checks.
6. Obtain authorization for full supervised training.
7. Train the supervised student.
8. Evaluate validation metrics and save the best checkpoint.
9. Freeze the student configuration.
10. Evaluate on the locked benchmark.
11. Evaluate on the external field and difficult-image manifests.
12. Convert the supervised student to LiteRT Float32 and Float16.
13. Verify format parity.
14. Benchmark the Float16 artifact on the target phone.
15. Compare the student with teacher v2.
16. Expand field validation in parallel.
17. Decide whether distillation solves a measured problem.
18. Run one controlled distillation comparison only if justified.
19. Produce the final student go/no-go report.
```

---

## Final recommendation

Start with the supervised tomato MobileNetV3 student now. Do not start with distillation.

The teacher v2 is a reasonable reference for controlled student development, but its field evidence remains preliminary because the external holdout is small and field calibration is not fully established.

The correct engineering decision is:

```text
teacher_v2: frozen reference for experiments
supervised student: build now
Float16 conversion: after supervised evaluation
physical phone test: required
expanded field validation: required
knowledge distillation: optional, evidence-dependent
production deployment: not approved yet
```

> **The supervised baseline tells us whether a small model is already sufficient. Distillation should only be added if comparison evidence shows that it solves a real problem.**

## References

[1]: file:///home/ubuntu/ipd_model_docs/tomato_teacher_v2_retraining_plan.md "Tomato teacher v2 retraining and validation plan"

[2]: file:///home/ubuntu/upload/TOMATO_TEACHER_V2_UNIFIED_MASTER_REPORT_FOR_MANUS.md "Tomato teacher v2 unified master report"

[3]: file:///home/ubuntu/ipd_model_docs/potato_app_team_integration_handoff.md "Potato mobile app integration handoff used as a project contract reference"

[4]: file:///home/ubuntu/ipd_model_docs/rice_student_model_development_plan.md "Rice student model development plan used as a project workflow reference"

# Tomato Teacher v2 Retraining and Validation Plan

**Project:** IPD tomato disease detection

**Model role:** Cloud teacher model for later mobile-student development

**Teacher architecture:** EfficientNetB3

**Target classes:** `healthy`, `early_blight`, `late_blight`

**Current teacher:** `tomato_teacher_v1_benchmark_candidate`

**Next model:** `tomato_teacher_v2`

**Document status:** Controlled retraining specification

**Author:** Manus AI

---

## 1. Executive decision

The tomato teacher should be retrained, but not by simply running more epochs on the current dataset.

The existing teacher has strong benchmark-distribution performance, but the available evidence indicates unresolved problems:

- The original field evaluation was too small and contained unreliable labels.
- The model may be using leaf color as a disease shortcut.
- Background sensitivity has been observed.
- White backgrounds and unusual aspect ratios cause failures.
- Field calibration and generalization are not established.
- The current teacher is not yet safe to supervise a final mobile student.

The correct objective is therefore:

```text
correct the data and evaluation design
→ retrain tomato_teacher_v2
→ test field-first robustness
→ freeze the teacher only if it passes independent gates
→ then create the tomato student
```

Do not distill `tomato_teacher_v1`. A student trained from an unreliable teacher can inherit its color and background shortcuts.

---

## 2. Current teacher status

Register the existing model as:

```text
model_id: tomato_teacher_v1_benchmark_candidate
role: benchmark teacher candidate
field_validation: incomplete
source_audit: incomplete or unresolved
color_shortcut_risk: present
background_robustness: unresolved
student_distillation_eligibility: no-go
production_status: no-go
```

Preserve the following before modifying anything:

```text
teacher checkpoint
training configuration
optimizer configuration
split manifest
label map
model checksum
preprocessing configuration
training log
benchmark report
field images
original labels
reviewed labels
source metadata
```

Never overwrite the v1 checkpoint or its evaluation reports.

---

## 3. Retraining objective

The goal of v2 is not merely to increase benchmark accuracy. The goal is to reduce shortcut learning and establish credible evidence that the teacher uses disease-relevant visual information across sources and conditions.

The v2 experiment must answer:

> After correcting source, label, color, background, and preprocessing problems, does EfficientNetB3 generalize better to independently reviewed field-like images?

The experiment must not claim that a higher benchmark score alone proves better disease reasoning.

---

## 4. Non-negotiable data rules

### 4.1 Do not use unreliable labels as ground truth

The original field images must be reviewed before they are used in training or final evaluation.

For each field image, store:

```text
image_id
crop
original_label
reviewed_label
label_status
reviewer_1
reviewer_2
expert_adjudication
label_confidence
symptoms_observed
source
plant_id
capture_session
location_if_known
camera_if_known
```

Use these label states:

```text
verified
probable
ambiguous
unverified
rejected
```

Only `verified` images should be used in the primary field holdout. `probable` images may be used in a secondary analysis. `ambiguous`, `unverified`, and `rejected` images must not be used as primary ground truth.

Do not correct a label solely because the model prediction looks visually plausible.

### 4.2 Preserve plant and capture groups

Images from the same plant, leaf, camera burst, or capture session must remain in one partition.

The split unit should be the strongest known group:

```text
plant_id > capture_session > image_family > exact image
```

If plant identity is unavailable, use capture-session and duplicate-family grouping and record that limitation.

### 4.3 Do not allow source to determine class

Create a cross-tabulation:

```text
source × class
```

The corrected training data should contain as much cross-source class coverage as the available data allows.

Avoid a structure such as:

```text
source_A → mostly healthy
source_B → mostly early blight
source_C → mostly late blight
```

If the data cannot provide source-balanced class coverage, maintain source-aware evaluation and explicitly report the limitation.

---

## 5. Data audit before retraining

The coding agent must complete this audit before launching full training.

### 5.1 File and image integrity

Check:

- Decodability.
- Corrupt files.
- Missing files.
- Duplicate paths.
- Exact SHA-256 duplicates.
- Perceptual duplicates.
- Invalid labels.
- Unsupported formats.
- Extreme dimensions.
- Empty or near-empty images.

### 5.2 Source metadata audit

Compare sources using:

```text
width
height
aspect_ratio
file_format
file_size
mean_rgb
brightness
contrast
sharpness
background_luminance
compression indicators
```

If a source classifier can identify the source with high accuracy, record a strong domain-shift warning.

### 5.3 Color-distribution audit

Compare by class and source:

```text
hue
saturation
brightness
contrast
leaf age if known
cultivar if known
disease severity
background brightness
```

Specifically inspect whether the current data resembles:

```text
dark olive-green → healthy
bright lime-green → late blight
```

If class color distributions are strongly separated, add counterexamples before retraining.

### 5.4 Required audit outputs

```text
reports/tomato/teacher_v2/data_audit_report.md
reports/tomato/teacher_v2/source_class_cross_tabulation.csv
reports/tomato/teacher_v2/color_distribution_report.json
reports/tomato/teacher_v2/duplicate_report.csv
reports/tomato/teacher_v2/label_review_log.csv
```

---

## 6. Build the v2 data partitions

Create a new versioned dataset and split. Do not silently modify the v1 manifest.

Recommended partitions:

```text
train: 70%
validation: 15%
locked benchmark test: 15%
external field holdout: separate and untouched
```

The external field holdout must not be included in the 70/15/15 split.

### 6.1 Training set

The training set may contain:

- Verified benchmark images.
- Verified field images.
- Healthy color hard negatives.
- Difficult disease images.
- Controlled augmentation.

### 6.2 Validation set

Use the validation set for:

- Checkpoint selection.
- Early stopping.
- Learning-rate scheduling.
- Threshold selection.
- Preprocessing comparison.

Do not use the locked test or external field holdout for these decisions.

### 6.3 Locked benchmark test

Keep the historical benchmark test for comparison, but do not claim that it proves field robustness.

If the v2 data changes the benchmark membership, create a new test manifest and retain the old result for historical comparison.

### 6.4 External field holdout

The preferred initial target is:

```text
100 healthy images
100 early_blight images
100 late_blight images
```

If this is not immediately possible, begin with at least 30–50 verified images per class and label the result preliminary.

The holdout should vary across:

- Location.
- Camera.
- Lighting.
- Soil, mulch, and natural backgrounds.
- Cultivar.
- Plant age.
- Leaf age.
- Disease severity.
- Leaf orientation.
- Partial and overlapping leaves.

The holdout must remain untouched until the final v2 evaluation.

### 6.5 Required split outputs

```text
manifests/tomato/teacher_v2_dataset_manifest.csv
manifests/tomato/teacher_v2_split_manifest.csv
manifests/tomato/teacher_v2_external_field_holdout.csv
reports/tomato/teacher_v2/split_integrity_report.md
```

---

## 7. Hard-negative and difficult-positive design

This is the most important data correction.

### 7.1 Healthy hard negatives

Add independently reviewed Healthy images with:

- Young pale leaves.
- Bright lime-green leaves.
- Direct flash.
- Strong sunlight.
- Shade.
- White backgrounds.
- Natural yellowing.
- Insect damage.
- Mechanical damage.
- Dust or soil spots.
- Shadows.
- Natural cultivar variation.
- Unusual but valid aspect ratios.

### 7.2 Difficult disease positives

Add verified Early Blight and Late Blight images with:

- Early subtle symptoms.
- Small lesions.
- Different backgrounds.
- Different lighting.
- Different cultivars.
- Partial leaves.
- Leaf-edge symptoms.
- Overlapping leaves.
- Different distances.
- Mature and early disease stages.

### 7.3 Ambiguous examples

Keep visually ambiguous cases in a separate set:

```text
ambiguous_review_set
```

Do not force them into a class merely to increase training size. Use them to evaluate uncertainty and abstention behavior.

---

## 8. Preprocessing contract

The same preprocessing policy must be used during training, evaluation, export, and future mobile integration.

Use:

```text
leaf localization or documented crop policy
aspect-preserving resize
controlled padding
no geometric stretching
quality checks
consistent color ordering
consistent normalization
```

Do not directly stretch panoramic images into a square if this changes lesion geometry.

Evaluate padding options using validation data only:

```text
0% padding
5% padding
10% padding
20% padding
```

Select one preprocessing version before the final external holdout evaluation.

Record:

```text
input resolution
resize method
padding value
padding color
RGB/BGR convention
normalization formula
quality thresholds
augmentation policy
```

Required output:

```text
manifests/tomato/teacher_v2_preprocessing_contract.json
```

---

## 9. Controlled augmentation policy

Use moderate augmentation to reduce shortcut learning without creating biologically unrealistic leaves.

Recommended augmentation categories:

- Horizontal flip when biologically acceptable.
- Small rotation.
- Small translation.
- Mild scale change.
- Hue jitter.
- Saturation jitter.
- Brightness variation.
- Contrast variation.
- White-balance variation.
- Mild shadow simulation.
- Mild glare simulation.
- Mild compression variation.

Avoid:

- Extreme hue changes.
- Severe geometric distortion.
- Augmentations that erase lesions.
- Unrealistic disease recoloring.
- Augmentation of validation or external holdout images.

Real field images are more valuable than synthetic color changes. Augmentation supports diversity; it does not replace field data.

---

## 10. Tomato teacher v2 training design

### 10.1 Architecture

Use EfficientNetB3 as the primary v2 teacher to isolate the effect of corrected data and preprocessing.

Do not change architecture, input size, preprocessing, optimizer, and dataset simultaneously in the first corrective experiment.

The purpose of v2 is to compare against v1 and determine whether the data correction works.

### 10.2 Training phases

Use a two-phase transfer-learning procedure.

#### Phase A: head warm-up

- Load ImageNet-pretrained EfficientNetB3.
- Keep the backbone frozen.
- Train only the classification head.
- Use the corrected training split.
- Monitor validation loss and macro-F1.
- Use early stopping.

#### Phase B: controlled fine-tuning

- Unfreeze the upper portion of the backbone first.
- Keep Batch Normalization layers frozen initially.
- Use a substantially smaller learning rate than Phase A.
- Unfreeze more layers only if validation improvement justifies it.
- Stop when validation loss and field-like validation metrics stop improving.

Do not use the external field holdout for early stopping or checkpoint selection.

### 10.3 Suggested initial configuration

Treat these as starting values, not guaranteed optimal settings:

```text
architecture: EfficientNetB3
input_size: 300x300 or the existing validated project size
optimizer: AdamW
head learning rate: 1e-3
fine-tuning learning rate: 1e-5 to 1e-4
weight decay: 1e-4
label smoothing: 0.0 to 0.05, selected on validation only
batch size: largest stable batch within VRAM budget
epoch budget: head warm-up 3–8 epochs
fine-tuning budget: 15–30 epochs
early stopping patience: 5–8 validation epochs
mixed precision: enabled only after numerical validation
random seeds: at least 3 if compute allows
```

Do not blindly copy these values if the existing codebase uses a different validated contract. Record the actual configuration.

### 10.4 Class imbalance

Compute class weights from the training partition only.

Do not compute weights using validation, benchmark test, or external holdout labels.

Compare weighted and unweighted training only if class imbalance is material. Select using validation metrics, not the locked test.

### 10.5 GPU execution

Use the RTX 4050 when available, but verify actual utilization and memory.

Record:

```text
GPU name
GPU utilization
peak VRAM
CPU utilization
RAM usage
batch size
mixed-precision status
images per second
time per epoch
validation time
augmentation status
```

Do not claim that GPU optimization is successful solely because training completes. The benchmark must demonstrate the bottleneck and measured improvement.

---

## 11. Required v2 experiments

Run experiments in controlled groups.

### Experiment A: corrected-data baseline

```text
v1 architecture
v2 corrected split
standard preprocessing
moderate augmentation
```

This is the primary retraining experiment.

### Experiment B: color-robustness comparison

Compare:

```text
standard augmentation
versus
moderate color and illumination augmentation
```

Select using validation and field-like validation data.

### Experiment C: preprocessing comparison

Compare padding and aspect-preserving policies using validation data only.

### Experiment D: background sensitivity

Evaluate controlled background perturbations without retraining:

```text
original
blurred background
darkened background
brightened background
background replacement
leaf-region occlusion
```

### Experiment E: optional ExG/necrosis gate

Treat the gate as a controlled safety experiment, not as the primary disease solution.

Compare:

```text
model only
model + ExG/necrosis gate
model + learned quality/OOD gate if available
```

The gate may output `uncertain`; it must not force `healthy`.

---

## 12. Evaluation protocol

Every v2 checkpoint must be evaluated in three stages.

### Stage 1: validation evaluation

Use for model and threshold decisions.

Report:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class recall.
- Confusion matrix.
- Calibration.
- Abstention behavior.

### Stage 2: locked benchmark evaluation

Run once after the configuration is frozen.

Report:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Confusion matrix.
- ECE.
- Brier score.

### Stage 3: external field holdout evaluation

Run after the model and thresholds are frozen.

Report:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Healthy false-positive rate.
- Disease false-negative rate.
- Field-versus-benchmark performance gap.
- Confidence distribution.
- High-confidence error count.
- Calibration.
- Abstention coverage.
- Performance by source, camera, lighting, cultivar, leaf age, and severity.

Do not average away a serious disease-class failure.

---

## 13. Tomato teacher v2 acceptance gates

The teacher is eligible to supervise a student only if all critical gates pass.

### Data and provenance gates

- Labels are independently reviewed or explicitly marked uncertain.
- Train, validation, benchmark test, and external holdout are group-disjoint.
- No exact duplicate leakage exists.
- No configured near-duplicate family crosses partitions.
- Source-class confounding is documented and reduced where possible.

### Robustness gates

- No unresolved major leaf-color shortcut.
- No major background-only class flips on the controlled perturbation test.
- White-background and unusual-aspect-ratio behavior is measured.
- Difficult early symptoms are evaluated.
- Healthy hard negatives are included.

### Performance gates

Use the project’s predefined thresholds, but do not accept a teacher solely because it exceeds the original benchmark score.

At minimum, require:

```text
acceptable per-class recall on the external field holdout
no unexplained high-confidence disease-to-Healthy failure pattern
field performance gap documented and judged acceptable
calibration and abstention behavior measured
```

If the external holdout is too small, the result remains preliminary and the teacher remains a benchmark candidate.

### Reproducibility gates

- Configuration is versioned.
- Model checksum is recorded.
- Random seed is recorded.
- Evaluation manifests are hashed.
- The result is reproducible across at least one rerun or seed comparison.

---

## 14. What to do if v2 still fails

If v2 still shows color or background shortcuts:

1. Do not distill it.
2. Do not declare the architecture inadequate immediately.
3. Inspect the failure cases and data distributions.
4. Add targeted, independently reviewed examples.
5. Recheck group and source structure.
6. Test a learned crop or quality model.
7. Repeat a controlled v3 data iteration.

If v2 fails only on a clearly out-of-scope input, decide whether the product should reject it. Do not silently exclude an input type that users are expected to submit.

If v2 fails on expected field inputs, the teacher remains no-go.

---

## 15. When to proceed to the tomato student

Proceed only after the teacher reaches:

```text
teacher v2 frozen
external field evaluation complete
known failures documented
preprocessing contract frozen
class order frozen
model checksum recorded
calibration and abstention behavior evaluated
```

Then use this sequence:

```text
teacher_v2 freeze
→ supervised tomato student baseline
→ evaluate supervised student
→ optional distillation comparison
→ convert candidates to LiteRT
→ quantization comparison
→ physical-device benchmark
→ mobile release gate
```

Do not start with distillation. A supervised student baseline is required to determine whether distillation adds value.

---

## 16. Distillation decision after v2

Distillation is optional, not automatic.

Run it only if:

- The teacher is independently validated.
- A smaller student is required for mobile constraints.
- The supervised student has a measurable weakness.
- The experiment has a predefined success criterion.

Compare:

```text
teacher v2
supervised student v2
one distilled student v2
```

Use the same benchmark, external field holdout, difficult-example set, calibration evaluation, conversion pipeline, and phone benchmark.

Accept the distilled student only if it provides a measured advantage in one or more of:

- External field per-class recall.
- Difficult-lesion recall.
- Calibration.
- Abstention behavior.
- Size.
- Latency.
- Memory.
- Quantization robustness.

Do not claim that distillation improves robustness merely because the student has a smaller file or because it performs well on the original benchmark.

---

## 17. Required artifact structure

Use a new versioned directory:

```text
models/tomato/teachers/v2/
├── teacher_best.keras
├── training_config.json
├── training_log.csv
├── model_manifest.json
├── checksum.sha256
└── evaluation_summary.json

manifests/tomato/teacher_v2/
├── dataset_manifest.csv
├── split_manifest.csv
├── external_field_holdout.csv
├── label_review_log.csv
└── group_assignments.csv

reports/tomato/teacher_v2/
├── data_audit_report.md
├── split_integrity_report.md
├── benchmark_evaluation.md
├── external_field_evaluation.md
├── calibration_report.md
├── background_sensitivity_report.md
├── preprocessing_comparison.md
├── failure_analysis.md
└── teacher_go_no_go_decision.md
```

Do not overwrite v1 artifacts.

---

## 18. Agent execution restrictions

The agent may run automatically:

- Dataset integrity checks.
- Duplicate and group audits.
- Source cross-tabulation.
- Color-distribution analysis.
- Contact-sheet generation.
- Small bounded preprocessing comparisons.
- Documentation and manifest validation.
- Model-shape and checksum checks.

The agent must request authorization before running:

```text
full tomato teacher retraining
multi-seed training
large hyperparameter sweeps
teacher training on newly added data
knowledge distillation
quantization-aware training
large external evaluation jobs
```

Before requesting authorization, report:

```text
exact command
purpose
training data version
split version
estimated runtime
GPU and VRAM requirement
RAM requirement
output directory
checkpoint behavior
whether existing files can be overwritten
```

The agent must never silently replace `tomato_teacher_v1`.

---

## 19. Final tomato teacher v2 go/no-go checklist

### Before training

- [ ] v1 artifact frozen.
- [ ] Dataset audit complete.
- [ ] Labels reviewed.
- [ ] Source-class table generated.
- [ ] Groups assigned.
- [ ] Duplicates checked.
- [ ] Healthy color hard negatives identified or added.
- [ ] Difficult disease examples identified or added.
- [ ] External holdout created and locked.
- [ ] Preprocessing contract written.
- [ ] Training configuration written.

### After training

- [ ] No NaN or training instability.
- [ ] Validation includes all classes.
- [ ] Best checkpoint selected only on validation data.
- [ ] Benchmark evaluation completed after freezing.
- [ ] External field evaluation completed.
- [ ] Background sensitivity measured.
- [ ] Color robustness measured.
- [ ] Calibration and abstention measured.
- [ ] High-confidence errors reviewed.
- [ ] Model checksum recorded.
- [ ] Teacher go/no-go decision documented.

### Before student work

- [ ] Teacher v2 passes independent field gates.
- [ ] Teacher limitations are documented.
- [ ] Teacher preprocessing is frozen.
- [ ] Student baseline plan is approved.
- [ ] Distillation has a defined reason and success criterion.

---

## Final recommendation

Retraining the tomato teacher is justified, but the retraining must be a **corrective data-and-evaluation iteration**, not a repetition of the previous training run.

The correct immediate sequence is:

```text
freeze tomato_teacher_v1
→ audit labels, sources, colors, backgrounds, and groups
→ create an independent field holdout
→ add healthy color hard negatives and difficult disease examples
→ freeze preprocessing and split contracts
→ train EfficientNetB3 tomato_teacher_v2
→ evaluate field-first robustness
→ freeze v2 only if it passes the gates
→ then build the tomato student
```

> **Do not transfer a teacher’s unresolved shortcuts into a mobile student. Fix and validate the teacher first.**

## References

[1]: /home/ubuntu/ipd_model_docs/current_status_next_steps.md "Tomato teacher current status and proper next steps"

[2]: /home/ubuntu/ipd_model_docs/potato_model_next_steps_plan.md "Potato model next-steps plan and execution controls"

[3]: /home/ubuntu/ipd_model_docs/rice_model_correction_and_release_plan.md "Rice model correction and release plan containing shared project governance principles"

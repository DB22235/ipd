# Rice Model Correction, Robustness, and Release Plan

**Project:** IPD rice plant disease detection

**Current models:**

- EfficientNetB3 teacher.
- Supervised MobileNetV3 student.
- Distilled MobileNetV3 student.

**Current classes:** Healthy, Blast, Brown Spot, Blight

**Document status:** Corrective action and release-control plan

**Primary purpose:** Resolve the problems that remain after the supervised and distilled rice models achieved very high performance on the current locked test set.

---

## 1. Current evidence and honest status

The current benchmark results are:

| Model | Test accuracy | Test macro-F1 | Blast recall | Approximate parameters |
|---|---:|---:|---:|---:|
| EfficientNetB3 teacher | 99.18% | 0.9879 | 95.83% | 12.24M |
| Supervised MobileNetV3 | 100.00% | 1.0000 | 100.00% | 3.00M |
| Distilled MobileNetV3 | 99.59% | 0.9937 | 97.22% | 3.00M |

The current audit established:

- No exact cryptographic hash overlap across train, validation, and test.
- No reported group overlap across train, validation, and test.
- No loader path overlap.
- Independent reproduction of the validation metric.
- Correct MobileNetV3-Large model identity.
- Strong performance on the current test distribution.

However, the audit also established a major unresolved issue:

```text
RiceDisease_Unknown:
    blast       960
    blight      1284
    brown_spot  1200
    healthy     0

RiceHealthyField_20190419:
    healthy     1488
    blast       0
    blight      0
    brown_spot  0
```

Therefore, **dataset source is perfectly correlated with class** in the current combined data.

This is source-label confounding. It is not the same as exact duplicate leakage, but it can produce inflated performance because the model may learn dataset-specific appearance instead of disease evidence.

### Current model status

The current models must be labeled as:

```text
benchmark candidates with unresolved source-domain confounding
```

They must not yet be labeled:

```text
field validated
production ready
robust disease detector
```

The supervised MobileNetV3 is currently the best predictive benchmark model. The distilled MobileNetV3 is currently the compact candidate, but it has lower benchmark performance and has not yet demonstrated a sufficient deployment advantage.

---

## 2. Problems this plan must solve

The corrective work must address these problems:

1. Perfect correlation between dataset source and class.
2. Unknown generalization to new field sources.
3. Overclaiming causal attribution from a small background perturbation test.
4. Unproven claim that distillation improved regularization.
5. Unproven claim that distillation improves INT8 robustness.
6. `.keras` file-size comparison being used as a proxy for mobile size.
7. No demonstrated phone latency or memory measurement.
8. No evidence yet that the distilled student is preferable to the supervised student.
9. Possible mismatch between benchmark success and real deployment success.
10. Potentially optimistic confidence and calibration results on a source-confounded test set.

The solution is not to blindly retrain repeatedly. First obtain the evidence needed to identify which problems are real and which are only documentation or evaluation problems.

---

## 3. Release-control decision

Do **not** release either student to users as the final rice model yet.

You may preserve and use them for:

- Controlled conversion tests.
- Quantization experiments.
- Source-held-out evaluation.
- Mobile latency benchmarking.
- Distillation analysis.

Do not use them for:

- Claims of field robustness.
- Automatic agronomic recommendations.
- Silent production deployment.
- Teacher-label generation for new data.
- Training a new student without recording the source-confounding limitation.

The correct status is:

```text
rice_teacher_v1_benchmark_candidate
rice_student_mobilenetv3_supervised_v1_benchmark_candidate
rice_student_mobilenetv3_distilled_v1_compact_benchmark_candidate
```

---

## 4. Corrective strategy overview

Follow this order:

```text
1. Freeze current artifacts and reports
2. Audit and document source differences
3. Determine whether source-balanced data exists
4. Create a source-aware evaluation design
5. Evaluate current students without changing them
6. Convert both students to mobile formats
7. Measure actual size, latency, and quantization degradation
8. Decide whether a new dataset or retraining is necessary
9. Retrain only after the source problem is addressed
10. Re-evaluate and release only the model that passes all gates
```

Do not begin with another temperature or alpha sweep. Distillation hyperparameters cannot repair a source-label-confounded dataset.

---

## 5. Step 1: freeze and preserve the current results

Preserve the following without overwriting:

```text
teacher checkpoint
supervised student checkpoint
distilled student checkpoint
training configurations
split manifest
label map
model checksums
benchmark metrics
confusion matrices
calibration reports
source cross-tabulation
```

Create a model registry entry containing:

```text
model_id
model_role
architecture
teacher_id if applicable
model_sha256
dataset_version
split_version
class_order
input_size
preprocessing_version
benchmark_metrics
source_confounding_status
mobile_conversion_status
release_status
known_limitations
```

Add this limitation to every current model card:

> All current classes are perfectly correlated with dataset source in the combined dataset. Benchmark results may therefore overestimate cross-source disease generalization. Independent source-balanced or source-held-out evaluation remains required.

---

## 6. Step 2: perform a source-domain audit

The immediate data task is to determine how different the two sources are visually and technically.

### 6.1 Create source contact sheets

Create balanced contact sheets for:

- Healthy images from `RiceHealthyField_20190419`.
- Disease images from `RiceDisease_Unknown`.
- Each disease class separately.
- Train, validation, and test samples separately.

Inspect:

- Background.
- Lighting.
- Camera style.
- Resolution.
- Aspect ratio.
- Compression.
- Crop framing.
- Leaf color.
- Leaf age.
- Lesion scale.
- Number of leaves.
- Field versus laboratory conditions.

Required outputs:

```text
reports/rice/source_audit/source_contact_sheet.png
reports/rice/source_audit/source_visual_comparison.md
```

### 6.2 Compare technical metadata

Report distributions by source for:

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
```

If the sources are distinguishable from technical metadata alone, document this as a strong domain-shift risk.

Required output:

```text
reports/rice/source_audit/source_metadata_comparison.csv
```

### 6.3 Train a source-only diagnostic classifier

Train a small, bounded diagnostic classifier with target labels:

```text
RiceDisease_Unknown
RiceHealthyField_20190419
```

This is not a disease model. It tests whether the source domains are visually easy to distinguish.

Interpretation:

- Near-random source accuracy: weaker evidence of source distinguishability.
- High source accuracy: strong domain gap.
- Near-perfect source accuracy: severe source confounding risk.

This diagnostic must not be presented as disease performance.

Required output:

```text
reports/rice/source_audit/source_classifier_report.md
```

---

## 7. Step 3: determine whether source-balanced data exists

Inspect all available datasets and classify them into:

| Data condition | Action |
|---|---|
| Healthy and disease images from the same source | Candidate for source-balanced training/evaluation |
| Multiple sources for every class | Best option for source-aware split |
| Only healthy field images from a second source | Useful healthy hard negatives, but insufficient for full source-held-out disease validation |
| Only one source for disease classes | Acquire or label additional disease sources |
| Unknown or weak labels | Quarantine until reviewed |

The goal is not merely to increase the image count. The goal is to break the relationship:

```text
source → class
```

A valid future dataset should contain as much of the following as possible:

```text
source A: healthy, blast, brown_spot, blight
source B: healthy, blast, brown_spot, blight
source C: healthy, blast, brown_spot, blight
```

If this is not possible, state the generalization limitation explicitly and use source-held-out tests only where class coverage makes them valid.

---

## 8. Step 4: create a source-aware evaluation design

Do not change the existing locked benchmark test set. Keep it for historical comparison.

Create an additional evaluation version with source metadata.

### Preferred design

Use at least one source or capture domain as an untouched holdout, while ensuring that the training data contains all target classes.

Example:

```text
training: source A + source B
validation: grouped subset of source A + source B
source-held-out test: source C
```

### If a source-held-out split cannot contain all classes

Do not report a misleading four-class accuracy. Instead:

- Report per-class results only for classes present.
- Mark the evaluation incomplete for full disease generalization.
- Acquire missing classes before making a full source-held-out claim.

### Required split metadata

```text
source_split_version
source_train_sources
source_validation_sources
source_test_sources
class_coverage_by_source
plant_or_leaf_group_rule
manifest_sha256
```

Required output:

```text
manifests/rice/rice_source_aware_split_v1.csv
reports/rice/source_audit/source_aware_split_report.md
```

---

## 9. Step 5: evaluate current models before retraining

Evaluate the existing teacher, supervised student, and distilled student on any valid source-aware or field-like data already available.

Do not retrain for this step.

Report for each model:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Healthy false-positive rate.
- Disease false-negative rate.
- Confusion matrix.
- Confidence distribution.
- Calibration.
- Abstention coverage.
- High-confidence errors.
- Source-wise performance.

Required comparison:

```text
supervised MobileNetV3
versus
distilled MobileNetV3
```

If both students fail similarly on source-aware data, this suggests the problem is inherited from the data or teacher rather than student architecture.

If the distilled student is better on source-aware data despite being slightly worse on the original benchmark, record that as a meaningful robustness result.

Required output:

```text
reports/rice/source_audit/current_models_source_aware_evaluation.md
```

---

## 10. Step 6: correct the claims in the distillation report

Update the current comparative report using the following language.

### Replace “parity with the large teacher” with

> The distilled MobileNetV3 remains highly accurate on the current locked benchmark and is substantially smaller in parameter count than the EfficientNetB3 teacher. Its performance must still be evaluated on source-aware and field-like data before generalization can be claimed.

### Replace “dark knowledge regularization” with

> The distillation objective successfully transferred softened teacher outputs using the selected temperature and loss weights. In this experiment, the distilled student performed below the supervised student on the current benchmark, so a beneficial regularization effect has not been demonstrated.

### Replace “significantly more robust to INT8” with

> INT8 robustness is a hypothesis requiring direct comparison of supervised and distilled float and quantized models. No robustness conclusion should be made before those measurements.

### Add this limitation

> The current benchmark has perfect source-label correlation: all healthy images originate from one source and all disease images originate from another. Consequently, the reported benchmark metrics may overestimate cross-source disease generalization.

---

## 11. Step 7: compare real mobile artifacts

The `.keras` file size must not be used as the final mobile-size comparison because saved optimizer state and serialization settings may differ.

Convert both students with the same settings:

```text
supervised MobileNetV3 → float32 LiteRT/TFLite
supervised MobileNetV3 → float16 LiteRT/TFLite
supervised MobileNetV3 → int8 LiteRT/TFLite

distilled MobileNetV3 → float32 LiteRT/TFLite
distilled MobileNetV3 → float16 LiteRT/TFLite
distilled MobileNetV3 → int8 LiteRT/TFLite
```

Use the same representative calibration set for both int8 conversions. The calibration set must cover all classes and relevant image variation and must not be the locked test set.

For every artifact, record:

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
model_sha256
```

Required output:

```text
reports/rice/conversion/student_conversion_comparison.csv
reports/rice/conversion/student_conversion_report.md
```

---

## 12. Step 8: measure quantization degradation honestly

Create this comparison:

| Model | Format | Accuracy | Macro-F1 | Blast recall | Blight recall | Brown Spot recall | Healthy recall | Size | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Supervised | Float32 | measure | measure | measure | measure | measure | measure | measure | measure |
| Supervised | Float16 | measure | measure | measure | measure | measure | measure | measure | measure |
| Supervised | Int8 | measure | measure | measure | measure | measure | measure | measure | measure |
| Distilled | Float32 | measure | measure | measure | measure | measure | measure | measure | measure |
| Distilled | Float16 | measure | measure | measure | measure | measure | measure | measure | measure |
| Distilled | Int8 | measure | measure | measure | measure | measure | measure | measure | measure |

Calculate:

```text
quantization_accuracy_delta
quantization_macro_f1_delta
quantization_per_class_recall_delta
float_to_litert_prediction_agreement
```

Only after this comparison may the report state whether the distilled model is more robust to INT8 conversion.

---

## 13. Step 9: measure actual phone performance

The final student choice must be based on the target phone, not on `.keras` size.

Measure for both selected float and quantized candidates:

- Cold-start latency.
- Warm median latency.
- P95 latency.
- Peak memory.
- Model package size.
- CPU delegate latency.
- NNAPI or hardware delegate latency if supported.
- Thermal or battery observations when available.

Do not claim “sub-50 ms” without measurements on named devices.

Decision rule:

```text
choose distilled model only if its actual mobile advantage justifies its measured accuracy loss
```

If both models meet the phone budget, prefer the model with better source-aware performance and per-class recall.

Required output:

```text
reports/rice/mobile/student_device_benchmark.md
```

---

## 14. Step 10: decide whether retraining is necessary

### Retraining is required if

- Source-aware evaluation is materially worse than the benchmark.
- Healthy and disease classes remain source-confounded in the training data.
- New source-balanced data changes the label distribution substantially.
- One class has unacceptable recall on field-like images.
- Preprocessing differs between training and deployment.
- Labels are shown to be unreliable.

### Retraining is not required immediately if

- The existing data is source-balanced after audit.
- Source-aware performance is strong.
- The problem is only missing conversion or latency measurements.
- The model passes independent test and field-like evaluation.
- The only issue is an overstrong wording in the report.

Do not retrain only because the distilled student is 0.41 percentage points below the supervised student on the current benchmark. First determine whether the distilled student provides a meaningful size or latency advantage.

---

## 15. If retraining is required: dataset correction protocol

When source confounding is confirmed, use this order:

1. Quarantine the source-confounded benchmark as historical.
2. Collect or obtain healthy and disease images from multiple sources.
3. Verify labels independently.
4. Remove exact and near duplicates.
5. Assign leaf, plant, capture-session, and source groups.
6. Create a source-aware split.
7. Reserve an untouched field-like holdout.
8. Train the supervised MobileNetV3 baseline again.
9. Evaluate the new baseline.
10. Distill only after the corrected teacher/student evidence is available.

Use hard negatives such as:

- Healthy field leaves.
- Healthy leaves with natural color variation.
- Healthy leaves under different light.
- Healthy leaves with soil, water, or sky backgrounds.
- Non-disease damage.
- Nutrient-stress examples clearly labeled as unsupported or separate.

Do not mix ambiguous plant-stress images into Healthy merely to increase diversity.

---

## 16. Distillation decision after correction

Distillation is optional, not automatically required.

Use the supervised MobileNetV3 if it:

- Meets the mobile size budget.
- Meets the device latency budget.
- Has better source-aware performance.
- Has better per-class recall.
- Has acceptable quantized degradation.

Use the distilled MobileNetV3 if it:

- Has a meaningful LiteRT/TFLite size or latency advantage.
- Does not introduce unacceptable class-specific failures.
- Performs competitively on source-aware and field-like data.
- Preserves calibration and abstention behavior.

If another distillation run is justified, run only a small targeted comparison, such as:

```text
T = 2, alpha = 0.75
T = 3, alpha = 0.75
```

The higher hard-label weight is reasonable because the supervised student already performs strongly, but this is an experiment, not an assumption.

Do not run a large temperature/alpha sweep without a clear deployment reason.

---

## 17. Release gates

A model can be called a **benchmark candidate** when:

- Model identity is verified.
- Metrics reproduce independently.
- Exact and group overlap checks pass.
- Test results are documented.

A model can be called a **mobile candidate** when:

- LiteRT/TFLite conversion succeeds.
- Class order and preprocessing are validated.
- Quantized metrics are measured.
- Model size is recorded.
- Device latency is measured.

A model can be called **field-validation candidate** when:

- It is evaluated on source-aware or independent field-like data.
- Class coverage is adequate.
- Labels are independently reviewed.
- Source and capture groups are separated.
- Per-class performance is acceptable.

A model can be called **production candidate** only when:

- It passes source-aware and field-like evaluation.
- Quantization and device behavior are acceptable.
- Calibration and abstention are defined.
- Known limitations are documented.
- The mobile package is reproducible and checksum-verified.
- Rollback and model-version metadata exist.

---

## 18. Required documentation updates

Update these files after corrective work:

```text
docs/rice_student_model_card.md
docs/rice_student_experiment_log.md
docs/rice_student_change_log.md
reports/rice/student/student_go_no_go_decision.md
reports/rice/distillation/distillation_report.md
reports/rice/source_audit/source_visual_comparison.md
reports/rice/source_audit/source_classifier_report.md
reports/rice/source_audit/current_models_source_aware_evaluation.md
reports/rice/conversion/student_conversion_report.md
reports/rice/mobile/student_device_benchmark.md
```

Every update must include:

- Date.
- Model version.
- Dataset version.
- Split version.
- Code commit.
- Exact files changed.
- Metrics.
- Known limitations.
- Decision.
- Next action.

Do not overwrite old reports. Create a new version or append a dated correction section.

---

## 19. Execution restrictions

The AI coding agent may perform without additional approval:

- Read-only inventory.
- Metadata inspection.
- Hash checks.
- Manifest checks.
- Small source-audit reports.
- Model identity checks.
- Small synthetic conversion tests.
- Documentation updates.

The agent must prepare commands and wait for user authorization before running:

- Full source-classifier training.
- Full source-aware evaluation if large.
- Retraining the student.
- Retraining the teacher.
- Distillation reruns.
- Quantization-aware training.
- Full device benchmarks.
- Large hyperparameter sweeps.

Before requesting an expensive run, report:

```text
command
purpose
expected runtime
hardware
RAM/VRAM expectation
outputs
whether old artifacts will be overwritten
```

Never overwrite the current benchmark models.

---

## 20. Immediate practical action list

The next actions are intentionally limited:

### Action 1: preserve current results

Freeze current model files, checksums, configurations, and reports.

### Action 2: document source confounding

Generate source contact sheets and metadata comparison. Do not retrain yet.

### Action 3: determine available source-balanced data

Inventory whether healthy and disease examples exist across multiple sources.

### Action 4: evaluate current students on any valid source-aware data

Do not alter the models. Compare supervised and distilled MobileNetV3.

### Action 5: convert both students

Create float32, float16, and int8 candidates using identical conversion settings.

### Action 6: measure actual mobile trade-off

Benchmark on the intended target phone before selecting the final student.

### Action 7: decide on retraining

Retrain only if source-aware evaluation shows a real generalization problem or if corrected data becomes available.

### Action 8: update model status

Use the most honest status supported by evidence:

```text
benchmark candidate
mobile candidate
field-validation candidate
production candidate
```

Do not skip statuses.

---

## 21. Final engineering conclusion

The current distillation experiment successfully produced a compact MobileNetV3, but the remaining problem is not primarily the distillation formula. The most serious unresolved problem is the perfect source-label correlation in the dataset.

The supervised student currently has the best benchmark metrics. The distilled student is only preferable if its converted mobile artifact provides a meaningful size, latency, memory, or quantization advantage without unacceptable loss of real-world performance.

The correct path is:

```text
freeze current results
→ audit and correct source confounding
→ evaluate current models on source-aware data
→ compare real LiteRT/TFLite artifacts
→ measure target-phone behavior
→ retrain only if evidence requires it
→ distill only when compression provides a justified benefit
```

> **Do not solve a data-domain problem with another distillation sweep. First prove whether the models generalize across sources, then choose the smallest model that preserves the required behavior.**

Author: **Manus AI**

# Rice Teacher Post-Training Evaluation and Go/No-Go Plan

**Document status:** Authoritative post-training instruction

**Project:** IPD dual-mode plant disease detection system

**Crop:** Rice

**Teacher classes:** Healthy, Blast, Brown Spot, Blight

**Primary instruction:** Training completion is not evidence of model readiness. Evaluate and validate the rice teacher before beginning student training, knowledge distillation, quantization, or mobile deployment.

---

## 1. Immediate decision

The rice teacher has completed training. The next task is **evaluation and validation**, not knowledge distillation.

Follow this sequence:

```text
verify model artifact
→ evaluate locked test set
→ evaluate independent field-like holdout
→ perform error analysis
→ test shortcut sensitivity
→ calibrate confidence and abstention
→ make teacher go/no-go decision
→ freeze teacher version if accepted
→ begin student experiments only after acceptance
```

Knowledge distillation must not begin until the teacher has been evaluated on data that was not used for training, model selection, threshold tuning, or preprocessing decisions.

---

## 2. Role of the evaluation agent

Act as a senior industrial ML engineer, computer-vision scientist, data scientist, and model-validation reviewer.

The evaluation agent must:

1. Inspect the repository and locate the exact trained model and configuration.
2. Identify the exact dataset and split version used for training.
3. Verify the class order and preprocessing contract.
4. Refuse to report production readiness if the test split is missing, leaked, or mutable.
5. Separate benchmark performance from field performance.
6. Report uncertainty, failure cases, and limitations honestly.
7. Never tune the model using the hidden test set.
8. Never report only accuracy.
9. Never begin distillation merely because training finished.
10. Preserve all evaluation artifacts and version them.

---

## 3. Artifact verification before evaluation

Locate and verify the following files:

```text
best rice teacher checkpoint
training configuration
model manifest
rice split manifest
class-label mapping
training log
validation log
training and validation curves
model checksum
environment details
preprocessing configuration
```

The model must use the intended class order:

```text
healthy
blast
brown_spot
blight
```

If the class order differs, record the actual order and ensure that every evaluation and mobile inference script uses the same mapping.

Confirm the following model metadata:

| Field | Required value |
|---|---|
| Crop | Rice |
| Number of classes | 4 |
| Input dimensions | Exact dimensions used during training |
| Color order | RGB or documented alternative |
| Normalization | Exact training and inference behavior |
| Backbone | Exact backbone and pretrained-weight status |
| Checkpoint | Best validation checkpoint, if available |
| Split version | Immutable split identifier |
| Class order | Four-class mapping |
| Model version | Unique version string |

Do not evaluate an unspecified or modified checkpoint as the official teacher result.

---

## 4. Locked test-set evaluation

Use the locked test set only for final reporting. Do not use it to change:

- Epoch count.
- Learning rate.
- Architecture.
- Augmentation.
- Class weights.
- Crop strategy.
- Confidence thresholds.
- Abstention thresholds.
- Distillation parameters.

If any of these decisions were changed after viewing test results, the test set is no longer a clean final test set. Record this and create a new evaluation version.

### Required test metrics

Generate all of the following:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Balanced accuracy.
- Per-class precision.
- Per-class recall.
- Per-class F1.
- Confusion matrix.
- One-vs-rest AUROC where meaningful.
- Confidence distribution.
- Expected calibration error.
- Reliability diagram.
- Selective accuracy at abstention thresholds.
- Abstention coverage.

The primary review must focus on macro-F1 and per-class recall, not only overall accuracy.

### Rice-specific confusion review

Inspect these confusion pairs carefully:

- Healthy versus Blast.
- Healthy versus Brown Spot.
- Healthy versus Blight.
- Brown Spot versus Blight.
- Blast versus Blight.

Record the number and percentage of errors for every pair. Save representative examples for every important confusion.

Required output files:

```text
reports/rice/teacher_test_metrics.json
reports/rice/teacher_test_classification_report.csv
reports/rice/teacher_test_confusion_matrix.png
reports/rice/teacher_test_reliability_diagram.png
reports/rice/teacher_test_error_manifest.csv
```

---

## 5. Independent field-like holdout evaluation

The field-like holdout is the most important evaluation after the locked test set.

The holdout must not have been used for:

- Training.
- Hyperparameter selection.
- Epoch selection.
- Threshold selection.
- Augmentation decisions.
- Architecture decisions.
- Label correction based on model output.

The holdout should include, where available:

- Natural field backgrounds.
- Soil and water reflections.
- Bright sunlight.
- Shade and overcast conditions.
- Different rice varieties.
- Different leaf ages.
- Healthy leaves with varied color.
- Small and large leaves.
- Partial leaves.
- Multiple leaves.
- Early disease symptoms.
- Advanced disease symptoms.
- Blur and camera noise.

If no independent field-like holdout exists, stop the readiness review and report:

> Benchmark evaluation is possible, but field robustness has not been established.

Do not call the model field-validated without this holdout.

### Field holdout label requirements

Each field image must have an independently reviewed label. Store:

```text
image_id
reviewer_1_label
reviewer_2_label
expert_adjudication
label_confidence
symptoms_observed
crop_confirmed
usable_leaf_confirmed
source
plant_id
capture_session
location_if_known
```

If reviewers disagree, classify the image as ambiguous and exclude it from the primary metric until resolved.

### Field metrics

Report the same metrics as the test set, plus:

- Benchmark-to-field performance gap.
- Per-source performance.
- Per-background performance.
- Per-lighting performance.
- Per-disease-severity performance.
- High-confidence error count.
- Healthy false-positive rate.
- Disease false-negative rate.
- Abstention rate.

Required output files:

```text
reports/rice/field_holdout_metrics.json
reports/rice/field_holdout_classification_report.csv
reports/rice/field_holdout_confusion_matrix.png
reports/rice/field_holdout_error_manifest.csv
reports/rice/field_holdout_model_card_section.md
```

---

## 6. Error analysis

Create a row for every incorrect, rejected, or uncertain prediction.

Required fields:

```text
image_id
ground_truth
predicted_class
confidence
second_class
second_confidence
source
background_type
lighting_condition
leaf_size_or_quality
crop_quality
error_category
reviewer_comment
recommended_action
```

Use these error categories:

```text
label_error
healthy_disease_confusion
disease_disease_confusion
background_shortcut
color_shortcut
lesion_not_visible
blur_or_quality
crop_failure
multiple_leaf_failure
unsupported_input
calibration_error
unknown
```

Review at least:

- Every high-confidence error.
- Every healthy-to-disease error.
- Every Brown Spot-to-Blight error.
- Every Blast-to-Blight error.
- At least 20 false positives when available.
- At least 20 false negatives when available.
- At least 20 uncertain or rejected images when available.

A high-confidence error is more important than a low-confidence error because it may indicate unsafe overconfidence or a strong shortcut.

Do not correct labels solely because the model disagrees. Label correction requires independent review.

---

## 7. Shortcut and robustness audit

Run controlled image perturbation tests on a representative sample from every class.

For every selected image, generate:

- Original image.
- Background blur.
- Background darkening.
- Background brightening.
- Background occlusion.
- Natural background replacement.
- Leaf-region occlusion.
- Lesion-region occlusion when lesions are visible.

Record:

```text
image_id
original_class
variant_type
predicted_class
confidence
logit_change
class_flip
```

### Interpretation rules

| Observation | Interpretation |
|---|---|
| Background-only change flips the class | Model is background-sensitive |
| Lesion occlusion has little effect | Model may not rely on disease evidence |
| Healthy images are consistently classified as disease after background changes | Possible leaf-color or morphology shortcut |
| Confidence changes strongly without class change | Calibration is fragile |
| Disease image remains unchanged after lesion removal | Investigate shortcut learning |

Grad-CAM may be used to support visual review, but it is not sufficient proof of biological reasoning. Do not use an arbitrary heatmap threshold as the only acceptance criterion.

Required output:

```text
reports/rice/shortcut_audit_report.json
reports/rice/shortcut_audit_examples/
```

---

## 8. Confidence calibration and abstention

Raw softmax confidence is not guaranteed to represent probability of correctness.

Measure:

- Expected calibration error.
- Reliability diagram.
- Confidence for correct predictions.
- Confidence for incorrect predictions.
- Selective accuracy.
- Coverage after abstention.
- Class-specific confidence behavior.

Select the abstention threshold using validation data only. Do not choose it from the hidden test set.

The mobile and online systems must be able to return:

```text
uncertain
```

when the image is low quality, out of distribution, unsupported, or below the calibrated confidence threshold.

Recommended user-facing message:

> Uncertain result. Capture a closer, well-lit image of one rice leaf or use online analysis.

Do not force every input into Healthy, Blast, Brown Spot, or Blight.

Required output:

```text
reports/rice/calibration_report.json
reports/rice/reliability_diagram.png
reports/rice/abstention_threshold_selection.json
```

---

## 9. Teacher go/no-go decision

Complete the following decision table.

| Gate | Result | Status |
|---|---|---|
| Model artifact verified | Record result | Pass/Fail |
| Class order verified | Record result | Pass/Fail |
| Preprocessing verified | Record result | Pass/Fail |
| Test split locked | Record result | Pass/Fail |
| Duplicate leakage absent | Record result | Pass/Fail |
| Benchmark metrics complete | Record result | Pass/Fail |
| Per-class recall acceptable | Record result | Pass/Fail |
| Independent field holdout exists | Record result | Pass/Fail |
| Field metrics complete | Record result | Pass/Fail |
| Label-review process documented | Record result | Pass/Fail |
| High-confidence errors analyzed | Record result | Pass/Fail |
| Shortcut audit complete | Record result | Pass/Fail |
| Calibration measured | Record result | Pass/Fail |
| Abstention behavior defined | Record result | Pass/Fail |
| Reproducibility verified | Record result | Pass/Fail |

### Go decision

Proceed to student development only when:

- No known leakage remains.
- Class mapping and preprocessing are correct.
- Benchmark metrics are complete.
- No class has unacceptable recall.
- Independent field-like evaluation does not collapse.
- High-confidence errors are understood or acceptably rare.
- Shortcut behavior has been investigated.
- Calibration and abstention are defined.
- The result is reproducible.

### No-go decision

Remain in teacher-validation mode if:

- No independent field-like holdout exists.
- Duplicate or group leakage is detected.
- Labels are unreliable or unresolved.
- One class has poor recall.
- Healthy leaves are frequently predicted as disease.
- Brown Spot and Blight are heavily confused.
- High-confidence errors are common.
- Background perturbations cause frequent class flips.
- Calibration is poor and no abstention policy exists.
- Preprocessing differs between training and evaluation.

Required decision artifact:

```text
reports/rice/teacher_go_no_go_decision.md
```

---

## 10. If the teacher passes: freeze the teacher version

Before student development, freeze the accepted teacher as a versioned artifact.

Record:

- Model file.
- Model checksum.
- Dataset version.
- Split version.
- Training configuration.
- Best epoch.
- Validation metrics.
- Test metrics.
- Field metrics.
- Calibration settings.
- Known limitations.
- Code commit.
- Environment.

Suggested version name:

```text
rice_teacher_v1_field_validated
```

Do not overwrite the accepted teacher when running student experiments.

---

## 11. Student work after teacher approval

Only after teacher approval follow this sequence:

```text
freeze accepted rice teacher
→ train supervised MobileNetV3-Large baseline
→ benchmark MobileNetV3-Small and MobileNetV2 if required
→ train distilled student
→ compare teacher, supervised student, and distilled student
→ convert float32 student to LiteRT/TFLite
→ evaluate float16, dynamic-range, and int8 candidates
→ test on target Android devices
```

Distillation uses a supervised plus soft-target objective:

```text
L = alpha * CE(y, student_logits)
  + (1 - alpha) * T^2 * KL(
        softmax(teacher_logits / T),
        softmax(student_logits / T)
    )
```

Do not assume that distillation corrects the teacher. Distillation compresses behavior and may transfer teacher errors.

---

## 12. Required final evaluation report

Create:

```text
reports/rice/rice_teacher_post_training_report.md
```

The report must include:

1. Model and environment information.
2. Dataset and split versions.
3. Class counts by partition.
4. Leakage and duplicate audit.
5. Locked-test metrics.
6. Field-holdout metrics.
7. Confusion matrices.
8. Calibration results.
9. Abstention analysis.
10. Shortcut audit results.
11. Error analysis.
12. Known limitations.
13. Reproducibility information.
14. Teacher go/no-go decision.
15. Exact next action.

The report must clearly distinguish:

```text
benchmark performance
field-like performance
production readiness
```

These are not interchangeable.

---

## 13. Final instruction

The immediate next implementation task is:

> Run a complete evaluation and error-analysis report on the locked rice test set and an independently labeled field-like holdout. Do not begin knowledge distillation until the teacher go/no-go review passes.

Training completion is only a milestone. The rice teacher becomes eligible for student development only after the evaluation proves that it generalizes beyond the training distribution and does not rely on unsafe shortcuts.

---

## References

[1]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[2]: https://keras.io/examples/keras_recipes/better_knowledge_distillation/ "Knowledge distillation recipes"
[3]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"
[4]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"

Author: **Manus AI**

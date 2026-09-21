# MobileNetV3 High-Results Diagnostic and Prevention Plan

**Project:** IPD rice disease detection

**Model:** MobileNetV3 student baseline

**Classes:** Healthy, Blast, Brown Spot, Blight

**Current observation:** MobileNetV3 has reached approximately 99.69% validation accuracy at epoch 8 and 100% validation accuracy at epoch 9 onward, with very low validation loss.

**Purpose:** Determine whether the high result is genuine, caused by an easy dataset, or inflated by leakage, preprocessing errors, source shortcuts, validation mistakes, or evaluation bugs. For every possible issue, run a practical check and apply a documented prevention method if the issue is found.

**Realistic constraint:** Do not run a large investigation blindly. Start with the highest-value, lowest-cost checks. Execute expensive checks only when an earlier check produces evidence of a problem or when the user authorizes them.

---

## 1. Current decision

The current result is **encouraging but unverified**.

Do not:

- Immediately switch to MobileNetV2.
- Immediately start knowledge distillation.
- Immediately claim that MobileNetV3 is field-ready.
- Discard the current checkpoint.
- Rerun the entire training job before checking the existing data and evaluation.

Do:

1. Let the current run finish or stop through the configured early-stopping policy.
2. Preserve the best checkpoint and training logs.
3. Run the low-cost diagnostic checks below.
4. Evaluate the saved checkpoint on the locked test set.
5. Decide whether additional investigation or retraining is actually necessary.

---

## 2. Possible causes of unusually high results

The high validation result may be caused by one or more of these conditions:

1. The rice classification problem is genuinely easy.
2. The validation set is too small or unrepresentative.
3. Exact duplicate leakage exists.
4. Near-duplicate or augmented-image leakage exists.
5. The same leaf, plant, capture session, or image sequence crosses partitions.
6. Dataset source or background is correlated with the class.
7. Preprocessing accidentally exposes the label or source.
8. The validation pipeline is not evaluating the intended images.
9. The labels are incorrect, simplified, or inconsistent.
10. The student is accidentally evaluated on training data or cached training data.
11. The model checkpoint or evaluation script is not the expected model.
12. The validation metric is computed incorrectly.
13. The model is overfitting despite high validation accuracy.
14. The test and validation distributions are much easier than deployment data.
15. The training pipeline differs from the final inference pipeline.

The agent must test these causes in priority order rather than assuming that 100% validation accuracy is either genuine or fraudulent.

---

## 3. Priority order for investigation

Use this order because it gives the highest information for the least compute:

| Priority | Check | Expected cost | Decision value |
|---:|---|---:|---|
| 1 | Manifest and split counts | Very low | Detects basic pipeline mistakes |
| 2 | Training/validation path audit | Very low | Detects accidental data reuse |
| 3 | Exact hash overlap | Low | Detects direct leakage |
| 4 | Class distribution and confusion matrix | Low | Shows whether the metric is meaningful |
| 5 | Checkpoint and evaluation verification | Low | Detects wrong-model evaluation |
| 6 | Near-duplicate and group overlap | Medium | Detects realistic leakage |
| 7 | Source/background/class cross-tabulation | Low to medium | Detects shortcut learning |
| 8 | Locked test evaluation | Low to medium | Measures generalization |
| 9 | Error and confidence inspection | Low | Detects hidden failure patterns |
| 10 | Field-like holdout | Medium | Measures deployment relevance |
| 11 | Repeated-seed or alternative split | High | Measures stability |

Do not start with retraining or a large hyperparameter sweep.

---

## 4. Check 1: verify dataset and split counts

### Issue being tested

The training script may be using a different dataset, split, or file count than expected.

### Check

Read the exact manifest loaded by the MobileNetV3 training script and report:

```text
train_image_count
validation_image_count
test_image_count
train_group_count
validation_group_count
test_group_count
class_counts_by_partition
source_counts_by_partition
manifest_path
manifest_sha256
```

Compare these values with the training log and configuration.

### Evidence of a problem

- Counts in the log do not match the manifest.
- Validation set is unexpectedly tiny.
- A class is absent or nearly absent from validation.
- The script creates a new random split instead of loading the locked split.
- The test or field holdout is included in training.

### Prevention or correction

- Require the script to load an immutable split manifest.
- Print manifest path and checksum at startup.
- Fail if counts differ from the configuration.
- Store train, validation, test, and field paths separately.
- Do not allow automatic re-splitting during training.

### Required artifact

```text
reports/rice/student/manifest_and_split_audit.json
```

---

## 5. Check 2: verify no exact duplicates cross partitions

### Issue being tested

The same file or pixel-identical image may appear in both training and validation.

### Check

Compute a cryptographic hash after reading the original files and, separately, after standardized decoding if necessary. Compare hashes across:

```text
train vs validation
train vs test
train vs field holdout
validation vs test
```

### Evidence of a problem

Any identical image exists across partitions.

### Prevention or correction

- Keep one canonical copy.
- Assign all duplicate copies to one group.
- Recreate the split manifest.
- Retrain and invalidate metrics generated from the contaminated split.
- Increment the split version, for example from `rice_split_v1` to `rice_split_v2`.

Do not merely remove the duplicate from validation while keeping the original split metadata unchanged.

### Required artifact

```text
reports/rice/student/exact_hash_overlap_report.csv
```

---

## 6. Check 3: verify no near-duplicate or augmentation leakage

### Issue being tested

The validation image may be a resized, compressed, cropped, flipped, brightness-adjusted, or otherwise modified version of a training image.

### Check

Use:

- Perceptual hashes.
- Image dimensions and filenames.
- Augmentation naming patterns.
- Optional image embeddings for difficult cases.
- Contact sheets of suspicious pairs.

Compare all partitions.

### Evidence of a problem

- Very similar images appear across partitions.
- One original image and its augmentations cross the split boundary.
- The same image sequence or capture burst is divided across partitions.

### Prevention or correction

- Create a shared `group_id` for every original/augmentation family.
- Split groups rather than files.
- Keep generated augmentations out of the stored dataset where possible; generate them only during training.
- Recreate and lock a group-safe split.
- Re-run training and invalidate the old result.

### Required artifact

```text
reports/rice/student/near_duplicate_review.csv
reports/rice/student/suspicious_pairs_contact_sheet.png
```

---

## 7. Check 4: verify leaf, plant, capture-session, and source grouping

### Issue being tested

Images of the same leaf, plant, plot, video sequence, or capture session may be split across training and validation even when file hashes differ.

### Check

Inspect available metadata, filenames, directory structure, EXIF data, source URLs, and visual clusters. Look for:

- Same leaf from different angles.
- Same plant captured repeatedly.
- Same background and framing sequence.
- Consecutive video frames.
- Same source folder appearing in multiple partitions.

### Evidence of a problem

Images from the same physical or acquisition group cross partitions.

### Prevention or correction

- Create the highest-confidence `group_id` available.
- Keep the whole group in one partition.
- If group identity is unknown, use source-held-out evaluation as a conservative substitute.
- Document the limitation instead of claiming complete independence.

### Required artifact

```text
reports/rice/student/group_overlap_report.csv
```

---

## 8. Check 5: inspect class balance and confusion matrix

### Issue being tested

The 100% validation accuracy may be caused by a trivial or unbalanced validation set.

### Check

Generate:

- Validation class counts.
- Test class counts.
- Per-class precision, recall, and F1.
- Confusion matrix.
- Majority-class baseline accuracy.

### Evidence of a problem

- Validation contains very few examples for a class.
- Accuracy is high but macro-F1 is materially lower.
- One class dominates the validation set.
- The confusion matrix is missing or only overall accuracy is reported.

### Prevention or correction

- Use grouped stratification where possible.
- Report macro-F1 and balanced accuracy.
- Require minimum class representation or mark the result provisional.
- Use class-aware sampling or class weights only after measuring imbalance.
- Do not use accuracy as the sole model-selection metric.

### Required artifact

```text
reports/rice/student/validation_class_distribution.csv
reports/rice/student/student_confusion_matrix.png
```

---

## 9. Check 6: verify the evaluation script is using the correct model

### Issue being tested

The evaluation may be loading the teacher, a previous student, a training checkpoint, or a different file than the reported MobileNetV3 checkpoint.

### Check

At evaluation startup, print and record:

```text
model_path
model_sha256
model_parameter_count
model_input_shape
model_output_shape
model_class_order
configuration_path
split_manifest_path
```

Compare the SHA-256 checksum with the intended epoch checkpoint.

### Evidence of a problem

- The evaluation path does not match the training output path.
- Parameter count does not match MobileNetV3.
- The model output has the wrong number of classes.
- The script evaluates a cached prediction file.
- The model checkpoint is not the best or intended checkpoint.

### Prevention or correction

- Require the model path as an explicit argument.
- Print and save the checksum.
- Disable stale prediction-cache reuse unless the model checksum matches.
- Store model metadata beside the model file.
- Fail on class-order mismatch.

### Required artifact

```text
reports/rice/student/evaluation_model_identity.json
```

---

## 10. Check 7: verify training and validation paths are separate

### Issue being tested

A path or data-loader bug may place training images into validation, or the validation loader may reuse a training cache.

### Check

For a sample of every partition, print:

- Relative path.
- Hash.
- Partition name.
- Label.
- Group ID.

Verify that:

```text
train loader paths ∩ validation loader paths = empty
train cache ∩ validation cache = empty
```

Inspect the data-loader code for:

- Reused mutable lists.
- Incorrect glob patterns.
- Directory-level labels applied incorrectly.
- Cache keys that omit partition name.
- Validation dataset accidentally pointing to the training directory.

### Evidence of a problem

Any validation item is also present in the training loader or cache.

### Prevention or correction

- Build train and validation datasets from separate immutable manifest subsets.
- Include `partition` in cache keys.
- Add an assertion that path and hash intersections are empty.
- Fail before training if the assertion fails.

### Required artifact

```text
reports/rice/student/loader_partition_assertion.json
```

---

## 11. Check 8: verify labels and class mapping

### Issue being tested

The model may be learning an easy folder/source distinction, or labels may be mapped incorrectly during evaluation.

### Check

Verify:

- Folder label.
- Manifest label.
- Integer encoding.
- `labels.txt` order.
- Model output index order.
- Confusion-matrix class names.
- Teacher class order.
- Student class order.

Manually inspect a contact sheet from every class.

### Evidence of a problem

- Class names differ between training and evaluation.
- Brown Spot and Blight are swapped in one script.
- The labels are derived from alphabetical order in one place and explicit order in another.
- A folder contains mixed or incorrect images.

### Prevention or correction

- Define one canonical `rice_label_map.json`.
- Load it everywhere.
- Never infer class order from filesystem sorting.
- Add a startup assertion comparing all mappings.
- Quarantine or relabel incorrect images with an audit log.

### Required artifact

```text
reports/rice/student/label_mapping_audit.json
```

---

## 12. Check 9: detect source, background, or camera shortcuts

### Issue being tested

The model may classify the data source instead of rice disease symptoms.

### Check

Create cross-tabulations of class against:

- Dataset source.
- Directory name.
- Background type.
- Camera or resolution.
- Image aspect ratio.
- Lighting.
- Compression pattern.
- Water, soil, sky, or laboratory background.

Train or inspect a simple source/background-only baseline if practical. If a simple model can predict the disease class from the background or metadata, the dataset contains a shortcut risk.

### Evidence of a problem

- One source maps almost entirely to one class.
- Background type is strongly class-specific.
- A background-only crop predicts the class above chance.
- Source-held-out performance drops sharply.

### Prevention or correction

- Add source-balanced data.
- Include diverse backgrounds for each class.
- Use source-held-out validation.
- Use natural crop/localization consistently.
- Add background perturbation augmentation.
- Do not rely on random image-level validation alone.

### Required artifact

```text
reports/rice/student/source_background_shortcut_report.md
```

---

## 13. Check 10: inspect color and symptom shortcuts

### Issue being tested

The model may associate color, brightness, lesion size, or image composition with classes instead of disease morphology.

### Check

Compare classes by:

- Hue.
- Saturation.
- Brightness.
- Leaf age if known.
- Lesion size.
- Background luminance.
- Image aspect ratio.

Run controlled variants:

- Mild hue shift.
- Brightness change.
- Contrast change.
- Background blur.
- Background replacement.
- Leaf-region occlusion.
- Lesion-region occlusion where possible.

### Evidence of a problem

- Predictions change when only color is changed.
- Lesion-region occlusion has little effect.
- Healthy and disease classes have non-overlapping color distributions caused by source bias.

### Prevention or correction

- Add real hard negatives with varied healthy colors.
- Add verified disease examples across lighting and severity.
- Use moderate color and illumination augmentation.
- Use natural crop and aspect-ratio handling.
- Add field-like validation by source and lighting.

Do not solve this with a fixed “green means healthy” rule.

### Required artifact

```text
reports/rice/student/color_shortcut_audit.json
```

---

## 14. Check 11: verify the validation metric implementation

### Issue being tested

The reported 100% may come from incorrect metric accumulation, wrong label shape, ignored samples, or a metric reset bug.

### Check

Compare validation accuracy computed independently after training:

1. Load the saved checkpoint.
2. Run prediction on the validation set.
3. Compute accuracy directly from predictions and labels.
4. Compare with the logged `val_accuracy`.
5. Compute macro-F1 independently.
6. Confirm the number of evaluated samples.

Check that:

- Metrics reset at the correct time.
- All validation batches are included.
- No samples are dropped unexpectedly.
- Labels are not compared to one-hot arrays incorrectly.
- Padding or repeated samples are not counted as real examples.

### Evidence of a problem

Independent accuracy differs from the logged metric, or evaluated sample count is wrong.

### Prevention or correction

- Use a standalone evaluation function.
- Report evaluated sample count.
- Add unit tests with known predictions and labels.
- Do not trust the training log until independent evaluation agrees.

### Required artifact

```text
reports/rice/student/metric_reproduction_report.json
```

---

## 15. Check 12: test for overfitting and checkpoint instability

### Issue being tested

The model may fit the current validation set unusually well while generalizing poorly elsewhere.

### Check

Compare:

- Epoch 9 checkpoint.
- Lowest-validation-loss checkpoint.
- Final checkpoint if available.
- Locked test results.
- Field-like holdout results.

Inspect the curves for:

- Training accuracy continuing to rise while validation worsens.
- Validation loss becoming unstable.
- Large metric changes between adjacent epochs.
- A single unusually lucky validation result.

### Evidence of a problem

- Validation is perfect but test performance drops materially.
- Results vary strongly by checkpoint.
- Field-like performance is poor.
- High-confidence errors increase.

### Prevention or correction

- Monitor validation loss or macro-F1 in addition to accuracy.
- Restore the best checkpoint using a field-relevant validation metric.
- Increase validation diversity.
- Use group-safe or source-held-out validation.
- Add hard negatives and retrain only if the error analysis supports it.

Do not rerun training merely because the score is high. Rerun only when evidence identifies a problem.

### Required artifact

```text
reports/rice/student/checkpoint_comparison.csv
```

---

## 16. Check 13: evaluate on the locked test set

### Issue being tested

The validation score may not transfer to unseen data.

### Check

Evaluate the intended best checkpoint once on the locked test set.

Report:

- Accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class precision and recall.
- Confusion matrix.
- Healthy false-positive rate.
- Confidence distribution.
- Calibration.

### Evidence of a problem

- Large validation-to-test drop.
- One class collapses.
- High-confidence errors are frequent.
- Test split is not actually independent.

### Prevention or correction

- If leakage exists, invalidate the old result and recreate the split.
- If the test is clean but performance drops, use the test errors to guide new training data.
- Keep the test set locked; do not tune directly on it.

### Required artifact

```text
reports/rice/student/mobilenetv3_locked_test_report.md
```

---

## 17. Check 14: evaluate on field-like or source-held-out data

### Issue being tested

The benchmark may be easier than real deployment data.

### Check

Evaluate on an untouched field-like or source-held-out set containing:

- Natural backgrounds.
- Different lighting.
- Different rice varieties.
- Different leaf ages.
- Small lesions.
- Partial leaves.
- Early disease.
- Healthy leaves with varied appearance.

### Evidence of a problem

- Field/source-held-out performance is materially lower than validation/test performance.
- Background or lighting changes cause class flips.
- Healthy images are frequently classified as disease.

### Prevention or correction

- Treat the model as benchmark-only.
- Add verified field/source-diverse data.
- Use source-held-out training validation.
- Add a quality gate and calibrated abstention.
- Retrain only after identifying the dominant error type.

### Required artifact

```text
reports/rice/student/mobilenetv3_field_holdout_report.md
```

---

## 18. Check 15: verify the student did not accidentally copy the teacher or use teacher outputs as labels

### Issue being tested

The supervised baseline may not be independent if it accidentally loads teacher predictions, teacher weights, or teacher-generated labels.

### Check

Inspect the baseline configuration and training script for:

- Teacher model loading.
- Distillation loss.
- Teacher logits.
- Pseudo-label files.
- Teacher-generated target files.
- Weight initialization from the teacher.

For a supervised baseline, the only target should be the ground-truth label. The student may use ImageNet weights if documented, but it must not use the rice teacher weights unless intentionally designed as transfer learning.

### Evidence of a problem

The baseline uses teacher outputs or teacher weights without recording this.

### Prevention or correction

- Separate baseline and distillation code paths.
- Add `model_role: student_baseline` to the configuration.
- Assert that the teacher is not loaded in baseline mode.
- Retrain the baseline if contamination occurred.

### Required artifact

```text
reports/rice/student/baseline_independence_check.json
```

---

## 19. Check 16: verify preprocessing consistency

### Issue being tested

The model may receive easier or different images during validation than during real inference.

### Check

Compare training, validation, test, and intended mobile preprocessing:

- Image decoder.
- RGB/BGR order.
- Resize method.
- Crop method.
- Aspect-ratio handling.
- Normalization.
- Quantization assumptions.
- Augmentation enabled/disabled state.

Run the same image through the Keras validation path and the standalone inference path and compare tensors.

### Evidence of a problem

- Validation applies a different crop or normalization.
- Mobile preprocessing differs from training.
- RGB/BGR mismatch exists.
- Images are stretched in one path and aspect-ratio-preserved in another.

### Prevention or correction

- Create one versioned preprocessing function or contract.
- Add tensor-equivalence tests.
- Put preprocessing metadata in the model manifest.
- Fail package validation on preprocessing mismatch.

### Required artifact

```text
reports/rice/student/preprocessing_equivalence_report.json
```

---

## 20. Check 17: inspect whether validation augmentation is accidentally active or inactive

### Issue being tested

The validation pipeline may be using training augmentation, or training may be receiving no intended augmentation.

### Check

Verify:

- Training augmentation is enabled only for training.
- Validation and test transformations are deterministic.
- Random augmentation is not applied to validation.
- Augmentation does not leak labels.
- The configuration reflects the actual runtime behavior.

### Evidence of a problem

- Validation images change between evaluations.
- Training and validation logs show identical transformations unexpectedly.
- Augmentation layer is disabled due to backend or data-pipeline incompatibility.

### Prevention or correction

- Separate `train_transform` and `eval_transform` explicitly.
- Run a small visual transform test.
- Record augmentation configuration in the experiment report.
- Use deterministic evaluation transforms.

### Required artifact

```text
reports/rice/student/augmentation_pipeline_report.md
```

---

## 21. Minimal practical diagnostic run

Do not run every expensive check immediately. The first practical run should execute only these checks:

1. Print manifest and split counts.
2. Verify model identity and checksum.
3. Verify class mapping.
4. Verify train/validation path separation.
5. Compute exact hash overlap.
6. Recompute validation metrics independently.
7. Evaluate the saved checkpoint on the locked test set.
8. Generate a confusion matrix and per-class report.

These checks should usually be completed before deciding whether a larger investigation is needed.

The agent may execute these only if they are bounded and do not run a large full-dataset job unexpectedly. If the evaluation involves a large dataset or significant runtime, prepare the command and ask the user to run it.

---

## 22. Decision matrix after diagnostics

| Finding | Decision |
|---|---|
| No leakage, independent metric reproduction, strong locked test | Keep MobileNetV3 baseline and continue to deployment comparison |
| Exact or near-duplicate leakage | Invalidate affected metrics, recreate split, retrain |
| Group/source overlap | Rebuild grouped or source-held-out split and retrain/evaluate |
| Wrong checkpoint or metric bug | Correct evaluation and rerun bounded evaluation |
| Strong benchmark but weak field/source-held-out result | Add representative data and retrain later; do not claim field readiness |
| Strong result and acceptable mobile budget | Keep MobileNetV3; distillation is optional |
| MobileNetV3 too slow or large | Benchmark MobileNetV2 using the same split and preprocessing |
| One disease class has weak recall | Add reviewed hard examples and investigate labels before architecture change |
| High confidence errors remain | Calibrate, add abstention, and inspect shortcut behavior |
| All checks pass and no field holdout exists | Call it benchmark-validated, not field-validated |

---

## 23. Prevention controls to add permanently

Regardless of whether a problem is found, add these controls to the codebase:

1. Immutable split manifest loading.
2. Manifest checksum printed at training and evaluation startup.
3. Exact train/validation/test path-disjoint assertion.
4. Group-overlap validation when group IDs exist.
5. Canonical label-map loading.
6. Model checksum and model identity logging.
7. Independent metric recomputation.
8. Per-class metrics by default.
9. Separate validation and test commands.
10. No automatic resplitting.
11. No stale prediction-cache reuse without checksum matching.
12. Preprocessing contract validation.
13. Baseline mode that refuses to load teacher outputs.
14. New output directory per experiment.
15. Model registry entry for every checkpoint selected for comparison.
16. A field/source-held-out evaluation path.
17. Calibration and abstention reporting.
18. Explicit runtime warnings when only benchmark data is available.

---

## 24. What not to conclude yet

Do not conclude any of the following from 100% validation accuracy alone:

- MobileNetV3 is production-ready.
- MobileNetV3 will work on field images.
- Knowledge distillation is unnecessary.
- MobileNetV2 is unnecessary forever.
- The dataset has no leakage.
- The model learned disease morphology.
- The model is calibrated.
- The model will work after TFLite conversion.
- The model will be fast enough on the target phone.

These conclusions require the relevant checks and measurements.

---

## 25. Realistic immediate plan

### Now

1. Let the current MobileNetV3 training run finish or stop via early stopping.
2. Preserve the checkpoint and log.
3. Run the minimal practical diagnostic run.
4. Evaluate the saved best checkpoint on the locked test set.

### If the minimal checks pass

1. Keep MobileNetV3 as the primary student candidate.
2. Measure model size and conversion compatibility.
3. Run a bounded field/source-held-out evaluation if available.
4. Decide whether distillation is worth the extra training.
5. Convert the best candidate only after the evaluation decision.

### If a problem is found

1. Record the exact failure.
2. Apply the prevention method in this document.
3. Invalidate affected metrics.
4. Recreate only the necessary split, evaluation, or training stage.
5. Do not rerun the entire project without evidence that it is necessary.

---

## 26. Final engineering conclusion

The MobileNetV3 result may be genuinely excellent, especially because the rice classes may be visually separable. However, 100% validation accuracy is also compatible with leakage, source shortcuts, evaluation mistakes, or an overly easy validation distribution.

The correct response is not to panic and not to celebrate prematurely.

> **Verify the existing run with a small, targeted audit. If the split, metric, model identity, and locked-test result are valid, keep MobileNetV3 and move forward. Retrain only when a specific problem is found.**

The first decision after this plan is therefore:

```text
minimal audit passes?
    yes → evaluate deployment suitability and continue with MobileNetV3
    no  → apply the relevant prevention, invalidate affected result, and repeat only the necessary stage
```

Author: **Manus AI**

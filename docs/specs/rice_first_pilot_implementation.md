# IPD Rice-First Pilot Implementation Specification

**Document status:** Authoritative implementation plan for the first complete crop pilot

**Project:** IPD dual-mode plant disease detection system

**Pilot crop:** Rice

**Initial classes:** Healthy, Blast, Brown Spot, Blight

**Primary objective:** Build and validate the first field-aware rice teacher model before developing the distilled offline student model.

**Non-negotiable rule:** Do not begin knowledge distillation, quantization, or mobile deployment until the rice teacher passes the data-quality, leakage, field-like evaluation, calibration, and shortcut-sensitivity gates defined in this document.

---

## 1. Strategic decision

Rice is approved as the first complete crop pilot because it provides an opportunity to establish the correct machine-learning process before repeating the tomato failure modes.

This decision does **not** mean that rice will automatically be easier or more accurate. The rice dataset currently contains unresolved risks, including duplicates, possible near-duplicates, unknown data leakage, source imbalance, and unverified label consistency.

The correct strategy is:

```text
rice dataset audit
→ leakage-safe split
→ label and source review
→ field-like holdout
→ EfficientNetB3 teacher baseline
→ shortcut and calibration audit
→ teacher improvement
→ teacher freeze
→ MobileNet student baseline
→ knowledge distillation
→ LiteRT/TFLite conversion
→ mobile device validation
```

Rice is the first pilot crop, not a shortcut that permits skipping data governance.

---

## 2. Product scope for the rice pilot

The first rice model is a crop-specific four-class image classifier.

| Class | Required interpretation |
|---|---|
| Healthy | Rice leaf with no disease symptoms according to the project label definition |
| Blast | Rice blast symptoms verified against the project label definition |
| Brown Spot | Brown spot symptoms verified against the project label definition |
| Blight | Blight symptoms verified against the project label definition |

The model must support an **uncertain or reject outcome** for images that are:

- Not rice.
- Not a usable leaf image.
- Too blurred or dark.
- Too small or partially visible.
- Outside the supported disease distribution.
- Low-confidence or poorly calibrated.

The model must not force every input into one of the four disease classes.

---

## 3. Responsibilities of the AI implementation agent

Act as a senior industrial ML engineer, computer-vision scientist, data scientist, MLOps engineer, and validation reviewer.

For every change:

1. Inspect the current repository and existing scripts before editing.
2. Preserve the immutable dataset and split manifests.
3. Record assumptions explicitly.
4. Separate observed evidence from hypotheses.
5. Avoid optimizing against the hidden test set.
6. Report missing data instead of inventing it.
7. Never claim field robustness from a random image split alone.
8. Produce reproducible scripts and configuration files rather than relying on notebook state.
9. Record dataset versions, code commit, environment, seeds, and model checksums.
10. Stop when a gate fails instead of silently continuing to the next phase.

---

## 4. Repository and artifact structure

Create or preserve the following structure:

```text
ipd/
├── data/
│   ├── raw/rice/
│   ├── interim/rice/
│   ├── processed/rice/
│   └── field_holdout/rice/
├── manifests/rice/
├── configs/rice/
├── src/rice/
├── experiments/rice/
├── models/rice/
├── reports/rice/
└── mobile/rice/
```

Required final artifacts include:

```text
reports/rice/
├── dataset_audit_report.md
├── duplicate_report.csv
├── label_review_log.csv
├── source_distribution_report.csv
├── class_distribution_report.csv
├── split_integrity_report.md
├── field_holdout_review.csv
├── teacher_metrics.json
├── teacher_model_card.md
├── shortcut_audit_report.json
├── calibration_report.json
└── go_no_go_decision.md
```

---

## 5. Phase 0: environment and reproducibility setup

Before data processing or training, record:

- Python version.
- Keras version.
- TensorFlow or PyTorch version.
- Keras backend.
- CUDA version.
- GPU name.
- Operating system.
- Installed package lock or environment file.
- Git commit.
- Dataset source paths.
- Dataset checksums.
- Random seeds.

The RTX 4050 training process must verify actual CUDA availability and model placement. The agent must not treat the existence of a GPU as proof that training is using it.

Create a configuration file such as:

```yaml
crop: rice
classes:
  - healthy
  - blast
  - brown_spot
  - blight
input_size: [300, 300]
seed: 42
split_version: rice_split_v1
teacher_backbone: EfficientNetB3
student_candidates:
  - MobileNetV3Large
  - MobileNetV3Small
  - MobileNetV2
```

Do not begin student experiments during the teacher-validation phases.

---

## 6. Phase 1: rice dataset audit

Create one manifest row per image with these fields:

```text
image_id
path
crop
label
source_dataset
source_url_or_reference
original_file_hash
perceptual_hash
width
height
format
capture_group
leaf_or_plant_id
capture_session
location_if_known
license
review_status
ambiguity_reason
```

### 6.1 File integrity checks

For every image:

- Verify that the file exists.
- Decode it successfully.
- Confirm that it contains an RGB or convertible image.
- Record width and height.
- Detect zero-byte or corrupted files.
- Detect unsupported formats.
- Record unusually small or large images.

Do not silently skip invalid files. Record the failure and reason.

### 6.2 Duplicate detection

Run all of the following:

1. Exact file hashing, such as SHA-256.
2. Pixel-level duplicate checks after standard decoding.
3. Perceptual hashing.
4. Optional image-embedding similarity for difficult cases.
5. Manual review of near-duplicate clusters.

Images from the same original image, leaf, plant, capture session, video sequence, or augmentation family must share one `group_id`.

Do not place duplicate families across train, validation, and test.

### 6.3 Label audit

Review samples from every source and every class. Confirm that folder names match actual visual labels.

The audit must identify:

- Wrong folder labels.
- Ambiguous disease symptoms.
- Images with no usable rice leaf.
- Images containing multiple leaves with conflicting states.
- Images where symptoms are too small to verify.
- Images that may show nutrient deficiency, pest damage, mechanical damage, or natural senescence instead of the target disease.
- Images whose class definition differs from the rest of the dataset.

Assign one of:

```text
verified
ambiguous
incorrect_label
unsupported
needs_expert_review
```

Images marked `ambiguous`, `incorrect_label`, or `unsupported` must not enter the primary benchmark until resolved.

### 6.4 New rice healthy dataset policy

Do not merge the newly found rice healthy dataset immediately.

First compare it with existing rice data by:

- Source.
- Camera.
- Lighting.
- Background.
- Leaf age.
- Cultivar if known.
- Image size and aspect ratio.
- Color distribution.
- Label definition.
- License and redistribution rights.

The new healthy data may be used in one of three ways:

| Condition | Action |
|---|---|
| Same visual domain and verified labels | Candidate for training after deduplication |
| Different visual domain but valid labels | Training data plus a source-stratified evaluation slice |
| Uncertain labels or incompatible definition | Quarantine until reviewed |

Never add healthy images only to increase class balance.

---

## 7. Phase 2: leakage-safe data split

Split groups, not individual files.

The initial target is:

| Partition | Target proportion | Purpose |
|---|---:|---|
| Train | 70% | Model fitting and augmentation |
| Validation | 15% | Model and threshold selection |
| Test | 15% | Final locked benchmark |

The exact percentages may change if the grouped data structure or rare classes makes stratification unstable. The group rule is more important than exact percentages.

### 7.1 Split rules

1. Deduplicate before assigning groups.
2. Assign each `group_id` to exactly one partition.
3. Preserve class balance as far as groups permit.
4. Preserve source metadata for subgroup evaluation.
5. Place at least one independent source or field-like collection in test when possible.
6. Save `rice_split_manifest_v1.csv`.
7. Lock the split manifest after approval.
8. Do not regenerate the split during hyperparameter tuning.
9. Do not use the hidden test set for threshold, architecture, or augmentation decisions.

### 7.2 Split integrity checks

The split audit must confirm:

- No exact-hash overlap.
- No perceptual-duplicate overlap.
- No group overlap.
- No source sequence leakage.
- No augmentation-family leakage.
- No accidental class-order mismatch.
- No validation/test images inside the training cache.

If any leakage is found, invalidate the split and create a new version, such as `rice_split_v2`.

---

## 8. Phase 3: field-like holdout

Create a separate field-like holdout that is not used for tuning.

The holdout should include variation in:

- Rice varieties.
- Leaf age.
- Healthy color and texture.
- Natural field backgrounds.
- Soil and water reflections.
- Bright sunlight.
- Shade and overcast conditions.
- Blur and camera noise.
- Partial leaves.
- Multiple leaves.
- Different disease severity levels.
- Early and advanced symptoms.

If the available set is small, call it a **preliminary field-like holdout**. Do not call it a production field benchmark.

The field holdout must have an independent label-review process. A label must not be changed solely because it disagrees with the model.

Recommended review fields:

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
```

Images with unresolved disagreement should be reported separately and excluded from the primary accuracy metric.

---

## 9. Phase 4: preprocessing and crop policy

Use a consistent preprocessing pipeline for training, validation, test, and deployment.

Recommended initial procedure:

1. Decode image as RGB.
2. Detect or confirm the rice leaf region.
3. Use a natural unmasked crop where possible.
4. Preserve aspect ratio.
5. Avoid stretching wide or tall images directly into a square.
6. Use documented crop or padding behavior.
7. Resize to 300 × 300 for the EfficientNetB3 teacher unless experiments justify another resolution.
8. Record crop failures.
9. Reject or abstain on images with no usable leaf.

Do not use black-background masking as an untested default. If masking is evaluated, train and infer with exactly matched preprocessing and compare against natural cropping.

Test crop padding values such as:

```text
0%, 5%, 10%, 20%
```

Select the preprocessing strategy using validation and field-like performance, not benchmark accuracy alone.

---

## 10. Phase 5: EfficientNetB3 teacher baseline

Use EfficientNetB3 as the first teacher candidate because it provides a strong transfer-learning baseline. If the implementation uses ImageNet weights, call it **ImageNet-pretrained transfer learning**, not training from scratch.

### 10.1 Initial model

Use:

- `include_top=False`.
- ImageNet weights.
- 300 × 300 input.
- Global average pooling.
- A simple classification head.
- Linear logits during training.
- Softmax only for reporting or inference.
- Ordinary `Dropout` after global pooling.
- Frozen BatchNorm during initial fine-tuning unless evidence supports updating it.

Start with a simple head before testing a larger dense head:

```text
Input
→ EfficientNetB3 backbone
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dropout
→ Dense(4, activation=None)
```

Do not use `SpatialDropout` on a two-dimensional feature vector unless the implementation has been explicitly verified.

### 10.2 Two-stage training

#### Stage 1: head training

- Backbone frozen.
- Train classification head only.
- Use early stopping.
- Save the best validation checkpoint.
- Validate once per epoch.

#### Stage 2: controlled fine-tuning

- Unfreeze the final one or two meaningful EfficientNet blocks.
- Keep BatchNorm frozen initially.
- Use a learning rate at least 10 times smaller than Stage 1.
- Recompile after changing trainability.
- Use early stopping and restore the best weights.

Do not define fine-tuning solely as “unfreeze the top N layers.” Use meaningful backbone blocks and record exactly which layers are trainable.

### 10.3 Training controls

Use conservative augmentation first:

- Mild horizontal flip only if biologically valid.
- Small rotations.
- Moderate crop and scale variation.
- Mild brightness and contrast variation.
- Moderate hue and saturation variation.
- Mild blur or compression perturbation.

Do not use extreme transformations that destroy lesion morphology.

Use class weights or class-aware sampling only after measuring class imbalance. Record the exact class-weight formula and values.

Use the GPU acceleration instructions for:

- CUDA verification.
- RAM ingestion.
- Batch-size tuning.
- Mixed precision.
- Input pipeline benchmarking.
- GPU and VRAM reporting.

---

## 11. Phase 6: teacher evaluation

Evaluate the teacher on validation, locked test, and field-like holdout separately.

Required metrics:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Balanced accuracy.
- Per-class precision.
- Per-class recall.
- Confusion matrix.
- Healthy false-positive rate.
- Blast false-negative rate.
- Brown Spot false-negative rate.
- Blight false-negative rate.
- Expected calibration error.
- Reliability diagram.
- Selective accuracy at abstention thresholds.
- Abstention coverage.
- Source-wise performance.
- Background-wise performance.
- Image-quality subgroup performance.
- Disease-severity performance where labels exist.

The model must not be accepted because of one combined accuracy number.

### 11.1 Rice-specific confusion checks

Inspect these pairs carefully:

- Healthy versus early disease.
- Brown Spot versus Blight.
- Blast versus Blight.
- Healthy versus Brown Spot.

Review false positives and false negatives manually. Save representative examples and explanations.

---

## 12. Phase 7: shortcut and robustness audits

Perform controlled perturbations on a representative sample from each class.

For each image, create:

- Original image.
- Background blur.
- Background darkening.
- Background brightening.
- Natural background replacement.
- Background occlusion.
- Leaf-region occlusion.
- Lesion-region occlusion where lesions are identifiable.

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

Interpretation:

- If background-only changes cause frequent class flips, the model has background sensitivity.
- If lesion occlusion has little effect, the model may not rely on disease evidence.
- If a healthy image is invariant but consistently misclassified, the model may be using a leaf appearance shortcut.
- If confidence changes drastically without a class flip, calibration is still fragile.

Use Grad-CAM as qualitative evidence only. Do not use an arbitrary heatmap threshold as the sole acceptance criterion.

---

## 13. Phase 8: teacher acceptance gate

The rice teacher may proceed to student development only when all applicable gates pass.

| Gate | Requirement |
|---|---|
| Data integrity | Corrupt files, duplicates, and ambiguous labels are documented |
| Leakage | No known duplicate, near-duplicate, group, or sequence overlap across partitions |
| Label quality | Primary benchmark labels are independently reviewed or documented as reliable |
| Benchmark quality | Strong macro-F1 with no catastrophic per-class failure |
| Field-like quality | Performance does not collapse on the field-like holdout |
| Healthy protection | Healthy false-positive rate is acceptable for the product risk level |
| Disease recall | No target disease class has unacceptable recall |
| Calibration | Confidence and abstention behavior are measured and calibrated |
| Shortcut audit | No unexplained dependence on background or non-disease artifacts |
| Reproducibility | Configuration, manifest, environment, and seed reproduce the result |
| Deployment path | Teacher/student export path is technically feasible |

The threshold values must be approved after inspecting the data size and label quality. Do not invent a high threshold merely to make the report look successful.

If a gate fails, remain in teacher-validation mode.

---

## 14. Phase 9: student development after teacher freeze

Only after the rice teacher is accepted should the student phase begin.

Benchmark at least:

- MobileNetV3-Large.
- MobileNetV3-Small.
- MobileNetV2.
- Optional EfficientNet-Lite candidate if supported by the deployment stack.

First train a supervised student baseline. Then train a distilled student using the frozen teacher.

Use a combined loss:

```text
L = alpha * CE(y, student_logits)
  + (1 - alpha) * T^2 * KL(
        softmax(teacher_logits / T),
        softmax(student_logits / T)
    )
```

Start with a small validation-only search over:

```text
T ∈ {2, 3, 4, 6}
alpha ∈ {0.25, 0.50, 0.75}
```

Compare:

- Teacher.
- Supervised student.
- Distilled student.
- Quantized student.

Do not assume distillation improves the model. Accept the student only if it meets the mobile budget and preserves important per-class recall and abstention behavior.

---

## 15. Phase 10: LiteRT/TFLite conversion

Convert the supervised/distilled student in this order:

1. Float32 model.
2. Float16 candidate.
3. Dynamic-range candidate.
4. Full-int8 candidate with representative calibration data.
5. Quantization-aware training only if post-training quantization causes unacceptable degradation.

The representative calibration set must:

- Come from training or validation data.
- Cover all rice classes.
- Cover lighting and background variation.
- Exclude the hidden test set.
- Be versioned and documented.

For every converted model, record:

- File size.
- Input and output tensor types.
- Class order.
- Preprocessing contract.
- Output agreement with Keras.
- Accuracy and macro-F1 delta.
- Per-class recall delta.
- Latency on target Android devices.
- Peak memory.
- Model checksum.

Do not assume that a smaller quantized model is automatically faster on every phone.

---

## 16. Phase 11: mobile pilot

The rice mobile pilot must include:

- Explicit crop selection or crop verification.
- Image-quality checks.
- Offline inference.
- Calibrated confidence.
- Abstention for uncertain input.
- Model version display.
- Exact preprocessing implementation.
- Local prediction logging with consent.
- No silent image upload.
- Clear online fallback.

The model package must contain:

```text
model.tflite
labels.txt
model_manifest.json
preprocessing.md
checksum.txt
release_notes.md
```

The manifest must specify:

- Crop.
- Class order.
- Input dimensions.
- Color order.
- Normalization.
- Quantization scale and zero-point if applicable.
- Abstention threshold.
- Model version.
- Minimum compatible app version.

---

## 17. Prohibited shortcuts

The implementation agent must not:

- Train before duplicate and leakage checks.
- Merge the new healthy rice data without compatibility review.
- Use random image-level splitting as the only evaluation.
- Tune on the hidden test set.
- Call ImageNet-pretrained training “from scratch.”
- Use an arbitrary color rule as the disease detector.
- Force every image into a disease class.
- Distill an unvalidated teacher.
- Claim field robustness from a small hand-labeled sample.
- Use Grad-CAM alone as proof of biological reasoning.
- Report accuracy without class-level metrics.
- Skip preprocessing or class-order checks during TFLite conversion.

---

## 18. Immediate implementation checklist

The coding agent must complete these tasks in order:

### Data tasks

- [ ] Locate and inventory all rice image sources.
- [ ] Generate the rice manifest.
- [ ] Compute exact hashes.
- [ ] Compute perceptual hashes.
- [ ] Create duplicate and near-duplicate groups.
- [ ] Review class definitions.
- [ ] Review the new healthy rice dataset.
- [ ] Record source, license, and capture information.
- [ ] Quarantine ambiguous and unsupported images.

### Split tasks

- [ ] Create `rice_split_manifest_v1.csv`.
- [ ] Verify group-disjoint train, validation, and test partitions.
- [ ] Preserve source metadata.
- [ ] Lock the split manifest.
- [ ] Produce the split integrity report.

### Teacher tasks

- [ ] Build the EfficientNetB3 baseline.
- [ ] Verify CUDA and GPU placement.
- [ ] Benchmark disk-backed and RAM-ingested pipelines.
- [ ] Train the frozen-backbone stage.
- [ ] Fine-tune selected upper blocks.
- [ ] Save the best checkpoint.
- [ ] Evaluate benchmark and field-like data.
- [ ] Run calibration and shortcut audits.
- [ ] Produce the teacher model card.

### Decision task

- [ ] Complete the rice teacher go/no-go review.
- [ ] Proceed to student development only if all required gates pass.

---

## 19. Final engineering conclusion

Starting with rice is approved and strategically sensible, but the advantage comes from applying a better process—not from assuming that rice is automatically free of the tomato problem.

The rice pilot succeeds only if it establishes a trustworthy sequence:

> clean and independently reviewed data, leakage-safe splitting, field-like evaluation, shortcut testing, calibrated abstention, and teacher validation before compression.

If this process succeeds, rice can become the first complete path from dataset to cloud teacher to offline student to mobile inference. If it fails, the failure should be treated as useful evidence about the data or task rather than hidden behind a high random-split accuracy.

---

## References

[1]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[2]: https://keras.io/examples/keras_recipes/better_knowledge_distillation/ "Knowledge distillation recipes"
[3]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"
[4]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"

Author: **Manus AI**

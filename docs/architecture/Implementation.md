# IPD Model Implementation Specification

## 1. Scope and recommended architecture

The first production milestone should contain **three independent crop-specific classifiers** rather than one four-way-or-more mixed classifier:

| Model | Classes to verify | Cloud role | Offline role |
|---|---|---|---|
| Potato | Healthy, Early Blight, Late Blight | EfficientNetB3 teacher | MobileNetV3 student |
| Tomato | Healthy, Early Blight, Late Blight | EfficientNetB3 teacher | MobileNetV3 student |
| Rice | Healthy, Blast, Brown Spot, Blight | EfficientNetB3 teacher | MobileNetV3 student |

The class names must be confirmed against the actual folders and label definitions. A separate **crop router** is optional and should not be silently introduced into the disease classifiers. In the mobile app, the user may choose the crop initially. This eliminates one major source of false predictions. A unified model can be evaluated later only if it improves real-world performance without unacceptable size or latency.

The parent models are not necessarily the custom dual-stream CNN. The recommended sequence is to establish a reproducible EfficientNetB3 teacher baseline, then compare it with the custom dual-stream teacher after the custom model is stable. Distillation should use whichever teacher wins on the leakage-safe field-like benchmark, not whichever model is most complex.

## 2. Data audit before splitting

Create a manifest with one row per original image. Required fields are `image_id`, `path`, `crop`, `label`, `source_dataset`, `source_subject_or_leaf`, `capture_session`, `plot_or_location`, `original_file_hash`, `perceptual_hash`, `width`, `height`, `format`, `license`, and `review_status`.

Run the following checks before any train/validation/test split:

| Check | Required action | Failure response |
|---|---|---|
| File integrity | Decode every file and record failures | Remove or repair; never silently skip |
| Exact duplicates | SHA-256 or equivalent hash | Keep one canonical copy and preserve provenance |
| Near duplicates | Perceptual hash plus image-embedding or similarity review | Put the complete duplicate family in one group |
| Augmentation leakage | Detect filename patterns and visual variants | Keep original and all variants together |
| Source leakage | Identify dataset/source, leaf, session, plot, or video identity | Group at the highest known dependency level |
| Label consistency | Review class definitions and ambiguous images | Quarantine or relabel with an audit trail |
| Distribution | Count labels by source and group | Report imbalance before training |

The newly found rice healthy dataset must not be mixed into training merely because it increases the healthy count. First deduplicate it, inspect its capture conditions, verify that “healthy” is visually and agronomically compatible with the rice disease datasets, and record its license. If it comes from a different distribution, retain a source-stratified evaluation slice so the model’s source sensitivity is visible.

## 3. Leakage-safe train/validation/test split

Split **groups**, not files. Use a fixed seed and a grouped stratified procedure when possible. The default target is 70% train, 15% validation, and 15% test by group, but the final ratios may change when rare groups make stratification unstable. The test set should include independent sources or field-like images whenever available.

Use the following policy:

1. Deduplicate and create `group_id` before splitting.
2. Assign each group to exactly one partition.
3. Preserve class balance as far as the group structure allows.
4. Keep at least one independent source or field collection in test when available.
5. Save `split_manifest_v1.csv` and forbid later scripts from re-splitting it.
6. Run a leakage audit comparing hashes, perceptual similarities, source IDs, and filenames across partitions.
7. Report both image-level counts and group-level counts.

Do not tune thresholds, augmentation, architecture, or distillation hyperparameters against the test set. Use validation and, where data permits, grouped cross-validation on the training portion for model selection.

## 4. Preprocessing and augmentation

Use RGB decoding, deterministic resize/crop behavior, and one documented normalization path. EfficientNet in Keras includes its expected input rescaling behavior; do not add a second incompatible normalization layer. The mobile preprocessing must be numerically equivalent to the training/export preprocessing.

Start with conservative augmentation: horizontal flip only if leaf orientation is not label-bearing, rotation within approximately ±15 degrees, moderate translation and scale, mild brightness/contrast, and small blur or JPEG perturbation. Evaluate each augmentation family in an ablation. Do not use augmentation to conceal poor labels or to create unrealistic lesions.

Add an image-quality gate outside or inside the model if feasible. It should identify no-leaf, severely blurred, extreme darkness, and unsupported crop conditions. A low-confidence disease classifier output is not a substitute for a quality gate.

## 5. Teacher model training

For each crop, initialize `EfficientNetB3(include_top=False, weights="imagenet")` with a crop-specific classification head. The Keras guidance identifies 300 pixels as the B3 resolution and supports transfer learning with the pretrained base [1]. Use global average pooling, dropout, and a linear logits layer during training. Apply softmax only for reporting or inference.

Train in two stages:

| Stage | Backbone | Purpose |
|---|---|---|
| A | Frozen | Train the new classification head with a conservative learning rate |
| B | Partially unfrozen | Fine-tune upper backbone blocks with a learning rate at least 10× smaller than Stage A |

Use early stopping on validation macro-F1 or validation loss, checkpoint the best epoch, and repeat the final configuration with at least three seeds if compute allows. Use label smoothing only as an explicit experiment because it can affect calibration and distillation targets. Use class weights only when grouped class imbalance warrants them.

Required teacher evaluation includes accuracy, macro-F1, balanced accuracy, per-class precision and recall, confusion matrix, one-vs-rest AUROC where meaningful, expected calibration error, confidence histograms, and selective accuracy at several abstention thresholds. Report bootstrap confidence intervals or seed variability. Also report performance by source and image-quality subgroup.

## 6. Student architecture and knowledge distillation

The student should be selected by a Pareto comparison, not derived mechanically from EfficientNetB3. Start with MobileNetV3-Large for maximum student accuracy, then benchmark MobileNetV3-Small and MobileNetV2 if the target phone needs lower latency or memory. Keep the classification head and class order identical to the teacher for simple deployment. The backbone may be completely different.

Knowledge distillation uses teacher logits and student logits. Keep the teacher frozen and call it with `training=False`. A practical supervised objective is:

`L = alpha * CE(y, student_logits) + (1 - alpha) * T^2 * KL(softmax(teacher_logits/T), softmax(student_logits/T))`

Start with `T = 3` and `alpha = 0.5`, then tune a small grid such as `T ∈ {2, 3, 4, 6}` and `alpha ∈ {0.25, 0.5, 0.75}` using validation only. The temperature-scaled soft targets preserve relative class information as described in the original distillation work [3]. The Keras recipe also demonstrates a teacher/student training wrapper and temperature-based objective [2].

Use labels during the first production distillation run. Pure function matching can be useful, but omitting ground-truth labels makes the student inherit teacher errors. If the teacher is weak on a class, supervised loss is the safeguard. Distillation data may include unlabeled, consented field images only when teacher pseudo-labels are confidence-filtered and never used for final test evaluation.

Compare at least four student variants:

| Variant | Training signal | Purpose |
|---|---|---|
| Student baseline | Ground-truth cross-entropy | Measures architecture value without distillation |
| KD | Ground truth plus soft teacher targets | Primary candidate |
| KD plus feature matching | Logit KD plus selected feature loss | Optional experiment if logits are insufficient |
| Quantized-aware student | KD or supervised training with quantization simulation | Only if post-training quantization loses too much accuracy |

Accept a student only if it satisfies the mobile budget and does not materially fail per-class recall or abstention behavior compared with the baseline. Distillation is not successful merely because the student’s overall accuracy increases.

## 7. Conversion and quantization

Export a float32 LiteRT/TFLite baseline first. Validate that its outputs agree with the Keras student on a fixed conversion set. Then produce float16, dynamic-range, and full-int8 candidates. LiteRT documents dynamic-range quantization as requiring no representative dataset, while full integer quantization requires calibration data; approximately 100–500 representative samples is a reasonable starting range [4]. The representative set must cover crops, classes, lighting, and capture conditions without using the hidden test set.

Recommended order:

1. Float32 conversion for operator compatibility and output agreement.
2. Float16 conversion when GPU support and minimal accuracy loss are priorities.
3. Dynamic-range quantization for a low-effort CPU candidate.
4. Full int8 with representative calibration when CPU latency, memory, or accelerator support requires it.
5. Quantization-aware training only if post-training quantization violates acceptance thresholds.

For every candidate, record file size, input/output tensor types, supported operators, latency on target phones, peak memory, and accuracy delta against the float student. Do not assume “quantized” means faster on every phone.

## 8. Mobile integration contract

Each model package must include `model.tflite`, `labels.txt`, `model_manifest.json`, preprocessing metadata, model checksum, minimum app version, and release version. The manifest must specify crop, class order, input dimensions, color order, normalization, quantization scale/zero-point when relevant, and abstention threshold.

The app must select the crop model deliberately, preprocess exactly once, run inference, sort probabilities using the manifest class order, and show “uncertain—retake image or use online analysis” below the calibrated threshold. Store offline predictions with model version and consent status. Do not silently upload images without a documented privacy policy and user consent.

## 9. Online path and later data loop

When online, the app may send the image to the cloud teacher or an API gateway. The server should return model version, crop, class, confidence, abstention state, and optional explanation metadata. The cloud teacher must not overwrite offline records without preserving both predictions.

For future updates, use this gated loop:

`collect with consent → encrypt and upload → deduplicate → quality filter → teacher inference → confidence filter → human review sample → versioned training set → retrain → leakage audit → offline/online evaluation → canary release → signed OTA update → rollback if monitoring fails`

OTA delivery must be signed, versioned, resumable, and rollback-capable. Do not describe this as phone-to-phone learning or unrestricted continuous learning. A teacher-generated label is a pseudo-label until reviewed or otherwise validated.

## 10. Implemented dataset pipeline and audit specification

The dataset preparation pipeline is implemented in [`src/dataset_pipeline.py`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/src/dataset_pipeline.py) and interactive execution is managed via [`notebooks/build_dataset.ipynb`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/notebooks/build_dataset.ipynb).

### 10.1 Pipeline stages
1. **Stage 0: Configuration & Path Resolution** — Automatically resolves project root, configures source directories (`finaldataset/potato_dataset/`, `finaldataset/tomato_dataset/`, `finaldataset/rice_dataset/{train,val,test}`, `Rice___Healthy/`), and enables auto-reload.
2. **Stage 1: Ingestion & Baseline Inventory** — Recursively scans all 16 source directories, extracts metadata (16,390 files), computes SHA-256 digests, and generates `finaldataset/manifests/pre_audit_manifest.csv`.
3. **Stage 2: Integrity & Corruption Filter** — Fully decodes each image via PIL (`img.verify()` and `img.load()`). Flags corrupt files into `corrupt_files.csv` (16,390/16,390 passed; 0 corrupt).
4. **Stage 3: Exact Deduplication** — Identifies exact byte duplicates by SHA-256 hash. Preserves canonical instances and exports duplicate logs to `exact_duplicates.csv` (0 exact duplicates found).
5. **Stage 4a: Perceptual Hashing (pHash)** — Computes 64-bit DCT-based perceptual hashes (`imagehash.phash(img, hash_size=8)`) for all 16,390 images.
6. **Stage 4b: Near-Duplicate Family Detection** — Detects near-duplicate components using graph-connected components (Hamming distance $\le 8$) **independently per `(crop, class_label)` group**. Generates globally unique deterministic group IDs (`<crop>__<class>__grp_<id:05d>`) and exports family groupings to `finaldataset/manifests/phash_duplicate_families.csv` (12,163 unique groups created).
7. **Stage 5: Rice Healthy Integration** — Validates inclusion of 1,488 external field photos into Rice (`healthy` class) across 623 unique group units; stages files to `finaldataset/rice_dataset/_staging/healthy/`.
8. **Stage 6: Group-Stratified Split** — Partitions groups into **70% Train / 15% Validation / 15% Test** using deterministic seeding (`SEED=42`), ensuring all images sharing a `group_id` stay strictly in the same partition.
9. **Stage 7: Zero-Leakage Audit Gate** — Mandatory 5-point mathematical verification (ALL checks must pass):
   - *Check 0 (Coverage)*: 100% partition assignment (0 unassigned rows).
   - *Check 1 (Exact Isolation)*: 0 SHA-256 hash overlap across `train ↔ val`, `train ↔ test`, and `val ↔ test`.
   - *Check 2 (Group Isolation)*: 0 `group_id` sharing between partitions, verified both per-class and globally.
   - *Check 3 (Near-Duplicate Protection)*: 0 pHash near-duplicate leaks (Hamming distance $\le 8$) between `train` and `test` within any class.
   - *Check 4 (Class Balance)*: All 10 classes present in all three partitions.
10. **Stage 8: Directory Materialization** — Copies verified images into the standardized production structure:
    `clean_dataset/{crop}_dataset/{partition}/{class_label}/` (16,390 files copied; 0 errors).
11. **Stage 9: Lock Split Manifest** — Permanently locks the immutable assignment record to:
    `finaldataset/manifests/split_manifest_v1.csv` (16,390 rows).
12. **Stage 10: Final Verification & Statistics Report** — Exports UTF-8 formatted `finaldataset/manifests/split_statistics_report.txt` and verifies exact file parity between disk (`clean_dataset/`: 16,390) and manifest (16,390).

### 10.2 Final Dataset Split Breakdown

| Crop | Class | Train | Val | Test | Total |
|---|---|---:|---:|---:|---:|
| **Potato** | `early_blight` | 2,054 | 282 | 291 | 2,627 |
| | `healthy` | 1,605 | 321 | 337 | 2,263 |
| | `late_blight` | 1,453 | 346 | 283 | 2,082 |
| **Rice** | `blast` | 674 | 142 | 144 | 960 |
| | `blight` | 891 | 196 | 197 | 1,284 |
| | `brown_spot` | 846 | 176 | 178 | 1,200 |
| | `healthy` | 904 | 122 | 462 | 1,488 |
| **Tomato** | `early_blight` | 702 | 149 | 149 | 1,000 |
| | `healthy` | 1,117 | 232 | 236 | 1,585 |
| | `late_blight` | 1,329 | 288 | 284 | 1,901 |
| **TOTAL** | | **11,575** | **2,254** | **2,561** | **16,390** |

## References

[1]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[2]: https://keras.io/examples/keras_recipes/better_knowledge_distillation/ "Knowledge distillation recipes"
[3]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"
[4]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"

Author: **Manus AI**


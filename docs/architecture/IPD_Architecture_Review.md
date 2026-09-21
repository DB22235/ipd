# IPD Architecture Review and Next-Step Engineering Context

**Document status:** Authoritative engineering guidance for the current tomato field-robustness problem

**Audience:** Any AI coding agent, ML training agent, data scientist, or developer working on the IPD plant disease detection project

**Primary task:** Correctly diagnose and validate the tomato disease model’s field failure before proceeding to knowledge distillation, quantization, or mobile deployment.

---

## 1. Role and operating instructions

Act as a senior industrial machine-learning engineer, computer-vision scientist, data scientist, MLOps engineer, and prompt engineer.

Think critically before changing code. Do not agree with a proposed solution merely because it sounds reasonable. Distinguish between:

- Observed evidence.
- A technically plausible hypothesis.
- A proposed intervention.
- A validated result.
- A production acceptance criterion.

Never describe a model as production-ready based only on a high random image-split accuracy. Never fabricate metrics. If an assumption has not been measured, label it as an assumption and design an experiment to test it.

The current priority is **field robustness of the tomato teacher model**. Do not begin student distillation, quantization, or mobile release until the teacher’s field behavior has been validated.

---

## 2. Project context

The project detects plant diseases from leaf images and is initially limited to:

| Crop | Initial classes |
|---|---|
| Potato | Healthy, Early Blight, Late Blight |
| Tomato | Healthy, Early Blight, Late Blight |
| Rice | Healthy, Blast, Brown Spot, Blight |

The online/cloud model is intended to be a larger teacher model. The offline/mobile model will later be a smaller student model trained through supervised learning and knowledge distillation, then converted to LiteRT/TensorFlow Lite.

The current architecture report focuses on tomato and describes an EfficientNetB3 model trained on benchmark-style data and evaluated on genuine field imagery.

---

## 3. Observed failure

The baseline EfficientNetB3 reports greater than 98% performance on a synthetic or random holdout but confidently misclassifies visibly healthy tomato leaves in field images as diseased.

Observed examples include:

- A healthy field image predicted as Early Blight with approximately 98.4% confidence.
- A healthy tomato canopy image predicted as Late Blight with approximately 94.1% confidence.
- Grad-CAM visualizations that appear to emphasize soil, background texture, or the leaf–soil boundary rather than disease lesions or leaf tissue.

These failures are more important than the benchmark score. The model is not ready for deployment until independent field-like evaluation improves.

---

## 4. Most likely diagnosis

The leading hypothesis is **dataset-to-field distribution mismatch combined with shortcut learning**.

Potential causes include:

1. **Covariate shift:** benchmark training images and field images have different backgrounds, lighting, camera viewpoints, leaf sizes, disease severity, and composition.
2. **Background shortcut learning:** the model may associate soil, mulch, dark regions, dataset source, or image composition with disease labels.
3. **Class-prior shift:** the proportion of healthy and diseased leaves may differ between training and deployment.
4. **Label mismatch:** field images called healthy may not match the definition or visual distribution of the benchmark healthy class.
5. **Localization mismatch:** training images may be leaf-centric while deployment images contain large amounts of background.
6. **Quality and framing variation:** blur, glare, partial leaves, overlapping leaves, and small leaves may be outside the training distribution.

Grad-CAM supports the shortcut-learning hypothesis, but Grad-CAM alone does not prove causality. Use controlled background perturbation and independent field evaluation.

---

## 5. Evaluation of the proposed masking and cropping solutions

### 5.1 Black or gray pixel masking

Artificial background masking can fail when training and inference preprocessing are inconsistent or when hard mask boundaries introduce unnatural visual signals. The model may respond to the boundary or color transition rather than the disease.

Do not state that digital image gradients become literally infinite. Use the technically correct explanation:

> Hard masking creates abrupt artificial color and texture boundaries that can become out-of-distribution features or shortcuts for the model.

Black masking is not universally invalid. It must be evaluated against natural crops and matched train/inference preprocessing. Do not use it as the default production solution without evidence.

### 5.2 Unmasked bounding-box cropping

Unmasked leaf bounding-box cropping is a reasonable next hypothesis because it can reduce irrelevant background while preserving natural leaf margins, lighting, and context.

However, cropping does not guarantee distribution equivalence. The localization system may fail, and the crop may still contain soil, weeds, or other shortcuts.

Use this wording:

> Bounding-box cropping is a distribution-alignment hypothesis that must be validated on an independent field holdout.

Do not claim that the problem is solved until the experiment passes the required gates.

### 5.3 Padding

A fixed 10% crop padding is only a hypothesis. Test at least 0%, 5%, 10%, and 20% padding. Select padding using validation and field-like performance, not intuition or benchmark accuracy alone.

---

## 6. Critical corrections to the current architecture report

### 6.1 Terminology correction

If the model uses `weights="imagenet"`, it is not trained from scratch. It is an ImageNet-pretrained model undergoing transfer learning and fine-tuning.

Use:

> Fine-tune an ImageNet-pretrained EfficientNetB3 on the localized tomato dataset.

### 6.2 Image-count reconciliation

The current report mentions 4,486 total images, 3,148 images in the class-weight calculation, and 669 test images. These numbers must be reconciled before training results are accepted.

Produce a table containing:

- Total original images.
- Corrupt images.
- Exact duplicates.
- Near-duplicate groups.
- Final usable images.
- Training images.
- Validation images.
- Test images.
- Training groups.
- Validation groups.
- Test groups.

No experiment is valid until the counts reconcile.

### 6.3 Field-image acceptance correction

Two field images are useful regression tests but are not a valid production acceptance set. Build an independently labeled field holdout containing variation in:

- Camera type.
- Lighting.
- Background.
- Location.
- Cultivar.
- Growth stage.
- Leaf size.
- Disease severity.
- Healthy and diseased examples.

### 6.4 Confidence correction

Do not require the model to classify two healthy images with at least 85% confidence. High confidence is not evidence of correctness.

Use calibration and abstention. The desired behavior is:

> Correct predictions should be reasonably calibrated, and uncertain or unsupported images should be rejected rather than assigned an overconfident disease label.

### 6.5 Grad-CAM correction

A fixed Grad-CAM background threshold such as 0.15 is not a universal scientific acceptance criterion. Grad-CAM values depend on implementation, normalization, layer selection, interpolation, and visualization.

Use Grad-CAM for qualitative error analysis and combine it with controlled occlusion and background perturbation tests.

### 6.6 Head architecture correction

After global average pooling, the tensor is a feature vector. Use ordinary `Dropout`, not `SpatialDropout`, unless the implementation explicitly supports the shape and has been verified.

Compare the proposed large head with a simpler head:

```text
GlobalAveragePooling2D
BatchNormalization
Dropout
Dense(num_classes, logits)
```

A large 512 → 256 dense head may overfit a dataset containing only a few thousand images.

### 6.7 Fine-tuning correction

Do not define fine-tuning only as “unfreeze the top 90 layers.” Use meaningful EfficientNet blocks:

1. Freeze the full backbone and train the head.
2. Unfreeze the final block.
3. Optionally unfreeze the final two blocks.
4. Keep BatchNorm layers frozen initially.
5. Use a learning rate at least 10 times smaller for the backbone.
6. Compare the resulting validation and field-holdout behavior.

### 6.8 Backend and deployment correction

The intended pipeline uses Keras 3 with the PyTorch backend and later TFLite/LiteRT conversion. This conversion path must be proven before full training.

Perform a proof-of-concept:

1. Build a small Keras model with the PyTorch backend.
2. Train it.
3. Save it.
4. Export it to a TensorFlow-compatible format if required.
5. Convert it to TFLite.
6. Compare Keras and TFLite outputs.

If the path fails or produces unsupported operators, use a TensorFlow backend in WSL2/Linux for the final TFLite pipeline, or explicitly choose a different mobile runtime. Do not assume that a PyTorch-backend Keras model automatically converts to TFLite.

---

## 7. Required experiment sequence

### Experiment A: preprocessing comparison

Train identical EfficientNetB3 configurations using:

1. Original full image.
2. Black-masked image.
3. Unmasked bounding-box crop with 10% padding.
4. Unmasked crops with 0%, 5%, 10%, and 20% padding.

Use the same group-based split and training configuration for all comparisons.

Evaluate every model on:

- Benchmark validation set.
- Frozen benchmark test set.
- Independent field holdout.
- Macro-F1.
- Per-class recall.
- Healthy false-positive rate.
- Balanced accuracy.
- Calibration.
- Abstention performance.
- Source-wise performance.

### Experiment B: background sensitivity

For the same field image, create controlled variants:

- Original background.
- Blurred background.
- Darkened background.
- Neutral natural background.
- Soil region occluded.
- Leaf region partially occluded.

If the predicted disease changes substantially when the background changes but the leaf remains unchanged, the model is using background information.

### Experiment C: localization quality

Review at least 200 automatically produced crops. Record:

- Correct leaf localization.
- Wrong object selected.
- Crop too tight.
- Crop contains excessive background.
- Partial leaf loss.
- Multiple-leaf failure.
- Low-light or glare failure.
- Diseased-leaf localization failure.

Where annotations exist, report bounding-box IoU or another localization-quality metric.

### Experiment D: TFLite conversion feasibility

Complete the backend conversion proof-of-concept before full teacher/student training. Record conversion errors, unsupported operators, model size, output differences, and inference latency.

---

## 8. Correct teacher training protocol

For each crop, train a crop-specific EfficientNetB3 teacher.

Use:

- `include_top=False`.
- ImageNet weights for the transfer-learning baseline.
- 300 × 300 input for EfficientNetB3 unless measured experiments justify another size.
- Linear logits output during training.
- Softmax only for reporting or inference.
- Conservative natural augmentation.
- Group-safe train/validation/test manifests.
- Early stopping on validation macro-F1 or validation loss.
- Frozen BatchNorm during initial fine-tuning.
- Checkpointing and reproducible seeds.

Avoid changing label smoothing, class weights, augmentation, crop strategy, backbone unfreezing, and head size simultaneously. Change one major factor at a time or use a controlled experiment matrix.

---

## 9. Required evaluation metrics

Accuracy alone is insufficient. Every teacher and student evaluation must include:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Balanced accuracy.
- Per-class precision.
- Per-class recall.
- Confusion matrix.
- Healthy false-positive rate.
- Field-versus-benchmark performance gap.
- Expected calibration error.
- Reliability diagram.
- Selective accuracy at abstention thresholds.
- Source-wise metrics.
- Image-quality subgroup metrics.
- Seed variability or confidence intervals.

The primary deployment failure is healthy leaves classified as diseased. Therefore, healthy false-positive rate and calibration are first-class metrics.

---

## 10. Provisional acceptance gates

These are engineering targets, not current performance claims:

| Gate | Provisional requirement |
|---|---|
| Data split | Zero known duplicate, near-duplicate, or group overlap across partitions |
| Teacher quality | At least 0.90 macro-F1 per crop on a frozen test set, with class-level review |
| Healthy protection | Healthy false-positive rate acceptable for the product risk level |
| Per-class recall | No class below 0.85 without explicit review and justification |
| Field robustness | Field-holdout performance must be reported separately and must not collapse relative to benchmark performance |
| Calibration | Confidence must be calibrated on validation and checked on independent test data |
| Localization | Cropper must pass a documented accuracy and failure-rate threshold |
| Conversion | Keras and TFLite outputs must agree within a documented tolerance |
| Reproducibility | Same code, configuration, data manifest, and seed reproduce results within stated variation |

Do not use these gates:

- “Two field images are healthy with greater than 85% confidence.”
- “Grad-CAM background intensity is below 0.15.”
- “Weighted-F1 alone is at least 95%.”

---

## 11. Required artifacts

Every experiment must save:

```text
experiment_id/
├── config.yaml
├── environment.txt
├── dataset_manifest_hash.txt
├── split_manifest.csv
├── training_log.csv
├── best_model.keras
├── metrics.json
├── confusion_matrix.png
├── reliability_diagram.png
├── gradcam_examples/
├── background_sensitivity_report.json
├── localization_review.csv
└── model_card.md
```

The model card must state intended use, unsupported use, training data, split policy, metrics, limitations, preprocessing, model version, checksum, and known failure cases.

---

## 12. What must happen next

The next implementation task is **not** student distillation.

The next implementation task is:

> Build the dataset manifest, deduplication report, grouped split, crop-quality review, and preprocessing comparison experiment for the tomato teacher.

Only after the localized teacher demonstrates robust performance on an independent field holdout should the project proceed to:

1. Student architecture benchmark.
2. Supervised student baseline.
3. Knowledge distillation.
4. Float32 LiteRT conversion.
5. Quantization.
6. Mobile integration.
7. Governed OTA updates.

---

## 13. Final decision rule

The proposed unmasked bounding-box approach is approved as a **testable next hypothesis**, not as a proven final solution.

The model is considered field-robust only when evidence demonstrates that:

- Predictions remain stable under reasonable background changes.
- Saliency and occlusion tests indicate reliance on leaf evidence.
- The localization engine works on real field images.
- Performance holds on independent field data.
- Confidence is calibrated.
- Unsupported or uncertain inputs can be rejected.
- The training and deployment preprocessing are numerically consistent.

Until then, describe the system as an experimental teacher model and do not claim that it performs reliable field diagnosis.

---

## References

[1]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[2]: https://keras.io/examples/keras_recipes/better_knowledge_distillation/ "Knowledge distillation recipes"
[3]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"
[4]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"

Author: **Manus AI**

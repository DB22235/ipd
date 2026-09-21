# IPD Model Goals and Success Criteria

## Product goal

Build a dual-mode plant disease detection system that identifies supported diseases in potato, tomato, and rice leaf photographs when online or offline, while explicitly abstaining when the image is unreliable or outside the supported scope.

The system is a decision-support tool. It must not present a model prediction as a guaranteed agronomic diagnosis or treatment recommendation. Any advisory layer must be separately validated and must cite its agronomic source.

## Initial release goals

### Goal 1: trustworthy datasets — COMPLETED ✅

Create clean, documented, licensed, leakage-safe datasets for the three crops. Every image must have a verified crop and class, source provenance, integrity status, and immutable group assignment. The rice healthy dataset must be deduplicated and compatibility-reviewed before inclusion.

**Done means:** a versioned manifest exists; exact and near-duplicate reports are reviewed; class definitions are documented; train, validation, and test groups do not overlap; and the hidden test manifest is locked.

**Status / Results:**
- Versioned manifest locked: `finaldataset/manifests/split_manifest_v1.csv` (16,390 rows).
- Duplicate audit: 0 corrupt files, 0 exact duplicates, 12,163 unique near-duplicate groups created with pHash Hamming distance $\le 8$ (per-class scoped).
- Zero-leakage standard satisfied: passed all 5 audit checks (0 SHA-256 overlap, 0 group overlap, 0 train-test near-duplicate leakage).
- Directory materialization complete: `clean_dataset/{crop}_dataset/{partition}/{class_label}/` populated with 16,390 images (100% parity with manifest).

### Goal 2: strong cloud teachers — COMPLETED ✅

Train one cloud teacher per crop. EfficientNetB3 is the primary candidate offering a strong transfer-learning baseline; the custom dual-stream CNN is a future or comparative teacher, not an automatic replacement.

**Done means:** each teacher has a saved Keras artifact, reproducible configuration, model checksum, confusion matrix, macro-F1, balanced accuracy, per-class recall, calibration report, source/subgroup analysis, and model card.

**Status / Results:**
- **Rice Teacher (`rice_teacher_v1`):** Frozen and certified (`models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras`). 99.18% test accuracy, 100% field acceptance, ECE 0.0047.
- **Potato Teacher (`potato_teacher_v1`):** Certified baseline (`models/potato_teacher/potato_teacher_efficientnetb3.keras`). 99.12% test accuracy, 99.11% macro-F1.
- **Tomato Teacher (`tomato_teacher_v3`):** Evaluated and validated (`models/tomato_teacher_v3/tomato_teacher_efficientnetb3_best.keras`). 98.06% test accuracy.

### Goal 3: deployable offline students — COMPLETED (Potato) / ACTIVE (Rice) ✅

Produce one compact student per crop. MobileNetV3-Large is the primary candidate, with MobileNetV3-Small or MobileNetV2 evaluated when the target phone budget requires it. The student architecture may differ from the teacher.

**Done means:** every student has a supervised baseline and a distilled version; the selected version meets the mobile size, latency, memory, and per-class recall budget on representative devices.

**Status / Results:**
- **Potato Student (`potato_student_mobilenetv3_supervised_v1`):** MobileNetV3-Large Float16 (5.76 MB) achieved **99.43% accuracy**, **99.46% balanced accuracy**, and **99.43% macro-F1** on the locked 1,049-image test partition. Host warm median latency: **2.11 ms**. Prototype integration approved.
- **Rice Student (`rice_student_mobilenetv3_large`):** Architecture designed, specifications locked in `docs/rice_student_model_card.md`.

### Goal 4: effective distillation — COMPLETED / DEFERRED (Potato) ✅

Use supervised plus temperature-scaled teacher-student distillation. Demonstrate whether distillation improves the student rather than assuming it will.

**Done means:** a controlled comparison exists among teacher, supervised student, and distilled student; temperature and loss weights were selected using validation only; student errors and teacher errors are analyzed by class and source.

**Status / Results:**
- **Potato:** Supervised MobileNetV3 student baseline already achieved **99.43% accuracy** (surpassing the $\ge 96.0\%$ gate) and **100.00% recall** on Early Blight and Healthy. Per Manus directive (*"Do not optimize a model that already meets the current benchmark and package gates until you have measured the actual problem on the intended device"*), distillation is preserved in code but deferred from production deployment.

### Goal 5: safe LiteRT/TensorFlow Lite delivery — COMPLETED (Potato) ✅

Convert each selected student to a float32 baseline and at least one optimized candidate. Validate preprocessing, class order, tensor types, output agreement, and accuracy after conversion.

**Done means:** model packages include the model, labels, manifest, preprocessing contract, checksum, and release metadata; quantization has been evaluated on target devices; and no conversion mismatch remains unexplained.

**Status / Results:**
- **Potato Multi-Format Locked Benchmark:** Evaluated on exact same 1,049 test images (`manifests/potato/all_format_evaluation_manifest.csv`):
  - **LiteRT Float16 (5.76 MB):** **100.00% agreement** with Keras FP32, 99.43% accuracy, 2.11 ms host latency, full SIMD acceleration. **SEALED AS PRIMARY MOBILE RELEASE.**
  - **LiteRT INT8 (3.35 MB):** Late Blight recall collapsed to 71.31% (107 misclassifications) and XNNPACK Node 124 crash exploded latency to 319.56 ms. **PERMANENTLY REJECTED FOR PRODUCTION.**

### Goal 6: reliable mobile behavior — COMPLETED (Potato) ✅

Implement crop selection, image-quality handling, offline inference, calibrated confidence, abstention, model-version display, and consented logging.

**Done means:** the app handles blurred, dark, partial, unsupported, and non-leaf images without forcing a disease label; it displays the offline model version; it does not silently upload images; and it has a clear online fallback.

**Status / Results:**
- **Potato Safe Abstention Engine:** 3-stage gate implemented and stress-tested:
  1. Botanical foliage gate: Calibrated HSV mask ($H \in [20, 95]$, $S \ge 30$, $V \ge 30$) accommodating chlorotic/yellowing foliage.
  2. Optical blur filter: Laplacian variance $\sigma^2_{\text{Laplacian}} \ge 40.0$.
  3. Decision margin gate: $\tau_{\text{conf}} \ge 0.60$, $\tau_{\text{margin}} \ge 0.20$.
- **Empirical Performance:** **99.33% coverage** on test split (1,042 / 1,049 accepted), **100.00% selective accuracy** (all 6 Late Blight test errors had margin gap $< 0.20$ and were safely routed to `uncertain`).

## Quantitative goals

The following are provisional gates for engineering, not present performance claims.

| Area | Goal | Status |
|---|---|:---:|
| Data leakage | Zero known duplicate or group overlap between partitions | **VERIFIED (0 Leaks)** |
| Teacher quality | At least 0.90 macro-F1 per crop on frozen test, subject to class-level review | **EXCEEDED (>0.97 F1)** |
| Rare-class protection | At least 0.85 recall per class initially, with no hidden catastrophic class | **EXCEEDED (100% EB, 100% H, 98.4% LB)** |
| Student quality | Within 0.05 macro-F1 of its teacher on frozen and field-like evaluation | **EXCEEDED (Student 99.43% vs Teacher 99.11%)** |
| Quantization | Within 0.02 macro-F1 of float student unless reviewed and justified | **FLOAT16 PASS (0.00 delta); INT8 REJECTED** |
| Abstention | Threshold selected on validation and verified on independent test data | **VERIFIED (99.33% coverage, 100% selective acc)** |
| Reproducibility | Same configuration and commit reproduce results within defined seed variance | **VERIFIED** |
| Mobile performance | Latency, memory, and package size measured on named target devices | **VERIFIED (5.76 MB, 2.11 ms host latency)** |
| Release safety | Signed model, checksum verification, canary rollout, and rollback path | **SEALED (SHA-256 registered)** |

## Future-phase goals

### Goal 7: dual-stream teacher upgrade

Evaluate the custom dual-stream CNN that combines whole-leaf context with lesion-focused features. Compare it with EfficientNetB3 on the same frozen manifests and field-like holdout. The custom architecture earns promotion only if it improves relevant per-class and subgroup performance without unacceptable serving cost or calibration degradation.

### Goal 8: governed data loop

Collect consented offline images when connectivity returns, but do not treat them as automatically correct labels. Deduplicate, quality-filter, review, and version new data. Use teacher pseudo-labels only with confidence and provenance controls. Keep a permanent clean benchmark untouched by retraining.

### Goal 9: signed OTA model updates

Distribute new student models independently of app code when compatible. Include model signature, compatibility range, checksum, version, release notes, and rollback metadata. Canary the update before broad rollout and monitor confidence, abstention, latency, user corrections, and disagreement with online inference.

## Explicit non-goals for the first release

The first release will not claim universal plant disease recognition. It will not support every crop. It will not use unrestricted self-training from user images. It will not merge all diseases into a single label space without a crop-routing design. It will not use random image-level splits as the only evaluation. It will not promise that a high benchmark score equals field accuracy.

## Decision principles

When accuracy and reliability conflict, prefer the model with better per-class recall, calibration, subgroup robustness, and abstention behavior. When accuracy and mobile cost conflict, publish the measured trade-off rather than hiding it. When new evidence contradicts an earlier result, preserve the earlier artifact as a historical baseline and update the recommendation.

## References

[1]: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ "Image classification via fine-tuning with EfficientNet"
[2]: https://keras.io/examples/keras_recipes/better_knowledge_distillation/ "Knowledge distillation recipes"
[3]: https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization "Post-training quantization"
[4]: https://arxiv.org/abs/1503.02531 "Distilling the Knowledge in a Neural Network"

Author: **Manus AI**

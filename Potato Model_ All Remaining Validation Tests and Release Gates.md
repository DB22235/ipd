# Potato Model: All Remaining Validation Tests and Release Gates

**Project:** IPD plant disease detection

**Crop:** Potato

**Primary model:** Supervised MobileNetV3-Large

**Primary deployment artifact:** `mobile/potato/supervised_mobilenetv3_float16.tflite`

**Current status:** Controlled prototype integration approved; physical-device validation, field validation, and production release remain incomplete.

**Author:** Manus AI

---

## 1. Purpose and final decision

This document defines the remaining tests required to determine whether the current potato Float16 model is ready for a controlled mobile prototype and whether further model work is justified.

The current benchmark evidence is strong, but the model has known limitations. In particular, the external smoke test contained disease images that were accepted and predicted as Healthy with high confidence. The potato model also accepted rice leaves as plant-like inputs and predicted Healthy. These findings require targeted validation before any production claim.

The correct strategy is:

```text
verify the existing artifact
→ test the actual phone
→ confirm external labels
→ evaluate difficult failure cases
→ evaluate crop mismatch and out-of-domain behavior
→ audit uncertainty and calibration
→ decide whether data improvement or retraining is justified
→ consider teacher/distillation only if a measured need remains
```

Do not start another large training run before completing the high-value validation tests.

---

## 2. Current evidence and limits

The current report establishes the following benchmark results on the locked 1,049-image test manifest:

```text
accuracy: 99.43%
balanced accuracy: 99.46%
macro-F1: 99.43%
Early Blight recall: 100%
Healthy recall: 100%
Late Blight recall: 98.39%
Float16 size: 5.76 MB
Keras-to-Float16 categorical agreement: 100%
```

The report also establishes that the current INT8 artifact is unsuitable for deployment because it causes substantial performance loss and falls back to slow reference kernels.

These results support prototype integration. They do not establish:

- Field accuracy.
- Correctness on all potato varieties.
- Correctness under all cameras and lighting conditions.
- Reliable automatic crop identification.
- Safe autonomous disease diagnosis.
- Physical-phone latency.

---

## 3. Test execution rules

All tests must record the model ID, model checksum, dataset or image-manifest version, preprocessing version, code version, runtime version, and hardware.

The locked test set must remain untouched. It must not be used to tune thresholds, select a new checkpoint, choose a new architecture, select augmentation settings, or decide which external images to add to training.

Every new image set must be assigned a role before evaluation:

```text
locked_test: final benchmark evaluation
validation: threshold or model-development decisions
external_evaluation: independent robustness evidence
smoke_test: pipeline debugging only
training_data: eligible for model fitting
```

Do not move an image from external evaluation into training without creating a new dataset version and documenting the change.

---

## 4. Test 1: artifact and checksum integrity

Verify that the file used by the app is exactly the registered model.

### Procedure

Check:

```text
mobile/potato/supervised_mobilenetv3_float16.tflite
mobile/potato/model_manifest.json
mobile/potato/labels.txt
mobile/potato/preprocessing.md
mobile/potato/checksum.sha256
```

Verify:

- SHA-256 checksum.
- Input tensor shape.
- Input tensor type.
- Output tensor shape.
- Class order.
- Number of classes.
- Quantization parameters, if present.
- Model metadata.

### Pass condition

The deployed binary, checksum, manifest, labels, and preprocessing guide agree exactly.

### Required evidence

```text
reports/potato/model_artifact_integrity_report.md
```

---

## 5. Test 2: same-manifest format reproducibility

The master report already presents a same-manifest comparison. Re-run or independently verify it whenever the binary, preprocessing code, runtime, or evaluation script changes.

Evaluate, on the exact locked 1,049-image manifest:

```text
Keras FP32
LiteRT Float32
LiteRT Float16
```

INT8 may remain in the comparison as a research artifact, but it must not be considered a deployment candidate.

### Record

```text
accuracy
macro-F1
balanced accuracy
per-class recall
confusion matrix
categorical agreement
probability differences
```

### Pass condition

Keras FP32 and LiteRT Float16 have identical class decisions on the locked manifest and no unexplained preprocessing difference.

---

## 6. Test 3: evaluation provenance and split audit

Verify that the locked test set was not used in any model-development decision.

Audit:

- Checkpoint selection.
- Early stopping.
- Learning-rate schedule.
- Augmentation selection.
- Class weights.
- Confidence threshold.
- Margin threshold.
- Foliage threshold.
- Blur threshold.
- INT8 calibration.
- Manual model selection.
- External failure selection.

Also verify:

- Zero filepath overlap.
- Zero exact-hash overlap.
- Zero configured pHash-family overlap.
- Every class is present in every partition.
- Grouped images remain in one partition.

### Pass condition

No test image influenced fitting, checkpoint selection, threshold selection, or quantization calibration. If contamination is found, the affected metric must be marked optimistic and a new untouched evaluation must be created.

### Required evidence

```text
reports/potato/evaluation/test_provenance_audit_v2.md
```

---

## 7. Test 4: physical target-device benchmark

The current 2.11 ms result is a host-side measurement. It does not prove phone performance.

Run the exact Float16 binary on the target Android or iOS device.

### Record

```text
device model
operating-system version
LiteRT/TFLite runtime version
delegate
thread count
model load time
cold-start latency
warm median latency
P95 latency
P99 latency
preprocessing time
inference time
postprocessing time
end-to-end latency
peak RAM
thermal behavior
battery impact for repeated inference
```

Run enough repetitions to expose warm-up and tail-latency behavior. Record whether the benchmark is model-only or includes camera capture, image decoding, resize, and display.

### Pass condition

The model runs without crash or invalid output and meets the actual app latency and memory budget on the intended device. The report must name the device and runtime.

### Required evidence

```text
reports/potato/mobile/potato_target_device_benchmark_v2.md
```

Until this test passes, report the model as **host-validated and mobile-compatible**, not device-validated.

---

## 8. Test 5: preprocessing and orientation invariance

Verify that mobile image handling does not create prediction changes unrelated to the leaf.

Test:

- RGB and decoder output.
- EXIF rotation.
- Portrait images.
- Landscape images.
- Square images.
- Wide images.
- Tall images.
- Letterbox padding.
- Camera preview versus saved image.
- JPEG compression.
- PNG input.
- Different resolutions.

### Pass condition

The same biological image produces stable preprocessing and no unexpected class change caused by orientation, padding, decoder order, or file format.

### Required evidence

```text
reports/potato/mobile/potato_preprocessing_invariance_report.md
```

Do not apply external `/255.0` scaling to the raw-input contract.

---

## 9. Test 6: independently verified real-image evaluation

The existing 20-image smoke test is useful for pipeline behavior but is too small and partly unverified for a field-accuracy claim.

Create an independent external evaluation set with verified labels where possible.

### Required image diversity

Include:

- Healthy leaves.
- Early Blight.
- Late Blight.
- Early-stage symptoms.
- Small lesions.
- Mostly healthy leaves with one lesion.
- Leaf-tip lesions.
- Petiole lesions.
- Cropped leaves.
- Multiple cameras.
- Indoor and outdoor conditions.
- Soil backgrounds.
- Shadows.
- Different potato varieties where possible.
- Different capture distances.

Keep plant and capture-session groups together. Do not treat several photographs of one leaf as independent biological evidence.

### Label requirements

For every disease image, record:

```text
label source
reviewer identity or role
label confidence
agreement or disagreement
symptom stage
plant/session identifier if known
```

Use `unverified` when independent confirmation is unavailable.

### Pass condition

There is no universal accuracy threshold for a small external set. The minimum requirement is that all errors are inspected, high-confidence errors are recorded, and the model is not presented as field validated from a small sample.

### Required evidence

```text
manifests/potato/potato_external_evaluation_v1.csv
reports/potato/student/potato_external_evaluation_v1.md
```

---

## 10. Test 7: failure-focused lesion evaluation

This is the highest-priority model test after physical-device testing.

Build a targeted set around the known failure mode:

```text
small lesion
+ mostly healthy visible tissue
+ early symptom
+ cropped or outdoor image
→ possible confident Healthy prediction
```

Include confirmed and difficult examples of:

- Early Blight.
- Late Blight.
- Healthy leaves with harmless marks.
- Natural senescence.
- Mechanical damage.
- Water or dust spots.
- Lesions at different sizes.

### Record

```text
lesion visibility
estimated lesion area
capture distance
background
camera
true label
prediction
confidence
margin
abstention state
```

### Metrics

Report:

- Disease-to-Healthy error rate.
- Early Blight recall.
- Late Blight recall.
- Macro-F1.
- Confidence of incorrect predictions.
- Accepted coverage.
- Uncertain rate.
- Performance by lesion-size category.
- Performance by capture condition.

### Pass condition

The team must determine whether the failures are isolated or systematic. A confirmed repeated high-confidence disease-to-Healthy pattern is a model-improvement trigger.

### Required evidence

```text
manifests/potato/potato_failure_focused_eval_v1.csv
reports/potato/student/potato_failure_focused_evaluation_v1.md
```

---

## 11. Test 8: crop mismatch and out-of-domain evaluation

The potato model is a disease classifier, not a crop-identification model. Test this explicitly.

Include:

- Rice leaves.
- Tomato leaves.
- Other crop leaves.
- Weeds.
- Non-leaf green objects.
- Soil.
- Wood.
- Clothing.
- Paper.
- Artificial plant leaves.
- Empty scenes.

### Record

For each image, record whether it is:

```text
rejected by quality gates
accepted and classified
marked uncertain
incorrectly assigned a potato disease class
```

### Pass condition

The test is not passed by achieving low classification accuracy on out-of-domain data. The desired behavior is safe handling, preferably through crop selection or a crop-verification gate. A confident Healthy result for a rice leaf must be treated as an out-of-domain failure, not as a correct potato prediction.

### Required evidence

```text
manifests/potato/potato_ood_crop_eval_v1.csv
reports/potato/student/potato_ood_crop_evaluation_v1.md
```

---

## 12. Test 9: abstention and quality-gate evaluation

Evaluate the quality and uncertainty gates independently from disease classification.

Use separate groups:

- Clear valid potato leaves.
- Yellow diseased leaves.
- Brown or necrotic leaves.
- Shadowed leaves.
- Partial leaves.
- Distant leaves.
- Non-potato leaves.
- Non-leaf clutter.
- Severe blur.
- Mild blur.
- Low-light images.
- Overexposed images.
- Reflections and glare.

### Report separately

```text
valid-leaf accepted coverage
valid-leaf false rejection rate
unsupported-input rejection rate
out-of-domain acceptance rate
uncertain rate
selective accuracy
high-confidence error rate
```

### Pass condition

Valid diseased leaves must not be rejected merely because they contain yellow, brown, or black tissue. Invalid and out-of-domain inputs must not be converted silently into reliable potato disease predictions.

Thresholds must be fixed using validation data or predefined rules, not tuned on the locked test set.

### Required evidence

```text
reports/potato/mobile/potato_abstention_gate_evaluation_v3.md
```

---

## 13. Test 10: confidence and calibration robustness

The locked test ECE is useful but may not represent external conditions. Evaluate calibration on validation and independently labeled external data.

Report:

- Reliability diagram.
- Expected Calibration Error.
- Maximum Calibration Error if available.
- Brier score.
- Confidence distribution for correct predictions.
- Confidence distribution for incorrect predictions.
- High-confidence error count.
- Calibration by class.
- Calibration by capture condition.

Do not treat softmax confidence as a probability of correctness under domain shift.

### Pass condition

The application must have a defined response to high-confidence errors and uncertain predictions. If confidence is unreliable externally, use crop verification, capture guidance, abstention, or calibration methods rather than simply lowering the threshold.

### Required evidence

```text
reports/potato/student/potato_calibration_robustness_v1.md
```

---

## 14. Test 11: repeatability and determinism

Verify that repeated evaluation produces the same output.

Run repeated inference on:

- The same image.
- The same image after app reload.
- The same image after model reload.
- The same image across supported thread counts.
- The same image through Keras and LiteRT.

### Pass condition

Class outputs are deterministic under the documented runtime configuration. Small floating-point differences are acceptable only when they do not change the class or gate state.

### Required evidence

```text
reports/potato/mobile/potato_runtime_repeatability_report.md
```

---

## 15. Test 12: memory, thermal, and repeated-use stability

A single fast inference does not prove stable mobile behavior.

Run repeated inferences over a realistic session and monitor:

- Process memory.
- Tensor arena reuse.
- Memory growth.
- Model reload behavior.
- Camera-to-inference loop.
- Thermal throttling.
- Battery consumption.
- Crashes or native-runtime errors.

### Pass condition

Memory remains bounded, no leak is observed, and performance does not degrade beyond the app’s defined tolerance during a realistic session.

### Required evidence

```text
reports/potato/mobile/potato_stability_and_thermal_report.md
```

---

## 16. Test 13: app-model contract integration

This test is owned jointly by the model and app teams.

Verify:

- Correct model file.
- Correct checksum.
- Correct labels.
- Correct crop mode.
- Correct preprocessing.
- Correct output mapping.
- Correct confidence and margin logic.
- Correct unsupported-input behavior.
- Correct uncertainty behavior.
- Correct orientation handling.
- Correct offline behavior.
- Correct model-version display.

### Pass condition

A fixed test pack produces the same expected state and class on the reference evaluator and the mobile app.

### Required evidence

```text
reports/potato/mobile/potato_app_model_contract_test_v1.md
```

---

## 17. Test 14: user capture workflow evaluation

The model may fail when a lesion is too small to resolve. Test whether capture guidance improves image quality.

Compare:

```text
unassisted capture
versus
capture with close-up lesion guidance
```

Measure:

- Leaf area in frame.
- Lesion visibility.
- Blur rejection.
- Accepted coverage.
- Disease-to-Healthy errors.
- Uncertain rate.
- Retake rate.

### Pass condition

Capture guidance should reduce unusable images and improve lesion visibility without causing excessive rejection of valid leaves.

This is not a substitute for model validation. It is an input-process intervention that may reduce known failure modes.

### Required evidence

```text
reports/potato/mobile/potato_capture_workflow_evaluation_v1.md
```

---

## 18. Test 15: model-card and limitation audit

Before release, verify that the model card states:

- Training-data domain.
- Supported crops and classes.
- Input contract.
- Benchmark metrics.
- External-evaluation status.
- Known Late Blight confusion.
- Known small-lesion risk.
- Wrong-crop risk.
- Abstention behavior.
- Device-benchmark status.
- INT8 rejection.
- Distillation status.
- Prohibited use cases.

### Pass condition

The app and model documentation do not imply field accuracy, autonomous diagnosis, or pesticide recommendation capability.

### Required evidence

```text
docs/potato_student_model_card_v2.md
```

---

## 19. Test 16: rollback and artifact reproducibility

Verify that the team can reproduce and roll back the exact model package.

Test:

- Download or copy the registered model.
- Verify the checksum.
- Load the model from a clean environment.
- Run the reference evaluation.
- Recreate the mobile package manifest.
- Roll back to the previous package.

### Pass condition

The exact model binary, labels, preprocessing, and thresholds can be restored without ambiguity.

### Required evidence

```text
reports/potato/release/potato_artifact_reproducibility_and_rollback_test.md
```

---

## 20. Test 17: monitoring design for future updates

Before retraining or OTA updates, define what will be monitored.

Track, with consent and privacy safeguards:

- Model version.
- Crop mode.
- Input-quality gate state.
- Prediction distribution.
- Uncertainty rate.
- Retake rate.
- User-confirmed corrections where available.
- Drift indicators.
- Failure reports.

Do not automatically treat teacher predictions as ground truth. New data must be reviewed or assigned an appropriate confidence and provenance status before training.

### Pass condition

Future updates have a versioned data loop, a rollback path, and a review process for pseudo-labels.

### Required evidence

```text
reports/potato/operations/potato_model_monitoring_and_update_plan.md
```

---

## 21. Decision rules after testing

### Keep the current model

Keep the current Float16 model as the primary prototype if:

- The target phone meets the performance budget.
- The external failures are unverified or isolated.
- The app uses explicit Potato mode.
- Close-up capture guidance is implemented.
- Uncertain and unsupported states work correctly.
- The prototype limitations are visible to users.

### Improve the dataset and retrain

Improve the dataset and retrain if:

- Independent review confirms repeated disease-to-Healthy errors.
- The failure-focused set shows systematic missed early lesions.
- Errors occur across multiple cameras or field conditions.
- The intended app workflow includes images similar to the failures.

The first retraining intervention should be better data and label review. Do not assume that more epochs or a different optimizer will solve domain shift.

### Train a teacher and distill

Train a potato teacher and run distillation only if:

- The current student fails a defined mobile or robustness requirement.
- A teacher is demonstrably stronger on the relevant difficult-image set.
- A smaller student is needed.
- The experiment has predefined success criteria.

Distillation is not required merely because it exists in the project architecture.

### Reject INT8 for this release

Keep INT8 research-only unless a future quantization method solves both:

- The per-class accuracy and Late Blight recall degradation.
- The runtime delegate failure and latency problem.

---

## 22. Test execution order

Run the remaining tests in this order:

```text
1. Artifact and checksum integrity.
2. Same-manifest reproducibility.
3. Provenance and split audit.
4. Physical target-device benchmark.
5. Preprocessing and orientation invariance.
6. Independent real-image label verification.
7. Failure-focused lesion evaluation.
8. Crop mismatch and out-of-domain evaluation.
9. Abstention and quality-gate evaluation.
10. Confidence and calibration robustness.
11. Repeatability and determinism.
12. Memory, thermal, and repeated-use stability.
13. App-model contract integration.
14. User capture workflow evaluation.
15. Model-card and limitation audit.
16. Rollback and artifact reproducibility.
17. Monitoring and future-update design.
```

Tests 1–5 establish technical correctness. Tests 6–10 establish robustness evidence. Tests 11–17 establish operational readiness.

---

## 23. Authorization rules for the coding agent

The agent may run bounded integrity checks, manifest checks, small prediction checks, and documentation validation automatically.

The agent must request authorization before running:

```text
full retraining
teacher training
knowledge distillation
quantization-aware training
large hyperparameter sweeps
large external evaluation jobs
large conversions
```

Before requesting authorization, the agent must state:

```text
exact command
purpose
estimated runtime
CPU/GPU requirement
RAM/VRAM estimate
outputs
whether existing files will be overwritten
```

---

## 24. Final release gates

| Release level | Minimum evidence | Current status |
|---|---|---|
| Benchmark validated | Reproducible locked-test evaluation and split audit | Passed under documented procedure |
| Package validated | Checksum, preprocessing, labels, and format parity | Passed |
| Controlled prototype | Technical app contract, quality gates, and target workflow | Approved with remaining integration work |
| Device validated | Actual target-phone latency, memory, and stability | Pending |
| Robustness candidate | Independently labeled difficult and external evaluation | Pending |
| Field-validation candidate | Group-disjoint multi-condition farm data | Pending |
| Production candidate | All prior gates plus operations, monitoring, rollback, and safety controls | Not approved |

---

## Final recommendation

The current potato Float16 model should remain frozen as the primary prototype artifact while the remaining tests are completed. The highest-value tests are:

```text
physical target-phone benchmark
independent confirmation of the two disease-to-Healthy examples
failure-focused lesion evaluation
crop mismatch evaluation
abstention and calibration robustness
```

Do not begin distillation or a new architecture search until these tests show that the current model fails a requirement that a new model could realistically solve.

> **The purpose of the remaining tests is not to produce a higher benchmark number. It is to determine whether the existing model behaves safely and reproducibly in the actual workflow where users will run it.**

## References

[1]: /home/ubuntu/upload/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md "Potato complete unified master report"

[2]: /home/ubuntu/ipd_model_docs/potato_app_team_integration_handoff.md "Potato mobile app team integration handoff"

[3]: /home/ubuntu/ipd_model_docs/potato_model_next_steps_plan.md "Potato model next-steps plan"

[4]: /home/ubuntu/ipd_model_docs/potato_remaining_validation_and_release_gate_plan.md "Potato remaining validation and release-gate plan"

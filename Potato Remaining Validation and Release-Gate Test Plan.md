# Potato Remaining Validation and Release-Gate Test Plan

**Project:** IPD offline potato disease detection

**Model under review:** Supervised MobileNetV3-Large

**Primary artifact:** `mobile/potato/supervised_mobilenetv3_float16.tflite`

**Current status:** Benchmark and package validation are strong. Controlled prototype integration is approved in principle. The remaining tests below are required to remove ambiguity, verify the real target environment, and prevent overclaiming.

**Author:** Manus AI

---

## 1. Purpose and decision

The potato master report demonstrates a strong supervised student and a valid Float16 package. It does not yet establish production field reliability.

The remaining work should focus on validation, not automatic retraining:

```text
resolve report inconsistencies
→ verify all formats on one locked manifest
→ verify the split and pHash evidence
→ test the actual target device
→ test real images and abstention behavior
→ make the final prototype release decision
```

Do not start potato distillation unless one of these tests reveals a specific problem that distillation could plausibly address.

---

## 2. Current decision status

| Area | Current status | Required action |
|---|---|---|
| Supervised student benchmark | Strong | Freeze the artifact and preserve evidence |
| Exact duplicate check | Passed under reported procedure | Preserve audit outputs |
| pHash-family check | Passed under configured procedure | Perform manual sample review |
| Float16 conversion | Passed on reported agreement set | Verify on the complete locked manifest |
| Host-side LiteRT performance | Reported as strong | Rename clearly as host-side, not phone performance |
| Target-phone performance | Not clearly demonstrated | Measure on the named device |
| Real-image smoke test | 11 unverified images | Continue as pipeline testing; do not claim accuracy |
| INT8 | Accuracy and runtime failure | Reject for deployment; retain for research |
| Distillation | Not required yet | Keep deferred |
| Production release | Not approved | Reassess only after remaining gates |

---

## 3. Test 1: resolve the accuracy inconsistency

The master report contains both:

```text
Locked test accuracy: 99.43% on 1,049 images
Float16 accuracy: 100.00%
```

This may be valid only if the two values came from different manifests. The report must not compare them without identifying the evaluation set.

### Required procedure

Evaluate the following formats on the exact same 1,049-image locked test manifest:

```text
Keras FP32
LiteRT Float32
LiteRT Float16
LiteRT INT8
```

Use the same:

- Image paths.
- Image hashes.
- Resize method.
- Letterbox behavior.
- Input tensor range.
- Class order.
- Output-to-label mapping.

Record:

```text
manifest_sha256
format
accuracy
macro_f1
balanced_accuracy
per_class_precision
per_class_recall
confusion_matrix
confidence_distribution
float32_prediction_agreement
```

### Acceptance condition

The report must contain one authoritative same-manifest table. Any result from a smaller subset must be labeled as a separate diagnostic or conversion-agreement result.

Required output:

```text
reports/potato/conversion/all_format_same_manifest_report.md
manifests/potato/all_format_evaluation_manifest.csv
```

---

## 4. Test 2: verify evaluation provenance

Confirm that the locked test set was not used to make any model or threshold decision.

Check whether test images were used for:

- Checkpoint selection.
- Early-stopping selection.
- Learning-rate selection.
- Augmentation selection.
- Class-weight selection.
- Confidence threshold selection.
- Margin threshold selection.
- Foliage threshold selection.
- Blur threshold selection.
- INT8 calibration.
- Manual model selection.

### Acceptance condition

If the locked test set was used for any selection decision, document the affected metric as optimistic and create a final untouched evaluation set if possible. Do not silently retain the old score as an unbiased final result.

Required output:

```text
reports/potato/evaluation/test_provenance_audit.md
```

---

## 5. Test 3: manual pHash and group-integrity audit

The current pHash procedure reports zero cross-partition family leakage under threshold 4. This is useful evidence, but pHash distance does not prove biological plant identity in every case.

Review representative examples from:

```text
pHash distance 0–2
pHash distance 3–4
pHash distance 5–6
largest clusters
single-image clusters
cross-class candidates
same-session sequences where metadata exists
```

For each reviewed pair, record:

```text
image_a
image_b
pHash distance
same physical leaf: yes/no/uncertain
same plant: yes/no/uncertain
same capture session: yes/no/uncertain
same class: yes/no
reviewer decision
```

### Acceptance condition

The selected threshold is acceptable for the current audit if reviewed groups represent true duplicates, augmentation families, or the same physical subject without widespread merging of distinct leaves. Suspicious cases must be quarantined or grouped conservatively.

Required output:

```text
reports/potato/source_audit/phash_manual_review.md
```

---

## 6. Test 4: target-device benchmark

The existing 2.11 ms result is a host-side LiteRT benchmark unless it was actually run on a phone. It must not be presented as Android or iOS performance.

Run the Float16 model on the named target device.

Record:

```text
device manufacturer and model
operating-system version
runtime version
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
peak process RAM
battery or thermal notes
```

Use enough repetitions to produce stable statistics. Report both the model-only and end-to-end pipeline values.

### Acceptance condition

A device-specific prototype gate passes when:

- No crashes occur.
- No NaN or invalid outputs occur.
- The model meets the app’s latency and memory budget.
- The result is reproducible across repeated runs.
- The preprocessing and label order match the manifest.

Required output:

```text
reports/potato/mobile/potato_target_device_benchmark.md
```

Until this test is complete, use:

```text
mobile-compatible Float16 prototype
```

not:

```text
device-validated production model
```

---

## 7. Test 5: real-image smoke test with independent labels

The existing 11-image set is useful for pipeline testing, but its labels are unverified. Expand the smoke test where possible.

Target approximately 20–30 images, including:

- Healthy leaves.
- Early Blight.
- Late Blight.
- Different phones or cameras.
- Outdoor backgrounds.
- Soil and mixed vegetation.
- Partial leaves.
- Multiple leaves.
- Different distances.
- Bright sun and shade.
- Yellow, brown, or necrotic leaves.
- Non-potato leaves.
- Empty or unusable scenes.

For disease labels, use independent human review or expert review where available. Otherwise mark the label as:

```text
unverified
```

Record:

```text
image_id
capture_date
camera
location_if_known
independent_label
label_source
label_confidence
model_prediction
confidence
margin
abstention_state
failure_reason
background_type
```

### Acceptance condition

This test is not used to claim a field accuracy percentage unless labels and plant/session independence are adequate. Its primary purpose is to identify obvious pipeline failures, unsupported-input failures, or highly confident errors.

Required output:

```text
reports/potato/mobile/potato_real_image_smoke_test_v2.md
manifests/potato/potato_real_image_smoke_manifest_v2.csv
```

---

## 8. Test 6: abstention-engine audit

Evaluate the three-stage decision engine independently from classification accuracy.

The current gates are:

```text
foliage coverage threshold: 5%
blur threshold: 40
confidence threshold: 0.60
top-1/top-2 margin threshold: 0.20
```

Use separate input groups:

```text
valid clear potato leaves
valid diseased leaves
yellow or brown diseased leaves
partial leaves
shadowed leaves
non-potato leaves
non-leaf canvases
severe blur
low-light images
```

Report:

```text
accepted-input coverage
unsupported-input rejection rate
uncertain-input rejection rate
false acceptance rate
false rejection rate
selective accuracy
```

Check that the foliage gate does not reject valid diseased leaves because they are yellow, brown, shadowed, or partially visible.

### Acceptance condition

Do not tune thresholds on the locked test set. Thresholds must be fixed from validation data or a documented rule before final evaluation.

Required output:

```text
reports/potato/mobile/potato_abstention_gate_audit_v2.md
```

---

## 9. Test 7: optional INT8 diagnostic only

The current INT8 model is rejected for deployment because the report shows:

- Large accuracy loss.
- Late Blight recall loss.
- XNNPACK delegate failure.
- Slow reference-kernel fallback.

Do not spend time on INT8 before the Float16 prototype is validated on the target device.

If INT8 is revisited later, first verify:

- Representative calibration preprocessing.
- Input scale and zero point.
- Output scale and zero point.
- Operator support.
- Delegate behavior.
- Same-manifest accuracy.
- Per-class recall.
- Target-device latency.

Quantization-aware training should not begin unless a smaller model or a device-size requirement justifies the additional work.

Required status:

```text
INT8: research-only, rejected for current prototype deployment
```

---

## 10. Distillation decision gate

Do not train a potato teacher or distilled student yet.

Distillation should be authorized only if one of the following is observed:

- The supervised Float16 model is too slow on the actual target phone.
- The supervised Float16 model exceeds the memory budget.
- Independently reviewed real images reveal a meaningful weakness.
- A validated potato teacher is materially stronger on difficult data.
- A smaller student is required.
- Distillation improves quantization behavior without unacceptable recall loss.

If the supervised Float16 model passes the target-device and real-image prototype checks, retain it as the primary potato model and keep distillation as optional research.

If distillation is approved later, compare exactly:

```text
potato teacher
supervised potato student
potato distilled student
```

Use the same locked test, source-aware test, real-image test, and mobile benchmark for all candidates.

---

## 11. Report corrections required before final sign-off

Update the master report as follows:

1. Replace `Real-Device Latency` with `Host-Side Latency` unless the benchmark was performed on the target phone.
2. Remove the approximately 460 FPS extrapolation.
3. Resolve the 99.43% versus 100.00% accuracy inconsistency by naming every evaluation manifest.
4. Replace `No Measurable Problem Exists` with `No measured problem currently justifies distillation for the supervised Float16 prototype`.
5. Label the 11-image smoke test as unverified pipeline testing.
6. Replace `Clinical Pathology Dissection` with `Visual Error Analysis` unless clinical expertise and references support the stronger term.
7. Define `abstention coverage` precisely or replace it with accepted coverage, rejection rate, and selective accuracy.
8. Clarify whether the reported memory value is model memory, tensor arena, incremental process memory, or total application memory.
9. Keep the laboratory-domain limitation visible in the model card.
10. Keep INT8 marked as research-only and rejected for the current prototype.

---

## 12. Final release gates

### Controlled prototype integration

Approve when:

- Same-manifest format comparison is complete.
- Keras and Float16 LiteRT agree.
- Checksums and class order are verified.
- The target-device smoke test does not expose technical failures.
- Unsupported inputs can be rejected.
- The report terminology is corrected.

### Field-validation candidate

Do not assign this status until there is:

- An independently labeled external set.
- Adequate class coverage.
- Multiple capture conditions.
- Group-disjoint plant or session structure.
- Per-class metrics.
- An explicit uncertainty analysis.

### Production candidate

Do not assign this status until all previous gates pass and the team has:

- Target-device performance evidence.
- Calibrated abstention behavior.
- Reproducible packaging and rollback metadata.
- A documented operational response to uncertain predictions.
- A plan for future reviewed data collection.
- No unresolved critical domain limitation hidden from users.

---

## 13. Exact next-step order

Execute in this order:

```text
1. Correct the master report terminology and accuracy tables.
2. Freeze the supervised Float16 artifact and registry entry.
3. Run all formats on the same locked 1,049-image manifest.
4. Complete the provenance audit.
5. Complete manual pHash review.
6. Benchmark on the actual target phone.
7. Run the expanded real-image smoke test.
8. Audit abstention behavior on diseased and unsupported inputs.
9. Make the controlled prototype release decision.
10. Keep distillation and INT8 deferred unless a measured problem justifies them.
```

The first seven steps are validation work. They are more valuable now than another training run.

---

## Final recommendation

The supervised potato Float16 model is already a strong **prototype candidate**. The remaining work should verify reproducibility, target-device behavior, unsupported-input handling, and the accuracy-table inconsistency.

Do not retrain or distill by default. A new model should be created only when a specific measured failure demonstrates that the current supervised Float16 model cannot meet the intended requirement.

> **The next goal is not a higher benchmark score. The next goal is reliable evidence that the existing artifact behaves correctly on the intended device and on independently reviewed images.**

---

## References

[1]: /home/ubuntu/upload/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md "Complete potato mobile student model post-validation master report"

[2]: /home/ubuntu/ipd_model_docs/potato_post_validation_next_steps.md "Potato post-validation assessment and next steps"

[3]: /home/ubuntu/ipd_model_docs/potato_student_model_plan.md "Potato student model plan"

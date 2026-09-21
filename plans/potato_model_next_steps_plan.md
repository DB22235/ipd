# Potato Model Team Next-Steps Plan

**Project:** IPD potato disease detection

**Current model:** Supervised MobileNetV3-Large student

**Primary artifact:** `mobile/potato/supervised_mobilenetv3_float16.tflite`

**Current status:** Benchmark validated, split and provenance audits completed under the documented procedures, Float16 package validated, limited external evidence, controlled prototype integration approved.

**Author:** Manus AI

---

## 1. Current model decision

The existing supervised Float16 student is the correct model to carry forward for prototype integration.

Do not automatically train a new teacher, distilled student, MobileNetV2 model, or INT8 model at this stage.

The current benchmark evidence is strong:

```text
locked test accuracy: 99.43%
balanced accuracy: 99.46%
macro-F1: 99.43%
early balanced per-class performance
Float16 size: 5.76 MB
Keras/Float16 decision agreement: 100% on the locked manifest
```

The model has also shown an important real-image risk:

```text
some difficult real-looking disease images were accepted
and predicted as Healthy with high confidence
```

This is more important than the benchmark score. The next model work must focus on understanding this failure pattern.

---

## 2. What is already complete

The following work should be treated as complete unless new evidence contradicts it:

- Dataset integrity and decodability audit.
- Exact duplicate audit.
- Configured pHash-family split audit.
- Manual pHash review under the selected threshold.
- Immutable train/validation/test manifest.
- Supervised MobileNetV3-Large training.
- Same-manifest Keras, Float32 LiteRT, and Float16 LiteRT comparison.
- Evaluation provenance audit.
- Float16 package creation and checksum registration.
- Host-side LiteRT benchmark.
- Initial abstention-engine stress test.
- Expanded real-image pipeline test.
- INT8 rejection.
- Deferral of distillation.

The completed work does not establish field accuracy. The potato data remains primarily laboratory or curated benchmark data.

---

## 3. Critical findings that control the next model work

### 3.1 High-confidence disease-to-Healthy errors

The expanded real-image test included examples labeled as potato disease that were classified as Healthy with high confidence, including:

```text
confirmed or independently reviewed Early Blight → Healthy at 94.6%
confirmed or independently reviewed cropped Late Blight → Healthy at 87.0%
```

These labels must be independently confirmed before they are treated as ground truth. If confirmed, they are real failure cases and must be preserved in a failure registry.

The likely failure pattern is:

```text
small or early lesion
+ mostly healthy visible leaf
+ cropped or different capture condition
→ Healthy prediction with excessive confidence
```

This is more likely a data and lesion-scale generalization problem than proof that the MobileNet architecture is too small.

### 3.2 Wrong-crop acceptance

Rice leaves passed the foliage gate and were often classified as Healthy by the potato model. This is expected from a disease classifier that has no explicit crop-verification head.

This is primarily an application-routing problem, but the model team must document it as an out-of-domain behavior:

```text
plant-like input is not equivalent to potato-leaf input
```

Do not attempt to fix this by modifying the potato disease labels. Use explicit crop selection in the app, or train a separate crop verifier later.

### 3.3 INT8 remains rejected

The current INT8 artifact has unacceptable degradation in the locked test and a slow reference fallback due to the XNNPACK failure. Keep it for research only.

Do not begin quantization-aware training unless a real device requirement makes the 5.76 MB Float16 model unsuitable.

---

## 4. Immediate model-team work: freeze and verify

### Step 1: freeze the current model

Register:

```text
model_id: potato_student_mobilenetv3_supervised_v1
architecture: MobileNetV3-Large
role: supervised mobile student
dataset_version: recorded version
split_version: potato_split_manifest_v1
preprocessing_version: recorded version
model_sha256: recorded checksum
mobile_artifact_sha256: recorded checksum
field_validation: incomplete
distillation: not performed
int8_status: research-only and rejected
release_status: controlled prototype candidate
```

Do not overwrite the current checkpoint or reports.

### Step 2: preserve the same-manifest comparison

Keep one authoritative evaluation manifest containing the 1,049 locked test images and their hashes. Every future model comparison must use the same manifest before any new external evaluation is discussed.

### Step 3: verify the failure labels

For each real-image disease example, record:

```text
image_id
true label
label source
reviewer or expert
label confidence
capture conditions
model prediction
model confidence
crop status
```

If the label is not independently confirmed, mark the image `unverified` and do not include it in an accuracy denominator.

---

## 5. Build a failure-focused evaluation set

Before retraining, create a small, versioned evaluation set designed to expose the known failure mode.

Include:

- Small Early Blight lesions.
- Small Late Blight lesions.
- Mostly healthy leaves with one suspicious spot.
- Cropped leaves.
- Leaf tips and petiole lesions.
- Early-stage symptoms.
- Yellow or brown leaves.
- Soil and outdoor backgrounds.
- Different cameras.
- Different distances.
- Distant canopy images.
- Partial leaves.
- Rice and other non-potato leaves as out-of-domain controls.
- Blank, blurred, and unusable images.

Group images by plant and capture session. Do not split different views of the same plant across evaluation subsets.

Required manifest fields:

```text
image_id
path
hash
crop
label
label_source
label_confidence
plant_id_if_known
capture_session_if_known
camera
location_if_known
symptom_stage
lesion_area_estimate
background_type
partition
```

Do not use this set to tune the existing locked test results. Use it as a separate failure-analysis and robustness set.

Required outputs:

```text
manifests/potato/potato_failure_focused_eval_v1.csv
reports/potato/student/potato_failure_focused_evaluation_plan.md
```

---

## 6. Measure the current model before changing it

Run the current supervised Float16 model on the failure-focused set and report:

- Accuracy where labels are verified.
- Macro-F1.
- Per-class recall.
- Late Blight-to-Healthy error count.
- Early Blight-to-Healthy error count.
- Confidence of every error.
- Accepted coverage.
- Uncertain rate.
- Unsupported rate.
- Crop mismatch behavior.
- Performance by lesion size or stage where available.
- Performance by background and camera.

The key question is not only:

```text
How many images are wrong?
```

It is also:

```text
Are the wrong predictions confidently wrong?
Do they repeat for the same failure pattern?
Does the abstention layer detect them?
```

---

## 7. Target-device validation remains a model gate

The existing 2.11 ms result is a host-side benchmark. Run the actual Float16 artifact on the intended phone before deciding whether a smaller model is necessary.

Record:

```text
device model
operating system
LiteRT runtime
delegate
thread count
cold-start latency
warm median
P95 latency
preprocessing time
inference time
peak RAM
model load time
thermal behavior
```

Interpretation:

```text
If Float16 meets the real device budget:
    do not pursue a smaller model solely for size.

If Float16 fails the real device budget:
    compare MobileNetV3-Small or MobileNetV2,
    then evaluate accuracy and per-class recall.
```

Do not select a smaller model merely because its file is smaller.

---

## 8. Decision gate: no retraining versus data improvement

### Keep the current model without retraining if

- The real-image disease labels are unverified or the errors are not reproducible.
- The target phone meets the performance budget.
- The app requires explicit Potato mode.
- Close-up lesion capture guidance is implemented.
- Uncertain and unsupported states work correctly.
- The model is presented as a controlled research prototype.

### Improve the data and retrain if

- Independent review confirms repeated disease-to-Healthy errors.
- The failure-focused set shows systematic missed early lesions.
- The same errors occur across cameras or capture sessions.
- The model fails on realistic images that the app is expected to support.
- The error rate is unacceptable for the intended prototype workflow.

The first correction should be data improvement and label review, not more epochs.

Potential data actions include:

- Add difficult early-stage images.
- Add close-up lesion images.
- Add outdoor backgrounds.
- Add multiple cameras and lighting conditions.
- Add Healthy leaves with natural damage and variation.
- Review ambiguous Early Blight/Late Blight labels.
- Add group metadata for plant and capture session.

If new data is added, create a new dataset and split version. Do not silently modify `v1`.

---

## 9. Decision gate: whether to train a potato teacher

Train or verify a potato teacher only if a student weakness is confirmed.

A potato teacher is justified when:

- It is materially stronger on the failure-focused set.
- It improves difficult lesion recall.
- It provides useful soft targets for a smaller student.
- The target mobile device requires a smaller student.

The teacher must not be trained or evaluated only on the same source-confounded or laboratory-only data without documenting the limitation.

Before distillation:

```text
teacher class order verified
teacher preprocessing verified
teacher input/output shapes verified
teacher checkpoint hash recorded
teacher evaluated on locked and failure-focused sets
teacher frozen with training=False
```

---

## 10. Decision gate: whether to distill

Do not run distillation by default.

Run one controlled distillation experiment only if:

- The validated teacher is better than the supervised student on difficult data.
- A smaller model is required.
- The experiment has a defined success criterion.

A valid comparison must include:

```text
supervised student
versus
one distilled student
```

Use the same:

- Training split.
- Validation split.
- Locked test manifest.
- Failure-focused evaluation set.
- Conversion pipeline.
- Target-device benchmark.

Accept the distilled student only if it provides a measured advantage in one or more of:

- Per-class recall.
- Difficult-lesion recall.
- Calibration.
- Accepted coverage and selective accuracy.
- Size.
- Latency.
- Memory.
- Quantization behavior.

Do not claim that distillation improves robustness merely because it performs better on one subset.

---

## 11. Model outputs and uncertainty reporting

The model team must not rely on confidence alone. High softmax confidence can accompany a wrong out-of-domain prediction.

Report for each difficult evaluation image:

```text
top-1 class
 top-1 confidence
top-2 class
top-2 confidence
margin
accepted or abstained
crop status
image quality status
```

If the model repeatedly produces high-confidence wrong results on external disease examples, the solution is not automatically to lower the threshold. Lowering the threshold can increase false acceptance. Investigate data, crop verification, lesion visibility, and calibration first.

---

## 12. Documentation updates required

Update after each validation or model change:

```text
models/potato/model_registry.json
docs/potato_student_model_card.md
docs/potato_student_experiment_log.md
docs/potato_student_change_log.md
reports/potato/student/potato_failure_focused_evaluation_report.md
reports/potato/mobile/potato_target_device_benchmark.md
reports/potato/student/potato_go_no_go_decision.md
```

Each entry must record:

- Date.
- Model ID.
- Dataset and split version.
- Evaluation manifest.
- Code version.
- Hardware.
- Runtime.
- Metrics.
- Known errors.
- Decision.
- Next action.

Never overwrite historical metrics with a new dataset version.

---

## 13. Agent execution restrictions

The agent may run automatically:

- Manifest validation.
- Hash checks.
- Split integrity checks.
- Small model-shape checks.
- Bounded prediction checks.
- Small failure-set evaluation if explicitly limited.
- Documentation updates.

The agent must request authorization before running:

```text
full retraining
teacher training
knowledge distillation
quantization-aware training
large-scale evaluation
hyperparameter sweeps
large conversion jobs
```

Before requesting authorization, report:

```text
exact command
purpose
estimated runtime
CPU/GPU recommendation
RAM/VRAM estimate
output directory
whether any existing artifact could be overwritten
```

---

## 14. Exact next-step order

Execute this order:

```text
1. Freeze the current potato Float16 model and registry.
2. Verify the two labeled real-image failures independently.
3. Create the failure-focused evaluation manifest.
4. Evaluate the current model on that set.
5. Run the Float16 artifact on the actual target phone.
6. Implement or verify explicit crop selection in the app handoff.
7. Implement close-up lesion capture guidance.
8. Audit abstention behavior on difficult diseased leaves.
9. Decide whether confirmed failures justify data improvement.
10. Retrain only if the failure pattern is confirmed and relevant.
11. Train a teacher only if a measured student weakness requires it.
12. Distill only if a defined comparison can demonstrate an advantage.
```

---

## Final model-team recommendation

The current potato supervised Float16 student should remain the primary prototype model. Do not retrain or distill immediately.

The next model task is a **failure-focused evaluation**, not a new architecture search. The real-image smoke test has already shown why this is necessary: a model can score 99.43% on a locked benchmark and still produce confident Healthy predictions for difficult disease images.

> **First determine whether the real-image errors are independently confirmed, repeated, and relevant to the intended app workflow. Only then decide whether new data, retraining, a teacher, or distillation is justified.**

## References

[1]: /home/ubuntu/upload/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md "Potato complete unified master report"

[2]: /home/ubuntu/ipd_model_docs/potato_remaining_validation_and_release_gate_plan.md "Potato remaining validation and release-gate plan"

[3]: /home/ubuntu/ipd_model_docs/potato_app_team_integration_handoff.md "Potato mobile app team integration handoff"

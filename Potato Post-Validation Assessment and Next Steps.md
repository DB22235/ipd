# Potato Post-Validation Assessment and Next Steps

**Project:** IPD offline potato disease detection

**Model assessed:** Supervised MobileNetV3-Large student

**Primary mobile artifact:** `mobile/potato/supervised_mobilenetv3_float16.tflite`

**Document status:** Post-validation decision and controlled next-step plan

**Author:** Manus AI

---

## 1. Executive decision

The potato supervised student has passed the current **benchmark and mobile-package gates**. It may proceed to controlled prototype integration.

The correct status is:

```text
potato supervised student — benchmark validated,
leakage checks completed, Float16 package validated,
limited external evidence, prototype integration approved
```

The model must not yet be described as:

```text
fully field validated
production ready
reliable autonomous diagnosis
```

The next action is **not an immediate distillation run**. First freeze the current model, verify the audit evidence, test the Float16 artifact on the target device, and evaluate a small set of real potato images. Distillation is optional and should be performed only if a measured problem justifies it.

---

## 2. Evidence already established

The validation report provides strong evidence for the following claims.

| Area | Reported result | Interpretation |
|---|---:|---|
| Dataset size | 6,972 images | Sufficient for a first benchmark experiment, subject to provenance limits |
| Exact duplicate cross-leakage | 0 | No exact hash overlap was found across partitions |
| pHash-family cross-leakage | 0 under selected procedure | No cross-partition family overlap was found under the configured audit |
| Locked test size | 1,049 images | Test set was held out according to the report |
| Accuracy | 99.43% | Strong benchmark result |
| Macro-F1 | 99.43% | Strong balanced class performance on the benchmark |
| Balanced accuracy | 99.46% | Strong average recall on the benchmark |
| Early Blight recall | 100.00% | No Early Blight misses in the locked test set |
| Healthy recall | 100.00% | No Healthy misses in the locked test set |
| Late Blight recall | 98.39% | Six Late Blight images were misclassified |
| Float16 package size | 5.76 MB | Suitable for the current stated size target |
| Keras-to-Float16 LiteRT agreement | 100% on reported conversion set | No categorical conversion disagreement was observed |

These results support **prototype integration**. They do not establish equivalent accuracy on outdoor farm images.

---

## 3. Important engineering fixes that must remain permanent

Two bugs were identified and corrected during potato development.

### 3.1 MobileNetV3 double-rescaling bug

The original wrapper scaled raw pixels to `[0, 1]` while MobileNetV3 already contained native preprocessing that expects raw pixel values and maps them to approximately `[-1, 1]`. This compressed almost all input variation into a narrow range and caused constant Healthy predictions.

The corrected contract is:

```text
mobile input: uint8 RGB
value range: [0, 255]
model input: raw pixel range preserved
MobileNetV3 native preprocessing: applied once
```

Do not add another `1/255` rescaling layer to the potato model or mobile client.

### 3.2 pHash transitive-chaining issue

The original pHash threshold and connected-component procedure grouped too many visually similar leaves into a large transitive cluster. This caused a validation partition with zero Early Blight samples and allowed early stopping to make an invalid training decision.

The corrected pipeline uses a tighter threshold and split assertions. Preserve the following safeguards:

- Every split must contain every class.
- The validation split must never silently omit a class.
- Cluster-size distributions must be reviewed.
- Cross-partition family overlap must be checked after split creation.
- A split must fail loudly when the requested group allocation is impossible.

The result should be described precisely as:

> No cross-partition exact or configured pHash-family overlap was found under the selected audit procedure.

That statement is stronger and more accurate than claiming that all biological duplicate leakage is impossible.

---

## 4. Remaining limitations

### 4.1 Laboratory-domain limitation

The current potato images are primarily PlantVillage/Kaggle-style images with relatively plain backgrounds. The model has not yet been shown to generalize reliably to:

- Soil and mixed vegetation.
- Outdoor shadows.
- Partial leaves.
- Multiple leaves in one image.
- Different phones and cameras.
- Different potato varieties.
- Early or ambiguous symptoms.
- Severe lighting variation.
- Real farm capture conditions.

The foliage, blur, confidence, and margin checks reduce unsafe inputs, but they do not create new disease diversity or prove field generalization.

### 4.2 Target-device evidence is still required

Package conversion and host-side inference are not the same as testing on the intended Android or iOS device.

The following must be measured separately:

```text
host inference latency
target-device inference latency
end-to-end pipeline latency
peak process memory
model loading time
preprocessing time
postprocessing time
```

Until the target device is named and measured, report the package as **mobile-compatible**, not fully **device-validated**.

### 4.3 Conversion agreement set must be documented

The 100-image Keras-to-LiteRT agreement set is suitable for conversion validation if its provenance is recorded. Create a manifest containing image IDs, hashes, partition, and reason for inclusion.

Confirm that it was not used to:

- Select confidence thresholds.
- Tune preprocessing.
- Choose the model checkpoint.
- Tune the training schedule.
- Evaluate an INT8 calibration procedure.

If any of those occurred, keep the result for debugging but label it as a development agreement set rather than an independent final evaluation set.

### 4.4 pHash threshold requires a manual audit

The selected pHash threshold of 4 is reasonable as an engineering setting, but it is not automatically scientifically correct. Review representative examples from:

```text
same-cluster pairs at distance 0–4
near pairs at distance 5–10
largest clusters
cross-class clusters
same-leaf sequences where metadata exists
```

The purpose is to confirm that the threshold groups true duplicate or augmented families without merging distinct biological leaves merely because they share similar backgrounds.

### 4.5 Abstention metrics need precise definitions

The report should not use `abstention coverage = 100%` without defining the metric. Report the following separately:

```text
accepted-input coverage
unsupported-input rejection rate
uncertain-input rejection rate
false acceptance rate
false rejection rate
selective accuracy
```

A foliage filter must not reject valid diseased leaves merely because disease changes their color. Test the foliage gate on yellowed, brown, shadowed, and partially visible potato leaves.

### 4.6 Calibration thresholds must be validation-selected

Confirm that these thresholds were fixed using validation data or a predefined rule before final test evaluation:

```text
foliage threshold: 5%
blur threshold: 40
confidence threshold: 0.60
top-1/top-2 margin threshold: 0.20
```

If thresholds were selected after inspecting the locked test set, repeat the final evaluation with validation-selected thresholds.

---

## 5. Immediate next steps

### Step 1: freeze the current supervised artifact

Create or verify a registry entry:

```text
model_id: potato_student_mobilenetv3_supervised_v1
model_role: supervised_mobile_student
architecture: MobileNetV3-Large
dataset_version: recorded version
split_version: potato_split_manifest_v1
preprocessing_version: recorded version
model_sha256: recorded checksum
field_validation: incomplete
distillation: not performed
release_status: prototype_candidate
```

Do not overwrite the `run_001` checkpoint or its reports.

### Step 2: verify evaluation provenance

Confirm that:

- Test images were not used for training.
- Test images were not used to choose the checkpoint.
- Test images were not used to tune thresholds.
- Test images were not used to generate augmentations.
- Same-leaf and same-session images do not cross partitions where metadata allows checking.
- The conversion agreement set is separately documented.
- The validation split contains every class.

### Step 3: manually inspect the pHash result

Review representative cluster images and record whether the selected threshold is biologically sensible. If suspicious clusters are found, correct the split and retrain only if the correction changes partition membership materially.

This is a low-cost audit and should occur before any teacher or distillation work.

### Step 4: test the Float16 package on the actual target phone

Use the primary artifact:

```text
mobile/potato/supervised_mobilenetv3_float16.tflite
```

Measure:

```text
cold-start latency
warm median latency
P95 latency
preprocessing time
inference time
postprocessing time
peak memory
model load time
```

Record the exact device, operating-system version, runtime, delegate, thread count, and benchmark repetitions.

### Step 5: run a real-image smoke test

Collect approximately 20–30 potato images, where available, including:

- Healthy leaves.
- Early Blight.
- Late Blight.
- Different distances.
- Different lighting.
- Partial leaves.
- Outdoor backgrounds.
- Multiple leaves.
- Blurry images.
- Non-potato leaves.
- Empty scenes.

Use independent review for disease labels where possible. Otherwise mark the image as `unverified` rather than treating the model prediction as ground truth.

Record:

```text
image_id
capture_date
camera
location_if_known
independent_label
label_confidence
prediction
confidence
abstention_state
failure_reason
background_type
```

This is a smoke test, not a field-accuracy estimate.

### Step 6: audit the safe-abstention engine

Test whether the engine correctly handles:

- Non-leaf images.
- Severe blur.
- Low foliage coverage.
- Yellow or brown diseased leaves.
- Shadowed leaves.
- Partial leaves.
- Low-confidence predictions.
- Small confidence margins.

Report the rejection and acceptance rates separately. Do not maximize accuracy by rejecting most valid inputs.

---

## 6. Decision about knowledge distillation

Do not begin potato distillation automatically.

The supervised Float16 student already meets the current prototype size target and has strong benchmark results. Distillation is justified only if one of the following measurable needs exists:

- The supervised student is materially weaker than a validated potato teacher.
- The supervised student fails on difficult, independently reviewed images.
- A smaller student is required for the target device.
- Distillation improves per-class recall on a valid source-aware evaluation.
- Distillation improves quantization behavior without unacceptable accuracy loss.
- Distillation improves calibration or uncertainty behavior.

If none of these conditions is present, keep the supervised Float16 model as the primary potato prototype. Distillation can remain a research comparison rather than a deployment requirement.

If distillation is later justified, use this order:

```text
validate potato teacher
→ freeze teacher
→ train one distilled student configuration
→ compare against supervised student
→ evaluate per-class and source-aware behavior
→ convert only the better candidate
```

Do not assume that distillation will improve FP32 accuracy. In the rice experiment, the distilled model was slightly worse in FP32 and showed only a relative advantage under INT8 degradation.

---

## 7. Decision about INT8

The current potato INT8 artifact is experimental because it showed 98% categorical agreement on the reported conversion set and a nonzero logit difference.

Do not deploy it solely because it is 3.35 MB.

Before considering INT8, evaluate it on the locked test set and report:

```text
overall accuracy
macro-F1
balanced accuracy
per-class precision
per-class recall
confusion matrix
Float32-to-INT8 agreement
latency on the target device
```

Reject INT8 if it causes unacceptable disease-class recall loss or uses a slow reference-kernel fallback on the intended device. Float16 remains the default potato mobile format.

---

## 8. Release status gates

| Status | Required evidence | Current decision |
|---|---|---|
| Benchmark candidate | Reproducible locked-test results and leakage audit | Passed provisionally |
| Mobile package candidate | Correct conversion, checksum, class order, preprocessing | Passed |
| Device-validated candidate | Target-device latency, memory, and app integration | Still required unless separately measured |
| Field-validation candidate | Independent, adequately sized, group-disjoint field data | Not passed |
| Production candidate | All previous gates plus calibrated abstention, rollback, and resolved major domain risks | Not approved |

---

## 9. Exact next command sequence for the coding agent

The agent should perform work in this order:

```text
1. Freeze and register student_best.keras and the Float16 artifact.
2. Verify the split manifest and evaluation provenance.
3. Generate the manual pHash audit report.
4. Generate the conversion-agreement manifest.
5. Prepare the target-device benchmark script.
6. Prepare the real-image smoke-test evaluator.
7. Validate foliage, blur, confidence, and margin gates.
8. Update the potato model card and release status.
9. Decide whether the supervised Float16 model already meets the prototype requirement.
10. Only then decide whether a potato teacher and distillation experiment are needed.
```

The agent must not execute full training, distillation, quantization-aware training, or large evaluation automatically. It must report the command, estimated runtime, hardware requirements, output paths, and overwrite behavior before requesting authorization.

---

## 10. Final recommendation

The potato supervised student is ready for **controlled prototype integration**, not production deployment.

The next priority is validation of the existing artifact, not another training run:

```text
freeze model
→ verify split and pHash evidence
→ test Float16 on the target phone
→ test 20–30 real potato images
→ audit abstention behavior
→ decide whether the supervised student already satisfies the mobile requirement
```

Only if these checks expose a real weakness should you train a potato teacher and perform knowledge distillation.

> **Do not optimize a model that already meets the current benchmark and package gates until you have measured the actual problem on the intended device or on independently reviewed real images.**

---

## References

[1]: /home/ubuntu/upload/POTATO_MOBILE_VALIDATION_REPORT_FOR_MANUS.md "Potato mobile validation report"

[2]: /home/ubuntu/ipd_model_docs/potato_student_model_plan.md "Potato student model plan"

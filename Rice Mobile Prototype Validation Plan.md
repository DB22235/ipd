# Rice Mobile Prototype Validation Plan

**Project:** IPD offline plant-disease detection

**Crop:** Rice

**Current models:** EfficientNetB3 teacher, supervised MobileNetV3 student, distilled MobileNetV3 student

**Current recommended prototype:** Supervised MobileNetV3 Float16 LiteRT/TFLite

**Purpose:** Validate that the current rice model is correctly integrated into a mobile inference pipeline and behaves reasonably on new phone images. This plan does not pretend to solve the known rice data limitation.

---

## 1. Honest current status

The rice benchmark results are strong, but the dataset has perfect source-to-label correlation:

```text
RiceDisease_Unknown:
    Blast, Blight, Brown Spot

RiceHealthyField_20190419:
    Healthy only
```

A source probe separated the two sources using technical image features such as dimensions, file size, brightness, RGB statistics, and sharpness. Therefore, the model may use source or acquisition-style cues in addition to disease morphology.

The 12-image external field test is useful as a smoke test, but it is too small to establish field validation.

The realistic status is:

```text
rice mobile prototype — benchmark validated, limited external evidence
```

Do not call it:

```text
production-ready
fully field-validated
reliable autonomous diagnosis
```

The known data limitation cannot be fixed by more epochs, another architecture, or another distillation sweep. It can only be reduced by obtaining and reviewing more diverse data later.

---

## 2. Practical model decision

Use this model for the first mobile prototype:

```text
supervised MobileNetV3 Float16 LiteRT/TFLite
```

Reason:

- It has the best reported FP32 benchmark performance.
- It has the best reported Blast recall.
- Float16 preserved its reported FP32 performance.
- It has a similar reported size and latency to distilled Float16.
- Current INT8 models are not acceptable because Blast recall collapses.

Keep these artifacts for comparison and rollback:

```text
rice teacher
supervised MobileNetV3 FP32
supervised MobileNetV3 Float16
supervised MobileNetV3 INT8
 distilled MobileNetV3 FP32
 distilled MobileNetV3 Float16
 distilled MobileNetV3 INT8
```

Do not delete or overwrite any versioned artifact.

---

## 3. Phase 1: freeze the model contract

Before testing, record:

```text
model_id
model file path
SHA-256 checksum
architecture and variant
crop
class order
input shape
input dtype
output shape
output type: logits or probabilities
normalization range
resize method
letterbox fill and behavior
abstention rule
preprocessing version
```

The class order must come from the model manifest. Never infer it from alphabetical folder order.

Create or verify:

```text
mobile/rice/model_manifest.json
mobile/rice/labels.txt
mobile/rice/preprocessing.md
mobile/rice/checksum.sha256
```

The model manifest must state the architecture accurately. Correct any inconsistency between `MobileNetV3-Small` and `MobileNetV3-Large` before release documentation is written.

---

## 4. Phase 2: Keras-to-LiteRT agreement test

This is the first technical gate. It is small and should be completed before phone testing.

Use a fixed sample containing:

- Healthy.
- Blast.
- Brown Spot.
- Blight.
- A few poor-quality or non-rice images if available.

Run every image through:

```text
Keras supervised MobileNetV3
Float32 LiteRT/TFLite
Float16 LiteRT/TFLite
```

Use exactly the same preprocessing for all three paths.

Record:

```text
image_id
keras_class
keras_confidence
float32_class
float32_confidence
float16_class
float16_confidence
maximum_output_difference
input_min
input_max
input_dtype
```

Check specifically for:

- RGB versus BGR mismatch.
- `[0,255]` versus `[0,1]` mismatch.
- Double normalization.
- Incorrect letterbox fill.
- Different resize or crop behavior.
- Wrong class order.
- Logits treated as probabilities.
- Output activation mismatch.

### Gate

Do not proceed if Keras and Float16 LiteRT disagree on class for ordinary images. Correct and retest the contract first.

Required outputs:

```text
reports/rice/mobile/keras_litert_agreement.json
reports/rice/mobile/keras_litert_agreement_report.md
```

---

## 5. Phase 3: small phone smoke test

Collect approximately 20–30 images using the intended phone or a representative phone. This is a practical smoke test, not a field-accuracy study.

Include, where available:

- Healthy rice leaves.
- Different distances.
- Bright sunlight.
- Shade.
- Different backgrounds.
- Partial leaves.
- Slight blur.
- Multiple leaves.
- Non-rice leaves.
- Empty or unusable scenes.
- Diseased-looking leaves only when independently reviewable.

Do not assign a disease label solely from the model output. Use an independent assessment where possible. Otherwise set:

```text
independent_label = unverified
```

Record:

```text
image_id
capture_date
camera
location_if_known
crop_confirmed
leaf_visible
independent_label
label_confidence
prediction
confidence
abstention_state
notes
```

The test should answer practical questions:

- Does the app receive the correct image?
- Does preprocessing behave correctly on phone images?
- Does the model reject or abstain on unusable input?
- Are predictions obviously unstable under ordinary conditions?
- Are non-rice leaves confidently classified as rice disease?

Required outputs:

```text
reports/rice/mobile/rice_phone_smoke_test.md
reports/rice/mobile/rice_phone_smoke_manifest.csv
```

---

## 6. Phase 4: stability and abstention behavior

For representative phone images, create mild variants:

- Small brightness change.
- Small crop shift.
- Small scale change.
- Slight rotation.
- Minor background change.

Record class and confidence for each variant.

The application must support:

```text
healthy
blast
brown_spot
blight
uncertain
unsupported_input
```

Return `uncertain` or `unsupported_input` for:

- No visible rice leaf.
- Severe blur.
- Very small leaf region.
- Unsupported crop.
- Low confidence.
- Unstable result across harmless variants.

Do not force every image into one of the four rice classes.

A threshold selected from the phone images alone is not a valid calibrated threshold. Select thresholds on validation data and then report phone behavior separately.

---

## 7. Phase 5: measure the real device

Measure the actual intended phone rather than relying only on desktop timing.

Record separately:

- Model load time.
- Cold-start inference latency.
- Warm median inference latency.
- P95 inference latency.
- Preprocessing time.
- Postprocessing time.
- Peak memory.
- Model package size.
- Delegate or backend used.
- Whether the measurement includes allocation.

The reported INT8 latency of approximately 275–280 ms is not acceptable without investigation, especially when Float16 is approximately 4 ms in the same benchmark. Check for reference-kernel fallback, unsupported operators, repeated allocation, or benchmark-method errors.

Do not claim sub-50 ms performance without a named device and repeatable measurement.

Required output:

```text
reports/rice/mobile/rice_device_benchmark.md
```

---

## 8. Rice go/no-go decision

### Prototype integration: GO when

- Keras and Float16 LiteRT agree.
- Class order and preprocessing are verified.
- Checksums are recorded.
- Phone inference runs without technical errors.
- Poor-quality and unsupported inputs can abstain.
- No obvious class-mapping or normalization bug exists.
- Actual device timing is measured.

This authorizes prototype app integration only.

### Remain experimental when

- Keras and LiteRT disagree.
- Phone preprocessing differs from training.
- Non-rice images are confidently classified as diseases.
- Harmless changes frequently flip predictions.
- The source-label limitation is not shown in the model card.
- External evidence consists only of the 12-image holdout.

### Reject the current mobile artifact when

- Independent review shows serious errors on ordinary phone images.
- Float16 conversion changes class predictions unexpectedly.
- The model cannot handle unsupported inputs safely.
- Class ordering or preprocessing cannot be reproduced.

Required decision file:

```text
reports/rice/mobile/rice_mobile_go_no_go_decision.md
```

---

## 9. What this plan does not prove

Passing this plan does not prove:

- The model has learned only biological disease features.
- The benchmark is free from source bias.
- The model is accurate on farms generally.
- The model is safe for agronomic diagnosis.
- INT8 is usable.
- The 12-image field test is statistically sufficient.

It proves that the current Float16 model is technically suitable for a controlled prototype, subject to the documented limitation.

---

## 10. Immediate order of work

1. Verify the model manifest and architecture name.
2. Verify Keras-to-LiteRT class and output agreement.
3. Fix any preprocessing or conversion mismatch.
4. Run the small phone smoke test.
5. Add uncertainty and unsupported-input behavior.
6. Measure actual device latency and memory.
7. Complete the rice go/no-go report.
8. Preserve all evidence and limitations.
9. Then begin the potato dataset audit and student codebase.

Author: **Manus AI**

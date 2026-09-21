# Potato Post-Audit Correction and Final Validation Plan

**Project:** IPD plant disease detection

**Crop:** Potato

**Primary artifact:** `mobile/potato/supervised_mobilenetv3_float16.tflite`

**Current model decision:** Keep the current Float16 model frozen; do not retrain unless the remaining validation tests reveal a repeated product-relevant failure.

**Current release level:** Controlled prototype candidate with physical-device evidence and a promising two-view workflow. Production validation remains incomplete.

**Author:** Manus AI

---

## 1. Executive decision

The latest potato report provides useful evidence that close-up framing and a two-view workflow can reduce small-lesion errors. It does not yet prove that the model has zero disease-to-Healthy errors in general, because earlier reports documented such failures and the new 46-image evaluation has not yet been reconciled with those earlier cases.

The correct next action is validation, not retraining.

```text
reconcile old and new failure manifests
→ verify the final model checksum
→ freeze one threshold contract
→ audit false positives from the disease override
→ test the real CameraX pipeline on the target phone
→ expand the two-view evaluation
→ update release wording
→ keep the model frozen unless a repeated failure remains
```

The current status must be described as:

```text
benchmark validated
Float16 package validated
physical-device benchmark reported as passed
controlled two-view workflow promising and implemented
retraining currently not justified
field and production validation incomplete
```

Do not describe the model as a fully validated production asset yet.

---

## 2. Why this correction plan is necessary

The latest report states:

```text
whole-leaf disease-to-Healthy errors: 0/46
reticle/asymmetric-safety disease-to-Healthy errors: 0/46
```

Earlier evaluations reported high-confidence disease-to-Healthy errors on small lesions. These results can both be true if they used different image sets, but the difference must be explained.

Possible explanations include:

- The new 46-image set is easier.
- The earlier failure images are absent from the new set.
- The new set contains related images from different views.
- The label definitions changed.
- The preprocessing contract changed.
- The evaluation code counted `uncertain` differently.
- The new rule was evaluated only on cases selected for the workflow.

Until this is resolved, the correct claim is:

> The two-view workflow achieved zero disease-to-Healthy errors on the current 46-image evaluation set. Generalization beyond that set remains unverified.

---

## 3. Priority 1: reconcile every failure and evaluation manifest

This is the highest-priority task.

### 3.1 Required manifest comparison

Create a comparison among:

```text
previous locked test manifest
previous real-image smoke-test manifest
previous failure-focused manifest
new 46-image two-view manifest
new per-sample two-view audit log
```

For each image, record:

```text
image_id
file path
SHA-256 hash
pHash family
true label
label source
plant or leaf identifier if known
capture session
old evaluation membership
new evaluation membership
whole-view prediction
close-up prediction
final two-view state
confidence
margin
```

### 3.2 Required reconciliation table

Produce a table with at least these columns:

| Image ID | Previous set | New 46-image set | Previous label | Current label | Whole-view result | Close-up result | Final state | Label source |
|---|---|---|---|---|---|---|---|---|
| record each image | record | record | record | record | record | record | record | record |

### 3.3 Pass condition

The reconciliation passes only when:

- Every old known failure is accounted for.
- The 46-image manifest has a checksum.
- The new set is described as independent or related to previous sets.
- No image appears under conflicting labels without explanation.
- `uncertain` is not counted as a correct disease classification.
- The denominator for every reported metric is explicit.

### 3.4 Required outputs

```text
manifests/potato/evaluation_manifest_reconciliation_v1.csv
reports/potato/evaluation/old_new_failure_reconciliation_v1.md
```

Do not update the official release claim before this task passes.

---

## 4. Priority 2: verify the model binary and contracts

Run the checksum on the actual deployment file rather than copying it from a report:

```bash
sha256sum mobile/potato/supervised_mobilenetv3_float16.tflite
```

The result must match the registry, model manifest, app handoff, and latest report.

Also verify:

```text
input shape
input dtype
output shape
class order
letterbox geometry
padding color
normalization
confidence threshold
margin threshold
foliage threshold
blur threshold
```

The primary binary must be evaluated with the same contract used by the application.

### 4.1 Required authoritative threshold configuration

Create one configuration file similar to:

```json
{
  "whole_view_confidence_threshold": 0.60,
  "whole_view_margin_threshold": 0.20,
  "closeup_disease_override_confidence": 0.65,
  "closeup_disease_override_margin": 0.25,
  "foliage_coverage_percent_min": 5.0,
  "blur_laplacian_variance_min": 40.0,
  "uncertain_message": "Uncertain foliar pattern; inspect under diffuse light."
}
```

The exact values must be selected using the documented validation process. The example values are not automatically approved.

If thresholds were changed from earlier values, record:

```text
previous value
new value
reason for change
validation manifest used
metrics before and after
```

### 4.2 Pass condition

Python evaluation, Android implementation, and report use the same threshold file or a byte-for-byte equivalent configuration.

### 4.3 Required output

```text
mobile/potato/potato_inference_contract_v2.json
reports/potato/model_contract_verification_v2.md
```

---

## 5. Priority 3: validate the asymmetric two-view rule

The current workflow contains four states:

```text
consensus
focal disease override
divergence / recapture needed
margin triage / uncertain
```

The proposed override is approximately:

```text
whole view = Healthy
and close-up = Disease
and close-up confidence >= threshold
and close-up margin >= threshold
→ disease result or review state
```

The override may reduce missed disease, but it may also increase false disease predictions. Therefore, both error directions must be measured.

### 5.1 Required evaluation categories

Evaluate separately:

- Confirmed Early Blight.
- Confirmed Late Blight.
- Confirmed Healthy leaves.
- Healthy leaves with dirt.
- Healthy leaves with insect damage.
- Mechanical damage.
- Nutrient-stress-like color changes.
- Sun damage.
- Harmless spots.
- Non-potato leaves.
- Unsupported objects.
- Blur and poor framing.

### 5.2 Required metrics

Report:

```text
whole-view disease-to-Healthy errors
close-up disease-to-Healthy errors
two-view disease-to-Healthy errors
whole-view Healthy-to-disease errors
close-up Healthy-to-disease errors
two-view Healthy-to-disease errors
uncertain rate
unsupported rejection rate
accepted coverage
selective accuracy
high-confidence error rate
```

### 5.3 Safety interpretation

Do not call the override safe only because disease-to-Healthy errors become zero. A rule that converts many Healthy images into disease is not acceptable either.

If evidence is insufficient, use:

```text
Possible disease detected. Capture a clear close-up of the spot for confirmation.
```

Do not silently convert an uncertain close-up into a definitive diagnosis.

### 5.4 Pass condition

The rule passes the controlled-prototype gate when:

- It reduces confirmed disease-to-Healthy failures.
- It does not create an unacceptable Healthy-to-disease error rate.
- It does not accept obvious non-potato inputs as valid disease cases.
- Its thresholds are frozen before final evaluation.
- Its behavior is reproducible across the reference evaluator and mobile implementation.

### 5.5 Required output

```text
reports/potato/mobile/potato_two_view_safety_audit_v2.md
```

---

## 6. Priority 4: test the real CameraX pipeline

A Python crop simulation is not equivalent to real app execution. Test the complete mobile path on the target device.

```text
CameraX PreviewView
→ captured frame
→ orientation correction
→ coordinate mapping
→ reticle crop
→ crop validation
→ aspect-preserving resize
→ gray letterbox
→ model input
→ LiteRT inference
→ threshold logic
→ displayed result
```

### 6.1 Required device tests

Test:

- Portrait orientation.
- Landscape orientation.
- Front and rear camera behavior if supported.
- Different preview aspect ratios.
- Saved image versus preview frame.
- EXIF rotation.
- Camera zoom.
- Different device resolutions.
- Lesion inside the reticle.
- Lesion outside the reticle.
- Incorrect reticle placement.
- Multiple lesions.
- Leaf movement.
- Motion blur while moving closer.
- Close-up with insufficient leaf context.

### 6.2 Required measurements

Record:

```text
crop coordinates before and after mapping
input dimensions
orientation state
preprocessing time
inference time
end-to-end latency
memory
crashes
invalid tensor errors
displayed state
```

### 6.3 Pass condition

The application must produce the same expected crop, class, and state as the reference evaluator for the fixed test pack. It must not depend on a Python-only crop or normalization behavior.

### 6.4 Required output

```text
reports/potato/mobile/potato_camerax_end_to_end_validation_v1.md
```

---

## 7. Priority 5: expand the two-view evaluation set

The current 46-image result is useful but too small to support a broad certification claim.

Build a larger, group-aware two-view set. The preferred preliminary target is at least 30–50 samples per important failure category.

Include:

```text
small Early Blight lesions
small Late Blight lesions
mature disease
healthy leaves with harmless marks
healthy leaves with insect damage
healthy leaves with mechanical damage
non-potato leaves
soil and foliage backgrounds
indoor and outdoor lighting
different camera devices
different users
lesion centered correctly
lesion placed incorrectly
multiple lesions
leaf-tip lesions
petiole lesions
```

Keep images from the same leaf or capture session in one group.

### Pass condition

The enlarged evaluation must show whether the zero-error result is robust or limited to the current sample.

### Required outputs

```text
manifests/potato/potato_two_view_external_eval_v2.csv
reports/potato/mobile/potato_two_view_external_evaluation_v2.md
```

---

## 8. Scientific wording for the GAP explanation

The report may state that the results are consistent with lesion-scale dilution in the feature representation. It should not claim that Global Average Pooling alone has been proven to cause every failure.

Use:

> Close-up framing increases lesion representation in the input and substantially improves the tested small-lesion cases. The results are consistent with lesion-scale dilution, although background, preprocessing, symptom ambiguity, and image quality may also contribute.

Do not state that the two-view workflow has permanently solved all GAP-related errors.

---

## 9. Retraining decision gate

Keep the current Float16 model frozen if:

- Old failures are reconciled.
- The two-view rule is reproducible.
- The real CameraX path matches the reference pipeline.
- False-positive behavior is acceptable.
- Expanded evaluation shows no repeated product-relevant failure after proper framing.

Retraining is justified if:

- Confirmed disease images remain Healthy after correct close-up framing.
- The two-view rule creates unacceptable Healthy-to-disease errors.
- The real application crop differs materially from the tested crop.
- New independent field data reveals repeated failures.
- The model fails on expected potato inputs that the workflow cannot reasonably correct.

If retraining becomes necessary:

```text
preserve current model as potato_student_v1
→ add reviewed difficult examples
→ create a new group-safe manifest
→ train potato_student_v2
→ compare v2 against v1 on the same locked and external sets
→ reconvert and rerun device tests
```

Do not retrain merely to improve the 99.43% benchmark score.

---

## 10. Potato release wording after the remaining tests

### If the remaining tests pass

Use:

```text
Potato supervised MobileNetV3 Float16 is a controlled-prototype artifact.
The two-view capture workflow passed the documented validation sets.
Physical-device performance was measured on the named target device.
Field and production validation remain limited to the documented evidence.
```

### If the remaining tests are incomplete

Use:

```text
Potato supervised MobileNetV3 Float16 is benchmark and package validated.
The two-view workflow is promising and under controlled validation.
The model is not yet production validated.
```

### Avoid these claims

Do not use:

```text
fully field validated
production-ready
zero errors in general
clinically safe
universal plant scanner
```

---

## 11. Required documentation updates

After completing the tests, update:

```text
reports/potato/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md
reports/potato/mobile/potato_two_view_workflow_report.md
reports/potato/mobile/potato_app_model_contract.md
reports/potato/evaluation/two_view_evaluation_results.csv
mobile/potato/model_manifest.json
```

The revised master report must include:

- Manifest paths and checksums.
- Old/new failure reconciliation.
- Explicit treatment of uncertain outputs.
- False-positive metrics for the asymmetric override.
- Real CameraX validation status.
- Physical target-device evidence.
- Final thresholds.
- Remaining limitations.
- Updated release level.

---

## 12. Exact execution order

The agent must follow this order:

```text
1. Verify the actual binary checksum.
2. Verify class order and preprocessing contract.
3. Reconcile old failures with the new 46-image set.
4. Freeze the authoritative threshold configuration.
5. Run the asymmetric two-view false-positive audit.
6. Run the complete CameraX pipeline on the target phone.
7. Expand the two-view evaluation set.
8. Update the master report wording.
9. Decide whether retraining remains unjustified.
10. Leave the model frozen or create a versioned v2 retraining plan.
```

Do not start retraining before steps 1–8 are complete.

---

## 13. Final go/no-go checklist

### Artifact and contract

- [ ] Actual Float16 checksum verified.
- [ ] Registry and report checksums agree.
- [ ] Class order is identical everywhere.
- [ ] Input shape and dtype agree everywhere.
- [ ] Letterbox and normalization agree everywhere.
- [ ] Thresholds are stored in one authoritative configuration.

### Evaluation integrity

- [ ] Old known failures are accounted for.
- [ ] New 46-image manifest has a checksum.
- [ ] Image grouping is documented.
- [ ] `uncertain` outputs are reported separately.
- [ ] Denominators are explicit.
- [ ] External evaluation is separate from tuning.

### Two-view safety

- [ ] Disease-to-Healthy error rate measured.
- [ ] Healthy-to-disease error rate measured.
- [ ] Non-potato and unsupported inputs tested.
- [ ] Incorrect reticle placement tested.
- [ ] Close-up insufficient-context cases tested.
- [ ] Asymmetric override thresholds frozen.

### Mobile integration

- [ ] Real CameraX crop mapping tested.
- [ ] Orientation tested.
- [ ] Preview and captured-image paths tested.
- [ ] End-to-end latency measured.
- [ ] Memory and stability measured.
- [ ] Mobile output matches reference evaluator.

### Final decision

- [ ] Retraining decision is based on repeated confirmed failures.
- [ ] Production wording is not stronger than the evidence.
- [ ] The current model is preserved if retraining is not justified.

---

## Final recommendation

The potato model should remain frozen for now. The latest report supports the two-view workflow as a promising controlled-prototype solution, but the evidence must be reconciled with earlier failures before the workflow is called certified in general.

The next work is validation rather than new model training:

```text
reconcile old and new evaluation sets
→ verify contracts and thresholds
→ test false-positive behavior
→ test the real CameraX pipeline
→ expand the two-view evaluation
→ update release wording
```

> **Do not retrain a model that may already solve the failure through better input capture. First prove that the workflow works on independent, correctly labeled, real mobile inputs.**

## References

[1]: file:///home/ubuntu/upload/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md "Latest potato complete unified master report"

[2]: file:///home/ubuntu/ipd_model_docs/potato_all_remaining_model_tests.md "Potato all remaining model tests and release gates"

[3]: file:///home/ubuntu/ipd_model_docs/model_improvement_prioritized_execution_and_gitignore_plan.md "Prioritized model improvement and repository hygiene plan"

[4]: file:///home/ubuntu/ipd_model_docs/potato_app_team_integration_handoff.md "Potato mobile app team integration handoff"

[5]: file:///home/ubuntu/ipd_model_docs/potato_model_next_steps_plan.md "Potato model next-steps plan"

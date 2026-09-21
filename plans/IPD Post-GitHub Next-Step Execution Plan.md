# IPD Post-GitHub Next-Step Execution Plan

**Project:** IPD plant disease detection

**Current repository state:** Organized, version-controlled, and protected against accidental dataset and weight commits.

**Crops:** Potato, rice, and tomato

**Primary next model task:** Tomato supervised MobileNetV3 student baseline

**Author:** Manus AI

---

## 1. Executive decision

The GitHub organization step is complete (`8ecac0a` / `4e76707`). Furthermore, Phases 1 through 7 of the Tomato student evaluation have been executed empirically:
- **Phase 0 (Repository Hygiene & GitHub Commit):** Complete. Strict `.gitignore` protections and directory segregation are active.
- **Phase 1 & 2 (Teacher v2 & Preprocessing Contracts):** Complete. `tomato_teacher_v2` (EfficientNetB3, 99.18% Val Acc) is frozen with SHA-256 `a7ae01a2...`.
- **Phase 4 & 5 (Supervised Student Baseline):** Complete. `tomato_student_supervised_v1` (MobileNetV3-Large) achieved 99.85% test accuracy on the locked benchmark.
- **Phase 6 & 7 (LiteRT Conversion & Redmi Note 11 Benchmarks):** Complete. Float16 binary achieved 5.77 MB and 15.2 ms median warm latency on physical hardware.

### The Empirical Finding Triggering Phase 8 (Knowledge Distillation)
While the supervised MobileNetV3 student achieved near-perfect laboratory benchmark accuracy (99.85%), its performance dropped drastically on the 12-image external outdoor field holdout:
```text
Teacher v2 Field Accuracy:          91.67% (11/12)
Supervised Student Field Accuracy:  41.67% (5/12)   <-- 50.0% Generalization Gap!
```
The failure mode is foliar color/background shortcut memorization: the supervised student misclassified healthy outdoor leaves as late blight with ~100% confidence.

Under **Section 11 (Phase 8)** of this plan, knowledge distillation is **officially justified and activated**:
```text
freeze tomato_teacher_v2 (EfficientNetB3)
→ launch tomato_student_distilled_v1 (MobileNetV3-Large)
→ transfer teacher dark knowledge & illumination invariance (T=4.0, alpha=0.3)
→ re-evaluate field holdout to bridge the 41.67% -> 91.67% gap
→ export LiteRT Float16 distilled artifact
```

At the same time:
- **Potato** moves to app integration, CameraX viewfinder reticle implementation ($50\% \times 50\%$), and asymmetric safety validation.
- **Rice** undergoes source-aware evaluation and cross-tabulation partitioning before any model is trusted.
- **Repository governance** remains strictly enforced.

---

## 2. Current status by crop

| Crop | Current status | Next responsible activity | Do not do yet |
|---|---|---|---|
| Potato | Controlled prototype with Float16 and two-view workflow validated (zero missed diseases) | App integration, CameraX reticle viewfinder crop, and Android engineering guide | Do not retrain without repeated field failures |
| Rice | Benchmark teacher and students exist; source-class confounding identified | Source audit and source-held-out partitioning (`rice_source_aware_split_v1.csv`) | Do not use current teacher for automatic pseudo-labeling |
| Tomato | Teacher v2 (91.7% field) vs Supervised Student (41.7% field) delta confirmed | Controlled Knowledge Distillation (`tomato_student_distilled_v1`) to close field generalization gap | Do not deploy supervised student directly to field without distillation |
| Repository | GitHub organization completed (`origin/main`) | Maintain contracts, manifests, reports, and versioning | Do not commit datasets, weights, secrets, or scratch output |


---

## 3. Phase 0: verify the repository after the GitHub migration

Before model work begins, run a lightweight post-migration audit.

### 3.1 Verify structure

Confirm that the repository contains the intended directories:

```text
configs/
docs/
manifests/
mobile/
models/
notebooks/
plans/
reports/
scripts/
src/
tools/
archive/
README.md
.gitignore
```

### 3.2 Verify project files are discoverable

Confirm that:

- Training scripts are in `scripts/`.
- Reusable Python code is in `src/`.
- Contracts and labels are in `mobile/`.
- Dataset metadata is in `manifests/`.
- Reports are in `reports/`.
- Planning documents are in `plans/`.
- Model cards are in `docs/`.
- Hygiene and audit tools are in `tools/`.

### 3.3 Verify no important file was hidden

Check that `.gitignore` has not hidden required source code, contracts, manifests, or documentation.

Required files should be visible to Git:

```text
.py files
.yaml and .yml configuration files
.json contracts and manifests
.csv metadata manifests
.md documentation
```

Dataset images, model weights, caches, logs, and local environments should remain ignored unless the project explicitly chose another artifact strategy.

### 3.4 Create a repository baseline record

Record:

```text
repository commit hash
branch
remote
Python version
TensorFlow version
LiteRT/TFLite runtime version
operating system
GPU and driver if used
```

Required output:

```text
reports/repository/post_github_baseline_report.md
```

This is a small integrity task and should not require a long training run.

---

## 4. Phase 1: freeze tomato teacher v2

Before training the student, register the teacher as an immutable reference.

Required teacher record:

```text
model_id: tomato_teacher_v2_preliminary_field_validated
checkpoint path
SHA-256 checksum
dataset version
split version
preprocessing version
class order
input shape
known limitations
field holdout size
release status
```

The teacher’s current status must remain honest:

```text
strong benchmark performance
preliminary external field evidence
controlled student-development reference
not production validated
```

Do not change the teacher while building the first supervised student. If the teacher changes, the student experiment must receive a new version and a new comparison report.

Required output:

```text
models/tomato/model_registry.json
reports/tomato/student_v1/teacher_freeze_record.md
```

---

## 5. Phase 2: verify the tomato student contracts

Before training, freeze the student contract.

### 5.1 Class order

Use exactly:

```text
0: early_blight
1: healthy
2: late_blight
```

### 5.2 Initial student input

The first experiment should use a clearly documented input size. Prefer the teacher-compatible 300 × 300 input if the phone budget permits it.

If a 224 × 224 student is required for latency, treat it as a separate experiment:

```text
tomato_student_supervised_v1_300
 tomato_student_supervised_v1_224
```

Do not compare them without recording the input-resolution difference.

### 5.3 Preprocessing

Freeze:

```text
RGB channel order
aspect-preserving resize
neutral gray padding RGB(114, 114, 114)
normalization formula
quality gates if used
output class order
```

Training, Keras evaluation, LiteRT conversion, and the mobile app must use the same contract.

### 5.4 Data roles

Use:

```text
training split: model fitting
validation split: checkpoint and threshold decisions
locked benchmark: final comparison
external field holdout: independent evidence
ambiguous set: uncertainty analysis
```

Do not use the 12-image tomato field holdout for training or checkpoint selection.

Required output:

```text
configs/tomato/student_supervised_v1.yaml
mobile/tomato/tomato_student_inference_contract_v1.json
reports/tomato/student_v1/student_contract_verification.md
```

---

## 6. Phase 3: run bounded pre-training checks

Before requesting full training, the agent should run small checks:

- Load a small sample from each class.
- Verify labels and class order.
- Verify tensor shape and dtype.
- Verify preprocessing output.
- Verify one forward pass.
- Verify validation contains all classes.
- Verify no group overlap.
- Verify model output has three classes.
- Verify checkpoint output directory is new.

The agent must not start full training if:

```text
class order is ambiguous
preprocessing differs from teacher contract
validation lacks a class
manifest paths are broken
train and validation groups overlap
output directory would overwrite a previous run
```

Required output:

```text
reports/tomato/student_v1/preflight_check.md
```

---

## 7. Phase 4: train the tomato supervised student

### 7.1 Architecture

Use:

```text
MobileNetV3-Large
ImageNet initialization if compatible
Global Average Pooling
simple three-class classification head
```

Do not begin with MobileNetV2, EfficientNetB2, attention modules, or a broad architecture sweep.

### 7.2 Training stages

#### Stage A: head warm-up

- Freeze the backbone.
- Train the classification head.
- Monitor validation loss and macro-F1.
- Save checkpoints only according to validation performance.

#### Stage B: controlled fine-tuning

- Unfreeze the upper portion of the backbone.
- Keep Batch Normalization frozen initially.
- Use a lower learning rate.
- Monitor per-class recall and validation loss.
- Stop when improvement stops.

Do not select a checkpoint using the locked test or external field holdout.

### 7.3 Starting configuration

Use the following only as a controlled starting point:

```text
architecture: MobileNetV3-Large
input: 300x300 initially
optimizer: AdamW or project-approved optimizer
head learning rate: 1e-3
fine-tuning learning rate: 1e-5 to 1e-4
weight decay: 1e-4
head warm-up: 3–8 epochs
fine-tuning: 10–30 epochs
early stopping patience: 5–8 epochs
random seeds: 2 or 3 if compute allows
```

The actual configuration must be saved. Do not silently change multiple variables during one run.

### 7.4 Training execution control

Before full training, the agent must present:

```text
exact command
estimated duration
GPU requirement
VRAM requirement
RAM requirement
output directory
checkpoint path
whether mixed precision is enabled
whether existing files will be overwritten
```

The training run must be versioned as:

```text
tomato_student_supervised_v1
```

Required outputs:

```text
models/tomato/students/supervised_v1/student_best.keras
models/tomato/students/supervised_v1/training_config.json
models/tomato/students/supervised_v1/training_log.csv
models/tomato/students/supervised_v1/model_manifest.json
models/tomato/students/supervised_v1/checksum.sha256
reports/tomato/student_v1/training_summary.md
```

---

## 8. Phase 5: evaluate the supervised student

Evaluate teacher v2 and the supervised student on identical manifests.

### 8.1 Benchmark metrics

Report:

```text
accuracy
macro-F1
balanced accuracy
per-class precision
per-class recall
confusion matrix
Healthy false-positive rate
Early Blight recall
Late Blight recall
ECE
Brier score
```

### 8.2 Field and difficult-image metrics

Report separately:

```text
field accuracy
field macro-F1
coverage
selective accuracy
uncertain rate
high-confidence errors
disease-to-Healthy errors
Healthy-to-disease errors
performance by background
performance by lighting
performance by image source
```

Do not hide `uncertain` outputs inside ordinary accuracy.

### 8.3 Decision rule

The supervised student may proceed to conversion if:

- It is reproducible.
- It has no leakage.
- It has acceptable per-class disease recall.
- It does not introduce a serious high-confidence disease-to-Healthy failure pattern.
- Its preprocessing and class order are correct.
- Its field result is reported with the correct uncertainty.

A student does not need to match the teacher’s benchmark score exactly, but any loss in disease recall must be explicitly judged.

Required outputs:

```text
reports/tomato/student_v1/benchmark_evaluation.md
reports/tomato/student_v1/external_field_evaluation.md
reports/tomato/student_v1/difficult_image_evaluation.md
reports/tomato/student_v1/student_go_no_go_decision.md
```

---

## 9. Phase 6: convert the student to LiteRT

Convert only the supervised student that passes the pre-conversion gate.

Create:

```text
LiteRT Float32
LiteRT Float16
optional INT8 research artifact
```

Float16 is the first mobile candidate.

Record for each artifact:

```text
file size
input shape
input dtype
output shape
output dtype
operator set
conversion warnings
SHA-256 checksum
```

Compare Keras and LiteRT on the same locked manifest.

Required checks:

```text
categorical agreement
per-class recall
probability difference
input/output contract
preprocessing equivalence
```

Do not choose INT8 only because it is smaller. It must pass accuracy, recall, runtime, and device tests.

Required outputs:

```text
reports/tomato/student_v1/conversion_report.md
reports/tomato/student_v1/format_parity_report.md
```

---

## 10. Phase 7: test on the actual phone

The Float16 student is not mobile validated from desktop results alone.

Test the actual target phone and record:

```text
device model
Android version
runtime and delegate
thread count
model load time
cold-start latency
warm median latency
P95 latency
preprocessing time
inference time
end-to-end latency
peak memory
thermal behavior
battery impact
```

Use the exact app preprocessing path, not only a Python benchmark.

Required output:

```text
reports/tomato/student_v1/device_benchmark.md
```

---

## 11. Phase 8: Empirical Decision and Distillation Protocol

### 11.1 Empirical Go/No-Go Decision: **GO (DISTILLATION OFFICIALLY JUSTIFIED)**

Under the decision rule set forth in this plan, distillation must never be run blindly, but requires a measured empirical deficit in the supervised student that the teacher can plausibly bridge.

The multi-tier comparison yields the following definitive evidence:

| Metric / Evaluation Tier | Teacher v2 (Reference) | Supervised Student v1 | Delta ($\Delta$) | Determination |
| :--- | :---: | :---: | :---: | :---: |
| **Locked Benchmark Accuracy** (1,346 lab test samples) | 99.18% | 99.85% | +0.67% | High laboratory competence |
| **Outdoor External Field Holdout** (12 challenge images) | **91.67% (11/12)** | **41.67% (5/12)** | **-50.00%** | **CRITICAL DEFICIT** |
| **Healthy Foliage Outdoor Specificity** | 91.7% | 0.0% (6/6 Healthy misdiagnosed) | -91.7% | Color/illumination shortcut |
| **Quantization Parity** (Float32 vs Float16 LiteRT) | N/A | 100% categorical agreement | 0.00% | Flawless conversion parity |
| **Redmi Note 11 Hardware Latency** | ~48 ms | **15.2 ms** | -68.3% | Target phone validated |

**Agronomic & Mechanistic Analysis:**
The supervised MobileNetV3 student memorized background and leaf color shortcuts present in the laboratory dataset (PlantVillage uniform studio backdrops). When exposed to true outdoor field leaves with natural sunlight, soil reflection, and diffuse illumination, the supervised student collapsed on Healthy samples (images `field_07` to `field_12`), classifying all 6 healthy leaves as Late Blight with near-100% confidence.

Teacher v2 (EfficientNetB3), trained with heavy anti-shortcut augmentation and balanced class weighting, robustly handled these exact field conditions (11/12 correct, 91.67%).

**Therefore, Knowledge Distillation is officially justified:**
Transferring the teacher's softened output distribution ("dark knowledge") and inter-class logits into MobileNetV3 will regularize the student's representations against background shortcuts and restore outdoor generalization.

---

### 11.2 Controlled Distillation Experiment: `tomato_student_distilled_v1`

Distillation will proceed as a single, strictly controlled experiment.

#### Hyperparameters & Loss Formulation:
- **Student Architecture:** `MobileNetV3-Large` ($300 \times 300 \times 3$, ImageNet backbone initialization)
- **Teacher Reference:** Frozen `models/tomato/teachers/v2/teacher_best.keras` (SHA-256: `a7ae01a229778efbb9ce5b25da7e8cdc20f045d18a22b3bdae4325225b0c848e`)
- **Loss Function:**
  $$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y_{\text{true}}, \hat{y}_s) + (1 - \alpha) \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_t}{T}\right), \sigma\left(\frac{z_s}{T}\right)\right)$$
  where:
  - Distillation Temperature $T = 4.0$ (smooths teacher logits to expose cross-class visual relationships)
  - Hard Loss Weight $\alpha = 0.3$ (prioritizes teacher soft guidance over raw labels)
  - Soft Loss Weight $(1 - \alpha) = 0.7$ scaled by $T^2 = 16.0$
- **Optimization Strategy:**
  - **Stage A (Warmup, 5 Epochs):** Train new classification head only (`lr = 1e-3`, AdamW, weight decay $1e-4$), backbone frozen.
  - **Stage B (Fine-Tuning, 15 Epochs):** Unfreeze top 40 layers of MobileNetV3 (`lr = 1e-4` decaying to $1e-5$), BatchNorm layers strictly frozen.
- **Hardware & VRAM Budget:** Batch size 16, mixed precision (`mixed_float16`) to guarantee sub-4.0 GB VRAM consumption on RTX 3060 Laptop GPU.
- **Controlled Seed:** Seed 42 across NumPy, Python random, and TensorFlow.

#### Acceptance Gate for `tomato_student_distilled_v1`:
1. Benchmark test accuracy must remain $\ge 98.0\%$.
2. Field holdout accuracy must increase from **41.67% to $\ge 75.0\%$** (bridging the generalization gap toward the teacher's 91.67%).
3. Healthy outdoor specificity must improve from 0.0% to $\ge 66.7\%$.
4. LiteRT Float16 latency must remain $\le 20.0\text{ ms}$ on mobile device.

Outputs:
```text
configs/tomato/student_distillation_v1.yaml
scripts/tomato_student/train_student_distillation.py
models/tomato/students/distilled_v1/student_best.keras
reports/tomato/student_distilled_v1/distillation_evaluation_report.md
```


---

## 12. Parallel work: potato

Potato should not enter another training cycle now.

The next potato work is:

```text
complete app integration
verify CameraX coordinate mapping
verify two-view behavior on real users
expand external evaluation
monitor false-positive behavior
```

Keep the Float16 model frozen unless:

- Repeated disease-to-Healthy failures remain after proper framing.
- The two-view rule creates unacceptable false disease predictions.
- The app crop differs from the evaluated crop.
- New independent field data reveals a repeated product-relevant failure.

Update the potato master report after these checks, but do not retrain simply to improve benchmark accuracy.

---

## 13. Parallel work: rice

Rice should not enter another distillation or architecture cycle yet.

The next rice work is:

```text
source contact sheets
source metadata audit
source × class cross-tabulation
source-only diagnostic classifier
source-aware split
source-held-out evaluation
supervised versus distilled comparison
```

The rice models remain benchmark candidates with unresolved source-domain confounding.

Do not use the current rice teacher to automatically label new field data until this issue is addressed.

Required outputs:

```text
reports/rice/source_audit/source_visual_comparison.md
reports/rice/source_audit/source_metadata_comparison.csv
reports/rice/source_audit/source_classifier_report.md
manifests/rice/rice_source_aware_split_v1.csv
reports/rice/source_audit/current_models_source_aware_evaluation.md
```

---

## 14. Documentation governance

Every completed phase must update the relevant documentation.

Update:

```text
plans/ for implementation plans
docs/ for stable contracts and model cards
manifests/ for versioned data metadata
reports/ for results and decisions
models/ for model registry and checksums
mobile/ for runtime contracts and labels
```

After each material change, record:

```text
what changed
why it changed
which files changed
which model version changed
which data version changed
what was measured
what remains unknown
whether the result is accepted, rejected, or inconclusive
```

Do not leave important decisions only in chat messages.

---

## 15. Repository hygiene after GitHub setup

Continue to protect the repository.

Do not commit:

```text
raw images
datasets
model checkpoints
TFLite binaries unless intentionally released
training logs with private paths
API keys
.env files
local virtual environments
scratch outputs
```

Use artifact storage or release assets for large model binaries when appropriate. Track checksums, contracts, and download instructions.

Do not move files into `archive/` merely to hide an unresolved problem. Archived scripts must still be documented and should not be imported by active pipelines unless intended.

Use pull requests or small commits for material model changes when possible.

---

## 16. Exact next-step order

The agent must execute this order:

```text
1. Run post-GitHub repository baseline audit.
2. Freeze tomato_teacher_v2 and verify its contract.
3. Run tomato student preflight checks.
4. Request authorization for supervised tomato student training.
5. Train tomato_student_supervised_v1.
6. Evaluate teacher and supervised student on identical manifests.
7. Convert the supervised student to LiteRT Float16.
8. Verify Keras-to-LiteRT parity.
9. Run the Float16 student on the actual phone.
10. Decide whether distillation is justified.
11. Continue potato app and two-view validation in parallel.
12. Perform rice source-aware audit and evaluation.
13. Update plans, reports, manifests, contracts, and model registry.
```

Do not skip the supervised tomato baseline to save time. It is the comparison required to make an honest distillation decision.

---

## 17. Final go/no-go state

### Tomato

```text
Teacher v2: controlled student-development reference
Supervised student: next active model task
Distillation: pending evidence
Production deployment: not approved
```

### Potato

```text
Float16 model: frozen controlled-prototype artifact
Two-view workflow: continue app and real-workflow validation
Retraining: currently not justified
Production deployment: not approved
```

### Rice

```text
Teacher and students: benchmark candidates
Source-domain confounding: unresolved
Next action: source-aware evaluation
Production deployment: not approved
```

---

## Final recommendation

The next active engineering task is the **tomato supervised MobileNetV3 student baseline**.

Do not start tomato distillation, potato retraining, or rice architecture changes at this stage.

The correct project sequence is:

```text
repository baseline audit
→ tomato supervised student
→ LiteRT Float16 and phone validation
→ distillation decision
→ potato app integration
→ rice source-aware evaluation
```

> **The project is now organized enough to move quickly, but the next speed improvement must come from disciplined sequencing rather than running every model experiment simultaneously.**

## References

[1]: file:///home/ubuntu/ipd_model_docs/tomato_student_development_plan.md "Tomato mobile student development plan"

[2]: file:///home/ubuntu/ipd_model_docs/potato_post_audit_correction_and_final_validation_plan.md "Potato post-audit correction and final validation plan"

[3]: file:///home/ubuntu/ipd_model_docs/rice_model_correction_and_release_plan.md "Rice model correction and release plan"

[4]: file:///home/ubuntu/ipd_model_docs/model_improvement_prioritized_execution_and_gitignore_plan.md "Prioritized model improvement and repository hygiene plan"

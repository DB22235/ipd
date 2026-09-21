
# IPD Model Improvement: Prioritized Execution Plan and Repository Hygiene

**Project:** IPD plant disease detection

**Crops:** Potato, rice, and tomato

**Purpose:** Improve existing models according to measured failure modes rather than blindly increasing benchmark accuracy.

**Primary rule:** The implementation agent must execute the highest-priority work before lower-priority experiments and must not launch expensive training without a documented reason and authorization.

**Author:** Manus AI

---

## 1. Executive decision

The project should not change every model at once. Each crop has a different dominant risk:

| Crop | Dominant current risk | Highest-value improvement |
|---|---|---|
| Potato | Small or early lesions are diluted in whole-leaf images | Improve capture workflow, close-up/reticle guidance, and two-view evaluation |
| Rice | Dataset source is strongly correlated with class | Source-aware evaluation and source-balanced data |
| Tomato | Field evidence is still small and the student has not yet been compared properly | Train and evaluate the supervised student, then expand independent field validation |

The implementation order is:

```text
repository and artifact safety
→ evaluation and contract verification
→ crop-specific highest-priority tests
→ confirmed failure collection
→ targeted data or workflow correction
→ controlled retraining
→ conversion and device validation
→ optional distillation or architecture experiments
```

Do not begin with a broad architecture search, blind epoch increase, or large hyperparameter sweep.

---

## 2. Non-negotiable engineering principles

### 2.1 Improve the failure mode, not the benchmark number

A change is an improvement only when it solves a documented problem without causing unacceptable regression elsewhere.

Every experiment must compare a frozen baseline against one controlled change. The report must state:

```text
baseline model
changed component
reason for the change
dataset version
split version
evaluation manifests
metrics before and after
new failures
final decision
```

### 2.2 Preserve all baseline artifacts

Never overwrite a released, benchmarked, or previously evaluated model.

Use versioned identifiers such as:

```text
potato_student_v1
rice_student_supervised_v1
rice_student_distilled_v1
tomato_teacher_v2
tomato_student_supervised_v1
```

A new experiment must receive a new model ID, configuration, output directory, and checksum.

### 2.3 Keep evaluation roles separate

Every image manifest must be assigned exactly one primary role:

```text
training
validation
locked_test
external_evaluation
field_holdout
smoke_test
```

The locked test and external field holdout must not be used for:

- Checkpoint selection.
- Threshold tuning.
- Architecture selection.
- Augmentation selection.
- Quantization calibration.
- Hard-negative selection based only on model errors.

### 2.4 Do not treat the teacher as ground truth automatically

Teacher predictions are pseudo-labels, not verified labels. Do not add teacher-labeled images to training unless their provenance and label quality are recorded and the data is reviewed according to the project’s labeling policy.

### 2.5 Expensive execution control

The agent may run small integrity checks and bounded evaluations automatically. It must request authorization before:

```text
full teacher retraining
full student retraining
multi-seed training
knowledge distillation
quantization-aware training
large hyperparameter sweeps
large-scale external evaluation
```

Before requesting authorization, the agent must report:

```text
exact command
purpose
model and data versions
estimated runtime
GPU/CPU requirement
RAM/VRAM estimate
output directory
checkpoint behavior
whether files will be overwritten
```

### 2.6 Group-disjoint partitioning and perceptual hash integrity

Foliage image collections frequently contain burst-shot sequences of the same leaf photographed from slightly different angles or lighting conditions. Naive random train/validation/test splitting places near-duplicate burst shots across both training and test partitions, artificially inflating initial accuracy (often jumping to 87%+ on epoch 1) while failing to learn true foliar disease morphology.

Every dataset partition must be generated using **group-disjoint perceptual hashing** (`dHash` / `pHash` with Hamming distance threshold $\le 10$):
- All images belonging to the same burst sequence (family ID) must be atomically locked into exactly one partition.
- Cross-partition duplicate leakage must be strictly $0.0\%$.
- The split manifest must explicitly record `family_id` and hash signatures for every sample.

### 2.7 Mobile quantization standards and delegate realities (Float16 vs INT8)

Quantization strategy must be governed by measured hardware execution performance and classification decision boundary stability, not merely file size reduction on disk:

1. **Float16 LiteRT as the Primary Mobile Standard:**
   - Float16 quantization yields ~50% binary size reduction (~5.7–5.8 MB for MobileNetV3-Large).
   - Executes with full hardware SIMD acceleration on mobile CPU/NNAPI delegates (e.g., TFLite `XNNPACK` delegate) at sub-5 ms latency (>200 FPS).
   - Preserves $\ge 99.5\%$ categorical decision agreement with unquantized Keras FP32, with maximum probability divergence $< 0.02$.
2. **INT8 Quantization Caveat & Delegate Preparation Failure:**
   - Full post-training INT8 quantization on architectures with HardSwish and Squeeze-and-Excitation blocks (such as MobileNetV3) frequently causes acceleration delegate preparation failure on edge runtimes (e.g. `RuntimeError: failed to create XNNPACK runtimeNode number 124 (TfLiteXNNPackDelegate) failed to prepare`).
   - This failure forces the runtime to fall back to unvectorized CPU reference kernels (`BUILTIN_WITHOUT_DEFAULT_DELEGATES`), making INT8 execution **~70x slower** than Float16 (~230+ ms vs ~3.3 ms).
   - Furthermore, INT8 can induce severe decision boundary collapse (e.g., dropping class recall below 10%).
   - **Mandate:** INT8 is restricted to diagnostic research. Float16 is the authoritative production standard for mobile edge deployment.

---

## 3. Priority 0: protect the repository and artifacts

This priority must be completed before model improvement work.

### 3.1 Inventory the repository

The agent must inspect the repository without moving or deleting files initially.

Create an inventory containing:

```text
tracked files
untracked files
directories by purpose
large files
model files
dataset directories
reports
logs
caches
virtual environments
notebooks
temporary exports
secrets or credential-like files
```

The agent must not commit or upload any dataset while performing the inventory.

### 3.2 Create or repair `.gitignore`

Create a root-level `.gitignore` if one does not exist. If one exists, preserve useful project-specific rules and add missing protections without deleting required source files.

The `.gitignore` must exclude datasets, generated model artifacts, caches, logs, local environments, secrets, and temporary files. It must not exclude source code, small configuration templates, manifests that intentionally contain metadata only, or required documentation.

Use a structure equivalent to the following, adapting paths to the actual repository:

```gitignore
# Operating-system files
.DS_Store
Thumbs.db
desktop.ini

# Editors and IDEs
.vscode/
.idea/
*.swp
*.swo

# Python bytecode and caches
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/
.pytype/

# Local environments and package directories
.venv/
venv/
env/
envs/
virtualenv/
.conda/
conda-env/
node_modules/

# Local configuration and secrets
.env
.env.*
!.env.example
*.pem
*.key
*.p12
*.pfx
credentials.json
service-account*.json

# Logs and runtime output
*.log
logs/
log/
runtime_logs/
tmp/
temp/

# Notebooks and tool checkpoints
.ipynb_checkpoints/
.checkpoints/

# Datasets and raw media
# Keep data outside Git or use a separate data store.
data/
dataset/
datasets/
raw_data/
raw-dataset/
clean_dataset/
field_data/
field_test_images/
uploads/
*.zip
*.tar
*.tar.gz
*.7z
*.rar
*.mp4
*.mov
*.avi
*.webm
*.wav
*.mp3

# Image collections; do not commit raw image trees.
*.jpg
*.jpeg
*.png
*.webp
*.bmp
*.tif
*.tiff

# Generated model and conversion artifacts
*.keras
*.h5
*.hdf5
*.ckpt
*.weights.h5
*.tflite
*.onnx
*.pb
*.savedmodel/
checkpoints/
weights/
artifacts/
exports/
conversions/

# Generated experiment output
reports/generated/
experiment_outputs/
evaluation_outputs/
training_outputs/
run_outputs/
tensorboard/
wandb/
mlruns/
*.sqlite

# Build and distribution output
build/
dist/
*.egg-info/
*.whl

# Coverage and test output
.coverage
coverage.xml
htmlcov/
.pytest_cache/

# OS and temporary files
*.tmp
*.temp
*.bak
*.old
*~
```

### 3.3 Important `.gitignore` caution

The generic image rules above are appropriate only if the repository must not contain any images. If the project intentionally stores small diagrams, icons, or documentation assets, replace the global image rules with targeted dataset rules such as:

```gitignore
clean_dataset/
field_test_images/
datasets/
raw_data/
```

Do not blindly ignore every image if the application needs checked-in UI assets.

Likewise, do not ignore all `.json` files. Some model manifests and small configuration files are required for reproducibility. Ignore only secret or generated JSON files and review every existing JSON file before adding a pattern.

### 3.4 Preserve metadata without committing data

The repository may contain small metadata-only files such as:

```text
README.md
DATA_CARD.md
schema.json
label_map.json
split_manifest_template.csv
model_manifest.example.json
.env.example
```

These should remain tracked when they do not contain private image paths, credentials, or large data.

For large or private datasets, store only:

```text
data card
source description
license information
download instructions
dataset version
manifest schema
checksums if appropriate
```

Do not commit image bytes merely to make a manifest reproducible.

### 3.5 Verify Git behavior

After writing `.gitignore`, run:

```bash
git status --short --ignored
git check-ignore -v path/to/sample_dataset_file.jpg
```

Create a temporary test file only if necessary, then remove it. Verify that:

- Dataset files are ignored.
- Raw images are ignored.
- Generated checkpoints are ignored.
- Logs and caches are ignored.
- Source code remains visible and trackable.
- Documentation remains visible and trackable.
- Required manifests and configuration templates remain visible.
- No secret is tracked.

The agent must not run `git add -A`, `git commit`, `git push`, or delete files as part of this plan unless explicitly instructed.

---

## 4. Priority 1: verify contracts and baselines

Before changing a model, verify the currently registered baseline.

For every crop, check:

```text
model file exists
model checksum matches the registry
class order is correct
input shape is correct
input dtype is correct
preprocessing matches the model contract
labels match output indices
benchmark manifest is locked
```

Also check for report inconsistencies. For example:

- Different model checksums in different reports.
- Different confidence-margin thresholds.
- Different padding or normalization rules.
- Different class orders.
- Host latency reported as phone latency.

If a conflict is found, stop downstream comparison until one contract is declared authoritative.

Required artifact:

```text
reports/project/model_contract_and_baseline_integrity_report.md
```

---

## 5. Priority 2: potato improvement path

### 5.1 Current potato problem

The potato model is strong on the locked benchmark and has passed the reported physical-device benchmark. Its main observed weakness is small or early lesions occupying too little of a whole-leaf image.

A previous reticle simulation showed that lesion-focused framing changed a wrong Healthy prediction into the correct disease result. This indicates that input framing may be more valuable than immediate retraining.

### 5.2 First potato experiment: capture workflow

Evaluate:

```text
Mode A: unassisted whole-leaf capture
Mode B: close-up reticle-guided capture
```

Use an independent failure-focused set containing:

- Small lesions.
- Early lesions.
- Leaf-tip lesions.
- Petiole lesions.
- Mostly green leaves with one lesion.
- Different phones and backgrounds.

Report:

```text
accuracy
per-class recall
disease-to-Healthy error rate
confidence of errors
uncertain rate
retake rate
accepted coverage
latency for one view and two views
```

### 5.2.1 CameraX viewfinder reticle technical specification

To implement Mode B consistently on mobile devices without relying on user framing intuition, the mobile client must provide an explicit viewfinder targeting reticle:

1. **Reticle Geometry:** Render a high-contrast, semi-transparent square reticle occupying the central **50% width $\times$ 50% height** of the CameraX viewfinder preview.
2. **Standoff & Fill Guidance:** Prompt the grower: *"Center the suspicious leaf spot inside the box at 15–20 cm distance."* The target lesion plus immediately surrounding tissue must fill $\ge 40\%$ of the reticle area.
3. **Software Cropping Contract:** Prior to resizing, crop the image tensor strictly to the reticle coordinates.
4. **Letterbox Standard:** Scale the cropped sub-tensor to $300 \times 300$ using aspect-preserving letterboxing with neutral gray **`RGB(114, 114, 114)`** padding, avoiding any geometric distortion of concentric fungal rings.

### 5.3 Second potato experiment: two-view inference

Test:

```text
View 1: unassisted whole-leaf view (context & leaf architecture)
+ View 2: close-up reticle-guided view (focal lesion morphology)
```

#### Asymmetric Agronomic Safety Rule:
Foliar diagnosis must not use symmetric probability averaging. It must enforce the **Asymmetric Agronomic Safety Rule**:

```text
1. Agreement between View 1 and View 2 → Accept diagnosis with combined confidence.
2. View 1 (Whole-leaf) = Healthy AND View 2 (Close-up) = Disease (p >= 0.65, margin >= 0.25):
   → OVERRIDE Healthy; prioritize the Disease diagnosis.
   Reason: Global Average Pooling (GAP) mathematically dilutes small focal lesions (<5% area) 
   when surrounded by 95% green leaf tissue in View 1. View 2 provides direct optical resolution 
   on the fungal mycelium or concentric rings. A false negative (missed blight) causes field 
   defoliation, while inspecting a suspicious spot is agronomically safe.
3. View 1 (Whole-leaf) = Disease AND View 2 (Close-up) = Healthy:
   → Return "Uncertain / Recapture Needed"; prompt grower to center the diseased area.
4. One or both views fall below confidence/margin thresholds:
   → Return "Uncertain foliar pattern; inspect in indirect sunlight."
```

The policy must be evaluated on validation data first and then on an untouched external set.

Do not average probabilities without measuring whether probability averaging helps.

### 5.4 Third potato experiment: calibration and abstention

Use validation data to compare:

```text
current confidence and margin gates
temperature-scaled confidence
stricter small-lesion uncertainty rule
whole-leaf/close-up disagreement rule
```

Report the trade-off among:

```text
accepted coverage
selective accuracy
false rejection
high-confidence error rate
uncertain rate
```

### 5.5 Potato retraining gate

Do not retrain the potato model unless the failure-focused set shows repeated failures even after correct lesion framing and workflow guidance.

If retraining is justified, add confirmed difficult lesions and preserve the current model as the baseline. Do not change architecture and data strategy simultaneously in the first corrective run.

### 5.6 Potato deployment boundary

The app must use explicit Potato mode. The potato classifier must not be treated as a universal crop detector. Rice or tomato leaves must not be silently classified as Healthy potato leaves.

---

## 6. Priority 3: rice improvement path

### 6.1 Current rice problem

The rice dataset contains strong source/class correlation. Healthy images and disease images come from materially different sources. Therefore, very high benchmark accuracy may overestimate disease generalization.

This is a data and evaluation problem before it is an architecture problem.

### 6.2 First rice experiment: source audit

Create:

```text
source contact sheets
source metadata comparison
source × class cross-tabulation
source-only diagnostic classifier
```

Measure differences in:

```text
background
lighting
camera style
resolution
aspect ratio
compression
leaf color
lesion scale
crop framing
```

### 6.3 Second rice experiment: source-aware evaluation

Create a split where at least one source or capture domain is held out, while documenting class coverage.

If the held-out source does not contain every class, do not report a misleading full-class accuracy. Report only valid per-class results and mark full generalization as untested.

Evaluate:

```text
teacher
supervised student
 distilled student
```

on exactly the same source-aware manifests.

### 6.4 Third rice improvement: source-balanced data

The preferred future structure is:

```text
Source A: Healthy, Blast, Brown Spot, Blight
Source B: Healthy, Blast, Brown Spot, Blight
Source C: Healthy, Blast, Brown Spot, Blight
```

If new data is unavailable, do not invent labels. Use existing data as source-specific evidence and state the limitation.

### 6.5 Fourth rice experiment: training strategy

Only after measuring source shift, compare one controlled strategy at a time:

```text
standard sampling
source-balanced batches
class-balanced batches
source-aware weighting
GroupDRO or domain-aware method, if justified
```

Select using source-held-out validation, not the original source-confounded benchmark.

### 6.6 Rice distillation boundary

Do not use distillation to hide source confounding. Distillation is justified only after source-aware evaluation identifies a mobile-specific problem that a validated teacher can solve.

Do not use current teacher predictions as automatic labels for new field data.

---

## 7. Priority 4: tomato improvement path

### 7.1 Current tomato status and empirical baseline

The tomato pipeline has achieved a major milestone:
1. **Teacher v2 Frozen Reference:** `tomato_teacher_v2` (`models/tomato/teachers/v2/teacher_best.keras`, SHA-256: `a7ae01a2...`, EfficientNetB3, $300 \times 300$) was retrained on group-disjoint Dataset v2 with 0% pHash duplicate leakage. Validated at **99.18% validation accuracy** and permanently frozen.
2. **Supervised Student Baseline Certified:** `tomato_student_supervised_v1` (`models/tomato/students/supervised_v1/student_best.keras`, MobileNetV3-Large, $300 \times 300$, 4.2M params) has been trained using two-phase AdamW with frozen BatchNorm:
   - **Benchmark Test Accuracy:** **99.85%** (1,344 / 1,346) on the locked, group-disjoint test set.
   - **Per-Class Sensitivity:** Early Blight **100.0%**, Healthy **100.0%**, Late Blight **99.64%**.
   - **LiteRT Float16 Conversion:** Packaged in `mobile/tomato/tomato_student_float16.tflite` (~5.77 MB).
   - **Format Parity:** **100.00% agreement** (162 / 162 samples) with Keras FP32; max probability delta = 0.01308; 100% (12/12) field holdout agreement.
   - **Device Latency:** **3.36 ms (297 FPS)** at 4 threads under XNNPACK SIMD acceleration.

With the supervised baseline certified and distillation safely deferred, the remaining tomato focus is expanding field validation and hard-negative robustness.

### 7.2 Primary tomato priority: expanded field validation and hard negatives

Rather than immediate distillation, the immediate priority for Tomato is field verification:
1. Expand external field testing beyond the initial 12-image regression set.
2. Collect and review difficult non-standard presentations (pale healthy leaves, studio white backings, dusty leaves).
3. Verify CameraX reticle targeting and Stage 3 margin triage on ambiguous petiole lesions (e.g. `field_05`).

### 7.3 Second tomato experiment: student evaluation

Compare teacher v2 and supervised student on the same:

```text
locked benchmark
12-image field regression set
expanded external field set when available
hard-negative set
difficult early-lesion set
ambiguous set
background perturbation set
```

Report:

```text
accuracy
macro-F1
balanced accuracy
per-class recall
Healthy false-positive rate
disease-to-Healthy errors
ECE
Brier score
selective accuracy
coverage
high-confidence errors
```

### 7.4 Third tomato improvement: hard-negative mining

Collect and independently review:

- Pale and lime-green Healthy leaves.
- White-background Healthy leaves.
- Direct-flash and sunlight examples.
- Natural yellowing.
- Mechanical or insect damage.
- Soil and dust contamination.
- Early and subtle disease.
- Different cameras and cultivars.

Add reviewed failures only through a new dataset version. Do not continuously add every model mistake to training because this can cause evaluation contamination.

### 7.5 Fourth tomato experiment: calibration and abstention

Use validation data to test:

```text
confidence threshold
margin threshold
temperature scaling
uncertain state
quality gates
```

The system must return `uncertain` when evidence is weak. It must not force ambiguous images into Healthy.

### 7.6 Tomato distillation boundary

Compare the supervised student first. Run one controlled distilled student experiment only if the supervised student has a defined weakness or the mobile budget requires additional compression.

Accept distillation only when it produces a measured advantage on relevant external or difficult-image metrics, size, latency, memory, or quantization.

---

## 8. Priority 5: shared improvements after crop-specific tests

### 8.1 Calibration

Use validation data to fit temperature scaling when overconfidence is observed. Report:

```text
ECE before and after
Brier score before and after
accuracy before and after
coverage before and after
```

Do not fit calibration on the locked test set.

### 8.2 Abstention

Separate the classifier from the decision layer. Report:

```text
raw classifier accuracy
accepted-input accuracy
selective accuracy
coverage
uncertain rate
unsupported-input rejection
false rejection
high-confidence error rate
```

### 8.2.1 Pre-inference edge quality gates (Tier 1 fast-fail)

Before executing the neural network on mobile devices, the application must execute lightweight, deterministic OpenCV/RenderScript quality gates in $< 2.0$ ms to reject bad captures immediately:

1. **Laplacian Blur Variance Gate:**
   - Compute grayscale variance: $\sigma^2(\nabla^2 I_{\text{gray}})$.
   - Threshold: $\text{Variance} \ge 100.0$.
   - Failure Action: Reject immediately; prompt user: *"Camera out of focus. Hold phone steady and tap to focus."*
2. **Botanical Foliage Coverage Gate (HSV):**
   - Convert frame to HSV color space.
   - Filter green foliage bounds: Hue $\in [20, 95]$, Saturation $\ge 30$, Value $\ge 30$.
   - Threshold: $\frac{\text{Foliage Pixels}}{\text{Total Frame Pixels}} \ge 15.0\%$.
   - Failure Action: Reject immediately; prompt user: *"No crop leaf detected. Center a leaf inside the frame."* (Prevents scanning soil, boots, fingers, or mulch).
3. **Specular Flash Overexposure Gate:**
   - Detect glare/flash burnout on waxy leaves: Saturated pixels ($V > 250$ with $S < 20$) must occupy $< 15.0\%$ of the leaf area.
   - Failure Action: Reject capture; prompt user: *"Glare detected. Turn off flash and use diffused outdoor light."*

### 8.2.2 Stage 3 tri-state decision layer and margin triage

The app must never force ambiguous or borderline symptoms into a false high-certainty diagnosis. Softmax outputs must pass through a strict tri-state triage gate:

Let $p_{(1)}$ be the highest class probability, and $p_{(2)}$ be the second-highest class probability:

$$\text{Confidence} = p_{(1)}, \quad \text{Margin } \Delta p = p_{(1)} - p_{(2)}$$

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             TRI-STATE DECISION FLOW                              │
├───────────────────┬───────────────────────────────┬──────────────────────────────┤
│ STATE             │ MATHEMATICAL CONDITION        │ USER-FACING ACTION           │
├───────────────────┼───────────────────────────────┼──────────────────────────────┤
│ 1. Accepted       │ p(1) >= 0.70 AND              │ Display diagnosed disease,   │
│    Diagnosis      │ delta_p >= 0.30               │ confidence bar, and agronomy │
│                   │                               │ management guide.            │
├───────────────────┼───────────────────────────────┼──────────────────────────────┤
│ 2. Borderline /   │ p(1) < 0.70 OR                │ Display amber advisory:      │
│    Uncertain      │ delta_p < 0.30                │ "Ambiguous foliar pattern.   │
│    Triage         │                               │ Retake photo in diffused     │
│                   │                               │ light or consult agronomist."│
├───────────────────┼───────────────────────────────┼──────────────────────────────┤
│ 3. Unsupported    │ Tripped Blur, Foliage, or     │ Display red prompt:          │
│    Input          │ Flash Quality Gate            │ Fast-fail before inference;  │
│                   │                               │ instructions on framing.     │
└───────────────────┴───────────────────────────────┴──────────────────────────────┘
```

**Clinical Justification:** This gate prevents false certainty on difficult real-world edge cases (e.g. Tomato `field_05` petiole collapse / stem lesion), successfully routing ambiguous visual evidence to human inspection rather than returning an erroneous classification.

### 8.3 Test-time augmentation

Consider only after baseline behavior is known. Test mild flip, crop, and brightness variants. Measure the extra latency and disagreement rate on the actual device before adopting it.

### 8.4 Ensembles

Consider an ensemble only if the device budget permits it and a single model cannot meet the requirement. Compare the accuracy benefit against model size, memory, latency, battery use, and update complexity.

### 8.5 Architecture changes

Architecture experiments such as MobileNetV2, MobileNetV3-Small, EfficientNetB2, or attention modules are lower priority. Run them only when:

```text
the current failure mode is identified

data and evaluation are trustworthy

the current architecture fails a product requirement

the new architecture has a measurable hypothesis
```

---

## 9. Experiments that are not allowed as default fixes

The agent must not perform these merely to produce a higher score:

```text
blindly increasing epochs
changing many hyperparameters at once
large architecture searches
changing thresholds using the locked test set
training on teacher pseudo-labels without review
adding synthetic images without label validation
distilling an unvalidated teacher
using INT8 only because the binary is smaller
removing difficult images from evaluation because they lower accuracy
```

If one of these is proposed, the agent must explain the specific failure mechanism and define an independent acceptance test first.

---

## 10. Required documentation updates

After every material experiment, update the relevant documentation folder.

At minimum, update:

```text
dataset version
split version
model registry
model manifest
training configuration
model checksum
evaluation report
known limitations
go/no-go decision
```

The report must state whether the change is:

```text
accepted
rejected
inconclusive
pending external validation
```

Do not report an experiment as successful merely because the process exited with code 0.

---

## 11. Required repository structure

Use a structure similar to:

```text
ipd/
├── app/
├── configs/
├── docs/
├── manifests/
│   ├── potato/
│   ├── rice/
│   └── tomato/
├── models/
│   ├── potato/
│   ├── rice/
│   └── tomato/
├── reports/
├── scripts/
│   ├── potato/
│   ├── rice/
│   └── tomato/
├── tests/
├── tools/
├── .gitignore
├── .env.example
└── README.md
```

Large data and generated artifacts should live outside Git, in a documented data or artifact store. The repository should contain code, documentation, schemas, lightweight manifests, and reproducibility metadata.

---

## 12. Final execution order for the agent

The agent must follow this order unless a documented blocker changes it:

```text
Priority 0: inventory repository and create/repair .gitignore
Priority 1: verify checksums, class orders, preprocessing, and baseline contracts
Priority 2: potato capture workflow, two-view inference, and small-lesion evaluation
Priority 3: rice source audit, source-aware split, and source-held-out evaluation
Priority 4: tomato supervised MobileNetV3 student baseline
Priority 5: shared calibration, abstention, and device checks
Priority 6: targeted data correction and controlled retraining only where justified
Priority 7: optional distillation
Priority 8: architecture experiments only if a product requirement remains unmet
```

The agent must stop and report if a higher-priority integrity issue is discovered.

---

## 13. Final go/no-go criteria

### Continue the current plan

Continue when:

- The baseline artifact is reproducible.
- The data role is clear.
- The evaluation manifest is independent.
- The experiment has a measurable hypothesis.
- The output will not overwrite existing artifacts.

### Pause and correct data or contracts

Pause when:

- Checksums conflict.
- Class order conflicts.
- Preprocessing differs between training and app.
- Test contamination is found.
- Labels are unreliable.
- Source and class are perfectly correlated.
- A report claims phone validation from host measurements.

### Retrain

Retrain only when:

- A confirmed failure repeats.
- The failure is relevant to the intended product workflow.
- The data correction or training change addresses that failure.
- A frozen baseline and independent evaluation set exist.

### Distill

Distill only when:

- The teacher is validated enough for the stated experiment.
- The supervised student baseline has been measured.
- There is a defined mobile or robustness need.
- Success criteria are written before training.

---

## Final recommendation

The highest-value improvements are different by crop:

```text
Potato: improve image framing and small-lesion capture before retraining.
Rice: fix source-domain evaluation and source diversity before trusting benchmark scores.
Tomato: build and evaluate the supervised student while expanding field validation.
All crops: verify contracts, calibrate uncertainty, preserve baselines, and keep datasets out of Git.
```

The agent should work from highest priority to lowest priority. It should prefer a small, measurable correction over a large speculative training run.

> **A higher score on a contaminated or mismatched evaluation set is not an improvement. An improvement is a measurable reduction in the intended failure mode without unacceptable regression.**

## References

[1]: file:///home/ubuntu/ipd_model_docs/potato_all_remaining_model_tests.md "Potato remaining model tests and release gates"

[2]: file:///home/ubuntu/ipd_model_docs/tomato_student_development_plan.md "Tomato mobile student development plan"

[3]: file:///home/ubuntu/ipd_model_docs/tomato_teacher_v2_retraining_plan.md "Tomato teacher v2 retraining and validation plan"

[4]: file:///home/ubuntu/ipd_model_docs/rice_model_correction_and_release_plan.md "Rice model correction and release plan"

[5]: file:///home/ubuntu/ipd_model_docs/potato_model_next_steps_plan.md "Potato model next-steps plan"

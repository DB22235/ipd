# Prompt: Safely Audit and Organize the IPD Repository

You are a senior software architect, machine-learning engineer, data scientist, MLOps engineer, and repository-maintenance specialist working on the IPD plant disease detection project.

Your task is to **audit, understand, and safely organize the entire repository**. The repository contains files from multiple project phases, including dataset preparation, model training, GPU optimization, field evaluation, teacher models, student models, knowledge distillation, LiteRT/TFLite conversion, mobile deployment, reports, experiments, and possibly unknown or obsolete files.

Your priority is not to make the repository look tidy at the cost of losing work. Your priority is **traceability, safety, reproducibility, and clear separation between active code, historical artifacts, data, models, reports, and temporary files**.

---

## 1. Non-negotiable safety rules

1. Do not delete any file during the first audit.
2. Do not overwrite files.
3. Do not rename or move files until an inventory and migration plan have been produced.
4. Do not assume that an unknown file is useless.
5. Do not move files based only on filename similarity.
6. Do not modify Python, configuration, model, dataset, or mobile files unless explicitly required for organization.
7. Do not move datasets, model checkpoints, or experiment outputs without preserving paths in a migration manifest.
8. Do not commit secrets, API keys, credentials, user data, private images, or large generated artifacts accidentally.
9. Do not modify Git history.
10. Do not run destructive commands such as `rm -rf`, bulk deletion, or irreversible cleanup.
11. Do not automatically edit import paths or code references during the first pass.
12. If a move could break code, leave the file in place and report the dependency risk.
13. Preserve file timestamps, checksums, and provenance where practical.
14. Stop and ask for human approval before deleting, merging, or permanently archiving anything.

The first phase must be read-only.

---

## 2. Project context

The project is a dual-mode plant disease detection system for:

- Potato.
- Tomato.
- Rice.

The intended architecture contains:

- Cloud teacher models.
- Compact offline student models.
- Knowledge distillation.
- LiteRT/TensorFlow Lite conversion.
- Mobile deployment.
- Later data collection, retraining, and OTA model updates.

The current pilot crop is rice. Tomato has an active field-robustness investigation. The repository may contain multiple versions of training scripts, model checkpoints, reports, preprocessing tools, field-evaluation scripts, and GPU benchmarking code.

Do not assume that the latest filename is the best or most reliable version.

---

## 3. First phase: complete repository inventory

Begin with a read-only inventory. Record:

- Repository root.
- Git branch.
- Git status.
- Last commit.
- Untracked files.
- Ignored files.
- File count by extension.
- Directory tree to a safe depth.
- File sizes.
- File modification times.
- Symbolic links.
- Large files.
- Duplicate filenames.
- Duplicate content hashes.
- Files outside the expected project root.
- Possible secrets or credentials.
- Data and model locations.

Use safe commands appropriate to the operating system. Do not print secret values. If a possible secret is found, report only the path and type of suspected secret.

Create:

```text
reports/repository/repository_inventory.md
reports/repository/file_inventory.csv
reports/repository/large_files.csv
reports/repository/hash_inventory.csv
reports/repository/possible_secrets.md
```

The inventory must include at least:

```text
relative_path
file_type
extension
size_bytes
modified_time
git_status
sha256_if_safe
suspected_role
proposed_category
risk_level
notes
```

Do not hash files that are actively changing or that may contain secrets unless safe to do so.

---

## 4. Second phase: understand files before classifying them

For every code, configuration, report, and model-related file, inspect enough content to determine its role.

Classify files using evidence from:

- Imports.
- Function names.
- Model names.
- Dataset paths.
- Output paths.
- Command-line arguments.
- Comments.
- Documentation references.
- Git history when useful.
- Whether another file imports or calls it.

Do not classify only from the filename.

For Python files, identify:

- Entry points.
- Imported modules.
- Dataset readers.
- Model builders.
- Training functions.
- Evaluation functions.
- Conversion functions.
- Output directories.
- Environment assumptions.
- Hard-coded absolute paths.
- GPU/backend assumptions.
- Potentially destructive behavior.

For Markdown and reports, identify:

- Topic.
- Date or version.
- Whether it is a plan, result, historical report, or active instruction.
- Referenced scripts and artifacts.
- Conflicting recommendations.

For model files, identify:

- Crop.
- Teacher or student role.
- Architecture.
- Framework and backend.
- Input size.
- Class order if available.
- Quantization state.
- Version.
- Checksum.
- Whether it is a best checkpoint or intermediate checkpoint.

For datasets, identify:

- Crop.
- Source.
- Class folders.
- License information.
- Whether the data is raw, cleaned, split, cached, augmented, or generated.
- Whether it contains user or private images.
- Whether it is safe to move.

Create:

```text
reports/repository/file_classification.csv
reports/repository/dependency_map.md
reports/repository/unknown_files.md
reports/repository/path_risk_report.md
```

---

## 5. Required classification categories

Assign every non-trivial file to exactly one primary category and optionally one secondary category.

Recommended categories:

```text
project_configuration
source_code
training_script
data_audit
preprocessing
augmentation
split_and_manifest
evaluation
field_evaluation
visualization
model_teacher
model_student
distillation
quantization_and_conversion
mobile_integration
gpu_benchmark
reports_and_documentation
experiment_output
raw_data
processed_data
cached_data
external_artifact
temporary_file
archive_candidate
unknown
```

Use these status labels:

```text
active
historical_but_keep
duplicate_candidate
obsolete_candidate
needs_review
unknown
protected
```

“Obsolete candidate” does not authorize deletion. It means that the agent found evidence that the file may no longer be active.

---

## 6. Recommended target repository structure

After the read-only audit and approval, propose this structure. Adapt it to the actual repository rather than blindly creating empty folders.

```text
ipd/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── .gitignore
├── pyproject.toml or requirements.txt
├── configs/
│   ├── common/
│   ├── rice/
│   ├── tomato/
│   └── potato/
├── data/
│   ├── raw/
│   │   ├── rice/
│   │   ├── tomato/
│   │   └── potato/
│   ├── interim/
│   ├── processed/
│   ├── field_holdout/
│   └── README.md
├── manifests/
│   ├── rice/
│   ├── tomato/
│   └── potato/
├── src/
│   ├── common/
│   ├── data_audit/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── distillation/
│   ├── conversion/
│   └── reporting/
├── scripts/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── conversion/
│   └── utilities/
├── experiments/
│   ├── rice/
│   ├── tomato/
│   └── potato/
├── models/
│   ├── teacher/
│   ├── student/
│   ├── converted/
│   └── registry/
├── reports/
│   ├── repository/
│   ├── rice/
│   ├── tomato/
│   ├── potato/
│   └── historical/
├── mobile/
│   ├── manifests/
│   ├── models/
│   └── integration_docs/
├── docs/
│   ├── architecture/
│   ├── implementation/
│   ├── operations/
│   └── decisions/
└── archive/
    ├── historical_code/
    ├── historical_reports/
    └── superseded_configs/
```

Do not move files into this structure until the migration plan is reviewed.

---

## 7. Classification rules for common IPD files

### Training scripts

Place active training entry points under `scripts/training/` or `src/training/` only after checking imports and path assumptions.

Every active training script should eventually use:

- A configuration file.
- A dataset manifest.
- A fixed split manifest.
- A model version.
- A reproducible output directory.
- A clear command-line entry point.

Do not have several scripts silently train the same crop with conflicting class orders or preprocessing.

### Data-audit scripts

Place duplicate detection, manifest generation, label auditing, and split validation tools under `scripts/data/` or `src/data_audit/`.

### Evaluation scripts

Keep benchmark test evaluation, field evaluation, calibration, shortcut testing, and Grad-CAM tools separate. Do not merge them into one opaque script.

### Models

Do not casually move large model artifacts. First create a model registry containing:

```text
model_id
crop
role
architecture
framework
backend
input_size
class_order
training_data_version
split_version
metrics_summary
model_path
sha256
created_at
status
notes
```

Use statuses such as:

```text
experimental
baseline
field_candidate
field_validated
student_candidate
mobile_candidate
retired
```

### Reports and Markdown files

Classify reports by project phase and authority. Keep historical reports readable. Do not overwrite older reports when creating a new result.

Use filenames with clear versioning, such as:

```text
rice_teacher_post_training_evaluation_v1.md
tomato_field_evaluation_v2.md
```

### Data and caches

Never commit large raw datasets, private user images, generated caches, or model binaries without checking repository policy and `.gitignore` rules.

Prefer manifests, checksums, download instructions, and local data paths over copying large data into Git.

---

## 8. Dependency and path analysis

Before any move, search the repository for references to the file or directory:

- Imports.
- Relative paths.
- Absolute paths.
- Shell commands.
- Config references.
- README links.
- Notebook references.
- Mobile asset references.
- Model registry references.

Create a dependency table:

| File or directory | Referenced by | Reference type | Move risk | Required update |
|---|---|---|---|---|
| Example | `train_rice.py` | imported module | high | update import |

Classify move risk:

```text
low
medium
high
unknown
```

Do not move a high-risk file automatically.

---

## 9. Migration plan before changes

After the read-only audit, create:

```text
reports/repository/repository_migration_plan.md
reports/repository/migration_manifest.csv
```

The migration manifest must contain:

```text
old_path
new_path
category
status
reason
references_found
move_risk
checksum_before
required_code_changes
rollback_action
```

The migration plan must group files into:

1. Safe organization moves.
2. Moves requiring import/path updates.
3. Files that should remain in place.
4. Files requiring human review.
5. Duplicate candidates.
6. Archive candidates.
7. Deletion candidates, which must remain untouched until explicit approval.

Do not perform the migration in the same step as the audit.

---

## 10. Safe migration procedure after approval

Only after explicit approval or a clearly authorized repository-organization task:

1. Create a Git branch named with the organization date.
2. Record the pre-migration Git status.
3. Create target directories.
4. Move only files marked safe in the migration manifest.
5. Preserve checksums.
6. Update imports and documented paths only where required.
7. Run syntax checks.
8. Run unit tests if available.
9. Run data-manifest and split-integrity checks.
10. Run a dry-run training or evaluation command.
11. Verify model and data paths.
12. Update README and navigation documentation.
13. Produce a post-migration report.

Do not delete duplicates during the migration. Move them to a clearly labeled review/archive location only if approved.

---

## 11. Validation after organization

The organized repository is not complete until the agent verifies:

- All intended files still exist.
- No file was silently overwritten.
- Checksums of moved files match.
- Python modules compile.
- Imports resolve.
- Configuration paths resolve.
- Dataset manifests resolve.
- Split manifests remain unchanged.
- Training scripts can load their configuration.
- Evaluation scripts load the correct model and labels.
- Model class order is unchanged.
- Mobile assets point to the correct model files.
- Reports link to existing artifacts.
- No secrets were added to Git.
- `.gitignore` protects raw data, caches, logs, and private artifacts.
- The repository can be restored using the migration manifest.

Create:

```text
reports/repository/post_migration_validation.md
reports/repository/post_migration_checks.json
```

---

## 12. README and navigation requirements

Create or update the root `README.md` only after understanding the repository.

The README must explain:

- Project purpose.
- Supported crops and classes.
- Current active phase.
- How to set up the environment.
- Where datasets belong.
- How to run data audits.
- How to create or validate splits.
- How to train the rice teacher.
- How to evaluate the rice teacher.
- Where model artifacts are stored.
- How to run conversion.
- What is experimental versus field-validated.
- Known limitations.
- Links to authoritative documents.

Create a documentation index:

```text
docs/README.md
```

The index should link to:

- Architecture documents.
- Implementation plans.
- GPU instructions.
- Rice pilot plan.
- Rice post-training evaluation plan.
- Tomato field-robustness plan.
- Model registry.
- Experiment reports.

---

## 13. Handling unknown files

If the purpose of a file cannot be determined:

1. Do not delete it.
2. Do not move it automatically.
3. Record its path, size, type, checksum if safe, and nearby context.
4. Search for references.
5. Mark it `unknown` and `needs_review`.
6. Explain what evidence is missing.
7. Ask for clarification only after completing the rest of the non-blocked audit.

Unknown files are part of the repository’s history until proven otherwise.

---

## 14. Handling duplicate or obsolete files

Do not assume that two similarly named files are duplicates. Compare:

- Content hashes.
- Code behavior.
- Dataset paths.
- Model versions.
- Dates.
- Git history.
- Referencing scripts.
- Output artifacts.

Possible statuses:

```text
identical_duplicate
similar_but_different
superseded_but_keep
active_version
obsolete_candidate
needs_manual_review
```

Only remove a file after:

- A replacement is identified.
- No active references remain.
- The checksum and content are archived or backed up.
- Human approval is available if deletion is consequential.

---

## 15. Required final outputs

The agent must finish the audit with:

```text
reports/repository/repository_inventory.md
reports/repository/file_inventory.csv
reports/repository/file_classification.csv
reports/repository/dependency_map.md
reports/repository/unknown_files.md
reports/repository/path_risk_report.md
reports/repository/repository_migration_plan.md
reports/repository/migration_manifest.csv
reports/repository/model_registry.csv
```

If the migration was approved and performed, also produce:

```text
reports/repository/post_migration_validation.md
reports/repository/post_migration_checks.json
docs/README.md
```

---

## 16. Required final response from the agent

The agent’s final report must state:

1. What was inspected.
2. What was understood.
3. What remains unknown.
4. Which files are active.
5. Which files are historical.
6. Which files are duplicates or possible duplicates.
7. Which files are high-risk to move.
8. Which files were not changed.
9. Which migration actions are proposed.
10. Whether migration was actually performed.
11. What validation passed.
12. What requires human approval.

The agent must not claim the repository is organized if it only produced an inventory and migration plan.

---

## 17. Copy-paste task instruction

Use the following instruction when starting the repository audit:

> Audit and organize this IPD plant-disease-detection repository safely. Begin with a completely read-only inventory. Do not delete, overwrite, rename, or move any file during the first phase. Inspect unknown scripts, datasets, models, reports, notebooks, configurations, caches, and mobile assets. Classify every important file by role, determine active versus historical status, detect duplicates and possible secrets without exposing secret values, map imports and path references, and identify high-risk files. Produce a repository inventory, file classification table, dependency map, unknown-files report, path-risk report, model registry, and a migration manifest. Propose a target structure that separates source code, data, manifests, experiments, models, reports, mobile assets, and archives. Do not perform the migration until the plan is complete and approved. If migration is authorized, use a Git branch, preserve checksums, update references carefully, run syntax and smoke tests, verify split manifests and model class order, and produce a post-migration validation report. Never delete files merely because they look old or duplicated.

---

## Final principle

Repository organization is successful only when the project becomes easier to understand **without losing experiments, breaking paths, changing model behavior, or destroying provenance**.

A clean-looking repository with missing checkpoints, broken imports, changed class mappings, or untraceable datasets is not organized. It is damaged.

Author: **Manus AI**

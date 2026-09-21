# Potato GitHub Pre-Push Safety and Repository Hygiene Plan

**Project:** IPD plant disease detection

**Scope:** Potato prototype repository and shared project files

**Purpose:** Determine whether the repository is safe to commit and push to GitHub without exposing datasets, model binaries, secrets, private paths, unnecessary generated files, or misleading release claims.

**Current recommendation:** The report indicates that a push may be appropriate, but the repository is not considered safe merely because `.gitignore` exists. The actual Git index, staged diff, file sizes, secrets, and remote must be checked immediately before pushing.

**Author:** Manus AI

---

## 1. Executive decision

You may push the potato project to GitHub **only after the pre-push checks in this document pass**.

The latest report provides evidence that the intended tracked content is lightweight and that datasets and model binaries are intended to be ignored. However, the report alone cannot prove that the current Git index is clean. A file can remain tracked even after it is added to `.gitignore`, and a project tree listing does not show whether a file is staged or already present in Git history.

The correct decision is:

```text
Push status: conditionally approved
Blind `git add .` and `git push`: not approved
Safe staged-diff review: required
```

Do not push until the commands below have been run from the repository root and their output has been inspected.

---

## 2. What is safe to push in principle

The following categories are generally appropriate for GitHub when they contain no secrets or private data:

```text
source code
unit tests
small configuration templates
model manifests without private absolute paths
class-label definitions
lightweight evaluation manifests
Markdown documentation
schemas
README files
small audit summaries
reproducibility instructions
.env.example with placeholder values only
```

The following categories should normally remain outside GitHub:

```text
raw datasets
image collections
private field photographs
trained model binaries
Keras checkpoints
LiteRT/TFLite binaries
training logs containing private paths
large generated reports or archives
virtual environments
caches
secrets
API keys
personal credentials
local absolute-path dumps
```

A model checksum and a model manifest may be tracked. The model binary itself should be distributed through a suitable artifact store or release process unless the project explicitly chooses another approach and confirms repository size and licensing requirements.

---

## 3. Important correction to the report’s push instructions

The report recommends:

```bash
git add .
git status
git commit -m "..."
git push origin main
```

Do not use this command sequence blindly. `git add .` stages every non-ignored file, including files that may be unnecessary, private, or incorrectly classified as safe.

Use the safer sequence in this document:

```bash
git status --short
git diff --stat
git diff --name-only
git add -p

git diff --cached --stat
git diff --cached --name-status
git diff --cached
```

Only push after the staged diff has been reviewed.

---

## 4. Step 0: confirm the repository root

Run:

```bash
git rev-parse --show-toplevel
git status --short --branch
```

Confirm that the reported root is the intended IPD repository and not a parent directory containing unrelated projects.

The current branch and remote must be recorded:

```bash
git branch --show-current
git remote -v
```

Do not push to an unexpected remote or branch.

---

## 5. Step 1: inspect the existing `.gitignore`

Open the root `.gitignore` and verify that it excludes, as appropriate:

```text
raw datasets
image files
archives
model checkpoints
Keras files
LiteRT/TFLite files
ONNX files
virtual environments
Python caches
training logs
TensorBoard or experiment runs
temporary files
secret files
local environment files
```

The `.gitignore` must not accidentally hide required source code or documentation.

Be especially careful with broad rules such as:

```gitignore
*.json
*.csv
*.md
*.py
```

These broad patterns can hide important project files and should not be used without a specific reason.

The repository should usually track:

```text
model_manifest.json
label_map.json
preprocessing_contract.json
lightweight metadata CSV files
README.md
```

provided they contain no secrets, private absolute paths, or raw personal data.

---

## 6. Step 2: check ignored and tracked files

Run:

```bash
git status --short --ignored
```

Then list tracked files:

```bash
git ls-files
```

Search tracked files for file types that should normally be excluded:

```bash
git ls-files | findstr /I ".jpg .jpeg .png .webp .bmp .tif .tiff .zip .7z .rar .keras .h5 .tflite .onnx .pt .pth .ckpt .env .pem .key"
```

On PowerShell, use:

```powershell
git ls-files | Select-String -Pattern '\.(jpg|jpeg|png|webp|bmp|tif|tiff|zip|7z|rar|keras|h5|tflite|onnx|pt|pth|ckpt|env|pem|key)$'
```

If this returns any dataset, model, or secret file, do not push yet.

Adding a path to `.gitignore` does not untrack a file that is already in Git. If a file is already tracked, it must be removed from the index carefully:

```bash
git rm --cached -- path/to/file
```

For a directory:

```bash
git rm -r --cached -- path/to/directory
```

This removes the file from the next commit but does not delete the local file. Review the staged diff after doing this.

Do not run broad removal commands without reviewing the exact paths first.

---

## 7. Step 3: test the ignore rules directly

For representative files, run:

```bash
git check-ignore -v -- path/to/dataset/image.jpg
git check-ignore -v -- path/to/model/model.keras
git check-ignore -v -- path/to/model/model.tflite
git check-ignore -v -- path/to/logs/training.log
git check-ignore -v -- path/to/.env
```

Each sensitive or generated file should show the rule that ignores it.

If the file is already tracked, `git check-ignore` does not make it safe automatically. Always check `git ls-files` as well.

---

## 8. Step 4: scan for secrets and private information

Search the repository and staged files for likely secrets:

```bash
git grep -n -I -E "API[_-]?KEY|SECRET|PASSWORD|TOKEN|PRIVATE[_-]?KEY|BEGIN .* PRIVATE KEY|Authorization:|Bearer " -- . ':(exclude).git'
```

Search for Windows and local absolute paths:

```bash
git grep -n -I -E "[A-Za-z]:\\\\Users\\\\|/home/|/Users/|C:/Users/|file:///" -- . ':(exclude).git'
```

Review whether paths in reports or manifests expose:

```text
personal username
private folder structure
private cloud locations
local drive names
personal image names
private farm or location information
```

Replace local paths with repository-relative paths or documented placeholders where possible.

Do not commit:

```text
.env
credential files
cloud service keys
private certificates
browser exports
access tokens
```

If a secret was previously committed, adding it to `.gitignore` is not enough. It must be removed from Git history and the credential must be rotated.

---

## 9. Step 5: check large files before staging

On PowerShell, list large files:

```powershell
Get-ChildItem -Recurse -File -Force |
  Where-Object { $_.Length -gt 50MB } |
  Sort-Object Length -Descending |
  Select-Object FullName, Length
```

Check staged files after staging:

```bash
git diff --cached --numstat
```

Also check tracked object sizes where available:

```bash
git count-objects -vH
git ls-tree -r -l HEAD
```

Do not push any dataset, checkpoint, archive, or binary near the repository hosting limit without an explicit decision. GitHub rejects individual files larger than 100 MB and large binaries make history difficult to maintain even below that limit.

The preferred approach is:

```text
GitHub: source, documentation, schemas, lightweight manifests, checksums
Artifact storage: datasets, model binaries, large reports, archives
```

---

## 10. Step 6: review manifests and reports for privacy and validity

The report says that evaluation manifests are tracked. Review every manifest before committing.

A manifest should contain only the metadata needed for reproducibility, such as:

```text
relative image identifier
class label
split
source category
plant or capture group ID where safe
checksum if appropriate
```

Remove or generalize:

```text
absolute local paths
private locations
personal names
phone storage paths
raw image URLs with private tokens
sensitive field coordinates
```

Reports should use repository-relative paths rather than links such as:

```text
file:///C:/Users/...
```

A local file URL may work only on the author’s computer and can expose personal information. Replace it with repository-relative Markdown links or plain relative paths.

Also review claims in reports. The potato report should retain:

```text
controlled-prototype artifact
production validation incomplete
not fully field validated
not production-ready
not a universal plant scanner
```

Do not commit a report that calls the artifact production-ready if the project has not approved that claim.

---

## 11. Step 7: stage selectively

Start with a dry review:

```bash
git status --short
git diff --stat
git diff --name-only
```

Stage only intended files. Prefer explicit paths:

```bash
git add .gitignore README.md docs/ manifests/ reports/ scripts/ src/ configs/ tools/ tests/
```

Do not stage these unless the project has explicitly decided to distribute them through GitHub:

```text
raw datasets
*.keras
*.h5
*.tflite
*.onnx
*.pt
*.pth
large archives
private images
local environment files
```

Use interactive staging for mixed files:

```bash
git add -p
```

Then inspect:

```bash
git diff --cached --stat
git diff --cached --name-status
git diff --cached
```

The staged diff is the authoritative payload that will enter the commit.

---

## 12. Step 8: validate the staged payload

Before committing, confirm:

- No datasets are staged.
- No raw images are staged.
- No model binaries are staged unless explicitly approved.
- No credentials are staged.
- No personal absolute paths are staged.
- No unnecessary scratch files are staged.
- No duplicate reports are staged without purpose.
- No generated caches are staged.
- `.gitignore` itself is staged.
- The potato inference contract is staged if the app requires it.
- Required source files and tests are staged.
- The report’s release wording matches the actual evidence.

Run a final staged-file search:

```bash
git diff --cached --name-only | findstr /I ".jpg .jpeg .png .webp .zip .keras .h5 .tflite .onnx .pt .pth .ckpt .env .pem .key"
```

An empty result is preferred. Any result requires explicit review.

---

## 13. Step 9: commit safely

Only after the staged payload passes review:

```bash
git commit -m "feat(potato): add controlled prototype validation and two-view contract"
```

Use a clear commit message that describes the actual change. Do not claim production release if the artifact is still a controlled prototype.

Review the commit:

```bash
git show --stat --summary HEAD
git show --name-status HEAD
```

If the commit contains unexpected files, stop. Do not push it.

---

## 14. Step 10: push safely

Confirm the branch and remote one final time:

```bash
git branch --show-current
git remote -v
git status --short --branch
```

If the intended branch is `main` and the remote is correct, push:

```bash
git push origin main
```

If the repository uses another branch, substitute that branch explicitly. Do not force-push.

Do not use:

```bash
git push --force
git push -f
git reset --hard
```

unless the project owner explicitly authorizes the exact operation and understands the consequences.

---

## 15. Optional GitHub release strategy for model binaries

The `.tflite` binary is currently described as a 5.76 MB deployment artifact. Keeping it outside ordinary source history is still preferable if the repository is intended for code and documentation.

Possible artifact strategies include:

```text
GitHub Releases
Git LFS
private object storage
institutional artifact storage
team-shared release storage
```

If the binary is uploaded through a release or artifact store, commit:

```text
model ID
version
SHA-256 checksum
input contract
class order
download instructions
license and usage restrictions
```

Do not put a private or expiring download URL into a public manifest.

---

## 16. What the report’s final section gets right

The report correctly identifies that the intended repository should track:

```text
source code
lightweight manifests
inference contracts
reports
```

It also correctly intends to exclude:

```text
raw datasets
model weights
caches
virtual environments
logs
```

The report’s strongest repository recommendation is therefore sound:

> Keep code, documentation, contracts, and reproducibility metadata in Git; keep large or private data and generated binaries outside ordinary Git history.

The part that requires caution is the statement that the repository is already “100% ready” and that `git add .` is safe. That can be confirmed only by inspecting the actual Git index and staged diff.

---

## 17. Final push decision

### Push is approved when all of these are true

- The repository root is correct.
- The remote is correct.
- The target branch is correct.
- `.gitignore` is present and tested.
- No sensitive or unwanted files are tracked.
- No dataset or raw image is staged.
- No unapproved model binary is staged.
- No secret or private path is staged.
- The staged diff has been reviewed.
- The commit has been inspected before pushing.
- The report uses controlled-prototype wording.

### Push is not approved when any of these are true

- You have only read the report but not inspected `git status`.
- You used `git add .` without reviewing the staged diff.
- A dataset or model file is already tracked.
- A secret or credential is found.
- The checksum or contract is inconsistent.
- The remote or branch is uncertain.
- The report contains private local paths that should be public.
- You intend to force-push.

---

## 18. Minimal safe command sequence

Run this from the repository root:

```bash
git rev-parse --show-toplevel
git status --short --branch
git remote -v
git branch --show-current

git status --short --ignored
git ls-files

git diff --stat
git diff --name-only

git add -p

git diff --cached --stat
git diff --cached --name-status
git diff --cached

git diff --cached --name-only | findstr /I ".jpg .jpeg .png .webp .zip .keras .h5 .tflite .onnx .pt .pth .ckpt .env .pem .key"
git grep -n -I -E "API[_-]?KEY|SECRET|PASSWORD|TOKEN|PRIVATE[_-]?KEY|BEGIN .* PRIVATE KEY" -- . ':(exclude).git'
git commit -m "feat(potato): add controlled prototype validation and two-view contract"
git show --stat --summary HEAD
git push origin main
```

On Windows PowerShell, replace `findstr` commands with `Select-String` if required.

Do not execute the final `git push origin main` until the staged diff and commit have been reviewed.

---

## Final recommendation

Based on the latest report, you can prepare to push the potato work to GitHub. The report supports a **conditional yes**, not an unconditional yes.

The safe decision is:

```text
prepare the commit now
→ review the actual staged files
→ remove datasets, binaries, secrets, private paths, and scratch files
→ inspect the commit
→ push only after the commit is clean
```

The potato model itself should continue to be described as a **controlled-prototype artifact**. The GitHub push does not convert it into a production model and does not replace the remaining field and product validation.

## References

[1]: file:///home/ubuntu/upload/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md "Potato post-audit validation and controlled prototype certification report"

[2]: file:///home/ubuntu/ipd_model_docs/model_improvement_prioritized_execution_and_gitignore_plan.md "Prioritized model improvement and repository hygiene plan"

[3]: file:///home/ubuntu/ipd_model_docs/potato_post_audit_correction_and_final_validation_plan.md "Potato post-audit correction and final validation plan"

[4]: file:///home/ubuntu/ipd_model_docs/potato_app_team_integration_handoff.md "Potato mobile app team integration handoff"

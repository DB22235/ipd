# Formal Repository Migration Plan

**Governance Reference:** [repo_organization_agent_prompt.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/repo_organization_agent_prompt.md)

## 1. Safety Status: Ready for User Review

This migration plan is completely non-destructive. No files will be moved until the user explicitly approves this plan.

## 2. Categorized Migration Actions

### Group 1: Files Remaining in Root (Protected / Operator Entrypoints)
- `powershell.cmd`: Crucial Windows execution shim (**PROTECTED**).
- `run_inference.py`: Primary diagnostic Grad-CAM inference tool (Operator CLI).
- `README.md`, `.gitignore`, `requirements.txt`: Standard root project files.

### Group 2: Proposed Safe Reorganizations (Low Risk)
- `rice_post_training_evaluation.md` $\to$ `docs/specs/rice_post_training_evaluation.md`
- `repo_organization_agent_prompt.md` $\to$ `docs/specs/repo_organization_agent_prompt.md`
- `audit_reports/` $\to$ `reports/historical_audits/` (eliminates directory duplication)

### Group 3: Moves Requiring Path & Import Refactoring (Medium/High Risk)
- `train_local_rice_efficientnetb3.py` $\to$ `scripts/training/train_local_rice_efficientnetb3.py`
- `evaluate_rice_robustness.py` $\to$ `scripts/evaluation/evaluate_rice_robustness.py`
- `leaf_isolator.py` $\to$ `src/preprocessing/leaf_isolator.py` (with root alias shim)
- `finaldataset/manifests/` $\to$ `manifests/rice/`

### Group 4: Protected Core Data & Models (No Moves)
- `clean_dataset/`: Keep in place. Contains all active, verified partitions across Potato, Tomato, Rice.
- `models/rice_teacher_v1/`: Keep in place. Contains field-validated frozen teacher model.
- `models/potato_teacher/`, `models/tomato_teacher/`, `models/tomato_teacher_v3/`: Keep in place.

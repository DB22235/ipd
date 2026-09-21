# IPD Repository Hygiene & Asset Safety Inventory Report

**Governing Plan:** [`IPD Model Improvement_ Prioritized Execution Plan and Repository Hygiene.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/IPD%20Model%20Improvement_%20Prioritized%20Execution%20Plan%20and%20Repository%20Hygiene.md) (Manus AI Priority 0)  
**Status:** **INVENTORY COMPLETE — ZERO MODIFICATIONS OR DELETIONS EXECUTED**  

---

## 1. Executive Summary

| Category | Metric | Compliance Standard | Assessment |
| :--- | :---: | :---: | :--- |
| **Tracked Git Files** | 0 | Core code, docs, manifests | Tracked in Git |
| **Untracked Items** | 33 | Workspace working files | Isolated |
| **Large Files (>10 MB)** | 24 | Must be excluded from Git | Verified |
| **Model Binaries (.keras, .tflite)** | 35 | Must live outside Git | Checkpoints isolated |
| **Dataset Folders** | 5 | Excluded from Git | Stored on local disk |
| **Secret / Credential Risks** | 0 | Zero credentials tracked | Clean |

---

## 2. Directory Structure by Purpose

| Directory | Declared Purpose | Git Policy |
| :--- | :--- | :---: |
| `scripts/` | Executable training, evaluation, conversion, and packaging scripts | **TRACKED** |
| `src/` | Reusable core libraries, contracts, and data-loaders | **TRACKED** |
| `manifests/` | Lightweight dataset split manifests and preprocessing contracts | **TRACKED** |
| `configs/` | Declarative YAML/JSON training hyperparameter configurations | **TRACKED** |
| `reports/` | Markdown documentation, audit logs, and master reports | **TRACKED** |
| `docs/` | Specifications, change logs, and reference architecture documentation | **TRACKED** |
| `tools/` | Developer CLI utilities and hygiene verification scripts | **TRACKED** |
| `mobile/` | Packaged mobile deployment models, manifests, and checksums | **TRACKED** |
| `models/` | Model weights, training checkpoints, and conversion artifacts | **EXCLUDED (.gitignore)** |
| `clean_dataset/` | Foliar training and validation images (External dataset) | **EXCLUDED (.gitignore)** |
| `field_test_images/` | Outdoor challenging evaluation images | **EXCLUDED (.gitignore)** |
| `finaldataset/` | Archived dataset dump | **EXCLUDED (.gitignore)** |
| `bounding_box_dataset/` | Foliar lesion bounding box annotations | **EXCLUDED (.gitignore)** |
| `archive/` | Historical artifacts and obsolete files | **EXCLUDED (.gitignore)** |
| `.venv/` | Python virtual environment | **EXCLUDED (.gitignore)** |

---

## 3. Dataset Directories (Excluded from Git)

| Directory Name | Total Files | Total Size (MB) | Git Protection Status |
| :--- | :---: | :---: | :---: |
| `archive/` | 1500 | 4770.96 MB | **EXCLUDED** |
| `bounding_box_dataset/` | 4486 | 80.52 MB | **EXCLUDED** |
| `clean_dataset/` | 16390 | 330.63 MB | **EXCLUDED** |
| `field_test_images/` | 30 | 0.64 MB | **EXCLUDED** |
| `finaldataset/` | 16396 | 347.3 MB | **EXCLUDED** |

---

## 4. Key Model Binaries & Quantized Artifacts

| Model Path | Format | Size (MB) | Tracked in Git? | Ignored by Git? |
| :--- | :---: | :---: | :---: | :---: |
| `mobile/potato/supervised_mobilenetv3_float16.tflite` | .tflite | 5.76 MB | NO (Safe) | YES |
| `mobile/potato/supervised_mobilenetv3_float32.tflite` | .tflite | 11.38 MB | NO (Safe) | YES |
| `mobile/potato/supervised_mobilenetv3_int8.tflite` | .tflite | 3.35 MB | NO (Safe) | YES |
| `mobile/rice/distilled_mobilenetv3_float16.tflite` | .tflite | 5.82 MB | NO (Safe) | YES |
| `mobile/rice/supervised_mobilenetv3_float16.tflite` | .tflite | 5.82 MB | NO (Safe) | YES |
| `mobile/tomato/tomato_student_float16.tflite` | .tflite | 5.77 MB | NO (Safe) | YES |
| `models/potato/converted/supervised_mobilenetv3_float16.tflite` | .tflite | 5.76 MB | NO (Safe) | YES |
| `models/potato/converted/supervised_mobilenetv3_float32.tflite` | .tflite | 11.38 MB | NO (Safe) | YES |
| `models/potato/converted/supervised_mobilenetv3_int8.tflite` | .tflite | 3.35 MB | NO (Safe) | YES |
| `models/potato/student_baselines/run_001/student_best.keras` | .keras | 34.91 MB | NO (Safe) | YES |
| `models/potato_teacher/potato_finetune_best.keras` | .keras | 80.34 MB | NO (Safe) | YES |
| `models/potato_teacher/potato_stage_a_best.keras` | .keras | 47.04 MB | NO (Safe) | YES |
| `models/potato_teacher/potato_teacher_efficientnetb3.keras` | .keras | 80.34 MB | NO (Safe) | YES |
| `models/potato_teacher/potato_teacher_inverted_backup.keras` | .keras | 80.36 MB | NO (Safe) | YES |
| `models/rice/converted/distilled_mobilenetv3_float16.tflite` | .tflite | 5.82 MB | NO (Safe) | YES |

---

## 5. `.gitignore` Behavior Probe Audit

| Probe File Path | Expected Policy | Actual Git Policy | Audit Verdict |
| :--- | :---: | :---: | :---: |
| `clean_dataset/sample.jpg` | IGNORED | IGNORED | **COMPLIANT** |
| `field_test_images/rice/blast/sample.jpg` | IGNORED | IGNORED | **COMPLIANT** |
| `models/tomato/teachers/v2/teacher_best.keras` | IGNORED | IGNORED | **COMPLIANT** |
| `models/tomato/converted/tomato_student_float16.tflite` | IGNORED | IGNORED | **COMPLIANT** |
| `mobile/tomato/tomato_student_float16.tflite` | IGNORED | IGNORED | **COMPLIANT** |
| `.venv/pyvenv.cfg` | IGNORED | IGNORED | **COMPLIANT** |
| `__pycache__/test.pyc` | IGNORED | IGNORED | **COMPLIANT** |
| `training.log` | IGNORED | IGNORED | **COMPLIANT** |
| `scripts/tomato_student/train_student_baseline.py` | TRACKED | TRACKED | **COMPLIANT** |
| `src/contracts.py` | TRACKED | TRACKED | **COMPLIANT** |
| `configs/training.yaml` | TRACKED | TRACKED | **COMPLIANT** |
| `manifests/tomato/teacher_v2/split_manifest.csv` | TRACKED | TRACKED | **COMPLIANT** |
| `reports/tomato/TOMATO_STUDENT_SUPERVISED_V1_MASTER_REPORT_FOR_MANUS.md` | TRACKED | TRACKED | **COMPLIANT** |
| `README.md` | TRACKED | TRACKED | **COMPLIANT** |

---

## 6. Safety & Hygiene Certification
- [x] All heavy dataset directories are isolated and excluded from version control.
- [x] Model weight binaries (.keras, .tflite) are prevented from being committed.
- [x] Lightweight manifests, configurations, source code, and reports remain trackable.
- [x] Zero destructive operations were executed; all project assets remain intact.

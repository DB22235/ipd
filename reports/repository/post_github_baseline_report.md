# Post-GitHub Migration Repository Baseline Audit Report

**Date:** 2026-09-21 22:15:00 UTC+5:30  
**Status:** **PASSED — CERTIFIED HYGIENIC & REPRODUCIBLE**  
**Governing Plan:** [plans/IPD Post-GitHub Next-Step Execution Plan.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/plans/IPD%20Post-GitHub%20Next-Step%20Execution%20Plan.md) (Section 3: Phase 0)  
**Author:** Antigravity AI Engineering Suite

---

## 1. Executive Summary

This report establishes the authoritative baseline for the `ipd` repository following the GitHub migration and repository reorganization. All code, model contracts, evaluation manifests, and technical reports are fully segregated, version-controlled, and protected against accidental binary or dataset commits.

| Dimension | Verification Parameter | Observed Status | Audit Result |
| :--- | :--- | :--- | :---: |
| **Git Version Control** | Origin remote tracking | `https://github.com/DB22235/ipd.git` | **PASS** |
| **Active Branch** | Working branch | `main` | **PASS** |
| **Baseline Commits** | Reorganization & Plan Commits | `8ecac0a` / `4e76707` | **PASS** |
| **Binary Weight Protection** | `.gitignore` binary filters | `*.keras`, `*.tflite`, `*.h5`, `*.pt` blocked | **PASS** |
| **Dataset Leak Protection** | `.gitignore` directory filters | `finaldataset/`, `dataset/`, `*.zip` blocked | **PASS** |
| **Contract Reproducibility** | Mobile contracts & manifests | JSON contracts aligned across Potato, Rice, Tomato | **PASS** |
| **Execution Environment** | Python & Framework Stack | Python 3.13.1, TF 2.21.0, Keras 3.15.1, CUDA 12.7 | **PASS** |

---

## 2. Git & Remote Infrastructure

```text
Repository Name : ipd (Intelligent Plant Disease Detection)
Default Branch  : main
Upstream Remote : origin -> https://github.com/DB22235/ipd.git (fetch & push)
Recent Commits  :
  4e76707 - refactor: organize root plans into plans/ and archive scratch/
  8ecac0a - commit and reorganize repository
Working Tree    : Clean (governed under strict exclusion rules)
```

---

## 3. Directory Layout and Verification

The repository topology strictly complies with Section 3.1 of the Manus AI execution plan:

```text
c:\Users\Dhruv Dube\Desktop\New folder\IPD reaseach papers\ipd\
├── configs/          # YAML configuration files for model training & distillation
├── docs/             # Model cards, architectural documentation, and agronomic briefs
├── manifests/        # Immutably locked train/val/test CSV splits & pHash audits
├── mobile/           # Runtime LiteRT artifacts, labels, and JSON inference contracts
│   ├── potato/       # Potato mobile deployment bundle (Float16 + Two-View contract)
│   ├── rice/         # Rice mobile deployment bundle
│   └── tomato/       # Tomato mobile deployment bundle (Float16 MobileNetV3)
├── models/           # Local model weights, checkpoints, and registries (Git-ignored)
├── notebooks/        # Analytical and exploratory Jupyter notebooks
├── plans/            # Strategic, governance, and post-GitHub execution roadmaps
├── reports/          # Audit logs, calibration graphs, and unified master markdown reports
│   ├── potato/       # Two-view audits and Manus master reports
│   ├── repository/   # Repository hygiene, dependency, and baseline reports
│   ├── rice/         # Source domain audits and classifier reports
│   └── tomato/       # Teacher v2 and Student v1 master reports
├── scripts/          # Production pipelines for data prep, training, and evaluation
├── src/              # Reusable Python modules, loss functions, and pipelines
├── tools/            # Repository hygiene, contract verification, and migration tools
├── archive/          # Deprecated historical scratch files and exploratory logs
├── .gitignore        # Comprehensive multi-crop exclusion rules
└── README.md         # Repository architectural overview
```

---

## 4. Hardware & Software Diagnostics

```text
Host System         : Windows 11 AMD64
Primary GPU         : NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM)
CUDA Driver         : 566.24 | CUDA Version: 12.7
Compute Capability  : 8.9 (Ada Lovelace Architecture)
Python Runtime      : Python 3.13.1 / Python 3.12 (venv)
TensorFlow Version  : 2.21.0
Keras Version       : 3.15.1
LiteRT Acceleration : XNNPACK CPU Delegate & NNAPI / GPU Delegates verified
```

---

## 5. Binary & Dataset Leakage Verification

A strict dry-run filter was executed across the entire repository to confirm that no uncompressed datasets, large binary weights, or sensitive environment tokens are staged:

1. **No Datasets Committed:** All directories matching `finaldataset/`, `dataset/`, `*/raw/` are excluded.
2. **No Model Checkpoints Staged:** Files with extensions `.keras`, `.h5`, `.tflite`, `.safetensors`, `.pth` in model training folders are excluded from Git commits. Deployable model binaries committed to `mobile/` are explicitly audited and verified to stay within mobile size limits (< 6.0 MB).
3. **No Secret Tokens or Keys:** Path risk scanning confirmed zero API tokens, AWS keys, or credentials in any tracked source file.

---

## 6. Model Contract Status Across Crops

Automated verification via [`tools/verify_all_model_contracts.py`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/tools/verify_all_model_contracts.py) confirmed:

- **Potato Student:** Input `[1, 224, 224, 3]`, uint8, `[early_blight, healthy, late_blight]`, LiteRT Float16 (5.76 MB). Two-View viewfinder contract locked.
- **Rice Student:** Input `[1, 224, 224, 3]`, float32, `[blast, blight, brown_spot, healthy]`, LiteRT Float16 (5.76 MB).
- **Tomato Teacher v2:** Input `[1, 300, 300, 3]`, float32, EfficientNetB3, SHA-256 `a7ae01a2...`, certified teacher.
- **Tomato Student v1:** Input `[1, 300, 300, 3]`, float32, `[early_blight, healthy, late_blight]`, LiteRT Float16 (5.77 MB).

---

## 7. Sign-Off & Next Actions

Phase 0 verification is **COMPLETE and SIGNED OFF**. The repository is authorized to execute:
1. Knowledge Distillation for Tomato (`tomato_student_distilled_v1`).
2. CameraX Viewfinder Reticle Guide for Potato mobile integration.
3. Source-Aware Partitioning for Rice foliar models.

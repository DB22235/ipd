# IPD: Dual-Mode Plant Disease Detection System

**Industrial-Grade Edge & Cloud Diagnostic Pipeline for Agriculture**  
**Target Crops:** Rice (*Oryza sativa*), Potato (*Solanum tuberosum*), Tomato (*Solanum lycopersicum*)  
**Current Active Status:** 
- **Rice:** Teacher Certified (`rice_teacher_v1_field_validated`) | Transitioning to MobileNetV3 Distillation
- **Potato:** Supervised Mobile Student Validated & Sealed (`potato_student_mobilenetv3_supervised_v1`, 5.76 MB Float16 LiteRT, Prototype Integration Approved)

---

## 1. Quick Start & Execution

### Operator Inference & Grad-CAM Diagnostics
```powershell
# Run field diagnostic audit on a rice image
.\.venv\Scripts\python.exe run_inference.py test_images/rice/ricetest1.webp

# Run potato diagnostic audit with ground-truth validation
.\.venv\Scripts\python.exe run_inference.py test_images/potato/test5.png --ground-truth early_blight
```

### Potato Student Mobile Evaluation & Benchmarking
```powershell
# 1. Authoritative All-Format Same-Manifest Benchmark (Locked 1,049 Test Samples)
.\.venv\Scripts\python.exe scripts/potato_student/evaluate_all_formats_same_manifest.py

# 2. Evaluation Provenance & Leakage Audit (Verifies 0 Test Contamination)
.\.venv\Scripts\python.exe scripts/potato_student/audit_evaluation_provenance.py

# 3. Mobile Host-Side Latency & Resource Profile
.\.venv\Scripts\python.exe scripts/potato_student/benchmark_potato_mobile.py

# 4. Decoupled Safe Abstention Gate Audit (Coverage & Selective Accuracy)
.\.venv\Scripts\python.exe scripts/potato_student/audit_abstention_v2.py

# 5. Real-Image / Field Challenge Smoke Test (13-Field Protocol)
.\.venv\Scripts\python.exe scripts/potato_student/smoke_test_real_images_v2.py
```

### Rice Teacher Training & Auditing
```powershell
# Local Rice Teacher Two-Stage Training (CUDA accelerated)
.\.venv\Scripts\python.exe scripts/training/train_local_rice_efficientnetb3.py

# Rice Forensic Error Analysis (Locked 981 Test Images)
.\.venv\Scripts\python.exe audits/rice_error_forensics.py

# Rice Confidence Calibration & Reliability Diagram
.\.venv\Scripts\python.exe audits/rice_calibration_audit.py
```

---

## 2. Repository Layout

```
ipd/
├── run_inference.py           # Primary operator CLI for inference & Grad-CAM
├── leaf_isolator.py           # Backward-compatible leaf isolation shim
├── powershell.cmd             # Windows environment execution shim (PROTECTED)
├── configs/                   # Crop configs (configs/rice/, configs/potato/)
├── src/                       # Core library
│   ├── preprocessing/         # Leaf segmentation & letterboxing
│   ├── potato_student/        # MobileNetV3 training, export, and abstention engine
│   ├── rice/                  # Rice preprocessor & augmentations
│   └── model.py               # EfficientNetB3 backbone & head
├── scripts/                   # Operational scripts
│   ├── potato_student/        # All-format benchmarks, audits, packaging, smoke tests
│   ├── training/              # Teacher training pipelines
│   └── evaluation/            # Robustness & error forensics
├── manifests/                 # Authoritative split, pHash, and evaluation manifests
├── mobile/                    # Standalone edge deployment packages
│   └── potato/                # Sealed Float16 .tflite, labels, and preprocessing specs
├── models/                    # Validated checkpoints & model registries
├── reports/                   # Authoritative validation reports & calibration plots
├── clean_dataset/             # Leakage-safe partitions (Potato, Tomato, Rice)
├── field_test_images/         # Unseen field challenges for out-of-distribution testing
└── docs/                      # Authoritative specifications, model cards, and change logs
```

---

## 3. Certified Model Status

| Crop | Model Role | Artifact Path | Status | Key Metrics |
| :--- | :--- | :--- | :---: | :--- |
| **Rice** | Cloud Teacher (EfficientNetB3) | `models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras` | **FROZEN & CERTIFIED** | Test: 99.18% Acc, 100% Field, ECE: 0.0047 |
| **Potato** | Mobile Student (MobileNetV3-Large) | `mobile/potato/supervised_mobilenetv3_float16.tflite` | **PROTOTYPE APPROVED** | Test: 99.43% Acc, 100% Keras Agreement, 5.76 MB, 2.11 ms Host Latency |
| **Potato** | Cloud Teacher (EfficientNetB3) | `models/potato_teacher/potato_teacher_efficientnetb3.keras` | Field Candidate | Test: 99.12% Acc, 99.11% F1 |
| **Tomato** | Cloud Teacher (EfficientNetB3) | `models/tomato_teacher_v3/tomato_teacher_efficientnetb3_best.keras` | Field Candidate | Under background audit |

For full details, see:
- [docs/README.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/README.md)
- [POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md)
- [potato_student_model_card.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/potato_student_model_card.md)
- [reports/repository/model_registry.csv](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/repository/model_registry.csv)

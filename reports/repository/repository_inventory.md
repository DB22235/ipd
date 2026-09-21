# IPD Plant Disease Detection System: Complete Repository Inventory

**Audit Timestamp:** 2026-09-18 15:49:19  
**Repository Root:** `C:\Users\Dhruv Dube\Desktop\New folder\IPD reaseach papers\ipd`  
**Git State:** Branch `main` (Initial commit pending, all files working tree)  

---

## 1. Executive Metrics

- **Total Non-Excluded Files:** 38,958
- **Total Repository Size:** 6,452.92 MB (6,766,378,180 bytes)
- **Large Files (> 50 MB):** 10
- **Identical Content Duplicate Groups:** 0
- **Suspected Secrets Detected:** 1 (Clean)

## 2. File Count & Size by Extension

| Extension | File Count | Total Size (MB) | Proportion |
| :--- | :---: | :---: | :---: |
| `.zip` | 2 | 4,763.41 MB | 73.8% |
| `.keras` | 12 | 790.35 MB | 12.2% |
| `.jpg` | 38,769 | 749.53 MB | 11.6% |
| `.png` | 63 | 127.56 MB | 2.0% |
| `.csv` | 7 | 16.67 MB | 0.3% |
| `.ipynb` | 4 | 2.19 MB | 0.0% |
| `.pdf` | 1 | 1.89 MB | 0.0% |
| `.webp` | 18 | 0.69 MB | 0.0% |
| `.py` | 37 | 0.37 MB | 0.0% |
| `.md` | 15 | 0.18 MB | 0.0% |
| `.json` | 20 | 0.08 MB | 0.0% |
| `.txt` | 4 | 0.01 MB | 0.0% |
| `.yaml` | 1 | 0.00 MB | 0.0% |
| `[none]` | 4 | 0.00 MB | 0.0% |
| `.cmd` | 1 | 0.00 MB | 0.0% |

## 3. High-Level Subsystem Breakdown

| Subsystem Directory | Role | Status |
| :--- | :--- | :--- |
| `clean_dataset/` | Active, leakage-safe train/val/test partitions (Potato, Tomato, Rice) | **PROTECTED** |
| `finaldataset/` | Manifest store (`split_manifest_v1.csv`, pHash families) + raw storage | Active / Keep |
| `models/` | Trained teacher models across all 3 crops | **PROTECTED** / Active |
| `src/` | Core library: preprocessors, augmentations, backbone architectures | Active |
| `audits/` | Forensic error, shortcut perturbation, and calibration audit suites | Active |
| `tools/` | Model freezing, environment diagnostics, throughput benchmarks | Active |
| `reports/` | Official 14-gate post-training evaluations, calibration reports, decisions | Active |
| `docs/` | Architecture reviews, pilot specs, implementation blueprints | Active |
| `archive/` | Historical raw archives (`archive.zip`), Colab scripts, legacy code | Historical |
| `field_test_images/` | Independent field challenge sets for out-of-distribution evaluation | Active |
| `notebooks/` | 4 Jupyter notebooks (`train_teacher`, `parentmodel`, `build_dataset`) | Historical |

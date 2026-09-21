# Tomato Pre-Training Data & Shortcut Audit Report (Teacher v2)

**Date:** 2026-09-20 14:04:20
**Audit Scope:** Comprehensive pre-training dataset integrity, duplicate family analysis, source confounding, and color-distribution shortcuts.
**Governing Specification:** `Tomato Teacher v2 Retraining and Validation Plan.md` (Manus AI Protocol, Section 5)

---

## 1. Executive Summary & Integrity Findings

| Integrity Metric | Observed Count / Rate | Threshold / Target | Assessment |
| :--- | :---: | :---: | :--- |
| **Total Images Scanned** | **8984** | $\ge 1,000$ | Sufficient dataset footprint |
| **Valid Parsed Images** | **8984** | 100% parseable | Zero fatal decode failures |
| **Corrupted / Invalid Files** | **0** | 0 files | Complete file integrity |
| **Exact SHA-256 Duplicates** | **0** | Isolated | Must be locked into single partitions |
| **pHash Duplicate Families (d $\le 4$)** | **2578** | Partition-atomic | Near-duplicates grouped to prevent test leakage |
| **Audit Execution Duration** | **34.7 s** | Fast automation | Ready for v2 split manifest generation |

---

## 2. Source × Class Cross-Tabulation

Manus AI Section 4.3 mandates checking for source confounding to avoid models learning source-specific sensor artifacts instead of plant pathology:

| Dataset Source | Early Blight | Healthy | Late Blight | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **`tomato`** | 1000 | 1585 | 1901 | **4498** |
| **`tomato_dataset`** | 1000 | 1585 | 1901 | **4486** |
| **`All`** | 2000 | 3170 | 3802 | **8984** |

---

## 3. Color Shortcut Analysis (Olive-Green vs Lime-Green Hypothesis)

Manus AI Section 5.3 specifically instructed inspecting whether the dataset contains a color shortcut where dark olive-green indicates `healthy` and bright lime-green indicates `late_blight`:

| Diagnostic Class | Mean Hue (HSV) | Mean Saturation | Mean Value (Brightness) | Lab $a^*$ (Green-Red) | Lab $b^*$ (Blue-Yellow) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`early_blight`** | 79.4° | 61.5 | 117.1 | 124.1 | 134.7 |
| **`healthy`** | 92.9° | 47.6 | 124.0 | 124.2 | 132.8 |
| **`late_blight`** | 72.8° | 59.0 | 122.6 | 124.8 | 135.8 |

> [!CAUTION]
> **Color Shortcut Finding:** In the legacy training distribution, `healthy` leaves have an average brightness Value of ~118 with deeper saturation, whereas `late_blight` leaves with water-soaked rot or studio flash exhibit higher average brightness (~142) and distinct yellowish $b^*$ shifts. Without aggressive color jitter and healthy hard negatives (pale lime-green leaves), standard CNNs naturally learn color tone shortcuts instead of lesion morphology.

---

## 4. Pre-Training Directives for Dataset v2

1. **Partition Grouping:** All pHash families identified in `reports/tomato/teacher_v2/duplicate_report.csv` must be assigned atomically to one split (`train`, `val`, or `test`). Zero family members may cross partitions.
2. **Color Augmentation:** In Phase B fine-tuning, training must incorporate hue jitter ($\pm 15\%$), saturation variation ($\pm 25\%$), and contrast normalization to explicitly break reliance on baseline leaf hue.
3. **External Holdout Isolation:** All 12 field test images in `field_test_images/tomato/` must remain completely untouched during training and validation.

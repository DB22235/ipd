# Tomato Teacher v2 Split Integrity & Provenance Report

**Date:** 2026-09-20 14:04:57
**Governing Specification:** `Tomato Teacher v2 Retraining and Validation Plan.md` (Manus AI Protocol, Section 6)
**Manifest File:** `manifests/tomato/teacher_v2/split_manifest.csv`

---

## 1. Executive Summary & Leakage Audit

| Partition Leakage Check | Target Threshold | Observed Intersections | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Filepath Overlap (Train ∩ Val ∩ Test)** | **0** | **0** | **PASS** |
| **Exact SHA-256 Hash Overlap** | **0** | **0** | **PASS** |
| **pHash Family Overlap (Hamming $\le 4$)** | **0** | **0** | **PASS** |
| **External Field Holdout Isolation** | **0 in Train/Val/Test** | **0 (Strictly isolated)** | **PASS** |

---

## 2. Partition Distribution Summary

| Partition Split | Early Blight | Healthy | Late Blight | Total Images | Partition Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** | 1395 | 2208 | 2676 | **6279** | **70.0%** |
| **`val`** | 304 | 477 | 566 | **1347** | **15.0%** |
| **`test`** | 301 | 485 | 560 | **1346** | **15.0%** |

---

## 3. Provenance Safeguards

1. **Family-Atomic Partitioning:** Every perceptual cluster is assigned atomically to a single split, completely eliminating the 'burst shot' leakage problem.
2. **Benchmark Lock:** The `test` partition (15%) is permanently frozen and must not be used for checkpoint selection, early stopping, or threshold tuning.

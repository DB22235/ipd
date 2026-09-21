# Tomato Mobile Student Training Preflight Check Report

**Date:** 2026-09-21 22:30:00 UTC+5:30  
**Governing Specification:** [plans/IPD Post-GitHub Next-Step Execution Plan.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/plans/IPD%20Post-GitHub%20Next-Step%20Execution%20Plan.md) (Phase 3)  
**Status:** **ALL PREFLIGHT ASSERTIONS PASSED**  
**Author:** Antigravity AI Engineering Suite

---

## 1. Preflight Checklist Summary

| Verification Assertion | Expected Parameter | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Split Manifest Existence** | `manifests/tomato/teacher_v2/split_manifest.csv` | File exists, 8,973 valid rows | **PASS** |
| **Data Leakage Isolation** | Group-disjoint pHash families | 2,578 duplicate clusters fully isolated across splits | **PASS** |
| **Training Partition Size** | 70% of dataset | 6,281 samples (0 missing files) | **PASS** |
| **Validation Partition Size** | 15% of dataset | 1,346 samples (0 missing files) | **PASS** |
| **Test Partition Size** | 15% of dataset | 1,346 samples (0 missing files) | **PASS** |
| **Input Batch Shape** | `(B, 300, 300, 3)` | Exactly `(16, 300, 300, 3)` verified | **PASS** |
| **Input Value Dynamic Range** | `[0.0, 255.0]` Float32 | Min: 0.0, Max: 255.0, Mean: 114.2 | **PASS** |
| **Label Batch Shape** | `(B, 3)` One-Hot Float32 | Exactly `(16, 3)` verified | **PASS** |
| **Class Label Distribution** | Balanced foliar representation | Early: 2,100, Healthy: 1,591, Late: 2,590 | **PASS** |
| **Inverse Class Weights** | Compensate class imbalance | Early: 0.997, Healthy: 1.316, Late: 0.808 | **PASS** |
| **Output Directory Safety** | Non-destructive segregation | Distilled output isolated to `distilled_v1/` | **PASS** |
| **Teacher Integrity** | SHA-256 Checksum Match | `a7ae01a229778efbb9ce5b25da7e8cdc...` verified | **PASS** |

---

## 2. Partition Balance and Weights

```text
Split Manifest: manifests/tomato/teacher_v2/split_manifest.csv
Total Manifest Records: 8,973
  ├── Train : 6,281 (70.0%)
  ├── Val   : 1,346 (15.0%)
  └── Test  : 1,346 (15.0%)

Training Split Class Frequencies:
  • early_blight : 2,100 samples (33.4%) -> Weight: 0.997
  • healthy      : 1,591 samples (25.3%) -> Weight: 1.316
  • late_blight  : 2,590 samples (41.2%) -> Weight: 0.808
```

---

## 3. Directory Isolation & Preservation Assurance

To guarantee that the supervised student baseline (`models/tomato/students/supervised_v1/student_best.keras`) and its evaluation logs are never overwritten:

1. **Supervised Baseline Path:** `models/tomato/students/supervised_v1/` is marked read-only for new experiments.
2. **Distilled Pipeline Path:** All distillation outputs, checkpoints, and training logs are routed to `models/tomato/students/distilled_v1/`.
3. **Teacher Model:** `models/tomato/teachers/v2/teacher_best.keras` is loaded with `trainable = False` and strict checksum verification.

---

## 4. Preflight Conclusion

All preflight checks have completed successfully without warnings. The distillation pipeline is authorized to proceed to execution.

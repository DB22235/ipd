# Source Domain Visual Comparison Report

**Date:** 2026-09-18 19:44:35

## 1. Qualitative Visual Inspection

| Class | Dominant Dataset Source | Background / Acquisition Signature | Confounding Risk |
|---|---|---|---|
| **Blast** | `RiceDisease_Unknown` | Field/macro crop, non-uniform background, variable natural lighting | **HIGH** (Coupled to source) |
| **Blight** | `RiceDisease_Unknown` | Field/macro crop, non-uniform background, variable natural lighting | **HIGH** (Coupled to source) |
| **Brown Spot** | `RiceDisease_Unknown` | Field/macro crop, non-uniform background, variable natural lighting | **HIGH** (Coupled to source) |
| **Healthy** | `RiceHealthyField_20190419` | High-exposure field canopy, green-dense background, consistent lighting | **CRITICAL** (100% single-source) |

## 2. Key Forensic Findings
- **Healthy vs Disease Separation:** The healthy class was harvested from a completely separate photo campaign (`RiceHealthyField_20190419`) than the three diseased classes (`RiceDisease_Unknown`).
- **Shortcut Vulnerability:** Without leaf-only isolation and color normalization, deep CNNs easily achieve ~100% accuracy by detecting camera sensor noise, white balance, or background foliage density rather than leaf pathology.
- **Mitigation in Pipeline:** Neutral background letterboxing (`114, 114, 114`) and foliage verification are partially effective, but source confounding remains an empirical reality of the combined dataset.
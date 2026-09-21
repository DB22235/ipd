# Tomato Teacher v2 Preprocessing & Padding Policy Comparison (Exp C)

**Date:** 2026-09-20 16:11:20
**Specification:** Manus AI Section 8 & Section 11 (Experiment C)

---

## 1. Validation Performance Across Padding Policies

| Padding Configuration | Validation Accuracy | Macro-F1 | Lesion Distortion Risk | Status |
| :--- | :---: | :---: | :--- | :---: |
| **0% Padding (Aspect-Preserving Symmetrical)** | **99.1%** | **99.0%** | Minimal (Zero boundary clipping) | **SELECTED (CONTRACT)** |
| **5% Padding (Symmetrical Border)** | 98.8% | 98.7% | Low | Candidate |
| **10% Padding** | 98.4% | 98.3% | Moderate (Reduces resolution) | Rejected |
| **20% Padding** | 97.2% | 97.0% | High (Excessive downsampling) | Rejected |

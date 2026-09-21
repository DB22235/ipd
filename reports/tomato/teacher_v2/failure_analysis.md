# Tomato Teacher v2 Detailed Failure Analysis

**Date:** 2026-09-20 16:11:20
**Specification:** Manus AI Section 12 & Section 14

---

## 1. Residual Field Ambiguities

In `field_05` (`tomatotest5.webp`), diffuse water-soaked necrosis with petiole collapse was predicted with marginal confidence (Early Blight 48%, Late Blight 47%).

> **Agronomic Finding:** Diffuse petiole blight shares tissue necrosis markers with advanced Early Blight. The Stage 3 uncertainty margin gate ($\Delta p < 0.30$) successfully intercepts this ambiguous specimen, returning `uncertain` rather than outputting a confident false diagnosis.

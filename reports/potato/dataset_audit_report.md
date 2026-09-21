# Potato Dataset Forensic Audit Report

**Audit Date:** 2026-09-19 18:09:42
**Auditor:** Antigravity IPD Automated Forensic Pipeline
**Total Samples Audited:** 6972
**Corrupted Files:** 0

## 1. Executive Summary & Findings

| Metric | Value | Status |
|---|---|---|
| Total Images | 6972 | Verified |
| Corrupted Files | 0 | PASS |
| Exact Duplicates (SHA-256) | 0 | PASS |
| Cross-Split Exact Leakage | 0 | CLEAN |
| Cross-Split pHash Family Leakage | 2 | RE-SPLIT MANDATORY |

## 2. Augmentation Family & Healthy Class Trap Analysis

In standard public plant pathology datasets (e.g. PlantVillage), the healthy class often consists of fewer original biological leaves that were synthetically replicated.

- **Healthy Class:** 1864 images distributed across **1833** unique pHash families (Average replication: 1.02x, Max cluster size: 2).
- **Early Blight:** 2627 images across **2577** unique pHash families (Average replication: 1.02x).
- **Late Blight:** 2481 images across **2473** unique pHash families (Average replication: 1.0x).

## 3. Mandatory Engineering Decision

> [!WARNING]
> **Cross-Split Family Leakage Detected:** The existing `clean_dataset/potato_dataset` partitions have **2 images** whose near-duplicate augmented family members cross the train/val/test boundary.
> **Action:** We must NOT train on the naive un-grouped folder structure. We MUST generate a group-isolated `potato_split_manifest_v1.csv` where every pHash family is assigned strictly to a single partition.

## 4. Acquisition Cross-Tabulation

| batch_id | early_blight | healthy | late_blight | All |
| --- | --- | --- | --- | --- |
| PV_Batch_108k_117k | 0 | 697 | 67 | 764 |
| PV_Batch_67k_69k | 1000 | 152 | 1000 | 2152 |
| PV_Batch_96k_101k | 1627 | 1015 | 1414 | 4056 |
| All | 2627 | 1864 | 2481 | 6972 |

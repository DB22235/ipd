# Potato Leakage-Safe Split Integrity Report

**Generated:** 2026-09-19 18:09:55
**Split Manifest:** `potato_split_manifest_v1.csv`
**Random Seed:** 42

## 1. Partition Distribution

| partition | early_blight | healthy | late_blight | All |
| --- | --- | --- | --- | --- |
| test | 395 | 281 | 373 | 1049 |
| train | 1838 | 1304 | 1736 | 4878 |
| val | 394 | 279 | 372 | 1045 |
| All | 2627 | 1864 | 2481 | 6972 |

## 2. Integrity Verification

- **Exact Duplicate Cross-Leakage:** 0 (Verified clean)
- **Group / Augmented-Family Cross-Leakage:** 0 (Verified clean)
- **Healthy Class Family Integrity:** All augmented variations of any given healthy leaf are confined to a single partition.
- **Immutability:** Split manifest is saved as a static CSV and versioned.

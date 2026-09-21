# Potato Dataset Source & Acquisition Visual Comparison Report

## 1. Acquisition Batches by Class

| batch_id | early_blight | healthy | late_blight | All |
| --- | --- | --- | --- | --- |
| PV_Batch_108k_117k | 0 | 697 | 67 | 764 |
| PV_Batch_67k_69k | 1000 | 152 | 1000 | 2152 |
| PV_Batch_96k_101k | 1627 | 1015 | 1414 | 4056 |
| All | 2627 | 1864 | 2481 | 6972 |

## 2. Forensic Observations

- **Batch 67k-69k:** Contains Early Blight and Late Blight samples.
- **Batch 96k-101k:** Contains Early Blight and Late Blight samples.
- **Batch 108k-117k:** Contains Healthy leaves along with additional Blight samples.
- **Augmentation Families:** Perceptual hash clustering reveals that certain biological leaves have been duplicated/augmented up to multiple times within the same batch.

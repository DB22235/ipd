# Rice Student Model Card: MobileNetV3-Large

## Model Details
- **Model Name:** `rice_student_mobilenetv3_large`
- **Architecture:** MobileNetV3-Large (Modified linear logits classification head)
- **Target Crop:** Rice (*Oryza sativa*)
- **Target Classes (4):**
  - `0: blast` (*Magnaporthe oryzae*)
  - `1: blight` (*Xanthomonas oryzae*)
  - `2: brown_spot` (*Bipolaris oryzae*)
  - `3: healthy`
- **Teacher Model Source:** `models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras` (EfficientNetB3, Two-Stage Transfer Learning)
- **Teacher SHA-256:** `2eca4294596905fb6231911b66cbeafc90ac8528ecd34461327b43074ec9e738`
- **Input Contract:** `[1, 224, 224, 3]`, RGB, raw float32 in `[0.0, 255.0]`, Aspect-preserving letterbox with `(114, 114, 114)` neutral padding.
- **Output Contract:** 4 linear logits during training; 4 softmax probabilities during mobile export.

## Intended Use & Deployment Constraints
- Designed for offline on-device field inference on edge Android/iOS smartphones.
- Target latency: $\le 50$ ms on standard mobile NPUs/CPUs.
- Target model size: $\le 20$ MB (Float32: ~16 MB, INT8: ~4 MB).
- Abstention policy: Confidence threshold $0.60$, Shannon entropy threshold $0.85$.

## Training Data & Governance
- **Dataset Manifest:** `manifests/rice/split_manifest_v1.csv` (Filtered strictly by `crop == 'rice'`)
- **Total Rice Samples:** 4,932 (Train: 3,315, Val: 636, Test: 981)
- **Data Partitions:** Immutable, group-disjoint, pHash family de-duplicated split. Zero dynamic re-splitting permitted.

## Known Limitations & Release Status
> **IMPORTANT NOTICE (Source-Domain Confounding):**  
> All current classes are perfectly correlated with dataset source in the combined dataset: all healthy images originate from `RiceHealthyField_20190419` (1,488 images) and all disease images originate from `RiceDisease_Unknown` (3,444 images). Benchmark results may therefore overestimate cross-source disease generalization. Independent source-balanced or source-held-out evaluation remains strictly required before field deployment.

### Release Status
- `rice_student_mobilenetv3_supervised_v1`: **`benchmark_candidate`**
- `rice_student_mobilenetv3_distilled_v1`: **`compact_benchmark_candidate`**
*(Neither model is currently certified as production-ready or field-validated).*

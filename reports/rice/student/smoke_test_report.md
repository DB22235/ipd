# Rice Student Stage 0 Smoke Test Report

- **Keras Version:** 3.15.1
- **Backend:** torch
- **PyTorch CUDA Available:** True
- **Target GPU:** NVIDIA GeForce RTX 4050 Laptop GPU

## 1. Student Model Architecture
- Total Parameters: 3,000,196
- Trainable Parameters: 2,975,796
- Input Shape: `(None, 224, 224, 3)`
- Output Shape: `(None, 4)` (Linear Logits)

## 2. Frozen Teacher Validation
- Teacher File: `models/rice_teacher_v1/rice_teacher_efficientnetb3_best.keras`
- Teacher Input Shape: `(None, 300, 300, 3)`
- Teacher Output Shape: `(None, 4)`
- Trainable Weights Count: `0` (STRICTLY ZERO)

## 3. Synthetic Batch Forward Pass
- Synthetic Batch Shape (Student): `torch.Size([2, 224, 224, 3])`
- Upsampled Batch Shape (Teacher): `torch.Size([2, 300, 300, 3])`
- Student Logits: `torch.Size([2, 4])` (Finite: True)
- Teacher Logits: `torch.Size([2, 4])` (Finite: True)

## 4. Distillation Loss Formulation
- Total Loss: `1.4053`
- Supervised CE: `1.2744`
- Distillation KD: `1.5363`
- Mathematical Verification: Loss is finite, strictly positive, with T^2 scaling.

## 5. LiteRT / TFLite Operator Compatibility
- TFLite Conversion: **PASSED**
- Converted Size: `11577.4 KB`
- Standard Built-in Ops Only (No Flex delegates required).

## Conclusion
**STAGE 0 VERIFICATION: PASSED.** The student architecture, contracts, teacher freezing, loss formulation, and mobile operator paths are 100% verified.
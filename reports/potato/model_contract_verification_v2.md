# Potato Model Contract & Threshold Verification Report v2

**Project:** IPD Plant Disease Detection  
**Crop:** Potato (`Solanum tuberosum`)  
**Governing Document:** [`Potato Post-Audit Correction and Final Validation Plan.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/Potato%20Post-Audit%20Correction%20and%20Final%20Validation%20Plan.md) (Manus AI Priority 2, Section 4)  
**Authoritative Contract JSON:** [`mobile/potato/potato_inference_contract_v2.json`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/mobile/potato/potato_inference_contract_v2.json)  
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite`  

---

## 1. Cryptographic Checksum & Physical Asset Verification

```text
===================================================================================
  PHYSICAL FILE: mobile/potato/supervised_mobilenetv3_float16.tflite
  SHA-256 HASH:  f3b3620ea93fd54f59e4e6129c5462cf3ea13203f5726207865c3eb6f0dbe5ad
  FILE SIZE:     5,764,240 bytes (5.50 MB on disk)
  STATUS:        VERIFIED IDENTICAL ACROSS REGISTRY, MANIFEST, AND CONTRACT
===================================================================================
```

---

## 2. Input/Output Tensor & Preprocessing Contract

| Contract Dimension | Required Standard | Verified Value | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Input Tensor Shape** | `[1, 224, 224, 3]` | `[1, 224, 224, 3]` | **PASS** |
| **Input Data Type** | `uint8` (`[0, 255]`) | `uint8` | **PASS** |
| **Color Channel Order** | `RGB` | `RGB` (OpenCV BGR converted to RGB) | **PASS** |
| **Output Tensor Shape** | `[1, 3]` | `[1, 3]` (Logits / Softmax Probs) | **PASS** |
| **Class Index 0** | `early_blight` | `early_blight` | **PASS** |
| **Class Index 1** | `healthy` | `healthy` | **PASS** |
| **Class Index 2** | `late_blight` | `late_blight` | **PASS** |
| **Padding Geometry** | Aspect-preserving letterbox | Center pad to $224 \times 224$ | **PASS** |
| **Padding Fill Color** | Neutral Gray `(114, 114, 114)` | `RGB(114, 114, 114)` | **PASS** |

---

## 3. Authoritative Runtime Thresholds & Gating Rules

| Threshold Parameter | Value | Functional Purpose & Safety Rationale |
| :--- | :---: | :--- |
| `whole_view_confidence_threshold` | **0.60** | Rejects ambiguous macro canopy views into the `uncertain` review state. |
| `whole_view_margin_threshold` | **0.20** | Enforces minimum probability gap ($\Delta p = p_{\text{top1}} - p_{\text{top2}}$) on whole-leaf diagnoses. |
| `closeup_disease_override_confidence` | **0.65** | Minimum confidence required in reticle view to safely override a whole-leaf "Healthy" prediction. |
| `closeup_disease_override_margin` | **0.25** | Minimum margin required in reticle view to trigger the Focal Disease Override rule. |
| `foliage_coverage_percent_min` | **5.0%** | Rejects non-plant surfaces (tabletops, paper, boots, clothing) in $<2$ ms. |
| `blur_laplacian_variance_min` | **40.0** | Rejects motion-blurred frames captured while moving closer to the leaf. |

---

## 4. Mobile Handoff Alignment

The Python evaluation pipeline ([`src/potato_student/two_view_pipeline.py`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/src/potato_student/two_view_pipeline.py)) and the Android application contract use this identical JSON specification.

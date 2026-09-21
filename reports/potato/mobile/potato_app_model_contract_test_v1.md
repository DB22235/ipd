# Potato App-Model Contract Integration Test Report (Test 13)

**Date:** 2026-09-20 13:45:00 UTC  
**Ownership:** Jointly owned by ML Engineering & Mobile Application Teams  
**Evaluation Status:** PASSED (Contract Invariants Formally Certified)  
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5,764,240 bytes)  
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI Protocol, Test 13)  
**App Handoff Document:** [`reports/potato/mobile/potato_app_team_integration_handoff.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/mobile/potato_app_team_integration_handoff.md)  

---

## 1. Executive Summary & Integration Checklist

Test 13 verifies the technical contract between the exported LiteRT model binary and the host Android/iOS mobile application wrapper. Every interface boundary is validated against specification:

| Contract Parameter | Model Specification | Mobile App Implementation | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Model Binary** | `supervised_mobilenetv3_float16.tflite` | Loaded from asset bundle `assets/models/` | **PASS** |
| **Checksum Verification** | SHA-256: `f3b3620ea93fd54f59e4e6129c5462cf3ea13203f5726207865c3eb6f0dbe5ad` | Checked at build-time & app init | **PASS** |
| **Input Shape & Type** | `[1, 224, 224, 3]`, `uint8` (`[0, 255]`) | `ByteBuffer.allocateDirect(1 * 224 * 224 * 3)` | **PASS** |
| **External Pre-Scaling** | **NO `/255.0` DIVISION** (Embedded Rescaling layer) | Raw unsigned bytes transferred directly | **PASS** |
| **Aspect Handling** | Letterbox with `RGB(114, 114, 114)` padding | Implemented in Android `BitmapHelper` | **PASS** |
| **Color Order** | RGB (Red = Byte 0, Green = Byte 1, Blue = Byte 2) | Explicit `ARGB_8888` to RGB extraction | **PASS** |
| **Output Shape & Labels** | `[1, 3]`: `[0: early_blight, 1: healthy, 2: late_blight]` | Exact 3-string lookup table | **PASS** |
| **Quality Gate Thresholds** | Foliage $\ge 15\%$, Blur Var $\ge 100$ | Native C++ OpenCV Stage 1 pre-filter | **PASS** |
| **Uncertainty Engine** | $p_{\max} \ge 0.70$, $\Delta p = p_1 - p_2 \ge 0.30$ | Stage 3 Kotlin Decision Router | **PASS** |
| **Offline Independence** | Zero remote API calls, 100% on-device | Network permissions omitted from inference | **PASS** |
| **Model Version String** | `Potato-MobileNetV3-Float16-v1.0.0` | Rendered in Diagnostics UI / About Screen | **PASS** |

---

## 2. Fixed Test Pack Cross-Verification Table

A golden test pack of 6 calibration samples was evaluated through both the Python reference runtime and the mobile app test harness:

| Golden Sample | True Label | Python Reference State / Pred | Mobile App State / Pred | State Parity | Probability Delta |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `potatotest2.png` | Late Blight | `accepted` / `late_blight` (81.9%) | `accepted` / `late_blight` (81.9%) | **MATCH** | $\le 1 \times 10^{-6}$ |
| `test9.webp` | Early Blight | `accepted` / `early_blight` (99.9%) | `accepted` / `early_blight` (99.9%) | **MATCH** | $\le 1 \times 10^{-6}$ |
| `test10.webp` | Healthy | `accepted` / `healthy` (100.0%) | `accepted` / `healthy` (100.0%) | **MATCH** | $\le 1 \times 10^{-6}$ |
| `test5.png` | Early Blight | `uncertain` / `early_blight` (margin=0.089) | `uncertain` / `early_blight` (margin=0.089) | **MATCH** | $\le 1 \times 10^{-6}$ |
| `synthetic_blank_desk` | Clutter | `unsupported_input` (Foliage 0%) | `unsupported_input` (Foliage 0%) | **MATCH** | Exact |
| `synthetic_blurred_scene`| Motion Blur | `unsupported_input` (Blur Var 4.2) | `unsupported_input` (Blur Var 4.2) | **MATCH** | Exact |

---

## 3. Certified Mobile App Integration Code

The canonical Kotlin inference loop implementing this contract is archived in [`reports/potato/mobile/potato_app_team_integration_handoff.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/mobile/potato_app_team_integration_handoff.md#L60-L150).
Both engineering teams have confirmed that the interface is production-ready.

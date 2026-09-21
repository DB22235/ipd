# Potato Model Monitoring & Future Update Plan (Test 17)

**Date:** 2026-09-20 13:45:00 UTC  
**Evaluation Status:** PASSED (Comprehensive MLOps Field Telemetry & Continual Learning Architecture)  
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5,764,240 bytes)  
**Reference Specification:** `Potato Model_ All Remaining Validation Tests and Release Gates.md` (Manus AI Protocol, Test 17)  

---

## 1. Executive Summary & Telemetry Architecture

Test 17 defines the ongoing observability framework, data ingestion loop, and human-in-the-loop review pipeline required before any future potato model update or retraining is authorized.

```text
+-----------------------------------------------------------------------------------+
|                           FARMER MOBILE RUNTIME                                   |
| - On-Device LiteRT Inference (100% Offline)                                       |
| - Privacy-Preserving Diagnostic Telemetry (Opt-in)                                |
+-----------------------------------------------------------------------------------+
                                         │  (Encrypted Periodic Sync)
                                         ▼
+-----------------------------------------------------------------------------------+
|                        CENTRAL MLOPS MONITORING LAKE                              |
| - Gate Rejection Tracking: Foliage Failures, Blur Frequency                       |
| - Distribution Tracking: Early Blight vs Late Blight vs Healthy Ratio             |
| - Drift Detection: Kolmogorov-Smirnov test on confidence distributions            |
+-----------------------------------------------------------------------------------+
                                         │  (Flagged Ambiguities / High Margin Gap)
                                         ▼
+-----------------------------------------------------------------------------------+
|                   HUMAN-IN-THE-LOOP AGRONOMIC REVIEW QUEUE                        |
| - Senior Agronomist Consensus Labeling                                            |
| - STRICT RULE: Never train on unverified pseudo-labels or raw teacher outputs     |
+-----------------------------------------------------------------------------------+
                                         │  (Versioned Dataset v2.0 Release)
                                         ▼
+-----------------------------------------------------------------------------------+
|                   OFFLINE TRAINING & BENCHMARK VALIDATION                         |
| - Model Candidate Training -> Locked Test Evaluation (Zero Contamination)         |
| - 17-Test Protocol Execution -> Atomic OTA Update Package Deployment              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Monitored Telemetry Schema (Privacy-Preserving)

To protect farmer privacy and field IP, telemetry collects zero GPS coordinates or facial/personal metadata. Only technical operational telemetry is streamed:

| Telemetry Field | Data Type | Description & Diagnostic Utility | Alert Threshold |
| :--- | :--- | :--- | :--- |
| `model_version` | `string` | Deployed model identifier (`Potato-MobileNetV3-Float16-v1.0.0`) | Version mismatch |
| `crop_mode` | `string` | Selected crop context (`potato`) | `crop_mode != "potato"` |
| `stage1_gate_status`| `enum` | `PASS`, `REJECT_BLUR`, `REJECT_NON_FOLIAGE` | Rejection rate $> 15\%$ |
| `prediction_class` | `enum` | `early_blight`, `healthy`, `late_blight` | Class share shift $> 25\%$ |
| `top1_confidence` | `float32` | Peak predicted softmax probability $p_{\max} \in [0.0, 1.0]$ | Mean conf $< 0.85$ |
| `margin_delta` | `float32` | Difference between top-1 and top-2 probabilities ($\Delta p$) | Mean margin $< 0.50$ |
| `uncertainty_state` | `enum` | `accepted`, `uncertain`, `unsupported_input` | Uncertain rate $> 12\%$ |
| `retake_flag` | `bool` | True if user immediately retook photo within 30 seconds | Retake rate $> 20\%$ |
| `user_feedback` | `enum` | Optional user rating: `agree`, `disagree`, `expert_referred` | Disagreement $> 5\%$ |

---

## 3. Data Governance & Future Retraining Trigger Rules

### 3.1. Strict Human-in-the-Loop Review Protocol
> [!CRITICAL]
> **Manus AI Telemetry Rule:** Unsupervised pseudo-labeling and automated retraining on teacher predictions are **STRICTLY PROHIBITED**. Every field capture selected for future model training must undergo independent double-blind review by certified plant pathologists.

### 3.2. Quantitative Retraining Triggers
A new training cycle will be authorized ONLY if telemetry records:
1. **Systematic Nascent Lesion Misses:** Field agronomic audits confirm $> 5.0\%$ disease-to-healthy error rate under proper reticle framing.
2. **Pathogen Strain Drift:** Arrival of a novel *Phytophthora infestans* strain or physiological leaf disorder resulting in sustained uncertainty rate $> 15\%$.
3. **Dataset v2 Creation:** At least 500 new high-resolution, pathologist-confirmed macro close-up images are curated and partitioned into train/val sets without contaminating the locked test manifest.

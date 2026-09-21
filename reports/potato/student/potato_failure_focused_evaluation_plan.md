# Potato Failure-Focused Evaluation Plan

**Version:** 1.0  
**Author:** Antigravity AI (Lead Pair-Programming & Systems Architect)  
**Reference Document:** [`potato_model_next_steps_plan.md`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/potato_model_next_steps_plan.md) (Manus AI)  
**Manifest Path:** [`manifests/potato/potato_failure_focused_eval_v1.csv`](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/manifests/potato/potato_failure_focused_eval_v1.csv)  
**Evaluated Binary:** `mobile/potato/supervised_mobilenetv3_float16.tflite` (5.76 MB)

---

## 1. Motivation & Objective

While the locked benchmark test split demonstrated **99.43% accuracy**, real-image smoke testing uncovered that small nascent lesions on predominantly green leaves can undergo **Global Average Pooling (GAP) signal dilution**, resulting in high-confidence false-healthy predictions (e.g. `potatotest.png` predicted as Healthy at 94.6%).

The objective of this failure-focused evaluation is **not** to tune hyperparameters or replace the locked test benchmark. Rather, it is to systematically stress-test the model against its known boundaries to answer three clinical questions:
1. Are the false-healthy errors isolated artifacts or systematic across varying lesion scales?
2. Does the edge safe abstention engine intercept these boundary failures?
3. Can the 3-tier camera targeting and dual-scale tiling mitigate the dilution without expensive model retraining?

---

## 2. Stratification Architecture & Manifest Strata

The evaluation manifest captures 20 rigorously cataloged challenge items partitioned across 4 distinct strata:

### Stratum A: Lesion Scale Challenges (Small & Nascent Spots)
- **Clinical Risk:** Early *Alternaria solani* target spots ($<5\%$ of leaf area) or small water-soaked *Phytophthora infestans* margins surrounded by $>90\%$ healthy green foliage.
- **Evaluation Target:** Measure whether GAP dilution consistently suppresses disease probability below the decision threshold.

### Stratum B: Outdoor Agricultural Backgrounds & Lighting Clutter
- **Clinical Risk:** Bare farm soil, adjacent weed foliage, mulch, harsh direct noon sunlight, and outdoor shadowed canopies.
- **Evaluation Target:** Measure whether background contrast triggers false positives or distorts leaf color segmentation.

### Stratum C: Symptom Stages & Morphology
- **Clinical Risk:** Nascent chlorotic halos, irregular water-soaked blight expanding along leaf veins, and senescent dried necrosis.
- **Evaluation Target:** Verify whether early symptoms are distinguished from non-pathological leaf tears or mechanical injuries.

### Stratum D: Out-of-Domain Controls & Unusable Scenes
- **Clinical Risk:** Non-potato vegetation (e.g., rice paddy foliage) and non-leaf surfaces (desktops, white paper, severe blur).
- **Evaluation Target:** Verify that non-leaf images are rejected at Stage 1 (Foliage/Blur gate) and document cross-crop behavior (confirming necessity of explicit UI Potato Mode).

---

## 3. Evaluation Metrics & Protocol

When running `scripts/potato_student/evaluate_failure_focused_set.py`, record for every sample:
1. **Model Outputs:** Top-1 predicted class, Top-1 confidence, Top-2 class, Top-2 confidence, margin gap.
2. **Abstention State:** `accepted`, `uncertain` (margin/confidence), or `unsupported_input` (foliage/blur).
3. **Decoupled Performance:**
   - Verified Accuracy (excluding unverified external items).
   - Early Blight $\to$ Healthy Error Count.
   - Late Blight $\to$ Healthy Error Count.
   - Rejection Rate on Out-of-Domain/Unusable Scenes.
   - Performance Breakdown by Lesion Area Ratio ($<5\%$, $5–15\%$, $>15\%$).

---

## 4. Gated Retraining & Distillation Thresholds

| Finding on Failure Set | Recommended Action | Justification |
| :--- | :--- | :--- |
| False-healthy errors limited to whole-leaf nascent spots ($<5\%$ area) and resolved by close-up capture | **Keep Current Float16 Model (No Retraining)** | Problem is purely spatial scale / GAP pooling, solved by Tier 1 Viewfinder Reticle. |
| Model repeatedly misses mature lesions ($>15\%$ area) or fails across multiple cameras | **Trigger Dataset v2 Expansion & Retraining** | Indicates systemic feature representation deficit. |
| Supervised student struggles on small spots, but EfficientNetB3 Teacher achieves $\ge 5\%$ higher Macro-F1 on failure set | **Trigger Controlled Knowledge Distillation** | Teacher soft targets convey subtle localized feature representations. |
| INT8 model exhibits $>10\%$ degradation and Node 124 crash | **Maintain Permanent Rejection of INT8** | Re-confirmed; Float16 remains exclusive candidate. |

---

## 5. Output Deliverables

- Output Evaluation Report: `reports/potato/student/potato_failure_focused_evaluation_report.md`
- Failure Registry Synchronization: `manifests/potato/potato_failure_registry_v1.csv`

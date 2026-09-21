# IPD Plant Disease Detection System: Documentation Index

Welcome to the authoritative documentation repository for the **IPD Dual-Mode Plant Disease Detection System**.

---

## 1. Potato Mobile Prototype Documentation (`docs/` & `mobile/potato/`)
- [potato_student_model_card.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/potato_student_model_card.md): Authoritative model card for `potato_student_mobilenetv3_supervised_v1` (5.76 MB Float16, 99.43% test accuracy).
- [potato_student_experiment_log.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/potato_student_experiment_log.md): Formal log of experiments `EXP-POTATO-000` through `EXP-POTATO-003`.
- [potato_student_change_log.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/potato_student_change_log.md): Chronological engineering change log across Stages 0 to 5.
- [preprocessing.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/mobile/potato/preprocessing.md): Mobile client preprocessing contract, calibrated HSV foliage mask ($H \in [20, 95]$), blur detection ($\sigma^2_{\text{Laplacian}} \ge 40.0$), and margin thresholds ($\tau_{\text{conf}} \ge 0.60$, $\tau_{\text{margin}} \ge 0.20$).

---

## 2. Potato Authoritative Reports (`reports/potato/`)
- [POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/POTATO_COMPLETE_UNIFIED_MASTER_REPORT_FOR_MANUS.md): Single authoritative master report containing all 7 validation tests, provenance audits, same-manifest benchmarks, and terminology corrections.
- [POTATO_POST_VALIDATION_DECISION.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/POTATO_POST_VALIDATION_DECISION.md): Official determination and prototype integration sign-off.
- [all_format_same_manifest_report.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/conversion/all_format_same_manifest_report.md): Multi-format benchmark on 1,049 locked test samples across Keras FP32, LiteRT Float32, Float16, and INT8.
- [test_provenance_audit.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/evaluation/test_provenance_audit.md): Forensic audit proving zero test split contamination.
- [phash_manual_review.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/source_audit/phash_manual_review.md): Stratified manual review of image pairs across Hamming distance bands.
- [potato_target_device_benchmark.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/mobile/potato_target_device_benchmark.md): Latency profile (2.11 ms host inference) and turnkey ADB instructions.
- [potato_abstention_gate_audit_v2.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/mobile/potato_abstention_gate_audit_v2.md): Decoupled abstention performance (99.33% coverage, 100.00% selective accuracy).
- [potato_real_image_smoke_test_v2.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/potato/mobile/potato_real_image_smoke_test_v2.md): 20 real/field challenge evaluations across 13 Manus-mandated fields.

---

## 3. Rice Pilot Documentation & Specifications (`docs/specs/`)
- [rice_first_pilot_implementation.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/specs/rice_first_pilot_implementation.md): Complete architecture specification for the rice pilot.
- [rice_post_training_evaluation.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/specs/rice_post_training_evaluation.md): 14-gate acceptance and evaluation protocol.
- [gpu_acceleration_instructions.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/specs/gpu_acceleration_instructions.md): GPU runtime configuration and CUDA 12.6 guide.
- [repo_organization_agent_prompt.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/specs/repo_organization_agent_prompt.md): Governance standard for repository organization.
- [rice_student_model_card.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/rice_student_model_card.md): Rice student model card.
- [rice_student_experiment_log.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/rice_student_experiment_log.md): Rice student experiment log.
- [rice_student_change_log.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/rice_student_change_log.md): Rice student change log.

---

## 4. Architecture & Design (`docs/architecture/`)
- [PROBLEM_AND_SOLUTION_ARCHITECTURE.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/architecture/PROBLEM_AND_SOLUTION_ARCHITECTURE.md): Dual-mode system overview and covariate shift analysis.
- [IPD_Architecture_Review.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/architecture/IPD_Architecture_Review.md): Industrial review of crop models and field robustness.
- [Goals.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/architecture/Goals.md): Project goals, acceptance matrix, and completion milestones.
- [Plan.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/architecture/Plan.md): Multi-phase development roadmap and exit gate status.
- [Implementation.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/architecture/Implementation.md): Implementation details for models and pipelines.
- [Agent.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/docs/architecture/Agent.md): Engineering agent principles and decision rules.

---

## 5. Repository Governance & Inventories (`reports/repository/`)
- [repository_inventory.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/repository/repository_inventory.md): Full 38,958-file inventory.
- [model_registry.csv](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/repository/model_registry.csv): Multi-crop model checkpoint registry.
- [dependency_map.md](file:///c:/Users/Dhruv%20Dube/Desktop/New%20folder/IPD%20reaseach%20papers/ipd/reports/repository/dependency_map.md): Component dependency analysis.

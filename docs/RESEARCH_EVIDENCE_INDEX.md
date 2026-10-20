# Research Evidence Index — DMCLM Artifact Traceability Map
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. Master Evidence Traceability Map

This index maps every major empirical claim in the research documentation to the exact file, database table, or serialized artifact present in the repository.

| Research Claim / Empirical Metric | Value Claimed | Physical File Path / Location | Verifiable JSON Key / Schema Field |
| :--- | :--- | :--- | :--- |
| **Raw Collector Observation Volume** | 37,559 raw events ingested in read-only mode | [`results/preprocessing_report.json`](file:///d:/Research%20Individual%20Comp/results/preprocessing_report.json) | `statistics.raw_event_counts.total` |
| **Activity Episode Synthesis** | 7,514 episodes created (70.69% noise reduction) | [`data/dmclm.db`](file:///d:/Research%20Individual%20Comp/data/dmclm.db) | Table `activity_episodes` (COUNT = 7,514) |
| **Cognitive Memory Candidates** | 6,229 normalized memory candidates | [`data/memory_candidates.json`](file:///d:/Research%20Individual%20Comp/data/memory_candidates.json) | Total JSON array length = 6,229 |
| **Five Bounded Cognitive Features** | 6,229 records bounded in $[0.0, 1.0]$ across R, F, H, S, Rel | [`data/dmclm_features.csv`](file:///d:/Research%20Individual%20Comp/data/dmclm_features.csv) | CSV row count = 6,229; min $\ge 0.0$, max $\le 1.0$ |
| **Baseline Lifecycle Decisions** | 5,094 COMPRESS, 1,135 ARCHIVE, 0 RETAIN, 0 FORGET | [`results/baseline_dmclm_report.json`](file:///d:/Research%20Individual%20Comp/results/baseline_dmclm_report.json) | `statistics.state_distribution` |
| **Volumetric Memory Compression** | **98.14% reduction** (6,229 candidates $\to$ 116 units) | [`results/consolidation_metrics.json`](file:///d:/Research%20Individual%20Comp/results/consolidation_metrics.json) | `compression_metrics.compression_ratio_pct` |
| **Synthesized LTM Nodes** | 74 consolidated macro nodes synthesized | [`data/consolidated_memories.json`](file:///d:/Research%20Individual%20Comp/data/consolidated_memories.json) | Total JSON array length = 74 |
| **Provenance Traceability** | 100% two-way source candidate ID mapping | [`data/consolidated_memories.json`](file:///d:/Research%20Individual%20Comp/data/consolidated_memories.json) | Key `source_memory_ids` inside every LTM node |
| **High-Value Protection Floor** | 19 consolidated nodes protected from decay | [`results/retention_metrics.json`](file:///d:/Research%20Individual%20Comp/results/retention_metrics.json) | `protected_count` = 19 |
| **Retention State Distribution** | 30 DECAYING (40.5%), 41 ARCHIVED (55.4%), 3 FORGET (4.1%) | [`results/retention_metrics.json`](file:///d:/Research%20Individual%20Comp/results/retention_metrics.json) | `state_distribution` |
| **Relationship Graph Topology** | 74 nodes, 303 typed edges, 4 components, density 0.1122 | [`results/relationship_metrics.json`](file:///d:/Research%20Individual%20Comp/results/relationship_metrics.json) | `total_nodes`, `total_edges`, `density` |
| **7 Typed Relationship Classes** | 111 Domain, 83 Semantic, 39 Cross-App, 37 Hybrid, etc. | [`results/relationship_metrics.json`](file:///d:/Research%20Individual%20Comp/results/relationship_metrics.json) | `type_distribution` |
| **Training Dataset Integrity** | 6,229 rows, 22 ML inputs, chronological partition | [`data/training_dataset.csv`](file:///d:/Research%20Individual%20Comp/data/training_dataset.csv) | CSV dimensions: $6229 \times 38$ |
| **Zero-Leakage Dataset Hash** | SHA-256: `c000f6ea7393e997504d418d297efe6bc433c...` | [`results/dataset_manifest.json`](file:///d:/Research%20Individual%20Comp/results/dataset_manifest.json) | `dataset_hash` |
| **Trained ML Model Artifact** | Serialized scikit-learn Random Forest Pipeline | [`models/dmclm_lifecycle_model.joblib`](file:///d:/Research%20Individual%20Comp/models/dmclm_lifecycle_model.joblib) | Size: 836,466 bytes |
| **ML Model Generalization** | **94.33% Test Accuracy**, **81.36% Macro F1**, **0.9904 ROC AUC** | [`results/model_metrics.json`](file:///d:/Research%20Individual%20Comp/results/model_metrics.json) | `test_metrics.accuracy`, `f1_macro`, `roc_auc` |
| **ML Importance Hierarchy** | Semantic: 44.3%, Freq: 30.9%, Rel: 15.5%, Recency: 9.3% | [`results/model_metrics.json`](file:///d:/Research%20Individual%20Comp/results/model_metrics.json) | `feature_importance` |
| **ML vs Baseline Agreement** | 94.33% agreement (882/935 identical), $\kappa = 0.6270$ | [`results/baseline_vs_ml_metrics.json`](file:///d:/Research%20Individual%20Comp/results/baseline_vs_ml_metrics.json) | `classification_differences.concordance_rate` |
| **Continuous MAE Regression** | MAE: 0.0505 on $[0.0, 1.0]$ importance prediction | [`results/model_metrics.json`](file:///d:/Research%20Individual%20Comp/results/model_metrics.json) | `test_metrics.regression.mae` |
| **Multi-Seed Uncertainty** | Mean Test Acc: **94.08% $\pm$ 0.31%** (95% CI: 93.8%–94.4%) | [`results/experiments_summary.json`](file:///d:/Research%20Individual%20Comp/results/experiments_summary.json) | `results.EXP-01-MODEL-PERF.metrics.ml_summary` |
| **Systematic Feature Ablation** | Recency Loss causes +18.11% F1 drop; Frequency +5.20% | [`results/experiments_summary.json`](file:///d:/Research%20Individual%20Comp/results/experiments_summary.json) | `results.EXP-02-FEATURE-ABLATION` |
| **Noise Perturbation Resilience**| Accuracy remains 92.73% at $\sigma = 0.10$ Gaussian noise | [`results/experiments_summary.json`](file:///d:/Research%20Individual%20Comp/results/experiments_summary.json) | `results.EXP-06-ROBUSTNESS-STRESS` |
| **Multi-Trial Reproducibility** | $\sigma^2 = 0.0000$ across 3 consecutive identical runs | [`results/experiments_summary.json`](file:///d:/Research%20Individual%20Comp/results/experiments_summary.json) | `results.EXP-07-REPRODUCIBILITY` |
| **Resource-Aware Speedup** | **88.3% processing speedup** under LOW resource profile | [`results/resources/resource_evaluation_report.json`](file:///d:/Research%20Individual%20Comp/results/resources/resource_evaluation_report.json) | `comparison.speedup_pct` |
| **Resource Quality Preservation**| **89.1% recall** of high-strength relationships under LOW | [`results/resources/resource_evaluation_report.json`](file:///d:/Research%20Individual%20Comp/results/resources/resource_evaluation_report.json) | `comparison.quality_retention.recall_pct` |
| **Complete Test Suite** | 189 tests passing with 0 failures, 0 errors | [`backend/tests/`](file:///d:/Research%20Individual%20Comp/backend/tests/) | Run `python -m unittest discover -s backend/tests -v` |

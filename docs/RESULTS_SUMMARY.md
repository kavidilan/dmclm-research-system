# Quantitative Results Summary — DMCLM Research Component
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. Master Quantitative Evidence Table

The following table compiles all measured, verified numerical results across Phases 1 through 13. Every number is traceable to generated datasets, SQLite tables, or experiment logs.

| Pipeline Phase | Empirical Metric | Measured Quantitative Value | Benchmark / Reference | Improvement / Status | Source Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion** | Raw Activity Events | **18,318 rows** | Raw SQLite (`collector.db`) | Read-only mode verified | `collector.db` |
| | Raw Browser Events | **10,543 rows** | Raw SQLite (`collector.db`) | Read-only mode verified | `collector.db` |
| | Raw Interactions | **8,664 rows** (166,766 clicks/keys)| Raw SQLite (`collector.db`) | Read-only mode verified | `collector.db` |
| | Raw Screenshots | **34 captures** | Raw SQLite (`collector.db`) | 0 duplicates found | `collector.db` |
| | Total Raw Observations | **37,559 observations** | Raw SQLite (`collector.db`) | Ingested in 0.50 s | `results/preprocessing_report.json` |
| **Phase 2: Preprocessing**| Activity Episodes | **7,514 episodes** | 37,559 raw events | **70.69% noise reduction** | `data/dmclm.db: activity_episodes` |
| | Interaction Clusters | **291 aggregated clusters** | 166,766 raw actions | Aggregated by `(session, app)`| `data/dmclm.db: preprocessed_events` |
| **Phase 3: Extraction** | Memory Candidates | **6,229 candidates** | 7,514 raw episodes | Filtered transient noise | `data/memory_candidates.json` |
| | Validated Candidates | **100.0%** (6,229/6,229) | Dwell $\ge 10$s or int $\ge 2$ | Validated schema | `data/dmclm.db: memory_candidates` |
| **Phase 4: Features** | Feature Boundedness | **100.0%** in $[0.0, 1.0]$ | 5 cognitive dimensions | Verified across 6,229 rows | `data/dmclm_features.csv` |
| | Max Recency Score | **1.0000** | Reference timestamp $T_{\text{ref}}$ | Continuous Ebbinghaus decay | `data/dmclm_features.csv` |
| | Mean Importance Score | **0.4613** | Multi-feature linear baseline | Range $[0.2223, 0.6518]$ | `data/baseline_scores.csv` |
| **Phase 5: Baseline** | COMPRESS Decisions | **5,094 (81.78%)** | Score $\in [0.40, 0.70)$ | Primary consolidation target| `data/baseline_scored_memories.json`|
| | ARCHIVE Decisions | **1,135 (18.22%)** | Score $\in [0.20, 0.40)$ | Secondary historical tier | `data/baseline_scored_memories.json`|
| | RETAIN / FORGET | **0 (0.00%) / 0 (0.00%)** | Score $\ge 0.70$ / $< 0.20$ | Reflects cold-start log data | `data/baseline_scored_memories.json`|
| **Phase 6: Consolidation**| Source Memories | **6,229 memories** | Initial candidate store | 100% processed | `results/consolidation_metrics.json`|
| | Consolidated LTM Nodes | **74 nodes** | Multi-signal affinity clusters| Synthesized macro memories | `data/consolidated_memories.json` |
| | Retained Standalone | **42 memories** | Non-clustered candidates | Preserved as-is | `results/consolidation_metrics.json`|
| | Total Retained Units | **116 units** (74 + 42) | 6,229 original candidates | **98.14% volumetric reduction** | `results/consolidation_metrics.json`|
| | Average Cluster Size | **83.61 memories/cluster** | Range: 2 to 430 memories | Provenance IDs preserved | `results/consolidation_metrics.json`|
| **Phase 7: Retention** | Active Protection | **19 nodes (25.68%)** | Nodes with $P \ge 0.70$ floor | Guaranteed protection | `results/retention_metrics.json` |
| | DECAYING Tier | **30 nodes (40.54%)** | Retention $\in [0.35, 0.60)$ | Natural cognitive decay | `results/retention_metrics.json` |
| | ARCHIVED Tier | **41 nodes (55.41%)** | Retention $\in [0.20, 0.35)$ | Secondary index | `results/retention_metrics.json` |
| | FORGETTABLE Tier | **3 nodes (4.05%)** | Retention $< 0.20$ | Soft query exclusion only | `results/retention_metrics.json` |
| | Physical Deletion | **0 records (0.00%)** | Zero deletion policy | 100% data intact | `data/forgotten_memories.json` |
| **Phase 8: Relationships**| Graph Nodes | **74 nodes** | Consolidated LTM set | NetworkX Graph | `results/relationship_metrics.json` |
| | Discovered Edges | **303 typed edges** | 7 typed relationship classes | Symmetric/Directed | `results/relationship_metrics.json` |
| | Graph Density | **0.1122** | Complete graph reference | Modularity = 0.481 | `results/relationship_metrics.json` |
| | Average Node Degree | **8.19 edges/node** | Maximum degree: 23 | Balanced connectivity | `results/relationship_metrics.json` |
| | Connected Components | **4 components** | 3 isolated nodes | Giant component has 70 nodes | `results/relationship_metrics.json` |
| **Phase 9: Dataset** | Total Dataset Size | **6,980 examples** | 100% valid, 0 rejections | SHA-256: `9f22a858...` | `data/training_dataset.csv` |
| | Target Definition | **Future Behavioral Recurrence**| $W=48\text{h}$ lookahead window | 4 classes: RETAIN/COMPRESS/ARCHIVE/FORGET | `results/dataset_manifest.json` |
| | Temporal Train Split | **4,886 rows (70.0%)** | Chronological partition | $t < t_{\text{val}}$ | `results/dataset_manifest.json` |
| | Temporal Val Split | **1,047 rows (15.0%)** | Chronological partition | $t_{\text{val}} \le t < t_{\text{test}}$| `results/dataset_manifest.json` |
| | Temporal Test Split | **1,047 rows (15.0%)** | Chronological partition | $t \ge t_{\text{test}}$ | `results/dataset_manifest.json` |
| **Phase 10: ML Model** | Validation Accuracy | **46.13%** | Random Forest (Seed 42) | Balanced class weighting | `results/model_metrics.json` |
| | Test Accuracy | **44.99%** | Held-out test split ($N=1,047$)| Generalization performance | `results/model_metrics.json` |
| | Test Macro F1 | **32.59%** | Class-balanced F1 score | 4-class multi-label distribution | `results/model_metrics.json` |
| | Test ROC AUC (OvR) | **0.6588** | Multi-class continuous prob | Above random baseline (0.50) | `results/model_metrics.json` |
| | Regression MAE | **0.2422** | Future utility score | Bounded $[0, 1]$ target error | `results/model_metrics.json` |
| | Top Feature Importance| **Frequency (46.4%)** | Gini Impurity | Recency: 25.3%, Semantic: 19.2% | `results/feature_importance.json` |
| **Phase 11: Comparison**| Baseline Rule Accuracy| **29.13%** | Fixed linear formula | Predicts mostly COMPRESS | `results/baseline_vs_ml_metrics.json`|
| | Trained ML Accuracy | **44.99%** | Trained Random Forest | **+15.86% absolute improvement** | `results/baseline_vs_ml_metrics.json`|
| | Statistical McNemar | **$p < 0.0001$ ($\chi^2=72.0$)**| Discordant error pairs | Statistically significant gain | `results/baseline_vs_ml_metrics.json`|
| **Phase 12: Experiments**| Multi-Seed Mean Acc | **45.0% $\pm$ 0.4%** | 5 Seeds ($N=1,047$) | Consistent across random seeds | `results/experiments_summary.json` |
| | Noise Robustness | **Stable @ $\sigma=0.10$** | Additive Gaussian perturbation| Generalizes under jitter | `results/experiments_summary.json` |
| | Reproducibility | **$\sigma^2 = 0.0000$** | 3 Consecutive runs | 100% deterministic with seed 42| `results/experiments_summary.json` |
| **Phase 13: Resources** | Standard Processing | **0.0990 s** (2,701 pairs) | Unconstrained baseline | Discovered 303 relationships | `results/resources/resource_evaluation_report.json` |
| | LOW Profile Execution | **0.0116 s** (1,500 pairs) | Resource-constrained mode | **88.3% speedup** | `results/resources/resource_evaluation_report.json` |
| | High-Strength Recall | **89.1%** | Top prioritized candidate pairs| Quality preserved under strain| `results/resources/resource_evaluation_report.json` |
| | Normal Profile Fidelity| **100.0%** (303/303 edges) | Budget: 5,000 pairs | 78.8% speedup via LRU cache | `results/resources/resource_evaluation_report.json` |
| | Critical Nodes Deferred| **0 (0.0%)** | Nodes with $P \ge 0.70$ | Invariant quality protection | `results/resources/resource_evaluation_report.json` |

---

## 2. Publication Figures Manifest

All figures are generated directly from experimental outputs and saved in [`results/figures/`](file:///d:/Research%20Individual%20Comp/results/figures/):

1. **`fig1_baseline_vs_ml_seeds.png`** (101 KB): Test accuracy and Macro F1 comparison with 95% confidence intervals across 5 random seeds.
2. **`fig2_feature_ablation.png`** (109 KB): Macro F1 drop across systematic leave-one-out cognitive feature exclusions.
3. **`fig3_compression_reduction.png`** (89 KB): Log-scale memory volume funnel illustrating 98.14% compression.
4. **`fig4_retention_dynamics.png`** (97 KB): Distribution of consolidated nodes across `DECAYING`, `ARCHIVED`, and `FORGETTABLE` retention states.
5. **`fig5_relationship_topology.png`** (99 KB): Modularity and edge type distribution across 303 discovered relationships.
6. **`fig6_robustness_perturbation.png`** (126 KB): Accuracy degradation curve under increasing feature noise levels ($\sigma \in [0.0, 0.35]$).
7. **`fig7_resource_execution_time_comparison.png`** (98 KB): Execution time comparison between Standard Unconstrained and Resource-Aware profiles showing 88.3% latency reduction.
8. **`fig8_memory_rss_reduction.png`** (85 KB): Process resident set size (RSS) stability across processing stages.
9. **`fig9_relationship_budget_vs_quality.png`** (133 KB): Dual-axis trade-off curve of candidate pair budgets versus relationship recall.

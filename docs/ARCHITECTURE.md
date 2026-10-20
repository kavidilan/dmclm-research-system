# System Architecture — Dynamic Memory Consolidation Learning Model (DMCLM)
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. High-Level Architectural Overview

The **Dynamic Memory Consolidation Learning Model (DMCLM)** provides biologically inspired, multi-dimensional long-term memory consolidation, adaptive retention, relationship topology management, and resource-aware reasoning for personal desktop intelligence agents.

The architecture ingests raw, passive, multi-modal desktop interaction telemetry collected by the **DesktopResearchCollector** (`collector.db`) strictly in **read-only mode**, structures it into episodic units, extracts multi-dimensional cognitive signals, evaluates memory lifecycle priority, consolidates redundant memory bursts, establishes relationship graph topology, trains transparent local machine learning models, and adaptively adjusts computational load based on real-time host system resources.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RAW DESKTOP INTERACTION TELEMETRY                               │
│            DesktopResearchCollector (collector.db: Read-Only ?mode=ro&immutable=1)    │
│            [18,318 Activity Events | 10,543 Browser Events | 8,664 Interactions]       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Phase 1 & Phase 2
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 2: PREPROCESSING & ACTIVITY EPISODE GROUPING                     │
│  ├── Screenshot Deduplication: Exact SHA-256 Hash Matching                             │
│  ├── Interaction Aggregation: Keystrokes & Clicks Aggregated by (Session, App)         │
│  └── Temporal Episode Grouping: Dwell Window tau = 300s                                │
│  ──> 37,559 Raw Records ──> 7,514 Cohesive Activity Episodes (70.69% Noise Reduction)  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Phase 3
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: STRUCTURED MEMORY EXTRACTION                               │
│  ├── Meaningfulness Filter: Dwell >= 10s OR Interactions >= 2 OR Browser Visits        │
│  └── Multi-Modal Context Enrichment: Window context, domains, application metadata    │
│  ──> 6,229 Structured Memory Candidates (data/dmclm.db: memory_candidates)             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Phase 4
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               PHASE 4: FIVE DMCLM COGNITIVE FEATURE EXTRACTION                         │
│  ├── 1. Recency (R)             : Exponential decay relative to reference time         │
│  ├── 2. Frequency (F)           : Log-normalized repetition & interaction density      │
│  ├── 3. Retrieval History (H)   : Local query audit trail (useful / total queries)     │
│  ├── 4. Semantic Relevance (S)  : Local TF-IDF session centroid cosine similarity      │
│  └── 5. Relationship (Rel)      : Temporal proximity & cross-app co-occurrence kernel  │
│  ──> Enriched Feature Records (data/dmclm_features.csv & feature_enriched_memories)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Phase 5
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 5: BASELINE DMCLM DECISION ENGINE                              │
│  ├── Weighted Linear Scoring: S_base = 0.20*R + 0.20*F + 0.25*H + 0.20*S + 0.15*Rel    │
│  └── 4-Way Lifecycle Gating: RETAIN (>=0.70) | COMPRESS (0.40-0.69)                    │
│                              ARCHIVE (0.20-0.39) | FORGET (<0.20)                      │
│  ──> 6,229 Scored Candidates (5,094 COMPRESS | 1,135 ARCHIVE)                          │
└─────────────────────┬───────────────────────────────────────────────────┬──────────────┘
                      │ Phase 6                                           │ Phase 9
                      ▼                                                   ▼
┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────┐
│   PHASE 6: CONSOLIDATION & COMPRESSION       │  │ PHASE 9: TRAINING DATASET GENERATION │
│  ├── Multi-Signal Affinity Clustering        │  │  ├── 38 Total, 22 ML Input Features  │
│  ├── Synthesized LTM Macro Nodes             │  │  ├── Deterministic Temporal Split    │
│  ├── 100% Provenance ID Traceability         │  │  │   (Train: 4,360 | Val: 934 |      │
│  ──> 6,229 Candidates ──> 74 LTM Nodes       │  │  │    Test: 935)                     │
│      (98.14% Volumetric Reduction)           │  │  └── Zero-Leakage Preprocessing      │
└─────────────────────┬────────────────────────┘  └───────────────────┬──────────────────┘
                      │ Phase 7                                       │ Phase 10
                      ▼                                               ▼
┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────┐
│   PHASE 7: ADAPTIVE RETENTION & FORGETTING   │  │   PHASE 10: LOCAL ML MODEL TRAINING  │
│  ├── Continuous Decay with Adaptive Halflife │  │  ├── Random Forest Classifier & Reg. │
│  ├── Resilience Dampening (R_adapt)          │  │  ├── 100 Trees, Depth 8, Seed 42     │
│  ├── 19 Protected Nodes (P >= 0.70 Floor)    │  │  ├── Validation F1: 84.30%           │
│  └── Zero Physical Deletion (Soft States)    │  │  └── Test Accuracy: 94.33%,          │
│  ──> 30 DECAYING | 41 ARCHIVED | 3 FORGET    │  │      Macro F1: 81.36%                │
└─────────────────────┬────────────────────────┘  └───────────────────┬──────────────────┘
                      │ Phase 8                                       │ Phase 11
                      ▼                                               ▼
┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────┐
│     PHASE 8: RELATIONSHIP MANAGEMENT         │  │    PHASE 11: ML VS BASELINE          │
│  ├── 7 Typed Cognitive Edges                 │  │  ├── Fair Held-Out Split Evaluation  │
│  ├── NetworkX Topology: 74 Nodes, 303 Edges  │  │  ├── 94.33% Agreement, Kappa = 0.627 │
│  ├── 4 Connected Components, Density 0.1122  │  │  └── Feature Importance Hierarchy    │
│  └── Isolated Node Detection (3 Nodes)       │  │      (Semantic 44.3%, Freq 30.9%)    │
└─────────────────────┬────────────────────────┘  └───────────────────┬──────────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │ Phase 12 & Phase 13
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 12: SYSTEMATIC EXPERIMENTAL EVALUATION                         │
│  ├── EXP-01: Multi-Seed Stability (94.08% ± 0.31% Test Acc, 95% CI)                   │
│  ├── EXP-02: Systematic Leave-One-Out Ablation (Semantic Loss: -1.05%, Recency: +18.1%)│
│  ├── EXP-03: Volumetric Reduction Funnel (98.14% Compression, 0.8587 Quality Ratio)    │
│  ├── EXP-04: Retention Dynamics & Protection Stability (25.68% Protected Nodes)        │
│  ├── EXP-05: Graph Topology & Modularity Distribution                                  │
│  ├── EXP-06: Robustness Under Noise (92.73% Acc @ sigma=0.10)                          │
│  └── EXP-07: Exact Deterministic Reproducibility (sigma^2 = 0.0000 across 3 trials)    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Phase 13
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 13: RESOURCE-AWARE REASONING & EDGE ADAPTATION                   │
│  ├── System & Process Telemetry via psutil (CPU %, Available RAM, Process RSS/VMS)     │
│  ├── Dynamic Profile Classification (LOW, NORMAL, HIGH)                                │
│  ├── Quality-Aware Processing Budgets (Protected P >= 0.70 Invariance)                 │
│  ├── Relationship Candidate Pruning & Symmetric LRU Cache (Capacity: 5k-20k)          │
│  ├── Zero Permanent Loss Deferred SQLite Queue                                         │
│  └── Real Hardware Benchmark: 88.3% Speedup under LOW Profile, 89.1% High-Strength Rec │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Data Flow Specification (Phases 1–13)

### Phase 1 — Real Collector Data Ingestion
- **Module:** [`backend/src/collector_reader.py`](file:///d:/Research%20Individual%20Comp/backend/src/collector_reader.py)
- **Source:** `C:\Users\Admin\AppData\Local\DesktopResearchCollector\collector.db`
- **Safety Mechanism:** Opened via SQLite URI `file:.../collector.db?mode=ro&immutable=1` with `PRAGMA query_only = ON;`. Zero write locks, zero table alterations.
- **Ingested Data Volume:**
  - `activity_events`: 18,318 rows
  - `browser_events`: 10,543 rows
  - `interaction_events`: 8,664 rows
  - `screenshots`: 34 rows
  - `sessions`: 37 rows
  - Total Raw Events: 37,559 observations.

### Phase 2 — Data Preprocessing & Activity Episodes
- **Module:** [`backend/src/phase2_preprocessor.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase2_preprocessor.py)
- **Inputs:** Raw collector tables from Phase 1.
- **Logic:**
  1. *Screenshot Deduplication:* Computes SHA-256 digests across image blobs; links identical captures to canonical screenshot IDs.
  2. *Interaction Aggregation:* Aggregates 166,766 micro-actions (clicks, keystrokes) into 291 task clusters grouped by `(session_id, application_name)`.
  3. *Activity Episode Construction:* Consolidates temporal bursts within dwell window $\tau = 300\text{ s}$ into cohesive episodes.
- **Outputs:** 7,514 structured activity episodes stored in SQLite table `activity_episodes` in [`data/dmclm.db`](file:///d:/Research%20Individual%20Comp/data/dmclm.db) (70.69% noise reduction from raw observations).

### Phase 3 — Memory Candidate Extraction & Normalization
- **Module:** [`backend/src/phase3_memory_extractor.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase3_memory_extractor.py)
- **Inputs:** `activity_episodes` from Phase 2.
- **Logic:** Evaluates cognitive meaningfulness criteria:
  $$\text{Valid} \iff (\text{duration} \ge 10\text{ s}) \lor (\text{interactions} \ge 2) \lor (\text{browser\_events} > 0)$$
  Attaches window titles, browser URL domains, and engagement metrics into normalized memory schemas.
- **Outputs:** 6,229 validated memory candidates stored in `data/memory_candidates.json` and SQLite table `memory_candidates`.

### Phase 4 — Five DMCLM Cognitive Features
- **Module:** [`backend/src/phase4_feature_extractor.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase4_feature_extractor.py)
- **Inputs:** 6,229 memory candidates.
- **Logic:** Extracts the five normalized cognitive signals in $[0.0, 1.0]$:
  1. *Recency ($R$):* Exponential decay relative to reference time $T_{\text{ref}}$.
  2. *Frequency ($F$):* Log-normalized occurrence and interaction density.
  3. *Retrieval History ($H$):* Local audit ratio of useful queries.
  4. *Semantic Relevance ($S$):* Local TF-IDF n-gram session centroid cosine similarity.
  5. *Relationship Strength ($Rel$):* Co-occurrence and temporal proximity kernel.
- **Outputs:** Stored in [`data/dmclm_features.csv`](file:///d:/Research%20Individual%20Comp/data/dmclm_features.csv) and SQLite table `feature_enriched_memories` (6,229 rows).

### Phase 5 — Baseline DMCLM Decision Engine
- **Module:** [`backend/src/phase5_baseline_dmclm.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase5_baseline_dmclm.py)
- **Inputs:** 6,229 feature-enriched memories.
- **Logic:** Computes weighted importance:
  $$S_{\text{base}} = 0.20 \cdot R + 0.20 \cdot F + 0.25 \cdot H + 0.20 \cdot S + 0.15 \cdot Rel$$
  Assigns 4-way lifecycle state:
  - $S_{\text{base}} \ge 0.70 \implies \text{RETAIN}$
  - $0.40 \le S_{\text{base}} < 0.70 \implies \text{COMPRESS}$
  - $0.20 \le S_{\text{base}} < 0.40 \implies \text{ARCHIVE}$
  - $S_{\text{base}} < 0.20 \implies \text{FORGET}$
- **Outputs:** 5,094 `COMPRESS` (81.78%), 1,135 `ARCHIVE` (18.22%), 0 `RETAIN`, 0 `FORGET`. Stored in [`data/baseline_scored_memories.json`](file:///d:/Research%20Individual%20Comp/data/baseline_scored_memories.json) and SQLite table `baseline_scored_memories`.

### Phase 6 — Memory Consolidation & Compression
- **Module:** [`backend/src/phase6_consolidation_compression.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase6_consolidation_compression.py)
- **Inputs:** 6,229 baseline-scored memories.
- **Logic:** Multi-signal affinity clustering (application match, domain match, context similarity, TF-IDF semantic similarity, temporal proximity). Groups candidates into macro-clusters ($\text{affinity} \ge 0.55$). Synthesizes consolidated Long-Term Memory (LTM) nodes while maintaining 100% provenance ID mapping.
- **Outputs:**
  - Original source memories: 6,229
  - Consolidated LTM nodes: 74 nodes
  - Unconsolidated standalone memories: 42
  - Total retained units: 116 units
  - Volumetric reduction: **98.14%**
  - Average cluster size: 83.61 memories/cluster
  - Stored in [`data/consolidated_memories.json`](file:///d:/Research%20Individual%20Comp/data/consolidated_memories.json) and SQLite table `consolidated_memories`.

### Phase 7 — Adaptive Retention & Cognitive Forgetting
- **Module:** [`backend/src/phase7_adaptive_retention_forgetting.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase7_adaptive_retention_forgetting.py)
- **Inputs:** 74 consolidated memory nodes.
- **Logic:** Evaluates temporal retention decay with adaptive half-life dampening:
  $$\lambda_{\text{adapt}} = \lambda_{\text{base}} \cdot (1 - \alpha \cdot \text{Resilience}), \quad \alpha = 0.60$$
  Protects high-value nodes ($P \ge 0.70$) with a retention floor ($0.40$).
  States: `RETAINED` ($\ge 0.60$), `DECAYING` ($[0.35, 0.60)$), `ARCHIVED` ($[0.20, 0.35)$), `FORGETTABLE` ($< 0.20$).
  *Zero physical deletion:* Forgetting is a soft eligibility status.
- **Outputs:** 30 `DECAYING` (40.54%), 41 `ARCHIVED` (55.41%), 3 `FORGETTABLE` (4.05%), 19 Protected Nodes (25.68%). Stored in [`data/retention_evaluated_memories.json`](file:///d:/Research%20Individual%20Comp/data/retention_evaluated_memories.json) and SQLite table `retention_evaluated_memories`.

### Phase 8 — Memory Relationship Management
- **Module:** [`backend/src/phase8_relationship_management.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase8_relationship_management.py)
- **Inputs:** 74 consolidated memory nodes.
- **Logic:** Computes pairwise multi-modal relationship kernel across 7 typed categories (`SEMANTIC_SIMILARITY`, `SHARED_DOMAIN`, `CROSS_APPLICATION_TASK`, `HYBRID_COMPOSITE`, `TEMPORAL_ASSOCIATION`, `SHARED_APPLICATION`, `WORKFLOW_CONTINUITY`). Constructs NetworkX graph.
- **Outputs:**
  - Nodes: 74, Edges: 303
  - Density: 0.1122, Average Degree: 8.19
  - Connected Components: 4, Isolated Nodes: 3
  - Stored in [`data/memory_relationships.json`](file:///d:/Research%20Individual%20Comp/data/memory_relationships.json) and SQLite table `memory_relationships`.

### Phase 9 — Training Dataset Generation
- **Module:** [`backend/src/phase9_training_dataset.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase9_training_dataset.py)
- **Inputs:** Outputs from Phases 3–8.
- **Logic:** Compiles 38 total features (22 strictly isolated ML input features). Enforces deterministic temporal split (Train 70%: 4,360, Val 15%: 934, Test 15%: 935) to eliminate lookahead data leakage.
- **Outputs:** [`data/training_dataset.csv`](file:///d:/Research%20Individual%20Comp/data/training_dataset.csv) (6,229 rows), [`results/dataset_manifest.json`](file:///d:/Research%20Individual%20Comp/results/dataset_manifest.json) (SHA-256 hash: `c000f6ea7393...`).

### Phase 10 — Local Machine Learning Model Training
- **Module:** [`backend/src/phase10_ml_training.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase10_ml_training.py)
- **Inputs:** Training and validation splits from Phase 9.
- **Logic:** Trains local Random Forest Classifier and Continuous Regressor using scikit-learn Pipeline with RobustScaler and StandardScaler. 100 trees, max depth 8, fixed seed 42.
- **Outputs:**
  - Artifact: [`models/dmclm_lifecycle_model.joblib`](file:///d:/Research%20Individual%20Comp/models/dmclm_lifecycle_model.joblib)
  - Test Accuracy: **94.33%**, Macro F1: **81.36%**, Weighted F1: **95.21%**
  - Regression MAE: **0.0505**, RMSE: **0.0525**
  - Feature Importance: Semantic Relevance (44.26%), Frequency (30.94%), Relationship (15.49%), Recency (9.31%), Retrieval History (0.00%).

### Phase 11 — ML vs Baseline Systematic Comparison
- **Module:** [`backend/src/phase11_comparison.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase11_comparison.py)
- **Inputs:** Held-out test split ($N = 935$).
- **Logic:** Head-to-head empirical comparison between rule-based Baseline and trained Random Forest.
- **Outputs:** Overall Agreement: **94.33%** (882/935 identical predictions), Cohen's Kappa: **0.6270** (substantial agreement). Stored in [`results/baseline_vs_ml_metrics.json`](file:///d:/Research%20Individual%20Comp/results/baseline_vs_ml_metrics.json).

### Phase 12 — Systematic Experiments & Evaluation Suite
- **Module:** [`backend/src/phase12_experiments.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase12_experiments.py)
- **Inputs:** Complete pipeline outputs.
- **Logic:** Executes 7 standardized research experiments (`EXP-01` to `EXP-07`) with multi-seed variance, leave-one-out feature ablation, compression funnel, retention dynamics, graph topology, perturbation stress, and determinism verification.
- **Outputs:** [`results/experiments_summary.json`](file:///d:/Research%20Individual%20Comp/results/experiments_summary.json), 6 publication figures in `results/figures/`.

### Phase 13 — Resource-Aware Reasoning & Edge Adaptation
- **Module:** [`backend/src/phase13_resource_aware.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase13_resource_aware.py)
- **Inputs:** Phase 8 relationship candidate pairs and Phase 10 frozen ML model.
- **Logic:** Observes host hardware via `psutil`. Derives profiles (`LOW`, `NORMAL`, `HIGH`). Enforces candidate pair budgets, symmetric LRU similarity cache, and zero-loss deferred SQLite queue.
- **Outputs:** 88.3% processing speedup under `LOW` profile while retaining 89.1% high-strength relationships. Stored in [`results/resources/resource_evaluation_report.json`](file:///d:/Research%20Individual%20Comp/results/resources/resource_evaluation_report.json) and Figures `fig7`–`fig9`.

---

## 3. Storage Architecture & Provenance Integrity

### SQLite Master Database (`data/dmclm.db`)
The system persists all derived state into a single SQLite database structured into 24 specialized relational tables:
1. `activity_episodes` (7,514 rows) — Preprocessed temporal task units.
2. `preprocessed_events` (11,007 rows) — Canonical deduplicated events.
3. `normalized_browser_events` (10,543 rows) — Domain-extracted navigation.
4. `memory_candidates` (6,229 rows) — Validated cognitive candidates.
5. `feature_enriched_memories` (6,229 rows) — 5-feature enriched records.
6. `baseline_scored_memories` (6,229 rows) — Baseline score and lifecycle states.
7. `consolidated_memories` (74 rows) — Synthesized LTM macro nodes.
8. `retention_evaluated_memories` (74 rows) — Adaptive decay and protection.
9. `memory_relationships` (303 rows) — Typed topological edges.
10. `deferred_memory_queue` — Queue for deferred memories during resource constraints.
11. `resource_monitoring_logs` (6 rows) — Hardware snapshots.
12. `resource_execution_runs` (6 rows) — Benchmarked execution metadata.
13. `experiment_runs` (11 rows) — Multi-seed evaluation records.

### Provenance Mapping Guarantee
Every consolidated LTM node (`c_node_...`) permanently stores the array of underlying source candidate IDs in JSON column `source_memory_ids`. Every candidate memory (`mem_cand_...`) permanently stores its source episode ID (`episode_...`), which links to exact raw `collector.db` row IDs. This guarantees **100% two-way forensic auditability** from high-level memory back to raw collector keystrokes and screenshots.

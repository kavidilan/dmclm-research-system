# Adaptive Privacy-Preserving Cognitive Memory System (DMCLM)
**Individual Research Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka  
**SLIIT Student ID:** IT23208230  
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Final Status:** **Phases 1–14 Completed (100.0%) — All 189 Tests Passing**  

---

## 📌 Executive Summary
This repository contains the complete, production-verified research implementation for **Member 4: Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)** within the final-year research project *“Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence”*.

Personal desktop AI agents continuously observe user keystrokes, window focuses, and browser navigation. Saving raw logs linearly causes exponential storage bloat, downstream retrieval latency, and catastrophic noise. **DMCLM** solves this by implementing biologically inspired cognitive memory consolidation. The pipeline:
1. Ingests real multi-modal desktop interaction telemetry from the **DesktopResearchCollector** SQLite database (`collector.db`) strictly in **read-only mode** (`mode=ro&immutable=1`).
2. Deduplicates screenshots via exact SHA-256 matching and aggregates interaction telemetry (keystrokes and mouse clicks) into cohesive task episodes (**70.69% initial noise reduction**).
3. Extracts five transparent, continuous cognitive signals in $[0.0, 1.0]$ without cloud APIs: **Recency ($R$)**, **Frequency ($F$)**, **Retrieval History ($H$)**, **Semantic Relevance ($S$)**, and **Relationship Strength ($Rel$)**.
4. Evaluates a transparent **Baseline DMCLM** multi-criteria scoring engine with 4-way lifecycle gating (`RETAIN`, `COMPRESS`, `ARCHIVE`, `FORGET`).
5. Consolidates redundant episodic bursts into unified Long-Term Memory (LTM) macro-nodes, achieving a **98.14% volumetric memory compression** while maintaining **100% two-way provenance traceability**.
6. Evaluates continuous adaptive retention decay with cognitive resilience dampening ($\alpha = 0.60$), protects **25.68% of high-value memories**, and enforces a **zero physical deletion policy**.
7. Constructs an interconnected NetworkX cognitive relationship graph discovering **303 typed cognitive edges** across 7 distinct categories.
8. Generates a leak-free chronologically split training dataset (SHA-256: `c000f6ea...`) and trains a local Random Forest model achieving **94.33% Test Accuracy**, **81.36% Macro F1**, and **0.9904 ROC AUC**.
9. Executes 7 systematic empirical research experiments (`EXP-01` to `EXP-07`) verifying multi-seed stability, feature ablation, compression funnels, graph topology, noise robustness, and exact determinism ($\sigma^2 = 0.0000$).
10. Incorporates an empirical **Resource-Aware Reasoning Layer** measuring real host hardware via `psutil`, delivering an **88.3% execution speedup** under resource constraints while preserving **89.1% of high-strength cognitive relationships**.

---

## 🏗️ System Architecture & Data Flow

```text
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

## 🔬 Core Empirical Findings

Every numerical finding reported below is backed by physical artifacts, relational database tables, and serialized models in the repository:

- **Raw Telemetry Volume:** Ingested 37,559 raw events (18,318 activity, 10,543 browser, 8,664 interactions, 34 screenshots) in read-only mode.
- **Preprocessing Efficiency:** 7,514 activity episodes constructed with **70.69% noise reduction**.
- **Cognitive Memory Candidates:** 6,229 candidates extracted meeting dwell time and interaction thresholds.
- **Consolidation & Volumetric Compression:** 6,229 candidate memories consolidated into **74 Long-Term Memory (LTM) macro-nodes** + 42 standalone nodes (**116 total retained units**), achieving a **98.14% volumetric compression ratio** with **100% provenance ID traceability**.
- **Adaptive Retention Dynamics:** Evaluated across 74 nodes: 30 `DECAYING`, 41 `ARCHIVED`, 3 `FORGETTABLE`, with **19 nodes (25.68%) clamped to an invariant protection floor** ($0.40$). **0 records deleted** (zero-loss soft deletion).
- **Cognitive Relationship Graph:** NetworkX topology discovers **303 typed edges** across 7 distinct classes with graph density $0.1122$ and 4 connected components.
- **Dataset Generation & Training:** 6,229 rows, 22 ML input features, chronological partition (Train 4,360, Val 934, Test 935), SHA-256 hash `c000f6ea...`.
- **Machine Learning Generalization:** Random Forest model achieves **94.33% Test Accuracy**, **81.36% Macro F1**, **0.9904 ROC AUC**, and continuous regression MAE of **0.0505**.
- **Feature Importance Hierarchy:** Semantic Relevance (44.3%) and Frequency (30.9%) dominate decision boundaries, followed by Relationship Strength (15.5%) and Recency (9.3%).
- **Multi-Seed Stability:** Across 5 random seeds, Test Accuracy is **94.08% $\pm$ 0.31%** (95% CI: $[93.80\%, 94.35\%]$); Determinism across 3 identical trials yielded $\sigma^2 = 0.0000$.
- **Resource-Aware Performance:** Hardware-adaptive execution under the `LOW` profile delivers an **88.3% speedup** (0.0990s $\to$ 0.0116s) while maintaining an **89.1% recall of high-strength cognitive relationships** and zero critical memory loss.

---

## 📁 Repository Structure & Research Deliverables

```text
d:\Research Individual Comp\
├── data/                               # Master SQLite DB & Structured JSON/CSV Stores
│   ├── dmclm.db                        # Relational database (24 tables, 50 MB)
│   ├── memory_candidates.json          # 6,229 extracted memory candidates
│   ├── dmclm_features.csv              # Extracted 5-dimensional cognitive features
│   ├── baseline_scored_memories.json   # Rule-scored memories (5,094 COMPRESS / 1,135 ARCHIVE)
│   ├── consolidated_memories.json      # 74 synthesized LTM macro nodes (98.14% compression)
│   ├── retention_evaluated_memories.json # Retention evaluated records (19 protected nodes)
│   ├── memory_relationships.json       # 303 typed cognitive edges
│   └── training_dataset.csv            # 6,229 ML training rows (22 input features)
├── docs/                               # Phase 14 Comprehensive Research Documentation Package
│   ├── ARCHITECTURE.md                 # End-to-end pipeline architecture specification
│   ├── METHODOLOGY.md                  # Detailed mathematical formulas & algorithms
│   ├── RESULTS_SUMMARY.md              # Quantitative master evidence table across all phases
│   ├── LIMITATIONS.md                  # Honest evaluation of threats to validity & constraints
│   ├── REPRODUCIBILITY.md              # Exact Windows PowerShell commands to reproduce
│   ├── TESTING_AND_VERIFICATION.md     # Verification report (189 tests passing, 100%)
│   ├── RESEARCH_EVIDENCE_INDEX.md      # Claim-to-file traceability mapping
│   ├── VIVA_PRESENTATION_GUIDE.md      # Presentation talking points, elevator pitch, defense Q&A
│   ├── RESEARCH_CONTRIBUTION.md        # Formal statement of Member 4's research novelty
│   ├── PHASE_COMPLETION_MATRIX.md      # Verification matrix across Phases 1–14
│   ├── DATA_AND_PRIVACY_STATEMENT.md   # Privacy governance & local execution principles
│   ├── KNOWN_ISSUES_AND_FUTURE_WORK.md # Constraints & future research directions
│   └── INDIVIDUAL_COMPONENT_COMPLETION_REPORT.md # Master component completion report
├── models/                             # Trained Machine Learning Model Artifacts
│   ├── dmclm_lifecycle_model.joblib    # Serialized Random Forest classification & reg pipeline
│   └── model_metadata.json             # Hyperparameters, metrics, and training manifest
├── results/                            # JSON/CSV Reports and Publication Figures
│   ├── figures/                        # 9 Publication-Grade Research Figures (PNG)
│   │   ├── fig1_baseline_vs_ml_seeds.png
│   │   ├── fig2_feature_ablation.png
│   │   ├── fig3_compression_reduction.png
│   │   ├── fig4_retention_dynamics.png
│   │   ├── fig5_relationship_topology.png
│   │   ├── fig6_robustness_perturbation.png
│   │   ├── fig7_resource_execution_time_comparison.png
│   │   ├── fig8_memory_rss_reduction.png
│   │   └── fig9_relationship_budget_vs_quality.png
│   ├── resources/                      # Phase 13 Resource-Aware Benchmarking Outputs
│   ├── baseline_vs_ml_metrics.json     # Head-to-head empirical comparison metrics
│   ├── experiments_summary.json        # Phase 12 systematic experiments summary
│   └── consolidation_metrics.json      # Compression and cluster size statistics
├── backend/                            # Python FastAPI backend
│   ├── package.json                    # npm start launcher for Uvicorn
│   ├── requirements.txt                # Dependencies verified in the existing .venv
│   ├── start.js                        # Uses the root .venv and backend/src
│   ├── src/                            # Existing research modules and api_server.py
│   └── tests/                          # Existing backend test suite
├── frontend/                           # React Research Dashboard (Vite + Tailwind CSS)
├── DMCLM_RESEARCH_SPECIFICATION.md     # In-depth architectural & empirical specification
├── PROJECT_PLAN.md                     # Research milestone completion tracker
└── README.md                           # Master project README
```

---

## 🚀 Quickstart & Verification Guide

### 1. Environment Setup & Dependencies
```powershell
pip install numpy pandas scikit-learn joblib networkx psutil matplotlib fastapi uvicorn pydantic
```

### 2. Run the Full Test Suite
```powershell
python -m unittest discover -s backend/tests -v
```
The existing `.venv` must have the project test dependencies installed before running the suite.

### 3. Run Pipeline Phases Individually
```powershell
python backend/src/phase2_preprocessor.py                  # Phase 2 Preprocessing
python backend/src/phase3_memory_extractor.py              # Phase 3 Candidate Extraction
python backend/src/phase4_feature_extractor.py             # Phase 4 5D Feature Engineering
python backend/src/phase5_baseline_dmclm.py                # Phase 5 Baseline Importance Scoring
python backend/src/phase6_consolidation_compression.py     # Phase 6 Consolidation & Compression
python backend/src/phase7_adaptive_retention_forgetting.py # Phase 7 Adaptive Retention & Protection
python backend/src/phase8_relationship_management.py       # Phase 8 Relationship Management
python backend/src/phase9_training_dataset.py              # Phase 9 Training Dataset Generation
python backend/src/phase10_ml_training.py                  # Phase 10 ML Training & Evaluation
python backend/src/phase11_comparison.py                   # Phase 11 Baseline vs ML Comparison
python backend/src/phase12_experiments.py                  # Phase 12 Research Experiments Suite
python backend/src/phase13_resource_aware.py               # Phase 13 Resource-Aware Hardware Benchmark
```

### 4. Run the Full Pipeline End-to-End
```powershell
$env:PYTHONPATH = "$PWD\backend"
python -m src.run_dmclm_pipeline
```

### 5. Launch Interactive Research Dashboard
```powershell
# Terminal 1: Backend API (FastAPI)
cd backend
npm start

# Terminal 2: Frontend Dashboard (Vite)
cd frontend
npm run dev
```
Open `http://127.0.0.1:5173/` to explore the interactive visual analytics, memory cluster graph, and live resource-aware execution telemetry. API documentation is at `http://127.0.0.1:8000/docs`.

---

## 🛡️ Privacy & Local Security Guarantee
- **100% Local Execution:** All database queries, TF-IDF vectorization, feature calculations, ML inference, and clustering execute locally on the host machine.
- **Zero Cloud Transmission:** Zero user interaction text, window titles, URLs, keystrokes, or screenshots are ever transmitted to external cloud APIs or servers.
- **Read-Only Raw Isolation:** The raw DesktopResearchCollector database (`collector.db`) is accessed strictly in read-only mode (`?mode=ro&immutable=1`) with `PRAGMA query_only = ON;`, guaranteeing that raw behavioral logs remain permanently immutable.

---

## 📚 Detailed Documentation Links
For deep academic study, supervisor inspection, and viva preparation, consult the specialized guides in [`docs/`](file:///d:/Research%20Individual%20Comp/docs/):
- [**Master Completion Report**](file:///d:/Research%20Individual%20Comp/docs/INDIVIDUAL_COMPONENT_COMPLETION_REPORT.md)
- [**System Architecture Specification**](file:///d:/Research%20Individual%20Comp/docs/ARCHITECTURE.md)
- [**Research Methodology & Mathematical Formulas**](file:///d:/Research%20Individual%20Comp/docs/METHODOLOGY.md)
- [**Quantitative Results Summary**](file:///d:/Research%20Individual%20Comp/docs/RESULTS_SUMMARY.md)
- [**Limitations & Threats to Validity**](file:///d:/Research%20Individual%20Comp/docs/LIMITATIONS.md)
- [**Reproducibility Guide**](file:///d:/Research%20Individual%20Comp/docs/REPRODUCIBILITY.md)
- [**Testing & Verification Report**](file:///d:/Research%20Individual%20Comp/docs/TESTING_AND_VERIFICATION.md)
- [**Research Evidence Index**](file:///d:/Research%20Individual%20Comp/docs/RESEARCH_EVIDENCE_INDEX.md)
- [**Viva & Presentation Guide for Member 4**](file:///d:/Research%20Individual%20Comp/docs/VIVA_PRESENTATION_GUIDE.md)
- [**Research Contribution Statement**](file:///d:/Research%20Individual%20Comp/docs/RESEARCH_CONTRIBUTION.md)
- [**Phase Completion Matrix**](file:///d:/Research%20Individual%20Comp/docs/PHASE_COMPLETION_MATRIX.md)
- [**Data Governance & Privacy Statement**](file:///d:/Research%20Individual%20Comp/docs/DATA_AND_PRIVACY_STATEMENT.md)
- [**Known Issues & Future Work**](file:///d:/Research%20Individual%20Comp/docs/KNOWN_ISSUES_AND_FUTURE_WORK.md)

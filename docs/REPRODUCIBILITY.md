# Reproducibility Guide — DMCLM Research Pipeline
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. System Requirements & Environment Setup

### 1.1 Hardware & OS Specifications
- **Operating System:** Windows 10 / Windows 11 (64-bit)
- **Processor:** Multi-core x86_64 CPU (4+ physical cores recommended)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** At least 2 GB free disk space

### 1.2 Software Prerequisites
- **Python:** Python 3.10+ (tested on Python 3.10/3.11/3.12 64-bit)
- **Node.js:** Node.js 18+ and npm (for frontend research dashboard)
- **Collector Database:** DesktopResearchCollector SQLite database located at:
  `C:\Users\Admin\AppData\Local\DesktopResearchCollector\collector.db`

### 1.3 Python Dependencies
Install required packages using PowerShell from project root (`d:\Research Individual Comp`):
```powershell
pip install numpy pandas scikit-learn joblib networkx psutil matplotlib fastapi uvicorn pydantic
```

---

## 2. Step-by-Step Phase Execution Commands

All commands must be executed from the repository root: `D:\Research Individual Comp`.

### Phase 2 — Data Preprocessing & Activity Episode Construction
Ingests `collector.db` in read-only mode, aggregates keystrokes/clicks, deduplicates screenshots, and builds episodic units:
```powershell
python backend/src/phase2_preprocessor.py
```
- **Expected Output:** Synthesizes 7,514 activity episodes in `data/dmclm.db` (table `activity_episodes`) and generates `results/preprocessing_report.json`.

### Phase 3 — Structured Memory Candidate Extraction
Applies cognitive meaningfulness criteria to filter noise and extract candidate memories:
```powershell
python backend/src/phase3_memory_extractor.py
```
- **Expected Output:** 6,229 memory candidates in `data/memory_candidates.json` and SQLite table `memory_candidates`.

### Phase 4 — Five DMCLM Cognitive Feature Extraction
Extracts Recency, Frequency, Retrieval History, Semantic Relevance, and Relationship Strength:
```powershell
python backend/src/phase4_feature_extractor.py
```
- **Expected Output:** `data/dmclm_features.csv` and SQLite table `feature_enriched_memories` with 6,229 rows.

### Phase 5 — Baseline DMCLM Decision Engine
Applies weighted multi-criteria importance scoring and 4-way lifecycle state gating:
```powershell
python backend/src/phase5_baseline_dmclm.py
```
- **Expected Output:** `data/baseline_scored_memories.json` (5,094 COMPRESS, 1,135 ARCHIVE) and `results/baseline_dmclm_report.json`.

### Phase 6 — Memory Consolidation & Compression
Performs multi-signal affinity clustering, synthesizes LTM macro nodes, and records compression ratios:
```powershell
python backend/src/phase6_consolidation_compression.py
```
- **Expected Output:** `data/consolidated_memories.json` (74 LTM nodes, 98.14% volumetric compression) and `results/consolidation_metrics.json`.

### Phase 7 — Adaptive Retention & Cognitive Forgetting
Evaluates temporal retention decay, resilience dampening, and high-value protection:
```powershell
python backend/src/phase7_adaptive_retention_forgetting.py
```
- **Expected Output:** `data/retention_evaluated_memories.json` (30 DECAYING, 41 ARCHIVED, 3 FORGETTABLE, 19 protected nodes) and `results/retention_metrics.json`.

### Phase 8 — Memory Relationship Management
Builds the multi-modal cognitive relationship topology and graph statistics:
```powershell
python backend/src/phase8_relationship_management.py
```
- **Expected Output:** `data/memory_relationships.json` (74 nodes, 303 typed edges, 4 components) and `results/relationship_metrics.json`.

### Phase 9 — Training Dataset Generation
Compiles 38 features (22 ML inputs), enforces zero-leakage chronological split, and generates manifest:
```powershell
python backend/src/phase9_training_dataset.py
```
- **Expected Output:** `data/training_dataset.csv` (6,229 rows: Train 4,360, Val 934, Test 935) and `results/dataset_manifest.json` (SHA-256: `c000f6ea...`).

### Phase 10 — Local Machine Learning Model Training
Trains the Random Forest classifier and regressor pipeline and evaluates validation/test metrics:
```powershell
python backend/src/phase10_ml_training.py
```
- **Expected Output:** Serialized artifact `models/dmclm_lifecycle_model.joblib` and metrics in `results/model_metrics.json` (94.33% Test Accuracy).

### Phase 11 — ML vs Baseline Systematic Comparison
Performs head-to-head evaluation on the held-out test split ($N = 935$):
```powershell
python backend/src/phase11_comparison.py
```
- **Expected Output:** `results/baseline_vs_ml_metrics.json` and Markdown report `results/baseline_vs_ml_report.md` (94.33% agreement, Kappa 0.6270).

### Phase 12 — Systematic Experiments & Evaluation Suite
Runs all 7 research experiments (`EXP-01` to `EXP-07`) and generates research figures:
```powershell
python backend/src/phase12_experiments.py
```
- **Expected Output:** `results/experiments_summary.json` and Figures `fig1`–`fig6` in `results/figures/`.

### Phase 13 — Resource-Aware Reasoning & Edge Adaptation
Runs hardware monitoring, profile classification, budgeted execution, and comparative benchmarking:
```powershell
python backend/src/phase13_resource_aware.py
```
- **Expected Output:** `results/resources/resource_evaluation_report.json`, Markdown report `results/resources/phase13_resource_aware_report.md`, and Figures `fig7`–`fig9`.

---

## 3. End-to-End One-Step Execution

To execute the entire DMCLM consolidation pipeline end-to-end in a single command:
```powershell
$env:PYTHONPATH = "$PWD\backend"
python -m src.run_dmclm_pipeline
```

---

## 4. Test Suite Execution

### 4.1 Run the Complete Test Suite (189 Tests)
```powershell
python -m unittest discover -s backend/tests -v
```
- **Expected Output:** All 189 tests across 13 test suites pass with **0 failures and 0 errors** in ~35–55 seconds.

### 4.2 Run Phase-Specific Test Suites
```powershell
# Phase 2 Preprocessing (8 tests)
python -m unittest backend/tests/test_phase2_preprocessing.py -v

# Phase 3 Memory Extraction (13 tests)
python -m unittest backend/tests/test_phase3_memory_extraction.py -v

# Phase 4 Feature Extraction (16 tests)
python -m unittest backend/tests/test_phase4_feature_extraction.py -v

# Phase 5 Baseline DMCLM (15 tests)
python -m unittest backend/tests/test_phase5_baseline_dmclm.py -v

# Phase 6 Consolidation & Compression (17 tests)
python -m unittest backend/tests/test_phase6_consolidation_compression.py -v

# Phase 7 Adaptive Retention & Forgetting (17 tests)
python -m unittest backend/tests/test_phase7_adaptive_retention_forgetting.py -v

# Phase 8 Relationship Management (17 tests)
python -m unittest backend/tests/test_phase8_relationship_management.py -v

# Phase 9 Training Dataset (14 tests)
python -m unittest backend/tests/test_phase9_training_dataset.py -v

# Phase 10 ML Model Training (16 tests)
python -m unittest backend/tests/test_phase10_ml_training.py -v

# Phase 11 Comparison Suite (18 tests)
python -m unittest backend/tests/test_phase11_comparison.py -v

# Phase 12 Systematic Experiments (10 tests)
python -m unittest backend/tests/test_phase12_experiments.py -v

# Phase 13 Resource-Aware Reasoning (12 tests)
python -m unittest backend/tests/test_phase13_resource_aware.py -v

# Integration Pipeline (16 tests)
python -m unittest backend/tests/test_dmclm_pipeline.py -v
```

---

## 5. Running the Research UI Dashboard

### Start Backend API Server (FastAPI on Port 8000)
```powershell
cd backend
npm start
```

### Start Frontend UI (Vite on Port 5173)
In a separate terminal:
```powershell
cd frontend
npm run dev
```
Open your browser at `http://127.0.0.1:5173/` to explore the interactive research dashboard, memory graph topology, consolidation analytics, and real-time resource-aware monitoring telemetry. API documentation is at `http://127.0.0.1:8000/docs`.

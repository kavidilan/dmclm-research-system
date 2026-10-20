# Research Methodology — Dynamic Memory Consolidation Learning Model (DMCLM)
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. Mathematical Definition of the Five Core DMCLM Features

The DMCLM cognitive model is grounded in five biologically inspired, transparent features extracted locally from desktop interaction logs without external API dependencies. All five features are bounded within $[0.0, 1.0]$.

### 1.1 Recency ($R$)
- **Purpose:** Quantifies how recently an interaction episode occurred, modeling human Ebbinghaus exponential decay.
- **Implemented Formula:**
  $$R(t) = \exp\left(-\lambda \cdot \Delta t_{\text{days}}\right)$$
  where $\lambda = 0.05$ (configurable in `dmclm_config.py`), and:
  $$\Delta t_{\text{days}} = \frac{T_{\text{ref}} - t_{\text{last\_seen}}}{86,400\text{ seconds}}$$
  $T_{\text{ref}}$ is the dataset reference timestamp ($\max(t_{\text{end}})$ across all observed episodes).
- **Normalized Range:** $[0.0, 1.0]$, where $1.0$ indicates immediate temporal proximity.
- **Evidence Used:** `last_seen` timestamp of the activity episode from `activity_events`.
- **Limitation:** In retrospective batch processing, recency measures distance to session termination rather than runtime wall-clock time.

### 1.2 Frequency ($F$)
- **Purpose:** Captures the behavioral intensity and habitual repetition of a task across user workflows.
- **Implemented Formula:**
  $$F = 0.75 \cdot \frac{\ln(1 + \text{occurrence\_count})}{\ln(1 + \text{max\_occurrence})} + 0.25 \cdot \frac{\ln(1 + \text{interaction\_count})}{\ln(1 + \text{max\_interaction})}$$
  where $\text{occurrence\_count}$ is the episode repetition count, and $\text{interaction\_count}$ is the aggregated number of keystrokes and mouse clicks.
- **Normalized Range:** $[0.0, 1.0]$.
- **Evidence Used:** Raw event counts grouped by application and window title; aggregated interaction telemetry.
- **Limitation:** Logarithmic dampening prevents extreme interaction spikes (e.g., rapid gaming or typing tests) from overwhelming the signal, but passive background apps with infrequent clicks receive lower frequency scores.

### 1.3 Retrieval History ($H$)
- **Purpose:** Models active memory reinforcement based on explicit downstream queries by the user or retrieval agents (Member 3).
- **Implemented Formula:**
  $$H = \begin{cases} \frac{\text{useful\_retrievals}}{\text{total\_retrievals}}, & \text{if } \text{total\_retrievals} > 0 \\ 0.0, & \text{if } \text{total\_retrievals} = 0 \end{cases}$$
- **Normalized Range:** $[0.0, 1.0]$.
- **Evidence Used:** Local DMCLM audit trail persisted in SQLite table `retrieval_events` and `data/retrieval_history.json`.
- **Limitation:** In retrospective historical logs before interactive agent deployment, $\text{total\_retrievals} = 0$, yielding a cold-start value $H = 0.0$ for all initial records. It activates dynamically during runtime agent queries.

### 1.4 Semantic Relevance ($S$)
- **Purpose:** Evaluates whether a memory aligns with the user's primary topical focus during the active session.
- **Implemented Formula:**
  $$S = \cos(v_i, C_{\text{session}(m_i)}) = \frac{v_i \cdot C_{\text{session}(m_i)}}{\|v_i\|_2 \|C_{\text{session}(m_i)}\|_2}$$
  where $v_i$ is the TF-IDF feature vector of memory candidate $m_i$ (derived from window title, application name, domain, and OCR keywords), and $C_{\text{session}}$ is the centroid of all TF-IDF vectors within that user session:
  $$C_{\text{session}} = \frac{1}{|M_{\text{session}}|} \sum_{m \in M_{\text{session}}} v_m$$
- **Normalized Range:** $[0.0, 1.0]$.
- **Evidence Used:** Textual window titles, browser page titles, and application names processed via a local scikit-learn `TfidfVectorizer` (sublinear TF scaling, max 1,000 features, English stop-word filtering).
- **Limitation:** Vectorizer operates on lexical n-grams; highly synonymous but lexically distinct concepts without overlapping tokens receive lower similarity.

### 1.5 Relationship Strength ($Rel$)
- **Purpose:** Quantifies multi-modal structural and contextual connectedness with surrounding episodic memories.
- **Implemented Formula:**
  $$Rel(m_i) = \frac{1}{|N(m_i)|} \sum_{m_j \in N(m_i)} \left(0.35 \cdot \text{AppMatch}(m_i, m_j) + 0.35 \cdot \text{DomainMatch}(m_i, m_j) + 0.30 \cdot \exp\left(-\frac{\Delta t_{ij}}{600\text{s}}\right)\right)$$
  where $N(m_i)$ denotes the set of temporal neighbor memories occurring within a $\pm 30\text{ minute}$ window.
- **Normalized Range:** $[0.0, 1.0]$.
- **Evidence Used:** Application executable names, browser domain matches, and temporal timestamps.
- **Limitation:** Isolated single-application sessions with sparse browser visits exhibit lower relationship strength scores.

---

## 2. Baseline DMCLM Decision Engine Methodology

### 2.1 Formula and Configured Weights
The Baseline DMCLM engine uses a transparent, linear multi-criteria weighting model:
$$S_{\text{base}} = w_R \cdot R + w_F \cdot F + w_H \cdot H + w_S \cdot S + w_{\text{Rel}} \cdot Rel$$
Configured weights in [`backend/src/dmclm_config.py`](file:///d:/Research%20Individual%20Comp/backend/src/dmclm_config.py) (`BaselineImportanceWeights`):
- $w_R = 0.20$ (Recency)
- $w_F = 0.20$ (Frequency)
- $w_H = 0.25$ (Retrieval History)
- $w_S = 0.20$ (Semantic Relevance)
- $w_{\text{Rel}} = 0.15$ (Relationship Strength)
$$\sum w_i = 1.00$$

### 2.2 Four-Way Lifecycle Gating
Memories are assigned deterministic lifecycle actions:
- **$\mathbf{RETAIN}$ ($S_{\text{base}} \ge 0.70$):** High-importance working memory retained in fast-access primary store.
- **$\mathbf{COMPRESS}$ ($0.40 \le S_{\text{base}} < 0.70$):** Moderate-to-high importance episodic bursts suitable for multi-memory consolidation into unified LTM nodes.
- **$\mathbf{ARCHIVE}$ ($0.20 \le S_{\text{base}} < 0.40$):** Lower-importance historical memories moved to secondary disk storage.
- **$\mathbf{FORGET}$ ($S_{\text{base}} < 0.20$):** Transient noise candidates eligible for cognitive pruning.

### 2.3 Actual Dataset Distribution
Evaluated across all 6,229 candidate memories:
- `COMPRESS`: **5,094 records (81.78%)** — reflects intensive, repetitive web research and development tasks.
- `ARCHIVE`: **1,135 records (18.22%)** — represents short, peripheral application switches.
- `RETAIN`: **0 records (0.00%)** — due to retrospective $H = 0.0$ cold-start; maximum observed score was $0.6518$.
- `FORGET`: **0 records (0.00%)** — lowest observed baseline score was $0.2223$.

---

## 3. Consolidation and Compression Methodology

### 3.1 Multi-Signal Affinity Clustering
Memories designated for consolidation are clustered using a composite affinity metric:
$$\text{Affinity}(m_a, m_b) = 0.20 \cdot \text{AppMatch} + 0.15 \cdot \text{DomainMatch} + 0.25 \cdot \text{ContextSim} + 0.25 \cdot \text{SemanticSim} + 0.10 \cdot \text{TemporalSim} + 0.05 \cdot \text{SessionMatch}$$
Clustering parameters:
- Affinity Threshold: $\ge 0.55$
- Minimum Group Size: 2 memories
- Temporal Window: 86,400 seconds (24 hours)

### 3.2 LTM Macro Node Synthesis
Each cluster is consolidated into a single Long-Term Memory (LTM) node:
- **Title & Summary:** Synthesized from the most frequent window context and topical domain.
- **Temporal Bounds:** $\min(\text{start\_time})$ to $\max(\text{end\_time})$.
- **Metrics Aggregation:** Sum of durations, sum of interactions, union of distinct applications and domains.
- **Provenance Traceability:** Complete list of source candidate IDs persisted in JSON field `source_memory_ids`.

### 3.3 Compression Metrics
From [`results/consolidation_metrics.json`](file:///d:/Research%20Individual%20Comp/results/consolidation_metrics.json):
- Source Memories: **6,229**
- Consolidated LTM Nodes: **74**
- Unconsolidated Standalone Nodes: **42**
- Total Retained Units: **116**
- Redundant Traces Reduced: **6,113**
- **Volumetric Compression Ratio:** **98.14%**
- Group-Only Compression Ratio: **98.80%**
- Average Group Size: **83.61 memories/cluster**

---

## 4. Adaptive Retention and Forgetting Methodology

### 4.1 Continuous Adaptive Decay Formula
Rather than a naive time-window purge, retention strength $S_{\text{ret}}(t)$ evolves continuously:
$$S_{\text{ret}}(t) = S_{\text{init}} \cdot \exp\left(-\lambda_{\text{adapt}} \cdot \Delta t_{\text{days}}\right)$$
where initial value $S_{\text{init}} = 0.50 \cdot S_{\text{base}} + 0.50 \cdot \text{Resilience}$, and the decay rate is adaptively dampened:
$$\lambda_{\text{adapt}} = \lambda_{\text{base}} \cdot (1 - \alpha \cdot \text{Resilience}), \quad \alpha = 0.60, \quad \lambda_{\text{base}} = 0.05$$
Resilience incorporates multi-signal cognitive reinforcement:
$$\text{Resilience} = 0.30 \cdot S_{\text{base}} + 0.20 \cdot F + 0.20 \cdot H + 0.15 \cdot S + 0.15 \cdot Rel$$

### 4.2 Cognitive Retention States & Thresholds
- **`RETAINED`** ($S_{\text{ret}} \ge 0.60$): Active, readily accessible tier.
- **`DECAYING`** ($0.35 \le S_{\text{ret}} < 0.60$): Intermediate tier undergoing natural cognitive attenuation.
- **`ARCHIVED`** ($0.20 \le S_{\text{ret}} < 0.35$): Dormant tier preserved in secondary index.
- **`FORGETTABLE`** ($S_{\text{ret}} < 0.20$): Inactive tier eligible for logical exclusion.

### 4.3 High-Value Protection & Zero Deletion
- **Protection Floor:** Nodes with baseline score $\ge 0.60$ or composite quality $P \ge 0.70$ are clamped to a minimum retention floor of $0.40$, preventing degradation to `ARCHIVED` or `FORGETTABLE`.
- **Zero Physical Deletion:** The system **never physically deletes** records. `FORGETTABLE` status is a soft query-exclusion flag in metadata. All raw collector data and derived LTM nodes remain intact on disk.

### 4.4 Measured Retention Distribution
From [`results/retention_metrics.json`](file:///d:/Research%20Individual%20Comp/results/retention_metrics.json) ($N = 74$ consolidated nodes):
- `RETAINED`: **0 (0.00%)**
- `DECAYING`: **30 (40.54%)**
- `ARCHIVED`: **41 (55.41%)**
- `FORGETTABLE`: **3 (4.05%)**
- **Protected Nodes:** **19 nodes (25.68%)**
- Mean Retention Strength: **0.3105** (range: $[0.1463, 0.5094]$).

---

## 5. Relationship Management Methodology

### 5.1 Seven Typed Cognitive Relationship Categories
Pairwise edges between consolidated memories are classified into 7 distinct relationship types:
1. `SEMANTIC_SIMILARITY`: Lexical/topical cosine similarity $\ge 0.40$ (83 edges).
2. `SHARED_DOMAIN`: Identical browser web domains (111 edges).
3. `CROSS_APPLICATION_TASK`: Different applications operating within the same task episode (39 edges).
4. `HYBRID_COMPOSITE`: Multi-signal co-occurrence meeting $\ge 3$ criteria (37 edges).
5. `TEMPORAL_ASSOCIATION`: Temporal delta $\Delta t \le 1\text{ hour}$ (8 edges).
6. `SHARED_APPLICATION`: Common desktop desktop executable without shared domain (1 edge).
7. `WORKFLOW_CONTINUITY`: Sequential transition between known complementary tools (24 edges).

### 5.2 Relationship Strength Calculation
$$w(e_{ij}) = 0.30 \cdot \text{SemanticSim} + 0.25 \cdot \text{ContextSim} + 0.20 \cdot \text{AppMatch} + 0.15 \cdot \text{DomainMatch} + 0.10 \cdot \text{TemporalSim}$$

### 5.3 Topology & Graph Statistics
Constructed using NetworkX and persisted to SQLite table `memory_relationships`:
- Total Nodes: **74**
- Total Directed/Undirected Edges: **303**
- Graph Density: **0.1122**
- Average Node Degree: **8.19**
- Average Relationship Strength: **0.4248**
- Connected Components: **4**
- Isolated Nodes: **3** (nodes with degree = 0).

---

## 6. Training Dataset Methodology

### 6.1 Schema & Feature Separation
The dataset in [`data/training_dataset.csv`](file:///d:/Research%20Individual%20Comp/data/training_dataset.csv) contains 6,229 rows and 38 total columns:
- **ML Input Features (22 columns):**
  - Core features: `feature_recency`, `feature_frequency`, `feature_retrieval_history`, `feature_semantic_relevance`, `feature_relationship_strength`
  - Engagement signals: `duration_ms`, `duration_seconds`, `interaction_count`, `browser_event_count`, `screenshot_count`, `has_browser_activity`, `has_window_context`
  - Temporal signals: `hour_of_day`, `day_of_week`, `is_weekend`
  - Graph signals: `is_consolidated`, `cluster_member_count`, `graph_degree`, `graph_avg_rel_strength`, `graph_max_rel_strength`, `is_graph_isolated`
  - Categorical context: `primary_application`
- **Target Variables (Excluded from Model Input):**
  - Categorical: `target_lifecycle_state` (`COMPRESS` vs `ARCHIVE`)
  - Continuous: `target_importance_score` ($S_{\text{base}}$)
  - Secondary targets: `target_consolidation_priority`, `target_retention_state`
- **Metadata (Excluded):** `candidate_id`, `session_id`, `start_time`, `end_time`.

### 6.2 Data Leakage Prevention & Temporal Splitting
To prevent future lookahead leakage, splitting strictly sorts candidate memories by `start_time`:
- **Train (70%):** 4,360 records ($t < t_{\text{val}}$)
- **Validation (15%):** 934 records ($t_{\text{val}} \le t < t_{\text{test}}$)
- **Test (15%):** 935 records ($t \ge t_{\text{test}}$)
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Dataset SHA-256 Checksum: `c000f6ea7393e997504d418d297efe6bc433c8f455040f82a1adf615da4d559a`.

### 6.3 Target Label Formulation (Weak Supervision Proxy)
Labels are derived from Phase 5 Baseline DMCLM rules. In private desktop systems without invasive human labeling questionnaires, baseline rules act as an objective, weak-supervision proxy. The machine learning model learns non-linear feature interactions and robustness across noise that rule boundaries cannot capture.

---

## 7. Machine Learning Model Methodology

### 7.1 Algorithm Selection
- **Model:** Random Forest Classifier and Random Forest Regressor (`sklearn.ensemble.RandomForestClassifier`, `RandomForestRegressor`).
- **Rationale:** High interpretability, natural handling of tabular feature skew, resistance to overfitting via bagging, and built-in Gini impurity feature importances. Does not require black-box pre-trained transformer embeddings.

### 7.2 Training Configuration & Preprocessing Pipeline
- Serialization: [`models/dmclm_lifecycle_model.joblib`](file:///d:/Research%20Individual%20Comp/models/dmclm_lifecycle_model.joblib)
- Hyperparameters: `n_estimators=100`, `max_depth=8`, `min_samples_split=5`, `class_weight='balanced'`, `random_state=42`.
- Preprocessing: Numerical features scaled with `RobustScaler` (median/IQR scaling resilient to interaction burst outliers); categorical `primary_application` encoded via `OneHotEncoder(handle_unknown='ignore')`.

### 7.3 Empirical Feature Importance Hierarchy
Gini impurity feature importances from the trained model:
1. `feature_semantic_relevance`: **44.26%**
2. `feature_frequency`: **30.94%**
3. `feature_relationship_strength`: **15.49%**
4. `feature_recency`: **9.31%**
5. `feature_retrieval_history`: **0.00%** (zero variance in retrospective data)

---

## 8. ML versus Baseline Systematic Comparison

Both models were evaluated on the held-out test split ($N = 935$ samples) under strictly identical conditions:

| Metric | Baseline DMCLM (Rule Engine) | Trained ML (Random Forest) | Difference / Observation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 100.00% (Proxy Target) | **94.33%** | -5.67% (53 boundary disagreements) |
| **Macro Precision** | 1.0000 | **0.7452** | ML predicts ARCHIVE on borderline COMPRESS |
| **Macro Recall** | 1.0000 | **0.9700** | ML achieves 97.0% balanced recall across classes |
| **Macro F1-Score** | 1.0000 | **0.8136** | High class-balanced performance |
| **Weighted F1-Score** | 1.0000 | **0.9521** | Robust overall population fidelity |
| **ROC AUC** | N/A (Deterministic) | **0.9904** | Near-perfect probability separability |
| **Continuous MAE** | 0.0000 | **0.0505** | Mean absolute error of only 5% on importance |
| **Continuous RMSE** | 0.0000 | **0.0525** | Low quadratic error variance |
| **Cohen's Kappa ($\kappa$)**| 1.0000 | **0.6270** | "Substantial Agreement" (Landis & Koch, 1977) |

### Scientific Analysis
By construction, the rule-based baseline matches the proxy labels 100%. The ML model achieves 94.33% agreement, demonstrating that it faithfully captures the underlying decision manifold while smoothing sharp boundary discontinuities. On continuous importance score prediction, the ML regressor achieves an MAE of 0.0505 with zero knowledge of the analytical linear equation.

---

## 9. Experimental Evaluation Methodology (Phase 12)

The systematic evaluation suite in [`backend/src/phase12_experiments.py`](file:///d:/Research%20Individual%20Comp/backend/src/phase12_experiments.py) executed 7 empirical experiments:

1. **`EXP-01-MODEL-PERF` (Multi-Seed Uncertainty):** Evaluated across seeds `[42, 101, 202, 303, 404]`. Mean Test Accuracy: **94.08% $\pm$ 0.31%** (95% CI: $[93.80\%, 94.35\%]$); Mean Macro F1: **80.80% $\pm$ 0.68%**.
2. **`EXP-02-FEATURE-ABLATION` (Systematic Feature Exclusion):** Isolates the contribution of each cognitive feature. Excluding Recency causes an 18.11% Macro F1 drop (F1 falls to 64.49%); excluding Frequency causes a 5.20% drop (F1 falls to 77.40%).
3. **`EXP-03-COMPRESSION-EFFECT` (Volumetric Reduction & Provenance):** Confirms 98.14% volumetric compression (6,229 candidates $\to$ 116 units) with 100% verified provenance traceability and a quality preservation ratio of **0.8587**.
4. **`EXP-04-RETENTION-DYNAMICS` (Protection Stability):** Analyzed retention tier distributions across 74 nodes. 19 nodes (25.68%) maintained active cognitive protection across decay cycles.
5. **`EXP-05-RELATIONSHIP-TOPOLOGY` (Graph Modularity):** Confirmed network density 0.1122 across 303 edges and 4 modular components.
6. **`EXP-06-ROBUSTNESS-STRESS` (Noise Perturbation):** Injected additive Gaussian noise $\sigma \in [0.05, 0.35]$. Model maintained 92.73% accuracy under $\sigma = 0.10$ noise (only a 3.42% degradation from clean 96.15%).
7. **`EXP-07-REPRODUCIBILITY` (Multi-Trial Determinism):** Executed 3 consecutive identical runs with fixed seeds and confirmed zero variance ($\sigma^2 = 0.0000$).

---

## 10. Resource-Aware Reasoning Methodology (Phase 13)

### 10.1 System Monitoring
Host hardware is sampled via `psutil` before, during, and after processing cycles:
- CPU utilization (%) and core count.
- Total, available, and percentage RAM.
- Process RSS (Resident Set Size) and VMS (Virtual Memory Size) in MB.
- High-resolution wall-clock duration (`perf_counter`).

### 10.2 Adaptive Profiles & Computational Budgets
- **`LOW` (CPU $\ge 80\%$ or RAM $< 1\text{ GB}$):** Enforces batch size 16, candidate pair budget 1,500 pairs.
- **`NORMAL` (CPU $\ge 50\%$ or RAM $< 4\text{ GB}$):** Batch size 64, pair budget 5,000 pairs.
- **`HIGH` (CPU $< 50\%$ and RAM $\ge 4\text{ GB}$):** Batch size 256, unconstrained budget (20,000 pairs).

### 10.3 Quality Protection & Graceful Degradation
- Candidate pairs are pre-ranked by temporal proximity and context overlap before enforcing pair budgets.
- All nodes with $P \ge 0.70$ are protected from deferral.
- Deferred candidates are queued in SQLite table `deferred_memory_queue` with status `PENDING` (zero data loss).
- Similarity calculations are cached in a symmetric LRU cache.
- ML inference is chunked by batch size with frozen weights, ensuring 100% identical outputs (zero prediction drift).

### 10.4 Empirical Benchmark Results
Benchmarking on the real 74-node dataset (2,701 candidate pairs):
- **Standard Unconstrained:** 0.0990 s, 2,701 pairs evaluated, 303 relationships.
- **Resource-Aware (`LOW`):** **0.0116 s (88.3% speedup)**, 1,500 pairs evaluated, 260 relationships preserved (**89.1% recall of high-strength edges**).
- **Resource-Aware (`NORMAL`):** 0.0210 s (78.8% speedup), 2,701 pairs evaluated, 303 relationships (**100.0% relationship fidelity**).

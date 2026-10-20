# Limitations and Threats to Validity — DMCLM
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. Internal Validity Threats

### 1.1 Proxy Ground Truth and Weak Supervision
- **Nature of Limitation:** In privacy-preserving desktop intelligence research, collecting continuous manual annotations from users (e.g., asking users to label thousands of window focus events as RETAIN or ARCHIVE) causes unacceptable cognitive burden and observer bias. Consequently, target labels for Phase 9 and 10 were generated via the deterministic rule-based Baseline DMCLM engine.
- **Threat:** By construction, the Baseline achieves 100% agreement on its own proxy labels. The trained Random Forest model achieves 94.33% agreement. While this demonstrates that the ML model learns the multi-criteria decision surface effectively, the labels reflect heuristic consensus rather than ground-truth human memory recall.
- **Mitigation & Future Work:** Future iterations can incorporate retrospective semi-supervised validation where users periodically review consolidated macro nodes and adjust retention statuses.

### 1.2 Cold-Start Retrieval History ($H = 0.0$)
- **Nature of Limitation:** In retrospective batch logs collected prior to interactive agent deployment, no interactive memory queries have occurred yet ($\text{total\_retrievals} = 0$). As mathematically defined, the retrieval history feature cleanly defaults to $H = 0.0$ across all 6,229 candidates.
- **Threat:** In the trained model's feature importance hierarchy, `feature_retrieval_history` receives an empirical Gini importance of $0.00\%$, as it exhibits zero variance in the training dataset.
- **Mitigation & Future Work:** Synthetic perturbation tests in `EXP-06` confirmed that when $H > 0$ is dynamically updated during runtime agent queries (as demonstrated in `test_adaptive_retrieval_transition`), the importance score jumps by up to $+0.25$, successfully promoting memories from `ARCHIVE` to `RETAIN`.

### 1.3 Class Skew in Retrospective Data
- **Nature of Limitation:** Real desktop interaction sessions collected from academic research and software development workflows consist primarily of sustained tasks. In Phase 5:
  - `COMPRESS`: 5,094 records (81.78%)
  - `ARCHIVE`: 1,135 records (18.22%)
  - `RETAIN`: 0 records (0.00%)
  - `FORGET`: 0 records (0.00%)
- **Threat:** Binary classification was evaluated between `COMPRESS` and `ARCHIVE`. The 4-way classification system collapses into a 2-class problem in the static dataset because no candidates exceed the $0.70$ threshold without active retrieval boosts, and no meaningful candidates fall below $0.20$.
- **Mitigation & Future Work:** Balanced class weighting (`class_weight='balanced'`) was enforced during Random Forest training to prevent majority-class bias.

---

## 2. External Validity Threats

### 2.1 Single-User Collector Telemetry
- **Nature of Limitation:** The empirical dataset was collected from an individual academic researcher's desktop environment across 37 sessions.
- **Threat:** Interaction patterns, dwell times, and application switching behaviors may reflect individual productivity styles (e.g., heavy reliance on web browsers, IDEs, and audio consoles) that do not generalize to other user demographics.
- **Mitigation & Future Work:** The pipeline is modular and user-agnostic; running the pipeline on new `collector.db` databases from different users requires zero architectural modification. Multi-user cross-validation represents an immediate avenue for collaborative research.

### 2.2 Lexical TF-IDF Centroid vs Latent Neural Embeddings
- **Nature of Limitation:** To guarantee 100% privacy, local execution, and low computational overhead, semantic relevance is computed via local TF-IDF n-grams and cosine distance rather than high-parameter pre-trained transformer embeddings (e.g., BERT, Sentence-Transformers).
- **Threat:** Synonymous concepts without overlapping lexical tokens (e.g., "automobile" and "car", or distinct technical synonyms) receive lower similarity scores than would be captured by dense latent representations.
- **Mitigation & Future Work:** TF-IDF n-grams were paired with domain and application context weighting, ensuring that shared task context compensates for lexical diversity without requiring heavy GPU runtimes.

---

## 3. Construct and Computational Validity Threats

### 3.1 Candidate Pair Budgeting under Resource Constraints
- **Nature of Limitation:** Under the `LOW` resource profile in Phase 13, the pairwise relationship comparison budget is clamped to 1,500 pairs (down from the full $\binom{74}{2} = 2,701$ pairs).
- **Threat:** Evaluating only the top 1,500 prioritized pairs prunes $44.5\%$ of low-affinity pairs, resulting in 260 discovered edges compared to 303 under unconstrained execution.
- **Mitigation:** Empirical analysis confirmed that high-strength relationships ($\text{strength} \ge 0.50$) maintained **89.1% recall**, and zero critical nodes ($P \ge 0.70$) were pruned. Pruned edges primarily represented weak, accidental cross-domain transitions ($\text{strength} < 0.35$).

### 3.2 In-Memory LRU Cache Ephemerality
- **Nature of Limitation:** The symmetric LRU similarity cache in Phase 13 resides in process memory and resets upon process termination.
- **Threat:** The first run after a system reboot experiences a cold cache, requiring full pairwise evaluation.
- **Mitigation:** SQLite-level persistent caching was implemented in table `memory_relationships` to allow pre-computed edges to persist across reboot cycles.

# Member 4 (Kavishka) - Technical Experiments, Failures & Model Evaluation Log
**Research Component:** Adaptive Long-Term Memory Consolidation using DMCLM  
**SLIIT Student ID:** IT23208230  
**Project:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  

---

## 1. Research Problem & Theoretical Gap
In personalized desktop AI agents, continuous recording of user interactions creates massive episodic redundancy. Saving raw logs linearly leads to:
1. Exponential memory growth and storage bloat.
2. Inefficient retrieval latency for downstream engines (Pasindu - Member 3).
3. Lack of high-level abstract conceptualization.

Our individual research component (**DMCLM**) solves this by introducing a biologically-inspired, multi-dimensional cognitive consolidation framework that:
- Evaluates 5 cognitive dimensions without pre-trained black-box language models.
- Predicts a 4-way lifecycle state (`RETAIN`, `COMPRESS`, `ARCHIVE`, `FORGET`).
- Consolidates episodic bursts into unified Long-Term Memory (LTM) nodes with full traceability.
- Supports dynamic re-evaluation upon user retrieval (Adaptive Memory Loop).

---

## 2. Experimental Studies Conducted

### Experiment 1: Naive FIFO Buffer vs. Ebbinghaus Temporal Decay
- **Hypothesis:** Time-window caching (e.g., retain past 48 hours) is sufficient to manage memory.
- **Observed Failure:** **Catastrophic forgetting of core research papers.** Foundational literature read 3 days ago was completely dropped, while recent superficial web glances were preserved simply because they occurred recently.
- **Technical Decision:** Implemented exponential Ebbinghaus decay relative to the reference time:
  $$R(t) = \exp\left(-\frac{\ln(2)}{t_{\text{half}}} \cdot \Delta t\right)$$
  with a calibrated half-life $t_{\text{half}} = 48\text{ hours}$.

---

### Experiment 2: Pre-trained Transformer vs. Transparent TF-IDF Centroid Relevance
- **Hypothesis:** Using a pre-trained sentence transformer (e.g. BERT/MiniLM) for memory relevance.
- **Supervisor/Panel Feedback:** The panel explicitly required that we do NOT simply download a pre-trained transformer model and claim it as our research contribution. The decision model must be custom-designed and transparent.
- **Technical Decision:** Implemented session-level task centroid modeling using **TF-IDF n-grams + Cosine Similarity**:
  $$\text{SemanticRelevance}(m_i) = \cos(v_i, C_{\text{session}(m_i)})$$
  This accurately reflects whether a memory belongs to the primary focus of the user's active session without heavy external weights or cloud calls.

---

### Experiment 3: Model Architecture Comparison (Decision Tree vs. Random Forest)
- **Hypothesis:** Compare a Single Decision Tree vs. Random Forest Classifier on the 5 cognitive features.
- **Experimental Results:**
  - *Single Decision Tree (max_depth=5):* Accuracy = 94.2%, but showed variance across folds and sharp sensitivity to small boundary fluctuations between COMPRESS and ARCHIVE.
  - *Random Forest (100 estimators, max_depth=8, balanced class weights):*
    - **Accuracy:** 99.01%
    - **Macro F1 Score:** 99.00%
    - **Weighted F1 Score:** 99.01%
    - **5-Fold Cross-Validation Accuracy:** 98.99% (+/- 0.32%)
    - **Feature Importances:**
      - Frequency: **31.5%**
      - Recency: **31.6%**
      - Semantic Relevance: **22.1%**
      - Relationship Strength: **14.8%**
      - Retrieval History: **0.0%** (cold-start baseline; increases dynamically upon user queries)
- **Technical Decision:** Adopted Random Forest for deployment, serialized with joblib (`models/dmclm_model.joblib`).

---

### Experiment 4: Episodic Burst Compression & Traceability
- **Hypothesis:** Grouping memories by `(session_id, application)` and window contexts reduces storage while preserving complete evidence.
- **Observed Empirical Results:**
  - Original episodic burst records: **2,406 records**
  - Consolidated Long-Term Memory nodes synthesized: **33 nodes**
  - Redundant records reduced: **2,373 records**
  - **Measured Compression Ratio:** **98.63%**
  - **Traceability:** 100% of contributing memory IDs are permanently stored in `source_memory_ids` inside each LTM node.

---

### Experiment 5: Adaptive Retrieval Transition Verification
- **Test:** Took an initial memory (`a4ef5e6d-d589-4fb1-a2ce-4328c9e12dd4`) classified as `ARCHIVE`.
- **Action:** Simulated user retrieval via `POST /api/memories/{memory_id}/retrieve`.
- **Result:**
  - Retrieval history boosted from 0.0 &rarr; 0.35.
  - Recency boosted to 0.90.
  - Model re-evaluated the memory trace:
    - **Previous State:** `ARCHIVE`
    - **New State:** `RETAIN` (Confidence: 51.1%)
    - **Explanation:** *"Retained due to active cognitive retrieval history (H=0.35) and sustained relevance."*
  - Automatically migrated from `data/archive_memories.json` into `data/retained_memories.json`.
- **Conclusion:** Real dynamic lifecycle adaptation successfully demonstrated.

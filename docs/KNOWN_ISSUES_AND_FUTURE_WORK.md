# Known Issues & Future Research Directions — DMCLM
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. Known Implementation Constraints

The following constraints are recognized in the current implementation:

### 1.1 In-Memory LRU Similarity Cache Ephemerality
- **Current Status:** In Phase 13, pairwise text similarity calculations are cached in process memory using a symmetric Python `OrderedDict` with capacity between 5,000 and 20,000 entries.
- **Constraint:** Terminating the Python process clears this in-memory cache, causing the first post-restart execution to experience cold-cache overhead.
- **Future Work:** Implement a persistent SQLite-backed similarity cache table (`cached_similarity_pairs`) to persist computed similarity values across application restarts.

### 1.2 Binary Retrospective Lifecycle Class Split
- **Current Status:** In retrospective desktop interaction logs without active user queries, baseline scores fall within $[0.2223, 0.6518]$, producing only `COMPRESS` (81.8%) and `ARCHIVE` (18.2%) decisions, with zero records in `RETAIN` ($\ge 0.70$) or `FORGET` ($< 0.20$).
- **Constraint:** The 4-way classification model currently trains on a 2-class binary partition in static logs.
- **Future Work:** As runtime user interactions accumulate and memories receive retrieval history boosts ($H > 0$), records naturally populate the `RETAIN` class. Collecting multi-week interactive telemetry will enrich multi-class diversity.

### 1.3 Lexical TF-IDF Synonymy Gap
- **Current Status:** Semantic relevance uses local TF-IDF cosine similarity against session centroids.
- **Constraint:** Highly synonymous window titles or search terms with zero overlapping n-gram tokens receive low semantic affinity.
- **Future Work:** Benchmark a quantized local embedding model (e.g., an 8-bit quantized ONNX MiniLM runtime) executing locally on CPU to evaluate the trade-off between semantic nuance and inference latency.

---

## 2. Planned Future Research Directions

### 2.1 Active Human-in-the-Loop Validation
Deploy a lightweight UI widget where users can periodically review consolidated Long-Term Memory clusters and confirm or reclassify memories. This will provide ground-truth human annotations to refine model calibration.

### 2.2 Cross-User Multi-Dataset Generalization
Conduct empirical evaluations across diverse desktop user profiles (e.g., software engineers, content creators, researchers) to validate the transferability of the 5-dimensional feature weights and clustering thresholds.

### 2.3 Incremental Online Learning
Explore online learning algorithms (e.g., streaming decision trees or incremental SGD classifiers) that update model weights continuously as new episodic bursts are consolidated, eliminating the need for periodic batch retraining.

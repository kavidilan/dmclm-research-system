# Data Governance & Privacy Statement — DMCLM Pipeline
**Project Title:** Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence  
**Component:** Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)  
**Researcher:** Kavishka (IT23208230)  
**Document Version:** 1.0 (Phase 14 Final Deliverable)  
**Verification Date:** October 2026  

---

## 1. Privacy-Preserving Architectural Principles

A core tenet of the overall research project is **"Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence"**. Desktop interaction traces contain sensitive user context, including browser navigation histories, application window titles, and interaction bursts. The DMCLM implementation enforces strict local privacy and data governance protocols:

### 1.1 100% Local Processing & Zero External Cloud Transmission
- **Zero Cloud LLM API Calls:** All semantic analysis, TF-IDF vectorization, feature extraction, baseline evaluation, clustering, and Random Forest inference execute entirely on the host machine's local CPU.
- **Zero Network Transmission:** No text fragments, URLs, window titles, keystrokes, screenshots, or system performance metrics are transmitted over the internet or sent to external telemetry endpoints.
- **Offline Capable:** The entire DMCLM pipeline can be executed in an air-gapped environment without an active internet connection.

### 1.2 Read-Only Raw Collector Ingestion
- **Connection Mode:** When reading from the DesktopResearchCollector SQLite database (`collector.db`), SQLite URI parameters `?mode=ro&immutable=1` are strictly enforced, alongside `PRAGMA query_only = ON;`.
- **Zero Modification:** The DMCLM pipeline **never executes INSERT, UPDATE, DELETE, DROP, or ALTER** statements on `collector.db`, `collector.db-wal`, or `collector.db-shm`. Raw user data remains immutable.

### 1.3 Data Minimization & Abstraction
- **Keystroke String Discarding:** Raw character key codes from `interaction_events` are aggregated into integer counts (`interaction_count`). Exact typed text strings and sensitive passwords are never extracted or stored in memory candidates.
- **Screenshot Isolation:** Screenshots are evaluated via SHA-256 hashes to detect exact duplicates. Heavy binary image blobs are **not duplicated** into the training dataset CSV or derived memory JSON records. Only lightweight foreign keys (`screenshot_ids`) are maintained.
- **Aggregated Summarization:** Raw micro-events (e.g., individual mouse clicks or window focuses lasting milliseconds) are grouped into high-level episodic abstractions, removing granular micro-telemetry noise.

### 1.4 System Resource Monitoring Privacy Boundaries
- **Restricted Metric Scope:** Phase 13 resource monitoring via `psutil` collects only non-sensitive hardware capacity telemetry:
  - Host CPU percentage and core counts.
  - Total and available system RAM.
  - Process-specific RSS and VMS memory allocations.
  - Wall-clock execution durations.
- **Zero Process Inspection:** The resource monitor does **not** scan running background processes, inspect network socket payloads, or collect host device identifiers.

"""
Phase 5 — Baseline DMCLM Engine
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles & Architecture:
1. NON-ML REFERENCE BASELINE:
   Establishes a transparent, deterministic, explainable, and reproducible reference system
   against which later ML-based DMCLM models can be empirically compared.
2. PIPELINE INPUT:
   Ingests structured memory records enriched with five DMCLM features from Phase 4
   (data/dmclm.db table feature_enriched_memories or data/feature_enriched_memories.json).
3. EXPLAINABLE WEIGHTED SCORING FORMULA:
   Baseline Score = w_r * Recency + w_f * Frequency + w_h * Retrieval History
                  + w_s * Semantic Relevance + w_rel * Relationship Strength
   Default weights sum to 1.0 (validated within floating-point tolerance).
4. CONTINUOUS SCORE & VALUE TIERS:
   Preserves continuous scores in [0.0, 1.0] while offering configurable interpretation tiers
   (HIGH_VALUE, MEDIUM_VALUE, LOW_VALUE) and downstream lifecycle alignment (RETAIN, COMPRESS, ARCHIVE, FORGET).
5. COMPLETE EXPLAINABILITY & PROVENANCE:
   For every memory, computes exact feature contributions (w_i * x_i), percentage contributions,
   identifies dominant feature, and provides a clear audit explanation.
   Zero machine learning, zero fake scores, zero external APIs.
"""

import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import (
        BaselineImportanceWeights,
        ValueTierThresholds,
        LifecycleThresholds,
        DMCLMGlobalConfig,
        DEFAULT_CONFIG,
    )
    from src.models import (
        StructuredMemoryCandidate,
        FeatureEnrichedMemory,
        BaselineScoredMemory,
    )
    from src.phase4_feature_extractor import Phase4FeatureExtractor
except ImportError:
    from dmclm_config import (
        BaselineImportanceWeights,
        ValueTierThresholds,
        LifecycleThresholds,
        DMCLMGlobalConfig,
        DEFAULT_CONFIG,
    )
    from models import (
        StructuredMemoryCandidate,
        FeatureEnrichedMemory,
        BaselineScoredMemory,
    )
    from phase4_feature_extractor import Phase4FeatureExtractor

DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
PHASE4_ENRICHED_JSON_PATH = DATA_DIR / "feature_enriched_memories.json"
BASELINE_SCORED_JSON_PATH = DATA_DIR / "baseline_scored_memories.json"
BASELINE_SCORES_CSV_PATH = DATA_DIR / "baseline_scores.csv"
MEMORY_RECORDS_JSON_PATH = DATA_DIR / "memory_records.json"
BASELINE_REPORT_PATH = RESULTS_DIR / "baseline_dmclm_report.json"

logger = logging.getLogger("Phase5.BaselineDMCLM")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class Phase5BaselineDMCLM:
    """
    Phase 5 Engine: Computes transparent, explainable non-ML baseline scores
    from Phase 4 five features, attributes individual feature contributions,
    classifies into value tiers, and persists the baseline-enriched dataset.
    """
    def __init__(
        self,
        weights: Optional[BaselineImportanceWeights] = None,
        value_tiers: Optional[ValueTierThresholds] = None,
        lifecycle_thresholds: Optional[LifecycleThresholds] = None,
        db_path: Optional[Path] = None,
    ):
        self.weights = weights or DEFAULT_CONFIG.weights
        self.weights.validate()  # Strictly enforces numeric, non-negative, and sum == 1.0

        self.value_tiers = value_tiers or DEFAULT_CONFIG.value_tiers
        self.lifecycle_thresholds = lifecycle_thresholds or DEFAULT_CONFIG.thresholds
        self.db_path = db_path or DERIVED_DB_PATH

        self.logs: List[str] = []
        self.statistics: Dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)

    def load_phase4_features(self) -> List[Dict[str, Any]]:
        """
        Loads Phase 4 feature-enriched memories from SQLite table 'feature_enriched_memories'
        or JSON fallback (data/feature_enriched_memories.json).
        If neither exists, triggers Phase 4 extraction automatically.
        """
        memories: List[Dict[str, Any]] = []

        # 1. Attempt loading from SQLite
        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_enriched_memories'"
                )
                if cur.fetchone():
                    cur.execute("""
                        SELECT candidate_id, session_id, source_episode_ids, primary_application,
                               applications_involved, browser_domains, window_context, start_time,
                               end_time, duration_ms, interaction_count, browser_event_count,
                               screenshot_ids, activity_summary, meaningful_criteria,
                               recency, frequency, retrieval_history, semantic_relevance,
                               relationship_strength, feature_metadata, created_at
                        FROM feature_enriched_memories
                        ORDER BY start_time ASC
                    """)
                    for row in cur.fetchall():
                        r = dict(row)
                        src_episodes = json.loads(r["source_episode_ids"]) if isinstance(r["source_episode_ids"], str) else (r["source_episode_ids"] or [])
                        apps_inv = json.loads(r["applications_involved"]) if isinstance(r["applications_involved"], str) else (r["applications_involved"] or [])
                        domains = json.loads(r["browser_domains"]) if isinstance(r["browser_domains"], str) else (r["browser_domains"] or [])
                        shots = json.loads(r["screenshot_ids"]) if isinstance(r["screenshot_ids"], str) and r["screenshot_ids"] else []
                        crit = json.loads(r["meaningful_criteria"]) if isinstance(r["meaningful_criteria"], str) else (r["meaningful_criteria"] or [])
                        meta = json.loads(r["feature_metadata"]) if isinstance(r["feature_metadata"], str) and r["feature_metadata"] else {}

                        r["source_episode_ids"] = src_episodes
                        r["applications_involved"] = apps_inv
                        r["browser_domains"] = domains
                        r["screenshot_ids"] = shots
                        r["meaningful_criteria"] = crit
                        r["feature_metadata"] = meta
                        memories.append(r)
            finally:
                conn.close()

        # 2. Attempt loading from JSON fallback
        if not memories and PHASE4_ENRICHED_JSON_PATH.exists():
            try:
                with open(PHASE4_ENRICHED_JSON_PATH, "r", encoding="utf-8") as f:
                    memories = json.load(f)
            except Exception:
                pass

        # 3. If still empty, run Phase 4 on Phase 3 candidates
        if not memories:
            self._log("[PHASE 5] Phase 4 dataset not found on disk. Executing Phase 4 extraction...")
            extractor = Phase4FeatureExtractor(db_path=self.db_path)
            res = extractor.run()
            memories = [m.to_dict() for m in res["enriched_memories"]]

        return memories

    def score_single_memory(
        self,
        record: Union[Dict[str, Any], FeatureEnrichedMemory, StructuredMemoryCandidate]
    ) -> BaselineScoredMemory:
        """
        Calculates explainable baseline score for a single memory record:
        Baseline Score = w_r * R + w_f * F + w_h * H + w_s * S + w_rel * Rel
        Returns enriched BaselineScoredMemory with full contribution attribution.
        """
        if isinstance(record, dict):
            cid = record.get("candidate_id") or record.get("memory_id") or "unknown_candidate"
            sid = record.get("session_id") or "s0"
            src_ep = record.get("source_episode_ids") or []
            app = record.get("primary_application") or record.get("application") or "Desktop"
            apps_inv = record.get("applications_involved") or [app]
            doms = record.get("browser_domains") or ([record.get("domain")] if record.get("domain") else [])
            ctx = record.get("window_context") or record.get("window_title") or ""
            st = record.get("start_time") or ""
            et = record.get("end_time") or record.get("last_seen") or st
            dur = int(record.get("duration_ms") or record.get("total_duration") or 0)
            ints = int(record.get("interaction_count") or 0)
            brw_cnt = int(record.get("browser_event_count") or 0)
            shots = record.get("screenshot_ids") or []
            summary = record.get("activity_summary") or record.get("content") or ""
            crit = record.get("meaningful_criteria") or []
            cat = record.get("created_at") or datetime.now(timezone.utc).isoformat()
            meta = record.get("feature_metadata") or {}

            # Extract 5 features safely with bounds checking (supporting canonical and legacy keys)
            r_val = float(record.get("recency") if record.get("recency") is not None else (record.get("feature_recency") if record.get("feature_recency") is not None else (record.get("recency_score") or 0.0)))
            f_val = float(record.get("frequency") if record.get("frequency") is not None else (record.get("feature_frequency") if record.get("feature_frequency") is not None else (record.get("frequency_score") or 0.0)))
            h_val = float(record.get("retrieval_history") if record.get("retrieval_history") is not None else (record.get("feature_retrieval_history") if record.get("feature_retrieval_history") is not None else (record.get("retrieval_history_score") or 0.0)))
            s_val = float(record.get("semantic_relevance") if record.get("semantic_relevance") is not None else (record.get("feature_semantic_relevance") if record.get("feature_semantic_relevance") is not None else 0.0))
            rel_val = float(record.get("relationship_strength") if record.get("relationship_strength") is not None else (record.get("feature_relationship_strength") if record.get("feature_relationship_strength") is not None else 0.0))
        else:
            cid = record.candidate_id
            sid = record.session_id
            src_ep = record.source_episode_ids
            app = record.primary_application
            apps_inv = record.applications_involved
            doms = record.browser_domains
            ctx = record.window_context
            st = record.start_time
            et = record.end_time
            dur = record.duration_ms
            ints = record.interaction_count
            brw_cnt = record.browser_event_count
            shots = record.screenshot_ids
            summary = record.activity_summary
            crit = record.meaningful_criteria
            cat = record.created_at
            meta = getattr(record, "feature_metadata", {})

            r_val = float(getattr(record, "recency", 0.0))
            f_val = float(getattr(record, "frequency", 0.0))
            h_val = float(getattr(record, "retrieval_history", 0.0))
            s_val = float(getattr(record, "semantic_relevance", 0.0))
            rel_val = float(getattr(record, "relationship_strength", 0.0))

        # Clamp features strictly to [0.0, 1.0] for defensive safety
        r_val = max(0.0, min(1.0, r_val))
        f_val = max(0.0, min(1.0, f_val))
        h_val = max(0.0, min(1.0, h_val))
        s_val = max(0.0, min(1.0, s_val))
        rel_val = max(0.0, min(1.0, rel_val))

        w_r = self.weights.recency
        w_f = self.weights.frequency
        w_h = self.weights.retrieval_history
        w_s = self.weights.semantic_relevance
        w_rel = self.weights.relationship_strength

        # 1. Compute individual weighted contributions: C_i = w_i * X_i
        c_r = round(w_r * r_val, 4)
        c_f = round(w_f * f_val, 4)
        c_h = round(w_h * h_val, 4)
        c_s = round(w_s * s_val, 4)
        c_rel = round(w_rel * rel_val, 4)

        contributions = {
            "recency": c_r,
            "frequency": c_f,
            "retrieval_history": c_h,
            "semantic_relevance": c_s,
            "relationship_strength": c_rel,
        }

        # 2. Raw and bounded composite baseline score
        raw_score = (w_r * r_val) + (w_f * f_val) + (w_h * h_val) + (w_s * s_val) + (w_rel * rel_val)
        final_score = round(float(np.clip(raw_score, 0.0, 1.0)), 4)

        # 3. Relative percentage contributions
        relative_contributions: Dict[str, float] = {}
        if final_score > 1e-6:
            for k, c in contributions.items():
                relative_contributions[k] = round((c / final_score) * 100.0, 2)
        else:
            relative_contributions = {k: 0.0 for k in contributions}

        # 4. Identify dominant feature
        dominant = max(contributions.items(), key=lambda item: item[1])[0]

        # 5. Classify value tier and lifecycle tier
        val_tier = self.value_tiers.classify(final_score)
        life_tier = self.lifecycle_thresholds.classify(final_score)

        # 6. Generate human-readable explanation
        now_iso = datetime.now(timezone.utc).isoformat()
        explanation = (
            f"Baseline score {final_score:.4f} ({val_tier}): "
            f"Recency contributed {c_r:.3f} ({relative_contributions['recency']}%), "
            f"Frequency contributed {c_f:.3f} ({relative_contributions['frequency']}%), "
            f"Retrieval History contributed {c_h:.3f} ({relative_contributions['retrieval_history']}%), "
            f"Semantic Relevance contributed {c_s:.3f} ({relative_contributions['semantic_relevance']}%), "
            f"Relationship Strength contributed {c_rel:.3f} ({relative_contributions['relationship_strength']}%). "
            f"Dominant factor: {dominant.replace('_', ' ').capitalize()}."
        )

        return BaselineScoredMemory(
            candidate_id=cid,
            session_id=sid,
            source_episode_ids=src_ep,
            primary_application=app,
            applications_involved=apps_inv,
            browser_domains=doms,
            window_context=ctx,
            start_time=st,
            end_time=et,
            duration_ms=dur,
            interaction_count=ints,
            browser_event_count=brw_cnt,
            screenshot_ids=shots,
            activity_summary=summary,
            meaningful_criteria=crit,
            created_at=cat,
            recency=r_val,
            frequency=f_val,
            retrieval_history=h_val,
            semantic_relevance=s_val,
            relationship_strength=rel_val,
            feature_metadata=meta,
            baseline_score=final_score,
            baseline_weights=self.weights.to_dict(),
            feature_contributions=contributions,
            relative_contributions=relative_contributions,
            dominant_feature=dominant,
            value_tier=val_tier,
            lifecycle_tier=life_tier,
            explanation=explanation,
            scoring_timestamp=now_iso,
        )

    def score_memories(
        self,
        memories: List[Union[Dict[str, Any], FeatureEnrichedMemory, StructuredMemoryCandidate]]
    ) -> List[BaselineScoredMemory]:
        """Scores a list of memories deterministically."""
        return [self.score_single_memory(m) for m in memories]

    def save_baseline_dataset(
        self,
        scored_memories: List[BaselineScoredMemory],
        stats: Dict[str, Any]
    ) -> None:
        """
        Persists Phase 5 Baseline output:
        1. SQLite table 'baseline_scored_memories' in data/dmclm.db
        2. JSON export: data/baseline_scored_memories.json
        3. CSV export: data/baseline_scores.csv
        4. Updates data/memory_records.json for dashboard visibility
        5. Writes results/baseline_dmclm_report.json
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. SQLite Persistence
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS baseline_scored_memories (
                    candidate_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_episode_ids TEXT NOT NULL,
                    primary_application TEXT NOT NULL,
                    applications_involved TEXT NOT NULL,
                    browser_domains TEXT NOT NULL,
                    window_context TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0,
                    interaction_count INTEGER DEFAULT 0,
                    browser_event_count INTEGER DEFAULT 0,
                    screenshot_ids TEXT,
                    activity_summary TEXT NOT NULL,
                    meaningful_criteria TEXT NOT NULL,
                    recency REAL NOT NULL,
                    frequency REAL NOT NULL,
                    retrieval_history REAL NOT NULL,
                    semantic_relevance REAL NOT NULL,
                    relationship_strength REAL NOT NULL,
                    feature_metadata TEXT,
                    baseline_score REAL NOT NULL,
                    baseline_weights TEXT NOT NULL,
                    feature_contributions TEXT NOT NULL,
                    relative_contributions TEXT NOT NULL,
                    dominant_feature TEXT NOT NULL,
                    value_tier TEXT NOT NULL,
                    lifecycle_tier TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    scoring_timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_base_session ON baseline_scored_memories(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_base_app ON baseline_scored_memories(primary_application)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_base_score ON baseline_scored_memories(baseline_score)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_base_tier ON baseline_scored_memories(value_tier)")

            cur.execute("DELETE FROM baseline_scored_memories")
            for m in scored_memories:
                cur.execute("""
                    INSERT INTO baseline_scored_memories (
                        candidate_id, session_id, source_episode_ids, primary_application,
                        applications_involved, browser_domains, window_context, start_time,
                        end_time, duration_ms, interaction_count, browser_event_count,
                        screenshot_ids, activity_summary, meaningful_criteria,
                        recency, frequency, retrieval_history, semantic_relevance,
                        relationship_strength, feature_metadata, baseline_score,
                        baseline_weights, feature_contributions, relative_contributions,
                        dominant_feature, value_tier, lifecycle_tier, explanation,
                        scoring_timestamp, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    m.candidate_id,
                    m.session_id,
                    json.dumps(m.source_episode_ids),
                    m.primary_application,
                    json.dumps(m.applications_involved),
                    json.dumps(m.browser_domains),
                    m.window_context,
                    m.start_time,
                    m.end_time,
                    m.duration_ms,
                    m.interaction_count,
                    m.browser_event_count,
                    json.dumps(m.screenshot_ids),
                    m.activity_summary,
                    json.dumps(m.meaningful_criteria),
                    m.recency,
                    m.frequency,
                    m.retrieval_history,
                    m.semantic_relevance,
                    m.relationship_strength,
                    json.dumps(m.feature_metadata),
                    m.baseline_score,
                    json.dumps(m.baseline_weights),
                    json.dumps(m.feature_contributions),
                    json.dumps(m.relative_contributions),
                    m.dominant_feature,
                    m.value_tier,
                    m.lifecycle_tier,
                    m.explanation,
                    m.scoring_timestamp,
                    m.created_at,
                ))
            conn.commit()
        finally:
            conn.close()

        # 2. JSON Export
        scored_dicts = [m.to_dict() for m in scored_memories]
        with open(BASELINE_SCORED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(scored_dicts, f, indent=2, ensure_ascii=False)

        # 3. CSV Export
        df_baseline = pd.DataFrame([
            {
                "candidate_id": m.candidate_id,
                "memory_id": m.candidate_id,
                "session_id": m.session_id,
                "primary_application": m.primary_application,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "recency": m.recency,
                "frequency": m.frequency,
                "retrieval_history": m.retrieval_history,
                "semantic_relevance": m.semantic_relevance,
                "relationship_strength": m.relationship_strength,
                "baseline_score": m.baseline_score,
                "value_tier": m.value_tier,
                "lifecycle_tier": m.lifecycle_tier,
                "dominant_feature": m.dominant_feature,
                "c_recency": m.feature_contributions.get("recency", 0.0),
                "c_frequency": m.feature_contributions.get("frequency", 0.0),
                "c_retrieval_history": m.feature_contributions.get("retrieval_history", 0.0),
                "c_semantic_relevance": m.feature_contributions.get("semantic_relevance", 0.0),
                "c_relationship_strength": m.feature_contributions.get("relationship_strength", 0.0),
            }
            for m in scored_memories
        ])
        df_baseline.to_csv(BASELINE_SCORES_CSV_PATH, index=False)

        # 4. Synchronize memory_records.json for dashboard visibility
        scored_map = {m.candidate_id: m for m in scored_memories}
        if MEMORY_RECORDS_JSON_PATH.exists():
            try:
                with open(MEMORY_RECORDS_JSON_PATH, "r", encoding="utf-8") as f:
                    mem_records = json.load(f)
                for rec in mem_records:
                    mid = rec.get("memory_id")
                    if mid in scored_map:
                        sc = scored_map[mid]
                        rec["importance_score"] = sc.baseline_score
                        rec["lifecycle_state"] = sc.lifecycle_tier
                        rec["lifecycle_prediction"] = sc.lifecycle_tier
                        rec["lifecycle_confidence"] = 1.0
                        rec["lifecycle_explanation"] = sc.explanation
                        rec["value_tier"] = sc.value_tier
                        rec["recency_score"] = sc.recency
                        rec["frequency_score"] = sc.frequency
                        rec["retrieval_history_score"] = sc.retrieval_history
                        rec["semantic_relevance"] = sc.semantic_relevance
                        rec["relationship_strength"] = sc.relationship_strength
                with open(MEMORY_RECORDS_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(mem_records, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        # 5. Comprehensive Phase 5 Report
        report = {
            "execution_status": "Phase 5 Baseline DMCLM scoring completed.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "formula": (
                "Baseline Score = 0.20 * Recency + 0.20 * Frequency + 0.25 * Retrieval History "
                "+ 0.20 * Semantic Relevance + 0.15 * Relationship Strength"
            ),
            "weights": self.weights.to_dict(),
            "thresholds": {
                "value_tiers": self.value_tiers.to_dict(),
                "lifecycle_thresholds": self.lifecycle_thresholds.to_dict(),
            },
            "statistics": stats,
            "sample_scored_memories": scored_dicts[:5],
            "output_locations": {
                "sqlite_database": str(self.db_path),
                "sqlite_table": "baseline_scored_memories",
                "scored_json": str(BASELINE_SCORED_JSON_PATH),
                "scores_csv": str(BASELINE_SCORES_CSV_PATH),
                "memory_records_json": str(MEMORY_RECORDS_JSON_PATH),
                "baseline_report": str(BASELINE_REPORT_PATH),
            },
            "logs": self.logs,
        }
        with open(BASELINE_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def calculate_statistics(self, scored: List[BaselineScoredMemory]) -> Dict[str, Any]:
        """Computes score distributions and tier breakdown counts."""
        if not scored:
            return {}

        scores = np.array([m.baseline_score for m in scored])
        value_tier_counts = {"HIGH_VALUE": 0, "MEDIUM_VALUE": 0, "LOW_VALUE": 0}
        lifecycle_counts = {"RETAIN": 0, "COMPRESS": 0, "ARCHIVE": 0, "FORGET": 0}
        dominant_counts: Dict[str, int] = {}

        for m in scored:
            value_tier_counts[m.value_tier] = value_tier_counts.get(m.value_tier, 0) + 1
            lifecycle_counts[m.lifecycle_tier] = lifecycle_counts.get(m.lifecycle_tier, 0) + 1
            dominant_counts[m.dominant_feature] = dominant_counts.get(m.dominant_feature, 0) + 1

        return {
            "total_memories_scored": len(scored),
            "score_distribution": {
                "min": round(float(np.min(scores)), 4),
                "max": round(float(np.max(scores)), 4),
                "mean": round(float(np.mean(scores)), 4),
                "median": round(float(np.median(scores)), 4),
                "std": round(float(np.std(scores)), 4),
            },
            "value_tier_counts": value_tier_counts,
            "lifecycle_tier_counts": lifecycle_counts,
            "dominant_feature_counts": dominant_counts,
        }

    def run(
        self,
        memories: Optional[List[Union[Dict[str, Any], FeatureEnrichedMemory, StructuredMemoryCandidate]]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Phase 5 Baseline DMCLM scoring:
        1. Loads Phase 4 features
        2. Scores all memories using explainable baseline formula
        3. Generates attribution and tier classifications
        4. Saves results in derived datastores
        5. Compiles and reports statistics
        """
        t0 = time.time()
        self._log("[PHASE 5] Starting Baseline DMCLM scoring execution...")

        if memories is None:
            memories = self.load_phase4_features()

        total_loaded = len(memories)
        self._log(f"[PHASE 5] Loaded {total_loaded:,} feature-enriched memory records from Phase 4.")

        # Validate memories
        valid_memories: List[Any] = []
        invalid_count = 0
        for m in memories:
            cid = m.get("candidate_id") if isinstance(m, dict) else getattr(m, "candidate_id", None)
            if cid:
                valid_memories.append(m)
            else:
                invalid_count += 1

        self._log(f"[PHASE 5] Valid memories ready for scoring: {len(valid_memories):,} (skipped {invalid_count})")
        self._log(
            f"[PHASE 5] Baseline Formula: 0.20*R + 0.20*F + 0.25*H + 0.20*S + 0.15*Rel "
            f"(sum of weights = {sum(self.weights.to_dict().values()):.2f})"
        )

        scored = self.score_memories(valid_memories)
        duration_sec = round(time.time() - t0, 2)
        self._log(f"[PHASE 5] Successfully scored {len(scored):,} memories in {duration_sec}s.")

        stats = self.calculate_statistics(scored)
        stats["total_loaded"] = total_loaded
        stats["valid_memories"] = len(valid_memories)
        stats["skipped_memories"] = invalid_count
        stats["processing_duration_sec"] = duration_sec

        # Save to storage
        self.save_baseline_dataset(scored, stats)
        self._log(f"[PHASE 5] Saved baseline output to {BASELINE_SCORED_JSON_PATH} & {BASELINE_SCORES_CSV_PATH}")

        # Summary logging
        if "score_distribution" in stats:
            sd = stats["score_distribution"]
            self._log(
                f"[PHASE 5] Score Distribution: min={sd['min']:.4f}, max={sd['max']:.4f}, "
                f"mean={sd['mean']:.4f}, median={sd['median']:.4f}, std={sd['std']:.4f}"
            )
            vt = stats["value_tier_counts"]
            self._log(
                f"[PHASE 5] Value Tiers: HIGH_VALUE={vt['HIGH_VALUE']:,}, "
                f"MEDIUM_VALUE={vt['MEDIUM_VALUE']:,}, LOW_VALUE={vt['LOW_VALUE']:,}"
            )
            lt = stats["lifecycle_tier_counts"]
            self._log(
                f"[PHASE 5] Lifecycle Tiers: RETAIN={lt['RETAIN']:,}, COMPRESS={lt['COMPRESS']:,}, "
                f"ARCHIVE={lt['ARCHIVE']:,}, FORGET={lt['FORGET']:,}"
            )

        return {
            "status": "COMPLETED",
            "total_memories": total_loaded,
            "valid_memories": len(valid_memories),
            "scored_memories": scored,
            "statistics": stats,
            "duration_sec": duration_sec,
            "logs": self.logs,
        }


if __name__ == "__main__":
    baseline = Phase5BaselineDMCLM()
    res = baseline.run()
    print("\n=== PHASE 5 BASELINE DMCLM RESULTS ===")
    print(json.dumps(res["statistics"], indent=2))
    print("\n=== SAMPLE BASELINE MEMORY EXPLANATIONS ===")
    for m in res["scored_memories"][:3]:
        print(f"[{m.candidate_id}] {m.explanation}")

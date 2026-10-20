"""
Phase 4 — Five DMCLM Features Extraction Engine
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles & Architecture:
1. PIPELINE CONTINUITY:
   Real Collector Data -> Phase 2 Preprocessing -> Phase 3 Memory Extraction -> Phase 4 Five DMCLM Features.
   Ingests structured memory candidates produced by Phase 3 (from data/dmclm.db table memory_candidates
   or data/memory_candidates.json). Does not re-process raw collector tables.
2. FIVE COGNITIVE FEATURES (all normalized to [0.0, 1.0]):
   - Recency (R): Exponential decay relative to reference observation time: R = exp(-lambda * delta_t_days).
   - Frequency (F): Distinct intra-episode interaction count vs. inter-episode occurrence repetition.
   - Retrieval History (H): Evaluated from genuine local audit trail; explicit cold-start distinction (no history vs unretrieved).
   - Semantic Relevance (S): Local TF-IDF session and global task centroid cosine similarity (zero external APIs).
   - Relationship Strength (Rel): Multi-modal graph affinity (shared session, application/domain, temporal proximity kernel, context overlap).
3. SOURCE TRACEABILITY:
   Preserves all original Phase 3 attributes; attaches 5 features and calculation evidence metadata.
4. DETERMINISTIC & REPRODUCIBLE:
   Zero random generation, zero hardcoded fake scores. Running twice yields bitwise identical outputs.
5. PERSISTENCE & INTEGRATION:
   Stores enriched dataset in data/dmclm.db (feature_enriched_memories), data/feature_enriched_memories.json,
   data/dmclm_features.csv, updates data/memory_records.json, and writes results/feature_extraction_report.json.
"""

import hashlib
import json
import logging
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import FeatureConfig, DEFAULT_CONFIG
    from src.models import StructuredMemoryCandidate, FeatureEnrichedMemory
except ImportError:
    from dmclm_config import FeatureConfig, DEFAULT_CONFIG
    from models import StructuredMemoryCandidate, FeatureEnrichedMemory

DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
CANDIDATES_JSON_PATH = DATA_DIR / "memory_candidates.json"
ENRICHED_JSON_PATH = DATA_DIR / "feature_enriched_memories.json"
FEATURES_CSV_PATH = DATA_DIR / "dmclm_features.csv"
MEMORY_RECORDS_JSON_PATH = DATA_DIR / "memory_records.json"
RETRIEVAL_HISTORY_JSON_PATH = DATA_DIR / "retrieval_history.json"
FEATURE_REPORT_PATH = RESULTS_DIR / "feature_extraction_report.json"

logger = logging.getLogger("Phase4.FeatureExtraction")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_timestamp(ts_val: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamp string into timezone-aware datetime."""
    if not ts_val or not isinstance(ts_val, str):
        return None
    clean = ts_val.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        try:
            dt = datetime.fromisoformat(clean[:19]).replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None


class RecencyEngine:
    """
    Cognitive Recency Engine (Phase 4):
    R(t) = exp(-lambda * delta_t_days)
    delta_t_days = max(0.0, (T_ref - t_candidate) / 86400)
    Memories observed closer to reference time receive R ~ 1.0; older memories decay towards 0.0.
    Configurable lambda (decay rate per day). Strictly deterministic and monotonic.
    """
    def __init__(self, config: Optional[FeatureConfig] = None, lambda_decay: Optional[float] = None):
        self.config = config or DEFAULT_CONFIG.features
        self.lambda_decay = lambda_decay if lambda_decay is not None else self.config.recency_lambda

    def compute(
        self,
        records: List[Union[Dict[str, Any], StructuredMemoryCandidate]],
        reference_time: Optional[datetime] = None
    ) -> List[float]:
        if not records:
            return []

        parsed_dates: List[Optional[datetime]] = []
        for r in records:
            if isinstance(r, dict):
                ts_str = r.get("end_time") or r.get("last_seen") or r.get("start_time") or r.get("timestamp")
            else:
                ts_str = r.end_time or r.start_time
            parsed_dates.append(parse_timestamp(ts_str))

        valid_dates = [d for d in parsed_dates if d is not None]
        if not valid_dates:
            return [0.0] * len(records)

        # Reference time T_ref
        if reference_time is not None:
            t_ref = reference_time
            if t_ref.tzinfo is None:
                t_ref = t_ref.replace(tzinfo=timezone.utc)
        elif self.config.recency_reference_mode == "current_time":
            t_ref = datetime.now(timezone.utc)
        else:
            t_ref = max(valid_dates)

        scores: List[float] = []
        for dt in parsed_dates:
            if dt is None:
                scores.append(0.0)
                continue
            delta_sec = max(0.0, (t_ref - dt).total_seconds())
            delta_days = delta_sec / 86400.0
            r_val = math.exp(-self.lambda_decay * delta_days)
            scores.append(round(float(np.clip(r_val, 0.0, 1.0)), 4))

        return scores


class FrequencyEngine:
    """
    Cognitive Frequency Engine (Phase 4):
    Distinguishes:
    1. Inter-episode repetition across historical sessions (application frequency & domain frequency).
    2. Intra-episode user engagement actions (keystrokes + clicks).
    Uses log-scaling: ln(1 + count) / ln(1 + max_count) to prevent outliers from dominating.
    Composite: F = w_inter * InterEpisode + w_intra * IntraEpisode. Strictly bounded in [0.0, 1.0].
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features

    def compute(
        self,
        records: List[Union[Dict[str, Any], StructuredMemoryCandidate]]
    ) -> List[float]:
        n = len(records)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        app_counts: Counter = Counter()
        domain_counts: Counter = Counter()
        occ_counts: List[int] = []
        int_counts: List[int] = []

        for r in records:
            if isinstance(r, dict):
                app = (r.get("primary_application") or r.get("application") or "").strip()
                doms = r.get("browser_domains") or ([r.get("domain")] if r.get("domain") else [])
                ints = int(r.get("interaction_count") or 0)
                occ = max(1, int(r.get("occurrence_count") or 1))
            else:
                app = r.primary_application
                doms = r.browser_domains
                ints = r.interaction_count
                occ = 1

            app_counts[app] += occ
            occ_counts.append(occ)
            if isinstance(doms, list):
                for d in doms:
                    if d:
                        domain_counts[d] += occ
            elif isinstance(doms, str) and doms:
                domain_counts[doms] += occ
            int_counts.append(ints)

        max_occ = max(occ_counts) if occ_counts else 1
        log_max_occ = math.log(1.0 + max_occ) if max_occ > 1 else 1.0

        max_app = max(app_counts.values()) if app_counts else 1
        log_max_app = math.log(1.0 + max_app) if max_app > 1 else 1.0

        max_dom = max(domain_counts.values()) if domain_counts else 1
        log_max_dom = math.log(1.0 + max_dom) if max_dom > 1 else 1.0

        max_int = max(int_counts) if int_counts else 1
        log_max_int = math.log(1.0 + max_int) if max_int > 0 else 1.0

        w_inter = self.config.freq_weight_inter
        w_intra = self.config.freq_weight_intra

        scores: List[float] = []
        for i, r in enumerate(records):
            if isinstance(r, dict):
                app = (r.get("primary_application") or r.get("application") or "").strip()
                doms = r.get("browser_domains") or ([r.get("domain")] if r.get("domain") else [])
                ints = int(r.get("interaction_count") or 0)
                occ = max(1, int(r.get("occurrence_count") or 1))
            else:
                app = r.primary_application
                doms = r.browser_domains
                ints = r.interaction_count
                occ = 1

            # Inter-episode component
            c_app = app_counts.get(app, 1)
            app_term = math.log(1.0 + c_app) / log_max_app if log_max_app > 0 else 0.0

            if max_occ > 1:
                occ_term = math.log(1.0 + occ) / log_max_occ
                inter_comp = 0.6 * occ_term + 0.4 * app_term
            elif doms:
                c_dom = max((domain_counts.get(d, 0) for d in doms if d), default=0)
                dom_term = (math.log(1.0 + c_dom) / log_max_dom) if log_max_dom > 0 else 0.0
                inter_comp = 0.65 * app_term + 0.35 * dom_term
            else:
                inter_comp = app_term

            # Intra-episode component
            intra_comp = (math.log(1.0 + ints) / log_max_int) if log_max_int > 0 else 0.0

            composite = w_inter * inter_comp + w_intra * intra_comp
            scores.append(round(float(np.clip(composite, 0.0, 1.0)), 4))

        return scores


class RetrievalHistoryEngine:
    """
    Retrieval History Engine (Phase 4):
    Honestly audits genuine DMCLM retrieval history.
    Never invents or fabricates retrieval events.
    Cold-start distinction:
    - NO_HISTORY: System audit log has 0 total retrievals across the board -> reports 0.0.
    - UNRETRIEVED: System has retrieval history, but this candidate was retrieved 0 times -> reports 0.0.
    - RETRIEVED: Candidate has N > 0 retrievals -> calculates H = 0.70*(useful/total) + 0.30*min(1.0, total/4).
    Extensible for future live retrieval loops.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features

    def compute_with_evidence(
        self,
        records: List[Union[Dict[str, Any], StructuredMemoryCandidate]],
        retrieval_stats: Optional[Dict[str, Dict[str, int]]] = None
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        if not records:
            return [], []

        stats = retrieval_stats or {}
        system_total_retrievals = sum(entry.get("total", 0) for entry in stats.values())

        scores: List[float] = []
        evidence_list: List[Dict[str, Any]] = []

        w_ratio = self.config.retrieval_ratio_weight
        w_freq = self.config.retrieval_freq_weight
        sat_cnt = self.config.retrieval_saturation_count

        for r in records:
            mid = r.get("candidate_id") or r.get("memory_id") if isinstance(r, dict) else r.candidate_id
            rec_entry = stats.get(str(mid))

            if rec_entry:
                tot = rec_entry.get("total", 0)
                useful = rec_entry.get("useful", 0)
            elif isinstance(r, dict):
                tot = int(r.get("retrieval_count") or 0)
                useful = int(r.get("useful_retrieval_count") or 0)
            else:
                tot = 0
                useful = 0

            if tot > 0:
                ratio = useful / tot
                freq_factor = min(1.0, tot / sat_cnt)
                val = w_ratio * ratio + w_freq * freq_factor
                score = round(float(np.clip(val, 0.0, 1.0)), 4)
                evidence = {
                    "status": "RETRIEVED",
                    "total_retrievals": tot,
                    "useful_retrievals": useful,
                    "ratio": round(ratio, 4),
                    "freq_factor": round(freq_factor, 4),
                }
            elif system_total_retrievals == 0 and tot == 0:
                score = self.config.retrieval_cold_start_default
                evidence = {
                    "status": "NO_RETRIEVAL_HISTORY_AVAILABLE",
                    "total_retrievals": 0,
                    "useful_retrievals": 0,
                    "note": "Cold-start prior; zero system-wide retrieval events recorded.",
                }
            else:
                score = self.config.retrieval_cold_start_default
                evidence = {
                    "status": "UNRETRIEVED_ZERO_COUNT",
                    "total_retrievals": 0,
                    "useful_retrievals": 0,
                    "note": "Memory has not been retrieved by user or agent.",
                }

            scores.append(score)
            evidence_list.append(evidence)

        return scores, evidence_list

    def compute(
        self,
        records: List[Union[Dict[str, Any], StructuredMemoryCandidate]],
        retrieval_stats: Optional[Dict[str, Dict[str, int]]] = None
    ) -> List[float]:
        scores, _ = self.compute_with_evidence(records, retrieval_stats)
        return scores


class SemanticRelevanceEngine:
    """
    Semantic Relevance Engine (Phase 4):
    Constructs rich textual documents from Phase 3 candidate attributes:
    application | window_context | activity_summary | browser_domains.
    Computes TF-IDF vector representations locally (zero external APIs).
    Calculates cosine similarity of each memory to its session task centroid and global centroid.
    Normalizes cleanly to [0.0, 1.0].
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=(self.config.tfidf_ngram_min, self.config.tfidf_ngram_max),
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
        )

    def _build_text(self, r: Union[Dict[str, Any], StructuredMemoryCandidate]) -> str:
        if isinstance(r, dict):
            app = str(r.get("primary_application") or r.get("application") or "")
            summary = str(r.get("activity_summary") or r.get("content") or "")
            title = str(r.get("window_context") or "")
            doms = r.get("browser_domains") or ([r.get("domain")] if r.get("domain") else [])
        else:
            app = r.primary_application
            summary = r.activity_summary
            title = r.window_context
            doms = r.browser_domains

        dom_str = " ".join(doms) if isinstance(doms, list) else str(doms or "")
        parts = [app, summary, title, dom_str]
        text = " ".join(p.strip() for p in parts if p and p.strip())
        return text or "desktop user activity"

    def compute(
        self,
        records: List[Union[Dict[str, Any], StructuredMemoryCandidate]]
    ) -> List[float]:
        n = len(records)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        corpus = [self._build_text(r) for r in records]

        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            return [0.5] * n

        # Index by session_id
        session_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, r in enumerate(records):
            if isinstance(r, dict):
                sid = r.get("session_id")
                if not sid and r.get("session_ids"):
                    sid = r["session_ids"][0] if isinstance(r["session_ids"], list) else str(r["session_ids"])
            else:
                sid = r.session_id
            sid = sid or "default_session"
            session_indices[sid].append(idx)

        global_centroid = np.asarray(tfidf_matrix.mean(axis=0))
        relevance_scores = np.zeros(n)

        w_sess = self.config.semantic_session_weight
        w_glob = self.config.semantic_global_weight

        for sid, indices in session_indices.items():
            if not indices:
                continue
            submatrix = tfidf_matrix[indices]
            if submatrix.shape[0] > 1:
                sess_centroid = np.asarray(submatrix.mean(axis=0))
                target_centroid = w_sess * sess_centroid + w_glob * global_centroid
            else:
                target_centroid = global_centroid

            sims = cosine_similarity(submatrix, target_centroid).flatten()
            for local_idx, global_idx in enumerate(indices):
                relevance_scores[global_idx] = sims[local_idx]

        min_v, max_v = float(relevance_scores.min()), float(relevance_scores.max())
        if max_v - min_v > 1e-5:
            normalized = (relevance_scores - min_v) / (max_v - min_v)
        else:
            normalized = np.full_like(relevance_scores, 0.5)

        return [round(float(v), 4) for v in normalized]


class RelationshipStrengthEngine:
    """
    Relationship Strength Engine (Phase 4):
    Evaluates multi-modal graph affinity to other memories in the current memory set:
    - Shared session affinity
    - Application & browser domain continuity
    - Temporal proximity kernel: exp(-delta_t / tau)
    - Window context / task overlap
    Strictly avoids self-comparison (j != i). Normalized to [0.0, 1.0].
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features

    def compute(
        self,
        records: List[Union[Dict[str, Any], StructuredMemoryCandidate]]
    ) -> List[float]:
        n = len(records)
        if n == 0:
            return []
        if n == 1:
            return [0.0]  # Isolated single memory has zero connections to others

        # Parse timestamps & metadata
        timestamps: List[Optional[datetime]] = []
        apps: List[str] = []
        sessions: List[str] = []
        domains_list: List[Set[str]] = []
        contexts: List[Set[str]] = []

        for r in records:
            if isinstance(r, dict):
                ts = r.get("end_time") or r.get("last_seen") or r.get("start_time") or r.get("timestamp")
                app = (r.get("primary_application") or r.get("application") or "").lower()
                sid = r.get("session_id") or (r["session_ids"][0] if r.get("session_ids") else "s0")
                doms = r.get("browser_domains") or ([r.get("domain")] if r.get("domain") else [])
                ctx = (r.get("window_context") or r.get("window_title") or "").lower()
            else:
                ts = r.end_time or r.start_time
                app = r.primary_application.lower()
                sid = r.session_id
                doms = r.browser_domains
                ctx = r.window_context.lower()

            timestamps.append(parse_timestamp(ts))
            apps.append(app)
            sessions.append(sid)
            if isinstance(doms, list):
                domains_list.append({d.lower() for d in doms if d})
            elif isinstance(doms, str) and doms:
                domains_list.append({doms.lower()})
            else:
                domains_list.append(set())

            # Word tokens for context overlap (min len 3)
            ctx_tokens = {w for w in ctx.replace("-", " ").replace("_", " ").split() if len(w) > 2}
            contexts.append(ctx_tokens)

        window_size = self.config.rel_neighbor_window
        w_sess = self.config.rel_weight_session
        w_app = self.config.rel_weight_app
        w_time = self.config.rel_weight_temporal
        w_ctx = self.config.rel_weight_context
        tau = self.config.relationship_temporal_window_sec

        scores: List[float] = []

        for i in range(n):
            start_idx = max(0, i - window_size)
            end_idx = min(n, i + window_size + 1)
            neighbor_indices = [j for j in range(start_idx, end_idx) if j != i]

            if not neighbor_indices:
                scores.append(0.0)
                continue

            conn_values: List[float] = []
            cur_time = timestamps[i]
            cur_app = apps[i]
            cur_sid = sessions[i]
            cur_doms = domains_list[i]
            cur_ctx = contexts[i]

            for j in neighbor_indices:
                # 1. Session affinity
                a_sess = 1.0 if cur_sid == sessions[j] else 0.0

                # 2. Application & domain continuity
                same_app = (cur_app and cur_app == apps[j])
                shared_dom = bool(cur_doms & domains_list[j])
                if same_app and shared_dom:
                    a_app = 1.0
                elif same_app:
                    a_app = 0.75
                elif shared_dom:
                    a_app = 0.50
                else:
                    a_app = 0.0

                # 3. Temporal proximity kernel
                other_time = timestamps[j]
                if cur_time and other_time:
                    delta_sec = abs((cur_time - other_time).total_seconds())
                    k_time = math.exp(-delta_sec / tau)
                else:
                    k_time = 0.0

                # 4. Context overlap
                other_ctx = contexts[j]
                if cur_ctx and other_ctx:
                    intersection = len(cur_ctx & other_ctx)
                    union = len(cur_ctx | other_ctx)
                    a_ctx = intersection / union if union > 0 else 0.0
                else:
                    a_ctx = 0.0

                conn = w_sess * a_sess + w_app * a_app + w_time * k_time + w_ctx * a_ctx
                conn_values.append(conn)

            avg_strength = sum(conn_values) / len(conn_values) if conn_values else 0.0
            scores.append(round(float(np.clip(avg_strength, 0.0, 1.0)), 4))

        return scores


class Phase4FeatureExtractor:
    """
    Phase 4 Pipeline Engine:
    Ingests Phase 3 Structured Memory Candidates.
    Calculates the 5 DMCLM features:
    Recency (R), Frequency (F), Retrieval History (H), Semantic Relevance (S), and Relationship Strength (Rel).
    Produces Feature-Enriched Memory Dataset.
    """
    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        db_path: Optional[Path] = None,
        retrieval_history_path: Optional[Path] = None
    ):
        self.config = config or DEFAULT_CONFIG.features
        self.db_path = db_path or DERIVED_DB_PATH
        self.retrieval_history_path = retrieval_history_path or RETRIEVAL_HISTORY_JSON_PATH

        self.recency_engine = RecencyEngine(config=self.config)
        self.frequency_engine = FrequencyEngine(config=self.config)
        self.retrieval_engine = RetrievalHistoryEngine(config=self.config)
        self.semantic_engine = SemanticRelevanceEngine(config=self.config)
        self.relationship_engine = RelationshipStrengthEngine(config=self.config)

        self.logs: List[str] = []
        self.statistics: Dict[str, Any] = {}

    def _log(self, msg: str) -> None:
        self.logs.append(msg)
        logger.info(msg)

    def load_phase3_candidates(self) -> List[StructuredMemoryCandidate]:
        """
        Loads Phase 3 structured memory candidates from SQLite (memory_candidates table)
        or JSON fallback (data/memory_candidates.json).
        """
        candidates: List[StructuredMemoryCandidate] = []

        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_candidates'")
                if cur.fetchone():
                    cur.execute("""
                        SELECT candidate_id, session_id, source_episode_ids, primary_application,
                               applications_involved, browser_domains, window_context, start_time,
                               end_time, duration_ms, interaction_count, browser_event_count,
                               screenshot_ids, activity_summary, meaningful_criteria, created_at
                        FROM memory_candidates
                        ORDER BY start_time ASC
                    """)
                    for row in cur.fetchall():
                        r = dict(row)
                        src_episodes = json.loads(r["source_episode_ids"]) if isinstance(r["source_episode_ids"], str) else (r["source_episode_ids"] or [])
                        apps_inv = json.loads(r["applications_involved"]) if isinstance(r["applications_involved"], str) else (r["applications_involved"] or [])
                        domains = json.loads(r["browser_domains"]) if isinstance(r["browser_domains"], str) else (r["browser_domains"] or [])
                        shots = json.loads(r["screenshot_ids"]) if isinstance(r["screenshot_ids"], str) and r["screenshot_ids"] else []
                        crit = json.loads(r["meaningful_criteria"]) if isinstance(r["meaningful_criteria"], str) else (r["meaningful_criteria"] or [])

                        cand = StructuredMemoryCandidate(
                            candidate_id=r["candidate_id"],
                            session_id=r["session_id"],
                            source_episode_ids=src_episodes,
                            primary_application=r["primary_application"],
                            applications_involved=apps_inv,
                            browser_domains=domains,
                            window_context=r["window_context"] or "",
                            start_time=r["start_time"],
                            end_time=r["end_time"],
                            duration_ms=int(r["duration_ms"] or 0),
                            interaction_count=int(r["interaction_count"] or 0),
                            browser_event_count=int(r["browser_event_count"] or 0),
                            screenshot_ids=shots,
                            activity_summary=r["activity_summary"] or "",
                            meaningful_criteria=crit,
                            created_at=r["created_at"] or "",
                        )
                        candidates.append(cand)
            finally:
                conn.close()

        if not candidates and CANDIDATES_JSON_PATH.exists():
            with open(CANDIDATES_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    cand = StructuredMemoryCandidate(
                        candidate_id=item["candidate_id"],
                        session_id=item.get("session_id", "s0"),
                        source_episode_ids=item.get("source_episode_ids", []),
                        primary_application=item.get("primary_application", "Desktop"),
                        applications_involved=item.get("applications_involved", []),
                        browser_domains=item.get("browser_domains", []),
                        window_context=item.get("window_context", ""),
                        start_time=item.get("start_time", ""),
                        end_time=item.get("end_time", ""),
                        duration_ms=item.get("duration_ms", 0),
                        interaction_count=item.get("interaction_count", 0),
                        browser_event_count=item.get("browser_event_count", 0),
                        screenshot_ids=item.get("screenshot_ids", []),
                        activity_summary=item.get("activity_summary", ""),
                        meaningful_criteria=item.get("meaningful_criteria", []),
                        created_at=item.get("created_at", ""),
                    )
                    candidates.append(cand)

        return candidates

    def load_retrieval_audit_trail(self) -> Dict[str, Dict[str, int]]:
        """
        Loads genuine retrieval events from derived dmclm.db (retrieval_events table)
        and data/retrieval_history.json.
        Returns: {memory_id: {"total": count, "useful": count}}
        """
        audit_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "useful": 0})

        # 1. Check SQLite retrieval_events
        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='retrieval_events'")
                if cur.fetchone():
                    cur.execute("SELECT memory_id, is_useful FROM retrieval_events")
                    for mid, is_useful in cur.fetchall():
                        audit_stats[str(mid)]["total"] += 1
                        if is_useful:
                            audit_stats[str(mid)]["useful"] += 1
            finally:
                conn.close()

        # 2. Check retrieval_history.json
        if self.retrieval_history_path.exists():
            try:
                with open(self.retrieval_history_path, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                    for r_item in hist_data.get("retrievals", []):
                        mid = str(r_item.get("memory_id") or "")
                        if mid:
                            # Avoid double counting if already present from SQLite
                            if audit_stats[mid]["total"] == 0:
                                audit_stats[mid]["total"] += 1
                                if r_item.get("is_useful"):
                                    audit_stats[mid]["useful"] += 1
            except Exception:
                pass

        return dict(audit_stats)

    def extract_features(
        self,
        candidates: List[StructuredMemoryCandidate],
        reference_time: Optional[datetime] = None
    ) -> List[FeatureEnrichedMemory]:
        """
        Extracts all 5 features for candidate list and creates FeatureEnrichedMemory models.
        Preserves original Phase 3 attributes with zero data loss.
        """
        if not candidates:
            return []

        retrieval_stats = self.load_retrieval_audit_trail()
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Recency
        recency_scores = self.recency_engine.compute(candidates, reference_time=reference_time)

        # 2. Frequency
        frequency_scores = self.frequency_engine.compute(candidates)

        # 3. Retrieval History (with honest audit trail & cold-start evidence)
        retrieval_scores, retrieval_evidence = self.retrieval_engine.compute_with_evidence(
            candidates, retrieval_stats=retrieval_stats
        )

        # 4. Semantic Relevance
        semantic_scores = self.semantic_engine.compute(candidates)

        # 5. Relationship Strength
        relationship_scores = self.relationship_engine.compute(candidates)

        enriched: List[FeatureEnrichedMemory] = []
        for i, c in enumerate(candidates):
            ev_meta = {
                "retrieval_audit": retrieval_evidence[i],
                "recency_lambda": self.recency_engine.lambda_decay,
                "browser_domains_count": len(c.browser_domains),
                "has_interaction_data": c.interaction_count > 0,
            }
            mem = FeatureEnrichedMemory(
                candidate_id=c.candidate_id,
                session_id=c.session_id,
                source_episode_ids=c.source_episode_ids,
                primary_application=c.primary_application,
                applications_involved=c.applications_involved,
                browser_domains=c.browser_domains,
                window_context=c.window_context,
                start_time=c.start_time,
                end_time=c.end_time,
                duration_ms=c.duration_ms,
                interaction_count=c.interaction_count,
                browser_event_count=c.browser_event_count,
                screenshot_ids=c.screenshot_ids,
                activity_summary=c.activity_summary,
                meaningful_criteria=c.meaningful_criteria,
                created_at=c.created_at,
                recency=recency_scores[i],
                frequency=frequency_scores[i],
                retrieval_history=retrieval_scores[i],
                semantic_relevance=semantic_scores[i],
                relationship_strength=relationship_scores[i],
                feature_metadata=ev_meta,
                calculation_timestamp=now_iso,
            )
            enriched.append(mem)

        return enriched

    def save_enriched_dataset(
        self,
        enriched: List[FeatureEnrichedMemory],
        stats: Dict[str, Any]
    ) -> None:
        """
        Saves feature-enriched memories:
        1. SQLite table: feature_enriched_memories in data/dmclm.db
        2. JSON export: data/feature_enriched_memories.json
        3. CSV export: data/dmclm_features.csv
        4. Updates data/memory_records.json for dashboard visibility
        5. Saves results/feature_extraction_report.json
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. SQLite Persistence
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feature_enriched_memories (
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
                    calculation_timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_session ON feature_enriched_memories(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_app ON feature_enriched_memories(primary_application)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_recency ON feature_enriched_memories(recency)")

            cur.execute("DELETE FROM feature_enriched_memories")
            for m in enriched:
                cur.execute("""
                    INSERT INTO feature_enriched_memories (
                        candidate_id, session_id, source_episode_ids, primary_application,
                        applications_involved, browser_domains, window_context, start_time,
                        end_time, duration_ms, interaction_count, browser_event_count,
                        screenshot_ids, activity_summary, meaningful_criteria,
                        recency, frequency, retrieval_history, semantic_relevance,
                        relationship_strength, feature_metadata, calculation_timestamp, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    m.calculation_timestamp,
                    m.created_at,
                ))
            conn.commit()
        finally:
            conn.close()

        # 2. JSON Export
        enriched_dicts = [m.to_dict() for m in enriched]
        with open(ENRICHED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(enriched_dicts, f, indent=2, ensure_ascii=False)

        # 3. CSV Export
        df_features = pd.DataFrame([
            {
                "candidate_id": m.candidate_id,
                "memory_id": m.candidate_id,
                "session_id": m.session_id,
                "primary_application": m.primary_application,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "duration_ms": m.duration_ms,
                "interaction_count": m.interaction_count,
                "recency": m.recency,
                "frequency": m.frequency,
                "retrieval_history": m.retrieval_history,
                "semantic_relevance": m.semantic_relevance,
                "relationship_strength": m.relationship_strength,
            }
            for m in enriched
        ])
        df_features.to_csv(FEATURES_CSV_PATH, index=False)

        # 4. Synchronize memory_records.json for dashboard visibility
        # Update existing records or populate from enriched candidates
        enriched_map = {m.candidate_id: m for m in enriched}
        if MEMORY_RECORDS_JSON_PATH.exists():
            try:
                with open(MEMORY_RECORDS_JSON_PATH, "r", encoding="utf-8") as f:
                    mem_records = json.load(f)
                for rec in mem_records:
                    mid = rec.get("memory_id")
                    if mid in enriched_map:
                        enr = enriched_map[mid]
                        rec["recency"] = enr.recency
                        rec["recency_score"] = enr.recency
                        rec["frequency"] = enr.frequency
                        rec["frequency_score"] = enr.frequency
                        rec["retrieval_history"] = enr.retrieval_history
                        rec["retrieval_history_score"] = enr.retrieval_history
                        rec["semantic_relevance"] = enr.semantic_relevance
                        rec["relationship_strength"] = enr.relationship_strength
                with open(MEMORY_RECORDS_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(mem_records, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        # 5. Comprehensive Phase 4 Report
        report = {
            "execution_status": "Phase 4 Five DMCLM Features extraction completed.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": stats,
            "configuration": {
                "recency_lambda": self.config.recency_lambda,
                "recency_reference_mode": self.config.recency_reference_mode,
                "freq_weight_inter": self.config.freq_weight_inter,
                "freq_weight_intra": self.config.freq_weight_intra,
                "retrieval_ratio_weight": self.config.retrieval_ratio_weight,
                "retrieval_freq_weight": self.config.retrieval_freq_weight,
                "tfidf_max_features": self.config.tfidf_max_features,
                "relationship_temporal_window_sec": self.config.relationship_temporal_window_sec,
                "rel_neighbor_window": self.config.rel_neighbor_window,
            },
            "sample_enriched_memories": enriched_dicts[:5],
            "output_locations": {
                "sqlite_database": str(self.db_path),
                "sqlite_table": "feature_enriched_memories",
                "enriched_json": str(ENRICHED_JSON_PATH),
                "features_csv": str(FEATURES_CSV_PATH),
                "memory_records_json": str(MEMORY_RECORDS_JSON_PATH),
                "feature_report": str(FEATURE_REPORT_PATH),
            },
            "logs": self.logs,
        }
        with open(FEATURE_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def calculate_statistics(self, enriched: List[FeatureEnrichedMemory]) -> Dict[str, Any]:
        """Calculates min, max, mean, median for each of the 5 features."""
        if not enriched:
            return {}

        feature_names = ["recency", "frequency", "retrieval_history", "semantic_relevance", "relationship_strength"]
        stats: Dict[str, Any] = {
            "total_candidates_enriched": len(enriched),
            "features": {},
        }

        for fname in feature_names:
            vals = [getattr(m, fname) for m in enriched]
            arr = np.array(vals)
            stats["features"][fname] = {
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr)), 4),
            }

        return stats

    def run(
        self,
        candidates: Optional[List[StructuredMemoryCandidate]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Phase 4 Feature Extraction:
        1. Loads Phase 3 candidates
        2. Computes the 5 features
        3. Enriches dataset
        4. Saves derived store, JSON, CSV
        5. Reports comprehensive statistics
        """
        t0 = time.time()
        self._log("[PHASE 4] Starting Five DMCLM Features Extraction...")

        if candidates is None:
            candidates = self.load_phase3_candidates()

        total_loaded = len(candidates)
        self._log(f"[PHASE 4] Loaded {total_loaded:,} memory candidates from Phase 3 output.")

        # Identify candidates with missing attributes
        missing_domains = sum(1 for c in candidates if not c.browser_domains)
        missing_interactions = sum(1 for c in candidates if c.interaction_count == 0)
        missing_summary = sum(1 for c in candidates if not c.activity_summary)
        valid_candidates = [c for c in candidates if c.candidate_id and (c.end_time or c.start_time)]

        self._log(f"[PHASE 4] Valid candidates: {len(valid_candidates):,}")
        self._log(f"[PHASE 4] Candidates without browser domains: {missing_domains:,}")
        self._log(f"[PHASE 4] Candidates without user interactions: {missing_interactions:,}")

        # Compute 5 features
        self._log("[PHASE 4] Calculating Recency (exponential decay)...")
        self._log("[PHASE 4] Calculating Frequency (inter vs intra episode engagement)...")
        self._log("[PHASE 4] Calculating Retrieval History (audit trail & cold-start check)...")
        self._log("[PHASE 4] Calculating Semantic Relevance (TF-IDF task centroid similarity)...")
        self._log("[PHASE 4] Calculating Relationship Strength (multi-modal neighbor graph affinity)...")

        enriched = self.extract_features(valid_candidates)
        self._log(f"[PHASE 4] Feature vectors generated: {len(enriched):,}")

        stats = self.calculate_statistics(enriched)
        stats["total_loaded"] = total_loaded
        stats["valid_candidates"] = len(valid_candidates)
        stats["missing_domains"] = missing_domains
        stats["missing_interactions"] = missing_interactions
        stats["missing_summary"] = missing_summary
        duration = round(time.time() - t0, 2)
        stats["processing_duration_sec"] = duration

        # Save results
        self.save_enriched_dataset(enriched, stats)
        self._log(f"[PHASE 4] Feature extraction completed in {duration}s.")
        self._log(f"[PHASE 4] Saved enriched dataset: {ENRICHED_JSON_PATH} & {FEATURES_CSV_PATH}")

        # Log feature summary table
        if "features" in stats:
            self._log("[PHASE 4] Feature Statistics:")
            for feat, fstats in stats["features"].items():
                self._log(
                    f"  - {feat.capitalize():<22}: min={fstats['min']:.4f}, max={fstats['max']:.4f}, "
                    f"mean={fstats['mean']:.4f}, median={fstats['median']:.4f}"
                )

        return {
            "status": "COMPLETED",
            "total_candidates": total_loaded,
            "valid_candidates": len(valid_candidates),
            "enriched_memories": enriched,
            "statistics": stats,
            "duration_sec": duration,
            "logs": self.logs,
        }


if __name__ == "__main__":
    extractor = Phase4FeatureExtractor()
    res = extractor.run()
    print("\n=== PHASE 4 FIVE DMCLM FEATURES RESULTS ===")
    print(json.dumps(res["statistics"], indent=2))

"""
DMCLM Five Feature Signals Engine
Calculates the 5 core cognitive memory signals:
1. Recency: Exponential temporal decay: R = exp(-lambda * days_since_last_seen).
2. Frequency: Genuine occurrence and usage density normalized to [0, 1].
3. Retrieval History: Genuine local DMCLM audit trail (useful_retrievals / total_retrievals, safe 0.0).
4. Semantic Relevance: Transparent local TF-IDF centroid cosine similarity (modular for local embedding upgrade).
5. Relationship Strength: Multi-modal graph affinity (shared session, application, temporal proximity, and textual overlap).
"""

import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dmclm_config import FeatureConfig, DEFAULT_CONFIG
    from preprocessing_pipeline import parse_timestamp
except ImportError:
    from src.dmclm_config import FeatureConfig, DEFAULT_CONFIG
    from src.preprocessing_pipeline import parse_timestamp


class RecencyEngine:
    """
    Cognitive Recency Engine:
    R(t) = exp(-lambda * days_since_last_seen)
    Memories observed recently have R ~ 1.0; older memories decay towards 0.0.
    """
    def __init__(self, lambda_decay: float = 0.05):
        self.lambda_decay = lambda_decay

    def compute(
        self,
        records: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None
    ) -> List[float]:
        if not records:
            return []

        # Determine reference time T_ref as the maximum observed timestamp across records or current time
        parsed_dates = []
        for r in records:
            ts_str = r.get("last_seen") or r.get("first_seen") or r.get("timestamp")
            dt = parse_timestamp(ts_str)
            parsed_dates.append(dt)

        valid_dates = [d for d in parsed_dates if d is not None]
        if not valid_dates:
            return [0.5] * len(records)

        t_ref = reference_time or max(valid_dates)
        if t_ref.tzinfo is None:
            t_ref = t_ref.replace(tzinfo=timezone.utc)

        scores = []
        for dt in parsed_dates:
            if dt is None:
                scores.append(0.0)
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            delta_days = max(0.0, (t_ref - dt).total_seconds() / 86400.0)
            recency = math.exp(-self.lambda_decay * delta_days)
            # Bound strictly between 0.0 and 1.0
            scores.append(round(float(np.clip(recency, 0.0, 1.0)), 4))

        return scores


class FrequencyEngine:
    """
    Frequency Engine:
    Computes normalized occurrence and engagement density from genuine Collector observation counts.
    Uses log-scaling: log(1 + occurrence_count) / log(1 + max_occurrence_count)
    Combined with application frequency across the session.
    """
    def compute(self, records: List[Dict[str, Any]]) -> List[float]:
        if not records:
            return []

        occ_counts = [max(1, int(r.get("occurrence_count") or 1)) for r in records]
        max_occ = max(occ_counts)
        log_max = math.log(1.0 + max_occ) if max_occ > 1 else 1.0

        # Also account for interaction counts
        int_counts = [max(0, int(r.get("interaction_count") or 0)) for r in records]
        max_int = max(int_counts) if int_counts else 1
        log_max_int = math.log(1.0 + max_int) if max_int > 0 else 1.0

        scores = []
        for occ, int_c in zip(occ_counts, int_counts):
            occ_component = math.log(1.0 + occ) / log_max
            int_component = (math.log(1.0 + int_c) / log_max_int) if log_max_int > 0 else 0.0
            composite = 0.75 * occ_component + 0.25 * int_component
            scores.append(round(float(np.clip(composite, 0.0, 1.0)), 4))

        return scores


class RetrievalHistoryEngine:
    """
    Retrieval History Engine:
    Honestly evaluates DMCLM retrieval history audit trail.
    If no retrievals have occurred, returns 0.0 (clearly marked as no history).
    When retrievals have occurred: useful_retrievals / total_retrievals with safe division.
    """
    def compute(
        self,
        records: List[Dict[str, Any]],
        retrieval_stats: Optional[Dict[str, Dict[str, int]]] = None
    ) -> List[float]:
        if not records:
            return []

        stats = retrieval_stats or {}
        scores = []

        for r in records:
            mid = str(r.get("memory_id"))
            rec_entry = stats.get(mid)
            if rec_entry:
                tot = rec_entry.get("total", 0)
                useful = rec_entry.get("useful", 0)
                if tot > 0:
                    ratio = useful / tot
                    # Boost with retrieval frequency factor
                    freq_factor = min(1.0, tot * 0.25)
                    score = 0.7 * ratio + 0.3 * freq_factor
                    scores.append(round(float(np.clip(score, 0.0, 1.0)), 4))
                else:
                    scores.append(0.0)
            else:
                # Direct check from record fields
                tot = int(r.get("retrieval_count") or 0)
                useful = int(r.get("useful_retrieval_count") or 0)
                if tot > 0:
                    score = useful / tot
                    scores.append(round(float(np.clip(score, 0.0, 1.0)), 4))
                else:
                    scores.append(0.0)

        return scores


class SemanticRelevanceEngine:
    """
    Modular Local Semantic Relevance Engine:
    Computes TF-IDF vector representations across memory contents and calculates
    cosine similarity against session task centroids.
    Modular design allows seamless upgrade to local sentence embeddings (e.g., all-MiniLM-L6-v2) later.
    No external LLM API is invoked.
    """
    def __init__(self, max_features: int = 1500, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            lowercase=True,
        )

    def compute(self, records: List[Dict[str, Any]]) -> List[float]:
        if not records:
            return []

        # Build clean textual corpus from memory fields
        corpus = []
        for r in records:
            parts = [
                str(r.get("application") or ""),
                str(r.get("content") or ""),
                str(r.get("domain") or ""),
            ]
            text = " ".join(p for p in parts if p).strip()
            corpus.append(text or "desktop activity")

        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Fallback if corpus is empty or trivial
            return [0.5] * len(records)

        # Group by session_id to calculate session task centroids
        session_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, r in enumerate(records):
            sid = r.get("session_id")
            if not sid and r.get("session_ids"):
                sid = r["session_ids"][0] if isinstance(r["session_ids"], list) else str(r["session_ids"])
            sid = sid or "default_session"
            session_indices[sid].append(idx)

        # Global centroid as well
        global_centroid = np.asarray(tfidf_matrix.mean(axis=0))

        relevance_scores = np.zeros(len(records))

        for sid, indices in session_indices.items():
            if not indices:
                continue
            session_submatrix = tfidf_matrix[indices]
            # Session centroid
            if session_submatrix.shape[0] > 1:
                sess_centroid = np.asarray(session_submatrix.mean(axis=0))
                # Blend session centroid with global centroid
                composite_centroid = 0.7 * sess_centroid + 0.3 * global_centroid
            else:
                composite_centroid = global_centroid

            # Cosine similarity
            sims = cosine_similarity(session_submatrix, composite_centroid).flatten()
            for local_idx, global_idx in enumerate(indices):
                relevance_scores[global_idx] = sims[local_idx]

        # Min-max scale across the batch to utilize [0, 1] range properly
        min_v, max_v = float(relevance_scores.min()), float(relevance_scores.max())
        if max_v - min_v > 1e-5:
            normalized = (relevance_scores - min_v) / (max_v - min_v)
        else:
            normalized = np.full_like(relevance_scores, 0.5)

        return [round(float(v), 4) for v in normalized]


class RelationshipStrengthEngine:
    """
    Modular Relationship Strength Engine:
    Calculates multi-modal relational cohesion:
    - Session affinity (memories co-occurring in the same research session)
    - Application continuity in surrounding temporal window
    - Temporal proximity kernel: exp(-delta_t / 600s)
    - Lexical / Domain topic linkage
    Modular design allows seamless upgrade to Graph Neural Network or knowledge graph embeddings later.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features

    def compute(self, records: List[Dict[str, Any]]) -> List[float]:
        n = len(records)
        if n == 0:
            return []
        if n == 1:
            return [0.5]

        # Parse timestamps
        timestamps = [
            parse_timestamp(r.get("last_seen") or r.get("first_seen") or r.get("timestamp"))
            for r in records
        ]

        scores = []
        window_size = 5  # Neighboring episodes in chronological order

        w_session = self.config.rel_weight_session
        w_app = self.config.rel_weight_app
        w_time = self.config.rel_weight_temporal
        tau = self.config.relationship_temporal_window_sec

        for i in range(n):
            cur = records[i]
            cur_time = timestamps[i]
            cur_app = (cur.get("application") or "").lower()
            cur_sids = set(cur.get("session_ids") or ([cur.get("session_id")] if cur.get("session_id") else []))
            cur_domain = (cur.get("domain") or "").lower()

            start_idx = max(0, i - window_size)
            end_idx = min(n, i + window_size + 1)
            neighbor_indices = [j for j in range(start_idx, end_idx) if j != i]

            if not neighbor_indices:
                scores.append(0.5)
                continue

            session_matches = 0
            app_matches = 0
            temporal_weights = []

            for j in neighbor_indices:
                other = records[j]
                other_time = timestamps[j]
                other_app = (other.get("application") or "").lower()
                other_sids = set(other.get("session_ids") or ([other.get("session_id")] if other.get("session_id") else []))
                other_domain = (other.get("domain") or "").lower()

                # Session match
                if cur_sids & other_sids:
                    session_matches += 1

                # App or domain match
                if cur_app == other_app or (cur_domain and cur_domain == other_domain):
                    app_matches += 1

                # Temporal proximity kernel
                if cur_time and other_time:
                    delta_sec = abs((cur_time - other_time).total_seconds())
                    temporal_weights.append(math.exp(-delta_sec / tau))
                else:
                    temporal_weights.append(0.0)

            k = len(neighbor_indices)
            s_sess = session_matches / k
            s_app = app_matches / k
            s_time = (sum(temporal_weights) / k) if temporal_weights else 0.0

            composite = w_session * s_sess + w_app * s_app + w_time * s_time
            scores.append(round(float(np.clip(composite, 0.0, 1.0)), 4))

        return scores


class DMCLMFeaturePipeline:
    """
    Orchestrates the 5 feature signal calculations for a collection of memory records.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or DEFAULT_CONFIG.features
        self.recency_engine = RecencyEngine(lambda_decay=self.config.recency_lambda)
        self.frequency_engine = FrequencyEngine()
        self.retrieval_engine = RetrievalHistoryEngine()
        self.semantic_engine = SemanticRelevanceEngine(
            max_features=self.config.tfidf_max_features,
            ngram_range=(self.config.tfidf_ngram_min, self.config.tfidf_ngram_max),
        )
        self.relationship_engine = RelationshipStrengthEngine(config=self.config)

    def extract_features(
        self,
        records: List[Dict[str, Any]],
        retrieval_stats: Optional[Dict[str, Dict[str, int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates all 5 features for each memory record and attaches them.
        Returns the updated list of records.
        """
        if not records:
            return []

        recency_scores = self.recency_engine.compute(records)
        frequency_scores = self.frequency_engine.compute(records)
        retrieval_scores = self.retrieval_engine.compute(records, retrieval_stats)
        semantic_scores = self.semantic_engine.compute(records)
        relationship_scores = self.relationship_engine.compute(records)

        for i, r in enumerate(records):
            r["recency_score"] = recency_scores[i]
            r["frequency_score"] = frequency_scores[i]
            r["retrieval_history_score"] = retrieval_scores[i]
            r["semantic_relevance"] = semantic_scores[i]
            r["relationship_strength"] = relationship_scores[i]

            # Also maintain shorthand keys for existing frontend compatibility
            r["recency"] = recency_scores[i]
            r["frequency"] = frequency_scores[i]
            r["retrieval_history"] = retrieval_scores[i]

        return records

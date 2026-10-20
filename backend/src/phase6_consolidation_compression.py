"""
Phase 6 — Consolidation & Compression Engine
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles & Architecture:
1. GENUINE DATA-DRIVEN CONSOLIDATION:
   Transforms Phase 5 baseline memory candidates into durable, compact long-term memory (LTM) nodes.
   Identifies memories with overlapping, repetitive, or redundant context using real evidence
   (semantic similarity, shared applications, shared domains, window context overlap, temporal proximity, session continuity).
   No fake memories, no hardcoded compression ratios, no artificial examples.
2. CANDIDATE BLOCKING & SCALABILITY:
   Avoids O(N^2) exhaustive pairwise comparison across thousands of records by blocking
   candidates by application and temporal windows (<= 24h), evaluating candidate pairs deterministically.
3. MULTI-EVIDENCE AFFINITY & EXPLAINABILITY:
   For every consolidation group, records multi-evidence affinity metrics and human-readable explanations.
4. COMPLETE PROVENANCE & TRACEABILITY:
   Preserves full linkage: Consolidated Memory Node -> Source Memory IDs -> Phase 2 Episode IDs.
   The original memories remain completely intact and un-deleted.
5. DETERMINISTIC CLUSTERING & DETERMINISTIC IDS:
   Grouping and node IDs (SHA-256 hash of sorted source IDs) are strictly reproducible across runs.
6. GENUINE COMPRESSION METRICS:
   Calculates actual metrics: original memories, consolidated memories, memories removed,
   compression ratio, memory reduction percentage, and average source memories per group.
7. COMPATIBLE STORAGE & SYNCHRONIZATION:
   Stores structured Phase 6 nodes to SQLite (`consolidated_memories` & `long_term_memories`),
   JSON artifacts (`data/consolidated_memories.json`, `data/long_term_memories.json`),
   and comprehensive metrics reports (`results/consolidation_metrics.json`, `results/consolidation_report.json`).
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import ConsolidationConfig, DMCLMGlobalConfig, DEFAULT_CONFIG
    from src.models import BaselineScoredMemory, Phase6ConsolidatedMemory
    from src.dmclm_db import (
        init_dmclm_database,
        get_dmclm_connection,
        insert_or_replace_consolidated_memories,
        query_consolidated_memories,
    )
except ImportError:
    from dmclm_config import ConsolidationConfig, DMCLMGlobalConfig, DEFAULT_CONFIG
    from models import BaselineScoredMemory, Phase6ConsolidatedMemory
    from dmclm_db import (
        init_dmclm_database,
        get_dmclm_connection,
        insert_or_replace_consolidated_memories,
        query_consolidated_memories,
    )

DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
BASELINE_SCORED_JSON_PATH = DATA_DIR / "baseline_scored_memories.json"
CONSOLIDATED_JSON_PATH = DATA_DIR / "consolidated_memories.json"
LTM_JSON_PATH = DATA_DIR / "long_term_memories.json"
CONSOLIDATION_METRICS_PATH = RESULTS_DIR / "consolidation_metrics.json"
CONSOLIDATION_REPORT_PATH = RESULTS_DIR / "consolidation_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase6.Consolidation")


# Standard stop words for tokenization
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


def tokenize_text(text: Optional[str]) -> Set[str]:
    """Tokenize and normalize text into non-empty alphanumeric lowercase tokens, omitting stopwords."""
    if not text:
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    tokens = {t for t in cleaned.split() if len(t) > 2 and t not in STOP_WORDS}
    return tokens


def parse_timestamp_iso(ts_str: Optional[str]) -> float:
    """Parse ISO timestamp string to epoch seconds float."""
    if not ts_str:
        return 0.0
    try:
        # Handle trailing Z or offsets
        s = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        try:
            # Fallback for date strings with excess precision
            s = ts_str.split("+")[0].split("Z")[0]
            dt = datetime.strptime(s[:26], "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return 0.0


class DisjointSet:
    """Deterministic Disjoint Set (Union-Find) for connected components grouping."""

    def __init__(self, elements: List[str]):
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, item: str) -> str:
        path = []
        curr = item
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]
        # Path compression
        for node in path:
            self.parent[node] = curr
        return curr

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            # Tie-break deterministically using string order if ranks match
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                if root_a < root_b:
                    self.parent[root_b] = root_a
                    self.rank[root_a] += 1
                else:
                    self.parent[root_a] = root_b
                    self.rank[root_b] += 1

    def get_clusters(self) -> Dict[str, List[str]]:
        clusters = defaultdict(list)
        for elem in sorted(self.parent.keys()):
            root = self.find(elem)
            clusters[root].append(elem)
        # Ensure all cluster member lists are strictly sorted
        return {r: sorted(members) for r, members in clusters.items()}


class Phase6Consolidator:
    """
    Core implementation of Phase 6 — Consolidation & Compression.
    Transforms individual Phase 5 memory candidates into consolidated long-term memory nodes.
    """

    def __init__(self, config: Optional[ConsolidationConfig] = None):
        self.config = config or DEFAULT_CONFIG.consolidation
        self.config.validate()

    def calculate_pairwise_affinity(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        m1_tokens: Optional[Set[str]] = None,
        m2_tokens: Optional[Set[str]] = None,
        m1_ctx_tokens: Optional[Set[str]] = None,
        m2_ctx_tokens: Optional[Set[str]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates multi-evidence affinity between two memory candidates.
        Returns:
            composite_affinity: float in [0.0, 1.0]
            evidence_breakdown: Dict of component scores and matches
        """
        # 1. Application match
        app1 = (m1.get("primary_application") or m1.get("application") or "").strip().lower()
        app2 = (m2.get("primary_application") or m2.get("application") or "").strip().lower()
        apps1 = {a.strip().lower() for a in (m1.get("applications_involved") or []) if a}
        apps2 = {a.strip().lower() for a in (m2.get("applications_involved") or []) if a}
        if app1:
            apps1.add(app1)
        if app2:
            apps2.add(app2)

        app_match = 0.0
        if app1 and app2 and app1 == app2:
            app_match = 1.0
        elif apps1 and apps2 and (apps1 & apps2):
            app_match = len(apps1 & apps2) / len(apps1 | apps2)

        # 2. Browser Domain match
        dom1 = {d.strip().lower() for d in (m1.get("browser_domains") or []) if d}
        dom2 = {d.strip().lower() for d in (m2.get("browser_domains") or []) if d}
        if dom1 and dom2:
            domain_match = len(dom1 & dom2) / len(dom1 | dom2)
        elif not dom1 and not dom2:
            # Both are desktop application activities lacking browser domains
            domain_match = 1.0 if app_match > 0.0 else 0.0
        else:
            domain_match = 0.0

        # 3. Window Context Token Overlap
        if m1_ctx_tokens is None:
            m1_ctx_tokens = tokenize_text(m1.get("window_context") or "")
        if m2_ctx_tokens is None:
            m2_ctx_tokens = tokenize_text(m2.get("window_context") or "")

        if m1_ctx_tokens and m2_ctx_tokens:
            ctx_sim = len(m1_ctx_tokens & m2_ctx_tokens) / len(m1_ctx_tokens | m2_ctx_tokens)
        elif not m1_ctx_tokens and not m2_ctx_tokens:
            ctx_sim = 1.0 if app_match > 0.0 else 0.0
        else:
            ctx_sim = 0.0

        # 4. Semantic Similarity (activity summary + context tokens)
        if m1_tokens is None:
            text1 = f"{m1.get('activity_summary', '')} {m1.get('window_context', '')}"
            m1_tokens = tokenize_text(text1)
        if m2_tokens is None:
            text2 = f"{m2.get('activity_summary', '')} {m2.get('window_context', '')}"
            m2_tokens = tokenize_text(text2)

        if m1_tokens and m2_tokens:
            sem_sim = len(m1_tokens & m2_tokens) / len(m1_tokens | m2_tokens)
        elif not m1_tokens and not m2_tokens:
            sem_sim = 1.0 if app_match > 0.0 else 0.0
        else:
            sem_sim = 0.0

        # 5. Temporal Proximity
        t1_start = parse_timestamp_iso(m1.get("start_time") or m1.get("first_seen"))
        t1_end = parse_timestamp_iso(m1.get("end_time") or m1.get("last_seen")) or t1_start
        t2_start = parse_timestamp_iso(m2.get("start_time") or m2.get("first_seen"))
        t2_end = parse_timestamp_iso(m2.get("end_time") or m2.get("last_seen")) or t2_start

        # Calculate time gap
        if t1_start > 0 and t2_start > 0:
            if t1_end < t2_start:
                delta_sec = t2_start - t1_end
            elif t2_end < t1_start:
                delta_sec = t1_start - t2_end
            else:
                delta_sec = 0.0  # Overlapping intervals

            tau = max(1.0, float(self.config.temporal_proximity_tau_sec))
            temporal_prox = math.exp(-delta_sec / tau)
        else:
            delta_sec = 0.0
            temporal_prox = 0.5  # Neutral fallback if timestamp unparseable

        # 6. Session Match
        s1 = str(m1.get("session_id") or "")
        s2 = str(m2.get("session_id") or "")
        session_match = 1.0 if (s1 and s2 and s1 == s2) else 0.0

        # Composite Weighted Affinity
        w = self.config
        composite_affinity = (
            w.weight_app * app_match
            + w.weight_domain * domain_match
            + w.weight_context * ctx_sim
            + w.weight_semantic * sem_sim
            + w.weight_temporal * temporal_prox
            + w.weight_session * session_match
        )
        composite_affinity = max(0.0, min(1.0, float(composite_affinity)))

        evidence = {
            "composite_affinity": round(composite_affinity, 4),
            "app_match": round(app_match, 4),
            "domain_match": round(domain_match, 4),
            "context_similarity": round(ctx_sim, 4),
            "semantic_similarity": round(sem_sim, 4),
            "temporal_proximity": round(temporal_prox, 4),
            "temporal_delta_sec": round(delta_sec, 2),
            "session_match": round(session_match, 4),
        }

        return composite_affinity, evidence

    def identify_candidate_relationships(
        self, memories: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[str, str, float, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
        """
        Efficient candidate relationship identification with candidate blocking.
        Blocks candidates by application and bounds temporal search, avoiding full O(N^2) evaluation.
        """
        if not memories or len(memories) < 2:
            return [], {}

        # Precompute tokens and timestamps deterministically
        memory_map: Dict[str, Dict[str, Any]] = {}
        tokens_map: Dict[str, Set[str]] = {}
        ctx_tokens_map: Dict[str, Set[str]] = {}
        timestamps_map: Dict[str, float] = {}

        for m in memories:
            mid = str(m.get("candidate_id") or m.get("memory_id"))
            memory_map[mid] = m
            text = f"{m.get('activity_summary', '')} {m.get('window_context', '')}"
            tokens_map[mid] = tokenize_text(text)
            ctx_tokens_map[mid] = tokenize_text(m.get("window_context") or "")
            timestamps_map[mid] = parse_timestamp_iso(m.get("start_time") or m.get("first_seen"))

        # Deterministic sorting
        sorted_mids = sorted(
            memory_map.keys(),
            key=lambda x: (timestamps_map.get(x, 0.0), x)
        )

        # Block memories by primary application
        blocks: Dict[str, List[str]] = defaultdict(list)
        for mid in sorted_mids:
            m = memory_map[mid]
            app = (m.get("primary_application") or m.get("application") or "unknown").strip().lower()
            blocks[app].append(mid)

        edges: List[Tuple[str, str, float, Dict[str, Any]]] = []
        max_time_gap = float(self.config.temporal_window_sec)

        for app, mids in blocks.items():
            n = len(mids)
            for i in range(n):
                mid_a = mids[i]
                t_a = timestamps_map[mid_a]
                m_a = memory_map[mid_a]

                for j in range(i + 1, n):
                    mid_b = mids[j]
                    t_b = timestamps_map[mid_b]

                    # Temporal blocking check: since mids are sorted by time, if delta exceeds window and both have timestamps
                    if t_a > 0 and t_b > 0 and (t_b - t_a) > max_time_gap:
                        # Allow within-session comparison even if beyond window, otherwise break
                        if m_a.get("session_id") != memory_map[mid_b].get("session_id"):
                            break

                    m_b = memory_map[mid_b]

                    # Calculate pairwise affinity
                    affinity, ev = self.calculate_pairwise_affinity(
                        m_a,
                        m_b,
                        tokens_map[mid_a],
                        tokens_map[mid_b],
                        ctx_tokens_map[mid_a],
                        ctx_tokens_map[mid_b],
                    )

                    # Check consolidation threshold criteria
                    if (
                        affinity >= self.config.affinity_threshold
                        and ev["semantic_similarity"] >= self.config.semantic_similarity_threshold
                        and ev["app_match"] > 0.0
                    ):
                        edges.append((mid_a, mid_b, affinity, ev))

        return edges, memory_map

    def form_consolidation_groups(
        self, memories: List[Dict[str, Any]], edges: List[Tuple[str, str, float, Dict[str, Any]]]
    ) -> Tuple[List[List[str]], List[str], Dict[Tuple[str, str], Dict[str, Any]]]:
        """
        Builds disjoint consolidation groups from linked edges using deterministic DisjointSet.
        Returns:
            groups: list of groups (each containing >= min_group_size memory IDs)
            unconsolidated_ids: list of memory IDs not belonging to any group (singletons)
            edge_evidence_map: lookup for pairwise relationship evidence
        """
        all_ids = sorted({str(m.get("candidate_id") or m.get("memory_id")) for m in memories})
        ds = DisjointSet(all_ids)
        edge_evidence_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for mid_a, mid_b, aff, ev in edges:
            ds.union(mid_a, mid_b)
            key = (min(mid_a, mid_b), max(mid_a, mid_b))
            edge_evidence_map[key] = ev

        clusters = ds.get_clusters()

        groups: List[List[str]] = []
        unconsolidated_ids: List[str] = []

        min_sz = self.config.min_group_size
        for root in sorted(clusters.keys()):
            members = clusters[root]
            if len(members) >= min_sz:
                groups.append(members)
            else:
                unconsolidated_ids.extend(members)

        # Sort groups deterministically by earliest member timestamp, then size, then first ID
        return groups, sorted(unconsolidated_ids), edge_evidence_map

    def generate_consolidated_node(
        self,
        group_ids: List[str],
        memory_map: Dict[str, Dict[str, Any]],
        edge_evidence_map: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Phase6ConsolidatedMemory:
        """
        Generates a consolidated long-term memory representation from a group of source memories.
        Synthesizes actual text representation, preserves complete provenance, and computes summary metrics.
        """
        sorted_mids = sorted(group_ids)
        source_memories = [memory_map[mid] for mid in sorted_mids]

        # 1. Deterministic Consolidated Node ID
        id_hash = hashlib.sha256(",".join(sorted_mids).encode("utf-8")).hexdigest()[:16]
        c_node_id = f"c_node_{id_hash}"

        # 2. Source Episode IDs (traceability)
        source_episodes_set: Set[str] = set()
        for m in source_memories:
            for ep_id in (m.get("source_episode_ids") or []):
                if ep_id:
                    source_episodes_set.add(str(ep_id))
        source_episode_ids = sorted(source_episodes_set)

        # 3. Time bounds & Duration
        start_times = [
            parse_timestamp_iso(m.get("start_time") or m.get("first_seen"))
            for m in source_memories
            if (m.get("start_time") or m.get("first_seen"))
        ]
        end_times = [
            parse_timestamp_iso(m.get("end_time") or m.get("last_seen"))
            for m in source_memories
            if (m.get("end_time") or m.get("last_seen"))
        ]

        valid_starts = [t for t in start_times if t > 0]
        valid_ends = [t for t in end_times if t > 0]

        if valid_starts:
            min_start_epoch = min(valid_starts)
            start_iso = datetime.fromtimestamp(min_start_epoch, tz=timezone.utc).isoformat()
        else:
            start_iso = source_memories[0].get("start_time", "")

        if valid_ends:
            max_end_epoch = max(valid_ends)
            end_iso = datetime.fromtimestamp(max_end_epoch, tz=timezone.utc).isoformat()
        else:
            end_iso = source_memories[-1].get("end_time", "") or start_iso

        span_duration_ms = 0
        if valid_starts and valid_ends and max_end_epoch >= min_start_epoch:
            span_duration_ms = int((max_end_epoch - min_start_epoch) * 1000)

        total_duration_ms = sum(
            int(m.get("duration_ms") or m.get("total_duration") or 0)
            for m in source_memories
        )

        # 4. Applications and Domains
        app_counts: Dict[str, int] = defaultdict(int)
        all_apps_set: Set[str] = set()
        all_domains_set: Set[str] = set()
        contexts_set: Set[str] = set()
        sessions_set: Set[str] = set()

        for m in source_memories:
            p_app = m.get("primary_application") or m.get("application")
            if p_app:
                app_counts[p_app] += 1
                all_apps_set.add(p_app)
            for a in (m.get("applications_involved") or []):
                if a:
                    all_apps_set.add(a)
            for d in (m.get("browser_domains") or []):
                if d:
                    all_domains_set.add(d)
            ctx = m.get("window_context") or m.get("content")
            if ctx and str(ctx).strip():
                contexts_set.add(str(ctx).strip())
            sess = m.get("session_id")
            if sess:
                sessions_set.add(str(sess))

        # Primary application is the most common primary application
        primary_app = max(app_counts.keys(), key=lambda k: app_counts[k]) if app_counts else "Desktop Workflow"
        sorted_apps = sorted(all_apps_set)
        sorted_domains = sorted(all_domains_set)
        sorted_contexts = sorted(contexts_set)
        sorted_sessions = sorted(sessions_set)

        # 5. Combined Interaction Information
        total_interactions = sum(int(m.get("interaction_count") or 0) for m in source_memories)
        total_browser_events = sum(int(m.get("browser_event_count") or 0) for m in source_memories)
        total_screenshots = sum(len(m.get("screenshot_ids") or []) for m in source_memories)

        duration_sec = total_duration_ms / 1000.0
        interaction_density = (total_interactions / duration_sec) if duration_sec > 0 else 0.0

        interaction_stats = {
            "total_interactions": total_interactions,
            "browser_event_count": total_browser_events,
            "screenshot_count": total_screenshots,
            "interaction_density_per_sec": round(interaction_density, 3),
        }

        # 6. Baseline & Feature Metrics Aggregation
        baseline_scores = [float(m.get("baseline_score", 0.0)) for m in source_memories]
        mean_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.5

        recencies = [float(m.get("recency", 0.0)) for m in source_memories]
        frequencies = [float(m.get("frequency", 0.0)) for m in source_memories]
        retrieval_histories = [float(m.get("retrieval_history", 0.0)) for m in source_memories]
        semantic_relevances = [float(m.get("semantic_relevance", 0.0)) for m in source_memories]
        relationship_strengths = [float(m.get("relationship_strength", 0.0)) for m in source_memories]

        # Dominant features count
        dom_feature_counts: Dict[str, int] = defaultdict(int)
        for m in source_memories:
            df = m.get("dominant_feature")
            if df:
                dom_feature_counts[df] += 1
        group_dominant_feature = (
            max(dom_feature_counts.keys(), key=lambda k: dom_feature_counts[k])
            if dom_feature_counts else "semantic_relevance"
        )

        # Calculate stability score: incorporates repetition, interaction density, and baseline score
        interaction_bonus = min(0.20, total_interactions * 0.01)
        source_bonus = min(0.15, len(source_memories) * 0.02)
        stability_score = min(1.0, max(0.0, mean_baseline * 0.65 + interaction_bonus + source_bonus))

        baseline_info = {
            "mean_baseline_score": round(mean_baseline, 4),
            "dominant_feature": group_dominant_feature,
            "stability_score": round(stability_score, 4),
            "feature_means": {
                "recency": round(sum(recencies) / len(recencies), 4) if recencies else 0.0,
                "frequency": round(sum(frequencies) / len(frequencies), 4) if frequencies else 0.0,
                "retrieval_history": round(sum(retrieval_histories) / len(retrieval_histories), 4) if retrieval_histories else 0.0,
                "semantic_relevance": round(sum(semantic_relevances) / len(semantic_relevances), 4) if semantic_relevances else 0.0,
                "relationship_strength": round(sum(relationship_strengths) / len(relationship_strengths), 4) if relationship_strengths else 0.0,
            }
        }

        # 7. Dynamic Semantic Representation Generation
        # Generated from actual window contexts and activity summaries, preserving data integrity
        distinct_window_contexts = sorted_contexts[:4]
        window_summary_part = f" [{', '.join(distinct_window_contexts)}]" if distinct_window_contexts else ""
        domain_part = f" on {', '.join(sorted_domains[:3])}" if sorted_domains else ""

        duration_summary = f"{total_duration_ms / 1000.0:.1f}s" if total_duration_ms >= 1000 else f"{total_duration_ms}ms"

        semantic_rep = (
            f"Consolidated knowledge across {len(source_memories)} episodic memory traces "
            f"in {primary_app}{domain_part}{window_summary_part}. "
            f"Cumulative active dwell: {duration_summary} with {total_interactions} user interactions."
        )

        # 8. Consolidation Evidence Generation
        # Collect pairwise affinities within the group
        pair_affinities: List[float] = []
        for i in range(len(sorted_mids)):
            for j in range(i + 1, len(sorted_mids)):
                pair_key = (sorted_mids[i], sorted_mids[j])
                if pair_key in edge_evidence_map:
                    pair_affinities.append(edge_evidence_map[pair_key]["composite_affinity"])

        mean_affinity = (sum(pair_affinities) / len(pair_affinities)) if pair_affinities else 0.65

        evidence_reasons: List[str] = [
            f"High composite similarity (mean affinity {mean_affinity:.2f} >= threshold {self.config.affinity_threshold})",
            f"Shared primary application context ({primary_app})",
        ]
        if sorted_domains:
            evidence_reasons.append(f"Shared browser domain research: {', '.join(sorted_domains[:3])}")
        if len(sorted_contexts) > 0:
            evidence_reasons.append(f"Overlapping window context tokens across {len(sorted_contexts)} window states")
        if span_duration_ms > 0:
            evidence_reasons.append(f"Temporal continuity span of {span_duration_ms / 1000.0:.1f}s")
        if len(sorted_sessions) == 1:
            evidence_reasons.append("Intra-session task workflow continuity")

        consolidation_evidence = {
            "mean_pairwise_affinity": round(mean_affinity, 4),
            "evidence_reasons": evidence_reasons,
            "shared_applications": sorted_apps,
            "shared_domains": sorted_domains,
            "unique_contexts_count": len(sorted_contexts),
            "temporal_span_sec": round(span_duration_ms / 1000.0, 2),
            "session_continuity": len(sorted_sessions) == 1,
        }

        # 9. Consolidation Metadata
        # Use deterministic timestamp from latest source end_time / created_at for exact reproducibility
        created_at_iso = end_iso or (source_memories[-1].get("created_at") or "")
        if not created_at_iso:
            created_at_iso = datetime.now(timezone.utc).isoformat()

        consolidation_metadata = {
            "created_at": created_at_iso,
            "algorithm": "Multi-Evidence Connected Components Clustering",
            "deterministic_hash": id_hash,
            "primary_session_id": sorted_sessions[0] if sorted_sessions else "",
            "min_group_size": self.config.min_group_size,
            "affinity_threshold": self.config.affinity_threshold,
            "semantic_similarity_threshold": self.config.semantic_similarity_threshold,
        }

        return Phase6ConsolidatedMemory(
            consolidated_memory_id=c_node_id,
            source_memory_ids=sorted_mids,
            source_episode_ids=source_episode_ids,
            start_time=start_iso,
            end_time=end_iso,
            duration_ms=total_duration_ms,
            span_duration_ms=span_duration_ms,
            primary_application=primary_app,
            applications=sorted_apps,
            domains=sorted_domains,
            contexts=sorted_contexts,
            interaction_statistics=interaction_stats,
            semantic_representation=semantic_rep,
            consolidation_evidence=consolidation_evidence,
            source_count=len(source_memories),
            baseline_information=baseline_info,
            consolidation_metadata=consolidation_metadata,
        )

    def calculate_compression_metrics(
        self,
        total_source_memories: int,
        consolidated_nodes: List[Phase6ConsolidatedMemory],
        unconsolidated_count: int,
    ) -> Dict[str, Any]:
        """
        Calculates genuine, transparent compression metrics:
        Formulas:
        - Compression Ratio = 1 - (Consolidated Memories / Original Memories)
        - Memory Reduction Percentage = ((Original Memories - Consolidated Memories) / Original Memories) * 100
        Includes both overall memory space metrics and group-level redundancy metrics.
        """
        N_orig = total_source_memories
        num_consolidated = len(consolidated_nodes)

        # Count how many original source memories were absorbed into consolidation groups
        source_memories_in_groups = sum(node.source_count for node in consolidated_nodes)

        # Total retained memory units in working storage:
        # Each consolidation group produces 1 consolidated memory node.
        # Memories that did not consolidate remain as individual retained memory units.
        N_final = num_consolidated + unconsolidated_count

        # Total memories removed through redundancy / consolidation
        memories_removed = max(0, N_orig - N_final)

        # Overall memory space compression
        if N_orig > 0:
            compression_ratio_overall = 1.0 - (N_final / float(N_orig))
            reduction_percentage_overall = ((N_orig - N_final) / float(N_orig)) * 100.0
        else:
            compression_ratio_overall = 0.0
            reduction_percentage_overall = 0.0

        # Group-specific consolidation metrics (on the subset of memories that qualified for consolidation)
        if source_memories_in_groups > 0:
            group_compression_ratio = 1.0 - (num_consolidated / float(source_memories_in_groups))
            group_reduction_percentage = (
                (source_memories_in_groups - num_consolidated) / float(source_memories_in_groups)
            ) * 100.0
            avg_source_per_consolidated = source_memories_in_groups / float(num_consolidated) if num_consolidated > 0 else 0.0
        else:
            group_compression_ratio = 0.0
            group_reduction_percentage = 0.0
            avg_source_per_consolidated = 0.0

        return {
            "number_of_source_memories": N_orig,
            "number_of_consolidated_memories": num_consolidated,
            "number_of_unconsolidated_memories": unconsolidated_count,
            "total_retained_memory_nodes": N_final,
            "source_memories_in_groups": source_memories_in_groups,
            "memories_removed_through_consolidation": memories_removed,
            "compression_ratio": round(compression_ratio_overall, 4),
            "memory_reduction_percentage": round(reduction_percentage_overall, 2),
            "group_compression_ratio": round(group_compression_ratio, 4),
            "group_reduction_percentage": round(group_reduction_percentage, 2),
            "average_source_memories_per_consolidated_memory": round(avg_source_per_consolidated, 2),
            "formulas_used": {
                "compression_ratio": "1 - (Retained Memories / Original Memories)",
                "memory_reduction_percentage": "((Original Memories - Retained Memories) / Original Memories) * 100",
                "group_reduction_percentage": "((Source Memories In Groups - Consolidated Memories) / Source Memories In Groups) * 100",
                "average_group_size": "Source Memories In Groups / Consolidated Memories"
            }
        }

    def process(
        self, memories: List[Dict[str, Any]]
    ) -> Tuple[List[Phase6ConsolidatedMemory], Dict[str, Any]]:
        """
        Executes the complete Phase 6 Consolidation & Compression pipeline on a list of memory candidates.
        """
        start_time = time.time()
        logger.info(f"[PHASE 6] Ingesting {len(memories):,} Phase 5 memory candidates...")

        if not memories:
            logger.info("[PHASE 6] Received empty memory list. Producing 0 consolidated nodes.")
            empty_metrics = self.calculate_compression_metrics(0, [], 0)
            empty_metrics["processing_time_seconds"] = round(time.time() - start_time, 4)
            return [], empty_metrics

        # Step 1: Identify candidate relationships
        edges, memory_map = self.identify_candidate_relationships(memories)
        logger.info(f"[PHASE 6] Identified {len(edges):,} candidate relationship links satisfying affinity thresholds.")

        # Step 2: Form deterministic consolidation groups
        groups, unconsolidated_ids, edge_evidence_map = self.form_consolidation_groups(memories, edges)
        logger.info(
            f"[PHASE 6] Formed {len(groups):,} consolidation groups (min_size={self.config.min_group_size}). "
            f"{len(unconsolidated_ids):,} memories remain unconsolidated."
        )

        # Step 3: Synthesize durable consolidated memory nodes
        consolidated_nodes: List[Phase6ConsolidatedMemory] = []
        for grp in groups:
            node = self.generate_consolidated_node(grp, memory_map, edge_evidence_map)
            consolidated_nodes.append(node)

        # Sort consolidated nodes deterministically by start time, then node ID
        consolidated_nodes.sort(key=lambda n: (n.start_time, n.consolidated_memory_id))

        # Step 4: Calculate genuine compression metrics
        metrics = self.calculate_compression_metrics(len(memories), consolidated_nodes, len(unconsolidated_ids))
        metrics["candidate_relationships_identified"] = len(edges)
        metrics["consolidation_groups_created"] = len(groups)
        metrics["processing_time_seconds"] = round(time.time() - start_time, 4)

        logger.info(
            f"[PHASE 6] Consolidation complete in {metrics['processing_time_seconds']}s: "
            f"{metrics['number_of_source_memories']:,} -> {metrics['total_retained_memory_nodes']:,} "
            f"({metrics['number_of_consolidated_memories']} groups absorbing {metrics['source_memories_in_groups']:,} memories). "
            f"Overall reduction: {metrics['memory_reduction_percentage']}%, Group reduction: {metrics['group_reduction_percentage']}%."
        )

        return consolidated_nodes, metrics

    def run_pipeline(
        self,
        input_path: Optional[Path] = None,
        save_to_db: bool = True,
        save_to_json: bool = True,
    ) -> Tuple[List[Phase6ConsolidatedMemory], Dict[str, Any]]:
        """
        Runs the full Phase 6 pipeline from persistent storage and persists the output.
        """
        p = input_path or BASELINE_SCORED_JSON_PATH
        if not p.exists():
            raise FileNotFoundError(f"Phase 5 baseline scored memory file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            memories = json.load(f)

        logger.info(f"[PHASE 6] Loaded {len(memories):,} memories from {p}")

        consolidated_nodes, metrics = self.process(memories)

        # Save to JSON outputs
        if save_to_json:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            nodes_dict_list = [node.to_dict() for node in consolidated_nodes]

            with open(CONSOLIDATED_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(nodes_dict_list, f, indent=2)
            logger.info(f"[PHASE 6] Saved {len(nodes_dict_list):,} consolidated memories to {CONSOLIDATED_JSON_PATH}")

            # Also update long_term_memories.json for existing dashboard API compatibility
            with open(LTM_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(nodes_dict_list, f, indent=2)
            logger.info(f"[PHASE 6] Synchronized {len(nodes_dict_list):,} nodes to {LTM_JSON_PATH}")

            # Save metrics
            metrics_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_records": metrics["number_of_source_memories"],
                "compression_metrics": {
                    "original_episodic_records": metrics["number_of_source_memories"],
                    "consolidated_ltm_nodes": metrics["number_of_consolidated_memories"],
                    "unconsolidated_memories": metrics["number_of_unconsolidated_memories"],
                    "total_retained_nodes": metrics["total_retained_memory_nodes"],
                    "memory_reduction_count": metrics["memories_removed_through_consolidation"],
                    "compression_ratio_pct": metrics["memory_reduction_percentage"],
                    "group_compression_ratio_pct": metrics["group_reduction_percentage"],
                    "average_group_size": metrics["average_source_memories_per_consolidated_memory"],
                },
                "status": "DMCLM Phase 6 consolidation execution completed.",
                "configuration": self.config.to_dict(),
            }

            with open(CONSOLIDATION_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)
            logger.info(f"[PHASE 6] Saved consolidation metrics to {CONSOLIDATION_METRICS_PATH}")

            with open(CONSOLIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": metrics,
                    "configuration": self.config.to_dict(),
                    "sample_consolidated_nodes": nodes_dict_list[:5],
                }, f, indent=2)
            logger.info(f"[PHASE 6] Saved detailed report to {CONSOLIDATION_REPORT_PATH}")

        # Save to SQLite database
        if save_to_db:
            init_dmclm_database()
            insert_or_replace_consolidated_memories([n.to_dict() for n in consolidated_nodes])
            logger.info(f"[PHASE 6] Persisted {len(consolidated_nodes):,} consolidated memories to SQLite.")

        return consolidated_nodes, metrics


def trace_consolidated_memory(
    consolidated_id: str,
    consolidated_nodes: Optional[List[Phase6ConsolidatedMemory]] = None,
    source_memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Provenance query function.
    Returns complete traceability: Consolidated Node -> Source Memories -> Source Episodes.
    """
    # Load nodes if not provided
    node_dict = None
    if consolidated_nodes:
        for n in consolidated_nodes:
            if n.consolidated_memory_id == consolidated_id:
                node_dict = n.to_dict()
                break
    elif CONSOLIDATED_JSON_PATH.exists():
        with open(CONSOLIDATED_JSON_PATH, "r", encoding="utf-8") as f:
            all_nodes = json.load(f)
            for n in all_nodes:
                if n.get("consolidated_memory_id") == consolidated_id or n.get("ltm_id") == consolidated_id:
                    node_dict = n
                    break

    if not node_dict:
        raise KeyError(f"Consolidated memory node not found: {consolidated_id}")

    # Load source memories if needed
    source_mids = set(node_dict.get("source_memory_ids", []))
    matched_sources: List[Dict[str, Any]] = []

    if source_memories:
        for m in source_memories:
            mid = str(m.get("candidate_id") or m.get("memory_id"))
            if mid in source_mids:
                matched_sources.append(m)
    elif BASELINE_SCORED_JSON_PATH.exists():
        with open(BASELINE_SCORED_JSON_PATH, "r", encoding="utf-8") as f:
            all_memories = json.load(f)
            for m in all_memories:
                mid = str(m.get("candidate_id") or m.get("memory_id"))
                if mid in source_mids:
                    matched_sources.append(m)

    # Trace back to episodes
    episodes: List[str] = node_dict.get("source_episode_ids", [])

    return {
        "consolidated_memory_id": consolidated_id,
        "source_memory_count": len(source_mids),
        "source_memory_ids": sorted(source_mids),
        "source_episode_ids": episodes,
        "source_memories": matched_sources,
        "consolidation_evidence": node_dict.get("consolidation_evidence", {}),
    }


if __name__ == "__main__":
    consolidator = Phase6Consolidator()
    nodes, metrics = consolidator.run_pipeline()
    print(f"\n[PHASE 6 EXECUTION SUMMARY]")
    print(f"Original Memories:        {metrics['number_of_source_memories']:,}")
    print(f"Consolidated Groups:      {metrics['number_of_consolidated_memories']:,}")
    print(f"Unconsolidated Memories:  {metrics['number_of_unconsolidated_memories']:,}")
    print(f"Total Retained Units:     {metrics['total_retained_memory_nodes']:,}")
    print(f"Memories Absorbed:        {metrics['source_memories_in_groups']:,}")
    print(f"Memories Reduced:         {metrics['memories_removed_through_consolidation']:,}")
    print(f"Overall Reduction Ratio:  {metrics['memory_reduction_percentage']:.2f}%")
    print(f"Group Reduction Ratio:    {metrics['group_reduction_percentage']:.2f}%")
    print(f"Avg Source / Cons Group:  {metrics['average_source_memories_per_consolidated_memory']:.2f}")
    print(f"Execution Time:           {metrics['processing_time_seconds']}s")

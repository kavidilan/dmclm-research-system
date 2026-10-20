"""
Phase 8 — Relationship Management Engine
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles & Architecture:
1. GENUINE DATA-DRIVEN RELATIONSHIP IDENTIFICATION:
   Builds an evidence-backed Memory Relationship layer connecting the consolidated memories
   produced by Phase 6 and evaluated by Phase 7.
   Connects memories only when measurable evidence satisfies configurable thresholds.
   No artificial connections, no demo data, no fake edges.
2. MULTI-MODAL EVIDENCE SIGNALS:
   Evaluates 5 distinct relationship signals across candidate pairs (u, v):
   - Semantic Similarity: Token overlap and topic keyword alignment
   - Context Overlap: Shared window titles and navigation contexts
   - Application Continuity: Shared primary or secondary desktop environments
   - Domain Continuity: Shared browser research domains
   - Temporal Proximity: Continuous or proximal task execution intervals
3. MULTIPLE MEANINGFUL RELATIONSHIP TYPES:
   Classifies edges into distinct, measurable relationship types:
   - WORKFLOW_CONTINUITY: Co-occurring window navigation and file workflow
   - SEMANTIC_SIMILARITY: Deep conceptual and task topic overlap
   - SHARED_DOMAIN: Shared web research domains
   - TEMPORAL_ASSOCIATION: High proximity temporal workflow sessions
   - SHARED_APPLICATION: Common desktop application environment
   - CROSS_APPLICATION_TASK: Multi-tool task bridging different applications
   - HYBRID_COMPOSITE: Multi-modal convergence across 3+ dimensions
4. CANONICAL DETERMINISM & ZERO REDUNDANCY:
   Strictly rejects self-loops (u == v).
   Enforces canonical ordering (min(u, v), max(u, v)) to avoid duplicate undirected edges.
   Relationship IDs are deterministic SHA-256 hashes of canonical endpoints and type.
5. NETWORKX GRAPH TOPOLOGY & HONEST METRICS:
   Builds a NetworkX graph maintaining all consolidated memory nodes and accepted edges.
   Preserves isolated memories as degree-0 nodes without forced connections.
   Calculates graph-level metrics: density, average degree, connected components,
   relationship type distribution, and average strength.
6. COMPLETE PROVENANCE & RETENTION PRESERVATION:
   Preserves Phase 7 retention states (RETAINED, DECAYING, ARCHIVED, FORGETTABLE).
   Even archived or forgettable nodes remain fully connected and traceable for research inspection.
   Zero permanent deletion of research data.
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import RelationshipConfig, DMCLMGlobalConfig, DEFAULT_CONFIG
    from src.models import MemoryRelationship, RelationshipGraphStats, Phase7RetainedMemory
    from src.dmclm_db import (
        init_dmclm_database,
        get_dmclm_connection,
        insert_or_replace_relationships,
        insert_or_replace_graph_stats,
        query_relationships,
        get_relationship_graph_stats,
    )
except ImportError:
    from dmclm_config import RelationshipConfig, DMCLMGlobalConfig, DEFAULT_CONFIG
    from models import MemoryRelationship, RelationshipGraphStats, Phase7RetainedMemory
    from dmclm_db import (
        init_dmclm_database,
        get_dmclm_connection,
        insert_or_replace_relationships,
        insert_or_replace_graph_stats,
        query_relationships,
        get_relationship_graph_stats,
    )

DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
CONSOLIDATED_JSON_PATH = DATA_DIR / "consolidated_memories.json"
RETENTION_JSON_PATH = DATA_DIR / "retention_evaluated_memories.json"
RELATIONSHIPS_JSON_PATH = DATA_DIR / "memory_relationships.json"
RELATIONSHIP_GRAPH_JSON_PATH = DATA_DIR / "relationship_graph.json"
RELATIONSHIP_METRICS_PATH = RESULTS_DIR / "relationship_metrics.json"
RELATIONSHIP_REPORT_PATH = RESULTS_DIR / "relationship_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase8.RelationshipManagement")

# Standard stop words for text normalization
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
        s = str(ts_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        try:
            s = str(ts_str).split("+")[0].split("Z")[0]
            dt = datetime.strptime(s[:26], "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return 0.0


class Phase8RelationshipManager:
    """
    Core implementation of Phase 8 — Relationship Management.
    Constructs an evidence-backed relationship graph from Phase 7 evaluated memories.
    """

    def __init__(self, config: Optional[RelationshipConfig] = None):
        self.config = config or DEFAULT_CONFIG.relationship
        self.config.validate()

    def calculate_pairwise_signals(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        m1_tokens: Optional[Set[str]] = None,
        m2_tokens: Optional[Set[str]] = None,
        m1_ctx_tokens: Optional[Set[str]] = None,
        m2_ctx_tokens: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates normalized relationship evidence signals between two distinct memories m1 and m2.
        Returns a dictionary of individual signal scores in [0.0, 1.0] and detailed evidence metrics.
        """
        # 1. Semantic Similarity
        if m1_tokens is None:
            t1 = f"{m1.get('summary', '')} {m1.get('topic', '')} {' '.join(m1.get('unique_windows_accessed') or [])}"
            m1_tokens = tokenize_text(t1)
        if m2_tokens is None:
            t2 = f"{m2.get('summary', '')} {m2.get('topic', '')} {' '.join(m2.get('unique_windows_accessed') or [])}"
            m2_tokens = tokenize_text(t2)

        if m1_tokens and m2_tokens:
            sem_sim = len(m1_tokens & m2_tokens) / len(m1_tokens | m2_tokens)
            shared_keywords = sorted(list(m1_tokens & m2_tokens))[:10]
        else:
            sem_sim = 0.0
            shared_keywords = []

        # 2. Context / Window Title Overlap
        if m1_ctx_tokens is None:
            c1 = " ".join(m1.get("unique_windows_accessed") or m1.get("contexts") or [])
            m1_ctx_tokens = tokenize_text(c1)
        if m2_ctx_tokens is None:
            c2 = " ".join(m2.get("unique_windows_accessed") or m2.get("contexts") or [])
            m2_ctx_tokens = tokenize_text(c2)

        if m1_ctx_tokens and m2_ctx_tokens:
            ctx_sim = len(m1_ctx_tokens & m2_ctx_tokens) / len(m1_ctx_tokens | m2_ctx_tokens)
            shared_context_words = sorted(list(m1_ctx_tokens & m2_ctx_tokens))[:8]
        else:
            ctx_sim = 0.0
            shared_context_words = []

        # 3. Application Match
        app1 = (m1.get("primary_application") or m1.get("application") or "").strip().lower()
        app2 = (m2.get("primary_application") or m2.get("application") or "").strip().lower()
        apps1 = {a.strip().lower() for a in (m1.get("applications") or []) if a}
        apps2 = {a.strip().lower() for a in (m2.get("applications") or []) if a}
        if app1:
            apps1.add(app1)
        if app2:
            apps2.add(app2)

        if app1 and app2 and app1 == app2:
            app_match = 1.0
        elif apps1 and apps2 and (apps1 & apps2):
            app_match = len(apps1 & apps2) / len(apps1 | apps2)
        else:
            app_match = 0.0

        shared_apps = sorted(list(apps1 & apps2))

        # 4. Browser Domain Match
        doms1 = {d.strip().lower() for d in (m1.get("domains") or []) if d}
        doms2 = {d.strip().lower() for d in (m2.get("domains") or []) if d}

        if doms1 and doms2:
            dom_match = len(doms1 & doms2) / len(doms1 | doms2)
            shared_doms = sorted(list(doms1 & doms2))
        elif not doms1 and not doms2 and app_match > 0.0:
            # Both are non-browser desktop applications sharing environment
            dom_match = 1.0 if app_match == 1.0 else 0.5
            shared_doms = []
        else:
            dom_match = 0.0
            shared_doms = []

        # 5. Temporal Proximity
        t1_start = parse_timestamp_iso(m1.get("start_time"))
        t1_end = parse_timestamp_iso(m1.get("end_time")) or t1_start
        t2_start = parse_timestamp_iso(m2.get("start_time"))
        t2_end = parse_timestamp_iso(m2.get("end_time")) or t2_start

        if t1_start > 0 and t2_start > 0:
            if t1_end < t2_start:
                delta_sec = t2_start - t1_end
            elif t2_end < t1_start:
                delta_sec = t1_start - t2_end
            else:
                delta_sec = 0.0  # Overlapping execution intervals

            tau = float(self.config.temporal_proximity_tau_sec)
            time_prox = math.exp(-delta_sec / tau)
        else:
            delta_sec = 0.0
            time_prox = 0.0

        return {
            "semantic_similarity": round(max(0.0, min(1.0, sem_sim)), 4),
            "context_similarity": round(max(0.0, min(1.0, ctx_sim)), 4),
            "application_match": round(max(0.0, min(1.0, app_match)), 4),
            "domain_match": round(max(0.0, min(1.0, dom_match)), 4),
            "temporal_proximity": round(max(0.0, min(1.0, time_prox)), 4),
            "temporal_delta_seconds": round(delta_sec, 2),
            "temporal_delta_hours": round(delta_sec / 3600.0, 2),
            "shared_keywords": shared_keywords,
            "shared_context_words": shared_context_words,
            "shared_applications": shared_apps,
            "shared_domains": shared_doms,
        }

    def compute_composite_strength(self, signals: Dict[str, Any]) -> float:
        """Computes weighted composite relationship strength in [0.0, 1.0]."""
        c = self.config
        strength = (
            c.weight_semantic * signals["semantic_similarity"]
            + c.weight_context * signals["context_similarity"]
            + c.weight_application * signals["application_match"]
            + c.weight_domain * signals["domain_match"]
            + c.weight_temporal * signals["temporal_proximity"]
        )
        return round(max(0.0, min(1.0, float(strength))), 4)

    def classify_relationship_type(
        self,
        signals: Dict[str, Any],
        strength: float
    ) -> Tuple[str, List[str], List[str]]:
        """
        Classifies relationship into a meaningful type based on qualifying evidence signals.
        Returns:
            (primary_relationship_type, qualifying_reasons, secondary_matching_types)
        """
        c = self.config
        reasons: List[str] = []
        qualifying_types: List[str] = []

        sem = signals["semantic_similarity"]
        ctx = signals["context_similarity"]
        app = signals["application_match"]
        dom = signals["domain_match"]
        time_prox = signals["temporal_proximity"]
        shared_doms = signals["shared_domains"]
        shared_apps = signals["shared_applications"]

        # 1. Check for WORKFLOW_CONTINUITY
        if ctx >= c.context_threshold and app >= 0.50:
            qualifying_types.append("WORKFLOW_CONTINUITY")
            reasons.append(f"Window context workflow overlap ({ctx:.2f} >= {c.context_threshold:.2f})")

        # 2. Check for SHARED_DOMAIN
        if shared_doms and dom >= c.domain_threshold:
            qualifying_types.append("SHARED_DOMAIN")
            reasons.append(f"Shared browser research domain ({', '.join(shared_doms[:3])})")

        # 3. Check for SEMANTIC_SIMILARITY
        if sem >= c.semantic_threshold:
            qualifying_types.append("SEMANTIC_SIMILARITY")
            reasons.append(f"Strong semantic token overlap ({sem:.2f} >= {c.semantic_threshold:.2f})")

        # 4. Check for TEMPORAL_ASSOCIATION
        if time_prox >= 0.50 and signals["temporal_delta_hours"] <= 24.0 and (app > 0 or sem >= 0.15):
            qualifying_types.append("TEMPORAL_ASSOCIATION")
            reasons.append(f"Close temporal proximity ({signals['temporal_delta_hours']:.1f}h gap, score {time_prox:.2f})")

        # 5. Check for CROSS_APPLICATION_TASK
        if app < 0.50 and (dom >= 0.25 or sem >= 0.25):
            qualifying_types.append("CROSS_APPLICATION_TASK")
            reasons.append(f"Cross-application synergy bridging tools via domain/task context")

        # 6. Check for SHARED_APPLICATION
        if app >= 0.80 and (ctx >= 0.10 or sem >= 0.15 or time_prox >= 0.20):
            qualifying_types.append("SHARED_APPLICATION")
            reasons.append(f"Common primary application environment ({', '.join(shared_apps)})")

        # Determine primary relationship type
        if len(qualifying_types) >= 3 and strength >= 0.50:
            primary_type = "HYBRID_COMPOSITE"
            reasons.insert(0, f"Multi-modal convergence across {len(qualifying_types)} relationship dimensions")
        elif qualifying_types:
            # Priority order for primary type: domain/workflow specific over generic app match
            type_priority = [
                "WORKFLOW_CONTINUITY",
                "SHARED_DOMAIN",
                "CROSS_APPLICATION_TASK",
                "SEMANTIC_SIMILARITY",
                "TEMPORAL_ASSOCIATION",
                "SHARED_APPLICATION",
            ]
            primary_type = next((t for t in type_priority if t in qualifying_types), qualifying_types[0])
        else:
            primary_type = "GENERAL_ASSOCIATION"
            reasons.append(f"Composite multi-signal association (strength {strength:.3f})")

        return primary_type, reasons, qualifying_types

    def evaluate_pair(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        m1_tokens: Optional[Set[str]] = None,
        m2_tokens: Optional[Set[str]] = None,
        m1_ctx: Optional[Set[str]] = None,
        m2_ctx: Optional[Set[str]] = None,
    ) -> Optional[MemoryRelationship]:
        """
        Evaluates two distinct memories and returns a MemoryRelationship if criteria are met.
        Strictly rejects self-loops and produces canonical deterministic ID.
        """
        id1 = str(m1.get("consolidated_memory_id") or m1.get("memory_id"))
        id2 = str(m2.get("consolidated_memory_id") or m2.get("memory_id"))

        if id1 == id2:
            return None  # Strictly reject self-relationships

        # Canonical ordering for undirected relationship
        canon_source = min(id1, id2)
        canon_target = max(id1, id2)

        signals = self.calculate_pairwise_signals(m1, m2, m1_tokens, m2_tokens, m1_ctx, m2_ctx)
        strength = self.compute_composite_strength(signals)

        if strength < self.config.min_strength_threshold:
            return None  # Below acceptance threshold

        primary_type, reasons, secondary_types = self.classify_relationship_type(signals, strength)

        # Deterministic relationship ID
        raw_sig = f"{canon_source}_{canon_target}_{primary_type}".encode("utf-8")
        rel_id = f"rel_{hashlib.sha256(raw_sig).hexdigest()[:16]}"

        evidence_payload = {
            "signals": {
                "semantic_similarity": signals["semantic_similarity"],
                "context_similarity": signals["context_similarity"],
                "application_match": signals["application_match"],
                "domain_match": signals["domain_match"],
                "temporal_proximity": signals["temporal_proximity"],
            },
            "temporal_delta_hours": signals["temporal_delta_hours"],
            "shared_keywords": signals["shared_keywords"],
            "shared_context_words": signals["shared_context_words"],
            "shared_applications": signals["shared_applications"],
            "shared_domains": signals["shared_domains"],
            "qualifying_reasons": reasons,
            "secondary_types": secondary_types,
            "threshold_used": self.config.min_strength_threshold,
        }

        # Source / episode mass metrics
        src_cnt_1 = int(m1.get("source_count") or len(m1.get("source_memory_ids") or []) or 1)
        src_cnt_2 = int(m2.get("source_count") or len(m2.get("source_memory_ids") or []) or 1)
        src_ep_1 = len(m1.get("source_episode_ids") or [])
        src_ep_2 = len(m2.get("source_episode_ids") or [])

        now_iso = datetime.now(timezone.utc).isoformat()

        return MemoryRelationship(
            relationship_id=rel_id,
            source_memory_id=canon_source,
            target_memory_id=canon_target,
            relationship_type=primary_type,
            relationship_strength=strength,
            evidence=evidence_payload,
            created_at=now_iso,
            source_memory_source_count=src_cnt_1 if canon_source == id1 else src_cnt_2,
            target_memory_source_count=src_cnt_2 if canon_source == id1 else src_cnt_1,
            source_episode_ids_count=src_ep_1 if canon_source == id1 else src_ep_2,
            target_episode_ids_count=src_ep_2 if canon_source == id1 else src_ep_1,
        )

    def build_relationship_graph(
        self,
        memories: List[Dict[str, Any]],
        relationships: List[MemoryRelationship]
    ) -> Tuple[nx.Graph, RelationshipGraphStats]:
        """
        Builds a NetworkX graph containing memory nodes and accepted relationship edges.
        Calculates exact graph-level topological metrics.
        """
        G = nx.Graph()

        # Add all memory nodes (including isolated nodes)
        for m in memories:
            mid = str(m.get("consolidated_memory_id") or m.get("memory_id"))
            G.add_node(
                mid,
                primary_application=str(m.get("primary_application") or m.get("application") or ""),
                retention_state=str(m.get("retention_state") or "RETAINED"),
                final_retention_strength=float(m.get("final_retention_strength") or m.get("retention_score") or 0.5),
                baseline_score=float(m.get("baseline_score") or 0.5),
                source_count=int(m.get("source_count") or len(m.get("source_memory_ids") or []) or 1),
                topic=str(m.get("topic") or ""),
                summary=str(m.get("summary") or m.get("semantic_representation") or ""),
                is_protected=bool(m.get("is_protected", False)),
                forgetting_eligible=bool(m.get("forgetting_eligible", False)),
            )

        # Add all relationship edges
        for r in relationships:
            G.add_edge(
                r.source_memory_id,
                r.target_memory_id,
                relationship_id=r.relationship_id,
                relationship_type=r.relationship_type,
                weight=r.relationship_strength,
                relationship_strength=r.relationship_strength,
                evidence=r.evidence,
            )

        # Graph Metrics Calculation
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()

        if total_nodes > 1:
            density = nx.density(G)
        else:
            density = 0.0

        if total_nodes > 0:
            avg_deg = sum(d for _, d in G.degree()) / float(total_nodes)
            isolated_count = sum(1 for _, d in G.degree() if d == 0)
            conn_components = nx.number_connected_components(G)
        else:
            avg_deg = 0.0
            isolated_count = 0
            conn_components = 0

        strengths = [r.relationship_strength for r in relationships]
        avg_strength = round(sum(strengths) / len(strengths), 4) if strengths else 0.0

        type_dist = Counter(r.relationship_type for r in relationships)

        # Sort top relationships by strength descending
        top_rels = sorted(
            [r.to_dict() for r in relationships],
            key=lambda x: -x["relationship_strength"]
        )[:10]

        now_iso = datetime.now(timezone.utc).isoformat()

        stats = RelationshipGraphStats(
            total_nodes=total_nodes,
            total_edges=total_edges,
            density=round(density, 4),
            average_degree=round(avg_deg, 2),
            average_strength=avg_strength,
            isolated_nodes_count=isolated_count,
            connected_components_count=conn_components,
            type_distribution=dict(type_dist),
            top_relationships=top_rels,
            created_at=now_iso,
        )

        return G, stats

    def process(
        self,
        memories: List[Dict[str, Any]]
    ) -> Tuple[List[MemoryRelationship], nx.Graph, RelationshipGraphStats]:
        """
        Executes the Phase 8 relationship extraction and graph construction pipeline.
        """
        start_time = time.time()
        logger.info(f"[PHASE 8] Processing {len(memories):,} memory nodes for relationship discovery...")

        if not memories:
            logger.info("[PHASE 8] Received empty memory list. Returning empty graph.")
            empty_graph = nx.Graph()
            empty_stats = RelationshipGraphStats(
                total_nodes=0,
                total_edges=0,
                density=0.0,
                average_degree=0.0,
                average_strength=0.0,
                isolated_nodes_count=0,
                connected_components_count=0,
                type_distribution={},
                top_relationships=[],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            return [], empty_graph, empty_stats

        # Precompute text tokens for all memories deterministically
        tokens_map: Dict[str, Set[str]] = {}
        ctx_tokens_map: Dict[str, Set[str]] = {}

        for m in memories:
            mid = str(m.get("consolidated_memory_id") or m.get("memory_id"))
            text = f"{m.get('summary', '')} {m.get('topic', '')} {' '.join(m.get('unique_windows_accessed') or [])}"
            tokens_map[mid] = tokenize_text(text)
            ctx_text = " ".join(m.get("unique_windows_accessed") or m.get("contexts") or [])
            ctx_tokens_map[mid] = tokenize_text(ctx_text)

        # Evaluate all pairs (u, v) with u < v (strictly preventing self-loops and duplicates)
        relationships: List[MemoryRelationship] = []
        n = len(memories)
        candidate_pairs_evaluated = 0

        # Sort memories deterministically by ID
        sorted_memories = sorted(
            memories,
            key=lambda m: str(m.get("consolidated_memory_id") or m.get("memory_id"))
        )

        for i in range(n):
            m1 = sorted_memories[i]
            id1 = str(m1.get("consolidated_memory_id") or m1.get("memory_id"))

            for j in range(i + 1, n):
                m2 = sorted_memories[j]
                id2 = str(m2.get("consolidated_memory_id") or m2.get("memory_id"))
                candidate_pairs_evaluated += 1

                rel = self.evaluate_pair(
                    m1,
                    m2,
                    tokens_map[id1],
                    tokens_map[id2],
                    ctx_tokens_map[id1],
                    ctx_tokens_map[id2],
                )

                if rel is not None:
                    relationships.append(rel)

        # Build NetworkX graph and compute statistics
        G, stats = self.build_relationship_graph(sorted_memories, relationships)

        elapsed = round(time.time() - start_time, 4)
        logger.info(
            f"[PHASE 8] Relationship discovery completed in {elapsed}s: "
            f"Evaluated {candidate_pairs_evaluated:,} candidate pairs -> Accepted {len(relationships):,} edges. "
            f"Nodes={stats.total_nodes}, Density={stats.density:.4f}, Avg Degree={stats.average_degree:.2f}, "
            f"Isolated={stats.isolated_nodes_count}, Connected Components={stats.connected_components_count}. "
            f"Types={stats.type_distribution}"
        )

        return relationships, G, stats

    def run_pipeline(
        self,
        input_path: Optional[Path] = None,
        save_to_db: bool = True,
        save_to_json: bool = True,
    ) -> Tuple[List[MemoryRelationship], nx.Graph, RelationshipGraphStats]:
        """
        Runs the full Phase 8 pipeline from persistent Phase 7 / Phase 6 storage and saves artifacts.
        """
        # Prefer Phase 7 retention evaluated memories, fallback to Phase 6 consolidated memories
        p = input_path
        if p is None:
            if RETENTION_JSON_PATH.exists():
                p = RETENTION_JSON_PATH
            elif CONSOLIDATED_JSON_PATH.exists():
                p = CONSOLIDATED_JSON_PATH
            else:
                raise FileNotFoundError(f"Neither {RETENTION_JSON_PATH} nor {CONSOLIDATED_JSON_PATH} found.")

        with open(p, "r", encoding="utf-8") as f:
            memories = json.load(f)

        logger.info(f"[PHASE 8] Loaded {len(memories):,} memory nodes from {p}")

        # If loading Phase 7 memories, enrich with domains/contexts from Phase 6 if available
        if CONSOLIDATED_JSON_PATH.exists() and p != CONSOLIDATED_JSON_PATH:
            try:
                with open(CONSOLIDATED_JSON_PATH, "r", encoding="utf-8") as f:
                    c_nodes = json.load(f)
                    c_lookup = {c["consolidated_memory_id"]: c for c in c_nodes}
                    for m in memories:
                        cid = m.get("consolidated_memory_id")
                        if cid in c_lookup:
                            c = c_lookup[cid]
                            if not m.get("domains") and c.get("domains"):
                                m["domains"] = c["domains"]
                            if not m.get("applications") and c.get("applications"):
                                m["applications"] = c["applications"]
                            if not m.get("contexts") and c.get("contexts"):
                                m["contexts"] = c["contexts"]
            except Exception as e:
                logger.warning(f"[PHASE 8] Could not enrich with Phase 6 contexts: {e}")

        relationships, G, stats = self.process(memories)

        if save_to_json:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            rel_dicts = [r.to_dict() for r in relationships]

            # 1. Save list of relationships
            with open(RELATIONSHIPS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(rel_dicts, f, indent=2)
            logger.info(f"[PHASE 8] Saved {len(rel_dicts):,} relationships to {RELATIONSHIPS_JSON_PATH}")

            # 2. Save graph representation (node-link format)
            graph_data = nx.node_link_data(G)
            with open(RELATIONSHIP_GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2)
            logger.info(f"[PHASE 8] Saved graph topology to {RELATIONSHIP_GRAPH_JSON_PATH}")

            # 3. Save graph metrics
            with open(RELATIONSHIP_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(stats.to_dict(), f, indent=2)
            logger.info(f"[PHASE 8] Saved graph metrics to {RELATIONSHIP_METRICS_PATH}")

            # 4. Save detailed audit report
            report_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "configuration": self.config.to_dict(),
                "graph_statistics": stats.to_dict(),
                "sample_relationships": rel_dicts[:10],
            }
            with open(RELATIONSHIP_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=2)
            logger.info(f"[PHASE 8] Saved detailed report to {RELATIONSHIP_REPORT_PATH}")

        if save_to_db:
            init_dmclm_database()
            insert_or_replace_relationships([r.to_dict() for r in relationships])
            insert_or_replace_graph_stats(stats.to_dict(), stat_id="latest")
            logger.info(f"[PHASE 8] Persisted {len(relationships):,} relationships and graph stats to SQLite.")

        return relationships, G, stats


def trace_memory_relationships(
    memory_id: str,
    relationships: Optional[List[MemoryRelationship]] = None,
    graph: Optional[nx.Graph] = None
) -> Dict[str, Any]:
    """
    Provenance query function for Phase 8.
    Returns all relationships, connected neighbors, evidence breakdowns,
    and underlying source candidate masses connected to a specific memory node.
    """
    # Load relationships if not provided
    rels_list: List[Dict[str, Any]] = []
    if relationships:
        rels_list = [r.to_dict() for r in relationships]
    elif RELATIONSHIPS_JSON_PATH.exists():
        with open(RELATIONSHIPS_JSON_PATH, "r", encoding="utf-8") as f:
            rels_list = json.load(f)

    incident_rels = [
        r for r in rels_list
        if r.get("source_memory_id") == memory_id or r.get("target_memory_id") == memory_id
    ]

    neighbors = []
    for r in incident_rels:
        partner_id = r["target_memory_id"] if r["source_memory_id"] == memory_id else r["source_memory_id"]
        neighbors.append({
            "neighbor_id": partner_id,
            "relationship_id": r["relationship_id"],
            "relationship_type": r["relationship_type"],
            "relationship_strength": r["relationship_strength"],
            "evidence": r["evidence"],
        })

    neighbors.sort(key=lambda x: -x["relationship_strength"])

    return {
        "memory_id": memory_id,
        "degree": len(incident_rels),
        "incident_relationships_count": len(incident_rels),
        "relationships": incident_rels,
        "connected_neighbors": neighbors,
    }


if __name__ == "__main__":
    manager = Phase8RelationshipManager()
    rels, G, stats = manager.run_pipeline()
    print("\n[PHASE 8 RELATIONSHIP MANAGEMENT SUMMARY]")
    print(f"Total Memory Nodes:           {stats.total_nodes:,}")
    print(f"Total Relationship Edges:     {stats.total_edges:,}")
    print(f"Network Density:              {stats.density:.4f}")
    print(f"Average Node Degree:          {stats.average_degree:.2f}")
    print(f"Average Relationship Strength:{stats.average_strength:.4f}")
    print(f"Isolated Memories:            {stats.isolated_nodes_count:,}")
    print(f"Connected Components:         {stats.connected_components_count:,}")
    print(f"Relationship Type Breakdown:")
    for k, v in stats.type_distribution.items():
        print(f"  - {k:<25}: {v:,}")

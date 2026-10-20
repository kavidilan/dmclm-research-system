"""
DMCLM Baseline Decision Engine & Genuine Memory Consolidation
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model

Responsibilities:
1. Calculates transparent experimental baseline DMCLM importance score:
   Importance = 0.20 * Recency + 0.20 * Frequency + 0.25 * Retrieval History
              + 0.20 * Semantic Relevance + 0.15 * Relationship Strength
2. Classifies each memory record into 4 cognitive lifecycle tiers:
   - RETAIN (>= 0.70)
   - COMPRESS (0.40 - 0.69)
   - ARCHIVE (0.20 - 0.39)
   - FORGET (< 0.20)
3. Genuine Memory Compression: Consolidates repeated episodic clusters into synthesized
   abstract Long-Term Memory (LTM) nodes with true compression ratio and source lineage.
4. Genuine Memory Forgetting: Excludes forgotten traces from active retrieval while
   preserving metadata in research records; NEVER modifies raw Collector data.
5. Generates the real labeled training dataset for future Machine Learning evaluation.
"""

import json
import logging
import math
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dmclm_config import (
        BaselineImportanceWeights,
        LifecycleThresholds,
        DMCLMGlobalConfig,
        DEFAULT_CONFIG,
    )
    from dmclm_db import (
        init_dmclm_database,
        insert_or_replace_memory_records,
        get_dmclm_connection,
    )
except ImportError:
    from src.dmclm_config import (
        BaselineImportanceWeights,
        LifecycleThresholds,
        DMCLMGlobalConfig,
        DEFAULT_CONFIG,
    )
    from src.dmclm_db import (
        init_dmclm_database,
        insert_or_replace_memory_records,
        get_dmclm_connection,
    )

logger = logging.getLogger("DMCLM.Baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class BaselineDecisionEngine:
    def __init__(self, config: Optional[DMCLMGlobalConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.weights = self.config.weights
        self.thresholds = self.config.thresholds

    def compute_importance_scores(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Computes the experimental baseline importance score for each memory:
        Importance = w_r * R + w_f * F + w_h * H + w_s * S + w_rel * Rel
        Stores individual feature values and the final score.
        """
        w_r = self.weights.recency
        w_f = self.weights.frequency
        w_h = self.weights.retrieval_history
        w_s = self.weights.semantic_relevance
        w_rel = self.weights.relationship_strength

        for r in records:
            r_val = float(r.get("recency_score") if r.get("recency_score") is not None else (r.get("recency") or 0.0))
            f_val = float(r.get("frequency_score") if r.get("frequency_score") is not None else (r.get("frequency") or 0.0))
            h_val = float(r.get("retrieval_history_score") if r.get("retrieval_history_score") is not None else (r.get("retrieval_history") or 0.0))
            s_val = float(r.get("semantic_relevance") or 0.0)
            rel_val = float(r.get("relationship_strength") or 0.0)

            raw_imp = (w_r * r_val) + (w_f * f_val) + (w_h * h_val) + (w_s * s_val) + (w_rel * rel_val)
            importance = round(float(max(0.0, min(1.0, raw_imp))), 4)

            # Assign lifecycle decision using configurable thresholds
            lifecycle = self.thresholds.classify(importance)

            r["importance_score"] = importance
            r["lifecycle_state"] = lifecycle
            # Frontend compatibility keys
            r["lifecycle_prediction"] = lifecycle
            r["lifecycle_confidence"] = 1.0  # Deterministic baseline formula
            r["lifecycle_explanation"] = (
                f"DMCLM baseline score {importance:.3f} = "
                f"({w_r}*R={r_val:.2f}) + ({w_f}*F={f_val:.2f}) + ({w_h}*H={h_val:.2f}) + "
                f"({w_s}*S={s_val:.2f}) + ({w_rel}*Rel={rel_val:.2f}) -> {lifecycle}"
            )

        return records

    def consolidate_compressed_memories(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Genuine Memory Consolidation:
        Finds all records categorized under COMPRESS (and related repeated activities),
        clusters them by application and session/domain, and synthesizes higher-level
        abstract Long-Term Memory (LTM) nodes.
        Calculates exact original memory count, consolidated memory count, and true compression ratio.
        """
        compress_items = [r for r in records if r.get("lifecycle_state") == "COMPRESS"]
        original_count = len(compress_items)

        if original_count == 0:
            return [], {
                "original_episodic_records": 0,
                "consolidated_ltm_nodes": 0,
                "memory_reduction_count": 0,
                "compression_ratio_pct": 0.0,
            }

        # Cluster by application and primary session
        clusters: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for m in compress_items:
            app = (m.get("application") or "Desktop").strip()
            # Group by session or domain
            dom = m.get("domain")
            group_key = dom if dom else (m.get("session_id") or (m.get("session_ids", ["main"])[0] if m.get("session_ids") else "session"))
            clusters[(app, str(group_key))].append(m)

        ltm_nodes = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for (app, group_ref), items in clusters.items():
            # Sort chronologically
            items.sort(key=lambda x: x.get("first_seen") or "")

            source_ids = [str(x["memory_id"]) for x in items]
            first_seen_vals = [x.get("first_seen") for x in items if x.get("first_seen")]
            last_seen_vals = [x.get("last_seen") for x in items if x.get("last_seen")]
            first_observed = first_seen_vals[0] if first_seen_vals else now_iso
            last_reinforced = last_seen_vals[-1] if last_seen_vals else now_iso

            total_duration = sum(int(x.get("total_duration") or x.get("duration_ms") or 0) for x in items)
            total_interactions = sum(int(x.get("interaction_count") or 0) for x in items)
            total_occurrences = sum(int(x.get("occurrence_count") or 1) for x in items)

            # Unique salient windows/titles
            unique_windows = list(dict.fromkeys([
                (x.get("content") or "").split(" | ")[1] if " | " in (x.get("content") or "") else (x.get("content") or "")
                for x in items if x.get("content")
            ]))[:5]

            # Average importance and stability
            avg_importance = sum(x.get("importance_score", 0.5) for x in items) / len(items)
            stability_score = round(min(1.0, 0.4 + (0.6 * avg_importance)), 4)

            windows_str = ", ".join(unique_windows) if unique_windows else "general desktop tasks"
            abstract_summary = (
                f"Consolidated knowledge across {len(items)} episodic traces ({total_occurrences} occurrences) in {app} "
                f"[{windows_str}]. Total duration: {round(total_duration / 1000, 1)}s with {total_interactions} user interactions."
            )

            ltm_id = f"ltm_{uuid.uuid4().hex[:10]}"
            ltm_node = {
                "ltm_id": ltm_id,
                "topic": f"{app}: {unique_windows[0] if unique_windows else 'Research Activity'}",
                "application": app,
                "session_id": str(group_ref),
                "summary": abstract_summary,
                "unique_windows_accessed": unique_windows,
                "source_count": len(items),
                "source_memory_ids": source_ids,
                "first_observed": first_observed,
                "last_reinforced": last_reinforced,
                "total_duration_ms": total_duration,
                "total_interactions": total_interactions,
                "stability_score": stability_score,
                "lifecycle_origin": "COMPRESS",
                "created_at": now_iso,
            }
            ltm_nodes.append(ltm_node)

        consolidated_count = len(ltm_nodes)
        reduction_count = original_count - consolidated_count
        compression_ratio_pct = round((reduction_count / max(1, original_count)) * 100, 2)

        metrics = {
            "original_episodic_records": original_count,
            "consolidated_ltm_nodes": consolidated_count,
            "memory_reduction_count": reduction_count,
            "compression_ratio_pct": compression_ratio_pct,
        }

        logger.info(
            f"Genuine Memory Compression completed: {original_count:,} episodic traces -> "
            f"{consolidated_count:,} durable LTM nodes ({compression_ratio_pct}% compression ratio)."
        )

        return ltm_nodes, metrics

    def partition_lifecycle_stores(
        self, records: List[Dict[str, Any]], ltm_nodes: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partitions derived memories into durable lifecycle tiers:
        - RETAIN: high-utility active working knowledge
        - COMPRESS: consolidated long-term memory nodes
        - ARCHIVE: secondary historical context
        - FORGET: logically pruned from active retrieval (raw collector data untouched!)
        """
        retained = [r for r in records if r.get("lifecycle_state") == "RETAIN"]
        compressed_source = [r for r in records if r.get("lifecycle_state") == "COMPRESS"]
        archived = [r for r in records if r.get("lifecycle_state") == "ARCHIVE"]
        forgotten = [r for r in records if r.get("lifecycle_state") == "FORGET"]

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        with open(DATA_DIR / "memory_records.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        with open(DATA_DIR / "long_term_memories.json", "w", encoding="utf-8") as f:
            json.dump(ltm_nodes, f, indent=2, ensure_ascii=False)

        with open(DATA_DIR / "retained_memories.json", "w", encoding="utf-8") as f:
            json.dump(retained, f, indent=2, ensure_ascii=False)

        with open(DATA_DIR / "archive_memories.json", "w", encoding="utf-8") as f:
            json.dump(archived, f, indent=2, ensure_ascii=False)

        with open(DATA_DIR / "forgotten_memories.json", "w", encoding="utf-8") as f:
            json.dump(forgotten, f, indent=2, ensure_ascii=False)

        return {
            "all": records,
            "retained": retained,
            "compressed_source": compressed_source,
            "ltm_nodes": ltm_nodes,
            "archived": archived,
            "forgotten": forgotten,
        }

    def generate_training_dataset(
        self, records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Prepares the genuine research training dataset from processed records and behavior-derived labels.
        Does NOT train the ML model yet (reserved for the future ML phase).
        Contains:
        - recency_score
        - frequency_score
        - retrieval_history_score
        - semantic_relevance
        - relationship_strength
        - importance_score
        - lifecycle_label (RETAIN, COMPRESS, ARCHIVE, FORGET)
        - metadata for lineage and evaluation
        """
        rows = []
        for r in records:
            rows.append({
                "memory_id": r.get("memory_id"),
                "recency_score": r.get("recency_score", 0.0),
                "frequency_score": r.get("frequency_score", 0.0),
                "retrieval_history_score": r.get("retrieval_history_score", 0.0),
                "semantic_relevance": r.get("semantic_relevance", 0.0),
                "relationship_strength": r.get("relationship_strength", 0.0),
                "importance_score": r.get("importance_score", 0.0),
                "lifecycle_label": r.get("lifecycle_state", "RETAIN"),
                "source_type": r.get("source_type"),
                "application": r.get("application"),
                "occurrence_count": r.get("occurrence_count", 1),
                "total_duration": r.get("total_duration", 0),
                "interaction_count": r.get("interaction_count", 0),
                "label_origin": "behavior_derived_baseline",
            })

        df = pd.DataFrame(rows)
        csv_path = DATA_DIR / "training_dataset.csv"
        df.to_csv(csv_path, index=False)

        # Also save features-only CSV
        feat_df = df[[
            "memory_id", "recency_score", "frequency_score",
            "retrieval_history_score", "semantic_relevance",
            "relationship_strength", "importance_score", "lifecycle_label"
        ]]
        feat_df.to_csv(DATA_DIR / "dmclm_features.csv", index=False)

        logger.info(f"Saved real training dataset with {len(df):,} records to {csv_path}.")
        return df

    def save_derived_database_and_metrics(
        self,
        records: List[Dict[str, Any]],
        ltm_nodes: List[Dict[str, Any]],
        comp_metrics: Dict[str, Any],
    ) -> None:
        """
        Stores all normalized memories and LTM nodes into `data/dmclm.db` SQLite database
        and saves consolidation metrics into `results/consolidation_metrics.json`.
        """
        init_dmclm_database()
        insert_or_replace_memory_records(records)

        # Insert LTM nodes into SQLite
        conn = get_dmclm_connection()
        try:
            cur = conn.cursor()
            for node in ltm_nodes:
                cur.execute("""
                    INSERT OR REPLACE INTO long_term_memories (
                        ltm_id, topic, application, session_id, summary,
                        unique_windows_accessed, source_count, source_memory_ids,
                        first_observed, last_reinforced, total_duration_ms,
                        total_interactions, stability_score, lifecycle_origin, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node["ltm_id"],
                    node["topic"],
                    node["application"],
                    node["session_id"],
                    node["summary"],
                    json.dumps(node.get("unique_windows_accessed", [])),
                    node["source_count"],
                    json.dumps(node.get("source_memory_ids", [])),
                    node["first_observed"],
                    node["last_reinforced"],
                    node["total_duration_ms"],
                    node["total_interactions"],
                    node["stability_score"],
                    node["lifecycle_origin"],
                    node["created_at"],
                ))
            conn.commit()
        finally:
            conn.close()

        # Distribution counts
        dist_counts = defaultdict(int)
        for r in records:
            dist_counts[r.get("lifecycle_state", "RETAIN").lower()] += 1

        total_records = len(records)
        consolidation_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": total_records,
            "weights": self.weights.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "lifecycle_distribution": {
                "retain": dist_counts["retain"],
                "compress": dist_counts["compress"],
                "archive": dist_counts["archive"],
                "forget": dist_counts["forget"],
            },
            "lifecycle_percentages": {
                "retain": round((dist_counts["retain"] / max(1, total_records)) * 100, 2),
                "compress": round((dist_counts["compress"] / max(1, total_records)) * 100, 2),
                "archive": round((dist_counts["archive"] / max(1, total_records)) * 100, 2),
                "forget": round((dist_counts["forget"] / max(1, total_records)) * 100, 2),
            },
            "compression_metrics": comp_metrics,
            "status": "DMCLM baseline execution completed.",
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "consolidation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(consolidation_report, f, indent=2)

        # Baseline evaluation report
        eval_report = {
            "baseline_type": "Deterministic DMCLM Cognitive Formula",
            "formula": "0.20*Recency + 0.20*Frequency + 0.25*RetrievalHistory + 0.20*SemanticRelevance + 0.15*RelationshipStrength",
            "total_evaluated_memories": total_records,
            "lifecycle_distribution": consolidation_report["lifecycle_distribution"],
            "compression_summary": comp_metrics,
            "ml_status": "Ready for Training (Training Dataset Generated)",
            "training_dataset_path": str(DATA_DIR / "training_dataset.csv"),
            "note": "Behavior-derived labels are baseline experimental labels for future comparative evaluation against ML models (e.g. Random Forest).",
        }
        with open(RESULTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        logger.info("Saved consolidation and evaluation reports.")

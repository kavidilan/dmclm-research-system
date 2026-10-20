"""
DMCLM Memory Consolidation & Lifecycle Execution Engine
Implements the 4-way lifecycle execution:
1. COMPRESS -> Consolidates repeated episodic bursts into abstracted Long-Term Memory (LTM) nodes.
2. ARCHIVE  -> Transfers decaying historical context into secondary storage (data/archive_memories.json).
3. FORGET   -> Logically prunes transient noise traces (data/forgotten_memories.json).
4. RETAIN   -> Preserves high-utility memories in active working tier (data/retained_memories.json).

Outputs:
- data/long_term_memories.json
- data/archive_memories.json
- data/forgotten_memories.json
- data/retained_memories.json
- results/consolidation_metrics.json
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

MEMORY_JSON_PATH = DATA_DIR / "memory_records.json"
LTM_JSON_PATH = DATA_DIR / "long_term_memories.json"
ARCHIVE_JSON_PATH = DATA_DIR / "archive_memories.json"
FORGOTTEN_JSON_PATH = DATA_DIR / "forgotten_memories.json"
RETAINED_JSON_PATH = DATA_DIR / "retained_memories.json"
CONSOLIDATION_METRICS_PATH = RESULTS_DIR / "consolidation_metrics.json"


def cluster_and_abstract_memories(
    compress_memories: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Groups episodic memories by (session_id, application) and temporal bursts,
    synthesizing each group into an abstracted Long-Term Memory (LTM) node.
    """
    if not compress_memories:
        return [], {"original_count": 0, "consolidated_count": 0, "reduction": 0, "compression_ratio_pct": 0.0}

    # Group by (session_id, application)
    clusters: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for m in compress_memories:
        sid = m.get("session_id") or "unassigned_session"
        app = m.get("application") or "System"
        clusters.setdefault((sid, app), []).append(m)

    ltm_nodes = []
    now_iso = datetime.now().isoformat()

    for (sid, app), items in clusters.items():
        # Sort chronologically
        items.sort(key=lambda x: x.get("timestamp") or "")

        source_ids = [str(x["memory_id"]) for x in items]
        timestamps = [x.get("timestamp") for x in items if x.get("timestamp")]
        start_time = timestamps[0] if timestamps else now_iso
        end_time = timestamps[-1] if timestamps else now_iso

        total_duration_ms = sum(x.get("duration_ms") or 0 for x in items)
        total_interactions = sum(x.get("interaction_count") or 0 for x in items)

        # Extract unique salient window titles / domains
        unique_windows = list(dict.fromkeys([
            x.get("content", "").split(" | ")[1]
            for x in items
            if " | " in x.get("content", "") and len(x.get("content", "").split(" | ")) > 1
        ]))[:4]

        # Calculate semantic stability / importance
        mean_relevance = sum(x.get("semantic_relevance", 0.5) for x in items) / len(items)
        mean_freq = sum(x.get("frequency", 0.5) for x in items) / len(items)
        stability_score = round(min(1.0, 0.4 + (0.3 * mean_relevance) + (0.3 * mean_freq)), 4)

        windows_str = ", ".join(unique_windows) if unique_windows else "general desktop tasks"
        abstract_summary = (
            f"Consolidated research activity across {len(items)} episodic traces in {app} "
            f"({windows_str}) within session {sid[:8]}... "
            f"Total duration: {round(total_duration_ms / 1000, 1)}s with {total_interactions} user interactions."
        )

        ltm_id = f"ltm_{uuid.uuid4().hex[:8]}"
        ltm_node = {
            "ltm_id": ltm_id,
            "topic": f"{app} Task Continuity",
            "application": app,
            "session_id": sid,
            "summary": abstract_summary,
            "unique_windows_accessed": unique_windows,
            "source_count": len(items),
            "source_memory_ids": source_ids,
            "first_observed": start_time,
            "last_reinforced": end_time,
            "total_duration_ms": total_duration_ms,
            "total_interactions": total_interactions,
            "stability_score": stability_score,
            "lifecycle_origin": "COMPRESSED_EPISODIC",
            "created_at": now_iso
        }
        ltm_nodes.append(ltm_node)

    orig_count = len(compress_memories)
    cons_count = len(ltm_nodes)
    reduction = orig_count - cons_count
    ratio = round((reduction / orig_count) * 100, 2) if orig_count > 0 else 0.0

    stats = {
        "original_count": orig_count,
        "consolidated_count": cons_count,
        "records_reduced": reduction,
        "compression_ratio_pct": ratio
    }

    return ltm_nodes, stats


def run_consolidation_pipeline() -> Dict[str, Any]:
    print("\n=== RUNNING DMCLM CONSOLIDATION & LIFECYCLE EXECUTION ===")
    if not MEMORY_JSON_PATH.exists():
        raise FileNotFoundError(f"Memory records not found at {MEMORY_JSON_PATH}")

    with open(MEMORY_JSON_PATH, "r", encoding="utf-8") as f:
        memories = json.load(f)

    now_iso = datetime.now().isoformat()

    # Bucket memories by predicted lifecycle (fallback to RETAIN if not predicted yet)
    retain_list = []
    compress_list = []
    archive_list = []
    forget_list = []

    for m in memories:
        decision = m.get("lifecycle_prediction", "RETAIN")
        if decision == "COMPRESS":
            compress_list.append(m)
        elif decision == "ARCHIVE":
            archive_list.append(m)
        elif decision == "FORGET":
            forget_list.append(m)
        else:
            retain_list.append(m)

    print(f"Total processed memories: {len(memories)}")
    print(f"  - COMPRESS candidates : {len(compress_list)}")
    print(f"  - ARCHIVE candidates  : {len(archive_list)}")
    print(f"  - FORGET candidates   : {len(forget_list)}")
    print(f"  - RETAIN candidates   : {len(retain_list)}")

    # 1. Execute Consolidation on COMPRESS records
    print("\n1. Consolidating COMPRESS records into Long-Term Memory (LTM)...")
    ltm_nodes, comp_stats = cluster_and_abstract_memories(compress_list)

    with open(LTM_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ltm_nodes, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(ltm_nodes)} Long-Term Memory nodes saved to: {LTM_JSON_PATH}")
    print(f"Memory Reduction: {comp_stats['records_reduced']} records ({comp_stats['compression_ratio_pct']}% reduction)")

    # 2. Execute ARCHIVE
    print("\n2. Preserving ARCHIVE records in secondary historical tier...")
    archive_records = [
        {
            "memory_id": str(m["memory_id"]),
            "source_type": m.get("source_type"),
            "application": m.get("application"),
            "content": m.get("content"),
            "timestamp": m.get("timestamp"),
            "session_id": m.get("session_id"),
            "archived_at": now_iso,
            "lifecycle_decision": "ARCHIVE",
            "archive_reason": m.get("lifecycle_explanation") or "Age decay with historical research value."
        }
        for m in archive_list
    ]
    with open(ARCHIVE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(archive_records, f, indent=2, ensure_ascii=False)
    print(f"Archived {len(archive_records)} records to: {ARCHIVE_JSON_PATH}")

    # 3. Execute FORGET (Pruning from active tier without deleting collector.db)
    print("\n3. Recording FORGET records in tombstone audit trail...")
    forget_records = [
        {
            "memory_id": str(m["memory_id"]),
            "source_type": m.get("source_type"),
            "application": m.get("application"),
            "content": m.get("content"),
            "forgotten_at": now_iso,
            "lifecycle_decision": "FORGET",
            "forget_reason": m.get("lifecycle_explanation") or "Sub-threshold semantic task utility and low engagement."
        }
        for m in forget_list
    ]
    with open(FORGOTTEN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(forget_records, f, indent=2, ensure_ascii=False)
    print(f"Recorded {len(forget_records)} forgotten records to: {FORGOTTEN_JSON_PATH}")

    # 4. Save RETAIN active working memory tier
    print("\n4. Preserving RETAIN records in active working episodic tier...")
    with open(RETAINED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(retain_list, f, indent=2, ensure_ascii=False)
    print(f"Preserved {len(retain_list)} active working records in: {RETAINED_JSON_PATH}")

    # 5. Save Consolidation Metrics
    metrics_summary = {
        "timestamp": now_iso,
        "total_source_memories": len(memories),
        "lifecycle_distribution": {
            "retain": len(retain_list),
            "compress": len(compress_list),
            "archive": len(archive_list),
            "forget": len(forget_list)
        },
        "compression_metrics": {
            "original_episodic_records": len(compress_list),
            "consolidated_ltm_nodes": len(ltm_nodes),
            "memory_reduction_count": comp_stats["records_reduced"],
            "compression_ratio_pct": comp_stats["compression_ratio_pct"]
        },
        "storage_breakdown": {
            "active_working_memories": len(retain_list),
            "consolidated_ltm_nodes": len(ltm_nodes),
            "archived_records": len(archive_records),
            "forgotten_records": len(forget_records)
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONSOLIDATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved complete consolidation metrics to: {CONSOLIDATION_METRICS_PATH}")

    return metrics_summary


if __name__ == "__main__":
    run_consolidation_pipeline()

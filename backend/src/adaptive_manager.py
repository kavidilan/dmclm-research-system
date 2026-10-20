
"""
DMCLM Adaptive Retrieval & Dynamic Lifecycle Re-evaluation Manager

Implements the core adaptive cognitive feedback loop:
User retrieves/inspects a memory ->
Audit log records retrieval event ->
Memory features (Retrieval History & Recency) are re-evaluated ->
Baseline DMCLM formula recalculates Importance Score ->
State transitions dynamically (e.g., ARCHIVE / COMPRESS -> RETAIN).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

RETRIEVAL_HISTORY_PATH = DATA_DIR / "retrieval_history.json"
MEMORY_JSON_PATH = DATA_DIR / "memory_records.json"
ARCHIVE_JSON_PATH = DATA_DIR / "archive_memories.json"
RETAINED_JSON_PATH = DATA_DIR / "retained_memories.json"

try:
    from dmclm_config import DEFAULT_CONFIG
    from dmclm_db import (
        record_retrieval_in_db,
        insert_or_replace_memory_records,
    )
except ImportError:
    from src.dmclm_config import DEFAULT_CONFIG
    from src.dmclm_db import (
        record_retrieval_in_db,
        insert_or_replace_memory_records,
    )


def record_memory_retrieval(
    memory_id: str,
    retrieval_type: str = "user_inspection",
    is_useful: bool = True,
) -> Dict[str, Any]:
    """
    Record a valid memory retrieval, update DMCLM features, and
    recalculate lifecycle state.

    Database errors are propagated instead of silently ignored.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    str_mid = str(memory_id)

    # 1. Validate the memory before recording the retrieval.
    if not MEMORY_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Memory records not found at {MEMORY_JSON_PATH}"
        )

    with open(MEMORY_JSON_PATH, "r", encoding="utf-8") as f:
        memories = json.load(f)

    target_mem = None
    for memory in memories:
        if str(memory.get("memory_id")) == str_mid:
            target_mem = memory
            break

    if target_mem is None:
        raise ValueError(
            f"Memory record {str_mid} not found in memory store."
        )

    # 2. Prepare retrieval history without writing it yet.
    retrievals = []
    if RETRIEVAL_HISTORY_PATH.exists():
        with open(RETRIEVAL_HISTORY_PATH, "r", encoding="utf-8") as f:
            ret_data = json.load(f)
            retrievals = ret_data.get("retrievals", [])

    retrievals.append(
        {
            "memory_id": str_mid,
            "retrieval_timestamp": now_iso,
            "retrieval_type": retrieval_type,
            "is_useful": is_useful,
        }
    )

    # 3. Record retrieval in SQLite first.
    # Allow errors to propagate; do not report success if this fails.
    record_retrieval_in_db(
        str_mid,
        retrieval_type=retrieval_type,
        is_useful=is_useful,
    )

    # Save the audit history only after the SQLite retrieval succeeds.
    with open(RETRIEVAL_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Local DMCLM retrieval audit log",
                "total_retrievals": len(retrievals),
                "retrievals": retrievals,
            },
            f,
            indent=2,
        )

    previous_lifecycle = (
        target_mem.get("lifecycle_state")
        or target_mem.get("lifecycle_prediction")
        or "UNKNOWN"
    )

    # 4. Recalculate retrieval counts and adaptive features.
    mem_retrievals = [
        event
        for event in retrievals
        if str(event.get("memory_id")) == str_mid
    ]
    mem_retrieval_count = len(mem_retrievals)
    useful_count = sum(
        1 for event in mem_retrievals if event.get("is_useful", True)
    )

    target_mem["recency_score"] = 1.0
    target_mem["recency"] = 1.0
    target_mem["last_seen"] = now_iso

    h_score = round(
        useful_count / max(1, mem_retrieval_count),
        4,
    )
    target_mem["retrieval_history_score"] = h_score
    target_mem["retrieval_history"] = h_score
    target_mem["retrieval_count"] = mem_retrieval_count
    target_mem["useful_retrieval_count"] = useful_count

    # 5. Recalculate the existing baseline DMCLM importance score.
    weights = DEFAULT_CONFIG.weights
    r_val = 1.0
    f_val = float(
        target_mem.get("frequency_score")
        or target_mem.get("frequency")
        or 0.2
    )
    h_val = h_score
    s_val = float(target_mem.get("semantic_relevance") or 0.5)
    rel_val = float(target_mem.get("relationship_strength") or 0.5)

    raw_importance = (
        weights.recency * r_val
        + weights.frequency * f_val
        + weights.retrieval_history * h_val
        + weights.semantic_relevance * s_val
        + weights.relationship_strength * rel_val
    )
    importance = round(
        float(max(0.0, min(1.0, raw_importance))),
        4,
    )

    new_lifecycle = DEFAULT_CONFIG.thresholds.classify(importance)
    new_confidence = 1.0
    new_explanation = (
        f"Adaptive Retrieval Feedback: Memory retrieved "
        f"({mem_retrieval_count}x). "
        f"Recency refreshed to 1.0, Retrieval History to {h_val:.2f}. "
        f"Updated baseline importance: {importance:.3f} -> "
        f"{new_lifecycle}."
    )

    target_mem["importance_score"] = importance
    target_mem["lifecycle_state"] = new_lifecycle
    target_mem["lifecycle_prediction"] = new_lifecycle
    target_mem["lifecycle_confidence"] = new_confidence
    target_mem["lifecycle_explanation"] = new_explanation
    target_mem["updated_time"] = now_iso

    # 6. Persist the updated memory to SQLite first.
    # Do not silently ignore database failures.
    insert_or_replace_memory_records([target_mem])

    # Update the JSON memory store only after the SQLite write succeeds.
    with open(MEMORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

    # 7. Handle lifecycle-state migration.
    state_transitioned = previous_lifecycle != new_lifecycle

    if state_transitioned and new_lifecycle == "RETAIN":
        if ARCHIVE_JSON_PATH.exists():
            with open(ARCHIVE_JSON_PATH, "r", encoding="utf-8") as f:
                archive_list = json.load(f)

            archive_list = [
                item
                for item in archive_list
                if str(item.get("memory_id")) != str_mid
            ]

            with open(ARCHIVE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(archive_list, f, indent=2, ensure_ascii=False)

        if RETAINED_JSON_PATH.exists():
            with open(RETAINED_JSON_PATH, "r", encoding="utf-8") as f:
                retained_list = json.load(f)

            if not any(
                str(item.get("memory_id")) == str_mid
                for item in retained_list
            ):
                retained_list.append(target_mem)

                with open(RETAINED_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(
                        retained_list,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

    return {
        "memory_id": str_mid,
        "retrieval_count": mem_retrieval_count,
        "previous_lifecycle": previous_lifecycle,
        "new_lifecycle": new_lifecycle,
        "state_transitioned": state_transitioned,
        "updated_features": {
            "recency": target_mem["recency"],
            "retrieval_history": target_mem["retrieval_history"],
            "importance_score": importance,
        },
        "confidence": new_confidence,
        "explanation": new_explanation,
    }

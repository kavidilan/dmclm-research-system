"""
DMCLM Feature Engineering Pipeline
Calculates the 5 core cognitive features for memory records:
1. Recency: Exponential temporal decay relative to reference observation time.
2. Frequency: Application and domain task repetition density.
3. Retrieval History: Interaction log based on local DMCLM retrieval audit trail.
4. Semantic Relevance: Transparent TF-IDF centroid similarity within temporal sessions.
5. Relationship Strength: Multi-modal co-occurrence and temporal proximity scoring.

Generates:
- data/dmclm_features.csv
- updates data/memory_records.json with computed features
- initializes data/retrieval_history.json
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

MEMORY_JSON_PATH = DATA_DIR / "memory_records.json"
FEATURES_CSV_PATH = DATA_DIR / "dmclm_features.csv"
RETRIEVAL_HISTORY_PATH = DATA_DIR / "retrieval_history.json"


def init_retrieval_history_store() -> Dict[str, Any]:
    """Initialize or load the local DMCLM retrieval history audit trail."""
    if not RETRIEVAL_HISTORY_PATH.exists():
        initial_data = {
            "description": "Local DMCLM retrieval audit log. Updated when memories are searched/inspected.",
            "total_retrievals": 0,
            "retrievals": []
        }
        with open(RETRIEVAL_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)
        return initial_data
    
    try:
        with open(RETRIEVAL_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"retrievals": []}


def parse_iso_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp strings with timezone handling."""
    if not ts_str:
        return None
    try:
        # Standard fromisoformat handles +00:00 or Z
        clean_ts = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean_ts)
    except Exception:
        # Fallback slicing for microseconds if too long
        try:
            return datetime.fromisoformat(clean_ts[:19])
        except Exception:
            return None


def calculate_recency(
    timestamps: List[Optional[datetime]],
    half_life_hours: float = 72.0
) -> np.ndarray:
    """
    Exponential cognitive forgetting curve:
    R(t) = exp(-lambda * delta_t_hours)
    lambda = ln(2) / half_life_hours
    Memories closer to T_ref have R ~ 1.0; older memories decay towards 0.0.
    """
    valid_times = [t for t in timestamps if t is not None]
    if not valid_times:
        return np.zeros(len(timestamps))

    t_ref = max(valid_times)
    lambda_decay = math.log(2.0) / max(1.0, half_life_hours)

    recency_scores = []
    for t in timestamps:
        if t is None:
            recency_scores.append(0.0)
            continue
        delta_hours = max(0.0, (t_ref - t).total_seconds() / 3600.0)
        score = math.exp(-lambda_decay * delta_hours)
        recency_scores.append(float(round(score, 4)))

    return np.array(recency_scores)


def calculate_frequency(records: List[Dict[str, Any]]) -> np.ndarray:
    """
    Calculates task engagement density:
    Combines application frequency, domain repetition, and interaction event count.
    Normalized to [0.0, 1.0] using log-scaling to avoid dominance by outlier sessions.
    """
    app_counts: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}

    for r in records:
        app = r.get("application") or "unknown"
        app_counts[app] = app_counts.get(app, 0) + 1
        dom = r.get("domain")
        if dom:
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

    raw_scores = []
    for r in records:
        app = r.get("application") or "unknown"
        dom = r.get("domain")
        int_count = r.get("interaction_count") or 0

        app_term = math.log(1.0 + app_counts.get(app, 1))
        dom_term = math.log(1.0 + domain_counts.get(dom, 0)) if dom else 0.0
        int_term = math.log(1.0 + int_count) * 0.5

        raw_score = app_term + dom_term + int_term
        raw_scores.append(raw_score)

    raw_arr = np.array(raw_scores)
    min_v, max_v = raw_arr.min(), raw_arr.max()
    if max_v - min_v > 1e-6:
        normalized = (raw_arr - min_v) / (max_v - min_v)
    else:
        normalized = np.full_like(raw_arr, 0.5)

    return np.round(normalized, 4)


def calculate_retrieval_history(
    records: List[Dict[str, Any]],
    retrieval_store: Dict[str, Any]
) -> np.ndarray:
    """
    Calculates retrieval history from the local DMCLM audit trail.
    If no retrievals exist yet, returns 0.0 (clearly documented as insufficient history).
    """
    retrievals = retrieval_store.get("retrievals", [])
    if not retrievals:
        return np.zeros(len(records))

    retrieval_counts: Dict[str, int] = {}
    for entry in retrievals:
        mid = entry.get("memory_id")
        if mid:
            retrieval_counts[mid] = retrieval_counts.get(mid, 0) + 1

    scores = []
    for r in records:
        mid = str(r.get("memory_id"))
        cnt = retrieval_counts.get(mid, 0)
        # Scaled: 1 retrieval = 0.25, 4+ retrievals = 1.0
        score = min(1.0, cnt * 0.25)
        scores.append(score)

    return np.array(scores)


def calculate_semantic_relevance(records: List[Dict[str, Any]]) -> np.ndarray:
    """
    Transparent TF-IDF session centroid relevance:
    Computes TF-IDF representation across all memory contents.
    Calculates cosine similarity of each memory to its session task centroid vector.
    High score indicates memory is central to the user's primary session task.
    """
    corpus = [r.get("content") or r.get("application") or "activity" for r in records]
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except Exception:
        return np.full(len(records), 0.5)

    # Group by session_id to find session centroids
    session_indices: Dict[str, List[int]] = {}
    for idx, r in enumerate(records):
        sid = r.get("session_id") or "default_session"
        session_indices.setdefault(sid, []).append(idx)

    relevance_scores = np.zeros(len(records))

    for sid, indices in session_indices.items():
        if not indices:
            continue
        session_vectors = tfidf_matrix[indices]
        # Centroid of session activity
        centroid = np.asarray(session_vectors.mean(axis=0))
        # Cosine similarity against session centroid
        sims = cosine_similarity(session_vectors, centroid).flatten()
        for local_idx, global_idx in enumerate(indices):
            relevance_scores[global_idx] = sims[local_idx]

    # Normalize to [0.0, 1.0]
    min_v, max_v = relevance_scores.min(), relevance_scores.max()
    if max_v - min_v > 1e-6:
        relevance_scores = (relevance_scores - min_v) / (max_v - min_v)
    else:
        relevance_scores = np.full_like(relevance_scores, 0.5)

    return np.round(relevance_scores, 4)


def calculate_relationship_strength(
    records: List[Dict[str, Any]],
    timestamps: List[Optional[datetime]]
) -> np.ndarray:
    """
    Calculates multi-modal relational linkage based on:
    - Same session affinity (weight 0.4)
    - Application continuity in adjacent time window (weight 0.3)
    - Temporal proximity (exponential proximity kernel within +/- 10 mins, weight 0.3)
    """
    n = len(records)
    scores = np.zeros(n)

    # Use sliding window of size 10 (chronologically sorted)
    window_size = 5

    for i in range(n):
        cur_rec = records[i]
        cur_time = timestamps[i]
        cur_app = cur_rec.get("application")
        cur_session = cur_rec.get("session_id")

        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)
        neighbors = [j for j in range(start_idx, end_idx) if j != i]

        if not neighbors:
            scores[i] = 0.5
            continue

        session_matches = 0
        app_matches = 0
        temporal_sum = 0.0

        for j in neighbors:
            other_rec = records[j]
            other_time = timestamps[j]

            if other_rec.get("session_id") == cur_session:
                session_matches += 1
            if other_rec.get("application") == cur_app:
                app_matches += 1

            if cur_time and other_time:
                delta_sec = abs((cur_time - other_time).total_seconds())
                # Kernel decays over 10 minutes (600s)
                temporal_sum += math.exp(-delta_sec / 600.0)

        r_session = session_matches / len(neighbors)
        r_app = app_matches / len(neighbors)
        r_time = min(1.0, temporal_sum / len(neighbors))

        composite = 0.4 * r_session + 0.3 * r_app + 0.3 * r_time
        scores[i] = composite

    return np.round(scores, 4)


def run_feature_engineering() -> pd.DataFrame:
    """
    Orchestrates the feature calculation pipeline.
    Updates memory_records.json and saves dmclm_features.csv.
    """
    print("\n=== DMCLM COGNITIVE FEATURE ENGINEERING ===")
    if not MEMORY_JSON_PATH.exists():
        raise FileNotFoundError(f"Memory records file not found at {MEMORY_JSON_PATH}")

    with open(MEMORY_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    total_records = len(records)
    print(f"Loaded {total_records} memory records from {MEMORY_JSON_PATH}.")

    # 1. Parse timestamps
    print("1. Parsing chronological timestamps...")
    timestamps = [parse_iso_timestamp(r.get("timestamp")) for r in records]

    # 2. Recency
    print("2. Calculating Recency scores (Ebbinghaus temporal decay)...")
    recency_scores = calculate_recency(timestamps, half_life_hours=48.0)

    # 3. Frequency
    print("3. Calculating Frequency scores (Task repetition density)...")
    frequency_scores = calculate_frequency(records)

    # 4. Retrieval History
    print("4. Calculating Retrieval History from DMCLM audit trail...")
    retrieval_store = init_retrieval_history_store()
    retrieval_scores = calculate_retrieval_history(records, retrieval_store)

    # 5. Semantic Relevance
    print("5. Calculating Semantic Relevance (TF-IDF Session Centroid Cosine)...")
    semantic_scores = calculate_semantic_relevance(records)

    # 6. Relationship Strength
    print("6. Calculating Relationship Strength (Temporal proximity & co-occurrence)...")
    relationship_scores = calculate_relationship_strength(records, timestamps)

    # Update records dictionary
    for i, r in enumerate(records):
        r["recency"] = float(recency_scores[i])
        r["frequency"] = float(frequency_scores[i])
        r["retrieval_history"] = float(retrieval_scores[i])
        r["semantic_relevance"] = float(semantic_scores[i])
        r["relationship_strength"] = float(relationship_scores[i])

    # Save updated memory_records.json
    with open(MEMORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Updated {len(records)} records in {MEMORY_JSON_PATH} with calculated features.")

    # Create DataFrame for model training
    df_features = pd.DataFrame({
        "memory_id": [r.get("memory_id") for r in records],
        "source_type": [r.get("source_type") for r in records],
        "timestamp": [r.get("timestamp") for r in records],
        "session_id": [r.get("session_id") for r in records],
        "application": [r.get("application") for r in records],
        "duration_ms": [r.get("duration_ms") or 0 for r in records],
        "interaction_count": [r.get("interaction_count") or 0 for r in records],
        "recency": recency_scores,
        "frequency": frequency_scores,
        "retrieval_history": retrieval_scores,
        "semantic_relevance": semantic_scores,
        "relationship_strength": relationship_scores
    })

    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(FEATURES_CSV_PATH, index=False)
    print(f"Saved feature dataset to: {FEATURES_CSV_PATH} ({len(df_features)} rows)")

    # Print summary statistics
    print("\nFeature Summary Statistics:")
    print(df_features[["recency", "frequency", "retrieval_history", "semantic_relevance", "relationship_strength"]].describe().round(4))

    return df_features


if __name__ == "__main__":
    run_feature_engineering()

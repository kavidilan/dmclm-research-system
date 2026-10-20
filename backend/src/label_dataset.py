"""
DMCLM Heuristic Reference Labeling Methodology
Generates documented ground-truth reference labels for training the custom DMCLM classifier.
Classes:
- RETAIN: High recency/retrieval active working memory.
- COMPRESS: High frequency & relational redundancy suitable for abstraction into LTM.
- ARCHIVE: Older, moderately relevant historical context preserved in secondary tier.
- FORGET: Low semantic relevance & low engagement noise falling below decay thresholds.

Generates:
- data/training_dataset.csv
- documents label distribution and justification
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

FEATURES_CSV_PATH = DATA_DIR / "dmclm_features.csv"
TRAINING_CSV_PATH = DATA_DIR / "training_dataset.csv"


def assign_reference_label(row: pd.Series) -> Tuple[str, str]:
    """
    Applies transparent, documented cognitive lifecycle rules to assign reference label.
    Returns: (label, rationale)
    """
    r = row["recency"]
    f = row["frequency"]
    h = row["retrieval_history"]
    s = row["semantic_relevance"]
    rel = row["relationship_strength"]

    # Rule 1: FORGET (Transient noise, low semantic relevance, low task frequency)
    if s < 0.18 and f < 0.35:
        return "FORGET", "Low semantic task relevance (<0.18) and low frequency (<0.35)"

    # Rule 2: RETAIN (Active working knowledge: high recency or retrieval + high relevance)
    if (r >= 0.68 or h > 0.0) and (s >= 0.38 or f >= 0.60):
        return "RETAIN", "Active working memory with high recency/retrieval and task engagement"

    # Rule 3: COMPRESS (Redundant episodic sequence: high frequency and strong relationship)
    if f >= 0.50 and rel >= 0.85:
        return "COMPRESS", "High session frequency (>=0.50) and strong relational cohesion (>=0.85)"

    # Rule 4: ARCHIVE (Older or moderate historical context)
    if r < 0.55 and (s >= 0.25 or rel >= 0.70):
        return "ARCHIVE", "Historical context with decayed recency (<0.55) suitable for secondary storage"

    # Fallback default rule based on utility composite
    utility = 0.3 * r + 0.3 * f + 0.2 * s + 0.2 * rel
    if utility >= 0.65:
        return "RETAIN", "High aggregate utility composite score"
    elif utility >= 0.45:
        return "COMPRESS", "Moderate utility with consolidation potential"
    elif utility >= 0.30:
        return "ARCHIVE", "Low-moderate utility historical record"
    else:
        return "FORGET", "Sub-threshold utility trace"


def generate_training_dataset() -> pd.DataFrame:
    """
    Loads dmclm_features.csv and produces training_dataset.csv with reference labels.
    """
    print("\n=== GENERATING RESEARCH TRAINING DATASET ===")
    if not FEATURES_CSV_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV_PATH}")

    df = pd.read_csv(FEATURES_CSV_PATH)
    print(f"Loaded {len(df)} records from {FEATURES_CSV_PATH}")

    labels = []
    reasons = []

    for _, row in df.iterrows():
        lbl, rsn = assign_reference_label(row)
        labels.append(lbl)
        reasons.append(rsn)

    df["lifecycle_label"] = labels
    df["label_rationale"] = reasons

    TRAINING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_CSV_PATH, index=False)
    print(f"Saved labeled training dataset to: {TRAINING_CSV_PATH}")

    print("\nReference Label Distribution (Ground Truth):")
    counts = df["lifecycle_label"].value_counts()
    percentages = df["lifecycle_label"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
    print(summary)

    return df


if __name__ == "__main__":
    generate_training_dataset()

import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

ROOT = Path.cwd()
CSV = ROOT / "data" / "training_dataset.csv"
OUT = ROOT / "results" / "feature_selection_validation.json"

CORE = [
    "feature_recency", "feature_frequency",
    "feature_retrieval_history", "feature_semantic_relevance",
    "feature_relationship_strength",
]

CONTEXT = [
    "duration_seconds", "interaction_count", "browser_event_count",
    "screenshot_count", "has_browser_activity", "has_window_context",
    "hour_of_day", "day_of_week", "is_weekend", "is_consolidated",
    "cluster_member_count", "graph_degree", "graph_avg_rel_strength",
    "graph_max_rel_strength", "is_graph_isolated",
]

FEATURE_SETS = {
    "all_20": CORE + CONTEXT,
    "core_5": CORE,
    "core_plus_activity": CORE + [
        "duration_seconds", "interaction_count",
        "browser_event_count", "screenshot_count",
        "has_browser_activity", "has_window_context",
    ],
    "core_plus_graph": CORE + [
        "is_consolidated", "cluster_member_count",
        "graph_degree", "graph_avg_rel_strength",
        "graph_max_rel_strength", "is_graph_isolated",
    ],
    "core_plus_activity_graph": CORE + [
        "duration_seconds", "interaction_count",
        "browser_event_count", "screenshot_count",
        "has_browser_activity", "has_window_context",
        "is_consolidated", "cluster_member_count",
        "graph_degree", "graph_avg_rel_strength",
        "graph_max_rel_strength", "is_graph_isolated",
    ],
}

if not CSV.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV}")
if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")

needed = ["split", "target_lifecycle_state"] + CORE + CONTEXT
df = pd.read_csv(CSV, usecols=lambda c: c in set(needed))
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Exclude test data before fitting or evaluation.
df = df[df["split"].isin(["train", "val"])].copy()
train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()

if train.empty or val.empty:
    raise ValueError("Training or validation data is empty.")
if train["target_lifecycle_state"].isna().any() or val["target_lifecycle_state"].isna().any():
    raise ValueError("Missing lifecycle labels.")

y_train = train["target_lifecycle_state"].astype(str)
y_val = val["target_lifecycle_state"].astype(str)
if set(y_val.unique()) - set(y_train.unique()):
    raise ValueError("Validation contains a class absent from training.")

results = {
    "training_rows": len(train),
    "validation_rows": len(val),
    "test_rows_loaded": 0,
    "selection_split": "validation",
    "sklearn_version": sklearn.__version__,
    "experiments": {},
}

for name, features in FEATURE_SETS.items():
    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_val = val[features].apply(pd.to_numeric, errors="coerce")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    report = classification_report(
        y_val, pred, output_dict=True, zero_division=0
    )

    results["experiments"][name] = {
        "features": features,
        "feature_count": len(features),
        "accuracy": round(float(accuracy_score(y_val, pred)), 4),
        "macro_f1": round(float(f1_score(
            y_val, pred, average="macro", zero_division=0
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_val, pred, average="weighted", zero_division=0
        )), 4),
        "per_class_f1": {
            cls: round(float(report[cls]["f1-score"]), 4)
            for cls in ["ARCHIVE", "COMPRESS", "FORGET", "RETAIN"]
        },
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

for name, r in results["experiments"].items():
    print(f"\n{name} ({r['feature_count']} features)")
    print("Accuracy:", r["accuracy"])
    print("Macro F1:", r["macro_f1"])
    print("Weighted F1:", r["weighted_f1"])
    print("Per-class F1:", r["per_class_f1"])

print("\nTest rows loaded: 0")
print("Saved report:", OUT)

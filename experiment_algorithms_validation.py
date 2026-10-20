import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

ROOT = Path.cwd()
CSV = ROOT / "data" / "training_dataset.csv"
OUT = ROOT / "results" / "algorithm_comparison_validation.json"

FEATURES = [
    "feature_recency", "feature_frequency",
    "feature_retrieval_history", "feature_semantic_relevance",
    "feature_relationship_strength", "duration_seconds",
    "interaction_count", "browser_event_count", "screenshot_count",
    "has_browser_activity", "has_window_context", "hour_of_day",
    "day_of_week", "is_weekend", "is_consolidated",
    "cluster_member_count", "graph_degree",
    "graph_avg_rel_strength", "graph_max_rel_strength",
    "is_graph_isolated",
]

if not CSV.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV}")
if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")

needed = ["split", "target_lifecycle_state"] + FEATURES
df = pd.read_csv(CSV, usecols=lambda c: c in set(needed))
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Exclude test rows before any experiment.
df = df[df["split"].isin(["train", "val"])].copy()
train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()

if train.empty or val.empty:
    raise ValueError("Training or validation partition is empty.")

y_train = train["target_lifecycle_state"].astype(str)
y_val = val["target_lifecycle_state"].astype(str)

if y_train.isin(["nan", "None"]).any() or y_val.isin(["nan", "None"]).any():
    raise ValueError("Missing lifecycle labels found.")
if set(y_val.unique()) - set(y_train.unique()):
    raise ValueError("Validation has a class absent from training.")

X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
X_val = val[FEATURES].apply(pd.to_numeric, errors="coerce")

models = {
    # Your current best benchmark
    "random_forest_benchmark": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    ),
}

results = {
    "dataset": "data/training_dataset.csv",
    "training_rows": len(train),
    "validation_rows": len(val),
    "test_rows_loaded": 0,
    "selection_split": "validation",
    "features": FEATURES,
    "sklearn_version": sklearn.__version__,
    "experiments": {},
}

for name, classifier in models.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    report = classification_report(
        y_val, pred, output_dict=True, zero_division=0
    )

    results["experiments"][name] = {
        "accuracy": round(float(accuracy_score(y_val, pred)), 4),
        "macro_f1": round(float(f1_score(
            y_val, pred, average="macro", zero_division=0
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_val, pred, average="weighted", zero_division=0
        )), 4),
        "per_class": {
            cls: {
                "precision": round(float(report[cls]["precision"]), 4),
                "recall": round(float(report[cls]["recall"]), 4),
                "f1": round(float(report[cls]["f1-score"]), 4),
            }
            for cls in ["ARCHIVE", "COMPRESS", "FORGET", "RETAIN"]
        },
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

for name, r in results["experiments"].items():
    print(f"\n{name}")
    print("Accuracy:", r["accuracy"])
    print("Macro F1:", r["macro_f1"])
    print("Weighted F1:", r["weighted_f1"])
    for cls, m in r["per_class"].items():
        print(f"  {cls}: F1={m['f1']:.4f}, recall={m['recall']:.4f}")

print("\nTest rows loaded: 0")
print("Saved report:", OUT)

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
OUT = ROOT / "results" / "class_weight_experiment_validation.json"

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

# Exclude test rows immediately.
df = df[df["split"].isin(["train", "val"])].copy()
train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()

if train.empty or val.empty:
    raise ValueError("Training or validation partition is empty.")
if train["target_lifecycle_state"].isna().any():
    raise ValueError("Training labels contain missing values.")
if val["target_lifecycle_state"].isna().any():
    raise ValueError("Validation labels contain missing values.")

y_train = train["target_lifecycle_state"].astype(str)
y_val = val["target_lifecycle_state"].astype(str)

if set(y_val.unique()) - set(y_train.unique()):
    raise ValueError("Validation contains classes absent from training.")

X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
X_val = val[FEATURES].apply(pd.to_numeric, errors="coerce")

# Same features and forest parameters; only class weights change.
strategies = {
    "balanced": "balanced",
    "balanced_subsample": "balanced_subsample",
    "custom_archive_forget": {
        "RETAIN": 1.0,
        "COMPRESS": 1.0,
        "ARCHIVE": 2.0,
        "FORGET": 1.5,
    },
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

for name, weights in strategies.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=5,
            class_weight=weights,
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
        "class_weight": weights,
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
                "support": int(report[cls]["support"]),
            }
            for cls in ["ARCHIVE", "COMPRESS", "FORGET", "RETAIN"]
        },
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

for name, result in results["experiments"].items():
    print(f"\n{name}")
    print("Accuracy:", result["accuracy"])
    print("Macro F1:", result["macro_f1"])
    print("Weighted F1:", result["weighted_f1"])
    for cls, metrics in result["per_class"].items():
        print(
            f"  {cls}: F1={metrics['f1']:.4f}, "
            f"recall={metrics['recall']:.4f}"
        )

print("\nTest rows loaded: 0")
print("Saved report:", OUT)

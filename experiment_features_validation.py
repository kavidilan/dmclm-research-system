import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

ROOT = Path.cwd()
CSV = ROOT / "data" / "training_dataset.csv"
OUT = ROOT / "results" / "feature_experiment_validation.json"

CORE = [
    "feature_recency",
    "feature_frequency",
    "feature_retrieval_history",
    "feature_semantic_relevance",
    "feature_relationship_strength",
]

CONTEXT = [
    "duration_seconds", "interaction_count", "browser_event_count",
    "screenshot_count", "has_browser_activity", "has_window_context",
    "hour_of_day", "day_of_week", "is_weekend", "is_consolidated",
    "cluster_member_count", "graph_degree", "graph_avg_rel_strength",
    "graph_max_rel_strength", "is_graph_isolated",
]

if not CSV.is_file():
    raise FileNotFoundError(f"Dataset not found: {CSV}")

# Load only train/validation rows. The test partition is excluded at read time.
df = pd.read_csv(
    CSV,
    usecols=lambda c: c in set(
        ["split", "target_lifecycle_state"] + CORE + CONTEXT
    ),
)
df = df[df["split"].isin(["train", "val"])].copy()

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()

if train.empty or val.empty:
    raise ValueError("Training or validation partition is empty.")

if train["target_lifecycle_state"].isna().any() or val["target_lifecycle_state"].isna().any():
    raise ValueError("Missing lifecycle labels found.")

if not set(train["target_lifecycle_state"].unique()).issuperset(
    set(val["target_lifecycle_state"].unique())
):
    raise ValueError("Validation contains a class absent from training.")

results = {
    "dataset": str(CSV.relative_to(ROOT)),
    "training_rows": len(train),
    "validation_rows": len(val),
    "test_rows_loaded": 0,
    "selection_split": "validation",
    "sklearn_version": sklearn.__version__,
    "experiments": {},
}

for name, features in [
    ("core_five", CORE),
    ("enriched", CORE + [c for c in CONTEXT if c in df.columns]),
]:
    if any(c not in df.columns for c in CORE):
        missing = [c for c in CORE if c not in df.columns]
        raise ValueError(f"Missing core features: {missing}")

    # Convert boolean/context/numeric columns safely to numeric.
    X_train = train[features].apply(pd.to_numeric, errors="coerce")
    X_val = val[features].apply(pd.to_numeric, errors="coerce")
    y_train = train["target_lifecycle_state"].astype(str)
    y_val = val["target_lifecycle_state"].astype(str)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    results["experiments"][name] = {
        "features": features,
        "accuracy": round(float(accuracy_score(y_val, pred)), 4),
        "macro_f1": round(float(f1_score(y_val, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_val, pred, average="weighted", zero_division=0)), 4),
        "per_class": classification_report(
            y_val, pred, output_dict=True, zero_division=0
        ),
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    raise FileExistsError(
        f"Output already exists; refusing to overwrite: {OUT}"
    )

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

for name, metrics in results["experiments"].items():
    print(f"\n{name}")
    print("Features:", len(metrics["features"]))
    print("Validation accuracy:", metrics["accuracy"])
    print("Validation macro F1:", metrics["macro_f1"])
    print("Validation weighted F1:", metrics["weighted_f1"])
    print("Per-class F1:")
    for cls, values in metrics["per_class"].items():
        if isinstance(values, dict) and "f1-score" in values:
            print(f"  {cls}: {values['f1-score']:.4f}")

print("\nTest rows loaded: 0")
print("Saved new report:", OUT)

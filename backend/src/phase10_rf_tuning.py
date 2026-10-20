"""Validation-only Random Forest tuning for the Phase 10 DMCLM classifier."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    from src.phase10_ml_training import CORE_FIVE_FEATURES, LIFECYCLE_MODEL_PATH, TRAINING_CSV_PATH
except ImportError:
    from phase10_ml_training import CORE_FIVE_FEATURES, LIFECYCLE_MODEL_PATH, TRAINING_CSV_PATH


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
TARGET_COLUMN = "target_lifecycle_state"
SPLIT_COLUMN = "split"
TIME_COLUMN = "start_time"
TARGET_CLASSES = ("ARCHIVE", "COMPRESS", "FORGET", "RETAIN")
OBSERVATION_HORIZON_HOURS = 48


def default_parameter_grid(n_estimators: int = 200) -> List[Dict[str, Any]]:
    """Return a deterministic grid spanning tree complexity and class weighting."""
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
        }
        for max_depth, min_samples_leaf, max_features, class_weight in itertools.product(
            (None, 8, 16),
            (1, 5, 15),
            ("sqrt", 1.0),
            (None, "balanced", "balanced_subsample"),
        )
    ]


def prepare_temporal_splits(
    dataframe: pd.DataFrame,
    observation_horizon_hours: int = OBSERVATION_HORIZON_HOURS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Purge train/validation labels whose future window crosses the next split."""
    required = {SPLIT_COLUMN, TIME_COLUMN, TARGET_COLUMN, *CORE_FIVE_FEATURES}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {sorted(missing)}")

    original = {
        name: dataframe.loc[dataframe[SPLIT_COLUMN] == name].copy()
        for name in ("train", "val", "test")
    }
    if any(part.empty for part in original.values()):
        raise ValueError("Dataset must contain non-empty train, val, and test partitions.")

    times = {
        name: pd.to_datetime(part[TIME_COLUMN], utc=True, errors="coerce")
        for name, part in original.items()
    }
    if any(series.isna().any() for series in times.values()):
        raise ValueError("Dataset contains invalid timestamps in a split partition.")

    val_start = times["val"].min()
    test_start = times["test"].min()
    dataset_max_time = max(series.max() for series in times.values())
    if times["train"].max() > val_start or times["val"].max() > test_start:
        raise ValueError("Dataset splits are not in chronological order.")

    horizon = pd.Timedelta(hours=observation_horizon_hours)
    train_keep = times["train"] + horizon < val_start
    val_keep = times["val"] + horizon < test_start
    train = original["train"].loc[train_keep].copy()
    validation = original["val"].loc[val_keep].copy()
    test = original["test"].copy()
    if train.empty or validation.empty:
        raise ValueError("48-hour boundary purging left no usable train or validation rows.")

    report = {
        "strategy": "precomputed_chronological_with_future-label_embargo",
        "observation_horizon_hours": observation_horizon_hours,
        "split_start_times": {
            name: series.min().isoformat() for name, series in times.items()
        },
        "original_counts": {name: len(part) for name, part in original.items()},
        "effective_counts": {
            "train": len(train), "validation": len(validation), "test": len(test)
        },
        "purged_counts": {
            "train": len(original["train"]) - len(train),
            "validation": len(original["val"]) - len(validation),
        },
        "test_count_with_complete_future_horizon": int(
            (times["test"] + horizon <= dataset_max_time).sum()
        ),
        "test_label_horizon_complete": bool(
            times["test"].max() + horizon <= dataset_max_time
        ),
    }
    return train, validation, test, report


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_labels: Sequence[str] = TARGET_CLASSES,
    probabilities: Optional[np.ndarray] = None,
    probability_classes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Calculate class-aware metrics with explicit, stable class ordering."""
    labels = list(class_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class: Dict[str, Dict[str, Any]] = {}
    for index, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "classes": labels,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "sample_count": len(y_true),
        "roc_auc_ovr_macro": None,
    }

    if probabilities is not None and probability_classes is not None:
        probability_indices = {label: index for index, label in enumerate(probability_classes)}
        for label in labels:
            if label not in probability_indices:
                per_class[label]["roc_auc_ovr"] = None
                continue
            binary_truth = np.asarray(y_true) == label
            if binary_truth.any() and (~binary_truth).any():
                per_class[label]["roc_auc_ovr"] = round(float(roc_auc_score(
                    binary_truth, probabilities[:, probability_indices[label]]
                )), 4)
            else:
                per_class[label]["roc_auc_ovr"] = None

        if all(label in probability_indices for label in labels):
            ordered_probabilities = probabilities[:, [probability_indices[label] for label in labels]]
            try:
                metrics["roc_auc_ovr_macro"] = round(float(roc_auc_score(
                    y_true,
                    ordered_probabilities,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )), 4)
            except ValueError:
                metrics["roc_auc_ovr_macro"] = None
    return metrics


def majority_class_baseline(
    training_labels: Sequence[str],
    evaluation_labels: Sequence[str],
    class_labels: Sequence[str] = TARGET_CLASSES,
) -> Dict[str, Any]:
    """Evaluate a constant prediction chosen from training labels only."""
    majority_label = pd.Series(training_labels).mode().iloc[0]
    predictions = [majority_label] * len(evaluation_labels)
    return {
        "prediction": str(majority_label),
        "metrics": classification_metrics(evaluation_labels, predictions, class_labels),
    }


def tune_random_forest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str] = CORE_FIVE_FEATURES,
    parameter_grid: Optional[Iterable[Dict[str, Any]]] = None,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select hyperparameters by validation macro-F1; test data is not accepted."""
    features = list(feature_names)
    for name, part in (("train", train), ("validation", validation)):
        missing = set(features + [TARGET_COLUMN]) - set(part.columns)
        if missing:
            raise ValueError(f"{name} partition is missing columns: {sorted(missing)}")
        if part[features].isna().any().any():
            raise ValueError(f"{name} feature matrix contains missing values.")
        if not all(pd.api.types.is_numeric_dtype(part[column]) for column in features):
            raise ValueError("Random Forest input features must be numeric.")

    candidates = list(parameter_grid or default_parameter_grid())
    if not candidates:
        raise ValueError("At least one Random Forest parameter candidate is required.")

    x_train, y_train = train[features], train[TARGET_COLUMN]
    x_validation, y_validation = validation[features], validation[TARGET_COLUMN]
    class_labels = list(TARGET_CLASSES)
    results: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None

    for candidate in candidates:
        parameters = {
            **candidate,
            "random_state": random_seed,
            "n_jobs": n_jobs,
        }
        classifier = RandomForestClassifier(**parameters)
        classifier.fit(x_train, y_train)
        probabilities = classifier.predict_proba(x_validation)
        predictions = classifier.predict(x_validation)
        metrics = classification_metrics(
            y_validation,
            predictions,
            class_labels,
            probabilities,
            classifier.classes_,
        )
        row = {"parameters": candidate, "validation_metrics": metrics}
        results.append(row)
        if best_metrics is None or (
            metrics["macro_f1"], metrics["accuracy"]
        ) > (best_metrics["macro_f1"], best_metrics["accuracy"]):
            best_params = dict(candidate)
            best_metrics = metrics

    assert best_params is not None and best_metrics is not None
    return best_params, {
        "selection_metric": "validation macro-F1, accuracy as tie-breaker",
        "candidate_count": len(results),
        "selected_validation_metrics": best_metrics,
        "candidates": results,
    }


def run_tuning(
    dataset_path: Path = TRAINING_CSV_PATH,
    parameter_grid: Optional[Iterable[Dict[str, Any]]] = None,
    observation_horizon_hours: int = OBSERVATION_HORIZON_HOURS,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Tune using purged train/validation rows, then evaluate the untouched test split."""
    dataframe = pd.read_csv(dataset_path)
    features = list(CORE_FIVE_FEATURES)
    required = {TARGET_COLUMN, SPLIT_COLUMN, TIME_COLUMN, *features}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {sorted(missing)}")
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Target labels contain missing values.")

    train, validation, test, split_report = prepare_temporal_splits(
        dataframe, observation_horizon_hours
    )
    best_params, tuning_report = tune_random_forest(
        train,
        validation,
        features,
        parameter_grid,
        n_jobs=n_jobs,
    )

    final_training = pd.concat([train, validation]).sort_values(TIME_COLUMN)
    final_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=n_jobs,
    )
    final_model.fit(final_training[features], final_training[TARGET_COLUMN])

    # Test labels and features are first used after model selection is complete.
    x_test = test[features]
    y_test = test[TARGET_COLUMN]
    tuned_metrics = classification_metrics(
        y_test,
        final_model.predict(x_test),
        TARGET_CLASSES,
        final_model.predict_proba(x_test),
        final_model.classes_,
    )
    majority = majority_class_baseline(final_training[TARGET_COLUMN], y_test)

    baseline_model_path = LIFECYCLE_MODEL_PATH
    baseline_metadata_path = MODELS_DIR / "model_metadata_baseline_20261009.json"
    original_baseline_metrics: Optional[Dict[str, Any]] = None
    if baseline_model_path.exists():
        baseline_model = joblib.load(baseline_model_path)
        baseline_metadata: Dict[str, Any] = {}
        if baseline_metadata_path.exists():
            baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
        baseline_features = baseline_metadata.get("feature_names", features)
        original_baseline_metrics = classification_metrics(
            y_test,
            baseline_model.predict(test[baseline_features]),
            TARGET_CLASSES,
            baseline_model.predict_proba(test[baseline_features]),
            baseline_model.classes_,
        )

    source_bytes = Path(dataset_path).read_bytes()
    raw_dataset_hash = hashlib.sha256(source_bytes).hexdigest()
    effective_max_time = pd.to_datetime(dataframe[TIME_COLUMN], utc=True).max()
    complete_test_count = int((
        pd.to_datetime(test[TIME_COLUMN], utc=True)
        + pd.Timedelta(hours=observation_horizon_hours)
        <= effective_max_time
    ).sum())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = MODELS_DIR / f"dmclm_lifecycle_model_v2_tuned_{run_id}.joblib"
    metadata_path = MODELS_DIR / f"model_metadata_v2_tuned_{run_id}.json"
    metrics_path = RESULTS_DIR / f"model_metrics_v2_tuned_{run_id}.json"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_version": "v2-random-forest-validation-tuned",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {**best_params, "random_state": 42, "n_jobs": n_jobs},
        "random_seed": 42,
        "feature_names": features,
        "target_column": TARGET_COLUMN,
        "target_classes": list(TARGET_CLASSES),
        "dataset_path": str(Path(dataset_path)),
        "dataset_sha256": raw_dataset_hash,
        "phase10_dataset_hash": hashlib.sha256(dataframe.to_csv(index=False).encode("utf-8")).hexdigest()[:16],
        "dataset_rows": len(dataframe),
        "label_origin_counts": dataframe.get("label_origin", pd.Series(dtype=str)).value_counts().to_dict(),
        "split_information": split_report,
        "tuning": tuning_report,
        "evaluation": {
            "test_metrics": tuned_metrics,
            "majority_class_baseline": majority,
            "original_model_baseline": original_baseline_metrics,
            "test_label_horizon_complete": complete_test_count == len(test),
            "test_rows_with_complete_48h_label_horizon": complete_test_count,
            "limitation": (
                "Test labels are right-censored: Phase 9 caps the future observation window at the "
                "dataset maximum timestamp. Treat these test metrics as provisional until a complete "
                "48-hour post-test observation period is available and labels are regenerated."
            ),
        },
        "model_artifact_path": str(model_path.relative_to(ROOT_DIR)),
    }
    metrics_report = {
        "model_version": metadata["model_version"],
        "training_timestamp": metadata["training_timestamp"],
        "algorithm": metadata["algorithm"],
        "hyperparameters": metadata["hyperparameters"],
        "split_information": split_report,
        "tuning": tuning_report,
        "test_metrics": tuned_metrics,
        "majority_class_baseline": majority,
        "original_model_baseline": original_baseline_metrics,
        "test_label_horizon_complete": complete_test_count == len(test),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")

    print(f"Selected validation macro-F1: {tuning_report['selected_validation_metrics']['macro_f1']:.4f}")
    print(f"Test accuracy: {tuned_metrics['accuracy']:.4f}; macro-F1: {tuned_metrics['macro_f1']:.4f}")
    print(f"Majority test accuracy: {majority['metrics']['accuracy']:.4f}")
    print(f"Test label horizon complete: {complete_test_count}/{len(test)}")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Metrics: {metrics_path}")
    return metadata


if __name__ == "__main__":
    run_tuning()
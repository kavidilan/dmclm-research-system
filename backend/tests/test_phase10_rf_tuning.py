"""Tests for leakage-aware Phase 10 Random Forest tuning and evaluation."""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.phase10_rf_tuning import (
    classification_metrics,
    majority_class_baseline,
    prepare_temporal_splits,
    tune_random_forest,
)
from src.phase10_ml_training import CORE_FIVE_FEATURES


def create_temporal_dataset() -> pd.DataFrame:
    rows = []
    split_dates = {
        "train": pd.date_range("2026-01-01", periods=6, freq="D"),
        "val": pd.date_range("2026-01-08", periods=6, freq="D"),
        "test": pd.date_range("2026-01-15", periods=4, freq="D"),
    }
    for split, dates in split_dates.items():
        for index, timestamp in enumerate(dates):
            label = "COMPRESS" if index % 2 else "ARCHIVE"
            row = {
                "memory_id": f"{split}_{index}",
                "split": split,
                "start_time": timestamp.isoformat(),
                "target_lifecycle_state": label,
            }
            row.update({feature: float((index + offset) % 5) / 5 for offset, feature in enumerate(CORE_FIVE_FEATURES)})
            rows.append(row)
    return pd.DataFrame(rows)


class TestPhase10RandomForestTuning(unittest.TestCase):
    def test_future_label_embargo_purges_boundaries_only(self):
        dataframe = create_temporal_dataset()
        train, validation, test, report = prepare_temporal_splits(dataframe, 48)

        self.assertEqual(len(train), 5)
        self.assertEqual(len(validation), 5)
        self.assertEqual(len(test), 4)
        self.assertEqual(report["purged_counts"], {"train": 1, "validation": 1})
        self.assertTrue((
            pd.to_datetime(train.start_time, utc=True) + pd.Timedelta(hours=48)
            < pd.to_datetime(dataframe.loc[dataframe.split.eq("val"), "start_time"], utc=True).min()
        ).all())
        self.assertTrue((
            pd.to_datetime(validation.start_time, utc=True) + pd.Timedelta(hours=48)
            < pd.to_datetime(dataframe.loc[dataframe.split.eq("test"), "start_time"], utc=True).min()
        ).all())
        self.assertListEqual(test.memory_id.tolist(), dataframe.loc[dataframe.split.eq("test"), "memory_id"].tolist())

    def test_tuning_uses_validation_metrics_and_records_all_candidates(self):
        dataframe = create_temporal_dataset()
        train, validation, _, _ = prepare_temporal_splits(dataframe, 48)
        parameters = [
            {"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
            {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 2, "max_features": 1.0, "class_weight": "balanced"},
        ]

        selected, report = tune_random_forest(train, validation, parameter_grid=parameters, n_jobs=1)

        self.assertIn(selected, parameters)
        self.assertEqual(report["candidate_count"], len(parameters))
        self.assertEqual(report["selection_metric"], "validation macro-F1, accuracy as tie-breaker")
        self.assertTrue(all("validation_metrics" in result for result in report["candidates"]))

    def test_metrics_include_per_class_confusion_and_ovr_auc(self):
        actual = np.array(["ARCHIVE", "COMPRESS", "FORGET", "RETAIN"] * 2)
        probabilities = np.eye(4)[[0, 1, 2, 3] * 2] * 0.8 + 0.05
        metrics = classification_metrics(actual, actual, probabilities=probabilities, probability_classes=("ARCHIVE", "COMPRESS", "FORGET", "RETAIN"))

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
        self.assertEqual(metrics["roc_auc_ovr_macro"], 1.0)
        self.assertEqual(metrics["per_class"]["ARCHIVE"]["support"], 2)

    def test_majority_baseline_chooses_only_from_training_labels(self):
        metrics = majority_class_baseline(
            ["RETAIN", "RETAIN", "COMPRESS"],
            ["RETAIN", "ARCHIVE", "COMPRESS"],
        )

        self.assertEqual(metrics["prediction"], "RETAIN")
        self.assertEqual(metrics["metrics"]["accuracy"], 0.3333)
        self.assertEqual(metrics["metrics"]["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
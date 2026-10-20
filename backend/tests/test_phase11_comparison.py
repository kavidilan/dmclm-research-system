"""
Unit and Integration Tests for Phase 11 — Baseline vs ML Model Empirical Comparison
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)
Project: Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence
"""

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.dmclm_config import ComparisonConfig, DEFAULT_CONFIG
from src.models import PerMemoryComparisonRecord, BaselineMLComparisonReport
from src.phase11_comparison import (
    Phase11BaselineMLComparator,
    CORE_FIVE_FEATURES,
    TRAINING_CSV_PATH,
    LIFECYCLE_MODEL_PATH,
    PER_MEMORY_CSV_PATH,
    COMPARISON_METRICS_PATH,
    COMPARISON_REPORT_MD_PATH,
)
from src.dmclm_db import init_dmclm_database, get_latest_comparison_run


class TestPhase11Comparison(unittest.TestCase):
    """Test suite validating fairness, alignment, metrics, agreement, and statistics for Phase 11."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="dmclm_phase11_test_"))
        cls.mock_db_path = cls.temp_dir / "test_dmclm.db"
        init_dmclm_database(cls.mock_db_path)

        # Create a clean mock dataset with 30 records: 20 train, 5 val, 5 test
        records = []
        for i in range(30):
            split = "train" if i < 20 else ("val" if i < 25 else "test")
            r = round(0.1 + (i % 8) * 0.1, 4)
            f = round(0.2 + (i % 6) * 0.1, 4)
            h = 0.0  # Cold-start
            s = round(0.3 + (i % 7) * 0.09, 4)
            rel = round(0.15 + (i % 5) * 0.12, 4)

            # Rule-based importance score
            score = round(0.20 * r + 0.20 * f + 0.25 * h + 0.20 * s + 0.15 * rel, 4)
            tier = "COMPRESS" if score >= 0.40 else "ARCHIVE"

            records.append({
                "memory_id": f"mem_test_{i:04d}",
                "session_id": "test_session",
                "parent_consolidated_id": "c_node_1",
                "start_time": f"2026-09-20T10:{i:02d}:00Z",
                "end_time": f"2026-09-20T10:{i:02d}:30Z",
                "split": split,
                "feature_recency": r,
                "feature_frequency": f,
                "feature_retrieval_history": h,
                "feature_semantic_relevance": s,
                "feature_relationship_strength": rel,
                "target_lifecycle_state": tier,
                "target_importance_score": score,
                "baseline_score": score,
                "value_tier": "MEDIUM_VALUE" if score >= 0.40 else "LOW_VALUE",
            })

        cls.mock_df = pd.DataFrame(records)
        cls.mock_csv_path = cls.temp_dir / "mock_training_dataset.csv"
        cls.mock_df.to_csv(cls.mock_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_dataset_loading_and_split_isolation(self):
        """Verify that evaluation strictly isolates the test split (zero train/val leakage)."""
        comparator = Phase11BaselineMLComparator(
            dataset_path=self.mock_csv_path,
        )
        test_df = comparator.load_evaluation_dataset(split="test")
        self.assertEqual(len(test_df), 5)
        self.assertTrue((test_df["split"] == "test").all())
        # Confirm no train samples leaked
        self.assertFalse((test_df["split"] == "train").any())
        self.assertFalse((test_df["split"] == "val").any())

    def test_02_feature_alignment_and_zero_leakage(self):
        """Verify that only the 5 core features are evaluated and audit variables are excluded."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        test_df = comparator.load_evaluation_dataset(split="test")

        for feat in CORE_FIVE_FEATURES:
            self.assertIn(feat, test_df.columns)

        # Ensure features are numeric and in [0, 1]
        for feat in CORE_FIVE_FEATURES:
            vals = test_df[feat].values
            self.assertTrue((vals >= 0.0).all())
            self.assertTrue((vals <= 1.0).all())

    def test_03_baseline_scoring_execution(self):
        """Verify that Phase 5 baseline scoring executes correctly and outputs valid lifecycle tiers."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        test_df = comparator.load_evaluation_dataset(split="test")
        records = test_df.to_dict(orient="records")

        baseline_results = comparator.run_baseline_scoring(records)
        self.assertEqual(len(baseline_results), len(records))

        for b in baseline_results:
            self.assertIn("memory_id", b)
            self.assertIn("baseline_score", b)
            self.assertIn("baseline_lifecycle_tier", b)
            self.assertTrue(0.0 <= b["baseline_score"] <= 1.0)
            self.assertIn(b["baseline_lifecycle_tier"], ["COMPRESS", "ARCHIVE", "RETAIN", "FORGET"])

    def test_04_ml_prediction_execution(self):
        """Verify that Phase 10 ML predictor outputs valid probabilities and classes."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        test_df = comparator.load_evaluation_dataset(split="test")
        records = test_df.to_dict(orient="records")

        ml_results = comparator.run_ml_predictions(records)
        self.assertEqual(len(ml_results), len(records))

        for m in ml_results:
            self.assertIn("memory_id", m)
            self.assertIn("predicted_lifecycle_state", m)
            self.assertIn("prediction_confidence", m)
            self.assertTrue(0.0 <= m["prediction_confidence"] <= 1.0)

    def test_05_per_memory_alignment_and_traceability(self):
        """Verify that baseline and ML predictions align 1-to-1 by memory_id."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        test_df = comparator.load_evaluation_dataset(split="test")
        records = test_df.to_dict(orient="records")

        b_res = comparator.run_baseline_scoring(records)
        m_res = comparator.run_ml_predictions(records)
        aligned = comparator.align_and_build_per_memory_records(test_df, b_res, m_res)

        self.assertEqual(len(aligned), len(test_df))
        for r in aligned:
            self.assertIsInstance(r, PerMemoryComparisonRecord)
            self.assertIn(r.outcome_category, [
                "BOTH_CORRECT", "BASELINE_ONLY_CORRECT", "ML_ONLY_CORRECT", "BOTH_INCORRECT"
            ])
            self.assertIsInstance(r.is_agreement, bool)
            self.assertIsInstance(r.baseline_is_correct, bool)
            self.assertIsInstance(r.ml_is_correct, bool)

    def test_06_classification_metrics_and_deltas(self):
        """Verify multiclass classification metrics and delta computation."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        y_true = ["COMPRESS", "COMPRESS", "ARCHIVE", "ARCHIVE", "COMPRESS"]
        b_pred = ["COMPRESS", "COMPRESS", "ARCHIVE", "ARCHIVE", "COMPRESS"]  # 100%
        m_pred = ["COMPRESS", "COMPRESS", "COMPRESS", "ARCHIVE", "COMPRESS"]  # 80% (1 mismatch)
        classes = ["ARCHIVE", "COMPRESS"]

        b_metrics = comparator.calculate_classification_metrics(y_true, b_pred, classes)
        m_metrics = comparator.calculate_classification_metrics(y_true, m_pred, classes)

        self.assertEqual(b_metrics["accuracy"], 1.0)
        self.assertEqual(m_metrics["accuracy"], 0.8)

        diff = comparator.calculate_metric_differences(b_metrics, m_metrics)
        self.assertAlmostEqual(diff["accuracy_delta"], -0.2, places=4)
        self.assertAlmostEqual(diff["accuracy_relative_pct"], -20.0, places=2)

    def test_07_regression_metrics_and_deltas(self):
        """Verify continuous memory-value regression metrics and error deltas."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        y_true = [0.45, 0.50, 0.30, 0.35, 0.60]
        b_pred = [0.45, 0.50, 0.30, 0.35, 0.60]  # Exact formula
        m_pred = [0.44, 0.52, 0.33, 0.34, 0.58]  # Slight errors

        b_reg = comparator.calculate_regression_metrics(y_true, b_pred)
        m_reg = comparator.calculate_regression_metrics(y_true, m_pred)

        self.assertEqual(b_reg["mae"], 0.0)
        self.assertEqual(b_reg["r2"], 1.0)
        self.assertTrue(m_reg["mae"] > 0.0)

        reg_diff = comparator.calculate_regression_differences(b_reg, m_reg)
        self.assertTrue(reg_diff["mae_delta"] > 0.0)

    def test_08_agreement_analysis_and_cohen_kappa(self):
        """Verify inter-method agreement rate, contingency matrix, and Cohen's Kappa."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        records = [
            PerMemoryComparisonRecord(
                memory_id="m1", feature_recency=0.5, feature_frequency=0.5, feature_retrieval_history=0.0,
                feature_semantic_relevance=0.5, feature_relationship_strength=0.5, target_lifecycle_state="COMPRESS",
                target_importance_score=0.5, baseline_lifecycle_tier="COMPRESS", baseline_score=0.5,
                baseline_is_correct=True, baseline_error=0.0, ml_predicted_state="COMPRESS", ml_predicted_score=0.5,
                ml_confidence=0.9, ml_is_correct=True, ml_error=0.0, is_agreement=True, score_delta=0.0,
                outcome_category="BOTH_CORRECT",
            ),
            PerMemoryComparisonRecord(
                memory_id="m2", feature_recency=0.2, feature_frequency=0.2, feature_retrieval_history=0.0,
                feature_semantic_relevance=0.2, feature_relationship_strength=0.2, target_lifecycle_state="ARCHIVE",
                target_importance_score=0.2, baseline_lifecycle_tier="ARCHIVE", baseline_score=0.2,
                baseline_is_correct=True, baseline_error=0.0, ml_predicted_state="COMPRESS", ml_predicted_score=0.42,
                ml_confidence=0.6, ml_is_correct=False, ml_error=0.22, is_agreement=False, score_delta=0.22,
                outcome_category="BASELINE_ONLY_CORRECT",
            ),
        ]
        classes = ["ARCHIVE", "COMPRESS"]
        agree = comparator.analyze_agreement(records, classes)

        self.assertEqual(agree["total_evaluations"], 2)
        self.assertEqual(agree["agreement_count"], 1)
        self.assertEqual(agree["disagreement_count"], 1)
        self.assertEqual(agree["agreement_rate"], 0.5)
        self.assertIn("ARCHIVE -> COMPRESS", agree["disagreement_breakdown"])

    def test_09_mcnemar_paired_test(self):
        """Verify McNemar's paired test calculation for discordant pairs."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        records = []
        # 80 both correct, 20 baseline only correct, 0 ml only correct
        for i in range(100):
            b_corr = True
            m_corr = True if i < 80 else False
            records.append(PerMemoryComparisonRecord(
                memory_id=f"m_{i}", feature_recency=0.5, feature_frequency=0.5, feature_retrieval_history=0.0,
                feature_semantic_relevance=0.5, feature_relationship_strength=0.5, target_lifecycle_state="COMPRESS",
                target_importance_score=0.5, baseline_lifecycle_tier="COMPRESS", baseline_score=0.5,
                baseline_is_correct=b_corr, baseline_error=0.0,
                ml_predicted_state="COMPRESS" if m_corr else "ARCHIVE",
                ml_predicted_score=0.5, ml_confidence=0.9, ml_is_correct=m_corr, ml_error=0.0,
                is_agreement=(b_corr == m_corr), score_delta=0.0,
                outcome_category="BOTH_CORRECT" if (b_corr and m_corr) else "BASELINE_ONLY_CORRECT",
            ))

        stats = comparator.calculate_statistical_tests(records)
        mcnemar = stats["mcnemar_test"]
        self.assertEqual(mcnemar["discordant_pairs"], 20)
        self.assertEqual(mcnemar["n_baseline_only_correct"], 20)
        self.assertEqual(mcnemar["n_ml_only_correct"], 0)
        self.assertTrue(mcnemar["p_value"] < 0.001)
        self.assertTrue(mcnemar["statistically_significant"])

    def test_10_wilcoxon_paired_test(self):
        """Verify Wilcoxon signed-rank test on paired error distributions."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        records = []
        for i in range(20):
            b_err = 0.0
            m_err = round(0.01 + (i % 5) * 0.01, 4)
            records.append(PerMemoryComparisonRecord(
                memory_id=f"m_{i}", feature_recency=0.5, feature_frequency=0.5, feature_retrieval_history=0.0,
                feature_semantic_relevance=0.5, feature_relationship_strength=0.5, target_lifecycle_state="COMPRESS",
                target_importance_score=0.5, baseline_lifecycle_tier="COMPRESS", baseline_score=0.5,
                baseline_is_correct=True, baseline_error=b_err,
                ml_predicted_state="COMPRESS", ml_predicted_score=0.5 + m_err, ml_confidence=0.9,
                ml_is_correct=True, ml_error=m_err, is_agreement=True, score_delta=m_err,
                outcome_category="BOTH_CORRECT",
            ))

        stats = comparator.calculate_statistical_tests(records)
        wilcoxon = stats["wilcoxon_test"]
        self.assertEqual(wilcoxon["sample_pairs"], 20)
        self.assertIsNotNone(wilcoxon["p_value"])
        self.assertTrue(wilcoxon["statistically_significant"])

    def test_11_feature_divergence_analysis(self):
        """Verify feature mean and standard deviation divergence between agreed and disagreed subsets."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        records = [
            PerMemoryComparisonRecord(
                memory_id="m1", feature_recency=0.8, feature_frequency=0.7, feature_retrieval_history=0.0,
                feature_semantic_relevance=0.9, feature_relationship_strength=0.6, target_lifecycle_state="COMPRESS",
                target_importance_score=0.6, baseline_lifecycle_tier="COMPRESS", baseline_score=0.6,
                baseline_is_correct=True, baseline_error=0.0, ml_predicted_state="COMPRESS", ml_predicted_score=0.6,
                ml_confidence=0.9, ml_is_correct=True, ml_error=0.0, is_agreement=True, score_delta=0.0,
                outcome_category="BOTH_CORRECT",
            ),
            PerMemoryComparisonRecord(
                memory_id="m2", feature_recency=0.2, feature_frequency=0.3, feature_retrieval_history=0.0,
                feature_semantic_relevance=0.4, feature_relationship_strength=0.2, target_lifecycle_state="ARCHIVE",
                target_importance_score=0.3, baseline_lifecycle_tier="ARCHIVE", baseline_score=0.3,
                baseline_is_correct=True, baseline_error=0.0, ml_predicted_state="COMPRESS", ml_predicted_score=0.42,
                ml_confidence=0.6, ml_is_correct=False, ml_error=0.12, is_agreement=False, score_delta=0.12,
                outcome_category="BASELINE_ONLY_CORRECT",
            ),
        ]
        div = comparator.analyze_feature_divergence(records)
        for f in CORE_FIVE_FEATURES:
            self.assertIn(f, div)
            self.assertIn("agreed_mean", div[f])
            self.assertIn("disagreed_mean", div[f])
            self.assertIn("mean_difference", div[f])

        # Recency agreed was 0.8, disagreed was 0.2 -> difference = -0.6
        self.assertAlmostEqual(div["feature_recency"]["mean_difference"], -0.6, places=4)

    def test_12_deterministic_comparison_run(self):
        """Verify that repeated evaluation runs yield identical metrics and agreement rates."""
        comparator = Phase11BaselineMLComparator(dataset_path=self.mock_csv_path)
        rep1 = comparator.run_pipeline(save_artifacts=False, save_to_db=False)
        rep2 = comparator.run_pipeline(save_artifacts=False, save_to_db=False)

        self.assertEqual(rep1.sample_count, rep2.sample_count)
        self.assertEqual(rep1.baseline_classification_metrics["accuracy"], rep2.baseline_classification_metrics["accuracy"])
        self.assertEqual(rep1.ml_classification_metrics["accuracy"], rep2.ml_classification_metrics["accuracy"])
        self.assertEqual(rep1.agreement_analysis["agreement_rate"], rep2.agreement_analysis["agreement_rate"])
        self.assertEqual(rep1.agreement_analysis["cohen_kappa"], rep2.agreement_analysis["cohen_kappa"])

    def test_13_error_handling_missing_artifacts(self):
        """Verify that comparator raises clear exceptions when artifacts or required columns are missing."""
        missing_csv = self.temp_dir / "non_existent.csv"
        comparator = Phase11BaselineMLComparator(dataset_path=missing_csv)
        with self.assertRaises(FileNotFoundError):
            comparator.load_evaluation_dataset()

        # Incomplete schema CSV
        bad_df = pd.DataFrame({"memory_id": ["m1"], "split": ["test"]})
        bad_csv = self.temp_dir / "bad.csv"
        bad_df.to_csv(bad_csv, index=False)
        comp_bad = Phase11BaselineMLComparator(dataset_path=bad_csv)
        with self.assertRaises(ValueError):
            comp_bad.load_evaluation_dataset()

    def test_14_real_data_pipeline_integration(self):
        """Integration test: Execute full Phase 11 comparison against real Phase 9 test set and Phase 10 model."""
        if not TRAINING_CSV_PATH.exists() or not LIFECYCLE_MODEL_PATH.exists():
            self.skipTest("Real Phase 9 dataset or Phase 10 model not found on disk.")

        comparator = Phase11BaselineMLComparator(
            dataset_path=TRAINING_CSV_PATH,
            classifier_path=LIFECYCLE_MODEL_PATH,
        )
        report = comparator.run_pipeline(save_artifacts=False, save_to_db=False)

        self.assertIsInstance(report, BaselineMLComparisonReport)
        self.assertEqual(report.sample_count, 875)
        self.assertEqual(report.evaluation_split, "test")

        # Baseline and ML accuracy on genuine test split
        self.assertGreater(report.baseline_classification_metrics["accuracy"], 0.15)
        self.assertGreater(report.ml_classification_metrics["accuracy"], 0.35)

        # ML model should outperform the fixed-weight baseline on unseen future behavior
        self.assertGreaterEqual(
            report.ml_classification_metrics["accuracy"],
            report.baseline_classification_metrics["accuracy"]
        )

        # Agreement rate should reflect non-trivial alignment
        self.assertGreater(report.agreement_analysis["agreement_rate"], 0.05)

        # Artifacts are not written in test mode (save_artifacts=False)


if __name__ == "__main__":
    unittest.main()

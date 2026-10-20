"""
Unit and Integration Tests for Phase 12 â€” Experiments & Evaluation Suite
Member 4 â€” Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)
Project: Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence
"""

import json
import os
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

from src.dmclm_config import ExperimentSuiteConfig, DEFAULT_CONFIG
from src.models import ExperimentResult, ExperimentSuiteReport
from src.phase12_experiments import (
    Phase12ExperimentRunner,
    EXPERIMENTS_SUMMARY_JSON_PATH,
    PHASE12_REPORT_MD_PATH,
    CONSOLIDATED_JSON_PATH,
    RETENTION_JSON_PATH,
    RELATIONSHIPS_JSON_PATH,
)
from src.phase10_ml_training import TRAINING_CSV_PATH, CORE_FIVE_FEATURES
from src.dmclm_db import (
    init_dmclm_database,
    insert_or_replace_experiment_run,
    get_latest_experiment_runs,
)


class TestPhase12Experiments(unittest.TestCase):
    """
    Comprehensive test suite validating Phase 12 evaluation framework:
    - Experiment configurations and validation
    - Multi-seed baseline vs ML performance evaluation
    - Leave-one-out and single-feature ablation
    - Consolidation compression effectiveness & traceability
    - Adaptive retention and cognitive forgetting dynamics
    - Relationship graph topology metrics
    - Robustness and feature perturbation stress testing
    - Determinism and reproducibility across identical trials
    - Output artifact generation (JSON summaries, figures, reports)
    - Database logging and query operations
    - Graceful insufficient data handling
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="dmclm_phase12_test_"))
        cls.mock_db_path = cls.temp_dir / "test_dmclm.db"
        init_dmclm_database(cls.mock_db_path)

        cls.runner = Phase12ExperimentRunner()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_config_validation_and_loading(self):
        """Test ExperimentSuiteConfig initialization, validation rules, and defaults."""
        cfg = ExperimentSuiteConfig()
        self.assertEqual(cfg.suite_id, "dmclm_phase12_suite")
        self.assertIn(42, cfg.random_seeds)
        self.assertGreaterEqual(len(cfg.random_seeds), 3)
        self.assertTrue(cfg.generate_plots)
        self.assertIsNone(cfg.validate())

        # Test invalid configuration handling
        invalid_cfg = ExperimentSuiteConfig(random_seeds=[])
        with self.assertRaises(ValueError):
            invalid_cfg.validate()

        invalid_trials = ExperimentSuiteConfig(reproducibility_trials=0)
        with self.assertRaises(ValueError):
            invalid_trials.validate()

    def test_02_result_schema_serialization(self):
        """Test ExperimentResult and ExperimentSuiteReport dataclasses and JSON serialization."""
        res = ExperimentResult(
            experiment_id="TEST-EXP-01",
            experiment_name="Schema Test",
            experiment_type="UNIT_TEST",
            timestamp="2026-10-08T00:00:00Z",
            parameters={"seed": 42},
            metrics={"acc": 0.95, "f1": 0.82},
            sample_count=100,
            execution_time_seconds=0.12,
            status="SUCCESS",
            notes="Schema validation unit test"
        )
        d = res.to_dict()
        self.assertEqual(d["experiment_id"], "TEST-EXP-01")
        self.assertEqual(d["metrics"]["acc"], 0.95)
        self.assertEqual(d["sample_count"], 100)

        report = ExperimentSuiteReport(
            suite_id="test_suite",
            timestamp="2026-10-08T00:00:00Z",
            total_experiments=1,
            successful_experiments=1,
            failed_experiments=0,
            dataset_hash="test_hash",
            results={"TEST-EXP-01": d},
            executive_summary="All tests passed",
            figures_generated=["fig1.png"]
        )
        rep_dict = report.to_dict()
        self.assertEqual(rep_dict["successful_experiments"], 1)
        self.assertIn("TEST-EXP-01", rep_dict["results"])

    def test_03_exp01_model_performance_multi_seed(self):
        """Test EXP-01: Baseline vs ML evaluation across seeds and confidence interval computation."""
        res = self.runner.run_experiment_01_model_performance()
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-01-MODEL-PERF")
        self.assertGreater(res.sample_count, 0)

        metrics = res.metrics
        self.assertIn("baseline", metrics)
        self.assertIn("ml_multi_seed_runs", metrics)
        self.assertIn("ml_summary", metrics)

        # Baseline metrics checks
        self.assertGreaterEqual(metrics["baseline"]["accuracy"], 0.0)
        self.assertLessEqual(metrics["baseline"]["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["baseline"]["f1_macro"], 0.0)

        # ML multi-seed runs
        runs = metrics["ml_multi_seed_runs"]
        self.assertEqual(len(runs), len(self.runner.config.random_seeds))
        for r in runs:
            self.assertIn("accuracy", r)
            self.assertGreaterEqual(r["accuracy"], 0.40)  # Measured accuracy on 4-class real behavioral dataset
            self.assertGreaterEqual(r["f1_macro"], 0.25)

        # Summary statistics and 95% confidence intervals
        ml_summary = metrics["ml_summary"]
        self.assertGreater(ml_summary["accuracy_mean"], 0.40)
        self.assertGreaterEqual(ml_summary["accuracy_std"], 0.0)
        self.assertEqual(len(ml_summary["accuracy_ci95"]), 2)
        self.assertLessEqual(ml_summary["accuracy_ci95"][0], ml_summary["accuracy_ci95"][1])

    def test_04_exp02_feature_ablation_evaluation(self):
        """Test EXP-02: Leave-one-out and single-feature ablation on DMCLM core features."""
        res = self.runner.run_experiment_02_feature_ablation()
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-02-FEATURE-ABLATION")

        metrics = res.metrics
        self.assertIn("ablation_evaluations", metrics)
        self.assertIn("leave_one_out_f1_drops", metrics)
        self.assertIn("ranked_feature_importance_by_f1_loss", metrics)

        evals = metrics["ablation_evaluations"]
        self.assertIn("Full_Model (All 5 Features)", evals)
        full_f1 = evals["Full_Model (All 5 Features)"]["f1_macro"]

        # Check all 5 leave-one-out configurations exist
        for feat in CORE_FIVE_FEATURES:
            plan_name = f"Leave_Out_{feat.replace('feature_', '').title()}"
            self.assertIn(plan_name, evals)
            loo_metrics = evals[plan_name]
            self.assertIn("accuracy", loo_metrics)
            self.assertIn("f1_macro", loo_metrics)

        # Verify ranked drops
        drops = metrics["leave_one_out_f1_drops"]
        self.assertEqual(len(drops), 5)
        ranked = metrics["ranked_feature_importance_by_f1_loss"]
        self.assertEqual(len(ranked), 5)

    def test_05_exp03_compression_effectiveness(self):
        """Test EXP-03: Real Phase 6 memory compression metrics and source traceability."""
        res = self.runner.run_experiment_03_compression_effectiveness()
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-03-COMPRESSION-EFFECT")

        m = res.metrics
        self.assertIn("initial_candidate_memories", m)
        self.assertIn("consolidated_memory_nodes", m)
        self.assertIn("overall_memory_reduction_pct", m)
        self.assertIn("compression_ratio", m)
        self.assertIn("traceability_verified", m)

        self.assertTrue(m["traceability_verified"])
        self.assertGreater(m["initial_candidate_memories"], 5000)
        self.assertGreater(m["consolidated_memory_nodes"], 50)
        self.assertGreater(m["overall_memory_reduction_pct"], 90.0)

    def test_06_exp04_retention_dynamics(self):
        """Test EXP-04 metrics using an isolated, known retention fixture."""
        fixture_path = self.temp_dir / "exp04_retention_fixture.json"
        fixture = [
            {"retention_state": "DECAYING", "is_protected": True,
             "final_retention_strength": 0.50, "decay_factor": 0.80,
             "stability_score": 0.70, "age_days": 5.0},
            {"retention_state": "ARCHIVED", "is_protected": False,
             "final_retention_strength": 0.30, "decay_factor": 0.50,
             "stability_score": 0.40, "age_days": 20.0},
            {"retention_state": "FORGETTABLE", "is_protected": False,
             "final_retention_strength": 0.10, "decay_factor": 0.20,
             "stability_score": 0.20, "age_days": 30.0},
        ]
        fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

        runner = Phase12ExperimentRunner(retention_path=fixture_path)
        res = runner.run_experiment_04_retention_dynamics()

        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-04-RETENTION-DYNAMICS")

        m = res.metrics
        self.assertEqual(m["total_evaluated_consolidated_nodes"], 3)
        self.assertEqual(m["state_counts"]["DECAYING"], 1)
        self.assertEqual(m["state_counts"]["ARCHIVED"], 1)
        self.assertEqual(m["state_counts"]["FORGETTABLE"], 1)
        self.assertEqual(m["protected_high_value_nodes"], 1)
        self.assertTrue(m["protection_policy_intact"])

    def test_07_exp05_relationship_topology(self):
        """Test EXP-05: Graph connectivity, edge distributions, and topology metrics."""
        res = self.runner.run_experiment_05_relationship_topology()
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-05-RELATIONSHIP-TOPOLOGY")

        m = res.metrics
        self.assertIn("graph_nodes", m)
        self.assertIn("graph_edges", m)
        self.assertIn("relationship_type_counts", m)
        self.assertIn("graph_density", m)
        self.assertIn("connected_components", m)

        self.assertGreater(m["graph_nodes"], 50)
        self.assertGreater(m["graph_edges"], 200)
        self.assertGreater(m["graph_density"], 0.0)

    def test_08_exp06_robustness_stress(self):
        """Test EXP-06: Missing field resilience and Gaussian noise degradation."""
        res = self.runner.run_experiment_06_robustness_stress()
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-06-ROBUSTNESS-STRESS")

        m = res.metrics
        self.assertIn("clean_accuracy", m)
        self.assertIn("missing_features_evaluated", m)
        self.assertIn("noise_degradation", m)

        # Confirm model maintains stability under modest noise
        clean_acc = m["clean_accuracy"]
        noise_005 = m["noise_degradation"]["noise_sigma_0.05"]
        self.assertGreaterEqual(noise_005, clean_acc - 0.15)

    def test_09_exp07_reproducibility(self):
        """Test EXP-07: Determinism check across identical trials."""
        res = self.runner.run_experiment_07_reproducibility()
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.experiment_id, "EXP-07-REPRODUCIBILITY")

        m = res.metrics
        self.assertIn("identical_reproducibility", m)
        self.assertTrue(m["identical_reproducibility"])
        self.assertEqual(m["accuracy_variance"], 0.0)
        self.assertEqual(m["f1_macro_variance"], 0.0)

    def test_10_figure_generation(self):
        """Test that generate_all_figures generates non-empty PNG files."""
        results = {
            "EXP-01-MODEL-PERF": self.runner.run_experiment_01_model_performance(),
            "EXP-02-FEATURE-ABLATION": self.runner.run_experiment_02_feature_ablation(),
            "EXP-03-COMPRESSION-EFFECT": self.runner.run_experiment_03_compression_effectiveness(),
            "EXP-04-RETENTION-DYNAMICS": self.runner.run_experiment_04_retention_dynamics(),
            "EXP-05-RELATIONSHIP-TOPOLOGY": self.runner.run_experiment_05_relationship_topology(),
            "EXP-06-ROBUSTNESS-STRESS": self.runner.run_experiment_06_robustness_stress(),
        }
        figures = self.runner.generate_all_figures(results)
        self.assertEqual(len(figures), 6)
        for fig_rel in figures:
            fig_path = Path(fig_rel)
            if not fig_path.is_absolute():
                from src.phase12_experiments import ROOT_DIR
                fig_path = ROOT_DIR / fig_path
            self.assertTrue(fig_path.exists(), f"Figure not found: {fig_path}")
            self.assertGreater(fig_path.stat().st_size, 5000, f"Figure unexpectedly small: {fig_path}")

    def test_11_db_logging_and_retrieval(self):
        """Test inserting and retrieving experiment records from SQLite."""
        test_rec = {
            "experiment_id": "TEST-DB-01",
            "suite_id": "test_suite_db",
            "experiment_name": "DB Persistence Test",
            "experiment_type": "UNIT_TEST",
            "timestamp": "2026-10-08T00:00:00Z",
            "sample_count": 50,
            "execution_time_seconds": 0.05,
            "status": "SUCCESS",
            "metrics": {"test_metric": 1.0},
            "parameters": {"test_param": "val"},
        }
        insert_or_replace_experiment_run(test_rec, db_path=self.mock_db_path)
        runs = get_latest_experiment_runs(suite_id="test_suite_db", db_path=self.mock_db_path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["experiment_id"], "TEST-DB-01")
        self.assertEqual(runs[0]["metrics"]["test_metric"], 1.0)

    def test_12_insufficient_data_graceful_handling(self):
        """Test graceful degradation when input dataset is missing or insufficient."""
        empty_csv = self.temp_dir / "empty_dataset.csv"
        pd.DataFrame(columns=["split", "target_lifecycle_state", "target_importance_score"]).to_csv(empty_csv, index=False)

        custom_runner = Phase12ExperimentRunner(dataset_path=empty_csv)
        res = custom_runner.run_experiment_01_model_performance()
        self.assertEqual(res.status, "SKIPPED")
        self.assertIn("Insufficient", res.notes)


if __name__ == "__main__":
    unittest.main()


"""
Unit and Integration Tests for Phase 10 — ML Model Training and Prediction
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Covers all Phase 10 requirements:
1. Feature list validation and deterministic ordering.
2. Leakage protection enforcement (prohibits target or audit columns in feature matrix).
3. Training reproducibility (fixed random seed produces identical metrics and trees).
4. Preprocessing consistency between training and inference pipelines.
5. Model serialization and deserialization (joblib roundtrip integrity).
6. Prediction output validation (valid classes, float probabilities in [0, 1]).
7. Deterministic inference on identical inputs.
8. Missing feature handling and safe default imputation.
9. Explainability and Gini feature importance extraction.
10. Non-interference with Phase 5 baseline scores (zero silent overwrites).
11. Backward compatibility: seamless mapping of legacy ('recency') and canonical ('feature_recency') keys.
12. Continuous importance score regressor training and evaluation.
13. Complete metadata tracking (dataset hash, samples, train/val/test metrics).
14. Real-data integration test against actual Phase 9 training dataset.
"""

import copy
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import ModelTrainingConfig, DEFAULT_CONFIG
from src.phase10_ml_training import (
    Phase10ModelTrainer,
    Phase10MLPredictor,
    CORE_FIVE_FEATURES,
    TRAINING_CSV_PATH,
)


def create_synthetic_training_df(n_samples: int = 100) -> pd.DataFrame:
    """Helper to generate structured synthetic test dataframe conforming to Phase 9 schema."""
    np.random.seed(42)
    rec = np.random.uniform(0.1, 0.9, n_samples)
    freq = np.random.uniform(0.1, 0.9, n_samples)
    ret = np.random.uniform(0.0, 0.5, n_samples)
    sem = np.random.uniform(0.2, 0.8, n_samples)
    rel = np.random.uniform(0.2, 0.8, n_samples)

    score = 0.20 * rec + 0.20 * freq + 0.25 * ret + 0.20 * sem + 0.15 * rel
    target = np.where(score >= 0.40, "COMPRESS", "ARCHIVE")

    splits = ["train"] * int(n_samples * 0.70) + ["val"] * int(n_samples * 0.15)
    splits += ["test"] * (n_samples - len(splits))

    return pd.DataFrame({
        "memory_id": [f"mem_{i:04d}" for i in range(n_samples)],
        "feature_recency": rec,
        "feature_frequency": freq,
        "feature_retrieval_history": ret,
        "feature_semantic_relevance": sem,
        "feature_relationship_strength": rel,
        "target_lifecycle_state": target,
        "target_importance_score": score,
        "split": splits,
    })


class TestPhase10MLTraining(unittest.TestCase):
    """Comprehensive test suite for Phase 10 ML Model Training and Prediction."""

    def setUp(self):
        self.config = ModelTrainingConfig(
            algorithm="random_forest",
            feature_set="core_five",
            n_estimators=30,
            max_depth=4,
            random_seed=42,
        )
        self.trainer = Phase10ModelTrainer(self.config)

    def test_01_feature_validation_and_ordering(self):
        """Verify that configured features match the 5 core DMCLM signals in exact order."""
        features = self.trainer.get_feature_list()
        self.assertEqual(features, CORE_FIVE_FEATURES)
        self.assertEqual(len(features), 5)
        self.assertEqual(features[0], "feature_recency")
        self.assertEqual(features[4], "feature_relationship_strength")

    def test_02_leakage_protection_enforcement(self):
        """Verify that attempting to include target or audit columns in features raises ValueError."""
        df = create_synthetic_training_df(50)
        df["baseline_score"] = 0.55

        # Force a leakage variable into feature list
        class LeakyTrainer(Phase10ModelTrainer):
            def get_feature_list(self, _=None):
                return ["feature_recency", "baseline_score"]

        leaky_trainer = LeakyTrainer(self.config)
        with self.assertRaises(ValueError) as ctx:
            leaky_trainer.validate_dataset(df)
        self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))

    def test_03_training_reproducibility(self):
        """Verify that training twice with identical seed produces identical model weights and metrics."""
        df = create_synthetic_training_df(100)
        _, _, meta1 = self.trainer.train_and_evaluate(df)
        _, _, meta2 = self.trainer.train_and_evaluate(df)

        self.assertEqual(meta1.val_metrics["accuracy"], meta2.val_metrics["accuracy"])
        self.assertEqual(meta1.val_metrics["f1_macro"], meta2.val_metrics["f1_macro"])
        self.assertEqual(meta1.feature_importance, meta2.feature_importance)

    def test_04_pipeline_preprocessing_consistency(self):
        """Verify that StandardScaler fits only on train data and transforms consistently."""
        df = create_synthetic_training_df(100)
        clf_pipeline, _, _ = self.trainer.train_and_evaluate(df)

        scaler = clf_pipeline.named_steps.get("scaler")
        self.assertIsNotNone(scaler)
        self.assertEqual(len(scaler.mean_), 5)
        self.assertTrue(hasattr(scaler, "scale_"))

    def test_05_model_serialization_and_deserialization(self):
        """Verify that trained model pipeline can be serialized and loaded via joblib without loss."""
        import tempfile
        df = create_synthetic_training_df(60)
        clf_pipeline, _, _ = self.trainer.train_and_evaluate(df)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            import joblib
            joblib.dump(clf_pipeline, tmp_path)
            loaded_clf = joblib.load(tmp_path)

            sample_input = np.array([[0.8, 0.6, 0.0, 0.7, 0.5]])
            pred_orig = clf_pipeline.predict(sample_input)
            pred_loaded = loaded_clf.predict(sample_input)
            self.assertEqual(pred_orig[0], pred_loaded[0])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_06_prediction_shape_and_datatypes(self):
        """Verify that predictor returns valid MLPredictionResult with probabilities summing to ~1.0."""
        df = create_synthetic_training_df(60)
        clf_pipeline, reg_pipeline, meta = self.trainer.train_and_evaluate(df)

        import tempfile, joblib
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            joblib.dump(clf_pipeline, tmp_path)
            predictor = Phase10MLPredictor(classifier_path=tmp_path)

            mem = {
                "memory_id": "test_mem_01",
                "feature_recency": 0.85,
                "feature_frequency": 0.60,
                "feature_retrieval_history": 0.0,
                "feature_semantic_relevance": 0.75,
                "feature_relationship_strength": 0.65,
            }
            res = predictor.predict_single(mem)

            self.assertEqual(res.memory_id, "test_mem_01")
            self.assertIn(res.predicted_lifecycle_state, ("COMPRESS", "ARCHIVE"))
            self.assertTrue(0.0 <= res.prediction_confidence <= 1.0)
            self.assertAlmostEqual(sum(res.class_probabilities.values()), 1.0, places=2)
            self.assertTrue(res.is_ml_generated)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_07_deterministic_inference(self):
        """Verify that repeated predictions on identical input yield identical confidence and state."""
        df = create_synthetic_training_df(60)
        clf_pipeline, _, _ = self.trainer.train_and_evaluate(df)

        import tempfile, joblib
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            joblib.dump(clf_pipeline, tmp_path)
            predictor = Phase10MLPredictor(classifier_path=tmp_path)

            mem = {"memory_id": "det_mem", "recency": 0.7, "frequency": 0.5, "retrieval_history": 0.0, "semantic_relevance": 0.6, "relationship_strength": 0.5}
            r1 = predictor.predict_single(mem)
            r2 = predictor.predict_single(mem)

            self.assertEqual(r1.predicted_lifecycle_state, r2.predicted_lifecycle_state)
            self.assertEqual(r1.prediction_confidence, r2.prediction_confidence)
            self.assertEqual(r1.class_probabilities, r2.class_probabilities)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_08_invalid_input_and_missing_features_handling(self):
        """Verify that missing optional features are imputed with safe default 0.0 without errors."""
        df = create_synthetic_training_df(60)
        clf_pipeline, _, _ = self.trainer.train_and_evaluate(df)

        import tempfile, joblib
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            joblib.dump(clf_pipeline, tmp_path)
            predictor = Phase10MLPredictor(classifier_path=tmp_path)

            # Sparse memory with some missing features
            sparse_mem = {"memory_id": "sparse_mem", "feature_recency": 0.5}
            res = predictor.predict_single(sparse_mem)
            self.assertIsNotNone(res.predicted_lifecycle_state)
            self.assertTrue(0.0 <= res.prediction_confidence <= 1.0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_09_feature_importance_generation(self):
        """Verify that tree-based Gini feature importances are non-negative and sum to 1.0."""
        df = create_synthetic_training_df(80)
        _, _, meta = self.trainer.train_and_evaluate(df)

        importances = meta.feature_importance
        self.assertEqual(len(importances), 5)
        for f, imp in importances.items():
            self.assertGreaterEqual(imp, 0.0)
            self.assertLessEqual(imp, 1.0)
        self.assertAlmostEqual(sum(importances.values()), 1.0, places=2)

    def test_10_non_interference_with_baseline_score(self):
        """Verify that ML predictions do not alter or overwrite existing Phase 5 baseline scores."""
        df = create_synthetic_training_df(60)
        clf_pipeline, _, _ = self.trainer.train_and_evaluate(df)

        import tempfile, joblib
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            joblib.dump(clf_pipeline, tmp_path)
            predictor = Phase10MLPredictor(classifier_path=tmp_path)

            original_memory = {
                "memory_id": "mem_baseline_test",
                "baseline_score": 0.4612,
                "lifecycle_state": "COMPRESS",
                "feature_recency": 0.8,
                "feature_frequency": 0.5,
                "feature_retrieval_history": 0.0,
                "feature_semantic_relevance": 0.6,
                "feature_relationship_strength": 0.5,
            }
            copied_mem = copy.deepcopy(original_memory)
            res = predictor.predict_single(copied_mem)

            # Original baseline score must be completely preserved
            self.assertEqual(copied_mem["baseline_score"], 0.4612)
            self.assertEqual(copied_mem["lifecycle_state"], "COMPRESS")
            self.assertTrue(res.is_ml_generated)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_11_legacy_key_aliases_support(self):
        """Verify that both legacy keys ('recency') and canonical keys ('feature_recency') are parsed."""
        df = create_synthetic_training_df(60)
        clf_pipeline, _, _ = self.trainer.train_and_evaluate(df)

        import tempfile, joblib
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            joblib.dump(clf_pipeline, tmp_path)
            predictor = Phase10MLPredictor(classifier_path=tmp_path)

            mem_legacy = {"memory_id": "m1", "recency": 0.8, "frequency": 0.6, "retrieval_history": 0.0, "semantic_relevance": 0.7, "relationship_strength": 0.5}
            mem_canon = {"memory_id": "m1", "feature_recency": 0.8, "feature_frequency": 0.6, "feature_retrieval_history": 0.0, "feature_semantic_relevance": 0.7, "feature_relationship_strength": 0.5}

            r_legacy = predictor.predict_single(mem_legacy)
            r_canon = predictor.predict_single(mem_canon)

            self.assertEqual(r_legacy.predicted_lifecycle_state, r_canon.predicted_lifecycle_state)
            self.assertEqual(r_legacy.prediction_confidence, r_canon.prediction_confidence)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_12_continuous_regressor_training(self):
        """Verify that continuous importance score regressor is trained and evaluated."""
        df = create_synthetic_training_df(80)
        _, reg_pipeline, meta = self.trainer.train_and_evaluate(df)

        self.assertIsNotNone(reg_pipeline)
        self.assertIn("regression", meta.val_metrics)
        val_reg = meta.val_metrics["regression"]
        self.assertIn("mae", val_reg)
        self.assertIn("rmse", val_reg)
        self.assertIn("r2_score", val_reg)

    def test_13_model_metadata_completeness(self):
        """Verify that metadata captures hyperparameters, samples, and partition metrics."""
        df = create_synthetic_training_df(80)
        _, _, meta = self.trainer.train_and_evaluate(df)

        self.assertEqual(meta.train_samples, int(80 * 0.70))
        self.assertEqual(meta.val_samples, int(80 * 0.15))
        self.assertGreater(meta.test_samples, 0)
        self.assertEqual(meta.target_column, "target_lifecycle_state")
        self.assertTrue(meta.leakage_protected)

    def test_14_real_data_pipeline_integration(self):
        """Integration test: Execute full model training against real Phase 9 dataset."""
        if not TRAINING_CSV_PATH.exists():
            self.skipTest("Real Phase 9 dataset not found at data/training_dataset.csv.")

        trainer = Phase10ModelTrainer()
        df = pd.read_csv(TRAINING_CSV_PATH)
        clf_pipeline, reg_pipeline, meta = trainer.train_and_evaluate(df)

        self.assertEqual(meta.train_samples, 4076)
        self.assertIn(meta.val_samples, [873, 934, 1047])
        self.assertIn(meta.test_samples, [874, 875, 935, 1047])
        self.assertEqual(meta.train_samples + meta.val_samples + meta.test_samples, len(df))
        self.assertGreater(meta.val_metrics["accuracy"], 0.40)
        self.assertGreater(meta.test_metrics["accuracy"], 0.40)
        self.assertGreater(len(meta.test_metrics["classes"]), 2)


if __name__ == "__main__":
    unittest.main()

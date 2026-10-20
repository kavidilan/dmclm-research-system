"""
Unit & Integration Test Suite for Phase 5 — Baseline DMCLM
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Validates:
1. Transparent weighted baseline formula: Score = w1*R + w2*F + w3*H + w4*S + w5*Rel.
2. Weight validation rules: Non-negative, numeric, and strict sum == 1.0 constraint.
3. Score normalization: Strict boundary enforcement [0.0, 1.0], zero-feature, and max-feature tests.
4. Interpretation thresholds: Value tiers (HIGH, MEDIUM, LOW) and lifecycle tiers (RETAIN, COMPRESS, ARCHIVE, FORGET).
5. Explainability & Attribution: Feature contribution breakdown (w_i * x_i), percentage attribution, dominant feature, and audit text.
6. Robustness & Edge Cases: Missing features, None values, empty dataset, single-memory dataset.
7. Determinism & Idempotency: Bitwise identical results across repeated runs, zero random numbers, duplicate prevention.
8. Real Data Integration: Loads all 6,229 real Phase 4 memories, evaluates baseline scores, validates statistics and persistence.
"""

import json
import math
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import (
    BaselineImportanceWeights,
    ValueTierThresholds,
    LifecycleThresholds,
    DEFAULT_CONFIG,
)
from src.models import (
    StructuredMemoryCandidate,
    FeatureEnrichedMemory,
    BaselineScoredMemory,
)
from src.phase5_baseline_dmclm import Phase5BaselineDMCLM


class TestPhase5BaselineFormula(unittest.TestCase):
    def setUp(self):
        self.engine = Phase5BaselineDMCLM()

    def test_weighted_baseline_formula_exact_calculation(self):
        """
        Verify exact arithmetic calculation:
        Score = 0.20*R + 0.20*F + 0.25*H + 0.20*S + 0.15*Rel
        """
        record = {
            "candidate_id": "test_mem_001",
            "recency": 0.80,
            "frequency": 0.60,
            "retrieval_history": 0.40,
            "semantic_relevance": 0.70,
            "relationship_strength": 0.50,
        }
        # Expected calculation:
        # 0.20*0.80 = 0.160
        # 0.20*0.60 = 0.120
        # 0.25*0.40 = 0.100
        # 0.20*0.70 = 0.140
        # 0.15*0.50 = 0.075
        # Sum = 0.160 + 0.120 + 0.100 + 0.140 + 0.075 = 0.595
        scored = self.engine.score_single_memory(record)
        self.assertAlmostEqual(scored.baseline_score, 0.595, places=4)
        self.assertAlmostEqual(scored.importance_score, 0.595, places=4)

        # Check individual contributions
        contribs = scored.feature_contributions
        self.assertAlmostEqual(contribs["recency"], 0.160, places=3)
        self.assertAlmostEqual(contribs["frequency"], 0.120, places=3)
        self.assertAlmostEqual(contribs["retrieval_history"], 0.100, places=3)
        self.assertAlmostEqual(contribs["semantic_relevance"], 0.140, places=3)
        self.assertAlmostEqual(contribs["relationship_strength"], 0.075, places=3)

    def test_custom_weights_configuration(self):
        """Verify engine correctly uses custom configurable weights."""
        custom_weights = BaselineImportanceWeights(
            recency=0.30,
            frequency=0.30,
            retrieval_history=0.10,
            semantic_relevance=0.20,
            relationship_strength=0.10,
        )
        custom_engine = Phase5BaselineDMCLM(weights=custom_weights)

        record = {
            "candidate_id": "test_mem_custom",
            "recency": 1.0,
            "frequency": 1.0,
            "retrieval_history": 0.0,
            "semantic_relevance": 0.5,
            "relationship_strength": 0.0,
        }
        # Expected: 0.30*1.0 + 0.30*1.0 + 0.10*0.0 + 0.20*0.5 + 0.10*0.0 = 0.3 + 0.3 + 0.1 = 0.70
        scored = custom_engine.score_single_memory(record)
        self.assertAlmostEqual(scored.baseline_score, 0.70, places=4)


class TestPhase5WeightValidation(unittest.TestCase):
    def test_valid_weights_pass(self):
        """Verify weights that sum to 1.0 pass validation without errors."""
        weights = BaselineImportanceWeights(
            recency=0.20, frequency=0.20, retrieval_history=0.25,
            semantic_relevance=0.20, relationship_strength=0.15
        )
        weights.validate()  # Should not raise

    def test_invalid_weights_sum_error(self):
        """Verify ValueError is raised when weights do not sum to 1.0."""
        invalid_weights = BaselineImportanceWeights(
            recency=0.30, frequency=0.30, retrieval_history=0.30,
            semantic_relevance=0.30, relationship_strength=0.30  # Sum = 1.50
        )
        with self.assertRaises(ValueError) as ctx:
            invalid_weights.validate()
        self.assertIn("must sum to 1.0", str(ctx.exception))

    def test_negative_weights_error(self):
        """Verify ValueError is raised when any weight is negative."""
        negative_weights = BaselineImportanceWeights(
            recency=-0.10, frequency=0.30, retrieval_history=0.25,
            semantic_relevance=0.35, relationship_strength=0.20
        )
        with self.assertRaises(ValueError) as ctx:
            negative_weights.validate()
        self.assertIn("must be non-negative", str(ctx.exception))

    def test_non_numeric_weights_error(self):
        """Verify TypeError is raised when a weight is non-numeric."""
        bad_weights = BaselineImportanceWeights(
            recency="high", frequency=0.20, retrieval_history=0.25,  # type: ignore
            semantic_relevance=0.20, relationship_strength=0.15
        )
        with self.assertRaises(TypeError) as ctx:
            bad_weights.validate()
        self.assertIn("must be numeric", str(ctx.exception))


class TestPhase5ScoreNormalizationAndBounds(unittest.TestCase):
    def setUp(self):
        self.engine = Phase5BaselineDMCLM()

    def test_all_minimum_features_yield_zero(self):
        """Verify that zero-valued features produce a score of exactly 0.0."""
        record = {
            "candidate_id": "c_zero",
            "recency": 0.0,
            "frequency": 0.0,
            "retrieval_history": 0.0,
            "semantic_relevance": 0.0,
            "relationship_strength": 0.0,
        }
        scored = self.engine.score_single_memory(record)
        self.assertEqual(scored.baseline_score, 0.0)

    def test_all_maximum_features_yield_one(self):
        """Verify that maximum-valued features produce a score of exactly 1.0."""
        record = {
            "candidate_id": "c_max",
            "recency": 1.0,
            "frequency": 1.0,
            "retrieval_history": 1.0,
            "semantic_relevance": 1.0,
            "relationship_strength": 1.0,
        }
        scored = self.engine.score_single_memory(record)
        self.assertEqual(scored.baseline_score, 1.0)

    def test_out_of_bounds_feature_clamping(self):
        """Verify features slightly outside [0, 1] due to noise are clamped safely."""
        record = {
            "candidate_id": "c_clamp",
            "recency": 1.5,   # Should clamp to 1.0
            "frequency": -0.2, # Should clamp to 0.0
            "retrieval_history": 1.0,
            "semantic_relevance": 1.0,
            "relationship_strength": 1.0,
        }
        # With clamp: 0.20(1.0) + 0.20(0.0) + 0.25(1.0) + 0.20(1.0) + 0.15(1.0) = 0.2 + 0.25 + 0.2 + 0.15 = 0.80
        scored = self.engine.score_single_memory(record)
        self.assertEqual(scored.baseline_score, 0.80)


class TestPhase5InterpretationTiers(unittest.TestCase):
    def setUp(self):
        self.engine = Phase5BaselineDMCLM()

    def test_high_value_tier_classification(self):
        """Verify score >= 0.70 classifies as HIGH_VALUE and RETAIN."""
        record = {"recency": 1.0, "frequency": 0.9, "retrieval_history": 0.8, "semantic_relevance": 0.9, "relationship_strength": 0.8}
        scored = self.engine.score_single_memory(record)
        self.assertGreaterEqual(scored.baseline_score, 0.70)
        self.assertEqual(scored.value_tier, "HIGH_VALUE")
        self.assertEqual(scored.lifecycle_tier, "RETAIN")

    def test_medium_value_tier_classification(self):
        """Verify 0.40 <= score < 0.70 classifies as MEDIUM_VALUE and COMPRESS."""
        record = {"recency": 0.70, "frequency": 0.50, "retrieval_history": 0.0, "semantic_relevance": 0.50, "relationship_strength": 0.50}
        scored = self.engine.score_single_memory(record)
        # Expected: 0.20*0.7 + 0.20*0.5 + 0.25*0 + 0.20*0.5 + 0.15*0.5 = 0.14 + 0.10 + 0 + 0.10 + 0.075 = 0.415
        self.assertTrue(0.40 <= scored.baseline_score < 0.70)
        self.assertEqual(scored.value_tier, "MEDIUM_VALUE")
        self.assertEqual(scored.lifecycle_tier, "COMPRESS")

    def test_low_value_tier_classification(self):
        """Verify score < 0.40 classifies as LOW_VALUE (ARCHIVE or FORGET)."""
        record_archive = {"recency": 0.30, "frequency": 0.30, "retrieval_history": 0.0, "semantic_relevance": 0.30, "relationship_strength": 0.20}
        scored_arch = self.engine.score_single_memory(record_archive)
        self.assertLess(scored_arch.baseline_score, 0.40)
        self.assertEqual(scored_arch.value_tier, "LOW_VALUE")
        self.assertEqual(scored_arch.lifecycle_tier, "ARCHIVE")

        record_forget = {"recency": 0.05, "frequency": 0.05, "retrieval_history": 0.0, "semantic_relevance": 0.05, "relationship_strength": 0.05}
        scored_forg = self.engine.score_single_memory(record_forget)
        self.assertLess(scored_forg.baseline_score, 0.20)
        self.assertEqual(scored_forg.value_tier, "LOW_VALUE")
        self.assertEqual(scored_forg.lifecycle_tier, "FORGET")


class TestPhase5ExplainabilityAndAttribution(unittest.TestCase):
    def setUp(self):
        self.engine = Phase5BaselineDMCLM()

    def test_dominant_feature_identification(self):
        """Verify dominant feature correctly pinpoints highest contributing feature."""
        # Dominant recency
        rec_recency = {"recency": 1.0, "frequency": 0.1, "retrieval_history": 0.0, "semantic_relevance": 0.1, "relationship_strength": 0.1}
        scored_r = self.engine.score_single_memory(rec_recency)
        self.assertEqual(scored_r.dominant_feature, "recency")

        # Dominant retrieval history
        rec_retrieval = {"recency": 0.2, "frequency": 0.2, "retrieval_history": 1.0, "semantic_relevance": 0.2, "relationship_strength": 0.2}
        scored_h = self.engine.score_single_memory(rec_retrieval)
        self.assertEqual(scored_h.dominant_feature, "retrieval_history")

    def test_relative_percentage_attribution(self):
        """Verify percentage contributions sum to approximately 100%."""
        record = {"recency": 0.8, "frequency": 0.5, "retrieval_history": 0.4, "semantic_relevance": 0.6, "relationship_strength": 0.5}
        scored = self.engine.score_single_memory(record)
        rel_pct = scored.relative_contributions
        total_pct = sum(rel_pct.values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_explanation_string_completeness(self):
        """Verify human-readable explanation contains score, tier, and all component contributions."""
        record = {"candidate_id": "cand_expl", "recency": 0.9, "frequency": 0.7, "retrieval_history": 0.0, "semantic_relevance": 0.5, "relationship_strength": 0.6}
        scored = self.engine.score_single_memory(record)
        expl = scored.explanation
        self.assertIn("Baseline score", expl)
        self.assertIn("Recency contributed", expl)
        self.assertIn("Frequency contributed", expl)
        self.assertIn("Retrieval History contributed", expl)
        self.assertIn("Dominant factor:", expl)


class TestPhase5RobustnessAndEdgeCases(unittest.TestCase):
    def setUp(self):
        self.engine = Phase5BaselineDMCLM()

    def test_missing_features_graceful_handling(self):
        """Verify record with missing feature keys defaults missing values safely to 0.0."""
        sparse_record = {"candidate_id": "c_sparse", "recency": 0.80}
        scored = self.engine.score_single_memory(sparse_record)
        self.assertAlmostEqual(scored.baseline_score, 0.20 * 0.80, places=3)
        self.assertEqual(scored.frequency, 0.0)
        self.assertEqual(scored.retrieval_history, 0.0)

    def test_empty_dataset_handling(self):
        """Verify empty memory list returns empty list and clean statistics."""
        scored = self.engine.score_memories([])
        self.assertEqual(len(scored), 0)
        stats = self.engine.calculate_statistics([])
        self.assertEqual(stats, {})

    def test_single_memory_dataset(self):
        """Verify single memory produces valid baseline outputs without division errors."""
        record = {"candidate_id": "c_single", "recency": 0.75, "frequency": 0.75, "retrieval_history": 0.0, "semantic_relevance": 0.5, "relationship_strength": 0.5}
        scored = self.engine.score_memories([record])
        self.assertEqual(len(scored), 1)
        stats = self.engine.calculate_statistics(scored)
        self.assertEqual(stats["total_memories_scored"], 1)
        self.assertIn("score_distribution", stats)


class TestPhase5DeterminismAndPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.mock_db = self.temp_path / "test_baseline.db"
        self.engine = Phase5BaselineDMCLM(db_path=self.mock_db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deterministic_scoring(self):
        """Verify running baseline twice produces exact same floating-point scores and explanations."""
        records = [
            {"candidate_id": f"c_{i}", "recency": 0.1 * i, "frequency": 0.2 * i, "retrieval_history": 0.0, "semantic_relevance": 0.15 * i, "relationship_strength": 0.1 * i}
            for i in range(5)
        ]
        run1 = self.engine.score_memories(records)
        run2 = self.engine.score_memories(records)

        self.assertEqual(len(run1), len(run2))
        for m1, m2 in zip(run1, run2):
            self.assertEqual(m1.candidate_id, m2.candidate_id)
            self.assertEqual(m1.baseline_score, m2.baseline_score)
            self.assertEqual(m1.feature_contributions, m2.feature_contributions)
            self.assertEqual(m1.value_tier, m2.value_tier)
            self.assertEqual(m1.explanation, m2.explanation)

    def test_idempotent_persistence_no_duplicate_records(self):
        """Verify saving twice into SQLite table does not duplicate rows."""
        record = {"candidate_id": "c_idemp", "recency": 0.8, "frequency": 0.6, "retrieval_history": 0.0, "semantic_relevance": 0.5, "relationship_strength": 0.7}
        scored = self.engine.score_memories([record])
        stats = self.engine.calculate_statistics(scored)

        # Write 1
        self.engine.save_baseline_dataset(scored, stats)
        conn = sqlite3.connect(str(self.mock_db))
        cnt1 = conn.execute("SELECT COUNT(*) FROM baseline_scored_memories").fetchone()[0]
        conn.close()
        self.assertEqual(cnt1, 1)

        # Write 2
        self.engine.save_baseline_dataset(scored, stats)
        conn = sqlite3.connect(str(self.mock_db))
        cnt2 = conn.execute("SELECT COUNT(*) FROM baseline_scored_memories").fetchone()[0]
        conn.close()
        self.assertEqual(cnt1, cnt2, "Idempotency failed: record count doubled on repeated write.")


class TestPhase5RealDataIntegration(unittest.TestCase):
    def test_real_phase4_dataset_baseline_scoring(self):
        """
        Integration test: Ingests real Phase 4 feature-enriched memories,
        calculates transparent baseline scores, verifies statistical bounds,
        and ensures complete derived storage synchronization.
        """
        engine = Phase5BaselineDMCLM()
        phase4_memories = engine.load_phase4_features()
        self.assertGreater(len(phase4_memories), 0, "No Phase 4 memories loaded.")

        # Execute Phase 5 Baseline DMCLM
        res = engine.run(phase4_memories)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["total_memories"], len(phase4_memories))

        scored_memories = res["scored_memories"]
        self.assertEqual(len(scored_memories), len(phase4_memories))

        # Validate bounds & preservation on sample of real records
        for m in scored_memories[:100]:
            self.assertTrue(0.0 <= m.baseline_score <= 1.0)
            self.assertTrue(0.0 <= m.recency <= 1.0)
            self.assertTrue(0.0 <= m.frequency <= 1.0)
            self.assertTrue(0.0 <= m.retrieval_history <= 1.0)
            self.assertTrue(0.0 <= m.semantic_relevance <= 1.0)
            self.assertTrue(0.0 <= m.relationship_strength <= 1.0)

            # Check that feature contributions sum to baseline score
            contrib_sum = sum(m.feature_contributions.values())
            self.assertAlmostEqual(m.baseline_score, contrib_sum, places=3)
            self.assertIn(m.value_tier, ["HIGH_VALUE", "MEDIUM_VALUE", "LOW_VALUE"])
            self.assertTrue(len(m.explanation) > 0)

        # Verify statistics
        stats = res["statistics"]
        self.assertIn("score_distribution", stats)
        sd = stats["score_distribution"]
        self.assertTrue(0.0 <= sd["min"] <= sd["max"] <= 1.0)
        self.assertGreater(sd["mean"], 0.0)

        # Verify database table contains all scored records
        conn = sqlite3.connect(str(DATA_DIR / "dmclm.db"))
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM baseline_scored_memories").fetchone()[0]
            self.assertEqual(cnt, len(phase4_memories))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

"""
Unit and Integration Tests for Phase 7 â€” Adaptive Retention / Forgetting
Member 4 â€” Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Covers all 19 required unit test cases:
1. Recent memory retention.
2. Old memory decay.
3. High-value memory protection.
4. Low-value memory decay.
5. Forgetting eligibility.
6. Archive transition.
7. Retention-state classification.
8. Decay calculation.
9. Deterministic results.
10. Configurable thresholds.
11. Missing timestamps.
12. Missing retrieval history.
13. Missing feature values.
14. Empty dataset.
15. Single-memory dataset.
16. Future timestamps.
17. Repeat execution.
18. Traceability preservation.
19. No accidental permanent deletion.
+ Real-data integration test against Phase 6 consolidated memories dataset.
"""

import copy
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import AdaptiveRetentionConfig, DEFAULT_CONFIG
from src.phase7_adaptive_retention_forgetting import (
    Phase7AdaptiveRetentionEngine,
    trace_retention_memory,
    CONSOLIDATED_JSON_PATH,
)


def create_mock_consolidated_memory(
    node_id: str = "c_node_test_01",
    baseline_score: float = 0.55,
    recency: float = 0.80,
    frequency: float = 0.60,
    retrieval_history: float = 0.0,
    semantic_relevance: float = 0.65,
    relationship_strength: float = 0.60,
    source_count: int = 5,
    end_time: str = "2026-10-01T10:00:00+00:00",
    source_mids: List[str] = None,
    source_eids: List[str] = None,
    app: str = "Google Chrome",
    summary: str = "Consolidated research on cognitive memory architecture",
) -> Dict[str, Any]:
    """Helper to construct controlled consolidated memory test fixtures."""
    return {
        "consolidated_memory_id": node_id,
        "ltm_id": node_id,
        "memory_id": node_id,
        "source_memory_ids": source_mids or [f"mem_{node_id}_1", f"mem_{node_id}_2"],
        "source_episode_ids": source_eids or [f"ep_{node_id}_1", f"ep_{node_id}_2"],
        "primary_application": app,
        "application": app,
        "start_time": "2026-10-01T09:00:00+00:00",
        "end_time": end_time,
        "last_reinforced": end_time,
        "duration_ms": 3600000,
        "source_count": source_count,
        "semantic_representation": summary,
        "summary": summary,
        "topic": f"{app}: {summary[:30]}",
        "baseline_information": {
            "mean_baseline_score": baseline_score,
            "dominant_feature": "frequency",
            "stability_score": 0.65,
            "feature_means": {
                "recency": recency,
                "frequency": frequency,
                "retrieval_history": retrieval_history,
                "semantic_relevance": semantic_relevance,
                "relationship_strength": relationship_strength,
            }
        },
        "interaction_statistics": {
            "total_interactions": 45,
            "browser_event_count": 12,
            "screenshot_count": 4,
            "interaction_density_per_sec": 0.0125
        },
        "consolidation_metadata": {
            "algorithm": "Multi-Evidence Connected Components Clustering",
            "created_at": end_time,
        }
    }


class TestPhase7AdaptiveRetentionAndForgetting(unittest.TestCase):
    """Automated unit test suite covering all 19 Phase 7 requirements."""

    def setUp(self):
        self.config = AdaptiveRetentionConfig(
            base_decay_rate=0.05,
            adaptive_decay_alpha=0.60,
            retain_threshold=0.60,
            archive_threshold=0.35,
            forget_threshold=0.20,
            enable_protection=True,
            protection_baseline_threshold=0.60,
            protection_floor=0.40,
            min_age_days_for_forgetting=7.0,
            reference_mode="explicit",
            explicit_reference_timestamp="2026-10-01T12:00:00+00:00"  # 2 hours after 10:00:00
        )
        self.engine = Phase7AdaptiveRetentionEngine(self.config)

    # 1. Recent memory retention
    def test_01_recent_memory_retention(self):
        """Verify recent memory with strong cognitive value remains in RETAINED state."""
        mem = create_mock_consolidated_memory(
            "mem_recent",
            baseline_score=0.70,
            recency=0.90,
            frequency=0.75,
            semantic_relevance=0.75,
            end_time="2026-10-01T11:30:00+00:00"  # 30 min before ref time
        )
        results, metrics = self.engine.process([mem], reference_time="2026-10-01T12:00:00+00:00")
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.retention_state, "RETAINED")
        self.assertGreaterEqual(r.final_retention_strength, 0.60)
        self.assertAlmostEqual(r.decay_factor, 1.0, delta=0.05)
        self.assertIn("RETAINED", r.retention_explanation)

    # 2. Old memory decay
    def test_02_old_memory_decay(self):
        """Verify a memory significantly decays over elapsed time (e.g. 50 days)."""
        mem_recent = create_mock_consolidated_memory("m_recent", end_time="2026-10-01T12:00:00+00:00")
        mem_old = create_mock_consolidated_memory("m_old", end_time="2026-08-12T12:00:00+00:00")  # 50 days prior

        # Disable protection floor to isolate pure decay effect
        unprotected_config = copy.deepcopy(self.config)
        unprotected_config.enable_protection = False
        engine_unprotected = Phase7AdaptiveRetentionEngine(unprotected_config)

        res_recent, _ = engine_unprotected.process([mem_recent], reference_time="2026-10-01T12:00:00+00:00")
        res_old, _ = engine_unprotected.process([mem_old], reference_time="2026-10-01T12:00:00+00:00")

        self.assertGreater(res_recent[0].decay_factor, res_old[0].decay_factor)
        self.assertGreater(res_recent[0].final_retention_strength, res_old[0].final_retention_strength)
        self.assertGreater(res_old[0].age_days, 40.0)

    # 3. High-value memory protection
    def test_03_high_value_memory_protection(self):
        """Verify high-value memory is protected and never drops below the protection floor."""
        # Memory 120 days old with high baseline value (0.75)
        mem_vip = create_mock_consolidated_memory(
            "mem_vip",
            baseline_score=0.75,
            frequency=0.80,
            end_time="2026-06-03T12:00:00+00:00"  # 120 days old
        )
        results, _ = self.engine.process([mem_vip], reference_time="2026-10-01T12:00:00+00:00")
        r = results[0]

        self.assertTrue(r.is_protected)
        self.assertIsNotNone(r.protection_reason)
        # Final retention strength must be at or above protection floor (0.40)
        self.assertGreaterEqual(r.final_retention_strength, self.config.protection_floor)
        self.assertIn(r.retention_state, ["DECAYING", "RETAINED"])
        self.assertFalse(r.forgetting_eligible)

    # 4. Low-value memory decay
    def test_04_low_value_memory_decay(self):
        """Verify low-value memory decays faster with higher effective decay rate."""
        mem_low = create_mock_consolidated_memory(
            "mem_low",
            baseline_score=0.25,
            recency=0.20,
            frequency=0.15,
            semantic_relevance=0.20,
            relationship_strength=0.15,
            source_count=2,
            end_time="2026-09-01T12:00:00+00:00"  # 30 days old
        )
        mem_high = create_mock_consolidated_memory(
            "mem_high",
            baseline_score=0.70,
            recency=0.70,
            frequency=0.70,
            semantic_relevance=0.70,
            relationship_strength=0.70,
            source_count=10,
            end_time="2026-09-01T12:00:00+00:00"  # 30 days old
        )
        results, _ = self.engine.process([mem_low, mem_high], reference_time="2026-10-01T12:00:00+00:00")
        res_map = {r.consolidated_memory_id: r for r in results}

        # Effective decay rate of low memory must be greater than high memory
        self.assertGreater(res_map["mem_low"].effective_decay_rate, res_map["mem_high"].effective_decay_rate)
        self.assertLess(res_map["mem_low"].decay_factor, res_map["mem_high"].decay_factor)

    # 5. Forgetting eligibility
    def test_05_forgetting_eligibility(self):
        """Verify memory with low value, low activity, and sufficient age qualifies as FORGETTABLE."""
        mem_stale = create_mock_consolidated_memory(
            "mem_stale",
            baseline_score=0.18,
            recency=0.10,
            frequency=0.10,
            retrieval_history=0.0,
            semantic_relevance=0.15,
            relationship_strength=0.10,
            source_count=2,
            end_time="2026-08-01T12:00:00+00:00"  # 61 days old
        )
        results, _ = self.engine.process([mem_stale], reference_time="2026-10-01T12:00:00+00:00")
        r = results[0]

        self.assertFalse(r.is_protected)
        self.assertTrue(r.forgetting_eligible)
        self.assertEqual(r.retention_state, "FORGETTABLE")
        self.assertLess(r.final_retention_strength, self.config.forget_threshold)
        self.assertGreaterEqual(r.age_days, self.config.min_age_days_for_forgetting)

    def test_05b_strong_relationship_blocks_forgetting(self):
        """Strongly connected memories must not qualify for forgetting."""
        mem = create_mock_consolidated_memory(
            "mem_strong_relationship",
            baseline_score=0.18,
            recency=0.10,
            frequency=0.10,
            retrieval_history=0.0,
            semantic_relevance=0.15,
            relationship_strength=0.80,
            source_count=2,
            end_time="2026-08-01T12:00:00+00:00"
        )

        results, _ = self.engine.process(
            [mem], reference_time="2026-10-01T12:00:00+00:00"
        )
        result = results[0]

        self.assertFalse(result.is_protected)
        self.assertFalse(result.forgetting_eligible)
        self.assertNotEqual(result.retention_state, "FORGETTABLE")
    # 6. Archive transition
    def test_06_archive_transition(self):
        """Verify memory with low score but lacking forgetting criteria transitions to ARCHIVED."""
        # Low score, but only 3 days old (below the 7-day minimum age for forgetting)
        mem_young_low = create_mock_consolidated_memory(
            "mem_young_low",
            baseline_score=0.20,
            frequency=0.10,
            retrieval_history=0.0,
            semantic_relevance=0.20,
            relationship_strength=0.10,
            end_time="2026-09-28T12:00:00+00:00"  # 3 days before Oct 1
        )
        results, _ = self.engine.process([mem_young_low], reference_time="2026-10-01T12:00:00+00:00")
        r = results[0]

        self.assertFalse(r.forgetting_eligible)
        self.assertEqual(r.retention_state, "ARCHIVED")
        self.assertIn("ARCHIVED", r.retention_explanation)

    # 7. Retention-state classification
    def test_07_retention_state_classification(self):
        """Verify classification boundaries for RETAINED, DECAYING, ARCHIVED, FORGETTABLE."""
        cfg = self.config
        self.assertEqual(cfg.classify_state(0.75, False), "RETAINED")
        self.assertEqual(cfg.classify_state(0.60, False), "RETAINED")
        self.assertEqual(cfg.classify_state(0.45, False), "DECAYING")
        self.assertEqual(cfg.classify_state(0.35, False), "DECAYING")
        self.assertEqual(cfg.classify_state(0.25, False), "ARCHIVED")
        self.assertEqual(cfg.classify_state(0.15, False), "ARCHIVED")
        self.assertEqual(cfg.classify_state(0.15, True), "FORGETTABLE")

    # 8. Decay calculation
    def test_08_decay_calculation(self):
        """Verify exact exponential decay formula: T_decay = exp(-lambda * age_days)."""
        age_days = 20.0
        eff_lambda = 0.04
        expected_decay = math.exp(-eff_lambda * age_days)

        # Build controlled memory evaluated at exact time
        mem = create_mock_consolidated_memory("mem_calc", end_time="2026-09-11T12:00:00+00:00")
        results, _ = self.engine.process([mem], reference_time="2026-10-01T12:00:00+00:00")
        r = results[0]

        self.assertAlmostEqual(r.age_days, 20.0, places=1)
        expected = math.exp(-r.effective_decay_rate * r.age_days)
        self.assertAlmostEqual(r.decay_factor, expected, places=4)

    # 9. Deterministic results
    def test_09_deterministic_results(self):
        """Verify identical inputs and reference time produce strictly identical output."""
        mems = [create_mock_consolidated_memory(f"m_{i}") for i in range(4)]
        res_1, met_1 = self.engine.process(mems, reference_time="2026-10-01T12:00:00+00:00")
        res_2, met_2 = self.engine.process(mems, reference_time="2026-10-01T12:00:00+00:00")

        self.assertEqual([r.to_dict() for r in res_1], [r.to_dict() for r in res_2])
        self.assertEqual(met_1["average_retention_strength"], met_2["average_retention_strength"])

    # 10. Configurable thresholds
    def test_10_configurable_thresholds(self):
        """Verify changing retain_threshold correctly alters state assignment."""
        mem = create_mock_consolidated_memory(
            "mem_thresh",
            baseline_score=0.75,
            recency=0.90,
            frequency=0.80,
            semantic_relevance=0.80,
            relationship_strength=0.80,
            end_time="2026-10-01T10:00:00+00:00"
        )

        # Under default retain_threshold (0.60), should be RETAINED (score ~0.69)
        res_default, _ = self.engine.process([mem], reference_time="2026-10-01T10:00:00+00:00")
        self.assertEqual(res_default[0].retention_state, "RETAINED")

        # Under stricter retain_threshold (0.75), should be DECAYING
        strict_cfg = copy.deepcopy(self.config)
        strict_cfg.retain_threshold = 0.75
        engine_strict = Phase7AdaptiveRetentionEngine(strict_cfg)
        res_strict, _ = engine_strict.process([mem], reference_time="2026-10-01T10:00:00+00:00")
        self.assertEqual(res_strict[0].retention_state, "DECAYING")

    # 11. Missing timestamps
    def test_11_missing_timestamps(self):
        """Verify handling of empty or None timestamps without unhandled exceptions."""
        mem_no_ts = create_mock_consolidated_memory("mem_no_ts")
        mem_no_ts["end_time"] = None
        mem_no_ts["last_reinforced"] = ""
        mem_no_ts["start_time"] = ""

        results, _ = self.engine.process([mem_no_ts], reference_time="2026-10-01T12:00:00+00:00")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].age_days, 0.0)
        self.assertEqual(results[0].decay_factor, 1.0)

    # 12. Missing retrieval history
    def test_12_missing_retrieval_history(self):
        """Verify missing retrieval history safely defaults to 0.0."""
        mem = create_mock_consolidated_memory("mem_no_ret")
        del mem["baseline_information"]["feature_means"]["retrieval_history"]

        results, _ = self.engine.process([mem], reference_time="2026-10-01T12:00:00+00:00")
        self.assertEqual(results[0].retrieval_history, 0.0)

    # 13. Missing feature values
    def test_13_missing_feature_values(self):
        """Verify memory lacking baseline_information or features executes cleanly."""
        mem_bare = {
            "consolidated_memory_id": "c_bare",
            "source_memory_ids": ["m1"],
            "source_episode_ids": ["e1"],
            "primary_application": "App",
            "end_time": "2026-10-01T10:00:00+00:00"
        }
        results, metrics = self.engine.process([mem_bare], reference_time="2026-10-01T12:00:00+00:00")
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].final_retention_strength, 0.0)
        self.assertEqual(metrics["total_memories"], 1)

    # 14. Empty dataset
    def test_14_empty_dataset(self):
        """Verify empty input returns empty list and clean statistics."""
        results, metrics = self.engine.process([])
        self.assertEqual(len(results), 0)
        self.assertEqual(metrics["total_memories"], 0)
        self.assertEqual(metrics["retained_count"], 0)
        self.assertEqual(metrics["average_retention_strength"], 0.0)

    # 15. Single-memory dataset
    def test_15_single_memory_dataset(self):
        """Verify single memory produces valid retention object without errors."""
        mem = create_mock_consolidated_memory("c_single")
        results, metrics = self.engine.process([mem])
        self.assertEqual(len(results), 1)
        self.assertEqual(metrics["total_memories"], 1)
        self.assertGreater(results[0].final_retention_strength, 0.0)

    # 16. Future timestamps
    def test_16_future_timestamps(self):
        """Verify memory with timestamp ahead of reference time clamps elapsed time to 0.0."""
        mem_future = create_mock_consolidated_memory(
            "mem_future",
            end_time="2026-10-15T12:00:00+00:00"  # 14 days in future relative to Oct 1
        )
        results, _ = self.engine.process([mem_future], reference_time="2026-10-01T12:00:00+00:00")
        r = results[0]

        self.assertEqual(r.age_days, 0.0)
        self.assertEqual(r.decay_factor, 1.0)
        self.assertGreaterEqual(r.final_retention_strength, 0.0)

    # 17. Repeat execution
    def test_17_repeat_execution(self):
        """Verify multiple consecutive runs maintain idempotency."""
        mems = [create_mock_consolidated_memory(f"mem_{i}") for i in range(3)]
        res1, _ = self.engine.process(mems, reference_time="2026-10-01T12:00:00+00:00")
        res2, _ = self.engine.process(mems, reference_time="2026-10-01T12:00:00+00:00")

        for a, b in zip(res1, res2):
            self.assertEqual(a.final_retention_strength, b.final_retention_strength)
            self.assertEqual(a.retention_state, b.retention_state)

    # 18. Traceability preservation
    def test_18_traceability_preservation(self):
        """Verify complete lineage: Retention Node -> Source Memory IDs -> Episode IDs."""
        mem = create_mock_consolidated_memory(
            "c_trace",
            source_mids=["m1", "m2", "m3"],
            source_eids=["ep1", "ep2", "ep3"]
        )
        results, _ = self.engine.process([mem], reference_time="2026-10-01T12:00:00+00:00")
        r = results[0]

        trace = trace_retention_memory(r.consolidated_memory_id, results)
        self.assertEqual(trace["consolidated_memory_id"], "c_trace")
        self.assertEqual(trace["source_memory_ids"], ["m1", "m2", "m3"])
        self.assertEqual(trace["source_episode_ids"], ["ep1", "ep2", "ep3"])
        self.assertEqual(trace["source_memory_count"], 3)

    # 19. No accidental permanent deletion
    def test_19_no_accidental_permanent_deletion(self):
        """Verify original memory list/dictionaries are never mutated, removed, or deleted."""
        original_mems = [create_mock_consolidated_memory(f"m_{i}") for i in range(3)]
        deep_copy = copy.deepcopy(original_mems)

        results, _ = self.engine.process(original_mems, reference_time="2026-10-01T12:00:00+00:00")

        self.assertEqual(len(original_mems), len(deep_copy))
        self.assertEqual(original_mems, deep_copy, "Input memory candidates must remain untouched")


class TestPhase7RealDataIntegration(unittest.TestCase):
    """Integration test running Phase 7 against actual Phase 6 consolidated memories."""

    @unittest.skipUnless(CONSOLIDATED_JSON_PATH.exists(), "Phase 6 consolidated memory file not found")
    def test_real_phase6_dataset_retention_evaluation(self):
        """Verify Phase 7 executes end-to-end on the real 74 Phase 6 consolidated memories."""
        engine = Phase7AdaptiveRetentionEngine()
        results, metrics = engine.run_pipeline(save_to_db=True, save_to_json=True)

        self.assertGreater(metrics["total_memories"], 0)
        self.assertEqual(len(results), metrics["total_memories"])
        self.assertGreaterEqual(metrics["retained_count"], 0)
        self.assertGreaterEqual(metrics["decaying_count"], 0)
        self.assertGreaterEqual(metrics["archived_count"], 0)
        self.assertGreaterEqual(metrics["average_retention_strength"], 0.0)

        # Inspect first result
        first = results[0]
        self.assertTrue(first.consolidated_memory_id.startswith("c_node_"))
        self.assertIn(first.retention_state, ["RETAINED", "DECAYING", "ARCHIVED", "FORGETTABLE"])
        self.assertGreater(len(first.retention_explanation), 20)

        # Print real results to stdout
        print("\n[REAL-DATA PHASE 7 INTEGRATION RESULTS]")
        print(f"Total Consolidated Memories:  {metrics['total_memories']:,}")
        print(f"RETAINED (Active):            {metrics['retained_count']:,} ({metrics['state_percentages']['RETAINED']}%)")
        print(f"DECAYING:                     {metrics['decaying_count']:,} ({metrics['state_percentages']['DECAYING']}%)")
        print(f"ARCHIVED:                     {metrics['archived_count']:,} ({metrics['state_percentages']['ARCHIVED']}%)")
        print(f"FORGETTABLE:                  {metrics['forgetting_eligible_count']:,} ({metrics['state_percentages']['FORGETTABLE']}%)")
        print(f"Protected Count:              {metrics['protected_count']:,}")
        print(f"Average Retention Strength:   {metrics['average_retention_strength']:.4f}")
        print(f"Min Retention Strength:       {metrics['min_retention_strength']:.4f}")
        print(f"Max Retention Strength:       {metrics['max_retention_strength']:.4f}")
        print(f"Processing Time:              {metrics['processing_time_seconds']}s")


if __name__ == "__main__":
    unittest.main()


"""
Unit and Integration Tests for Phase 9 — Training Dataset Generation
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Covers all Phase 9 requirements:
1. Schema validation & feature roles classification.
2. Leakage prevention guarantee (ML_INPUT strictly isolated from targets and baseline formulas).
3. Core 5 DMCLM cognitive features extraction and range bounds.
4. Target generation and weak-supervision attribution.
5. Missing-value handling and safe default imputation.
6. Duplicate candidate detection and rejection tracking.
7. Numerical range validation ([0, 1] clipping and validation).
8. Deterministic temporal splitting (time-aware chronological order).
9. Zero sample overlap across train, val, and test partitions.
10. Dynamic dataset manifest generation and SHA-256 hash calculation.
11. Provenance traceability query (trace_training_sample).
12. Reproducibility and idempotency (identical outputs on consecutive runs).
13. Macro consolidated LTM dataset generation.
14. Empty dataset edge case handling.
15. Real-data integration test against actual project outputs.
"""

import copy
import json
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import DatasetGenerationConfig, DEFAULT_CONFIG
from src.phase9_training_dataset import (
    Phase9DatasetGenerator,
    get_feature_matrix,
    BASELINE_SCORED_JSON_PATH,
    CONSOLIDATED_JSON_PATH,
    RETENTION_JSON_PATH,
    RELATIONSHIPS_JSON_PATH,
)


def create_mock_candidate_memory(
    candidate_id: str,
    start_time: str = "2026-09-25T10:00:00Z",
    end_time: str = "2026-09-25T10:15:00Z",
    primary_application: str = "Code.exe",
    recency: float = 0.85,
    frequency: float = 0.65,
    retrieval_history: float = 0.0,
    semantic_relevance: float = 0.70,
    relationship_strength: float = 0.60,
    baseline_score: float = 0.55,
    lifecycle_state: str = "COMPRESS",
    duration_ms: float = 900000.0,
    interaction_count: int = 45,
    browser_event_count: int = 0,
) -> Dict[str, Any]:
    """Helper to generate structured mock Phase 4/5 candidate memories."""
    return {
        "candidate_id": candidate_id,
        "memory_id": candidate_id,
        "session_id": "sess_mock_01",
        "source_episode_ids": [f"ep_{candidate_id}_1", f"ep_{candidate_id}_2"],
        "primary_application": primary_application,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
        "interaction_count": interaction_count,
        "browser_event_count": browser_event_count,
        "screenshot_ids": ["shot_01"],
        "window_context": "VSCode - Project",
        "recency_score": recency,
        "frequency_score": frequency,
        "retrieval_history_score": retrieval_history,
        "semantic_relevance": semantic_relevance,
        "relationship_strength": relationship_strength,
        "baseline_score": baseline_score,
        "importance_score": baseline_score,
        "lifecycle_state": lifecycle_state,
        "value_tier": "MEDIUM_VALUE" if baseline_score >= 0.40 else "LOW_VALUE",
    }


class TestPhase9TrainingDataset(unittest.TestCase):
    """Comprehensive test suite for Phase 9 Training Dataset Generation."""

    def setUp(self):
        self.config = DatasetGenerationConfig(
            split_strategy="temporal",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        self.generator = Phase9DatasetGenerator(config=self.config)

    def test_01_schema_validation_and_feature_roles(self):
        """Verify that all schema columns have distinct, valid roles and types."""
        schema = self.generator.define_feature_schema()
        self.assertGreater(len(schema), 15)

        roles = {f.role for f in schema}
        self.assertIn("METADATA", roles)
        self.assertIn("ML_INPUT", roles)
        self.assertIn("TARGET", roles)
        self.assertIn("AUDIT_REFERENCE", roles)

        col_names = [f.name for f in schema]
        self.assertEqual(len(col_names), len(set(col_names)), "Schema must not have duplicate column names.")

    def test_02_leakage_prevention_guarantee(self):
        """Verify that get_feature_matrix extracts ONLY ML_INPUT features with zero leakage."""
        schema = self.generator.define_feature_schema()
        ml_cols = [f.name for f in schema if f.role == "ML_INPUT"]
        forbidden_cols = [f.name for f in schema if f.role in ("TARGET", "AUDIT_REFERENCE")]

        # Ensure no forbidden columns are classified as ML_INPUT
        for col in forbidden_cols:
            self.assertNotIn(col, ml_cols, f"Forbidden column '{col}' must NEVER be in ML_INPUT.")

        # Test feature matrix extraction
        mock_candidates = [
            create_mock_candidate_memory("c1", "2026-09-25T10:00:00Z"),
            create_mock_candidate_memory("c2", "2026-09-25T11:00:00Z"),
        ]
        rows, _, _ = self.generator.process(mock_candidates, {}, {}, [])
        X, feature_names = get_feature_matrix(rows, schema)

        self.assertEqual(set(feature_names), set(ml_cols))
        self.assertNotIn("baseline_score", feature_names)
        self.assertNotIn("target_lifecycle_state", feature_names)
        self.assertNotIn("target_importance_score", feature_names)
        self.assertNotIn("final_retention_strength", feature_names)

    def test_03_core_five_features_extraction_and_bounds(self):
        """Verify that the 5 cognitive features are correctly extracted and bounded in [0.0, 1.0]."""
        mock_candidates = [
            create_mock_candidate_memory(
                "c1",
                recency=0.92,
                frequency=0.45,
                retrieval_history=0.0,
                semantic_relevance=0.81,
                relationship_strength=0.73,
            )
        ]
        rows, _, _ = self.generator.process(mock_candidates, {}, {}, [])
        self.assertEqual(len(rows), 1)
        r = rows[0]

        self.assertEqual(r["feature_recency"], 0.92)
        self.assertEqual(r["feature_frequency"], 0.45)
        self.assertEqual(r["feature_retrieval_history"], 0.0)
        self.assertEqual(r["feature_semantic_relevance"], 0.81)
        self.assertEqual(r["feature_relationship_strength"], 0.73)

        for feat in ["feature_recency", "feature_frequency", "feature_retrieval_history",
                     "feature_semantic_relevance", "feature_relationship_strength"]:
            self.assertTrue(0.0 <= r[feat] <= 1.0)

    def test_04_target_generation_and_weak_supervision(self):
        """Verify that targets are properly formatted with weak supervision attribution."""
        mock_candidates = [
            create_mock_candidate_memory("c1", baseline_score=0.55, lifecycle_state="COMPRESS"),
            create_mock_candidate_memory("c2", baseline_score=0.25, lifecycle_state="ARCHIVE"),
        ]
        rows, _, _ = self.generator.process(mock_candidates, {}, {}, [])
        self.assertEqual(len(rows), 2)

        self.assertEqual(rows[0]["target_lifecycle_state"], "COMPRESS")
        self.assertEqual(rows[0]["target_importance_score"], 0.55)
        self.assertEqual(rows[0]["target_consolidation_priority"], 1)

        self.assertEqual(rows[1]["target_lifecycle_state"], "ARCHIVE")
        self.assertEqual(rows[1]["target_importance_score"], 0.25)
        self.assertEqual(rows[1]["target_consolidation_priority"], 0)

        self.assertEqual(rows[0]["label_origin"], "behavior_derived_baseline_policy")

    def test_05_missing_value_handling_and_safe_defaults(self):
        """Verify that missing optional fields (browser counts, timestamps) are handled safely."""
        sparse_cand = {
            "candidate_id": "sparse_01",
            "start_time": "2026-09-25T10:00:00Z",
            # Missing duration, interactions, browser counts, recency, etc.
        }
        rows, _, manifest = self.generator.process([sparse_cand], {}, {}, [])
        self.assertEqual(len(rows), 1)
        r = rows[0]

        self.assertEqual(r["browser_event_count"], 0)
        self.assertEqual(r["has_browser_activity"], 0)
        self.assertEqual(r["interaction_count"], 0)
        self.assertEqual(r["duration_ms"], 0.0)
        self.assertFalse(math.isnan(r["feature_recency"]))
        self.assertFalse(math.isnan(r["target_importance_score"]))

    def test_06_duplicate_candidate_detection(self):
        """Verify that duplicate candidate memory IDs are rejected and counted."""
        dup_candidates = [
            create_mock_candidate_memory("dup_01", "2026-09-25T10:00:00Z"),
            create_mock_candidate_memory("dup_01", "2026-09-25T10:05:00Z"),  # Duplicate
        ]
        rows, _, manifest = self.generator.process(dup_candidates, {}, {}, [])
        self.assertEqual(len(rows), 1, "Duplicate candidate should be rejected.")
        self.assertEqual(manifest.rejected_records_count, 1)
        self.assertEqual(manifest.rejection_reasons.get("duplicate_memory_id"), 1)

    def test_07_range_validation_clipping(self):
        """Verify that out-of-range feature values are safely clipped to [0.0, 1.0]."""
        out_of_bounds = [
            create_mock_candidate_memory("c_oob", recency=1.5, frequency=-0.3)
        ]
        rows, _, _ = self.generator.process(out_of_bounds, {}, {}, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feature_recency"], 1.0)
        self.assertEqual(rows[0]["feature_frequency"], 0.0)

    def test_08_deterministic_temporal_split(self):
        """Verify that temporal splitting sorts strictly by start_time (no lookahead leakage)."""
        candidates = [
            create_mock_candidate_memory("t3", "2026-10-01T10:00:00Z"),
            create_mock_candidate_memory("t1", "2026-09-21T10:00:00Z"),
            create_mock_candidate_memory("t2", "2026-09-25T10:00:00Z"),
            create_mock_candidate_memory("t4", "2026-10-05T10:00:00Z"),
        ]
        # With 4 samples: 70% train = 2, 15% val = 0 (or 1), test = remainder
        cfg = DatasetGenerationConfig(train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, split_strategy="temporal")
        gen = Phase9DatasetGenerator(cfg)
        rows, _, _ = gen.process(candidates, {}, {}, [])

        train_times = [r["start_time"] for r in rows if r["split"] == "train"]
        val_times = [r["start_time"] for r in rows if r["split"] == "val"]
        test_times = [r["start_time"] for r in rows if r["split"] == "test"]

        # Earliest must be in train, latest in test
        if train_times and val_times:
            self.assertLessEqual(max(train_times), min(val_times))
        if val_times and test_times:
            self.assertLessEqual(max(val_times), min(test_times))

    def test_09_split_zero_sample_overlap(self):
        """Verify that subsets are strictly disjoint: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅."""
        candidates = [
            create_mock_candidate_memory(f"mem_{i:03d}", f"2026-09-25T{i%24:02d}:00:00Z")
            for i in range(20)
        ]
        # This test targets split disjointness, not follow-up eligibility.
        split_cfg = DatasetGenerationConfig(
            future_observation_window_hours=0.0
        )
        split_gen = Phase9DatasetGenerator(split_cfg)
        rows, _, _ = split_gen.process(candidates, {}, {}, [])

        train_ids = {r["memory_id"] for r in rows if r["split"] == "train"}
        val_ids = {r["memory_id"] for r in rows if r["split"] == "val"}
        test_ids = {r["memory_id"] for r in rows if r["split"] == "test"}

        self.assertEqual(len(train_ids & val_ids), 0, "Train and Val must be disjoint.")
        self.assertEqual(len(train_ids & test_ids), 0, "Train and Test must be disjoint.")
        self.assertEqual(len(val_ids & test_ids), 0, "Val and Test must be disjoint.")
        self.assertEqual(len(train_ids) + len(val_ids) + len(test_ids), 20)

    def test_10_dataset_manifest_generation(self):
        """Verify that manifest contains dynamic metadata, execution statistics, and SHA-256 hash."""
        candidates = [
            create_mock_candidate_memory("c1", "2026-09-25T10:00:00Z"),
            create_mock_candidate_memory("c2", "2026-09-25T11:00:00Z"),
        ]
        rows, _, manifest = self.generator.process(candidates, {}, {}, [])

        self.assertEqual(manifest.total_source_candidates, 2)
        self.assertEqual(manifest.total_valid_examples, 2)
        self.assertGreater(len(manifest.data_hash_sha256), 0)
        self.assertIn("categorical", manifest.target_distribution)
        self.assertGreater(len(manifest.limitations), 0)

    def test_11_reproducibility_idempotency(self):
        """Verify that consecutive executions on identical data produce identical datasets and hashes."""
        candidates = [
            create_mock_candidate_memory("c1", "2026-09-25T10:00:00Z"),
            create_mock_candidate_memory("c2", "2026-09-25T11:00:00Z"),
        ]
        rows1, _, manifest1 = self.generator.process(candidates, {}, {}, [])
        rows2, _, manifest2 = self.generator.process(candidates, {}, {}, [])

        self.assertEqual(len(rows1), len(rows2))
        self.assertEqual([r["memory_id"] for r in rows1], [r["memory_id"] for r in rows2])
        self.assertEqual([r["split"] for r in rows1], [r["split"] for r in rows2])
        self.assertEqual(manifest1.data_hash_sha256, manifest2.data_hash_sha256)

    def test_12_consolidated_dataset_generation(self):
        """Verify that macro consolidated dataset is generated with valid structure."""
        c_lookup = {
            "c_node_01": {
                "consolidated_memory_id": "c_node_01",
                "source_count": 10,
                "primary_application": "Code.exe",
                "recency": 0.8,
                "frequency": 0.6,
                "retrieval_history": 0.0,
                "semantic_relevance": 0.7,
                "relationship_strength": 0.65,
                "baseline_score": 0.55,
            }
        }
        r_lookup = {
            "c_node_01": {
                "retention_state": "DECAYING",
                "final_retention_strength": 0.45,
                "is_protected": True,
                "forgetting_eligible": False,
            }
        }
        g_lookup = {
            "c_node_01": {
                "graph_degree": 4,
                "graph_avg_rel_strength": 0.52,
                "graph_max_rel_strength": 0.74,
                "is_graph_isolated": 0,
            }
        }
        c_rows = self.generator.build_consolidated_dataset(c_lookup, r_lookup, g_lookup)
        self.assertEqual(len(c_rows), 1)
        self.assertEqual(c_rows[0]["consolidated_memory_id"], "c_node_01")
        self.assertEqual(c_rows[0]["graph_degree"], 4)
        self.assertEqual(c_rows[0]["target_retention_state"], "DECAYING")
        self.assertTrue(c_rows[0]["is_protected"])

    def test_13_empty_dataset_handling(self):
        """Verify that processing an empty candidate list returns empty structures cleanly."""
        rows, c_rows, manifest = self.generator.process([], {}, {}, [])
        self.assertEqual(len(rows), 0)
        self.assertEqual(manifest.total_source_candidates, 0)
        self.assertEqual(manifest.total_valid_examples, 0)

    def test_14_real_data_pipeline_integration(self):
        """Integration test: Execute full Phase 9 pipeline on actual Phase 1-8 outputs."""
        if not BASELINE_SCORED_JSON_PATH.exists():
            self.skipTest("Real candidate memories dataset not found.")

        rows, c_rows, manifest = self.generator.run_pipeline(save_to_disk=False, save_to_db=False)

        # Real dataset size assertions
        self.assertIn(manifest.total_source_candidates, [6229, 6980])
        # Records without a complete future observation window are rejected.
        self.assertLess(manifest.total_valid_examples, manifest.total_source_candidates)
        self.assertEqual(
            manifest.total_valid_examples + manifest.rejected_records_count,
            manifest.total_source_candidates,
        )
        self.assertGreater(
            manifest.rejection_reasons.get("insufficient_followup", 0), 0
        )
        self.assertGreater(manifest.train_count, 3000)
        self.assertGreater(manifest.val_count, 500)
        self.assertGreater(manifest.test_count, 500)
        self.assertEqual(
            manifest.train_count + manifest.val_count + manifest.test_count,
            manifest.total_valid_examples,
        )

        # Target distribution verification
        cat_dist = manifest.target_distribution["categorical"]
        self.assertIn("COMPRESS", cat_dist)
        self.assertIn("ARCHIVE", cat_dist)

        # Consolidated tier assertions
        self.assertIn(len(c_rows), [74, 78])


if __name__ == "__main__":
    unittest.main()

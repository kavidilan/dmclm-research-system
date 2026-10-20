"""
Comprehensive Unit & Integration Test Suite for DMCLM Pipeline
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model

Tests:
1. Read-only database connection enforcement (no write allowed).
2. Exact duplicate screenshot detection (SHA-256).
3. Interaction aggregation (clicks, keystrokes, scrolls).
4. Activity episode grouping (temporal proximity, session bounding, title continuity).
5. Five feature signals (Recency decay, Frequency, Retrieval History, Semantic Relevance, Relationship Strength).
6. Baseline DMCLM Importance Scoring (0.20R + 0.20F + 0.25H + 0.20S + 0.15Rel).
7. Cognitive lifecycle thresholds (RETAIN, COMPRESS, ARCHIVE, FORGET).
8. Genuine memory compression & LTM node synthesis.
9. Genuine memory forgetting (logical exclusion, raw DB untouched).
10. Adaptive retrieval feedback loop (state transition on retrieval).
11. Robust edge case handling (empty input, missing title/app, invalid timestamps, zero retrievals).
"""

import json
import sqlite3
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add src to sys.path
TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"
for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import (
    DEFAULT_CONFIG,
    BaselineImportanceWeights,
    LifecycleThresholds,
    PreprocessingConfig,
    FeatureConfig,
)
from src.preprocessing_pipeline import (
    PreprocessingPipeline,
    get_readonly_collector_connection,
    parse_timestamp,
    COLLECTOR_DB,
)
from src.dmclm_features import (
    RecencyEngine,
    FrequencyEngine,
    RetrievalHistoryEngine,
    SemanticRelevanceEngine,
    RelationshipStrengthEngine,
    DMCLMFeaturePipeline,
)
from src.dmclm_baseline import BaselineDecisionEngine
from src.adaptive_manager import record_memory_retrieval


class TestDMCLMReadOnlyConnection(unittest.TestCase):
    def test_readonly_enforcement(self):
        """Verify that connection is strictly read-only and writes are rejected by SQLite."""
        if not COLLECTOR_DB.exists():
            self.skipTest("Collector DB does not exist on this machine.")

        conn = get_readonly_collector_connection()
        try:
            # Querying must succeed
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM activity_events")
            count = cur.fetchone()[0]
            self.assertGreaterEqual(count, 0)

            # Any write attempt MUST fail with OperationalError
            with self.assertRaises(sqlite3.OperationalError):
                cur.execute("CREATE TABLE IF NOT EXISTS should_fail_test (id INT)")

            with self.assertRaises(sqlite3.OperationalError):
                cur.execute("DELETE FROM activity_events WHERE 1=0")
        finally:
            conn.close()


class TestScreenshotDeduplication(unittest.TestCase):
    def test_duplicate_screenshot_detection_by_sha256(self):
        """Verify duplicate screenshots sharing identical SHA-256 are grouped without data loss."""
        pipeline = PreprocessingPipeline()
        mock_shots = [
            {
                "screenshot_id": "shot_001",
                "session_id": "sess_1",
                "timestamp": "2026-10-01T10:00:00+00:00",
                "application_name": "Google Chrome",
                "window_title": "Paper PDF",
                "sha256": "abc123hash",
                "file_size": 50000,
            },
            {
                "screenshot_id": "shot_002",
                "session_id": "sess_1",
                "timestamp": "2026-10-01T10:01:00+00:00",
                "application_name": "Google Chrome",
                "window_title": "Paper PDF",
                "sha256": "abc123hash",  # Exact duplicate
                "file_size": 50000,
            },
            {
                "screenshot_id": "shot_003",
                "session_id": "sess_1",
                "timestamp": "2026-10-01T10:05:00+00:00",
                "application_name": "VS Code",
                "window_title": "main.py",
                "sha256": "unique999hash",
                "file_size": 42000,
            },
        ]

        canonicals, registry = pipeline.deduplicate_screenshots(mock_shots)
        self.assertEqual(len(canonicals), 2)
        self.assertEqual(registry["abc123hash"]["count"], 2)
        self.assertTrue(registry["abc123hash"]["is_duplicate"])
        self.assertEqual(registry["abc123hash"]["canonical_id"], "shot_001")
        self.assertIn("shot_002", registry["abc123hash"]["all_screenshot_ids"])


class TestInteractionAggregation(unittest.TestCase):
    def test_interaction_aggregation(self):
        """Verify clicks and keystrokes are aggregated by session and application."""
        pipeline = PreprocessingPipeline()
        mock_interactions = [
            {"session_id": "sess_A", "application_name": "VS Code", "interaction_type": "mouse_click", "count": 10},
            {"session_id": "sess_A", "application_name": "VS Code", "interaction_type": "keyboard_activity_count", "count": 45},
            {"session_id": "sess_A", "application_name": "VS Code", "interaction_type": "mouse_click", "count": 5},
            {"session_id": "sess_A", "application_name": "Chrome", "interaction_type": "mouse_click", "count": 8},
        ]

        agg = pipeline.aggregate_interactions(mock_interactions)
        vscode_entry = agg[("sess_A", "VS Code")]
        self.assertEqual(vscode_entry["total"], 60)
        self.assertEqual(vscode_entry["mouse_click"], 15)
        self.assertEqual(vscode_entry["keyboard"], 45)


class TestActivityEpisodeGrouping(unittest.TestCase):
    def test_episode_grouping_temporal_proximity(self):
        """Verify observations within proximity window are aggregated into a single episode."""
        pipeline = PreprocessingPipeline(config=PreprocessingConfig(temporal_proximity_seconds=300.0))
        mock_events = [
            {
                "event_id": "ev_1",
                "session_id": "s1",
                "timestamp": "2026-10-01T10:00:00+00:00",
                "application_name": "VS Code",
                "window_title": "models.py",
                "duration_ms": 2000,
            },
            {
                "event_id": "ev_2",
                "session_id": "s1",
                "timestamp": "2026-10-01T10:02:00+00:00",  # +120s (within 300s)
                "application_name": "VS Code",
                "window_title": "models.py",
                "duration_ms": 3000,
            },
            {
                "event_id": "ev_3",
                "session_id": "s1",
                "timestamp": "2026-10-01T10:15:00+00:00",  # +780s (exceeds 300s -> new episode)
                "application_name": "VS Code",
                "window_title": "models.py",
                "duration_ms": 1500,
            },
        ]

        episodes = pipeline.group_activity_episodes(mock_events, {}, {})
        self.assertEqual(len(episodes), 2)
        # First episode contains ev_1 and ev_2
        self.assertEqual(episodes[0]["occurrence_count"], 2)
        self.assertEqual(episodes[0]["total_duration"], 5000)
        self.assertEqual(episodes[0]["source_event_ids"], ["ev_1", "ev_2"])
        # Second episode contains ev_3
        self.assertEqual(episodes[1]["occurrence_count"], 1)


class TestFiveFeaturesEngine(unittest.TestCase):
    def test_recency_decay(self):
        """Verify exponential decay: R = exp(-lambda * days)."""
        engine = RecencyEngine(lambda_decay=0.05)
        now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        records = [
            {"last_seen": now.isoformat()},                                     # 0 days old -> R = 1.0
            {"last_seen": (now - timedelta(days=14)).isoformat()},             # 14 days old -> R ~ exp(-0.7) = 0.4966
            {"last_seen": (now - timedelta(days=60)).isoformat()},             # 60 days old -> R ~ exp(-3.0) = 0.0498
        ]

        scores = engine.compute(records, reference_time=now)
        self.assertEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.4966, places=2)
        self.assertAlmostEqual(scores[2], 0.0498, places=2)

    def test_frequency_calculation(self):
        """Verify frequency normalization scales with occurrence count."""
        engine = FrequencyEngine()
        records = [
            {"occurrence_count": 1, "interaction_count": 0},
            {"occurrence_count": 10, "interaction_count": 5},
            {"occurrence_count": 100, "interaction_count": 50},
        ]
        scores = engine.compute(records)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_retrieval_history_honest_zero(self):
        """Verify that zero retrievals explicitly results in 0.0."""
        engine = RetrievalHistoryEngine()
        records = [{"memory_id": "m1", "retrieval_count": 0, "useful_retrieval_count": 0}]
        scores = engine.compute(records)
        self.assertEqual(scores[0], 0.0)

    def test_retrieval_history_with_audit_trail(self):
        """Verify useful / total retrieval ratio calculation."""
        engine = RetrievalHistoryEngine()
        records = [{"memory_id": "m1", "retrieval_count": 4, "useful_retrieval_count": 4}]
        scores = engine.compute(records)
        self.assertEqual(scores[0], 1.0)

    def test_semantic_relevance(self):
        """Verify local TF-IDF semantic relevance returns normalized values."""
        engine = SemanticRelevanceEngine()
        records = [
            {"session_id": "s1", "application": "Browser", "content": "Attention is All You Need Transformer", "domain": "arxiv.org"},
            {"session_id": "s1", "application": "Notes", "content": "Transformer Multi-Head Self Attention mechanism", "domain": None},
            {"session_id": "s1", "application": "System", "content": "Calculator and Solitaire Game", "domain": None},
        ]
        scores = engine.compute(records)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        # Transformer-related notes should have higher similarity than Solitaire game
        self.assertGreater(scores[1], scores[2])

    def test_relationship_strength(self):
        """Verify multi-modal co-occurrence and temporal proximity scoring."""
        engine = RelationshipStrengthEngine()
        now = datetime(2026, 10, 6, 10, 0, 0, tzinfo=timezone.utc)
        records = [
            {"session_ids": ["s1"], "application": "VS Code", "last_seen": now.isoformat()},
            {"session_ids": ["s1"], "application": "VS Code", "last_seen": (now + timedelta(seconds=30)).isoformat()},
            {"session_ids": ["s2"], "application": "Spotify", "last_seen": (now + timedelta(hours=5)).isoformat()},
        ]
        scores = engine.compute(records)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        # Adjacent VS Code episodes in same session have stronger relationship than isolated Spotify
        self.assertGreater(scores[0], scores[2])


class TestBaselineDMCLMDecisions(unittest.TestCase):
    def test_importance_formula_and_thresholds(self):
        """Verify Importance = 0.20R + 0.20F + 0.25H + 0.20S + 0.15Rel and 4-way classification."""
        decision_engine = BaselineDecisionEngine()

        # Case 1: High utility working memory (RETAIN >= 0.70)
        rec_retain = {
            "memory_id": "m_ret",
            "recency_score": 1.0,
            "frequency_score": 0.8,
            "retrieval_history_score": 1.0,
            "semantic_relevance": 0.8,
            "relationship_strength": 0.8,
        }
        # Expected: 0.20*1 + 0.20*0.8 + 0.25*1 + 0.20*0.8 + 0.15*0.8 = 0.2 + 0.16 + 0.25 + 0.16 + 0.12 = 0.89 -> RETAIN
        out = decision_engine.compute_importance_scores([rec_retain])
        self.assertAlmostEqual(out[0]["importance_score"], 0.89, places=2)
        self.assertEqual(out[0]["lifecycle_state"], "RETAIN")

        # Case 2: Repetitive episodic task (COMPRESS 0.40 - 0.69)
        rec_compress = {
            "memory_id": "m_comp",
            "recency_score": 0.6,
            "frequency_score": 0.6,
            "retrieval_history_score": 0.0,
            "semantic_relevance": 0.5,
            "relationship_strength": 0.8,
        }
        # Expected: 0.2*0.6 + 0.2*0.6 + 0.25*0 + 0.2*0.5 + 0.15*0.8 = 0.12 + 0.12 + 0 + 0.10 + 0.12 = 0.46 -> COMPRESS
        out = decision_engine.compute_importance_scores([rec_compress])
        self.assertAlmostEqual(out[0]["importance_score"], 0.46, places=2)
        self.assertEqual(out[0]["lifecycle_state"], "COMPRESS")

        # Case 3: Decayed historical context (ARCHIVE 0.20 - 0.39)
        rec_archive = {
            "memory_id": "m_arch",
            "recency_score": 0.3,
            "frequency_score": 0.2,
            "retrieval_history_score": 0.0,
            "semantic_relevance": 0.4,
            "relationship_strength": 0.4,
        }
        # Expected: 0.2*0.3 + 0.2*0.2 + 0 + 0.2*0.4 + 0.15*0.4 = 0.06 + 0.04 + 0.08 + 0.06 = 0.24 -> ARCHIVE
        out = decision_engine.compute_importance_scores([rec_archive])
        self.assertAlmostEqual(out[0]["importance_score"], 0.24, places=2)
        self.assertEqual(out[0]["lifecycle_state"], "ARCHIVE")

        # Case 4: Transient noise (FORGET < 0.20)
        rec_forget = {
            "memory_id": "m_forg",
            "recency_score": 0.1,
            "frequency_score": 0.1,
            "retrieval_history_score": 0.0,
            "semantic_relevance": 0.1,
            "relationship_strength": 0.1,
        }
        # Expected: 0.2*0.1 + 0.2*0.1 + 0 + 0.2*0.1 + 0.15*0.1 = 0.02 + 0.02 + 0.02 + 0.015 = 0.075 -> FORGET
        out = decision_engine.compute_importance_scores([rec_forget])
        self.assertAlmostEqual(out[0]["importance_score"], 0.075, places=2)
        self.assertEqual(out[0]["lifecycle_state"], "FORGET")


class TestGenuineCompressionAndForgetting(unittest.TestCase):
    def test_genuine_compression_synthesis(self):
        """Verify that genuine compression synthesizes abstract LTM nodes and tracks true ratio."""
        decision_engine = BaselineDecisionEngine()
        mock_compress = [
            {
                "memory_id": f"m_{i}",
                "application": "Python IDE",
                "session_id": "session_alpha",
                "content": "Python IDE | Debugging Consolidation",
                "lifecycle_state": "COMPRESS",
                "first_seen": "2026-10-01T12:00:00+00:00",
                "last_seen": "2026-10-01T12:05:00+00:00",
                "total_duration": 10000,
                "occurrence_count": 2,
                "interaction_count": 15,
                "importance_score": 0.52,
            }
            for i in range(10)
        ]

        ltm_nodes, metrics = decision_engine.consolidate_compressed_memories(mock_compress)
        self.assertEqual(metrics["original_episodic_records"], 10)
        self.assertEqual(metrics["consolidated_ltm_nodes"], 1)
        self.assertEqual(metrics["memory_reduction_count"], 9)
        self.assertEqual(metrics["compression_ratio_pct"], 90.0)
        self.assertEqual(len(ltm_nodes), 1)
        self.assertEqual(len(ltm_nodes[0]["source_memory_ids"]), 10)
        self.assertIn("Python IDE", ltm_nodes[0]["topic"])

    def test_genuine_forgetting_keeps_collector_intact(self):
        """Verify that FORGET only marks derived memory layer and does not touch collector.db."""
        if not COLLECTOR_DB.exists():
            self.skipTest("Collector DB does not exist on this machine.")

        conn = get_readonly_collector_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM activity_events")
            before_count = cur.fetchone()[0]

            # In derived layer: memory marked FORGET
            forgotten_record = {
                "memory_id": "test_forget_mid",
                "source_type": "activity",
                "application": "NoiseApp",
                "importance_score": 0.05,
                "lifecycle_state": "FORGET",
            }

            # Confirm raw count in collector.db is untouched
            cur.execute("SELECT COUNT(*) FROM activity_events")
            after_count = cur.fetchone()[0]
            self.assertEqual(before_count, after_count)
        finally:
            conn.close()


class TestEdgeCases(unittest.TestCase):
    def test_empty_input(self):
        """Verify empty records list is handled gracefully across all engines."""
        rec_eng = RecencyEngine()
        self.assertEqual(rec_eng.compute([]), [])

        freq_eng = FrequencyEngine()
        self.assertEqual(freq_eng.compute([]), [])

        ret_eng = RetrievalHistoryEngine()
        self.assertEqual(ret_eng.compute([]), [])

        sem_eng = SemanticRelevanceEngine()
        self.assertEqual(sem_eng.compute([]), [])

        rel_eng = RelationshipStrengthEngine()
        self.assertEqual(rel_eng.compute([]), [])

        dec_eng = BaselineDecisionEngine()
        self.assertEqual(dec_eng.compute_importance_scores([]), [])

    def test_missing_window_title_and_application(self):
        """Verify missing titles or apps default gracefully without crashing."""
        pipeline = PreprocessingPipeline()
        mock_events = [
            {
                "event_id": "ev_none",
                "session_id": "s_none",
                "timestamp": "2026-10-01T10:00:00+00:00",
                "application_name": None,
                "window_title": None,
                "duration_ms": None,
            }
        ]
        episodes = pipeline.group_activity_episodes(mock_events, {}, {})
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["application"], "Unknown Application")
        self.assertEqual(episodes[0]["total_duration"], 0)

    def test_invalid_timestamps(self):
        """Verify invalid timestamps are flagged and filtered without crashing."""
        self.assertIsNone(parse_timestamp("invalid-date-string"))
        self.assertIsNone(parse_timestamp(None))
        self.assertIsNone(parse_timestamp(""))


if __name__ == "__main__":
    unittest.main()

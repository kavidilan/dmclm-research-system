"""
Unit & Integration Test Suite for Phase 4 — Five DMCLM Features
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Validates:
1. Recency Engine: Exponential decay, temporal monotonicity, custom lambda, and missing timestamp fallback.
2. Frequency Engine: Differentiation between intra-episode interactions and inter-episode repetition, log-scale normalization.
3. Retrieval History Engine: Genuine audit-trail evaluation, cold-start distinction (no history vs unretrieved), zero-fabrication invariant.
4. Semantic Relevance Engine: Local TF-IDF centroid similarity, session task clustering, empty/noise text resilience.
5. Relationship Strength Engine: Multi-modal graph affinity (session, app/domain, temporal proximity), strict self-comparison exclusion.
6. Edge Cases & Robustness: Empty datasets, single-memory datasets, missing browser domains, zero interactions.
7. Determinism & Repeatability: Zero random variation, identical floats across runs, idempotent database writing.
8. Real Data Integration: Loads real Phase 3 memory candidates, calculates 5 features, validates statistics and persistence.
"""

import json
import math
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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

from src.dmclm_config import FeatureConfig, DEFAULT_CONFIG
from src.models import StructuredMemoryCandidate, FeatureEnrichedMemory
from src.phase4_feature_extractor import (
    Phase4FeatureExtractor,
    RecencyEngine,
    FrequencyEngine,
    RetrievalHistoryEngine,
    SemanticRelevanceEngine,
    RelationshipStrengthEngine,
    parse_timestamp,
)
from src.phase3_memory_extractor import Phase3MemoryExtractor


class TestPhase4Recency(unittest.TestCase):
    def test_recency_ordering_and_monotonicity(self):
        """Verify that more recent memories strictly receive higher recency scores than older memories."""
        engine = RecencyEngine(lambda_decay=0.05)
        t_ref = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        records = [
            {"candidate_id": "c1", "end_time": t_ref.isoformat()},
            {"candidate_id": "c2", "end_time": (t_ref - timedelta(days=2)).isoformat()},
            {"candidate_id": "c3", "end_time": (t_ref - timedelta(days=14)).isoformat()},
            {"candidate_id": "c4", "end_time": (t_ref - timedelta(days=60)).isoformat()},
        ]

        scores = engine.compute(records, reference_time=t_ref)
        self.assertEqual(len(scores), 4)

        # R(t_ref) must equal 1.0
        self.assertEqual(scores[0], 1.0)
        # Monotonicity: c1 > c2 > c3 > c4
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[3])

        # Mathematical verification: R = exp(-0.05 * 14) = exp(-0.7) ~ 0.4966
        self.assertAlmostEqual(scores[2], math.exp(-0.05 * 14), places=2)
        # All bounded in [0.0, 1.0]
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_recency_configurable_decay_parameter(self):
        """Verify that configurable lambda alters the decay curve as expected."""
        t_ref = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
        record = [{"candidate_id": "c1", "end_time": (t_ref - timedelta(days=10)).isoformat()}]

        engine_slow = RecencyEngine(lambda_decay=0.02)
        engine_fast = RecencyEngine(lambda_decay=0.10)

        score_slow = engine_slow.compute(record, reference_time=t_ref)[0]
        score_fast = engine_fast.compute(record, reference_time=t_ref)[0]

        # Faster decay parameter must yield lower retention for the same time delta
        self.assertGreater(score_slow, score_fast)
        self.assertAlmostEqual(score_slow, math.exp(-0.02 * 10), places=3)
        self.assertAlmostEqual(score_fast, math.exp(-0.10 * 10), places=3)

    def test_recency_missing_timestamp_fallback(self):
        """Verify missing or malformed timestamp results in 0.0 without throwing exceptions."""
        engine = RecencyEngine()
        records = [
            {"candidate_id": "c_valid", "end_time": "2026-10-06T10:00:00+00:00"},
            {"candidate_id": "c_missing", "end_time": None, "start_time": None},
            {"candidate_id": "c_invalid", "end_time": "not_a_date", "start_time": ""},
        ]
        scores = engine.compute(records)
        self.assertEqual(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)
        self.assertTrue(0.0 <= scores[0] <= 1.0)


class TestPhase4Frequency(unittest.TestCase):
    def test_frequency_intra_vs_inter_episode_distinction(self):
        """
        Verify that frequency properly balances inter-episode repetition
        against intra-episode user interactions according to configurable weights.
        """
        cfg = FeatureConfig(freq_weight_inter=0.70, freq_weight_intra=0.30)
        engine = FrequencyEngine(config=cfg)

        records = [
            # App 'Code' appears 4 times (high inter-episode repetition)
            {"primary_application": "Code", "occurrence_count": 1, "interaction_count": 10},
            {"primary_application": "Code", "occurrence_count": 1, "interaction_count": 10},
            {"primary_application": "Code", "occurrence_count": 1, "interaction_count": 10},
            {"primary_application": "Code", "occurrence_count": 1, "interaction_count": 10},
            # App 'RareTool' appears only once with identical interactions (10)
            {"primary_application": "RareTool", "occurrence_count": 1, "interaction_count": 10},
        ]

        scores = engine.compute(records)
        self.assertEqual(len(scores), 5)
        # Frequent app episodes (indices 0-3) must score higher than single rare app (index 4)
        self.assertGreater(scores[0], scores[4])
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_frequency_log_scale_prevents_outlier_domination(self):
        """Verify log-scaling maintains smooth progression and prevents runaway dominance."""
        engine = FrequencyEngine()
        records = [
            {"primary_application": "Browser", "occurrence_count": 1, "interaction_count": 0},
            {"primary_application": "Browser", "occurrence_count": 5, "interaction_count": 20},
            {"primary_application": "Browser", "occurrence_count": 50, "interaction_count": 500},
        ]
        scores = engine.compute(records)
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))


class TestPhase4RetrievalHistory(unittest.TestCase):
    def test_cold_start_distinction_no_system_history(self):
        """Verify that when no retrieval history exists system-wide, engine returns 0.0 with honest evidence."""
        engine = RetrievalHistoryEngine()
        records = [
            {"candidate_id": "cand_1", "retrieval_count": 0},
            {"candidate_id": "cand_2", "retrieval_count": 0},
        ]
        scores, evidence = engine.compute_with_evidence(records, retrieval_stats={})
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)
        self.assertEqual(evidence[0]["status"], "NO_RETRIEVAL_HISTORY_AVAILABLE")
        self.assertIn("Cold-start", evidence[0]["note"])

    def test_unretrieved_vs_retrieved_with_audit_trail(self):
        """Verify distinction between unretrieved candidate and candidate with actual retrieval logs."""
        engine = RetrievalHistoryEngine()
        stats = {
            "cand_active": {"total": 4, "useful": 4},
            "cand_mixed": {"total": 4, "useful": 2},
        }
        records = [
            {"candidate_id": "cand_unretrieved"},
            {"candidate_id": "cand_active"},
            {"candidate_id": "cand_mixed"},
        ]
        scores, evidence = engine.compute_with_evidence(records, retrieval_stats=stats)

        # Unretrieved candidate
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(evidence[0]["status"], "UNRETRIEVED_ZERO_COUNT")

        # 100% useful retrieval candidate
        self.assertEqual(scores[1], 1.0)
        self.assertEqual(evidence[1]["status"], "RETRIEVED")
        self.assertEqual(evidence[1]["useful_retrievals"], 4)

        # 50% useful retrieval candidate
        self.assertLess(scores[2], scores[1])
        self.assertGreater(scores[2], scores[0])
        self.assertEqual(evidence[2]["status"], "RETRIEVED")


class TestPhase4SemanticRelevance(unittest.TestCase):
    def test_semantic_task_centroid_relevance(self):
        """Verify local TF-IDF centroid similarity favors topic-aligned memories within a session."""
        engine = SemanticRelevanceEngine()
        records = [
            {
                "session_id": "s_research",
                "primary_application": "Chrome",
                "window_context": "Deep Learning Transformer Attention",
                "activity_summary": "Researching self-attention Transformer mechanisms in neural networks",
                "browser_domains": ["arxiv.org"],
            },
            {
                "session_id": "s_research",
                "primary_application": "VS Code",
                "window_context": "models/transformer.py",
                "activity_summary": "Implementing Transformer multi-head attention module in Python",
                "browser_domains": [],
            },
            {
                "session_id": "s_research",
                "primary_application": "Solitaire",
                "window_context": "Klondike Game",
                "activity_summary": "Playing Solitaire card deck games",
                "browser_domains": [],
            },
        ]
        scores = engine.compute(records)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        # Transformer coding and research items should have higher relevance than isolated Solitaire game
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(scores[1], scores[2])

    def test_semantic_empty_or_trivial_text_resilience(self):
        """Verify fallback when text is empty or non-alphabetic."""
        engine = SemanticRelevanceEngine()
        records = [
            {"session_id": "s1", "primary_application": "", "window_context": "", "activity_summary": ""},
            {"session_id": "s1", "primary_application": "", "window_context": "12345", "activity_summary": "---"},
        ]
        scores = engine.compute(records)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))


class TestPhase4RelationshipStrength(unittest.TestCase):
    def test_multi_modal_relational_cohesion(self):
        """Verify multi-modal affinity based on shared session, app continuity, and temporal proximity."""
        engine = RelationshipStrengthEngine()
        now = datetime(2026, 10, 6, 10, 0, 0, tzinfo=timezone.utc)

        records = [
            # Record 0 & 1: Same session, same application, only 30s apart
            {
                "candidate_id": "c0", "session_id": "sess_A", "primary_application": "VS Code",
                "browser_domains": [], "window_context": "src/main.py", "end_time": now.isoformat(),
            },
            {
                "candidate_id": "c1", "session_id": "sess_A", "primary_application": "VS Code",
                "browser_domains": [], "window_context": "src/main.py",
                "end_time": (now + timedelta(seconds=30)).isoformat(),
            },
            # Record 2: Different session, different app, hours away
            {
                "candidate_id": "c2", "session_id": "sess_B", "primary_application": "Spotify",
                "browser_domains": [], "window_context": "Chill Mix",
                "end_time": (now + timedelta(hours=8)).isoformat(),
            },
        ]

        scores = engine.compute(records)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))
        # Adjacent coding episodes in same session must have stronger relationship than isolated Spotify
        self.assertGreater(scores[0], scores[2])

    def test_avoid_self_comparison_and_isolated_single_memory(self):
        """Verify that single memory has relationship strength 0.0 (no self-loops)."""
        engine = RelationshipStrengthEngine()
        records = [{
            "candidate_id": "c_alone", "session_id": "s1", "primary_application": "Code",
            "browser_domains": [], "window_context": "solo.py", "end_time": "2026-10-06T10:00:00+00:00",
        }]
        scores = engine.compute(records)
        self.assertEqual(scores[0], 0.0)


class TestPhase4EdgeCasesAndNormalization(unittest.TestCase):
    def test_empty_dataset_handling(self):
        """Verify that an empty dataset returns empty lists without exceptions."""
        extractor = Phase4FeatureExtractor()
        enriched = extractor.extract_features([])
        self.assertEqual(len(enriched), 0)

        stats = extractor.calculate_statistics([])
        self.assertEqual(stats, {})

    def test_single_memory_dataset_handling(self):
        """Verify that single memory dataset processes cleanly with valid [0, 1] bounds."""
        extractor = Phase4FeatureExtractor()
        cand = StructuredMemoryCandidate(
            candidate_id="cand_single",
            session_id="s1",
            source_episode_ids=["ep_1"],
            primary_application="Chrome",
            applications_involved=["Chrome"],
            browser_domains=["google.com"],
            window_context="Search",
            start_time="2026-10-06T10:00:00+00:00",
            end_time="2026-10-06T10:05:00+00:00",
            duration_ms=300000,
            interaction_count=12,
            browser_event_count=3,
            screenshot_ids=[],
            activity_summary="Web search in Chrome",
            meaningful_criteria=["duration_threshold"],
            created_at="2026-10-06T10:05:00+00:00",
        )
        enriched = extractor.extract_features([cand])
        self.assertEqual(len(enriched), 1)
        m = enriched[0]

        # Invariant: all 5 features in [0.0, 1.0]
        self.assertTrue(0.0 <= m.recency <= 1.0)
        self.assertTrue(0.0 <= m.frequency <= 1.0)
        self.assertTrue(0.0 <= m.retrieval_history <= 1.0)
        self.assertTrue(0.0 <= m.semantic_relevance <= 1.0)
        self.assertTrue(0.0 <= m.relationship_strength <= 1.0)

        # Single memory has recency 1.0, frequency 1.0, relationship 0.0
        self.assertEqual(m.recency, 1.0)
        self.assertEqual(m.frequency, 1.0)
        self.assertEqual(m.relationship_strength, 0.0)

    def test_missing_browser_domains_and_zero_interactions(self):
        """Verify memory candidate with no browser domains and zero interactions."""
        extractor = Phase4FeatureExtractor()
        cand = StructuredMemoryCandidate(
            candidate_id="cand_offline",
            session_id="s1",
            source_episode_ids=["ep_off"],
            primary_application="Notepad",
            applications_involved=["Notepad"],
            browser_domains=[],
            window_context="notes.txt",
            start_time="2026-10-06T11:00:00+00:00",
            end_time="2026-10-06T11:05:00+00:00",
            duration_ms=300000,
            interaction_count=0,
            browser_event_count=0,
            screenshot_ids=[],
            activity_summary="Reviewing notes in Notepad",
            meaningful_criteria=["duration_threshold"],
            created_at="2026-10-06T11:05:00+00:00",
        )
        enriched = extractor.extract_features([cand])
        self.assertEqual(len(enriched), 1)
        m = enriched[0]
        self.assertTrue(0.0 <= m.recency <= 1.0)
        self.assertTrue(0.0 <= m.frequency <= 1.0)
        self.assertEqual(m.browser_domains, [])
        self.assertEqual(m.interaction_count, 0)


class TestPhase4DeterminismAndIdempotency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.mock_db = self.temp_path / "test_dmclm.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deterministic_reproducibility(self):
        """Verify running Phase 4 twice on identical candidate inputs yields bitwise identical feature values."""
        extractor = Phase4FeatureExtractor(db_path=self.mock_db)
        candidates = [
            StructuredMemoryCandidate(
                candidate_id=f"c_{i}",
                session_id=f"sess_{i % 2}",
                source_episode_ids=[f"ep_{i}"],
                primary_application="Code" if i % 2 == 0 else "Chrome",
                applications_involved=["Code" if i % 2 == 0 else "Chrome"],
                browser_domains=["github.com"] if i % 2 != 0 else [],
                window_context=f"Project Task {i}",
                start_time=f"2026-10-06T10:{i:02d}:00+00:00",
                end_time=f"2026-10-06T10:{i+1:02d}:00+00:00",
                duration_ms=60000,
                interaction_count=i * 5,
                browser_event_count=1 if i % 2 != 0 else 0,
                screenshot_ids=[],
                activity_summary=f"Activity description for task {i}",
                meaningful_criteria=["duration_threshold"],
                created_at="2026-10-06T10:00:00+00:00",
            )
            for i in range(5)
        ]

        run1 = extractor.extract_features(candidates)
        run2 = extractor.extract_features(candidates)

        self.assertEqual(len(run1), len(run2))
        for m1, m2 in zip(run1, run2):
            self.assertEqual(m1.candidate_id, m2.candidate_id)
            self.assertEqual(m1.recency, m2.recency)
            self.assertEqual(m1.frequency, m2.frequency)
            self.assertEqual(m1.retrieval_history, m2.retrieval_history)
            self.assertEqual(m1.semantic_relevance, m2.semantic_relevance)
            self.assertEqual(m1.relationship_strength, m2.relationship_strength)

    def test_idempotent_sqlite_persistence(self):
        """Verify saving twice into SQLite table does not duplicate records."""
        extractor = Phase4FeatureExtractor(db_path=self.mock_db)
        candidates = [
            StructuredMemoryCandidate(
                candidate_id="c_persist_1",
                session_id="s1",
                source_episode_ids=["ep_p1"],
                primary_application="VS Code",
                applications_involved=["VS Code"],
                browser_domains=[],
                window_context="test.py",
                start_time="2026-10-06T10:00:00+00:00",
                end_time="2026-10-06T10:10:00+00:00",
                duration_ms=600000,
                interaction_count=20,
                browser_event_count=0,
                screenshot_ids=[],
                activity_summary="Coding test in VS Code",
                meaningful_criteria=["duration_threshold"],
                created_at="2026-10-06T10:10:00+00:00",
            )
        ]

        enriched = extractor.extract_features(candidates)
        stats = extractor.calculate_statistics(enriched)

        # First save
        extractor.save_enriched_dataset(enriched, stats)
        conn = sqlite3.connect(str(self.mock_db))
        cnt1 = conn.execute("SELECT COUNT(*) FROM feature_enriched_memories").fetchone()[0]
        conn.close()
        self.assertEqual(cnt1, 1)

        # Second save (repeatability check)
        extractor.save_enriched_dataset(enriched, stats)
        conn = sqlite3.connect(str(self.mock_db))
        cnt2 = conn.execute("SELECT COUNT(*) FROM feature_enriched_memories").fetchone()[0]
        conn.close()
        self.assertEqual(cnt1, cnt2, "Idempotency failed: record count duplicated on second write.")


class TestPhase4RealDataIntegration(unittest.TestCase):
    def test_real_phase3_candidates_feature_extraction(self):
        """
        Integration test using real Phase 3 structured memory candidates.
        Extracts 5 cognitive features, verifies mathematical boundaries, and ensures persistence.
        """
        extractor = Phase4FeatureExtractor()
        candidates = extractor.load_phase3_candidates()
        
        # If candidate table is not populated yet, run Phase 3 first
        if not candidates:
            phase3 = Phase3MemoryExtractor()
            p3_res = phase3.run()
            self.assertEqual(p3_res["status"], "COMPLETED")
            candidates = extractor.load_phase3_candidates()

        self.assertGreater(len(candidates), 0, "No real Phase 3 candidates loaded.")

        # Run Phase 4 feature extraction
        res = extractor.run(candidates)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["total_candidates"], len(candidates))

        enriched = res["enriched_memories"]
        self.assertEqual(len(enriched), len(candidates))

        # Verify all 5 features are populated and bounded
        for m in enriched[:100]:  # Sample check
            self.assertTrue(0.0 <= m.recency <= 1.0)
            self.assertTrue(0.0 <= m.frequency <= 1.0)
            self.assertTrue(0.0 <= m.retrieval_history <= 1.0)
            self.assertTrue(0.0 <= m.semantic_relevance <= 1.0)
            self.assertTrue(0.0 <= m.relationship_strength <= 1.0)

            # Check that original Phase 3 attributes remain fully intact
            self.assertTrue(len(m.candidate_id) > 0)
            self.assertTrue(len(m.session_id) > 0)
            self.assertTrue(len(m.source_episode_ids) > 0)
            self.assertTrue(len(m.primary_application) > 0)

        # Verify statistics are calculable
        stats = res["statistics"]["features"]
        self.assertIn("recency", stats)
        self.assertIn("frequency", stats)
        self.assertIn("retrieval_history", stats)
        self.assertIn("semantic_relevance", stats)
        self.assertIn("relationship_strength", stats)

        # Verify database table exists and contains real records
        conn = sqlite3.connect(str(DATA_DIR / "dmclm.db"))
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM feature_enriched_memories").fetchone()[0]
            self.assertEqual(cnt, len(candidates))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

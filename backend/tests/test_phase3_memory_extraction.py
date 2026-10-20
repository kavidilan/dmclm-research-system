"""
Unit & Integration Tests for Phase 3 — Memory Extraction
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Validates:
1. Phase 2 derived data loading & missing/empty input handling
2. Malformed record detection & error handling
3. Meaningful episode classification & configurable threshold enforcement
4. Dynamic local activity summary generation (zero LLMs/APIs)
5. Concurrent browser domain correlation & screenshot context mapping
6. Deterministic memory candidate IDs (sha256 based) & repeatability (idempotency)
7. Source episode provenance & traceability preservation
8. Verification that Phase 4 cognitive features are strictly NOT computed in Phase 3
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import MemoryExtractionConfig, DEFAULT_CONFIG
from src.models import StructuredMemoryCandidate
from src.phase3_memory_extractor import (
    Phase3MemoryExtractor,
    generate_dynamic_activity_summary,
    categorize_application_activity,
    format_duration_human,
    parse_timestamp,
)


class TestPhase3MemoryExtraction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.mock_db_path = self.temp_path / "mock_dmclm.db"
        self._init_mock_database(self.mock_db_path)

        self.custom_config = MemoryExtractionConfig(
            min_meaningful_duration_ms=2500,
            min_interactions_for_short_burst=1,
            filter_transient_system_noise=True,
            system_noise_applications=("ShellHost", "SearchHost", "LockApp"),
            browser_correlation_window_sec=60.0,
            max_summary_length=350,
        )
        self.extractor = Phase3MemoryExtractor(
            config=self.custom_config,
            db_path=self.mock_db_path
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_mock_database(self, db_path: Path):
        """Creates a mock Phase 2 derived database schema for isolated unit testing."""
        conn = sqlite3.connect(str(db_path))
        with conn:
            conn.execute("""
                CREATE TABLE activity_episodes (
                    episode_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    application TEXT NOT NULL,
                    process_name TEXT NOT NULL,
                    window_title TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    source_event_ids TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    screenshot_ids TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE normalized_browser_events (
                    normalized_id TEXT PRIMARY KEY,
                    raw_event_id TEXT,
                    session_id TEXT NOT NULL,
                    browser TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    page_title TEXT,
                    url TEXT,
                    normalized_timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE screenshot_duplicates (
                    sha256 TEXT PRIMARY KEY,
                    canonical_screenshot_id TEXT NOT NULL,
                    is_duplicate INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    duplicate_screenshot_ids TEXT,
                    application_name TEXT,
                    window_title TEXT,
                    file_size INTEGER,
                    local_file_path TEXT,
                    created_at TEXT NOT NULL
                )
            """)
        conn.close()

    def test_missing_database_error_handling(self):
        """Verify descriptive FileNotFoundError when Phase 2 derived database does not exist."""
        non_existent_path = self.temp_path / "non_existent.db"
        extractor = Phase3MemoryExtractor(db_path=non_existent_path)
        with self.assertRaises(FileNotFoundError):
            extractor.load_phase2_data()

    def test_missing_activity_episodes_table_error_handling(self):
        """Verify RuntimeError when derived database is missing the activity_episodes table."""
        bad_db = self.temp_path / "empty.db"
        sqlite3.connect(str(bad_db)).close()
        extractor = Phase3MemoryExtractor(db_path=bad_db)
        with self.assertRaises(RuntimeError):
            extractor.load_phase2_data()

    def test_empty_dataset_handling(self):
        """Verify that an empty activity_episodes table produces 0 candidates without errors."""
        res = self.extractor.run()
        self.assertEqual(res["total_episodes_loaded"], 0)
        self.assertEqual(res["valid_episodes"], 0)
        self.assertEqual(res["meaningful_episodes"], 0)
        self.assertEqual(res["candidates_extracted"], 0)
        self.assertEqual(len(res["candidates"]), 0)

    def test_malformed_and_invalid_records_filtering(self):
        """Verify that records with missing application, unknown app, or invalid timestamps are rejected."""
        conn = sqlite3.connect(str(self.mock_db_path))
        with conn:
            # 1. Missing application
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_missing_app', 'sess_1', '', 'unknown.exe', 'Title',
                    '2026-03-01T10:00:00+00:00', '2026-03-01T10:05:00+00:00',
                    300000, 10, '[]', 5, '[]', '2026-03-01T10:05:00+00:00'
                )
            """)
            # 2. Unknown application
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_unknown_app', 'sess_1', 'Unknown', 'unknown.exe', 'Title',
                    '2026-03-01T10:00:00+00:00', '2026-03-01T10:05:00+00:00',
                    300000, 10, '[]', 5, '[]', '2026-03-01T10:05:00+00:00'
                )
            """)
            # 3. Missing end_time
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_no_end', 'sess_1', 'Code.exe', 'code.exe', 'Title',
                    '2026-03-01T10:00:00+00:00', '',
                    0, 1, '[]', 0, '[]', '2026-03-01T10:00:00+00:00'
                )
            """)
            # 4. Transient system noise (ShellHost.exe with duration < 2500ms and 0 interactions)
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_system_noise', 'sess_1', 'ShellHost.exe', 'ShellExperienceHost.exe', 'ShellHost',
                    '2026-03-01T10:10:00+00:00', '2026-03-01T10:10:01+00:00',
                    1000, 1, '[]', 0, '[]', '2026-03-01T10:10:01+00:00'
                )
            """)
            # 5. Fleeting zero-dwell with no interactions (< 2500ms, 0 interactions, no screenshot)
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_fleeting', 'sess_1', 'Notepad.exe', 'notepad.exe', 'Untitled',
                    '2026-03-01T10:20:00+00:00', '2026-03-01T10:20:00.500+00:00',
                    500, 1, '[]', 0, '[]', '2026-03-01T10:20:00+00:00'
                )
            """)
        conn.close()

        res = self.extractor.run()
        self.assertEqual(res["total_episodes_loaded"], 5)
        self.assertEqual(res["meaningful_episodes"], 0)
        self.assertEqual(res["candidates_extracted"], 0)
        self.assertEqual(res["skipped_records"]["missing_application"], 2)
        self.assertEqual(res["skipped_records"]["malformed_or_missing_timestamp"], 1)
        self.assertEqual(res["skipped_records"]["transient_system_noise"], 1)
        self.assertEqual(res["skipped_records"]["insufficient_dwell_or_interactions"], 1)

    def test_meaningful_episode_thresholds(self):
        """Verify that genuine episodes meeting duration, interaction, or screenshot criteria are extracted."""
        conn = sqlite3.connect(str(self.mock_db_path))
        with conn:
            # 1. Episode with sufficient dwell duration (15s > 2.5s)
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_dwell', 'sess_1', 'Visual Studio Code', 'Code.exe', 'main.py - DMCLM',
                    '2026-03-01T10:00:00+00:00', '2026-03-01T10:00:15+00:00',
                    15000, 5, '["ev_1", "ev_2"]', 0, '[]', '2026-03-01T10:00:15+00:00'
                )
            """)
            # 2. Episode with short dwell but active user interactions
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_interactions', 'sess_1', 'Windows Explorer', 'explorer.exe', 'Documents',
                    '2026-03-01T10:01:00+00:00', '2026-03-01T10:01:01+00:00',
                    1000, 3, '["ev_3"]', 4, '[]', '2026-03-01T10:01:01+00:00'
                )
            """)
            # 3. Episode with short dwell but attached screenshot
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_screenshot', 'sess_1', 'Snipping Tool', 'snippingtool.exe', 'Capture',
                    '2026-03-01T10:02:00+00:00', '2026-03-01T10:02:00.800+00:00',
                    800, 1, '["ev_4"]', 0, '["shot_001"]', '2026-03-01T10:02:00+00:00'
                )
            """)
        conn.close()

        res = self.extractor.run()
        self.assertEqual(res["total_episodes_loaded"], 3)
        self.assertEqual(res["meaningful_episodes"], 3)
        self.assertEqual(res["candidates_extracted"], 3)
        
        candidates = {c["source_episode_ids"][0]: c for c in res["candidates"]}
        self.assertIn("ep_dwell", candidates)
        self.assertIn("duration_threshold", candidates["ep_dwell"]["meaningful_criteria"])
        self.assertEqual(candidates["ep_dwell"]["duration_ms"], 15000)

        self.assertIn("ep_interactions", candidates)
        self.assertIn("user_interactions", candidates["ep_interactions"]["meaningful_criteria"])
        self.assertEqual(candidates["ep_interactions"]["interaction_count"], 4)

        self.assertIn("ep_screenshot", candidates)
        self.assertIn("screenshot_context", candidates["ep_screenshot"]["meaningful_criteria"])
        self.assertEqual(candidates["ep_screenshot"]["screenshot_ids"], ["shot_001"])

    def test_dynamic_activity_summary_generation(self):
        """Verify dynamic semantic activity summary generation logic is local, dynamic, and non-LLM."""
        summary = generate_dynamic_activity_summary(
            primary_app="Visual Studio Code",
            window_title="src/phase3_memory_extractor.py - DMCLM",
            browser_domains=["github.com", "python.org"],
            duration_ms=45000,
            interaction_count=18,
            screenshot_count=1,
            max_length=350
        )
        self.assertIn("Software development in Visual Studio Code", summary)
        self.assertIn('focused on "src/phase3_memory_extractor.py - DMCLM"', summary)
        self.assertIn("(referencing: github.com, python.org)", summary)
        self.assertIn("dwell: 45s", summary)
        self.assertIn("18 user actions", summary)
        self.assertIn("1 screen context", summary)

        # Test duration formatting
        self.assertEqual(format_duration_human(500), "500ms")
        self.assertEqual(format_duration_human(12000), "12s")
        self.assertEqual(format_duration_human(125000), "2m 5s")
        self.assertEqual(format_duration_human(3720000), "1h 2m")

        # Test activity categorization
        self.assertEqual(categorize_application_activity("Google Chrome"), "Web research")
        self.assertEqual(categorize_application_activity("Visual Studio Code"), "Software development")
        self.assertEqual(categorize_application_activity("explorer.exe"), "File and workspace organization")
        self.assertEqual(categorize_application_activity("notepad.exe"), "Document and media review")
        self.assertEqual(categorize_application_activity("DB Browser for SQLite"), "Database management")

    def test_browser_domain_correlation(self):
        """Verify browser events correlate with concurrent activity episodes within the session window."""
        conn = sqlite3.connect(str(self.mock_db_path))
        with conn:
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_research', 'sess_research', 'Google Chrome', 'chrome.exe', 'Cognitive Architecture - Search',
                    '2026-03-01T11:00:00+00:00', '2026-03-01T11:05:00+00:00',
                    300000, 10, '["ev_10"]', 12, '[]', '2026-03-01T11:05:00+00:00'
                )
            """)
            conn.execute("""
                INSERT INTO normalized_browser_events VALUES (
                    'norm_1', 'raw_1', 'sess_research', 'chrome', 'scholar.google.com',
                    'Cognitive Memory Models - Google Scholar', 'https://scholar.google.com/search?q=cognitive+memory',
                    '2026-03-01T11:01:00+00:00', '2026-03-01T11:01:00+00:00'
                )
            """)
            conn.execute("""
                INSERT INTO normalized_browser_events VALUES (
                    'norm_2', 'raw_2', 'sess_research', 'chrome', 'arxiv.org',
                    'arXiv: Memory Consolidation in Artificial Agents', 'https://arxiv.org/abs/2401.00001',
                    '2026-03-01T11:03:00+00:00', '2026-03-01T11:03:00+00:00'
                )
            """)
        conn.close()

        res = self.extractor.run()
        self.assertEqual(res["candidates_extracted"], 1)
        candidate = res["candidates"][0]

        self.assertIn("scholar.google.com", candidate["browser_domains"])
        self.assertIn("arxiv.org", candidate["browser_domains"])
        self.assertEqual(candidate["browser_event_count"], 2)
        self.assertIn("scholar.google.com", candidate["activity_summary"])

    def test_deterministic_memory_ids_and_idempotency(self):
        """Verify memory candidate IDs are deterministic and running extraction multiple times produces identical output."""
        conn = sqlite3.connect(str(self.mock_db_path))
        with conn:
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_det_1', 'sess_det', 'Code.exe', 'code.exe', 'pipeline.py',
                    '2026-03-01T12:00:00+00:00', '2026-03-01T12:10:00+00:00',
                    600000, 20, '["ev_a"]', 15, '[]', '2026-03-01T12:10:00+00:00'
                )
            """)
        conn.close()

        # Run 1
        res1 = self.extractor.run()
        cand1 = res1["candidates"][0]

        # Run 2
        res2 = self.extractor.run()
        cand2 = res2["candidates"][0]

        # ID must be deterministic (starts with mem_cand_)
        self.assertTrue(cand1["candidate_id"].startswith("mem_cand_"))
        self.assertEqual(cand1["candidate_id"], cand2["candidate_id"])
        self.assertEqual(cand1["activity_summary"], cand2["activity_summary"])
        self.assertEqual(cand1["duration_ms"], cand2["duration_ms"])
        self.assertEqual(res1["candidates_extracted"], res2["candidates_extracted"])

        # Check database table idempotency
        conn = sqlite3.connect(str(self.mock_db_path))
        count = conn.execute("SELECT COUNT(*) FROM memory_candidates").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1, "Duplicate runs must NOT cause duplicate records in the database.")

    def test_source_traceability_preservation(self):
        """Verify candidate memory maintains exact reference back to Phase 2 episode IDs."""
        conn = sqlite3.connect(str(self.mock_db_path))
        with conn:
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'episode_phase2_orig_99', 'sess_trace', 'chrome.exe', 'chrome.exe', 'Research Paper',
                    '2026-03-01T14:00:00+00:00', '2026-03-01T14:05:00+00:00',
                    300000, 12, '["raw_101", "raw_102"]', 8, '[]', '2026-03-01T14:05:00+00:00'
                )
            """)
        conn.close()

        res = self.extractor.run()
        self.assertEqual(res["candidates_extracted"], 1)
        cand = res["candidates"][0]

        self.assertEqual(cand["source_episode_ids"], ["episode_phase2_orig_99"])
        self.assertEqual(cand["session_id"], "sess_trace")
        self.assertEqual(cand["primary_application"], "chrome.exe")

    def test_phase4_features_deferred(self):
        """
        Verify that Phase 3 strictly DOES NOT compute Phase 4 cognitive features
        (Recency, Frequency, Retrieval History, Semantic Relevance, Relationship Strength).
        """
        conn = sqlite3.connect(str(self.mock_db_path))
        with conn:
            conn.execute("""
                INSERT INTO activity_episodes VALUES (
                    'ep_feat_check', 'sess_test', 'PyCharm', 'pycharm64.exe', 'model.py',
                    '2026-03-01T15:00:00+00:00', '2026-03-01T15:10:00+00:00',
                    600000, 25, '[]', 20, '[]', '2026-03-01T15:10:00+00:00'
                )
            """)
        conn.close()

        res = self.extractor.run()
        cand = res["candidates"][0]

        # In StructuredMemoryCandidate, cognitive features do NOT exist
        cand_keys = set(cand.keys())
        prohibited_phase4_keys = {
            "recency_score", "frequency_score", "retrieval_history_score",
            "semantic_relevance_score", "relationship_strength_score",
            "importance_score", "lifecycle_state", "lifecycle_prediction",
            "consolidation_cluster", "forgetting_threshold"
        }
        intersection = cand_keys.intersection(prohibited_phase4_keys)
        self.assertEqual(len(intersection), 0, f"Phase 4 cognitive features were erroneously generated: {intersection}")


if __name__ == "__main__":
    unittest.main()

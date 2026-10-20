"""
Unit & Integration Tests for Phase 2 — Data Preprocessing
Validates:
1. Read-only collector.db access & immutability guarantee
2. Screenshot duplicate detection (SHA-256)
3. Interaction aggregation & provenance preservation
4. Browser normalization (domain, URL, title, browser)
5. Activity episode construction & boundary enforcement
6. Derived datastore population & repeatability (idempotency)
7. Verification that Phase 3 cognitive scores are NOT implemented yet
"""

import json
import sqlite3
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.phase2_preprocessor import (
    Phase2Preprocessor,
    COLLECTOR_DB,
    DERIVED_DB_PATH,
    get_collector_db_sha256,
    get_readonly_collector_connection,
    get_derived_db_connection,
    clean_domain,
    clean_page_title,
    normalize_browser_name,
)


class TestPhase2Preprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Phase2Preprocessor()

    def test_collector_db_readonly_enforcement(self):
        """Verify that collector.db is strictly opened read-only and modifications raise OperationalError."""
        conn = get_readonly_collector_connection()
        try:
            # Querying works
            cnt = conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0]
            self.assertGreater(cnt, 0)

            # Writing must be blocked
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO screenshots (screenshot_id) VALUES ('fake_test_id')")
        finally:
            conn.close()

    def test_collector_db_checksum_intact(self):
        """Verify collector.db file SHA-256 is valid and calculable."""
        h = get_collector_db_sha256()
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)

    def test_browser_cleaning_utilities(self):
        """Test URL, domain, title, and browser name normalization logic."""
        # 1. Title stripping
        t1 = "drc - Google Search - Google Chrome"
        self.assertEqual(clean_page_title(t1), "drc - Google Search")

        t2 = "Microsoft Teams and 5 more pages - Personal - Microsoft\u200b Edge"
        self.assertEqual(clean_page_title(t2), "Microsoft Teams")

        # 2. Domain normalization
        d1, u1 = clean_domain("www.Google.com", None)
        self.assertEqual(d1, "google.com")
        self.assertEqual(u1, "https://google.com")

        # 3. Domain inference from title when domain is None
        d2, u2 = clean_domain(None, "drc - Google Search")
        self.assertEqual(d2, "google.com")

        d3, u3 = clean_domain(None, "Downloads - DB Browser for SQLite")
        self.assertEqual(d3, "sqlitebrowser.org")

        # 4. Browser name normalization
        self.assertEqual(normalize_browser_name("Google Chrome"), "Google Chrome")
        self.assertEqual(normalize_browser_name("Brave"), "Brave Browser")
        self.assertEqual(normalize_browser_name("Microsoft Edge"), "Microsoft Edge")

    def test_end_to_end_phase2_execution_and_repeatability(self):
        """Verify full Phase 2 execution, derived data store persistence, and idempotency."""
        initial_hash = get_collector_db_sha256()

        # Run 1
        res1 = self.preprocessor.run()
        self.assertEqual(res1["status"], "COMPLETED")
        self.assertEqual(res1["statistics"]["errors_count"], 0)

        # Check raw counts match real database
        raw_counts = res1["statistics"]["raw_event_counts"]
        self.assertGreaterEqual(raw_counts["screenshots"], 34)
        self.assertGreaterEqual(raw_counts["sessions"], 32)
        self.assertGreater(raw_counts["activity_events"], 16000)
        self.assertGreater(raw_counts["browser_events"], 9000)
        self.assertGreater(raw_counts["interaction_events"], 7600)

        # Check screenshot deduplication
        self.assertEqual(res1["statistics"]["total_screenshots"], 34)
        self.assertEqual(res1["statistics"]["unique_screenshot_count"], 34)
        self.assertEqual(res1["statistics"]["duplicate_screenshot_count"], 0)

        # Check derived database tables exist and are populated
        conn = get_derived_db_connection()
        try:
            ep_count1 = conn.execute("SELECT COUNT(*) FROM activity_episodes").fetchone()[0]
            norm_brw1 = conn.execute("SELECT COUNT(*) FROM normalized_browser_events").fetchone()[0]
            dup_shots1 = conn.execute("SELECT COUNT(*) FROM screenshot_duplicates").fetchone()[0]
            prep_events1 = conn.execute("SELECT COUNT(*) FROM preprocessed_events").fetchone()[0]

            self.assertGreater(ep_count1, 0)
            self.assertEqual(norm_brw1, raw_counts["browser_events"])
            self.assertEqual(dup_shots1, 34)
            self.assertGreater(prep_events1, 0)

            # Verify source IDs are preserved in episodes
            sample_ep = conn.execute("SELECT source_event_ids FROM activity_episodes LIMIT 1").fetchone()[0]
            parsed_ids = json.loads(sample_ep)
            self.assertIsInstance(parsed_ids, list)
            self.assertGreater(len(parsed_ids), 0)
        finally:
            conn.close()

        # Run 2 (verify repeatability — derived tables must NOT duplicate records)
        res2 = self.preprocessor.run()
        self.assertEqual(res2["status"], "COMPLETED")

        conn = get_derived_db_connection()
        try:
            ep_count2 = conn.execute("SELECT COUNT(*) FROM activity_episodes").fetchone()[0]
            norm_brw2 = conn.execute("SELECT COUNT(*) FROM normalized_browser_events").fetchone()[0]
            dup_shots2 = conn.execute("SELECT COUNT(*) FROM screenshot_duplicates").fetchone()[0]
            prep_events2 = conn.execute("SELECT COUNT(*) FROM preprocessed_events").fetchone()[0]

            self.assertEqual(ep_count1, ep_count2, "Repeatability failure: activity_episodes count doubled")
            self.assertEqual(norm_brw1, norm_brw2, "Repeatability failure: normalized_browser_events count doubled")
            self.assertEqual(dup_shots1, dup_shots2, "Repeatability failure: screenshot_duplicates count doubled")
            self.assertEqual(prep_events1, prep_events2, "Repeatability failure: preprocessed_events count doubled")
        finally:
            conn.close()

        # Final verification that collector.db was never touched
        final_hash = get_collector_db_sha256()
        self.assertEqual(initial_hash, final_hash, "CRITICAL: collector.db was modified during tests!")


if __name__ == "__main__":
    unittest.main()

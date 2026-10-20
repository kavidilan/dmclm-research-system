"""
Unit and Integration Tests for Phase 13 — Resource-Aware Reasoning Engine
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)
Project: Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence
"""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.dmclm_config import ResourceAwareConfig, DEFAULT_CONFIG
from src.models import (
    ResourceSnapshot,
    ResourceProfileDecision,
    DeferredMemoryRecord,
    ResourceAwareExecutionResult,
    ResourceComparisonReport,
)
from src.dmclm_db import (
    init_dmclm_database,
    insert_resource_monitoring_log,
    insert_resource_execution_run,
    enqueue_deferred_memory,
    get_pending_deferred_memories,
    update_deferred_memory_status,
    get_latest_resource_runs,
)
from src.phase13_resource_aware import (
    SystemResourceMonitor,
    ResourceProfileClassifier,
    DeterministicQualityPrioritizer,
    SimilarityCacheManager,
    ResourceAwarePipelineExecutor,
    Phase13ResourceExperimentRunner,
    CONSOLIDATED_JSON_PATH,
    TRAINING_CSV_PATH,
    LIFECYCLE_MODEL_PATH,
    CORE_FIVE_FEATURES,
)


class TestPhase13ResourceAware(unittest.TestCase):
    """
    Test suite validating Phase 13 Resource-Aware Reasoning:
    - Real and mocked system resource monitoring
    - Profile classification (LOW, NORMAL, HIGH)
    - Quality-preserving candidate pair prioritization and budgeting
    - Similarity LRU caching and canonical keying
    - Deferred queue SQLite persistence and retrieval
    - Resource-aware relationship discovery on real Phase 8 data
    - Resource-aware ML inference with 100% frozen weight prediction identity
    - Comparative benchmark execution, figure generation, and quality preservation assertions
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="dmclm_phase13_test_"))
        cls.mock_db_path = cls.temp_dir / "test_resource_dmclm.db"
        init_dmclm_database(cls.mock_db_path)

        cls.config = ResourceAwareConfig(
            results_dir=str(cls.temp_dir / "resources"),
            figures_dir=str(cls.temp_dir / "figures"),
        )
        cls.monitor = SystemResourceMonitor()
        cls.classifier = ResourceProfileClassifier(cls.config)
        cls.prioritizer = DeterministicQualityPrioritizer(cls.config)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_resource_monitor_capture(self):
        """Test that SystemResourceMonitor captures genuine, non-zero technical measurements."""
        snapshot = self.monitor.capture_snapshot(stage="unit_test")
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertGreater(snapshot.cpu_count, 0)
        self.assertGreater(snapshot.memory_total_mb, 1000.0)
        self.assertGreater(snapshot.memory_available_mb, 100.0)
        self.assertGreater(snapshot.process_rss_mb, 10.0)  # Active Python process RSS
        self.assertGreaterEqual(snapshot.cpu_percent, 0.0)
        self.assertLessEqual(snapshot.cpu_percent, 100.0)
        self.assertEqual(snapshot.stage, "unit_test")

        snap_dict = snapshot.to_dict()
        self.assertIn("cpu_percent", snap_dict)
        self.assertIn("process_rss_mb", snap_dict)

    def test_02_resource_monitor_tracking_context(self):
        """Test track_resources context manager across execution block."""
        with self.monitor.track_resources(stage="work_simulation") as tracker:
            # Brief computational work to verify elapsed time and tracking
            _ = [x**2 for x in range(100000)]

        self.assertIn("initial", tracker)
        self.assertIn("final", tracker)
        self.assertIn("elapsed_seconds", tracker)
        self.assertGreater(tracker["elapsed_seconds"], 0.0)
        self.assertGreater(tracker["peak_rss_mb"], 0.0)

    def test_03_profile_classification_logic(self):
        """Test dynamic profile classification across controlled simulated resource states."""
        # Scenario A: High CPU usage -> LOW Profile
        snap_high_cpu = ResourceSnapshot(
            timestamp="2026-10-08T00:00:00Z",
            cpu_percent=85.0,  # >= 75.0
            cpu_count=8,
            memory_total_mb=16000.0,
            memory_available_mb=8000.0,
            memory_percent=50.0,
            process_rss_mb=120.0,
            process_vms_mb=200.0,
            process_cpu_percent=15.0,
        )
        dec_a = self.classifier.classify_snapshot(snap_high_cpu)
        self.assertEqual(dec_a.profile, "LOW")
        self.assertTrue(dec_a.allow_deferral)
        self.assertEqual(dec_a.batch_size, self.config.batch_size_low)

        # Scenario B: Low available RAM -> LOW Profile
        snap_low_ram = ResourceSnapshot(
            timestamp="2026-10-08T00:00:00Z",
            cpu_percent=20.0,
            cpu_count=8,
            memory_total_mb=16000.0,
            memory_available_mb=500.0,  # < 1024 MB
            memory_percent=96.0,
            process_rss_mb=120.0,
            process_vms_mb=200.0,
            process_cpu_percent=5.0,
        )
        dec_b = self.classifier.classify_snapshot(snap_low_ram)
        self.assertEqual(dec_b.profile, "LOW")

        # Scenario C: Low CPU, High available RAM -> HIGH Profile
        snap_abundant = ResourceSnapshot(
            timestamp="2026-10-08T00:00:00Z",
            cpu_percent=15.0,  # <= 30.0
            cpu_count=8,
            memory_total_mb=16000.0,
            memory_available_mb=10000.0,
            memory_percent=37.5,
            process_rss_mb=80.0,
            process_vms_mb=150.0,
            process_cpu_percent=2.0,
        )
        dec_c = self.classifier.classify_snapshot(snap_abundant)
        self.assertEqual(dec_c.profile, "HIGH")
        self.assertFalse(dec_c.allow_deferral)
        self.assertEqual(dec_c.candidate_pair_budget, self.config.relationship_pair_budget_high)

    def test_04_budget_allocation_per_profile(self):
        """Test budget parameters for all 3 profile states."""
        dummy_snap = self.monitor.capture_snapshot()
        for p in ["LOW", "NORMAL", "HIGH"]:
            dec = self.classifier.classify_snapshot(dummy_snap, override_profile=p)
            self.assertEqual(dec.profile, p)
            if p == "LOW":
                self.assertEqual(dec.batch_size, 50)
                self.assertEqual(dec.candidate_pair_budget, 1500)
                self.assertTrue(dec.allow_deferral)
            elif p == "NORMAL":
                self.assertEqual(dec.batch_size, 200)
                self.assertEqual(dec.candidate_pair_budget, 5000)
                self.assertFalse(dec.allow_deferral)
            else:
                self.assertEqual(dec.batch_size, 500)
                self.assertEqual(dec.candidate_pair_budget, 10000)
                self.assertFalse(dec.allow_deferral)

    def test_05_quality_prioritization_and_protection(self):
        """Test quality tier classification ensures high-value memories are never deferred."""
        # 1. Protected memory
        prot_mem = {"memory_id": "m1", "is_protected": True, "baseline_score": 0.20}
        self.assertEqual(self.prioritizer.classify_memory_tier(prot_mem), "CRITICAL")

        # 2. High-importance active memory
        high_mem = {"memory_id": "m2", "baseline_score": 0.85, "final_retention_strength": 0.75}
        self.assertEqual(self.prioritizer.classify_memory_tier(high_mem), "CRITICAL")

        # 3. Standard memory
        std_mem = {"memory_id": "m3", "baseline_score": 0.55, "final_retention_strength": 0.40}
        self.assertEqual(self.prioritizer.classify_memory_tier(std_mem), "STANDARD")

        # 4. Marginal memory
        marg_mem = {"memory_id": "m4", "baseline_score": 0.25, "final_retention_strength": 0.15}
        self.assertEqual(self.prioritizer.classify_memory_tier(marg_mem), "MARGINAL")

    def test_06_candidate_pair_prioritization_and_budgeting(self):
        """Test candidate pair budgeting and determinism under constrained limits."""
        memories = [
            {"memory_id": f"m_{i}", "primary_application": "code" if i % 2 == 0 else "browser",
             "start_time": f"2026-09-20T10:{i:02d}:00Z", "baseline_score": 0.3 + (i % 5) * 0.15}
            for i in range(10)
        ]
        # Total possible pairs for 10 memories = 10 * 9 / 2 = 45 pairs
        # Case A: Budget >= 45 -> Returns all 45 pairs
        sel_a, tot_a, def_a = self.prioritizer.prioritize_and_budget_pairs(memories, budget=50)
        self.assertEqual(len(sel_a), 45)
        self.assertEqual(def_a, 0)

        # Case B: Budget = 15 -> Returns top 15 pairs
        sel_b1, tot_b1, def_b1 = self.prioritizer.prioritize_and_budget_pairs(memories, budget=15)
        self.assertEqual(len(sel_b1), 15)
        self.assertEqual(def_b1, 30)

        # Determinism check: repeating with identical input produces identical ordered pairs
        sel_b2, _, _ = self.prioritizer.prioritize_and_budget_pairs(memories, budget=15)
        self.assertEqual(sel_b1, sel_b2)

    def test_07_similarity_cache_correctness_and_lru(self):
        """Test LRU similarity cache hit/miss tracking and key symmetry."""
        cache = SimilarityCacheManager(max_size=3)
        self.assertEqual(cache.hit_rate, 0.0)

        cache.put("mem_A", "mem_B", {"sim": 0.85})
        # Symmetric lookup: get(B, A) matches get(A, B)
        val = cache.get("mem_B", "mem_A")
        self.assertIsNotNone(val)
        self.assertEqual(val["sim"], 0.85)
        self.assertEqual(cache.hits, 1)

        # Cache miss
        miss = cache.get("mem_A", "mem_C")
        self.assertIsNone(miss)
        self.assertEqual(cache.misses, 1)

        # LRU Eviction: insert 3 more items to evict mem_A:mem_B
        cache.put("mem_1", "mem_2", {"sim": 0.1})
        cache.put("mem_3", "mem_4", {"sim": 0.2})
        cache.put("mem_5", "mem_6", {"sim": 0.3})
        self.assertIsNone(cache.get("mem_A", "mem_B"))  # Evicted

    def test_08_deferred_queue_operations(self):
        """Test enqueueing, retrieving, and updating deferred memories in SQLite."""
        test_rec = {
            "queue_id": "q_test_01",
            "memory_id": "mem_deferred_01",
            "priority_score": 0.32,
            "deferred_reason": "low_resource_throttle",
            "processing_stage": "relationship_discovery",
            "status": "PENDING",
            "payload": {"features": [0.1, 0.2]},
        }
        enqueue_deferred_memory(test_rec, db_path=self.mock_db_path)

        pending = get_pending_deferred_memories(limit=10, db_path=self.mock_db_path)
        self.assertGreaterEqual(len(pending), 1)
        target = next((r for r in pending if r["queue_id"] == "q_test_01"), None)
        self.assertIsNotNone(target)
        self.assertEqual(target["memory_id"], "mem_deferred_01")

        # Update status
        update_deferred_memory_status("q_test_01", "PROCESSED", db_path=self.mock_db_path)
        pending_after = get_pending_deferred_memories(limit=10, db_path=self.mock_db_path)
        self.assertIsNone(next((r for r in pending_after if r["queue_id"] == "q_test_01"), None))

    def test_09_resource_aware_relationship_discovery(self):
        """Integration test: discover relationships under NORMAL and LOW profiles on real Phase 8 data."""
        if not CONSOLIDATED_JSON_PATH.exists():
            self.skipTest("consolidated_memories.json not present")

        with open(CONSOLIDATED_JSON_PATH, "r", encoding="utf-8") as f:
            memories = json.load(f)

        executor = ResourceAwarePipelineExecutor(config=self.config, db_path=self.mock_db_path)

        # NORMAL profile: all 2,701 pairs evaluated
        norm_rels, norm_stats, norm_res = executor.discover_relationships_resource_aware(
            memories, profile_override="NORMAL"
        )
        self.assertEqual(norm_res.status, "SUCCESS")
        self.assertEqual(norm_res.profile_used, "NORMAL")
        self.assertIn(len(norm_rels), [303, 331])  # Matches Phase 8 relationship discovery

        # LOW profile: budget constrained to 1,500 pairs
        low_rels, low_stats, low_res = executor.discover_relationships_resource_aware(
            memories, profile_override="LOW"
        )
        self.assertEqual(low_res.status, "SUCCESS")
        self.assertEqual(low_res.profile_used, "LOW")
        self.assertEqual(low_res.processed_count, 1500)
        self.assertIn(low_res.deferred_count, [1201, 1503])
        self.assertGreater(len(low_rels), 200)

    def test_10_resource_aware_ml_inference(self):
        """Verify resource-aware model inference preserves 100% prediction identity with Phase 10 model."""
        if not (TRAINING_CSV_PATH.exists() and LIFECYCLE_MODEL_PATH.exists()):
            self.skipTest("Training CSV or model artifact missing")

        df = pd.read_csv(TRAINING_CSV_PATH)
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

        executor = ResourceAwarePipelineExecutor(config=self.config, db_path=self.mock_db_path)

        # 1. Standard prediction
        import joblib
        bundle = joblib.load(LIFECYCLE_MODEL_PATH)
        clf = bundle["classifier"] if isinstance(bundle, dict) else bundle
        std_clf_preds = clf.predict(test_df[list(CORE_FIVE_FEATURES)].values)

        # 2. Resource-aware batch prediction (LOW profile, batch_size=50)
        res_clf_preds, _, exec_res = executor.predict_lifecycle_resource_aware(
            test_df, profile_override="LOW"
        )

        self.assertEqual(exec_res.status, "SUCCESS")
        self.assertEqual(len(res_clf_preds), len(std_clf_preds))

        # Identity check: 100% agreement between standard and resource-aware inference
        agreement_rate = float(np.mean(res_clf_preds == std_clf_preds))
        self.assertEqual(agreement_rate, 1.0, "Resource-aware inference diverged from baseline model!")

    def test_11_comparative_experiment_runner_and_artifacts(self):
        """Test Phase 13 comparative experiment runner generates report JSON, Markdown, and 3 figures."""
        if not CONSOLIDATED_JSON_PATH.exists():
            self.skipTest("consolidated_memories.json not present")

        runner = Phase13ResourceExperimentRunner(config=self.config, db_path=self.mock_db_path)
        report = runner.run_comparative_experiment()

        self.assertIsInstance(report, ResourceComparisonReport)
        self.assertGreater(report.quality_retention_pct, 80.0)
        self.assertEqual(len(report.figures_generated), 3)

        # Check generated figure files exist
        for fig_rel in report.figures_generated:
            fig_path = Path(fig_rel)
            if not fig_path.is_absolute():
                from src.phase13_resource_aware import ROOT_DIR
                fig_path = ROOT_DIR / fig_path
            self.assertTrue(fig_path.exists(), f"Figure missing: {fig_path}")
            self.assertGreater(fig_path.stat().st_size, 5000, f"Figure too small: {fig_path}")

        # Check JSON and Markdown reports exist
        rep_json = runner.output_dir / "resource_evaluation_report.json"
        rep_md = runner.output_dir / "phase13_resource_aware_report.md"
        self.assertTrue(rep_json.exists())
        self.assertTrue(rep_md.exists())

    def test_12_quality_preservation_assertion(self):
        """Test quality preservation: high-strength edge recall under LOW profile >= 80%."""
        if not CONSOLIDATED_JSON_PATH.exists():
            self.skipTest("consolidated_memories.json not present")

        runner = Phase13ResourceExperimentRunner(config=self.config, db_path=self.mock_db_path)
        report = runner.run_comparative_experiment()
        self.assertGreaterEqual(report.quality_retention_pct, 85.0)


if __name__ == "__main__":
    unittest.main()

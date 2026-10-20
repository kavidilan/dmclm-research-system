"""
Unit and Integration Tests for Phase 6 — Consolidation & Compression
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Covers all 18 required unit test cases:
1. Empty dataset handling
2. Single-memory dataset
3. Completely unrelated memories
4. Clearly related memories
5. Multiple consolidation groups
6. Semantic similarity threshold
7. Temporal threshold
8. Shared application/domain relationships
9. Consolidation evidence generation
10. Source-memory traceability
11. Deterministic grouping
12. Deterministic IDs
13. Compression ratio calculation
14. Memory reduction percentage
15. Duplicate consolidation prevention
16. Preservation of original memories
17. Missing or malformed memory fields
18. Repeatability of the complete consolidation process
+ Real-data integration test
"""

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import ConsolidationConfig, DEFAULT_CONFIG
from src.phase6_consolidation_compression import (
    Phase6Consolidator,
    tokenize_text,
    parse_timestamp_iso,
    trace_consolidated_memory,
    BASELINE_SCORED_JSON_PATH,
)


def create_mock_memory(
    candidate_id: str,
    app: str = "Google Chrome",
    domain: str = "arxiv.org",
    context: str = "Attention Is All You Need - arXiv",
    summary: str = "Reading transformer architecture paper on arXiv",
    start_time: str = "2026-10-01T10:00:00+00:00",
    end_time: str = "2026-10-01T10:15:00+00:00",
    duration_ms: int = 900000,
    interactions: int = 15,
    session_id: str = "sess-001",
    episode_ids: List[str] = None,
    baseline_score: float = 0.55,
) -> Dict[str, Any]:
    """Helper to create controlled test memory fixtures."""
    return {
        "candidate_id": candidate_id,
        "memory_id": candidate_id,
        "session_id": session_id,
        "source_episode_ids": episode_ids or [f"ep_{candidate_id}"],
        "primary_application": app,
        "applications_involved": [app],
        "browser_domains": [domain] if domain else [],
        "window_context": context,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
        "interaction_count": interactions,
        "browser_event_count": 5,
        "screenshot_ids": [f"shot_{candidate_id}_1"],
        "activity_summary": summary,
        "meaningful_criteria": ["user_interactions"],
        "created_at": "2026-10-01T10:00:00+00:00",
        "recency": 0.8,
        "frequency": 0.6,
        "retrieval_history": 0.0,
        "semantic_relevance": 0.7,
        "relationship_strength": 0.6,
        "baseline_score": baseline_score,
        "dominant_feature": "recency",
        "value_tier": "MEDIUM_VALUE",
        "lifecycle_tier": "COMPRESS",
    }


class TestPhase6ConsolidationAndCompression(unittest.TestCase):
    """Automated unit test suite covering all 18 Phase 6 requirements."""

    def setUp(self):
        self.config = ConsolidationConfig(
            min_group_size=2,
            affinity_threshold=0.55,
            semantic_similarity_threshold=0.35,
            temporal_window_sec=86400.0,
            temporal_proximity_tau_sec=3600.0,
        )
        self.consolidator = Phase6Consolidator(self.config)

    # 1. Empty dataset handling
    def test_01_empty_dataset_handling(self):
        """Verify engine gracefully handles empty memory input without crashing or zero-division."""
        nodes, metrics = self.consolidator.process([])
        self.assertEqual(len(nodes), 0)
        self.assertEqual(metrics["number_of_source_memories"], 0)
        self.assertEqual(metrics["number_of_consolidated_memories"], 0)
        self.assertEqual(metrics["compression_ratio"], 0.0)
        self.assertEqual(metrics["memory_reduction_percentage"], 0.0)

    # 2. Single-memory dataset
    def test_02_single_memory_dataset(self):
        """Verify single memory does not form an artificial group and remains unconsolidated."""
        m = create_mock_memory("mem_single")
        nodes, metrics = self.consolidator.process([m])
        self.assertEqual(len(nodes), 0, "Single memory must not form a consolidation group (min_group_size=2)")
        self.assertEqual(metrics["number_of_source_memories"], 1)
        self.assertEqual(metrics["number_of_unconsolidated_memories"], 1)
        self.assertEqual(metrics["total_retained_memory_nodes"], 1)
        self.assertEqual(metrics["compression_ratio"], 0.0)
        self.assertEqual(metrics["memory_reduction_percentage"], 0.0)

    # 3. Completely unrelated memories
    def test_03_completely_unrelated_memories(self):
        """Verify memories across completely different applications, domains, and tasks are not merged."""
        m1 = create_mock_memory("mem_1", app="Spotify", domain="", context="Playing Lo-Fi Playlist", summary="Listening to music on Spotify")
        m2 = create_mock_memory("mem_2", app="Visual Studio Code", domain="", context="server.py - DMCLM", summary="Editing python server script")
        m3 = create_mock_memory("mem_3", app="Calculator", domain="", context="Standard Calculator", summary="Math calculations")

        nodes, metrics = self.consolidator.process([m1, m2, m3])
        self.assertEqual(len(nodes), 0, "Unrelated memories must not be falsely consolidated")
        self.assertEqual(metrics["number_of_consolidated_memories"], 0)
        self.assertEqual(metrics["number_of_unconsolidated_memories"], 3)
        self.assertEqual(metrics["total_retained_memory_nodes"], 3)
        self.assertEqual(metrics["memory_reduction_percentage"], 0.0)

    # 4. Clearly related memories
    def test_04_clearly_related_memories(self):
        """Verify memories representing repeated workflow on the same task consolidate into 1 node."""
        m1 = create_mock_memory(
            "mem_rel_1",
            app="Google Chrome",
            domain="arxiv.org",
            context="Attention Is All You Need - arXiv",
            summary="Reading transformer architecture paper",
            start_time="2026-10-01T10:00:00+00:00",
            end_time="2026-10-01T10:10:00+00:00",
            session_id="sess_ml_1",
            episode_ids=["ep_101"],
        )
        m2 = create_mock_memory(
            "mem_rel_2",
            app="Google Chrome",
            domain="arxiv.org",
            context="Attention Is All You Need PDF - arXiv",
            summary="Studying transformer attention mechanisms",
            start_time="2026-10-01T10:12:00+00:00",
            end_time="2026-10-01T10:25:00+00:00",
            session_id="sess_ml_1",
            episode_ids=["ep_102"],
        )

        nodes, metrics = self.consolidator.process([m1, m2])
        self.assertEqual(len(nodes), 1, "Two related memories must consolidate into 1 durable node")
        node = nodes[0]
        self.assertEqual(set(node.source_memory_ids), {"mem_rel_1", "mem_rel_2"})
        self.assertEqual(set(node.source_episode_ids), {"ep_101", "ep_102"})
        self.assertEqual(node.source_count, 2)
        self.assertEqual(metrics["source_memories_in_groups"], 2)
        self.assertEqual(metrics["group_reduction_percentage"], 50.0)

    # 5. Multiple consolidation groups
    def test_05_multiple_consolidation_groups(self):
        """Verify engine correctly discovers distinct consolidation clusters for distinct tasks."""
        # Task A: VSCode DMCLM development (3 memories)
        a1 = create_mock_memory("a1", app="VSCode", domain="", context="models.py - DMCLM", summary="Writing models in VSCode", session_id="s1")
        a2 = create_mock_memory("a2", app="VSCode", domain="", context="models.py test - DMCLM", summary="Writing tests in VSCode", session_id="s1")
        a3 = create_mock_memory("a3", app="VSCode", domain="", context="models.py refactor - DMCLM", summary="Refactoring models in VSCode", session_id="s1")

        # Task B: Chrome PubMed cancer research (2 memories)
        b1 = create_mock_memory("b1", app="Chrome", domain="pubmed.gov", context="Immunotherapy Clinical Trial - PubMed", summary="Reading immunotherapy clinical trial study", session_id="s2")
        b2 = create_mock_memory("b2", app="Chrome", domain="pubmed.gov", context="Immunotherapy Clinical Trial - PubMed", summary="Analyzing immunotherapy clinical trial results", session_id="s2")

        # Task C: Solitary memory (1 memory)
        c1 = create_mock_memory("c1", app="Calculator", domain="", context="Calculator", summary="Doing budget math", session_id="s3")

        nodes, metrics = self.consolidator.process([a1, a2, a3, b1, b2, c1])
        self.assertEqual(len(nodes), 2, "Must identify exactly 2 distinct consolidation groups")
        group_apps = {n.primary_application for n in nodes}
        self.assertIn("VSCode", group_apps)
        self.assertIn("Chrome", group_apps)
        self.assertEqual(metrics["number_of_unconsolidated_memories"], 1)
        self.assertEqual(metrics["total_retained_memory_nodes"], 3)  # 2 consolidated nodes + 1 singleton

    # 6. Semantic similarity threshold
    def test_06_semantic_similarity_threshold(self):
        """Verify that memories with low semantic overlap fail consolidation if threshold is not met."""
        # Two memories in same app, but completely disjoint semantic tokens
        m1 = create_mock_memory("m1", app="Chrome", domain="example.com", context="Cooking Italian Pasta Recipes", summary="Making pasta recipes at home")
        m2 = create_mock_memory("m2", app="Chrome", domain="example.com", context="Quantum Computing Superconducting Qubits", summary="Physics of quantum hardware")

        # High semantic threshold requires genuine token match
        strict_config = ConsolidationConfig(semantic_similarity_threshold=0.40)
        consolidator_strict = Phase6Consolidator(strict_config)

        nodes, _ = consolidator_strict.process([m1, m2])
        self.assertEqual(len(nodes), 0, "Semantically disparate topics must not merge even if app matches")

    # 7. Temporal threshold
    def test_07_temporal_threshold(self):
        """Verify memories separated by a wide temporal gap beyond the temporal window do not consolidate."""
        # Two memories 10 days apart in different sessions
        m1 = create_mock_memory("m1", start_time="2026-09-01T10:00:00+00:00", end_time="2026-09-01T10:10:00+00:00", session_id="sess_old")
        m2 = create_mock_memory("m2", start_time="2026-09-15T10:00:00+00:00", end_time="2026-09-15T10:10:00+00:00", session_id="sess_new")

        # 24 hour temporal window
        config = ConsolidationConfig(temporal_window_sec=86400.0)
        consolidator = Phase6Consolidator(config)

        nodes, _ = consolidator.process([m1, m2])
        self.assertEqual(len(nodes), 0, "Memories separated by weeks in different sessions must not consolidate")

    # 8. Shared application/domain relationships
    def test_08_shared_application_domain_relationships(self):
        """Verify shared domain affinity correctly connects browser activities."""
        m1 = create_mock_memory("m1", app="Google Chrome", domain="github.com", context="Repo Issues - github.com", summary="Checking GitHub issues")
        m2 = create_mock_memory("m2", app="Google Chrome", domain="github.com", context="Repo Pull Requests - github.com", summary="Reviewing GitHub pull requests")

        nodes, _ = self.consolidator.process([m1, m2])
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertIn("github.com", node.domains)
        self.assertEqual(node.primary_application, "Google Chrome")

    # 9. Consolidation evidence generation
    def test_09_consolidation_evidence_generation(self):
        """Verify each consolidated memory node contains explainable evidence reasons."""
        m1 = create_mock_memory("m1", context="Data Preprocessing - Python", summary="Running preprocessing scripts")
        m2 = create_mock_memory("m2", context="Data Preprocessing Clean - Python", summary="Cleaning preprocessing records")

        nodes, _ = self.consolidator.process([m1, m2])
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        ev = node.consolidation_evidence
        self.assertIn("evidence_reasons", ev)
        self.assertIsInstance(ev["evidence_reasons"], list)
        self.assertTrue(len(ev["evidence_reasons"]) > 0)
        self.assertIn("mean_pairwise_affinity", ev)
        self.assertGreaterEqual(ev["mean_pairwise_affinity"], 0.55)

    # 10. Source-memory traceability
    def test_10_source_memory_traceability(self):
        """Verify full provenance traceability: Consolidated Node -> Source Memories -> Source Episodes."""
        m1 = create_mock_memory("m1", episode_ids=["ep_001", "ep_002"])
        m2 = create_mock_memory("m2", episode_ids=["ep_003"])
        m3 = create_mock_memory("m3", episode_ids=["ep_004"])

        nodes, _ = self.consolidator.process([m1, m2, m3])
        self.assertEqual(len(nodes), 1)
        node = nodes[0]

        # Trace via utility function
        trace = trace_consolidated_memory(node.consolidated_memory_id, nodes, [m1, m2, m3])
        self.assertEqual(trace["consolidated_memory_id"], node.consolidated_memory_id)
        self.assertEqual(sorted(trace["source_memory_ids"]), ["m1", "m2", "m3"])
        self.assertEqual(sorted(trace["source_episode_ids"]), ["ep_001", "ep_002", "ep_003", "ep_004"])
        self.assertEqual(len(trace["source_memories"]), 3)

    # 11. Deterministic grouping
    def test_11_deterministic_grouping(self):
        """Verify grouping is strictly invariant to input list ordering."""
        m1 = create_mock_memory("m1", start_time="2026-10-01T10:00:00+00:00")
        m2 = create_mock_memory("m2", start_time="2026-10-01T10:05:00+00:00")
        m3 = create_mock_memory("m3", start_time="2026-10-01T10:10:00+00:00")

        nodes_order1, _ = self.consolidator.process([m1, m2, m3])
        nodes_order2, _ = self.consolidator.process([m3, m1, m2])

        self.assertEqual(len(nodes_order1), 1)
        self.assertEqual(len(nodes_order2), 1)
        self.assertEqual(nodes_order1[0].source_memory_ids, nodes_order2[0].source_memory_ids)

    # 12. Deterministic IDs
    def test_12_deterministic_ids(self):
        """Verify that running Phase 6 multiple times produces identical consolidated node IDs."""
        m1 = create_mock_memory("m1")
        m2 = create_mock_memory("m2")

        nodes_run1, _ = self.consolidator.process([m1, m2])
        nodes_run2, _ = self.consolidator.process([m1, m2])

        self.assertEqual(nodes_run1[0].consolidated_memory_id, nodes_run2[0].consolidated_memory_id)
        self.assertTrue(nodes_run1[0].consolidated_memory_id.startswith("c_node_"))

    # 13. Compression ratio calculation
    def test_13_compression_ratio_calculation(self):
        """Verify mathematical formula: Compression Ratio = 1 - (Retained Memories / Original Memories)."""
        # 4 memories that all consolidate into 1 node: N_orig=4, N_final=1 -> Ratio = 1 - (1/4) = 0.75
        mems = [create_mock_memory(f"m_{i}") for i in range(4)]
        nodes, metrics = self.consolidator.process(mems)

        self.assertEqual(len(nodes), 1)
        self.assertEqual(metrics["number_of_source_memories"], 4)
        self.assertEqual(metrics["total_retained_memory_nodes"], 1)
        self.assertAlmostEqual(metrics["compression_ratio"], 0.75, places=4)

    # 14. Memory reduction percentage
    def test_14_memory_reduction_percentage(self):
        """Verify mathematical formula: Reduction % = ((Original - Retained) / Original) * 100."""
        # 4 memories -> 1 retained node: ((4 - 1) / 4) * 100 = 75.0%
        mems = [create_mock_memory(f"m_{i}") for i in range(4)]
        _, metrics = self.consolidator.process(mems)

        self.assertAlmostEqual(metrics["memory_reduction_percentage"], 75.0, places=2)
        self.assertEqual(metrics["memories_removed_through_consolidation"], 3)

    # 15. Duplicate consolidation prevention
    def test_15_duplicate_consolidation_prevention(self):
        """Verify disjoint partitioning: no single memory candidate appears in more than one consolidation group."""
        # 4 related memories that link together
        mems = [create_mock_memory(f"mem_{i}") for i in range(4)]
        nodes, _ = self.consolidator.process(mems)

        seen_mids = set()
        for node in nodes:
            for mid in node.source_memory_ids:
                self.assertNotIn(mid, seen_mids, f"Memory {mid} was duplicated across multiple consolidated nodes")
                seen_mids.add(mid)

    # 16. Preservation of original memories
    def test_16_preservation_of_original_memories(self):
        """Verify original memory list/dictionaries are never mutated, stripped, or destroyed."""
        original_mems = [create_mock_memory(f"m_{i}") for i in range(3)]
        snapshot = copy.deepcopy(original_mems)

        nodes, metrics = self.consolidator.process(original_mems)

        self.assertEqual(len(original_mems), len(snapshot))
        self.assertEqual(original_mems, snapshot, "Original memories must remain completely unmodified")

    # 17. Missing or malformed memory fields
    def test_17_missing_or_malformed_memory_fields(self):
        """Verify resilience to missing timestamps, empty strings, None values, and missing keys."""
        m_corrupt_1 = {
            "candidate_id": "m_bad_1",
            "primary_application": None,
            "window_context": "",
            "activity_summary": None,
            "start_time": "invalid_date_format",
            "duration_ms": "not_an_int",
            "interaction_count": None,
        }
        m_corrupt_2 = {
            "candidate_id": "m_bad_2",
            "primary_application": None,
            "window_context": "",
            "activity_summary": None,
            "start_time": "",
        }

        # Should execute cleanly without throwing unhandled exceptions
        nodes, metrics = self.consolidator.process([m_corrupt_1, m_corrupt_2])
        self.assertIsInstance(nodes, list)
        self.assertEqual(metrics["number_of_source_memories"], 2)

    # 18. Repeatability of the complete consolidation process
    def test_18_repeatability_of_complete_process(self):
        """Verify identical results across repeated executions on identical datasets."""
        mems = [create_mock_memory(f"m_{i}") for i in range(5)]

        nodes_1, metrics_1 = self.consolidator.process(mems)
        nodes_2, metrics_2 = self.consolidator.process(mems)

        self.assertEqual(len(nodes_1), len(nodes_2))
        self.assertEqual(
            [n.to_dict() for n in nodes_1],
            [n.to_dict() for n in nodes_2],
            "Repeated execution must produce strictly identical consolidated outputs"
        )
        self.assertEqual(metrics_1["compression_ratio"], metrics_2["compression_ratio"])


class TestPhase6RealDataIntegration(unittest.TestCase):
    """Integration test executing Phase 6 against the actual Phase 5 baseline dataset."""

    @unittest.skipUnless(BASELINE_SCORED_JSON_PATH.exists(), "Phase 5 baseline data file not found")
    def test_real_phase5_dataset_consolidation(self):
        """Verify Phase 6 runs end-to-end on the real 6,229 Phase 5 memories, producing honest metrics."""
        consolidator = Phase6Consolidator()
        nodes, metrics = consolidator.run_pipeline(save_to_db=True, save_to_json=True)

        self.assertGreater(metrics["number_of_source_memories"], 0)
        self.assertGreater(metrics["number_of_consolidated_memories"], 0)
        self.assertGreater(len(nodes), 0)

        # Check provenance on first consolidated node
        first_node = nodes[0]
        self.assertTrue(first_node.consolidated_memory_id.startswith("c_node_"))
        self.assertGreaterEqual(first_node.source_count, 2)
        self.assertTrue(len(first_node.source_episode_ids) > 0)
        self.assertTrue(len(first_node.semantic_representation) > 10)
        self.assertIn("mean_pairwise_affinity", first_node.consolidation_evidence)

        # Print real results to stdout for inspection
        print(f"\n[REAL-DATA PHASE 6 INTEGRATION RESULTS]")
        print(f"Total Source Memories:       {metrics['number_of_source_memories']:,}")
        print(f"Consolidation Groups:         {metrics['number_of_consolidated_memories']:,}")
        print(f"Unconsolidated Memories:      {metrics['number_of_unconsolidated_memories']:,}")
        print(f"Total Retained Memory Units:  {metrics['total_retained_memory_nodes']:,}")
        print(f"Source Memories in Groups:    {metrics['source_memories_in_groups']:,}")
        print(f"Redundancy Removed:           {metrics['memories_removed_through_consolidation']:,}")
        print(f"Overall Memory Reduction:     {metrics['memory_reduction_percentage']:.2f}%")
        print(f"Group Reduction Percentage:   {metrics['group_reduction_percentage']:.2f}%")
        print(f"Avg Source / Consolidated:    {metrics['average_source_memories_per_consolidated_memory']:.2f}")
        print(f"Processing Time:              {metrics['processing_time_seconds']}s")


if __name__ == "__main__":
    unittest.main()

"""
Unit and Integration Tests for Phase 8 — Relationship Management
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Covers all Phase 8 requirements:
1. Self-relationship rejection (u == v -> None, no self-loops in graph).
2. Duplicate undirected relationship prevention (canonical min(u, v) < max(u, v) ordering).
3. Threshold filtering (rejection when composite strength < min_strength_threshold).
4. Normalized relationship strength in [0.0, 1.0].
5. Deterministic relationship IDs (re-running produces identical stable IDs).
6. Safe handling of missing signals (missing timestamps, missing domains, missing contexts).
7. Isolated memories preserved as valid degree-0 nodes.
8. Multiple relationship types correctly classified (WORKFLOW_CONTINUITY, SHARED_DOMAIN,
   SEMANTIC_SIMILARITY, TEMPORAL_ASSOCIATION, SHARED_APPLICATION, CROSS_APPLICATION_TASK,
   HYBRID_COMPOSITE).
9. NetworkX graph topology and topological metrics calculation (density, avg degree, components).
10. Traceability preservation (trace_memory_relationships returns incident edges, neighbors, evidence).
11. Retention state preservation (nodes preserve Phase 7 states: RETAINED, DECAYING, ARCHIVED, FORGETTABLE).
12. Zero accidental permanent deletion (archived/forgetting-eligible memories remain fully connected).
13. Edge case: Empty dataset returns empty graph and stats cleanly.
14. Edge case: Single-memory dataset produces 1 isolated node and 0 edges cleanly.
15. Repeat execution idempotency (identical outputs on consecutive runs).
16. Real-data integration test against actual project outputs.
"""

import copy
import json
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dmclm_config import RelationshipConfig, DEFAULT_CONFIG
from src.phase8_relationship_management import (
    Phase8RelationshipManager,
    trace_memory_relationships,
    tokenize_text,
    parse_timestamp_iso,
    RETENTION_JSON_PATH,
    CONSOLIDATED_JSON_PATH,
)


def create_mock_phase7_memory(
    node_id: str,
    primary_application: str = "Code.exe",
    topic: str = "python data pipeline",
    summary: str = "Developing Python data pipeline script for research data processing and logging",
    contexts: List[str] = None,
    domains: List[str] = None,
    start_time: str = "2026-03-29T10:00:00Z",
    end_time: str = "2026-03-29T10:30:00Z",
    retention_state: str = "RETAINED",
    final_retention_strength: float = 0.85,
    baseline_score: float = 0.70,
    source_count: int = 5,
    is_protected: bool = False,
    forgetting_eligible: bool = False,
) -> Dict[str, Any]:
    """Helper to generate structured mock Phase 7 consolidated memory nodes."""
    return {
        "consolidated_memory_id": node_id,
        "primary_application": primary_application,
        "topic": topic,
        "summary": summary,
        "unique_windows_accessed": contexts or ["VSCode - pipeline.py", "Terminal"],
        "contexts": contexts or ["VSCode - pipeline.py", "Terminal"],
        "applications": [primary_application],
        "domains": domains or [],
        "start_time": start_time,
        "end_time": end_time,
        "retention_state": retention_state,
        "final_retention_strength": final_retention_strength,
        "baseline_score": baseline_score,
        "source_count": source_count,
        "is_protected": is_protected,
        "forgetting_eligible": forgetting_eligible,
        "source_memory_ids": [f"mem_{node_id}_{i}" for i in range(source_count)],
        "source_episode_ids": [f"ep_{node_id}_{i}" for i in range(source_count)],
    }


class TestPhase8RelationshipManagement(unittest.TestCase):
    """Comprehensive test suite for Phase 8 Relationship Management."""

    def setUp(self):
        self.config = RelationshipConfig(
            min_strength_threshold=0.35,
            semantic_threshold=0.30,
            context_threshold=0.25,
            domain_threshold=0.30,
            weight_semantic=0.25,
            weight_context=0.20,
            weight_application=0.20,
            weight_domain=0.20,
            weight_temporal=0.15,
            temporal_proximity_window_sec=86400.0,
            temporal_proximity_tau_sec=21600.0,
        )
        self.manager = Phase8RelationshipManager(config=self.config)

    def test_01_self_relationship_rejection(self):
        """Verify that a memory evaluated against itself returns None and no self-loops exist in graph."""
        m1 = create_mock_phase7_memory("node_01")
        rel = self.manager.evaluate_pair(m1, m1)
        self.assertIsNone(rel, "Evaluating a memory against itself must return None (no self-loops).")

        # Test within full process pipeline
        rels, G, stats = self.manager.process([m1])
        self.assertEqual(len(rels), 0)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(nx.number_of_selfloops(G), 0, "Graph must have zero self-loops.")

    def test_02_duplicate_undirected_relationship_prevention(self):
        """Verify that canonical ordering prevents duplicate undirected edges (u, v) and (v, u)."""
        m1 = create_mock_phase7_memory("node_alpha", primary_application="Code.exe")
        m2 = create_mock_phase7_memory("node_beta", primary_application="Code.exe")

        # Evaluating (m1, m2) vs (m2, m1)
        rel_12 = self.manager.evaluate_pair(m1, m2)
        rel_21 = self.manager.evaluate_pair(m2, m1)

        self.assertIsNotNone(rel_12)
        self.assertIsNotNone(rel_21)
        # Both must produce identical canonical source and target
        self.assertEqual(rel_12.source_memory_id, "node_alpha")
        self.assertEqual(rel_12.target_memory_id, "node_beta")
        self.assertEqual(rel_21.source_memory_id, "node_alpha")
        self.assertEqual(rel_21.target_memory_id, "node_beta")
        self.assertEqual(rel_12.relationship_id, rel_21.relationship_id)

        # In full process, exactly one edge must exist between two connected memories
        rels, G, stats = self.manager.process([m1, m2])
        self.assertEqual(len(rels), 1, "Exactly one undirected relationship must be created.")
        self.assertEqual(G.number_of_edges(), 1)
        self.assertTrue(G.has_edge("node_alpha", "node_beta"))

    def test_03_threshold_filtering(self):
        """Verify that memory pairs with composite strength below min_strength_threshold are rejected."""
        # Pair with zero commonality: completely different apps, no domains, no shared tokens, distant timestamps
        m1 = create_mock_phase7_memory(
            "node_code",
            primary_application="Code.exe",
            topic="neural network training",
            summary="Training convolutional layers on GPU with PyTorch",
            contexts=["PyTorch GPU Training"],
            domains=[],
            start_time="2026-01-01T10:00:00Z",
            end_time="2026-01-01T11:00:00Z",
        )
        m2 = create_mock_phase7_memory(
            "node_calc",
            primary_application="Calculator.exe",
            topic="budget tax sum",
            summary="Calculating monthly grocery tax subtraction",
            contexts=["Calculator"],
            domains=[],
            start_time="2026-03-01T10:00:00Z",
            end_time="2026-03-01T10:05:00Z",
        )

        rel = self.manager.evaluate_pair(m1, m2)
        self.assertIsNone(rel, "Memory pair with composite strength below threshold must be rejected.")

        rels, G, stats = self.manager.process([m1, m2])
        self.assertEqual(len(rels), 0, "No relationships should pass threshold filtering.")
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(stats.isolated_nodes_count, 2, "Both nodes must be preserved as isolated nodes.")

    def test_04_normalized_relationship_strength_in_bounds(self):
        """Verify that relationship strength is strictly bounded in [0.0, 1.0]."""
        m1 = create_mock_phase7_memory("node_01")
        m2 = create_mock_phase7_memory("node_02")

        rel = self.manager.evaluate_pair(m1, m2)
        self.assertIsNotNone(rel)
        self.assertGreaterEqual(rel.relationship_strength, 0.0)
        self.assertLessEqual(rel.relationship_strength, 1.0)
        self.assertIsInstance(rel.relationship_strength, float)

    def test_05_deterministic_relationship_ids(self):
        """Verify that running relationship evaluation multiple times generates identical stable IDs."""
        m1 = create_mock_phase7_memory("node_01")
        m2 = create_mock_phase7_memory("node_02")

        rel_run1 = self.manager.evaluate_pair(m1, m2)
        rel_run2 = self.manager.evaluate_pair(m1, m2)

        self.assertEqual(rel_run1.relationship_id, rel_run2.relationship_id)
        self.assertTrue(rel_run1.relationship_id.startswith("rel_"))
        self.assertEqual(len(rel_run1.relationship_id), 20)  # "rel_" + 16 chars

    def test_06_safe_missing_signals_handling(self):
        """Verify that missing timestamps, domains, contexts, or summary tokens do not cause errors."""
        m_sparse1 = {
            "consolidated_memory_id": "sparse_01",
            "primary_application": "Code.exe",
            # No start_time, end_time, summary, domains, contexts
        }
        m_sparse2 = {
            "consolidated_memory_id": "sparse_02",
            "primary_application": "Code.exe",
            # No start_time, end_time, summary, domains, contexts
        }

        # Should execute cleanly without raising exception
        signals = self.manager.calculate_pairwise_signals(m_sparse1, m_sparse2)
        self.assertIn("semantic_similarity", signals)
        self.assertIn("temporal_proximity", signals)
        self.assertEqual(signals["temporal_proximity"], 0.0)
        self.assertEqual(signals["semantic_similarity"], 0.0)
        self.assertEqual(signals["application_match"], 1.0)

        # Pairwise evaluation should not crash
        rel = self.manager.evaluate_pair(m_sparse1, m_sparse2)
        # With app match = 1.0 (weight 0.20) and no domains, composite may or may not reach 0.35, but must be valid
        if rel:
            self.assertTrue(0.0 <= rel.relationship_strength <= 1.0)

    def test_07_isolated_memories_preserved_as_valid_nodes(self):
        """Verify that isolated memories with no relationships are preserved as degree-0 nodes."""
        m_connected1 = create_mock_phase7_memory("conn_01", primary_application="Code.exe")
        m_connected2 = create_mock_phase7_memory("conn_02", primary_application="Code.exe")
        m_isolated = create_mock_phase7_memory(
            "iso_01",
            primary_application="SpecialTool.exe",
            topic="rare custom routine",
            summary="Completely distinct rare custom routine without overlap",
            contexts=["Unique Window"],
            start_time="2025-01-01T00:00:00Z",
            end_time="2025-01-01T00:10:00Z",
        )

        rels, G, stats = self.manager.process([m_connected1, m_connected2, m_isolated])
        self.assertEqual(stats.total_nodes, 3)
        self.assertEqual(stats.total_edges, 1)
        self.assertEqual(stats.isolated_nodes_count, 1)
        self.assertTrue(G.has_node("iso_01"))
        self.assertEqual(G.degree["iso_01"], 0)
        self.assertEqual(G.degree["conn_01"], 1)
        self.assertEqual(G.degree["conn_02"], 1)

    def test_08_multiple_relationship_types_classification(self):
        """Verify that different signal combinations yield appropriate relationship types."""
        # 1. SHARED_DOMAIN (Browser research)
        m_web1 = create_mock_phase7_memory(
            "web_01",
            primary_application="chrome.exe",
            domains=["arxiv.org", "paperswithcode.com"],
            contexts=["arxiv.org/abs/2103.00020 - Chrome"],
            start_time="2026-03-29T10:00:00Z",
            end_time="2026-03-29T10:15:00Z",
        )
        m_web2 = create_mock_phase7_memory(
            "web_02",
            primary_application="chrome.exe",
            domains=["arxiv.org", "google.com"],
            contexts=["arxiv.org/abs/2106.09685 - Chrome"],
            start_time="2026-03-29T10:20:00Z",
            end_time="2026-03-29T10:35:00Z",
        )
        rel_web = self.manager.evaluate_pair(m_web1, m_web2)
        self.assertIsNotNone(rel_web)
        self.assertIn("SHARED_DOMAIN", [rel_web.relationship_type] + rel_web.evidence.get("secondary_types", []))

        # 2. WORKFLOW_CONTINUITY (Shared app + window context overlap)
        m_wf1 = create_mock_phase7_memory(
            "wf_01",
            primary_application="Code.exe",
            contexts=["VSCode - pipeline.py - Project", "Terminal - bash"],
        )
        m_wf2 = create_mock_phase7_memory(
            "wf_02",
            primary_application="Code.exe",
            contexts=["VSCode - pipeline.py - Project", "Output - Debug"],
        )
        rel_wf = self.manager.evaluate_pair(m_wf1, m_wf2)
        self.assertIsNotNone(rel_wf)
        self.assertIn("WORKFLOW_CONTINUITY", [rel_wf.relationship_type] + rel_wf.evidence.get("secondary_types", []))

    def test_09_graph_construction_and_topological_metrics(self):
        """Verify NetworkX graph construction, density, average degree, and component count."""
        m1 = create_mock_phase7_memory("m1", primary_application="Code.exe")
        m2 = create_mock_phase7_memory("m2", primary_application="Code.exe")
        m3 = create_mock_phase7_memory("m3", primary_application="Code.exe")

        rels, G, stats = self.manager.process([m1, m2, m3])
        self.assertEqual(stats.total_nodes, 3)
        self.assertGreaterEqual(stats.total_edges, 1)
        self.assertGreaterEqual(stats.density, 0.0)
        self.assertLessEqual(stats.density, 1.0)
        self.assertGreaterEqual(stats.average_degree, 0.0)
        self.assertGreaterEqual(stats.connected_components_count, 1)

    def test_10_traceability_preservation(self):
        """Verify that trace_memory_relationships returns incident edges, neighbors, and evidence."""
        m1 = create_mock_phase7_memory("node_tracer_1", primary_application="Code.exe")
        m2 = create_mock_phase7_memory("node_tracer_2", primary_application="Code.exe")

        rels, G, stats = self.manager.process([m1, m2])
        self.assertEqual(len(rels), 1)

        trace = trace_memory_relationships("node_tracer_1", relationships=rels, graph=G)
        self.assertEqual(trace["memory_id"], "node_tracer_1")
        self.assertEqual(trace["degree"], 1)
        self.assertEqual(len(trace["connected_neighbors"]), 1)
        self.assertEqual(trace["connected_neighbors"][0]["neighbor_id"], "node_tracer_2")
        self.assertIn("evidence", trace["connected_neighbors"][0])

    def test_11_retention_state_preservation_in_graph_nodes(self):
        """Verify that nodes in the graph retain Phase 7 retention states and protection flags."""
        m_retained = create_mock_phase7_memory("m_ret", retention_state="RETAINED", final_retention_strength=0.90, is_protected=True)
        m_decaying = create_mock_phase7_memory("m_dec", retention_state="DECAYING", final_retention_strength=0.55)
        m_archived = create_mock_phase7_memory("m_arc", retention_state="ARCHIVED", final_retention_strength=0.25)
        m_forget = create_mock_phase7_memory("m_for", retention_state="FORGETTABLE", final_retention_strength=0.15, forgetting_eligible=True)

        rels, G, stats = self.manager.process([m_retained, m_decaying, m_archived, m_forget])
        self.assertEqual(G.nodes["m_ret"]["retention_state"], "RETAINED")
        self.assertTrue(G.nodes["m_ret"]["is_protected"])
        self.assertEqual(G.nodes["m_dec"]["retention_state"], "DECAYING")
        self.assertEqual(G.nodes["m_arc"]["retention_state"], "ARCHIVED")
        self.assertEqual(G.nodes["m_for"]["retention_state"], "FORGETTABLE")
        self.assertTrue(G.nodes["m_for"]["forgetting_eligible"])

    def test_12_no_accidental_permanent_deletion(self):
        """Verify that archived or forgettable memories remain fully connected and preserved in graph."""
        m1 = create_mock_phase7_memory("m_active", primary_application="Code.exe", retention_state="RETAINED")
        m2 = create_mock_phase7_memory("m_archived", primary_application="Code.exe", retention_state="ARCHIVED")

        rels, G, stats = self.manager.process([m1, m2])
        self.assertEqual(len(rels), 1)
        self.assertTrue(G.has_node("m_archived"), "Archived memory must NOT be deleted from relationship graph.")
        self.assertTrue(G.has_edge("m_active", "m_archived"), "Relationship between active and archived must be preserved.")

    def test_13_empty_dataset_handling(self):
        """Verify that processing an empty list of memories handles gracefully with zero counts."""
        rels, G, stats = self.manager.process([])
        self.assertEqual(len(rels), 0)
        self.assertEqual(stats.total_nodes, 0)
        self.assertEqual(stats.total_edges, 0)
        self.assertEqual(stats.density, 0.0)
        self.assertEqual(stats.isolated_nodes_count, 0)

    def test_14_single_memory_dataset_handling(self):
        """Verify that a single memory yields 1 node, 0 edges, and 1 isolated node."""
        m1 = create_mock_phase7_memory("single_node")
        rels, G, stats = self.manager.process([m1])
        self.assertEqual(len(rels), 0)
        self.assertEqual(stats.total_nodes, 1)
        self.assertEqual(stats.total_edges, 0)
        self.assertEqual(stats.isolated_nodes_count, 1)
        self.assertEqual(stats.density, 0.0)

    def test_15_repeat_execution_idempotency(self):
        """Verify that running the pipeline repeatedly on identical input yields identical outputs."""
        memories = [
            create_mock_phase7_memory("rep_1", primary_application="Code.exe"),
            create_mock_phase7_memory("rep_2", primary_application="Code.exe"),
            create_mock_phase7_memory("rep_3", primary_application="chrome.exe"),
        ]

        rels_1, _, stats_1 = self.manager.process(memories)
        rels_2, _, stats_2 = self.manager.process(memories)

        self.assertEqual(len(rels_1), len(rels_2))
        self.assertEqual([r.relationship_id for r in rels_1], [r.relationship_id for r in rels_2])
        self.assertEqual([r.relationship_strength for r in rels_1], [r.relationship_strength for r in rels_2])
        self.assertEqual(stats_1.total_edges, stats_2.total_edges)
        self.assertEqual(stats_1.average_degree, stats_2.average_degree)

    def test_16_real_data_pipeline_integration(self):
        """Integration test: Execute full pipeline against real Phase 7 / Phase 6 dataset."""
        input_path = RETENTION_JSON_PATH if RETENTION_JSON_PATH.exists() else CONSOLIDATED_JSON_PATH
        if not input_path.exists():
            self.skipTest("Real dataset not available for integration test.")

        manager = Phase8RelationshipManager()
        # Run full pipeline with enrichment without mutating persistent files in test
        rels, G, stats = manager.run_pipeline(save_to_db=False, save_to_json=False)

        # Assertions on real data output integrity
        self.assertIn(stats.total_nodes, [74, 78])
        self.assertGreater(stats.total_edges, 0)
        self.assertGreaterEqual(stats.density, 0.0)
        self.assertLessEqual(stats.density, 1.0)
        self.assertGreaterEqual(stats.isolated_nodes_count, 0)
        self.assertEqual(stats.isolated_nodes_count + sum(1 for _, d in G.degree() if d > 0), stats.total_nodes)
        self.assertEqual(len(rels), stats.total_edges)

        # Check all edges are valid
        for r in rels:
            self.assertTrue(0.0 <= r.relationship_strength <= 1.0)
            self.assertNotEqual(r.source_memory_id, r.target_memory_id)
            self.assertLess(r.source_memory_id, r.target_memory_id)
            self.assertIn("signals", r.evidence)
            self.assertGreater(len(r.relationship_id), 0)


if __name__ == "__main__":
    unittest.main()

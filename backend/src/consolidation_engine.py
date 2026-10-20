"""
Consolidation Engine
Performs semantic clustering of short-term episodic items,
synthesizes abstract concepts, and migrates consolidated representations
into Long-Term Memory (LTM).
"""

from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import ShortTermMemoryItem, ConsolidatedMemoryNode


class ConsolidationEngine:
    def __init__(self, similarity_threshold: float = 0.38):
        """
        similarity_threshold: Cosine similarity cutoff to consider two episodic memories
                              part of the same research topic/theme.
        """
        self.similarity_threshold = similarity_threshold
        self.long_term_memory: Dict[str, ConsolidatedMemoryNode] = {}
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

    def cluster_episodic_memories(
        self,
        candidate_items: List[ShortTermMemoryItem]
    ) -> List[List[ShortTermMemoryItem]]:
        """
        Groups episodic short-term memories into semantic clusters based on text similarity.
        """
        if not candidate_items:
            return []
        if len(candidate_items) == 1:
            return [candidate_items]

        corpus = [f"{item.title} {item.content} {' '.join(item.tags)}" for item in candidate_items]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception:
            # Fallback if vocabulary is too sparse
            return [[item] for item in candidate_items]

        n = len(candidate_items)
        visited = [False] * n
        clusters: List[List[ShortTermMemoryItem]] = []

        for i in range(n):
            if visited[i]:
                continue
            current_cluster = [candidate_items[i]]
            visited[i] = True

            for j in range(i + 1, n):
                if not visited[j] and sim_matrix[i, j] >= self.similarity_threshold:
                    current_cluster.append(candidate_items[j])
                    visited[j] = True

            clusters.append(current_cluster)

        return clusters

    def synthesize_cluster(self, cluster: List[ShortTermMemoryItem]) -> ConsolidatedMemoryNode:
        """
        Consolidates a cluster of episodic memories into a single high-level Long-Term Memory Node.
        Extracts salient points, resolves timestamps, and determines stability.
        """
        # Determine primary concept title from most reinforced item
        cluster_sorted = sorted(cluster, key=lambda x: (x.interaction_weight, x.base_importance), reverse=True)
        primary_title = cluster_sorted[0].title or "Research Topic"

        # Deduplicate and consolidate key points / sentences
        extracted_points = []
        all_tags = set()
        contributing_ids = []
        timestamps = []

        for item in cluster:
            contributing_ids.append(item.id)
            timestamps.append(item.timestamp)
            all_tags.update(item.tags)
            
            # Sentence extraction
            sentences = [s.strip() for s in item.content.split('.') if len(s.strip()) > 15]
            for s in sentences:
                if s not in extracted_points:
                    extracted_points.append(s)

        first_obs = min(timestamps)
        last_obs = max(timestamps)
        
        # Stability score calculation (Synaptic reinforcement based on cluster size and interaction weight)
        aggregate_weight = sum(item.interaction_weight for item in cluster)
        stability = min(1.0, 0.4 + (0.15 * len(cluster)) + (0.05 * aggregate_weight))

        # Generate abstractive concept summary
        summary = f"Consolidated concept from {len(cluster)} research interactions between {first_obs.strftime('%Y-%m-%d %H:%M')} and {last_obs.strftime('%H:%M')}."

        ltm_node = ConsolidatedMemoryNode(
            primary_concept=primary_title,
            summary=summary,
            key_points=extracted_points[:5],  # Top salient insights
            contributing_stm_ids=contributing_ids,
            first_observed=first_obs,
            last_reinforced=last_obs,
            reinforcement_count=len(cluster),
            stability_score=round(stability, 3),
            semantic_tags=sorted(list(all_tags))
        )
        return ltm_node

    def run_consolidation_cycle(
        self,
        memory_buffer: List[ShortTermMemoryItem]
    ) -> Tuple[List[ConsolidatedMemoryNode], Dict[str, float]]:
        """
        Executes a consolidation cycle over the short-term memory buffer:
        1. Filters active candidates (not pruned, not yet consolidated)
        2. Clusters them semantically
        3. Synthesizes LTM nodes
        4. Marks STM items as consolidated
        5. Returns created LTM nodes and compression metrics
        """
        active_candidates = [m for m in memory_buffer if not m.is_pruned and not m.is_consolidated]
        if not active_candidates:
            return [], {"raw_count": 0, "consolidated_nodes": 0, "compression_ratio_pct": 0.0}

        clusters = self.cluster_episodic_memories(active_candidates)
        new_ltm_nodes = []

        for cluster in clusters:
            node = self.synthesize_cluster(cluster)
            self.long_term_memory[node.node_id] = node
            new_ltm_nodes.append(node)
            
            # Mark STM items as consolidated
            for item in cluster:
                item.is_consolidated = True

        raw_count = len(active_candidates)
        node_count = len(new_ltm_nodes)
        compression = round(((raw_count - node_count) / raw_count) * 100, 2) if raw_count > 0 else 0.0

        stats = {
            "raw_count": raw_count,
            "consolidated_nodes": node_count,
            "compression_ratio_pct": compression
        }
        return new_ltm_nodes, stats

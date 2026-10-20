"""
Desktop Research Collector Adapter
Ingests raw streams and session exports from the Desktop Research Collector App,
transforming them into standardized memory items for the Consolidation Pipeline.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.models import RawCollectorItem, ShortTermMemoryItem
from src.decay_engine import CognitiveDecayEngine


class CollectorAdapter:
    def __init__(self, decay_engine: CognitiveDecayEngine):
        self.decay_engine = decay_engine

    def load_from_json(self, file_path: str) -> List[RawCollectorItem]:
        """Loads collector items from a JSON export produced by the desktop app."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Collector file not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = []
        for entry in data:
            item = RawCollectorItem(
                id=entry.get("id", ""),
                timestamp=datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat())),
                source_app=entry.get("source_app", "DesktopCollector"),
                title=entry.get("title", "Untitled Research Note"),
                content=entry.get("content", ""),
                tags=entry.get("tags", []),
                interaction_type=entry.get("interaction_type", "read"),
                metadata=entry.get("metadata", {})
            )
            items.append(item)
        return items

    def convert_to_short_term_memory(
        self,
        raw_items: List[RawCollectorItem]
    ) -> List[ShortTermMemoryItem]:
        """Converts raw collector records into Short-Term Memory Items with initial importance scores."""
        stm_items = []
        for raw in raw_items:
            base_imp = self.decay_engine.compute_initial_importance(raw)
            multiplier = self.decay_engine.interaction_multipliers.get(raw.interaction_type, 1.0)
            
            stm = ShortTermMemoryItem(
                id=raw.id or f"stm_{len(stm_items)+1}",
                timestamp=raw.timestamp,
                title=raw.title,
                content=raw.content,
                source_app=raw.source_app,
                tags=raw.tags,
                access_count=1,
                interaction_weight=multiplier,
                base_importance=base_imp,
                current_retention_score=base_imp
            )
            stm_items.append(stm)
        return stm_items

    @staticmethod
    def generate_synthetic_collector_data(output_path: str = "data/sample_collector_data.json") -> str:
        """
        Generates realistic sample session data simulating the user's Desktop Research Collector App.
        Simulates research sessions covering Memory Systems, Privacy Sanitization, and Retrieval Engines.
        """
        now = datetime.now()
        samples = [
            # Topic A: Transformer Architecture & Attention (3 items, closely related)
            {
                "id": "rec_001",
                "timestamp": (now - timedelta(hours=6)).isoformat(),
                "source_app": "ArxivReader",
                "title": "Attention Is All You Need Paper Analysis",
                "content": "The Transformer uses self-attention mechanisms to compute representations of input sequences without using sequence-aligned RNNs or convolution.",
                "tags": ["deep-learning", "attention", "transformer"],
                "interaction_type": "highlight"
            },
            {
                "id": "rec_002",
                "timestamp": (now - timedelta(hours=5, minutes=45)).isoformat(),
                "source_app": "ResearchNotes",
                "title": "Multi-Head Attention Notes",
                "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Scaled dot-product attention formula.",
                "tags": ["transformer", "attention", "neural-networks"],
                "interaction_type": "note"
            },
            {
                "id": "rec_003",
                "timestamp": (now - timedelta(hours=5, minutes=10)).isoformat(),
                "source_app": "BrowserPdf",
                "title": "Transformer Positional Encodings",
                "content": "Since models contain no recurrence, positional encodings are injected at the bottoms of the encoder and decoder stacks to inject sequence order information.",
                "tags": ["transformer", "positional-encoding"],
                "interaction_type": "read"
            },

            # Topic B: Cognitive Memory & Consolidation (3 items, user's individual focus)
            {
                "id": "rec_004",
                "timestamp": (now - timedelta(hours=3, minutes=30)).isoformat(),
                "source_app": "ResearchNotes",
                "title": "Memory Consolidation in AI Agents",
                "content": "Memory consolidation transforms volatile short-term episodic experiences into stable semantic knowledge representations. Biological hippocampus-to-cortex replay model.",
                "tags": ["memory-consolidation", "agentic-ai", "long-term-memory"],
                "interaction_type": "note"
            },
            {
                "id": "rec_005",
                "timestamp": (now - timedelta(hours=2, minutes=15)).isoformat(),
                "source_app": "ArxivReader",
                "title": "Generative Agents Memory Architecture",
                "content": "Generative agents implement a memory stream with reflection and consolidation. Reflection synthesizes low-level observations into higher-level abstract insights.",
                "tags": ["memory-consolidation", "reflection", "agentic-ai"],
                "interaction_type": "highlight"
            },
            {
                "id": "rec_006",
                "timestamp": (now - timedelta(hours=1, minutes=40)).isoformat(),
                "source_app": "BrowserPdf",
                "title": "Synaptic Consolidation Mechanics",
                "content": "Ebbinghaus decay curve governs decay rate. Frequent retrieval reinforces synaptic stability and prevents premature forgetting in long-term memory stores.",
                "tags": ["forgetting-curve", "memory-consolidation", "ebbinghaus"],
                "interaction_type": "note"
            },

            # Topic C: Ephemeral / Noisy web browses (Low importance, should be pruned or decayed)
            {
                "id": "rec_007",
                "timestamp": (now - timedelta(hours=12)).isoformat(),
                "source_app": "WebBrowser",
                "title": "Google Search - Weather Forecast Colombo",
                "content": "Weather forecast for Colombo today sunny with light showers in the evening.",
                "tags": ["weather"],
                "interaction_type": "passive_view"
            },
            {
                "id": "rec_008",
                "timestamp": (now - timedelta(hours=10)).isoformat(),
                "source_app": "WebBrowser",
                "title": "StackOverflow - Git commit error",
                "content": "How to resolve fatal: refusing to merge unrelated histories in git repository.",
                "tags": ["git", "devtools"],
                "interaction_type": "passive_view"
            }
        ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        return output_path

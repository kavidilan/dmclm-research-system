"""
Decay and Forgetting Curve Engine
Implements biologically-inspired memory decay, cognitive importance calculation,
and retention threshold gating before consolidation.
"""

import math
from datetime import datetime
from typing import List
from src.models import ShortTermMemoryItem, RawCollectorItem


class CognitiveDecayEngine:
    def __init__(
        self,
        decay_constant: float = 0.05,        # Lambda: rate at which episodic trace fades per hour
        pruning_threshold: float = 0.25,     # Memory items below this score are candidates for deletion/decay
        consolidation_threshold: float = 0.65 # Memory items above this score are prioritized for consolidation
    ):
        self.decay_constant = decay_constant
        self.pruning_threshold = pruning_threshold
        self.consolidation_threshold = consolidation_threshold
        
        # Interaction type multipliers reflecting cognitive effort in collector app
        self.interaction_multipliers = {
            "passive_view": 0.6,
            "read": 1.0,
            "search": 1.1,
            "bookmark": 1.4,
            "highlight": 1.6,
            "note": 2.0
        }

    def compute_initial_importance(self, raw_item: RawCollectorItem) -> float:
        """
        Calculates baseline importance (0.0 to 1.0) based on:
        1. Interaction depth (passive view vs written note)
        2. Content density (information length / tags)
        """
        multiplier = self.interaction_multipliers.get(raw_item.interaction_type, 1.0)
        
        # Word count heuristic (diminishing returns log scale)
        words = len(raw_item.content.split())
        length_factor = min(1.0, math.log10(max(10, words)) / 3.0)  # ~0.33 for 10 words, 1.0 for 1000 words
        
        # Tag heuristic
        tag_factor = min(1.0, len(raw_item.tags) * 0.15)
        
        base_score = (0.5 * length_factor + 0.3 * tag_factor + 0.2) * multiplier
        return round(min(1.0, max(0.1, base_score)), 4)

    def calculate_retention(self, item: ShortTermMemoryItem, current_time: datetime) -> float:
        """
        Ebbinghaus-inspired retention formula with spaced repetition reinforcement:
        R(t) = Base * e^(-lambda * dt) + ReinforcementBonus
        """
        delta_hours = max(0.0, (current_time - item.timestamp).total_seconds() / 3600.0)
        
        # Exponential memory decay
        decay_factor = math.exp(-self.decay_constant * delta_hours)
        
        # Spaced repetition / access reinforcement (logarithmic scaling)
        reinforcement = 0.12 * math.log(1 + item.access_count) * item.interaction_weight
        
        retention = (item.base_importance * decay_factor) + reinforcement
        return round(min(1.0, max(0.0, retention)), 4)

    def update_buffer_retention(self, memory_buffer: List[ShortTermMemoryItem], current_time: datetime):
        """Updates retention scores for all short-term memories and flags candidates."""
        for item in memory_buffer:
            if not item.is_pruned:
                item.current_retention_score = self.calculate_retention(item, current_time)
                if item.current_retention_score < self.pruning_threshold and not item.is_consolidated:
                    item.is_pruned = True

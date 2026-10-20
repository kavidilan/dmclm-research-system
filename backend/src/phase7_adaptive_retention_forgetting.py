"""
Phase 7 â€” Adaptive Retention / Forgetting Engine
Member 4 â€” Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles & Architecture:
1. MATHEMATICALLY EXPLAINABLE TEMPORAL DECAY:
   Evaluates elapsed time between memory observation and reference time.
   Applies an adaptive exponential decay curve where high-value, resilient memories decay slower
   and low-value memories decay faster:
   lambda_eff = lambda_base * (1.0 - alpha * resilience)
   T_decay = exp(-lambda_eff * age_days)
2. ADAPTIVE COGNITIVE EVIDENCE & RETENTION STRENGTH:
   Synthesizes Phase 4 five features, Phase 5 baseline scores, and consolidation information into
   cognitive resilience factor (Omega) and retention strength (R):
   Retention Strength = (w_val * V_0 + (1 - w_val) * Omega) * T_decay
3. HIGH-VALUE MEMORY PROTECTION:
   Protects memories with high initial baseline scores, frequent activity, or large consolidation mass
   from accidental forgetting through a guaranteed protection floor (R >= 0.40).
4. MULTI-SIGNAL FORGETTING ELIGIBILITY (NO BLIND DELETION):
   Memories are classified as FORGETTABLE only when multiple signals converge:
   retention strength < 0.20, age >= 7 days, low frequency, low retrieval history,
   low semantic relevance, and low relationship strength.
   Original memories are NEVER permanently deleted; they remain preserved for research evaluation.
5. DISCRETE RETENTION STATES:
   - RETAINED: Active, strongly retained knowledge (R >= 0.60)
   - DECAYING: Declining retention, active monitoring (0.35 <= R < 0.60)
   - ARCHIVED: Preserved secondary tier with low current activity (R < 0.35, not forgettable)
   - FORGETTABLE: Qualified for logical forgetting under multi-signal criteria
6. COMPLETE TRACEABILITY & REVERSIBILITY:
   Full provenance maintained: Retention Decision -> Consolidated Memory -> Source Memories -> Activity Episodes.
"""

import json
import logging
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import AdaptiveRetentionConfig, DMCLMGlobalConfig, DEFAULT_CONFIG
    from src.models import Phase6ConsolidatedMemory, Phase7RetainedMemory
    from src.dmclm_db import (
        init_dmclm_database,
        get_dmclm_connection,
        insert_or_replace_retention_memories,
        query_retention_memories,
    )
except ImportError:
    from dmclm_config import AdaptiveRetentionConfig, DMCLMGlobalConfig, DEFAULT_CONFIG
    from models import Phase6ConsolidatedMemory, Phase7RetainedMemory
    from dmclm_db import (
        init_dmclm_database,
        get_dmclm_connection,
        insert_or_replace_retention_memories,
        query_retention_memories,
    )

DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
CONSOLIDATED_JSON_PATH = DATA_DIR / "consolidated_memories.json"
RETENTION_JSON_PATH = DATA_DIR / "retention_evaluated_memories.json"
RETENTION_METRICS_PATH = RESULTS_DIR / "retention_metrics.json"
RETENTION_REPORT_PATH = RESULTS_DIR / "retention_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase7.AdaptiveRetention")


def parse_timestamp_iso(ts_str: Optional[str]) -> float:
    """Parse ISO timestamp string to epoch seconds float."""
    if not ts_str:
        return 0.0
    try:
        s = str(ts_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        try:
            s = str(ts_str).split("+")[0].split("Z")[0]
            dt = datetime.strptime(s[:26], "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return 0.0


class Phase7AdaptiveRetentionEngine:
    """
    Core implementation of Phase 7 â€” Adaptive Retention / Forgetting.
    Evaluates temporal decay, adaptive cognitive resilience, protection rules,
    and multi-signal forgetting eligibility for consolidated memory representations.
    """

    def __init__(self, config: Optional[AdaptiveRetentionConfig] = None):
        self.config = config or DEFAULT_CONFIG.retention
        self.config.validate()

    def resolve_reference_timestamp(
        self,
        memories: List[Dict[str, Any]],
        reference_time: Optional[Union[str, datetime]] = None
    ) -> float:
        """
        Determines the reference/evaluation timestamp deterministically.
        Supports:
        1. Explicit argument `reference_time` (string or datetime).
        2. Config `reference_mode == 'max_observed'`: picks max timestamp across memories (reproducible).
        3. Config `reference_mode == 'current_time'`: current UTC time.
        4. Config `explicit_reference_timestamp`: static configured timestamp.
        """
        if reference_time is not None:
            if isinstance(reference_time, datetime):
                return reference_time.timestamp()
            elif isinstance(reference_time, str):
                parsed = parse_timestamp_iso(reference_time)
                if parsed > 0.0:
                    return parsed

        if self.config.explicit_reference_timestamp:
            parsed = parse_timestamp_iso(self.config.explicit_reference_timestamp)
            if parsed > 0.0:
                return parsed

        if self.config.reference_mode == "max_observed" and memories:
            timestamps = []
            for m in memories:
                ts = parse_timestamp_iso(
                    m.get("end_time") or m.get("last_reinforced") or m.get("start_time") or m.get("created_at")
                )
                if ts > 0.0:
                    timestamps.append(ts)
            if timestamps:
                return max(timestamps)

        # Fallback to current UTC timestamp
        return datetime.now(timezone.utc).timestamp()

    def extract_feature_values(self, memory: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, int]:
        """
        Extracts baseline and 5D feature values from consolidated memory, supporting nested schemas safely.
        Returns:
            (baseline_score, recency, frequency, retrieval_history, semantic_relevance, relationship_strength, source_count)
        """
        binfo = memory.get("baseline_information") or {}
        fmeans = binfo.get("feature_means") or {}

        # 1. Baseline Score
        baseline_score = float(
            binfo.get("mean_baseline_score")
            if binfo.get("mean_baseline_score") is not None
            else (memory.get("baseline_score") if memory.get("baseline_score") is not None else 0.5)
        )

        # 2. Five features
        recency = float(
            fmeans.get("recency")
            if fmeans.get("recency") is not None
            else (memory.get("recency") if memory.get("recency") is not None else 0.5)
        )
        frequency = float(
            fmeans.get("frequency")
            if fmeans.get("frequency") is not None
            else (memory.get("frequency") if memory.get("frequency") is not None else 0.5)
        )
        retrieval_history = float(
            fmeans.get("retrieval_history")
            if fmeans.get("retrieval_history") is not None
            else (memory.get("retrieval_history") if memory.get("retrieval_history") is not None else 0.0)
        )
        semantic_relevance = float(
            fmeans.get("semantic_relevance")
            if fmeans.get("semantic_relevance") is not None
            else (memory.get("semantic_relevance") if memory.get("semantic_relevance") is not None else 0.5)
        )
        relationship_strength = float(
            fmeans.get("relationship_strength")
            if fmeans.get("relationship_strength") is not None
            else (memory.get("relationship_strength") if memory.get("relationship_strength") is not None else 0.5)
        )

        # 3. Source count
        source_count = int(
            memory.get("source_count")
            if memory.get("source_count") is not None
            else len(memory.get("source_memory_ids") or [])
        )

        # Clamping to valid bounds [0.0, 1.0]
        baseline_score = max(0.0, min(1.0, baseline_score))
        recency = max(0.0, min(1.0, recency))
        frequency = max(0.0, min(1.0, frequency))
        retrieval_history = max(0.0, min(1.0, retrieval_history))
        semantic_relevance = max(0.0, min(1.0, semantic_relevance))
        relationship_strength = max(0.0, min(1.0, relationship_strength))
        source_count = max(1, source_count)

        return (
            baseline_score,
            recency,
            frequency,
            retrieval_history,
            semantic_relevance,
            relationship_strength,
            source_count,
        )

    def calculate_adaptive_resilience(
        self,
        baseline_score: float,
        frequency: float,
        retrieval_history: float,
        semantic_relevance: float,
        relationship_strength: float
    ) -> float:
        """
        Computes composite cognitive resilience Omega in [0.0, 1.0].
        Weights high baseline value, repeated frequency, and active retrieval utility.
        """
        c = self.config
        resilience = (
            c.resilience_weight_baseline * baseline_score
            + c.resilience_weight_frequency * frequency
            + c.resilience_weight_retrieval * retrieval_history
            + c.resilience_weight_semantic * semantic_relevance
            + c.resilience_weight_relationship * relationship_strength
        )
        return max(0.0, min(1.0, float(resilience)))

    def evaluate_protection(
        self,
        baseline_score: float,
        frequency: float,
        retrieval_history: float,
        semantic_relevance: float,
        relationship_strength: float,
        source_count: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates the high-value memory protection mechanism.
        Returns:
            (is_protected, protection_reason)
        """
        if not self.config.enable_protection:
            return False, None

        c = self.config
        reasons: List[str] = []

        if baseline_score >= c.protection_baseline_threshold:
            reasons.append(f"High baseline importance ({baseline_score:.3f} >= {c.protection_baseline_threshold:.2f})")
        if frequency >= c.protection_frequency_threshold:
            reasons.append(f"High task repetition frequency ({frequency:.3f} >= {c.protection_frequency_threshold:.2f})")
        if retrieval_history >= c.protection_retrieval_threshold:
            reasons.append(f"Proven retrieval utility ({retrieval_history:.3f} >= {c.protection_retrieval_threshold:.2f})")
        if source_count >= c.protection_source_count_threshold:
            reasons.append(f"Extensive consolidation mass ({source_count} source memories >= {c.protection_source_count_threshold})")
        if semantic_relevance >= 0.70 and relationship_strength >= 0.70:
            reasons.append(f"Strong semantic and relational cohesion (S={semantic_relevance:.2f}, Rel={relationship_strength:.2f})")

        if reasons:
            return True, "; ".join(reasons)
        return False, None

    def evaluate_forgetting_eligibility(
        self,
        retention_strength: float,
        age_days: float,
        is_protected: bool,
        frequency: float,
        retrieval_history: float,
        semantic_relevance: float,
        relationship_strength: float
    ) -> Tuple[bool, List[str]]:
        """
        Evaluates multi-signal forgetting criteria.
        Requires convergence of:
        - retention strength below forget threshold
        - memory not protected
        - sufficient elapsed age (>= 7 days)
        - low frequency, retrieval, semantic relevance, and relationship strength.
        """
        reasons: List[str] = []
        if is_protected:
            return False, ["Memory is protected by high-value policy."]

        c = self.config
        if retention_strength >= c.forget_threshold:
            return False, [f"Retention strength ({retention_strength:.3f}) exceeds forgetting threshold ({c.forget_threshold:.2f})."]

        # Check age requirement
        if age_days < c.min_age_days_for_forgetting:
            return False, [f"Memory age ({age_days:.1f} days) is less than minimum retention window ({c.min_age_days_for_forgetting:.1f} days)."]

        # Strong relationships are a mandatory safeguard against forgetting.
        if relationship_strength > c.forgetting_relationship_max:
            return False, [
                f"Relationship strength ({relationship_strength:.2f}) exceeds "
                f"the forgetting limit ({c.forgetting_relationship_max:.2f})."
            ]

        # Check the remaining three low-utility signals.
        signals_met = []
        if frequency <= c.forgetting_freq_max:
            signals_met.append(f"low frequency ({frequency:.2f} <= {c.forgetting_freq_max:.2f})")
        if retrieval_history <= c.forgetting_retrieval_max:
            signals_met.append(f"low retrieval utility ({retrieval_history:.2f} <= {c.forgetting_retrieval_max:.2f})")
        if semantic_relevance <= c.forgetting_semantic_max:
            signals_met.append(f"low semantic relevance ({semantic_relevance:.2f} <= {c.forgetting_semantic_max:.2f})")

        # Relationship strength must be low, and all three remaining signals must be low.
        if len(signals_met) == 3:
            reasons = [
                f"Retention strength ({retention_strength:.3f} < {c.forget_threshold:.2f})",
                f"Elapsed age ({age_days:.1f}d >= {c.min_age_days_for_forgetting:.1f}d)",
            ] + signals_met
            return True, reasons
        else:
            return False, [f"Insufficient multi-signal convergence (only {len(signals_met)}/4 low-activity signals met)."]

    def construct_retention_explanation(
        self,
        retention_state: str,
        final_strength: float,
        decay_factor: float,
        effective_lambda: float,
        age_days: float,
        is_protected: bool,
        protection_reason: Optional[str],
        forgetting_eligible: bool,
        forgetting_reasons: List[str],
        baseline_score: float,
        resilience: float
    ) -> str:
        """Constructs human-readable, data-driven explanation for the retention decision."""
        parts: List[str] = [
            f"Retention strength: {final_strength:.4f} -> State: [{retention_state}]."
        ]

        if age_days < 0.05:
            parts.append("Recent observation with zero/negligible temporal decay.")
        else:
            parts.append(
                f"Elapsed time of {age_days:.2f} days yielded temporal decay factor of {decay_factor:.4f} "
                f"(effective decay rate: {effective_lambda:.4f}/day)."
            )

        parts.append(f"Cognitive resilience: {resilience:.3f} based on initial baseline score {baseline_score:.3f}.")

        if is_protected:
            parts.append(f"Protected by high-value policy: {protection_reason} (enforced floor: {self.config.protection_floor:.2f}).")

        if retention_state == "RETAINED":
            parts.append("Active knowledge tier maintained due to strong cognitive value and minimal decay.")
        elif retention_state == "DECAYING":
            parts.append("Retention strength is gradually decaying over time; remains actively available.")
        elif retention_state == "ARCHIVED":
            parts.append("Decayed below active threshold; preserved in secondary archive without forgetting.")
        elif retention_state == "FORGETTABLE":
            parts.append(f"Qualified for logical forgetting under multi-signal criteria: {'; '.join(forgetting_reasons)}.")

        return " ".join(parts)

    def evaluate_memory(
        self,
        memory: Dict[str, Any],
        ref_timestamp: float
    ) -> Phase7RetainedMemory:
        """
        Evaluates retention strength, temporal decay, protection, and retention state for a single memory.
        """
        cid = str(memory.get("consolidated_memory_id") or memory.get("ltm_id") or memory.get("candidate_id") or "")
        src_mids = list(memory.get("source_memory_ids") or [cid])
        src_eids = list(memory.get("source_episode_ids") or [])
        primary_app = str(memory.get("primary_application") or memory.get("application") or "Desktop App")

        # 1. Feature extraction
        (
            baseline_score,
            recency,
            frequency,
            retrieval_history,
            semantic_relevance,
            relationship_strength,
            source_count,
        ) = self.extract_feature_values(memory)

        # 2. Elapsed time calculation
        mem_ts = parse_timestamp_iso(
            memory.get("end_time") or memory.get("last_reinforced") or memory.get("start_time") or memory.get("created_at")
        )

        if mem_ts > 0.0 and ref_timestamp > 0.0:
            if mem_ts > ref_timestamp:
                delta_sec = 0.0  # Future timestamp safe clamp
            else:
                delta_sec = ref_timestamp - mem_ts
        else:
            delta_sec = 0.0  # Missing timestamp safe fallback

        age_days = delta_sec / 86400.0

        # 3. Adaptive resilience & Effective decay rate
        resilience = self.calculate_adaptive_resilience(
            baseline_score, frequency, retrieval_history, semantic_relevance, relationship_strength
        )
        effective_lambda = self.config.base_decay_rate * (
            1.0 - self.config.adaptive_decay_alpha * resilience
        )
        effective_lambda = max(0.001, effective_lambda)

        # 4. Temporal decay factor
        decay_factor = math.exp(-effective_lambda * age_days)
        decay_factor = max(0.0, min(1.0, decay_factor))

        # 5. Composite initial value & raw retention strength
        w_init = self.config.initial_value_weight
        initial_val = w_init * baseline_score + (1.0 - w_init) * resilience
        adaptive_factor = (1.0 + resilience) / 2.0  # Adaptive evidence factor in [0.5, 1.0]

        raw_retention = initial_val * decay_factor
        raw_retention = max(0.0, min(1.0, raw_retention))

        # 6. High-value protection check
        is_protected, protection_reason = self.evaluate_protection(
            baseline_score, frequency, retrieval_history, semantic_relevance, relationship_strength, source_count
        )

        if is_protected:
            final_strength = max(raw_retention, self.config.protection_floor)
        else:
            final_strength = raw_retention

        final_strength = round(max(0.0, min(1.0, final_strength)), 4)

        # 7. Forgetting eligibility & Retention state
        forgetting_eligible, forgetting_reasons = self.evaluate_forgetting_eligibility(
            final_strength, age_days, is_protected, frequency, retrieval_history, semantic_relevance, relationship_strength
        )

        retention_state = self.config.classify_state(final_strength, forgetting_eligible)

        # 8. Explanation generation
        explanation = self.construct_retention_explanation(
            retention_state,
            final_strength,
            decay_factor,
            effective_lambda,
            age_days,
            is_protected,
            protection_reason,
            forgetting_eligible,
            forgetting_reasons,
            baseline_score,
            resilience,
        )

        eval_iso = datetime.fromtimestamp(ref_timestamp, tz=timezone.utc).isoformat()

        # Stability score carries over or incorporates retention strength
        stability = float(memory.get("stability_score") or memory.get("baseline_information", {}).get("stability_score") or baseline_score)

        return Phase7RetainedMemory(
            consolidated_memory_id=cid,
            source_memory_ids=src_mids,
            source_episode_ids=src_eids,
            primary_application=primary_app,
            baseline_score=round(baseline_score, 4),
            recency=round(recency, 4),
            frequency=round(frequency, 4),
            retrieval_history=round(retrieval_history, 4),
            semantic_relevance=round(semantic_relevance, 4),
            relationship_strength=round(relationship_strength, 4),
            stability_score=round(stability, 4),
            source_count=source_count,
            age_days=round(age_days, 3),
            decay_factor=round(decay_factor, 4),
            effective_decay_rate=round(effective_lambda, 5),
            adaptive_resilience=round(resilience, 4),
            adaptive_retention_factor=round(adaptive_factor, 4),
            final_retention_strength=final_strength,
            retention_state=retention_state,
            is_protected=is_protected,
            forgetting_eligible=forgetting_eligible,
            protection_reason=protection_reason,
            retention_explanation=explanation,
            evaluation_timestamp=eval_iso,
            topic=str(memory.get("topic") or f"{primary_app}: {cid}"),
            summary=str(memory.get("semantic_representation") or memory.get("summary") or ""),
            start_time=str(memory.get("start_time") or ""),
            end_time=str(memory.get("end_time") or ""),
            total_duration_ms=int(memory.get("duration_ms") or memory.get("total_duration_ms") or 0),
            total_interactions=int(memory.get("total_interactions") or memory.get("interaction_statistics", {}).get("total_interactions", 0)),
            unique_windows_accessed=list(memory.get("contexts") or memory.get("unique_windows_accessed") or []),
            domains=list(memory.get("domains") or []),
        )

    def process(
        self,
        memories: List[Dict[str, Any]],
        reference_time: Optional[Union[str, datetime]] = None
    ) -> Tuple[List[Phase7RetainedMemory], Dict[str, Any]]:
        """
        Executes the complete Phase 7 Adaptive Retention pipeline on a list of memory nodes.
        Returns:
            retained_memories: List of Phase7RetainedMemory objects
            metrics: Dict containing aggregate retention counts, distributions, and stats
        """
        start_time = time.time()
        logger.info(f"[PHASE 7] Ingesting {len(memories):,} memories for adaptive retention evaluation...")

        if not memories:
            logger.info("[PHASE 7] Received empty memory list. Returning empty evaluation.")
            return [], {
                "total_memories": 0,
                "retained_count": 0,
                "decaying_count": 0,
                "archived_count": 0,
                "forgetting_eligible_count": 0,
                "protected_count": 0,
                "average_retention_strength": 0.0,
                "min_retention_strength": 0.0,
                "max_retention_strength": 0.0,
                "processing_time_seconds": round(time.time() - start_time, 4),
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            }

        ref_timestamp = self.resolve_reference_timestamp(memories, reference_time)
        eval_iso = datetime.fromtimestamp(ref_timestamp, tz=timezone.utc).isoformat()
        logger.info(f"[PHASE 7] Reference/Evaluation Timestamp: {eval_iso}")

        results: List[Phase7RetainedMemory] = []
        for m in memories:
            res = self.evaluate_memory(m, ref_timestamp)
            results.append(res)

        # Sort deterministically by retention strength descending, then memory ID
        results.sort(key=lambda x: (-x.final_retention_strength, x.consolidated_memory_id))

        # Metrics aggregation
        strengths = [r.final_retention_strength for r in results]
        state_counts = {
            "RETAINED": sum(1 for r in results if r.retention_state == "RETAINED"),
            "DECAYING": sum(1 for r in results if r.retention_state == "DECAYING"),
            "ARCHIVED": sum(1 for r in results if r.retention_state == "ARCHIVED"),
            "FORGETTABLE": sum(1 for r in results if r.retention_state == "FORGETTABLE"),
        }
        protected_count = sum(1 for r in results if r.is_protected)
        forgetting_eligible_count = sum(1 for r in results if r.forgetting_eligible)

        metrics = {
            "total_memories": len(results),
            "retained_count": state_counts["RETAINED"],
            "decaying_count": state_counts["DECAYING"],
            "archived_count": state_counts["ARCHIVED"],
            "forgetting_eligible_count": forgetting_eligible_count,
            "protected_count": protected_count,
            "state_distribution": state_counts,
            "state_percentages": {
                k: round((v / len(results)) * 100.0, 2) for k, v in state_counts.items()
            },
            "average_retention_strength": round(sum(strengths) / len(strengths), 4) if strengths else 0.0,
            "min_retention_strength": round(min(strengths), 4) if strengths else 0.0,
            "max_retention_strength": round(max(strengths), 4) if strengths else 0.0,
            "evaluation_timestamp": eval_iso,
            "processing_time_seconds": round(time.time() - start_time, 4),
            "configuration": self.config.to_dict(),
        }

        logger.info(
            f"[PHASE 7] Evaluated {len(results):,} memories in {metrics['processing_time_seconds']}s: "
            f"RETAINED={state_counts['RETAINED']}, DECAYING={state_counts['DECAYING']}, "
            f"ARCHIVED={state_counts['ARCHIVED']}, FORGETTABLE={state_counts['FORGETTABLE']}. "
            f"Avg Strength: {metrics['average_retention_strength']:.4f} [min: {metrics['min_retention_strength']:.4f}, max: {metrics['max_retention_strength']:.4f}]."
        )

        return results, metrics

    def run_pipeline(
        self,
        input_path: Optional[Path] = None,
        reference_time: Optional[Union[str, datetime]] = None,
        save_to_db: bool = True,
        save_to_json: bool = True,
    ) -> Tuple[List[Phase7RetainedMemory], Dict[str, Any]]:
        """
        Runs the full Phase 7 pipeline from persistent Phase 6 output and saves the results.
        """
        p = input_path or CONSOLIDATED_JSON_PATH
        if not p.exists():
            raise FileNotFoundError(f"Phase 6 consolidated memory file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            memories = json.load(f)

        logger.info(f"[PHASE 7] Loaded {len(memories):,} consolidated memories from {p}")

        results, metrics = self.process(memories, reference_time=reference_time)

        if save_to_json:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            dict_results = [r.to_dict() for r in results]

            with open(RETENTION_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(dict_results, f, indent=2)
            logger.info(f"[PHASE 7] Saved {len(dict_results):,} evaluated memories to {RETENTION_JSON_PATH}")

            with open(RETENTION_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[PHASE 7] Saved retention metrics to {RETENTION_METRICS_PATH}")

            with open(RETENTION_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": metrics,
                    "configuration": self.config.to_dict(),
                    "sample_evaluations": dict_results[:5],
                }, f, indent=2)
            logger.info(f"[PHASE 7] Saved detailed report to {RETENTION_REPORT_PATH}")

        if save_to_db:
            init_dmclm_database()
            insert_or_replace_retention_memories([r.to_dict() for r in results])
            logger.info(f"[PHASE 7] Persisted {len(results):,} evaluated memories to SQLite.")

        return results, metrics


def trace_retention_memory(
    consolidated_id: str,
    retention_nodes: Optional[List[Phase7RetainedMemory]] = None
) -> Dict[str, Any]:
    """
    Provenance query function for Phase 7.
    Returns complete lineage:
    Retention Decision -> Consolidated Node -> Source Candidates -> Source Episodes.
    """
    node_dict = None
    if retention_nodes:
        for n in retention_nodes:
            if n.consolidated_memory_id == consolidated_id:
                node_dict = n.to_dict()
                break
    elif RETENTION_JSON_PATH.exists():
        with open(RETENTION_JSON_PATH, "r", encoding="utf-8") as f:
            all_nodes = json.load(f)
            for n in all_nodes:
                if n.get("consolidated_memory_id") == consolidated_id:
                    node_dict = n
                    break

    if not node_dict:
        raise KeyError(f"Retention evaluated memory not found: {consolidated_id}")

    return {
        "consolidated_memory_id": consolidated_id,
        "retention_state": node_dict.get("retention_state"),
        "final_retention_strength": node_dict.get("final_retention_strength"),
        "is_protected": node_dict.get("is_protected"),
        "protection_reason": node_dict.get("protection_reason"),
        "forgetting_eligible": node_dict.get("forgetting_eligible"),
        "retention_explanation": node_dict.get("retention_explanation"),
        "source_memory_count": len(node_dict.get("source_memory_ids", [])),
        "source_memory_ids": node_dict.get("source_memory_ids", []),
        "source_episode_ids": node_dict.get("source_episode_ids", []),
        "evaluation_timestamp": node_dict.get("evaluation_timestamp"),
    }


if __name__ == "__main__":
    engine = Phase7AdaptiveRetentionEngine()
    results, metrics = engine.run_pipeline()
    print("\n[PHASE 7 RETENTION & FORGETTING SUMMARY]")
    print(f"Total Consolidated Memories:  {metrics['total_memories']:,}")
    print(f"RETAINED (Active):            {metrics['retained_count']:,} ({metrics['state_percentages']['RETAINED']}%)")
    print(f"DECAYING:                     {metrics['decaying_count']:,} ({metrics['state_percentages']['DECAYING']}%)")
    print(f"ARCHIVED:                     {metrics['archived_count']:,} ({metrics['state_percentages']['ARCHIVED']}%)")
    print(f"FORGETTABLE:                  {metrics['forgetting_eligible_count']:,} ({metrics['state_percentages']['FORGETTABLE']}%)")
    print(f"High-Value Protected:         {metrics['protected_count']:,}")
    print(f"Average Retention Strength:   {metrics['average_retention_strength']:.4f}")
    print(f"Strength Range:               [{metrics['min_retention_strength']:.4f}, {metrics['max_retention_strength']:.4f}]")
    print(f"Processing Time:              {metrics['processing_time_seconds']}s")


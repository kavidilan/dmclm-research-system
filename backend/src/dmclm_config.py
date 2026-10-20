"""
DMCLM Configuration Module
Contains transparent, configurable parameters for:
1. Activity Episode Grouping & Preprocessing
2. Five Feature Signals (Recency decay, Frequency normalization, etc.)
3. Baseline DMCLM Importance Scoring Weights
4. Cognitive Lifecycle Decision Thresholds
5. Consolidation & Compression Parameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class PreprocessingConfig:
    """Configuration for raw observation aggregation into activity episodes."""
    # Maximum gap in seconds between consecutive activities to be considered part of the same episode
    temporal_proximity_seconds: float = 300.0  # 5 minutes
    
    # Whether episodes must strictly be contained within a single collection session
    bound_by_session: bool = True
    
    # Minimum duration (ms) for an activity to be considered non-noise (unless repeated or interacted with)
    noise_duration_threshold_ms: int = 250
    
    # Maximum character length for window titles to normalize truncation/noise
    max_title_length: int = 300
    
    # Whether to normalize browser page titles by stripping trailing browser suffixes
    clean_browser_titles: bool = True


@dataclass
class MemoryExtractionConfig:
    """Configuration for Phase 3 — Memory Extraction from Phase 2 Activity Episodes."""
    # Minimum duration (ms) for an episode to be considered a meaningful candidate on its own
    min_meaningful_duration_ms: int = 2500  # 2.5 seconds dwell

    # Minimum interaction count to qualify short burst episodes (< min_meaningful_duration_ms)
    min_interactions_for_short_burst: int = 1

    # Filter transient system background processes that lack user engagement
    filter_transient_system_noise: bool = True
    system_noise_applications: tuple = (
        "ShellHost", "SearchHost", "Windows Shell Experience Host", "OpenWith"
    )

    # Window (seconds) to correlate concurrent browser research events with activity episodes
    browser_correlation_window_sec: float = 60.0

    # Max character length for generated activity summary
    max_summary_length: int = 350


@dataclass
class FeatureConfig:
    """
    Configuration for the 5 DMCLM cognitive feature signals (Phase 4):
    1. Recency: Exponential temporal decay R = exp(-lambda * delta_t_days)
    2. Frequency: Composite of inter-episode repetition and intra-episode interactions
    3. Retrieval History: Audit-log ratio with cold-start distinction
    4. Semantic Relevance: Local TF-IDF session and global centroid similarity
    5. Relationship Strength: Multi-modal graph affinity across neighbors
    """
    # 1. Recency Parameters
    # Recency decay parameter lambda in exp(-lambda * days_since_last_seen)
    # With lambda = 0.05, half-life is ~13.86 days. After 14 days, recency is ~0.496.
    recency_lambda: float = 0.05
    recency_reference_mode: str = "max_observed"  # "max_observed" or "current_time"

    # 2. Frequency Parameters
    # Distinguishes repeated occurrence across episodes vs user actions within one episode
    freq_weight_inter: float = 0.70  # Inter-episode task repetition across sessions/apps/domains
    freq_weight_intra: float = 0.30  # Intra-episode user interaction density (clicks/keystrokes)

    # 3. Retrieval History Parameters
    retrieval_ratio_weight: float = 0.70  # Weight for useful/total retrieval ratio
    retrieval_freq_weight: float = 0.30   # Weight for retrieval frequency factor
    retrieval_saturation_count: float = 4.0  # Number of retrievals for full frequency factor
    retrieval_cold_start_default: float = 0.0  # Scientifically defensible cold-start score

    # 4. Semantic Relevance Parameters
    tfidf_max_features: int = 1500
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    semantic_session_weight: float = 0.70  # Weight for session task centroid
    semantic_global_weight: float = 0.30   # Weight for global task centroid

    # 5. Relationship Strength Parameters
    rel_neighbor_window: int = 10  # Chronological neighbor window for pairwise graph evaluation
    relationship_temporal_window_sec: float = 600.0  # 10 minutes temporal proximity kernel tau
    rel_weight_session: float = 0.35   # Shared session affinity
    rel_weight_app: float = 0.30       # Shared application and domain affinity
    rel_weight_temporal: float = 0.25  # Temporal proximity kernel
    rel_weight_context: float = 0.10   # Window context / title overlap affinity


@dataclass
class BaselineImportanceWeights:
    """
    Transparent experimental baseline weights for DMCLM importance score:
    Baseline Score = w_r * Recency + w_f * Frequency + w_h * Retrieval History
                   + w_s * Semantic Relevance + w_rel * Relationship Strength
    Note: These weights are experimental baseline values for this implementation (sum = 1.0).
    """
    recency: float = 0.20
    frequency: float = 0.20
    retrieval_history: float = 0.25
    semantic_relevance: float = 0.20
    relationship_strength: float = 0.15

    def validate(self) -> None:
        """Validate that all weights are non-negative and sum to 1.0 within tolerance."""
        weights_dict = self.to_dict()
        for k, v in weights_dict.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"Baseline weight '{k}' must be numeric, got {type(v).__name__}: {v}")
            if v < 0.0:
                raise ValueError(f"Baseline weight '{k}' must be non-negative, got {v}")

        total = sum(weights_dict.values())
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"Baseline weights must sum to 1.0 within floating-point tolerance (got sum={total:.6f}). "
                f"Weights: {weights_dict}"
            )

    def to_dict(self) -> Dict[str, float]:
        return {
            "recency": self.recency,
            "frequency": self.frequency,
            "retrieval_history": self.retrieval_history,
            "semantic_relevance": self.semantic_relevance,
            "relationship_strength": self.relationship_strength,
        }


@dataclass
class ValueTierThresholds:
    """
    Interpretation thresholds for Phase 5 Baseline DMCLM scores:
    - Score >= high_threshold (0.70): HIGH_VALUE
    - medium_threshold (0.40) <= Score < high_threshold: MEDIUM_VALUE
    - Score < medium_threshold: LOW_VALUE
    """
    high_threshold: float = 0.70
    medium_threshold: float = 0.40

    def classify(self, score: float) -> str:
        if score >= self.high_threshold:
            return "HIGH_VALUE"
        elif score >= self.medium_threshold:
            return "MEDIUM_VALUE"
        else:
            return "LOW_VALUE"

    def to_dict(self) -> Dict[str, float]:
        return {
            "high_threshold": self.high_threshold,
            "medium_threshold": self.medium_threshold,
        }


@dataclass
class LifecycleThresholds:
    """
    Experimental baseline lifecycle policy thresholds:
    - Importance >= retain_min: RETAIN
    - compress_min <= Importance < retain_min: COMPRESS
    - archive_min <= Importance < compress_min: ARCHIVE
    - Importance < archive_min: FORGET
    """
    retain_min: float = 0.70
    compress_min: float = 0.40
    archive_min: float = 0.20

    def classify(self, importance: float) -> str:
        if importance >= self.retain_min:
            return "RETAIN"
        elif importance >= self.compress_min:
            return "COMPRESS"
        elif importance >= self.archive_min:
            return "ARCHIVE"
        else:
            return "FORGET"

    def to_dict(self) -> Dict[str, float]:
        return {
            "retain_min": self.retain_min,
            "compress_min": self.compress_min,
            "archive_min": self.archive_min,
        }


@dataclass
class ConsolidationConfig:
    """
    Configuration for Phase 6 — Consolidation & Compression.
    Parameters governing candidate pairing, multi-evidence affinity, and deterministic clustering.
    """
    # Minimum cluster size to form a consolidated long-term memory node
    min_group_size: int = 2

    # Composite affinity threshold required to consider two memories redundant/related
    affinity_threshold: float = 0.55

    # Minimum text/semantic token overlap similarity threshold
    semantic_similarity_threshold: float = 0.35

    # Temporal blocking window (seconds) to bound candidate pair evaluation
    temporal_window_sec: float = 86400.0  # 24 hours

    # Temporal proximity exponential decay kernel tau (seconds)
    temporal_proximity_tau_sec: float = 3600.0  # 1 hour

    # Multi-evidence affinity component weights (must sum to 1.0)
    weight_app: float = 0.20
    weight_domain: float = 0.15
    weight_context: float = 0.25
    weight_semantic: float = 0.25
    weight_temporal: float = 0.10
    weight_session: float = 0.05

    def validate(self) -> None:
        """Validate that all weights are non-negative and sum to 1.0 within tolerance."""
        weights = {
            "weight_app": self.weight_app,
            "weight_domain": self.weight_domain,
            "weight_context": self.weight_context,
            "weight_semantic": self.weight_semantic,
            "weight_temporal": self.weight_temporal,
            "weight_session": self.weight_session,
        }
        for k, v in weights.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"Consolidation weight '{k}' must be numeric, got {type(v).__name__}: {v}")
            if v < 0.0:
                raise ValueError(f"Consolidation weight '{k}' must be non-negative, got {v}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"Consolidation weights must sum to 1.0 within floating-point tolerance (got sum={total:.6f})."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_group_size": self.min_group_size,
            "affinity_threshold": self.affinity_threshold,
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "temporal_window_sec": self.temporal_window_sec,
            "temporal_proximity_tau_sec": self.temporal_proximity_tau_sec,
            "weights": {
                "weight_app": self.weight_app,
                "weight_domain": self.weight_domain,
                "weight_context": self.weight_context,
                "weight_semantic": self.weight_semantic,
                "weight_temporal": self.weight_temporal,
                "weight_session": self.weight_session,
            }
        }


@dataclass
class AdaptiveRetentionConfig:
    """
    Configuration for Phase 7 — Adaptive Retention / Forgetting.
    Parameters governing:
    1. Scientifically reasonable temporal decay (exponential decay with adaptive rate).
    2. High-value memory protection thresholds.
    3. Multi-signal forgetting eligibility criteria.
    4. Discrete retention state boundaries (RETAINED, DECAYING, ARCHIVED, FORGETTABLE).
    """
    # Base temporal decay rate lambda_base (per day). With 0.05, half-life is ~13.86 days.
    base_decay_rate: float = 0.05

    # Adaptive decay sensitivity alpha: determines how much high cognitive resilience can reduce lambda.
    # lambda_eff = lambda_base * (1.0 - adaptive_decay_alpha * resilience)
    # With alpha=0.60, max decay rate reduction is 60%, slowing decay for high-value memories.
    adaptive_decay_alpha: float = 0.60

    # Cognitive resilience weights (sum = 1.0)
    resilience_weight_baseline: float = 0.30
    resilience_weight_frequency: float = 0.20
    resilience_weight_retrieval: float = 0.20
    resilience_weight_semantic: float = 0.15
    resilience_weight_relationship: float = 0.15

    # Weight balancing initial baseline value vs composite adaptive evidence
    initial_value_weight: float = 0.50

    # Retention State Thresholds
    retain_threshold: float = 0.60      # R >= 0.60 -> RETAINED
    archive_threshold: float = 0.35     # 0.35 <= R < 0.60 -> DECAYING
    forget_threshold: float = 0.20      # R < 0.20 -> FORGETTABLE (if criteria met), else ARCHIVED

    # High-Value Protection Parameters
    enable_protection: bool = True
    protection_baseline_threshold: float = 0.60     # Memories with baseline score >= this are protected
    protection_frequency_threshold: float = 0.70    # High repetition protection
    protection_retrieval_threshold: float = 0.40    # High retrieval audit protection
    protection_source_count_threshold: int = 50     # Large consolidated mass protection
    protection_floor: float = 0.40                  # Guaranteed minimum retention score for protected nodes

    # Forgetting Eligibility Multi-Signal Criteria
    min_age_days_for_forgetting: float = 7.0        # Memory must be at least this old to be eligible
    forgetting_freq_max: float = 0.35               # Low frequency required
    forgetting_retrieval_max: float = 0.15          # Low retrieval required
    forgetting_semantic_max: float = 0.35           # Low semantic relevance required
    forgetting_relationship_max: float = 0.35       # Low relationship strength required

    # Reference Time Configuration
    reference_mode: str = "max_observed"            # "max_observed", "current_time", or explicit ISO string
    explicit_reference_timestamp: Optional[str] = None

    def validate(self) -> None:
        """Validate configuration parameters and weight sums."""
        weights = {
            "resilience_weight_baseline": self.resilience_weight_baseline,
            "resilience_weight_frequency": self.resilience_weight_frequency,
            "resilience_weight_retrieval": self.resilience_weight_retrieval,
            "resilience_weight_semantic": self.resilience_weight_semantic,
            "resilience_weight_relationship": self.resilience_weight_relationship,
        }
        for k, v in weights.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"Resilience weight '{k}' must be numeric, got {type(v).__name__}: {v}")
            if v < 0.0:
                raise ValueError(f"Resilience weight '{k}' must be non-negative, got {v}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"Resilience weights must sum to 1.0 within tolerance (got sum={total:.6f})."
            )
        if not (0.0 < self.archive_threshold < self.retain_threshold <= 1.0):
            raise ValueError("Threshold ordering must satisfy: 0.0 < archive_threshold < retain_threshold <= 1.0")
        if not (0.0 <= self.forget_threshold <= self.archive_threshold):
            raise ValueError("forget_threshold must be between 0.0 and archive_threshold")
        if self.base_decay_rate < 0.0:
            raise ValueError("base_decay_rate must be non-negative")
        if not (0.0 <= self.adaptive_decay_alpha <= 1.0):
            raise ValueError("adaptive_decay_alpha must be in [0.0, 1.0]")

    def classify_state(self, retention_strength: float, is_forgetting_eligible: bool) -> str:
        if retention_strength >= self.retain_threshold:
            return "RETAINED"
        elif retention_strength >= self.archive_threshold:
            return "DECAYING"
        elif is_forgetting_eligible:
            return "FORGETTABLE"
        else:
            return "ARCHIVED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_decay_rate": self.base_decay_rate,
            "adaptive_decay_alpha": self.adaptive_decay_alpha,
            "initial_value_weight": self.initial_value_weight,
            "retain_threshold": self.retain_threshold,
            "archive_threshold": self.archive_threshold,
            "forget_threshold": self.forget_threshold,
            "enable_protection": self.enable_protection,
            "protection_baseline_threshold": self.protection_baseline_threshold,
            "protection_floor": self.protection_floor,
            "min_age_days_for_forgetting": self.min_age_days_for_forgetting,
            "reference_mode": self.reference_mode,
            "resilience_weights": {
                "baseline": self.resilience_weight_baseline,
                "frequency": self.resilience_weight_frequency,
                "retrieval": self.resilience_weight_retrieval,
                "semantic": self.resilience_weight_semantic,
                "relationship": self.resilience_weight_relationship,
            }
        }


@dataclass
class RelationshipConfig:
    """
    Configuration for Phase 8 — Relationship Management.
    Parameters governing:
    1. Multi-modal relationship strength evaluation (semantic, context, application, domain, temporal).
    2. Meaningful relationship type classification thresholds.
    3. Graph topology constraints and filtering.
    """
    # Minimum composite relationship strength to accept an edge in the graph
    min_strength_threshold: float = 0.35

    # Specific relationship type threshold criteria
    semantic_threshold: float = 0.30       # Minimum semantic similarity to qualify for SEMANTIC_SIMILARITY
    context_threshold: float = 0.25        # Minimum window context token overlap for WORKFLOW_CONTINUITY
    domain_threshold: float = 0.30         # Minimum domain overlap for SHARED_DOMAIN
    temporal_proximity_window_sec: float = 86400.0  # Max temporal gap (24h) for TEMPORAL_ASSOCIATION
    temporal_proximity_tau_sec: float = 21600.0     # 6 hours exponential kernel scale

    # Composite relationship strength weights (must sum to 1.0)
    weight_semantic: float = 0.30
    weight_context: float = 0.25
    weight_application: float = 0.20
    weight_domain: float = 0.15
    weight_temporal: float = 0.10

    # Graph constraints
    max_neighbors_per_node: Optional[int] = None  # None for unbounded complete evidence graph

    def validate(self) -> None:
        """Validate relationship weights and parameters."""
        weights = {
            "weight_semantic": self.weight_semantic,
            "weight_context": self.weight_context,
            "weight_application": self.weight_application,
            "weight_domain": self.weight_domain,
            "weight_temporal": self.weight_temporal,
        }
        for k, v in weights.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"Relationship weight '{k}' must be numeric, got {type(v).__name__}: {v}")
            if v < 0.0:
                raise ValueError(f"Relationship weight '{k}' must be non-negative, got {v}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"Relationship weights must sum to 1.0 within tolerance (got sum={total:.6f})."
            )
        if not (0.0 <= self.min_strength_threshold <= 1.0):
            raise ValueError("min_strength_threshold must be in [0.0, 1.0]")
        if self.temporal_proximity_window_sec <= 0:
            raise ValueError("temporal_proximity_window_sec must be positive")
        if self.temporal_proximity_tau_sec <= 0:
            raise ValueError("temporal_proximity_tau_sec must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_strength_threshold": self.min_strength_threshold,
            "semantic_threshold": self.semantic_threshold,
            "context_threshold": self.context_threshold,
            "domain_threshold": self.domain_threshold,
            "temporal_proximity_window_sec": self.temporal_proximity_window_sec,
            "temporal_proximity_tau_sec": self.temporal_proximity_tau_sec,
            "weights": {
                "semantic": self.weight_semantic,
                "context": self.weight_context,
                "application": self.weight_application,
                "domain": self.weight_domain,
                "temporal": self.weight_temporal,
            }
        }


@dataclass
class DatasetGenerationConfig:
    """
    Configuration for Phase 9 — Training Dataset Generation:
    Extracts, enriches, and structures ML-ready training, validation, and testing datasets
    from real Phase 1-8 processed memory records and relationship graphs.
    """
    split_strategy: str = "temporal"  # "temporal", "random", or "stratified"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Target settings
    primary_categorical_target: str = "target_lifecycle_state"
    primary_continuous_target: str = "target_importance_score"
    weak_supervision_label_origin: str = "behavior_derived_baseline_policy"
    future_observation_window_hours: float = 48.0

    # Preprocessing and enrichment flags
    include_graph_features: bool = True
    include_temporal_features: bool = True
    include_activity_features: bool = True
    export_formats: tuple = ("csv", "json")
    min_meaningful_features: int = 5

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0 within tolerance, got {total:.4f}")
        if self.split_strategy not in ("temporal", "random", "stratified"):
            raise ValueError(f"Unsupported split strategy: {self.split_strategy}")
        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative with positive train ratio.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_strategy": self.split_strategy,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "primary_categorical_target": self.primary_categorical_target,
            "primary_continuous_target": self.primary_continuous_target,
            "weak_supervision_label_origin": self.weak_supervision_label_origin,
            "future_observation_window_hours": self.future_observation_window_hours,
            "include_graph_features": self.include_graph_features,
            "include_temporal_features": self.include_temporal_features,
            "include_activity_features": self.include_activity_features,
            "export_formats": list(self.export_formats),
            "min_meaningful_features": self.min_meaningful_features,
        }


@dataclass
class ModelTrainingConfig:
    """
    Configuration for Phase 10 — ML Model Training and Prediction:
    Configures lightweight, interpretable, local machine learning models
    trained strictly on Phase 9 dataset partitions.
    """
    algorithm: str = "random_forest"  # "random_forest", "gradient_boosting", "logistic_regression"
    feature_set: str = "core_five"  # "core_five" (5 core DMCLM features) or "enriched"
    target_column: str = "target_lifecycle_state"
    regression_target_column: str = "target_importance_score"
    n_estimators: int = 100
    max_depth: int = 6
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    random_seed: int = 42
    scale_features: bool = True
    train_regressor: bool = True

    def validate(self) -> None:
        if self.algorithm not in ("random_forest", "gradient_boosting", "logistic_regression"):
            raise ValueError(f"Unsupported ML algorithm: {self.algorithm}")
        if self.feature_set not in ("core_five", "enriched"):
            raise ValueError(f"Unsupported feature set: {self.feature_set}")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "feature_set": self.feature_set,
            "target_column": self.target_column,
            "regression_target_column": self.regression_target_column,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_seed": self.random_seed,
            "scale_features": self.scale_features,
            "train_regressor": self.train_regressor,
        }


@dataclass
class ComparisonConfig:
    """
    Configuration for Phase 11 — ML vs Baseline Comparison.
    Governs evaluation split isolation, statistical significance thresholds, and artifact export.
    """
    split: str = "test"
    significance_alpha: float = 0.05
    compute_mcnemar: bool = True
    compute_wilcoxon: bool = True
    export_per_memory_csv: bool = True

    def validate(self) -> None:
        if self.split not in ("test", "val", "train", "all"):
            raise ValueError(f"Invalid evaluation split: {self.split}")
        if not (0.0 < self.significance_alpha < 1.0):
            raise ValueError("significance_alpha must be strictly within (0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "significance_alpha": self.significance_alpha,
            "compute_mcnemar": self.compute_mcnemar,
            "compute_wilcoxon": self.compute_wilcoxon,
            "export_per_memory_csv": self.export_per_memory_csv,
        }


@dataclass
class ExperimentSuiteConfig:
    """
    Configuration for Phase 12 — Experiments & Evaluation.
    Governs multi-seed evaluations, feature ablation configurations, reproducibility trials,
    and visualization generation.
    """
    suite_id: str = "dmclm_phase12_suite"
    random_seeds: List[int] = field(default_factory=lambda: [42, 101, 202, 303, 404])
    output_dir: str = "results/experiments"
    figures_dir: str = "results/figures"
    generate_plots: bool = True
    significance_alpha: float = 0.05
    reproducibility_trials: int = 3

    def validate(self) -> None:
        if not self.random_seeds:
            raise ValueError("random_seeds cannot be empty")
        if not (0.0 < self.significance_alpha < 1.0):
            raise ValueError("significance_alpha must be strictly within (0, 1)")
        if self.reproducibility_trials < 2:
            raise ValueError("reproducibility_trials must be at least 2")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "random_seeds": self.random_seeds,
            "output_dir": self.output_dir,
            "figures_dir": self.figures_dir,
            "generate_plots": self.generate_plots,
            "significance_alpha": self.significance_alpha,
            "reproducibility_trials": self.reproducibility_trials,
        }


@dataclass
class ResourceAwareConfig:
    """
    Configuration for Phase 13 — Resource-Aware Reasoning.
    Governs local system resource monitoring, dynamic profile classification (LOW, NORMAL, HIGH),
    computational budgets, candidate pair prioritization, and graceful degradation.
    """
    enabled: bool = True
    cpu_high_threshold: float = 75.0        # Above this, CPU is considered constrained
    cpu_low_threshold: float = 30.0         # Below this, CPU is unconstrained
    memory_low_mb: float = 1024.0           # Below this available RAM (MB), memory is constrained
    memory_low_pct: float = 15.0            # Below this available RAM (%), memory is constrained
    process_memory_budget_mb: float = 512.0 # Target process RSS ceiling
    batch_size_high: int = 500              # Batch size under HIGH availability
    batch_size_normal: int = 200            # Batch size under NORMAL availability
    batch_size_low: int = 50                # Batch size under LOW availability
    relationship_pair_budget_high: int = 10000
    relationship_pair_budget_normal: int = 5000
    relationship_pair_budget_low: int = 1500
    enable_similarity_caching: bool = True
    max_cache_size: int = 10000
    quality_priority_threshold: float = 0.70  # Memories >= 0.70 or protected are never deferred
    defer_marginal_threshold: float = 0.40    # Low tier cutoff for deferral under LOW profile
    graceful_degradation: bool = True
    results_dir: str = "results/resources"
    figures_dir: str = "results/figures"

    def validate(self) -> None:
        if not (0.0 < self.cpu_low_threshold < self.cpu_high_threshold <= 100.0):
            raise ValueError("CPU thresholds must satisfy 0 < cpu_low < cpu_high <= 100")
        if self.memory_low_mb <= 0:
            raise ValueError("memory_low_mb must be positive")
        if not (0 < self.batch_size_low < self.batch_size_normal < self.batch_size_high):
            raise ValueError("Batch sizes must satisfy 0 < low < normal < high")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cpu_high_threshold": self.cpu_high_threshold,
            "cpu_low_threshold": self.cpu_low_threshold,
            "memory_low_mb": self.memory_low_mb,
            "memory_low_pct": self.memory_low_pct,
            "process_memory_budget_mb": self.process_memory_budget_mb,
            "batch_size_high": self.batch_size_high,
            "batch_size_normal": self.batch_size_normal,
            "batch_size_low": self.batch_size_low,
            "relationship_pair_budget_high": self.relationship_pair_budget_high,
            "relationship_pair_budget_normal": self.relationship_pair_budget_normal,
            "relationship_pair_budget_low": self.relationship_pair_budget_low,
            "enable_similarity_caching": self.enable_similarity_caching,
            "max_cache_size": self.max_cache_size,
            "quality_priority_threshold": self.quality_priority_threshold,
            "defer_marginal_threshold": self.defer_marginal_threshold,
            "graceful_degradation": self.graceful_degradation,
            "results_dir": self.results_dir,
            "figures_dir": self.figures_dir,
        }


@dataclass
class DMCLMGlobalConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    memory_extraction: MemoryExtractionConfig = field(default_factory=MemoryExtractionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    weights: BaselineImportanceWeights = field(default_factory=BaselineImportanceWeights)
    thresholds: LifecycleThresholds = field(default_factory=LifecycleThresholds)
    value_tiers: ValueTierThresholds = field(default_factory=ValueTierThresholds)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    retention: AdaptiveRetentionConfig = field(default_factory=AdaptiveRetentionConfig)
    relationship: RelationshipConfig = field(default_factory=RelationshipConfig)
    dataset: DatasetGenerationConfig = field(default_factory=DatasetGenerationConfig)
    model: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    experiments: ExperimentSuiteConfig = field(default_factory=ExperimentSuiteConfig)
    resources: ResourceAwareConfig = field(default_factory=ResourceAwareConfig)


DEFAULT_CONFIG = DMCLMGlobalConfig()







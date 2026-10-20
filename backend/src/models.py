"""
Memory Consolidation Data Models
Represents Short-Term Memory (STM) items, Consolidated Long-Term Memory (LTM) units, and Consolidation Clusters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class RawCollectorItem:
    """Raw item ingested from Desktop Research Collector App."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    source_app: str = "CollectorApp"  # e.g., 'browser_pdf', 'research_notes', 'arxiv_reader'
    title: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    interaction_type: str = "read"   # 'read', 'highlight', 'note', 'search', 'bookmark'


@dataclass
class ShortTermMemoryItem:
    """Working Memory / Episodic representation in short-term buffer."""
    id: str
    timestamp: datetime
    content: str
    title: str
    source_app: str
    tags: List[str]
    access_count: int = 1
    interaction_weight: float = 1.0  # notes/highlights have higher initial weight than passive reads
    base_importance: float = 0.5
    current_retention_score: float = 1.0
    embedding: Optional[List[float]] = None
    is_consolidated: bool = False
    is_pruned: bool = False


@dataclass
class ConsolidatedMemoryNode:
    """Consolidated Long-Term Memory (LTM) representation.
    Produced by the Consolidation Algorithm from multiple episodic STM items.
    """
    node_id: str = field(default_factory=lambda: f"ltm_{uuid.uuid4().hex[:8]}")
    primary_concept: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    contributing_stm_ids: List[str] = field(default_factory=list)
    first_observed: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    reinforcement_count: int = 1
    stability_score: float = 0.5   # Synaptic strength in long-term memory
    semantic_tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class StructuredMemoryCandidate:
    """
    Phase 3 — Structured Memory Candidate
    Represents a meaningful desktop activity episode transformed into a structured memory candidate.
    Maintains full traceability back to Phase 2 source episode(s).
    """
    candidate_id: str
    session_id: str
    source_episode_ids: List[str]
    primary_application: str
    applications_involved: List[str]
    browser_domains: List[str]
    window_context: str
    start_time: str
    end_time: str
    duration_ms: int
    interaction_count: int
    browser_event_count: int
    screenshot_ids: List[str]
    activity_summary: str
    meaningful_criteria: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "source_episode_ids": self.source_episode_ids,
            "primary_application": self.primary_application,
            "applications_involved": self.applications_involved,
            "browser_domains": self.browser_domains,
            "window_context": self.window_context,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "interaction_count": self.interaction_count,
            "browser_event_count": self.browser_event_count,
            "screenshot_ids": self.screenshot_ids,
            "activity_summary": self.activity_summary,
            "meaningful_criteria": self.meaningful_criteria,
            "created_at": self.created_at,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class FeatureEnrichedMemory:
    """
    Phase 4 — Feature-Enriched Memory Representation.
    Combines original Phase 3 structured memory candidate with five DMCLM cognitive features:
    Recency (R), Frequency (F), Retrieval History (H), Semantic Relevance (S), and Relationship Strength (Rel).
    Preserves original Phase 3 data and source traceability.
    """
    candidate_id: str
    session_id: str
    source_episode_ids: List[str]
    primary_application: str
    applications_involved: List[str]
    browser_domains: List[str]
    window_context: str
    start_time: str
    end_time: str
    duration_ms: int
    interaction_count: int
    browser_event_count: int
    screenshot_ids: List[str]
    activity_summary: str
    meaningful_criteria: List[str]
    created_at: str

    # Five DMCLM Cognitive Features (normalized strictly to [0.0, 1.0])
    recency: float
    frequency: float
    retrieval_history: float
    semantic_relevance: float
    relationship_strength: float

    # Traceability & evidence metadata
    feature_metadata: Dict[str, Any] = field(default_factory=dict)
    calculation_timestamp: str = ""

    # Aliases for backward compatibility with existing systems
    @property
    def memory_id(self) -> str:
        return self.candidate_id

    @property
    def recency_score(self) -> float:
        return self.recency

    @property
    def frequency_score(self) -> float:
        return self.frequency

    @property
    def retrieval_history_score(self) -> float:
        return self.retrieval_history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "memory_id": self.candidate_id,
            "session_id": self.session_id,
            "source_episode_ids": self.source_episode_ids,
            "primary_application": self.primary_application,
            "applications_involved": self.applications_involved,
            "browser_domains": self.browser_domains,
            "window_context": self.window_context,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "interaction_count": self.interaction_count,
            "browser_event_count": self.browser_event_count,
            "screenshot_ids": self.screenshot_ids,
            "activity_summary": self.activity_summary,
            "meaningful_criteria": self.meaningful_criteria,
            "created_at": self.created_at,
            "recency": self.recency,
            "recency_score": self.recency,
            "frequency": self.frequency,
            "frequency_score": self.frequency,
            "retrieval_history": self.retrieval_history,
            "retrieval_history_score": self.retrieval_history,
            "semantic_relevance": self.semantic_relevance,
            "relationship_strength": self.relationship_strength,
            "feature_metadata": self.feature_metadata,
            "calculation_timestamp": self.calculation_timestamp,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class BaselineScoredMemory:
    """
    Phase 5 — Baseline DMCLM Scored Memory Representation.
    Combines Phase 3 candidates and Phase 4 five features with Phase 5 explainable
    baseline scoring, feature contributions, and value classification.
    Maintains full provenance back to Phase 4 features and Phase 3 candidates.
    """
    candidate_id: str
    session_id: str
    source_episode_ids: List[str]
    primary_application: str
    applications_involved: List[str]
    browser_domains: List[str]
    window_context: str
    start_time: str
    end_time: str
    duration_ms: int
    interaction_count: int
    browser_event_count: int
    screenshot_ids: List[str]
    activity_summary: str
    meaningful_criteria: List[str]
    created_at: str

    # Five DMCLM Features (from Phase 4)
    recency: float
    frequency: float
    retrieval_history: float
    semantic_relevance: float
    relationship_strength: float
    feature_metadata: Dict[str, Any] = field(default_factory=dict)

    # Phase 5 Baseline DMCLM Outputs
    baseline_score: float = 0.0
    baseline_weights: Dict[str, float] = field(default_factory=dict)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    relative_contributions: Dict[str, float] = field(default_factory=dict)
    dominant_feature: str = ""
    value_tier: str = "MEDIUM_VALUE"
    lifecycle_tier: str = "COMPRESS"
    explanation: str = ""
    scoring_timestamp: str = ""

    # Aliases for backward compatibility with existing systems
    @property
    def memory_id(self) -> str:
        return self.candidate_id

    @property
    def importance_score(self) -> float:
        return self.baseline_score

    @property
    def recency_score(self) -> float:
        return self.recency

    @property
    def frequency_score(self) -> float:
        return self.frequency

    @property
    def retrieval_history_score(self) -> float:
        return self.retrieval_history

    @property
    def lifecycle_state(self) -> str:
        return self.lifecycle_tier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "memory_id": self.candidate_id,
            "session_id": self.session_id,
            "source_episode_ids": self.source_episode_ids,
            "primary_application": self.primary_application,
            "applications_involved": self.applications_involved,
            "browser_domains": self.browser_domains,
            "window_context": self.window_context,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "interaction_count": self.interaction_count,
            "browser_event_count": self.browser_event_count,
            "screenshot_ids": self.screenshot_ids,
            "activity_summary": self.activity_summary,
            "meaningful_criteria": self.meaningful_criteria,
            "created_at": self.created_at,
            "recency": self.recency,
            "recency_score": self.recency,
            "frequency": self.frequency,
            "frequency_score": self.frequency,
            "retrieval_history": self.retrieval_history,
            "retrieval_history_score": self.retrieval_history,
            "semantic_relevance": self.semantic_relevance,
            "relationship_strength": self.relationship_strength,
            "feature_metadata": self.feature_metadata,
            "baseline_score": self.baseline_score,
            "importance_score": self.baseline_score,
            "baseline_weights": self.baseline_weights,
            "feature_contributions": self.feature_contributions,
            "relative_contributions": self.relative_contributions,
            "dominant_feature": self.dominant_feature,
            "value_tier": self.value_tier,
            "lifecycle_tier": self.lifecycle_tier,
            "lifecycle_state": self.lifecycle_tier,
            "explanation": self.explanation,
            "scoring_timestamp": self.scoring_timestamp,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class Phase6ConsolidatedMemory:
    """
    Phase 6 — Consolidated Memory Representation.
    Produced by the genuine Consolidation & Compression pipeline from Phase 5 Baseline Memories.
    Maintains complete provenance back to original memory candidates and source episodes.
    """
    consolidated_memory_id: str
    source_memory_ids: List[str]
    source_episode_ids: List[str]
    start_time: str
    end_time: str
    duration_ms: int
    span_duration_ms: int
    primary_application: str
    applications: List[str]
    domains: List[str]
    contexts: List[str]
    interaction_statistics: Dict[str, Any]
    semantic_representation: str
    consolidation_evidence: Dict[str, Any]
    source_count: int
    baseline_information: Dict[str, Any]
    consolidation_metadata: Dict[str, Any]

    # Dashboard & Storage compatibility properties
    @property
    def ltm_id(self) -> str:
        return self.consolidated_memory_id

    @property
    def memory_id(self) -> str:
        return self.consolidated_memory_id

    @property
    def application(self) -> str:
        return self.primary_application

    @property
    def session_id(self) -> str:
        return self.consolidation_metadata.get("primary_session_id", "")

    @property
    def topic(self) -> str:
        ctx = self.contexts[0] if self.contexts else self.primary_application
        return f"{self.primary_application}: {ctx}"

    @property
    def summary(self) -> str:
        return self.semantic_representation

    @property
    def unique_windows_accessed(self) -> List[str]:
        return self.contexts

    @property
    def first_observed(self) -> str:
        return self.start_time

    @property
    def last_reinforced(self) -> str:
        return self.end_time

    @property
    def total_duration_ms(self) -> int:
        return self.duration_ms

    @property
    def total_interactions(self) -> int:
        return self.interaction_statistics.get("total_interactions", 0)

    @property
    def stability_score(self) -> float:
        return self.baseline_information.get("stability_score", 0.5)

    @property
    def lifecycle_origin(self) -> str:
        return "COMPRESS"

    @property
    def created_at(self) -> str:
        return self.consolidation_metadata.get("created_at", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consolidated_memory_id": self.consolidated_memory_id,
            "ltm_id": self.consolidated_memory_id,
            "memory_id": self.consolidated_memory_id,
            "source_memory_ids": self.source_memory_ids,
            "source_episode_ids": self.source_episode_ids,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "span_duration_ms": self.span_duration_ms,
            "primary_application": self.primary_application,
            "application": self.primary_application,
            "applications": self.applications,
            "domains": self.domains,
            "contexts": self.contexts,
            "unique_windows_accessed": self.contexts,
            "interaction_statistics": self.interaction_statistics,
            "semantic_representation": self.semantic_representation,
            "summary": self.semantic_representation,
            "topic": self.topic,
            "consolidation_evidence": self.consolidation_evidence,
            "source_count": self.source_count,
            "baseline_information": self.baseline_information,
            "consolidation_metadata": self.consolidation_metadata,
            "session_id": self.session_id,
            "first_observed": self.first_observed,
            "last_reinforced": self.last_reinforced,
            "total_duration_ms": self.total_duration_ms,
            "total_interactions": self.total_interactions,
            "stability_score": self.stability_score,
            "lifecycle_origin": self.lifecycle_origin,
            "created_at": self.created_at,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class Phase7RetainedMemory:
    """
    Phase 7 — Adaptive Retention and Forgetting Representation.
    Produced by applying temporal decay and adaptive cognitive retention to Phase 6 Consolidated Memories.
    Maintains complete provenance back to Phase 6 consolidated nodes, Phase 3 candidates, and Phase 2 episodes.
    """
    consolidated_memory_id: str
    source_memory_ids: List[str]
    source_episode_ids: List[str]
    primary_application: str
    baseline_score: float
    recency: float
    frequency: float
    retrieval_history: float
    semantic_relevance: float
    relationship_strength: float
    stability_score: float
    source_count: int
    age_days: float
    decay_factor: float
    effective_decay_rate: float
    adaptive_resilience: float
    adaptive_retention_factor: float
    final_retention_strength: float
    retention_state: str                    # 'RETAINED', 'DECAYING', 'ARCHIVED', 'FORGETTABLE'
    is_protected: bool
    forgetting_eligible: bool
    protection_reason: Optional[str]
    retention_explanation: str
    evaluation_timestamp: str

    # Context / provenance fields carried forward
    topic: str = ""
    summary: str = ""
    start_time: str = ""
    end_time: str = ""
    total_duration_ms: int = 0
    total_interactions: int = 0
    unique_windows_accessed: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)

    # Dashboard & Storage compatibility properties
    @property
    def memory_id(self) -> str:
        return self.consolidated_memory_id

    @property
    def ltm_id(self) -> str:
        return self.consolidated_memory_id

    @property
    def importance_score(self) -> float:
        return self.final_retention_strength

    @property
    def retention_score(self) -> float:
        return self.final_retention_strength

    @property
    def lifecycle_state(self) -> str:
        return self.retention_state

    @property
    def application(self) -> str:
        return self.primary_application

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consolidated_memory_id": self.consolidated_memory_id,
            "memory_id": self.consolidated_memory_id,
            "ltm_id": self.consolidated_memory_id,
            "source_memory_ids": self.source_memory_ids,
            "source_episode_ids": self.source_episode_ids,
            "primary_application": self.primary_application,
            "application": self.primary_application,
            "baseline_score": self.baseline_score,
            "recency": self.recency,
            "frequency": self.frequency,
            "retrieval_history": self.retrieval_history,
            "semantic_relevance": self.semantic_relevance,
            "relationship_strength": self.relationship_strength,
            "stability_score": self.stability_score,
            "source_count": self.source_count,
            "age_days": self.age_days,
            "decay_factor": self.decay_factor,
            "effective_decay_rate": self.effective_decay_rate,
            "adaptive_resilience": self.adaptive_resilience,
            "adaptive_retention_factor": self.adaptive_retention_factor,
            "final_retention_strength": self.final_retention_strength,
            "importance_score": self.final_retention_strength,
            "retention_score": self.final_retention_strength,
            "retention_state": self.retention_state,
            "lifecycle_state": self.retention_state,
            "is_protected": self.is_protected,
            "forgetting_eligible": self.forgetting_eligible,
            "protection_reason": self.protection_reason,
            "retention_explanation": self.retention_explanation,
            "evaluation_timestamp": self.evaluation_timestamp,
            "topic": self.topic,
            "summary": self.summary,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "total_interactions": self.total_interactions,
            "unique_windows_accessed": self.unique_windows_accessed,
            "domains": self.domains,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class MemoryRelationship:
    """
    Phase 8 — Memory Relationship Representation.
    Represents an evidence-backed relationship between two consolidated/retained memory nodes.
    Maintains deterministic identity, normalized strength, comprehensive evidence breakdown,
    and complete provenance back to the constituent memories.
    """
    relationship_id: str
    source_memory_id: str
    target_memory_id: str
    relationship_type: str
    relationship_strength: float
    evidence: Dict[str, Any]
    created_at: str
    source_memory_source_count: int = 1
    target_memory_source_count: int = 1
    source_episode_ids_count: int = 0
    target_episode_ids_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_memory_id": self.source_memory_id,
            "target_memory_id": self.target_memory_id,
            "relationship_type": self.relationship_type,
            "relationship_strength": self.relationship_strength,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "source_memory_source_count": self.source_memory_source_count,
            "target_memory_source_count": self.target_memory_source_count,
            "source_episode_ids_count": self.source_episode_ids_count,
            "target_episode_ids_count": self.target_episode_ids_count,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class RelationshipGraphStats:
    """
    Graph-level metrics and topology summary for Phase 8 relationship graph.
    """
    total_nodes: int
    total_edges: int
    density: float
    average_degree: float
    average_strength: float
    isolated_nodes_count: int
    connected_components_count: int
    type_distribution: Dict[str, int]
    top_relationships: List[Dict[str, Any]]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "density": self.density,
            "average_degree": self.average_degree,
            "average_strength": self.average_strength,
            "isolated_nodes_count": self.isolated_nodes_count,
            "connected_components_count": self.connected_components_count,
            "type_distribution": self.type_distribution,
            "top_relationships": self.top_relationships,
            "created_at": self.created_at,
        }


# ==============================================================================
# PHASE 9: TRAINING DATASET GENERATION MODELS
# ==============================================================================

@dataclass
class FeatureSchemaField:
    """
    Defines metadata and leakage-protection attributes for each column in the training dataset.
    """
    name: str
    role: str  # "METADATA", "ML_INPUT", "TARGET", "AUDIT_REFERENCE"
    dtype: str  # "float", "int", "str", "bool"
    source_phase: str  # "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6", "Phase 7", "Phase 8", "Phase 9"
    description: str
    is_leakage_risk: bool = False
    imputation_strategy: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "source_phase": self.source_phase,
            "description": self.description,
            "is_leakage_risk": self.is_leakage_risk,
            "imputation_strategy": self.imputation_strategy,
        }


@dataclass
class DatasetManifest:
    """
    Audit manifest for generated Phase 9 training, validation, and test datasets.
    """
    dataset_version: str
    created_at: str
    split_strategy: str
    total_source_candidates: int
    total_valid_examples: int
    rejected_records_count: int
    rejection_reasons: Dict[str, int]
    total_features_count: int
    ml_input_features_count: int
    ml_input_features: List[str]
    target_variables: List[str]
    primary_categorical_target: str
    primary_continuous_target: str
    train_count: int
    val_count: int
    test_count: int
    target_distribution: Dict[str, Any]
    split_distribution: Dict[str, Dict[str, int]]
    data_hash_sha256: str
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "split_strategy": self.split_strategy,
            "total_source_candidates": self.total_source_candidates,
            "total_valid_examples": self.total_valid_examples,
            "rejected_records_count": self.rejected_records_count,
            "rejection_reasons": self.rejection_reasons,
            "total_features_count": self.total_features_count,
            "ml_input_features_count": self.ml_input_features_count,
            "ml_input_features": self.ml_input_features,
            "target_variables": self.target_variables,
            "primary_categorical_target": self.primary_categorical_target,
            "primary_continuous_target": self.primary_continuous_target,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "target_distribution": self.target_distribution,
            "split_distribution": self.split_distribution,
            "data_hash_sha256": self.data_hash_sha256,
            "limitations": self.limitations,
        }


# ==============================================================================
# PHASE 10: ML MODEL TRAINING & PREDICTION MODELS
# ==============================================================================

@dataclass
class MLPredictionResult:
    """
    Structured outcome of ML model inference on an episodic memory record.
    """
    memory_id: str
    predicted_lifecycle_state: str  # "COMPRESS" or "ARCHIVE"
    prediction_confidence: float  # Maximum class probability in [0.0, 1.0]
    class_probabilities: Dict[str, float]  # Probability distribution over classes
    predicted_importance_score: Optional[float] = None  # Regressor output in [0.0, 1.0]
    dominant_contributing_feature: Optional[str] = None
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    model_version: str = "v1.0"
    algorithm: str = "RandomForestClassifier"
    is_ml_generated: bool = True

    @property
    def confidence(self) -> float:
        """Alias for prediction_confidence."""
        return self.prediction_confidence

    @property
    def probabilities(self) -> Dict[str, float]:
        """Alias for class_probabilities."""
        return self.class_probabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "predicted_lifecycle_state": self.predicted_lifecycle_state,
            "prediction_confidence": self.prediction_confidence,
            "class_probabilities": self.class_probabilities,
            "predicted_importance_score": self.predicted_importance_score,
            "dominant_contributing_feature": self.dominant_contributing_feature,
            "feature_contributions": self.feature_contributions,
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "is_ml_generated": self.is_ml_generated,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class ModelTrainingMetadata:
    """
    Audit and provenance record for a trained DMCLM machine learning model.
    """
    model_version: str
    training_timestamp: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    target_column: str
    dataset_hash: str
    train_samples: int
    val_samples: int
    test_samples: int
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    leakage_protected: bool = True
    model_artifact_path: str = "models/dmclm_lifecycle_model.joblib"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "training_timestamp": self.training_timestamp,
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "target_column": self.target_column,
            "dataset_hash": self.dataset_hash,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "feature_importance": self.feature_importance,
            "leakage_protected": self.leakage_protected,
            "model_artifact_path": self.model_artifact_path,
        }


@dataclass
class PerMemoryComparisonRecord:
    """
    Aligned evaluation unit comparing Baseline DMCLM against trained ML Model
    for a single real memory record with 1-to-1 provenance traceability.
    """
    memory_id: str
    feature_recency: float
    feature_frequency: float
    feature_retrieval_history: float
    feature_semantic_relevance: float
    feature_relationship_strength: float
    target_lifecycle_state: str
    target_importance_score: float
    baseline_lifecycle_tier: str
    baseline_score: float
    baseline_is_correct: bool
    baseline_error: float
    ml_predicted_state: str
    ml_predicted_score: Optional[float]
    ml_confidence: float
    ml_is_correct: bool
    ml_error: Optional[float]
    is_agreement: bool
    score_delta: Optional[float]
    outcome_category: str  # BOTH_CORRECT, BASELINE_ONLY_CORRECT, ML_ONLY_CORRECT, BOTH_INCORRECT
    ml_dominant_feature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "feature_recency": self.feature_recency,
            "feature_frequency": self.feature_frequency,
            "feature_retrieval_history": self.feature_retrieval_history,
            "feature_semantic_relevance": self.feature_semantic_relevance,
            "feature_relationship_strength": self.feature_relationship_strength,
            "target_lifecycle_state": self.target_lifecycle_state,
            "target_importance_score": self.target_importance_score,
            "baseline_lifecycle_tier": self.baseline_lifecycle_tier,
            "baseline_score": self.baseline_score,
            "baseline_is_correct": self.baseline_is_correct,
            "baseline_error": self.baseline_error,
            "ml_predicted_state": self.ml_predicted_state,
            "ml_predicted_score": self.ml_predicted_score,
            "ml_confidence": self.ml_confidence,
            "ml_is_correct": self.ml_is_correct,
            "ml_error": self.ml_error,
            "is_agreement": self.is_agreement,
            "score_delta": self.score_delta,
            "outcome_category": self.outcome_category,
            "ml_dominant_feature": self.ml_dominant_feature,
        }

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        d = self.to_dict()
        if item in d:
            return d[item]
        raise KeyError(item)

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class BaselineMLComparisonReport:
    """
    Comprehensive scientific comparison report between Baseline DMCLM and ML Model.
    """
    comparison_id: str
    timestamp: str
    evaluation_split: str
    dataset_hash: str
    sample_count: int
    baseline_version: str
    ml_version: str
    ml_algorithm: str
    features_evaluated: List[str]
    target_classification: str
    target_regression: str
    baseline_classification_metrics: Dict[str, Any]
    ml_classification_metrics: Dict[str, Any]
    classification_differences: Dict[str, Any]
    baseline_regression_metrics: Dict[str, Any]
    ml_regression_metrics: Dict[str, Any]
    regression_differences: Dict[str, Any]
    agreement_analysis: Dict[str, Any]
    feature_agreement_divergence: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    overall_conclusion: str
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "timestamp": self.timestamp,
            "evaluation_split": self.evaluation_split,
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "baseline_version": self.baseline_version,
            "ml_version": self.ml_version,
            "ml_algorithm": self.ml_algorithm,
            "features_evaluated": self.features_evaluated,
            "target_classification": self.target_classification,
            "target_regression": self.target_regression,
            "baseline_classification_metrics": self.baseline_classification_metrics,
            "ml_classification_metrics": self.ml_classification_metrics,
            "classification_differences": self.classification_differences,
            "baseline_regression_metrics": self.baseline_regression_metrics,
            "ml_regression_metrics": self.ml_regression_metrics,
            "regression_differences": self.regression_differences,
            "agreement_analysis": self.agreement_analysis,
            "feature_agreement_divergence": self.feature_agreement_divergence,
            "statistical_tests": self.statistical_tests,
            "overall_conclusion": self.overall_conclusion,
            "limitations": self.limitations,
        }


@dataclass
class ExperimentResult:
    """
    Standardized result entity for an individual research evaluation experiment.
    """
    experiment_id: str
    experiment_name: str
    experiment_type: str
    timestamp: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    sample_count: int
    execution_time_seconds: float
    status: str = "SUCCESS"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "sample_count": self.sample_count,
            "execution_time_seconds": self.execution_time_seconds,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class ExperimentSuiteReport:
    """
    Centralized experimental evaluation suite report for Phase 12.
    """
    suite_id: str
    timestamp: str
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    dataset_hash: str
    results: Dict[str, Any]
    executive_summary: str
    figures_generated: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "timestamp": self.timestamp,
            "total_experiments": self.total_experiments,
            "successful_experiments": self.successful_experiments,
            "failed_experiments": self.failed_experiments,
            "dataset_hash": self.dataset_hash,
            "results": self.results,
            "executive_summary": self.executive_summary,
            "figures_generated": self.figures_generated,
        }


# =============================================================================
# PHASE 13: RESOURCE-AWARE REASONING DATA MODELS
# =============================================================================

@dataclass
class ResourceSnapshot:
    """
    Point-in-time technical system performance measurement.
    Strictly local, non-sensitive system indicators only.
    """
    timestamp: str
    cpu_percent: float
    cpu_count: int
    memory_total_mb: float
    memory_available_mb: float
    memory_percent: float
    process_rss_mb: float
    process_vms_mb: float
    process_cpu_percent: float
    stage: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 2),
            "cpu_count": self.cpu_count,
            "memory_total_mb": round(self.memory_total_mb, 2),
            "memory_available_mb": round(self.memory_available_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "process_rss_mb": round(self.process_rss_mb, 2),
            "process_vms_mb": round(self.process_vms_mb, 2),
            "process_cpu_percent": round(self.process_cpu_percent, 2),
            "stage": self.stage,
        }


@dataclass
class ResourceProfileDecision:
    """
    Dynamic resource availability profile decision derived from real measurements.
    """
    profile: str  # "LOW", "NORMAL", "HIGH"
    timestamp: str
    reasons: List[str]
    measurements: Dict[str, Any]
    batch_size: int
    candidate_pair_budget: int
    allow_deferral: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "timestamp": self.timestamp,
            "reasons": self.reasons,
            "measurements": self.measurements,
            "batch_size": self.batch_size,
            "candidate_pair_budget": self.candidate_pair_budget,
            "allow_deferral": self.allow_deferral,
        }


@dataclass
class DeferredMemoryRecord:
    """
    Record of a memory whose non-critical processing was deferred under resource pressure.
    """
    queue_id: str
    memory_id: str
    priority_score: float
    deferred_reason: str
    queued_at: str
    processing_stage: str
    status: str = "PENDING"  # "PENDING", "PROCESSED", "EXPIRED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "memory_id": self.memory_id,
            "priority_score": round(self.priority_score, 4),
            "deferred_reason": self.deferred_reason,
            "queued_at": self.queued_at,
            "processing_stage": self.processing_stage,
            "status": self.status,
        }


@dataclass
class ResourceAwareExecutionResult:
    """
    Execution report for a resource-aware memory processing cycle.
    """
    execution_id: str
    timestamp: str
    profile_used: str
    stage: str
    duration_seconds: float
    initial_snapshot: Dict[str, Any]
    peak_snapshot: Dict[str, Any]
    final_snapshot: Dict[str, Any]
    total_candidates: int
    processed_count: int
    deferred_count: int
    cache_hits: int
    cache_misses: int
    quality_metrics: Dict[str, Any]
    status: str = "SUCCESS"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "profile_used": self.profile_used,
            "stage": self.stage,
            "duration_seconds": round(self.duration_seconds, 4),
            "initial_snapshot": self.initial_snapshot,
            "peak_snapshot": self.peak_snapshot,
            "final_snapshot": self.final_snapshot,
            "total_candidates": self.total_candidates,
            "processed_count": self.processed_count,
            "deferred_count": self.deferred_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "quality_metrics": self.quality_metrics,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class ResourceComparisonReport:
    """
    Comparative evaluation between standard unconstrained vs resource-aware processing.
    """
    report_id: str
    timestamp: str
    standard_run: Dict[str, Any]
    resource_aware_run: Dict[str, Any]
    execution_time_delta_pct: float
    memory_rss_delta_pct: float
    throughput_gain_pct: float
    quality_retention_pct: float
    figures_generated: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "standard_run": self.standard_run,
            "resource_aware_run": self.resource_aware_run,
            "execution_time_delta_pct": round(self.execution_time_delta_pct, 2),
            "memory_rss_delta_pct": round(self.memory_rss_delta_pct, 2),
            "throughput_gain_pct": round(self.throughput_gain_pct, 2),
            "quality_retention_pct": round(self.quality_retention_pct, 2),
            "figures_generated": self.figures_generated,
            "summary": self.summary,
        }










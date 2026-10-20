export interface MemoryRecord {
  memory_id: string;
  source_type: 'activity' | 'browser' | 'interaction' | 'screenshot';
  timestamp: string;
  session_id: string;
  application: string;
  content: string;
  duration_ms: number;
  domain: string | null;
  interaction_count: number;
  recency: number;
  frequency: number;
  retrieval_history: number;
  semantic_relevance: number;
  relationship_strength: number;
  importance_score?: number;
  lifecycle_state?: string;
  lifecycle_prediction?: 'RETAIN' | 'COMPRESS' | 'ARCHIVE' | 'FORGET';
  lifecycle_confidence?: number;
  lifecycle_explanation?: string;
  occurrence_count?: number;
  screenshot_count?: number;
  first_seen?: string;
  last_seen?: string;
}

export interface RelatedMemorySummary {
  memory_id: string;
  application: string;
  source_type: string;
  content: string;
  timestamp: string;
  lifecycle?: string;
}

export interface MemoryDetailResponse {
  memory: MemoryRecord;
  related_memories: RelatedMemorySummary[];
  lifecycle_state: {
    status: string;
    confidence?: number;
    reason: string;
  };
}

export interface LongTermMemoryNode {
  ltm_id: string;
  topic: string;
  application: string;
  session_id: string;
  summary: string;
  unique_windows_accessed: string[];
  source_count: number;
  source_memory_ids: string[];
  first_observed: string;
  last_reinforced: string;
  total_duration_ms: number;
  total_interactions: number;
  stability_score: number;
  lifecycle_origin: string;
  created_at: string;
}

export interface AdaptiveRetrievalResponse {
  memory_id: string;
  retrieval_count: number;
  previous_lifecycle: string;
  new_lifecycle: string;
  state_transitioned: boolean;
  updated_features: {
    recency: number;
    retrieval_history: number;
  };
  confidence: number;
  explanation: string;
}

export interface ConsolidationStats {
  total_episodic_records: number;
  engine_status: string;
  features: {
    recency: { status: string; description: string };
    frequency: { status: string; description: string };
    retrieval_history: { status: string; description: string };
    semantic_relevance: { status: string; description: string };
    relationship_strength: { status: string; description: string };
  };
  lifecycle_states: {
    retain: { count: number; percentage: number; status: string; description: string };
    compress: { count: number; percentage: number; status: string; description: string };
    archive: { count: number; percentage: number; status: string; description: string };
    forget: { count: number; percentage: number; status: string; description: string };
  };
  compression_summary?: {
    original_episodic_records: number;
    consolidated_ltm_nodes: number;
    memory_reduction_count: number;
    compression_ratio_pct: number;
    unconsolidated_memories?: number;
    total_retained_nodes?: number;
    average_group_size?: number;
    group_compression_ratio_pct?: number;
  };
  research_metrics: {
    memory_reduction: string;
    compression_ratio: string;
    retrieval_utility: string;
    recall_at_k: string;
    ndcg: string;
    f1_score: string;
    auc: string;
    response_time: string;
  };
}

export interface ModelMetricsResponse {
  evaluation_dataset_size: number;
  classes: string[];
  cross_validation_5_fold: {
    mean_accuracy: number;
    std_accuracy: number;
    mean_weighted_f1: number;
    fold_accuracies: number[];
  };
  model_performance: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_f1: number;
    roc_auc_score: number;
  };
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  per_class_metrics: Record<
    string,
    {
      precision: number;
      recall: number;
      f1_score: number;
      support: number;
    }
  >;
  feature_importance: Array<{ name: string; importance: number }>;
  model_status?: string;
  training_dataset_ready?: boolean;
  note?: string;
}

export interface AnalyticsData {
  top_applications: Array<{ name: string; count: number }>;
  top_domains: Array<{ domain: string; count: number }>;
  interaction_types: Array<{ type: string; count: number }>;
  lifecycle_distribution?: Array<{ name: string; value: number; color: string }>;
  feature_distributions: {
    status: string;
    means?: {
      recency: number;
      frequency: number;
      retrieval_history: number;
      semantic_relevance: number;
      relationship_strength: number;
    };
  };
}

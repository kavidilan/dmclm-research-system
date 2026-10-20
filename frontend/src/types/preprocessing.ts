export interface ActivityEpisode {
  episode_id: string;
  session_id: string;
  application: string;
  process_name: string;
  window_title: string;
  start_time: string;
  end_time: string;
  duration_ms: number;
  event_count: number;
  source_event_ids: string[];
  interaction_count: number;
  screenshot_ids: string[];
  created_at: string;
}

export interface ScreenshotDuplicate {
  sha256: string;
  canonical_screenshot_id: string;
  is_duplicate: number;
  total_count: number;
  duplicate_screenshot_ids: string[];
  application_name: string;
  window_title: string;
  file_size: number;
  local_file_path: string;
  created_at: string;
}

export interface ScreenshotDuplicatesResponse {
  total_screenshots: number;
  unique_screenshots: number;
  duplicate_screenshots: number;
  items: ScreenshotDuplicate[];
  detection_method: string;
}

export interface PreprocessingRawCounts {
  activity_events: number;
  browser_events: number;
  interaction_events: number;
  screenshots: number;
  sessions: number;
  total: number;
}

export interface PreprocessingStatistics {
  raw_event_counts: PreprocessingRawCounts;
  normalized_event_counts: number;
  duplicate_screenshot_count: number;
  unique_screenshot_count: number;
  total_screenshots: number;
  aggregated_interaction_count: number;
  interaction_clusters_count: number;
  browser_normalization_count: number;
  activity_episode_count: number;
  reduction_ratio_pct: number;
  processing_duration_sec: number;
  errors_count: number;
  preprocessing_timestamp: string;
  run_id?: string;
  status?: string;
}

export interface PreprocessingSummary {
  status: string;
  last_run: string | null;
  run_id: string;
  duration_sec: number;
  raw_counts: PreprocessingRawCounts;
  statistics: PreprocessingStatistics;
  database_integrity: {
    collector_db_path: string;
    collector_db_mode: string;
    collector_db_unmodified: boolean;
    collector_db_sha256: string;
  };
}

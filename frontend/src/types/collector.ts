export interface ActivityEvent {
  event_id: string;
  session_id: string;
  timestamp: string;
  event_type: string;
  application_name: string;
  process_name: string;
  window_title: string;
  duration_ms: number;
}

export interface BrowserEvent {
  browser_event_id: string;
  session_id: string;
  timestamp: string;
  browser_name: string;
  domain: string;
  page_title: string;
  url: string;
}

export interface InteractionEvent {
  interaction_id: string;
  session_id: string;
  timestamp: string;
  interaction_type: string;
  application_name: string;
  count: number;
}

export interface ScreenshotEvent {
  screenshot_id: string;
  session_id: string;
  timestamp: string;
  application_name: string;
  window_title: string;
  capture_reason: string;
  file_size: number;
}

export interface SessionEvent {
  session_id: string;
  started_at: string;
  ended_at: string | null;
  end_reason: string | null;
}

export interface DashboardStats {
  activity_events: number;
  browser_events: number;
  interaction_events: number;
  screenshots: number;
  sessions: number;
  memory_records: number;
}

export interface SystemStatus {
  status: 'connected' | 'disconnected';
  database_path: string;
  database_size_mb: number;
  last_sync: string | null;
  mode: string;
  table_counts: Record<string, number>;
  memory_records_count: number;
  privacy: {
    status: string;
    source: string;
    storage: string;
    network_upload: string;
    sanitization_status: string;
  };
  pipeline: {
    collector: string;
    memory_extraction: string;
    feature_calculation: string;
    semantic_analysis: string;
    dmclm_decision_engine: string;
    consolidation: string;
  };
}

export interface TimelineDataPoint {
  time: string;
  full_time: string;
  activity: number;
  browser: number;
  interactions: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

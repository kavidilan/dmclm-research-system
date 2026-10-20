import {
  ActivityEvent,
  BrowserEvent,
  DashboardStats,
  InteractionEvent,
  PaginatedResponse,
  ScreenshotEvent,
  SessionEvent,
  SystemStatus,
  TimelineDataPoint
} from '../types/collector';
import {
  AdaptiveRetrievalResponse,
  AnalyticsData,
  ConsolidationStats,
  LongTermMemoryNode,
  MemoryDetailResponse,
  MemoryRecord,
  ModelMetricsResponse
} from '../types/memory';
import {
  ActivityEpisode,
  PreprocessingSummary,
  ScreenshotDuplicatesResponse
} from '../types/preprocessing';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`API Error ${response.status}: ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  getSystemStatus: (): Promise<SystemStatus> =>
    fetchJson<SystemStatus>(`${API_BASE}/status`),

  getDashboardStats: (): Promise<DashboardStats> =>
    fetchJson<DashboardStats>(`${API_BASE}/dashboard/stats`),

  getTimelineActivity: (): Promise<TimelineDataPoint[]> =>
    fetchJson<TimelineDataPoint[]>(`${API_BASE}/dashboard/timeline`),

  getActivityEvents: (
    page = 1,
    limit = 25,
    search?: string,
    appFilter?: string
  ): Promise<PaginatedResponse<ActivityEvent>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (search) params.append('search', search);
    if (appFilter) params.append('app_filter', appFilter);
    return fetchJson<PaginatedResponse<ActivityEvent>>(`${API_BASE}/raw-data/activity-events?${params}`);
  },

  getBrowserEvents: (
    page = 1,
    limit = 25,
    search?: string,
    domainFilter?: string
  ): Promise<PaginatedResponse<BrowserEvent>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (search) params.append('search', search);
    if (domainFilter) params.append('domain_filter', domainFilter);
    return fetchJson<PaginatedResponse<BrowserEvent>>(`${API_BASE}/raw-data/browser-events?${params}`);
  },

  getInteractionEvents: (
    page = 1,
    limit = 25,
    search?: string
  ): Promise<PaginatedResponse<InteractionEvent>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (search) params.append('search', search);
    return fetchJson<PaginatedResponse<InteractionEvent>>(`${API_BASE}/raw-data/interaction-events?${params}`);
  },

  getScreenshots: (
    page = 1,
    limit = 25
  ): Promise<PaginatedResponse<ScreenshotEvent>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    return fetchJson<PaginatedResponse<ScreenshotEvent>>(`${API_BASE}/raw-data/screenshots?${params}`);
  },

  getSessions: (
    page = 1,
    limit = 25
  ): Promise<PaginatedResponse<SessionEvent>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    return fetchJson<PaginatedResponse<SessionEvent>>(`${API_BASE}/raw-data/sessions?${params}`);
  },

  getMemoryRecords: (
    page = 1,
    limit = 20,
    sourceType?: string,
    search?: string,
    appFilter?: string,
    lifecycleFilter?: string
  ): Promise<PaginatedResponse<MemoryRecord>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (sourceType) params.append('source_type', sourceType);
    if (search) params.append('search', search);
    if (appFilter) params.append('app_filter', appFilter);
    if (lifecycleFilter) params.append('lifecycle_filter', lifecycleFilter);
    return fetchJson<PaginatedResponse<MemoryRecord>>(`${API_BASE}/memories?${params}`);
  },

  getMemoryById: (memoryId: string): Promise<MemoryDetailResponse> =>
    fetchJson<MemoryDetailResponse>(`${API_BASE}/memories/${memoryId}`),

  retrieveMemory: (memoryId: string): Promise<AdaptiveRetrievalResponse> =>
    fetchJson<AdaptiveRetrievalResponse>(`${API_BASE}/memories/${memoryId}/retrieve`, {
      method: 'POST'
    }),

  getConsolidationStats: (): Promise<ConsolidationStats> =>
    fetchJson<ConsolidationStats>(`${API_BASE}/consolidation/stats`),

  getLongTermMemories: (): Promise<LongTermMemoryNode[]> =>
    fetchJson<LongTermMemoryNode[]>(`${API_BASE}/consolidation/long-term-memories`),

  getModelMetrics: (): Promise<ModelMetricsResponse> =>
    fetchJson<ModelMetricsResponse>(`${API_BASE}/model/metrics`),

  getAnalytics: (): Promise<AnalyticsData> =>
    fetchJson<AnalyticsData>(`${API_BASE}/analytics`),

  // Phase 2 Preprocessing Endpoints
  getPreprocessingSummary: (): Promise<PreprocessingSummary> =>
    fetchJson<PreprocessingSummary>(`${API_BASE}/preprocessing/summary`),

  getPreprocessingEpisodes: (
    page = 1,
    limit = 25,
    search?: string,
    appFilter?: string
  ): Promise<PaginatedResponse<ActivityEpisode>> => {
    const params = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (search) params.append('search', search);
    if (appFilter) params.append('app_filter', appFilter);
    return fetchJson<PaginatedResponse<ActivityEpisode>>(`${API_BASE}/preprocessing/episodes?${params}`);
  },

  getPreprocessingDuplicates: (): Promise<ScreenshotDuplicatesResponse> =>
    fetchJson<ScreenshotDuplicatesResponse>(`${API_BASE}/preprocessing/duplicates`),

  runPreprocessing: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/preprocessing/run`, { method: 'POST' }),

  // Relationships
  getRelationshipsStats: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/relationships/stats`),

  getRelationships: (limit = 20, offset = 0): Promise<any> => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    return fetchJson<any>(`${API_BASE}/relationships?${params}`);
  },

  getRelationshipGraph: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/relationships/graph`),

  // Retention
  getRetentionStats: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/retention/stats`),

  // Model
  getModelStatus: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/model/status`),

  getModelFeatures: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/model/features`),

  // Dataset
  getDatasetManifest: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/dataset/manifest`),

  getDatasetSchema: (): Promise<any> =>
    fetchJson<any>(`${API_BASE}/dataset/schema`),
};

import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { DashboardStats, SystemStatus, TimelineDataPoint } from '../types/collector';

export function useCollectorData() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [timeline, setTimeline] = useState<TimelineDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsData, statusData, timelineData] = await Promise.all([
        api.getDashboardStats().catch(() => null),
        api.getSystemStatus().catch(() => null),
        api.getTimelineActivity().catch(() => [])
      ]);

      if (!statsData && !statusData) {
        throw new Error('Unable to connect to DMCLM API backend. Ensure python backend server is running on port 8000.');
      }

      setStats(statsData);
      setStatus(statusData);
      setTimeline(timelineData || []);
      setLastRefreshed(new Date());
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch collector data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    stats,
    status,
    timeline,
    loading,
    error,
    lastRefreshed,
    refetch: fetchData
  };
}

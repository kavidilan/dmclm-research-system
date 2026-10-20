import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { MemoryRecord } from '../types/memory';
import { PaginatedResponse } from '../types/collector';

export function useMemoryData(
  page: number = 1,
  limit: number = 20,
  sourceType?: string,
  search?: string,
  appFilter?: string,
  lifecycleFilter?: string
) {
  const [data, setData] = useState<PaginatedResponse<MemoryRecord>>({
    items: [],
    total: 0,
    page: 1,
    limit: 20,
    pages: 0
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMemories = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getMemoryRecords(page, limit, sourceType, search, appFilter, lifecycleFilter);
      setData(res);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch memory records');
    } finally {
      setLoading(false);
    }
  }, [page, limit, sourceType, search, appFilter, lifecycleFilter]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  return {
    memories: data.items,
    total: data.total,
    page: data.page,
    pages: data.pages,
    loading,
    error,
    refetch: fetchMemories
  };
}

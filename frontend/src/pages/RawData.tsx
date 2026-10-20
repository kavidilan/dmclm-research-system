import React, { useState, useEffect, useCallback } from 'react';
import { Database, Globe, MousePointer, Camera, Layers } from 'lucide-react';
import { DataFilters } from '../components/data/DataFilters';
import { DataTable } from '../components/data/DataTable';
import { api } from '../services/api';
import {
  ActivityEvent,
  BrowserEvent,
  InteractionEvent,
  ScreenshotEvent,
  SessionEvent
} from '../types/collector';

export const RawData: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'activity' | 'browser' | 'interaction' | 'screenshot' | 'session'>('activity');
  const [search, setSearch] = useState<string>('');
  const [appFilter, setAppFilter] = useState<string>('');
  const [page, setPage] = useState<number>(1);
  const [limit] = useState<number>(25);

  const [loading, setLoading] = useState<boolean>(true);
  const [data, setData] = useState<any[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [totalPages, setTotalPages] = useState<number>(1);

  const fetchTableData = useCallback(async () => {
    setLoading(true);
    try {
      if (activeTab === 'activity') {
        const res = await api.getActivityEvents(page, limit, search, appFilter);
        setData(res.items);
        setTotal(res.total);
        setTotalPages(res.pages);
      } else if (activeTab === 'browser') {
        const res = await api.getBrowserEvents(page, limit, search);
        setData(res.items);
        setTotal(res.total);
        setTotalPages(res.pages);
      } else if (activeTab === 'interaction') {
        const res = await api.getInteractionEvents(page, limit, search);
        setData(res.items);
        setTotal(res.total);
        setTotalPages(res.pages);
      } else if (activeTab === 'screenshot') {
        const res = await api.getScreenshots(page, limit);
        setData(res.items);
        setTotal(res.total);
        setTotalPages(res.pages);
      } else if (activeTab === 'session') {
        const res = await api.getSessions(page, limit);
        setData(res.items);
        setTotal(res.total);
        setTotalPages(res.pages);
      }
    } catch {
      setData([]);
      setTotal(0);
      setTotalPages(1);
    } finally {
      setLoading(false);
    }
  }, [activeTab, page, limit, search, appFilter]);

  useEffect(() => {
    fetchTableData();
  }, [fetchTableData]);

  // Reset page when tab or search changes
  const handleTabChange = (tab: any) => {
    setActiveTab(tab);
    setPage(1);
  };

  const handleSearchChange = (val: string) => {
    setSearch(val);
    setPage(1);
  };

  const formatTimestamp = (ts: string) => {
    if (!ts) return '—';
    try {
      const d = new Date(ts);
      return d.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return ts;
    }
  };

  const tabs = [
    { id: 'activity', label: 'Activity Events', icon: Database },
    { id: 'browser', label: 'Browser Events', icon: Globe },
    { id: 'interaction', label: 'Interactions', icon: MousePointer },
    { id: 'screenshot', label: 'Screenshots', icon: Camera },
    { id: 'session', label: 'Sessions', icon: Layers }
  ];

  return (
    <div className="space-y-5">
      {/* Page Header */}
      <div>
        <div className="section-eyebrow mb-1 flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5" />
          Raw SQLite Ingestion Layer
        </div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Desktop Research Collector Store</h1>
      </div>

      {/* Table Navigation Tabs */}
      <div className="tab-bar overflow-x-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`tab-item ${isActive ? 'active' : ''}`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Filter Controls */}
      <DataFilters
        search={search}
        onSearchChange={handleSearchChange}
        appFilter={activeTab === 'activity' ? appFilter : undefined}
        onAppFilterChange={activeTab === 'activity' ? setAppFilter : undefined}
        onRefresh={fetchTableData}
        isRefreshing={loading}
        totalRecords={total}
      />

      {/* Tabular View */}
      {activeTab === 'activity' && (
        <DataTable<ActivityEvent>
          columns={[
            {
              key: 'timestamp',
              header: 'Timestamp',
              render: (i) => <span className="font-mono text-[11px] text-slate-500">{formatTimestamp(i.timestamp)}</span>,
              width: 'w-44'
            },
            {
              key: 'application_name',
              header: 'Application',
              render: (i) => <span className="font-semibold text-slate-900">{i.application_name || '—'}</span>
            },
            {
              key: 'process_name',
              header: 'Process',
              render: (i) => <span className="font-mono text-[11px] text-slate-600">{i.process_name || '—'}</span>
            },
            {
              key: 'window_title',
              header: 'Window Title',
              render: (i) => (
                <span className="truncate max-w-sm block text-slate-700" title={i.window_title}>
                  {i.window_title || '—'}
                </span>
              )
            },
            {
              key: 'event_type',
              header: 'Event Type',
              render: (i) => (
                <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                  {i.event_type}
                </span>
              )
            },
            {
              key: 'duration_ms',
              header: 'Duration',
              align: 'right',
              render: (i) => (
                <span className="text-slate-600 font-mono">
                  {i.duration_ms ? `${(i.duration_ms / 1000).toFixed(1)}s` : '0s'}
                </span>
              )
            }
          ]}
          data={data}
          loading={loading}
          page={page}
          totalPages={totalPages}
          totalItems={total}
          onPageChange={setPage}
        />
      )}

      {activeTab === 'browser' && (
        <DataTable<BrowserEvent>
          columns={[
            {
              key: 'timestamp',
              header: 'Timestamp',
              render: (i) => <span className="font-mono text-[11px] text-slate-500">{formatTimestamp(i.timestamp)}</span>,
              width: 'w-44'
            },
            {
              key: 'browser_name',
              header: 'Browser',
              render: (i) => <span className="font-semibold text-slate-900">{i.browser_name}</span>
            },
            {
              key: 'domain',
              header: 'Domain',
              render: (i) => <span className="font-mono text-emerald-700 font-semibold">{i.domain || '—'}</span>
            },
            {
              key: 'page_title',
              header: 'Page Title',
              render: (i) => (
                <span className="truncate max-w-sm block text-slate-700" title={i.page_title}>
                  {i.page_title || '—'}
                </span>
              )
            },
            {
              key: 'url',
              header: 'URL',
              render: (i) => (
                <span className="truncate max-w-xs block font-mono text-[11px] text-slate-500" title={i.url}>
                  {i.url || '—'}
                </span>
              )
            }
          ]}
          data={data}
          loading={loading}
          page={page}
          totalPages={totalPages}
          totalItems={total}
          onPageChange={setPage}
        />
      )}

      {activeTab === 'interaction' && (
        <DataTable<InteractionEvent>
          columns={[
            {
              key: 'timestamp',
              header: 'Timestamp',
              render: (i) => <span className="font-mono text-[11px] text-slate-500">{formatTimestamp(i.timestamp)}</span>,
              width: 'w-44'
            },
            {
              key: 'application_name',
              header: 'Application',
              render: (i) => <span className="font-semibold text-slate-900">{i.application_name}</span>
            },
            {
              key: 'interaction_type',
              header: 'Interaction Type',
              render: (i) => (
                <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
                  {i.interaction_type}
                </span>
              )
            },
            {
              key: 'count',
              header: 'Interaction Count',
              align: 'right',
              render: (i) => <span className="font-mono font-bold text-slate-800">{i.count}</span>
            }
          ]}
          data={data}
          loading={loading}
          page={page}
          totalPages={totalPages}
          totalItems={total}
          onPageChange={setPage}
        />
      )}

      {activeTab === 'screenshot' && (
        <DataTable<ScreenshotEvent>
          columns={[
            {
              key: 'timestamp',
              header: 'Timestamp',
              render: (i) => <span className="font-mono text-[11px] text-slate-500">{formatTimestamp(i.timestamp)}</span>,
              width: 'w-44'
            },
            {
              key: 'application_name',
              header: 'Application',
              render: (i) => <span className="font-semibold text-slate-900">{i.application_name}</span>
            },
            {
              key: 'window_title',
              header: 'Window',
              render: (i) => <span className="truncate max-w-sm block text-slate-700">{i.window_title || '—'}</span>
            },
            {
              key: 'capture_reason',
              header: 'Capture Reason',
              render: (i) => (
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                  {i.capture_reason || 'Periodic'}
                </span>
              )
            },
            {
              key: 'file_size',
              header: 'File Size',
              align: 'right',
              render: (i) => (
                <span className="font-mono text-slate-600">
                  {i.file_size ? `${(i.file_size / 1024).toFixed(1)} KB` : '—'}
                </span>
              )
            }
          ]}
          data={data}
          loading={loading}
          page={page}
          totalPages={totalPages}
          totalItems={total}
          onPageChange={setPage}
        />
      )}

      {activeTab === 'session' && (
        <DataTable<SessionEvent>
          columns={[
            {
              key: 'session_id',
              header: 'Session ID',
              render: (i) => <span className="font-mono text-[11px] font-bold text-blue-700">{i.session_id}</span>
            },
            {
              key: 'started_at',
              header: 'Started At',
              render: (i) => <span className="font-mono text-[11px] text-slate-500">{formatTimestamp(i.started_at)}</span>
            },
            {
              key: 'ended_at',
              header: 'Ended At',
              render: (i) => (
                <span className="font-mono text-[11px] text-slate-500">
                  {i.ended_at ? formatTimestamp(i.ended_at) : 'Active'}
                </span>
              )
            },
            {
              key: 'end_reason',
              header: 'End Reason',
              render: (i) => (
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                  {i.end_reason || 'In-Progress'}
                </span>
              )
            }
          ]}
          data={data}
          loading={loading}
          page={page}
          totalPages={totalPages}
          totalItems={total}
          onPageChange={setPage}
        />
      )}
    </div>
  );
};

import React, { useState } from 'react';
import {
  Brain, Search, Globe, Monitor, MousePointer, Camera,
  Eye, ChevronLeft, ChevronRight
} from 'lucide-react';
import { MemoryRecord } from '../types/memory';
import { MemoryFeatureBar } from '../components/memory/MemoryFeatureBar';
import { useMemoryData } from '../hooks/useMemoryData';
import { MemoryDetailDrawer } from '../components/memory/MemoryDetailDrawer';

const getSourceIcon = (type: string) => {
  switch (type) {
    case 'browser':     return <Globe       className="w-3.5 h-3.5" style={{ color: '#059669' }} />;
    case 'activity':    return <Monitor     className="w-3.5 h-3.5" style={{ color: '#2563eb' }} />;
    case 'interaction': return <MousePointer className="w-3.5 h-3.5" style={{ color: '#d97706' }} />;
    case 'screenshot':  return <Camera      className="w-3.5 h-3.5" style={{ color: '#7c3aed' }} />;
    default:            return <Monitor     className="w-3.5 h-3.5" style={{ color: '#94a3b8' }} />;
  }
};

const lcBadgeClass: Record<string, string> = {
  RETAIN: 'lc-retain', COMPRESS: 'lc-compress', ARCHIVE: 'lc-archive', FORGET: 'lc-forget'
};

const LIFECYCLE_OPTIONS = ['', 'RETAIN', 'COMPRESS', 'ARCHIVE', 'FORGET'];
const SOURCE_OPTIONS = [
  { value: '', label: 'All' },
  { value: 'activity',    label: 'Activity' },
  { value: 'browser',     label: 'Browser' },
  { value: 'interaction', label: 'Interaction' },
  { value: 'screenshot',  label: 'Screenshot' },
];

interface MemoryPageProps {
  initialLifecycleFilter?: string;
}

export const Memory: React.FC<MemoryPageProps> = ({ initialLifecycleFilter = '' }) => {
  const [page, setPage]             = useState<number>(1);
  const [sourceType, setSourceType] = useState<string>('');
  const [lifecycleFilter, setLCF]   = useState<string>(initialLifecycleFilter);
  const [search, setSearch]         = useState<string>('');
  const [selectedMemory, setSelected] = useState<MemoryRecord | null>(null);

  const { memories, total, pages, loading } = useMemoryData(
    page, 20,
    sourceType || undefined,
    search || undefined,
    undefined,
    lifecycleFilter || undefined
  );

  const displayed = lifecycleFilter
    ? memories.filter((m) => m.lifecycle_prediction === lifecycleFilter)
    : memories;

  return (
    <div className="space-y-4">
      {/* ── Filter toolbar ─────────────────────────────────────────── */}
      <div className="card p-4">
        <div className="flex flex-col md:flex-row gap-3 items-start md:items-center">
          {/* Search */}
          <div className="relative flex-1 w-full">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              id="memory-search"
              placeholder="Search content, application, domain, or memory ID…"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              className="search-input"
            />
          </div>

          {/* Source filter */}
          <div className="seg-control flex-shrink-0">
            {SOURCE_OPTIONS.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => { setSourceType(value); setPage(1); }}
                className={`seg-item ${sourceType === value ? 'active' : ''}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Lifecycle filter chips */}
        <div className="flex items-center flex-wrap gap-2 mt-3 pt-3 border-t" style={{ borderColor: 'var(--line)' }}>
          <span className="label-sm">Lifecycle:</span>
          {LIFECYCLE_OPTIONS.map((lc) => (
            <button
              key={lc || 'ALL'}
              onClick={() => { setLCF(lc); setPage(1); }}
              className={`chip ${lifecycleFilter === lc ? 'active' : ''}`}
            >
              {lc || 'ALL'}
            </button>
          ))}
          <span className="ml-auto text-[11px] text-slate-400 whitespace-nowrap">
            {total.toLocaleString()} records total
          </span>
        </div>
      </div>

      {/* ── Table ──────────────────────────────────────────────────── */}
      <div className="card overflow-hidden">
        {loading ? (
          <div className="p-8 flex items-center justify-center gap-3">
            <div className="spinner" />
            <span className="text-[13px] text-slate-500">Loading memory records…</span>
          </div>
        ) : displayed.length === 0 ? (
          <div className="empty-state py-16">
            <Brain className="w-8 h-8" style={{ color: '#cbd5e1' }} />
            <p className="font-medium text-slate-500 mt-3">No records match the selected filters.</p>
            <p className="text-slate-400 mt-1 text-[12px]">Adjust the lifecycle or source type filter.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="text-left">Memory ID</th>
                  <th className="text-left">Source</th>
                  <th className="text-left">Application</th>
                  <th className="text-left">Content</th>
                  <th className="text-center">Importance</th>
                  <th className="text-center">Lifecycle</th>
                  <th className="text-left">Recency</th>
                  <th className="text-left">Frequency</th>
                  <th className="text-left">Timestamp</th>
                  <th className="text-center">Details</th>
                </tr>
              </thead>
              <tbody>
                {displayed.map((mem) => (
                  <tr key={mem.memory_id} onClick={() => setSelected(mem)} style={{ cursor: 'pointer' }}>
                    <td>
                      <span className="font-mono text-[11px] text-slate-500 truncate block max-w-[120px]">
                        {mem.memory_id}
                      </span>
                    </td>
                    <td>
                      <div className="flex items-center gap-1.5">
                        {getSourceIcon(mem.source_type)}
                        <span className="text-[11px] capitalize">{mem.source_type}</span>
                      </div>
                    </td>
                    <td>
                      <span className="font-medium text-[12px] truncate block max-w-[120px]">
                        {mem.application || '—'}
                      </span>
                    </td>
                    <td>
                      <p className="text-[11px] text-slate-500 truncate max-w-[200px]">
                        {mem.content || '—'}
                      </p>
                    </td>
                    <td className="text-center">
                      <span className="font-mono font-bold text-[12px]" style={{ color: 'var(--accent)' }}>
                        {Number(mem.importance_score ?? 0.5).toFixed(3)}
                      </span>
                    </td>
                    <td className="text-center">
                      <span className={`badge ${lcBadgeClass[mem.lifecycle_prediction || ''] || 'badge-slate'}`}>
                        {mem.lifecycle_prediction || 'PENDING'}
                      </span>
                    </td>
                    <td>
                      <div className="w-16">
                        <MemoryFeatureBar label="Recency" value={mem.recency ?? 0} isCalculated />
                      </div>
                    </td>
                    <td>
                      <div className="w-16">
                        <MemoryFeatureBar label="Frequency" value={mem.frequency ?? 0} isCalculated />
                      </div>
                    </td>
                    <td>
                      <span className="font-mono text-[11px] text-slate-400 whitespace-nowrap">
                        {mem.timestamp ? mem.timestamp.substring(0, 19).replace('T', ' ') : '—'}
                      </span>
                    </td>
                    <td className="text-center">
                      <button
                        className="flex items-center gap-1 text-[11px] font-medium mx-auto"
                        style={{ color: 'var(--accent)' }}
                        onClick={(e) => { e.stopPropagation(); setSelected(mem); }}
                      >
                        <Eye className="w-3.5 h-3.5" />
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Pagination ─────────────────────────────────────────────── */}
      {!loading && pages > 1 && (
        <div className="card flex items-center justify-between p-3.5 px-4">
          <span className="text-[12px] text-slate-500">
            Page <strong className="text-slate-700">{page}</strong> of {Math.max(1, pages)} &mdash; {total.toLocaleString()} records
          </span>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPage(page - 1)}
              disabled={page <= 1}
              className="btn btn-secondary h-8 px-2.5"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="text-[12px] font-mono text-slate-600 px-1">{page}</span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page >= pages}
              className="btn btn-secondary h-8 px-2.5"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ── Detail Drawer ───────────────────────────────────────────── */}
      <MemoryDetailDrawer
        memory={selectedMemory}
        onClose={() => setSelected(null)}
      />
    </div>
  );
};

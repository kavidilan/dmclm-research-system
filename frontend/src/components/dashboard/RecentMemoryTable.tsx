import React from 'react';
import { MemoryRecord } from '../../types/memory';
import { ExternalLink, Globe, Monitor, MousePointer, Camera } from 'lucide-react';

interface RecentMemoryTableProps {
  memories: MemoryRecord[];
  loading?: boolean;
  onSelectMemory: (memory: MemoryRecord) => void;
  onViewAll: () => void;
}

const getSourceIcon = (type: string) => {
  switch (type) {
    case 'browser':     return <Globe      className="w-3.5 h-3.5" style={{ color: '#059669' }} />;
    case 'interaction': return <MousePointer className="w-3.5 h-3.5" style={{ color: '#d97706' }} />;
    case 'screenshot':  return <Camera     className="w-3.5 h-3.5" style={{ color: '#7c3aed' }} />;
    default:            return <Monitor    className="w-3.5 h-3.5" style={{ color: '#2563eb' }} />;
  }
};

const getLifecycleBadge = (status?: string) => {
  switch (status) {
    case 'RETAIN':   return 'lc-retain';
    case 'COMPRESS': return 'lc-compress';
    case 'ARCHIVE':  return 'lc-archive';
    case 'FORGET':   return 'lc-forget';
    default:         return 'badge-slate';
  }
};

const formatTime = (ts: string) => {
  try {
    return new Date(ts).toLocaleString([], {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  } catch { return ts; }
};

export const RecentMemoryTable: React.FC<RecentMemoryTableProps> = ({
  memories, loading, onSelectMemory, onViewAll
}) => {
  return (
    <div className="card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-900 tracking-tight">Recent Memory Records</h2>
            <span className="badge badge-real">REAL DATA</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Extracted from desktop sessions · structured for DMCLM consolidation
          </p>
        </div>
        <button
          onClick={onViewAll}
          className="btn btn-ghost text-[12px] gap-1"
          style={{ color: 'var(--accent)' }}
        >
          View All Records
          <ExternalLink className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th className="text-left">Timestamp</th>
              <th className="text-left">Application</th>
              <th className="text-left">Source</th>
              <th className="text-left">Content / Window</th>
              <th className="text-left">Duration</th>
              <th className="text-left">Importance</th>
              <th className="text-right">Lifecycle</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}>
                  {Array.from({ length: 7 }).map((__, j) => (
                    <td key={j} className="py-3 px-4">
                      <div className="skeleton h-3 rounded" style={{ width: j === 3 ? '140px' : '70px' }} />
                    </td>
                  ))}
                </tr>
              ))
            ) : memories.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-10 text-center">
                  <div className="empty-state py-0">
                    <p>No memory records found.</p>
                  </div>
                </td>
              </tr>
            ) : (
              memories.slice(0, 8).map((mem) => (
                <tr key={mem.memory_id} onClick={() => onSelectMemory(mem)}>
                  <td className="whitespace-nowrap">
                    <span className="font-mono text-[11px] text-slate-500">{formatTime(mem.timestamp)}</span>
                  </td>
                  <td className="whitespace-nowrap">
                    <span className="font-semibold text-slate-800 text-[12px]">{mem.application || 'Unknown'}</span>
                  </td>
                  <td className="whitespace-nowrap">
                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 border border-slate-200 text-[11px] font-medium text-slate-700">
                      {getSourceIcon(mem.source_type)}
                      <span className="capitalize">{mem.source_type}</span>
                    </span>
                  </td>
                  <td className="max-w-[240px] truncate text-slate-600 text-[11px]" title={mem.content}>
                    {mem.content || '—'}
                  </td>
                  <td className="whitespace-nowrap text-slate-500 text-[11px] font-mono">
                    {mem.duration_ms ? `${(mem.duration_ms / 1000).toFixed(1)}s` : '—'}
                  </td>
                  <td className="whitespace-nowrap">
                    {mem.importance_score != null ? (
                      <span className="font-mono font-bold text-[12px]" style={{ color: 'var(--accent)' }}>
                        {Number(mem.importance_score).toFixed(3)}
                      </span>
                    ) : (
                      <span className="text-slate-400 text-[11px]">—</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap text-right">
                    <span
                      className={`badge ${getLifecycleBadge(mem.lifecycle_prediction)} text-[9px] tracking-wider`}
                    >
                      {mem.lifecycle_prediction || 'PENDING'}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

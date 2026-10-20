import React, { useEffect, useState } from 'react';
import {
  CheckCircle, Layers, Archive, Trash2, Brain,
  ChevronLeft, ChevronRight, Search, Eye
} from 'lucide-react';
import { api } from '../services/api';
import { ConsolidationStats, MemoryRecord } from '../types/memory';
import { useMemoryData } from '../hooks/useMemoryData';
import { MemoryDetailDrawer } from '../components/memory/MemoryDetailDrawer';
import { MemoryFeatureBar } from '../components/memory/MemoryFeatureBar';

const LC_CONFIG = {
  RETAIN: {
    icon: CheckCircle, label: 'RETAIN',
    desc: 'High-value memories kept in active working memory.',
    longDesc: 'Retained memories are immediately available for active retrieval. They have high semantic relevance, frequent access history, or high recency. The system safeguards these records against premature archival.',
    colorClass: 'lc-retain', textColor: '#166534', bg: 'var(--lc-retain-bg)', border: 'var(--lc-retain-border)',
  },
  COMPRESS: {
    icon: Layers, label: 'COMPRESS',
    desc: 'Repeated episodic bursts consolidated into durable Long-Term Memory nodes.',
    longDesc: 'Memories eligible for compression are grouped by application and semantic similarity. Multiple episodic records are abstracted into a single LTM node, reducing storage while preserving useful knowledge.',
    colorClass: 'lc-compress', textColor: '#1d4ed8', bg: 'var(--lc-compress-bg)', border: 'var(--lc-compress-border)',
  },
  ARCHIVE: {
    icon: Archive, label: 'ARCHIVE',
    desc: 'Historical context with decayed recency preserved in secondary storage.',
    longDesc: 'Archived memories remain queryable but are not in the active retrieval pool. They represent historical context that is no longer frequently needed but may be valuable for future recall.',
    colorClass: 'lc-archive', textColor: '#854d0e', bg: 'var(--lc-archive-bg)', border: 'var(--lc-archive-border)',
  },
  FORGET: {
    icon: Trash2, label: 'FORGET',
    desc: 'Transient background noise excluded from active retrieval.',
    longDesc: 'Forgotten memories are logically excluded from retrieval. The underlying raw collector data in collector.db is never deleted (read-only). Forgotten records can be reviewed but are not served in normal queries.',
    colorClass: 'lc-forget', textColor: '#991b1b', bg: 'var(--lc-forget-bg)', border: 'var(--lc-forget-border)',
  },
};

type LifecycleKey = keyof typeof LC_CONFIG;

const LCCard: React.FC<{
  lifecycle: LifecycleKey;
  count: number | null;
  percentage: number | null;
  loading: boolean;
  selected: boolean;
  onClick: () => void;
}> = ({ lifecycle, count, percentage, loading, selected, onClick }) => {
  const cfg = LC_CONFIG[lifecycle];
  const Icon = cfg.icon;
  return (
    <button
      onClick={onClick}
      className="card p-4 text-left w-full transition-all"
      style={{
        borderColor: selected ? cfg.border : 'var(--line)',
        background: selected ? cfg.bg : '#fff',
        boxShadow: selected ? `0 0 0 1.5px ${cfg.border}` : 'var(--shadow)',
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4" style={{ color: cfg.textColor }} />
          <span className="font-bold text-[13px]" style={{ color: cfg.textColor }}>{cfg.label}</span>
        </div>
      </div>
      <p className="text-[11px] text-slate-600 leading-relaxed mb-3">{cfg.desc}</p>
      {loading ? (
        <div className="skeleton h-6 w-24" />
      ) : (
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono" style={{ color: cfg.textColor }}>
            {count != null ? count.toLocaleString() : '—'}
          </span>
          <span className="text-[12px] text-slate-400">
            {percentage != null ? `${percentage.toFixed(1)}%` : ''}
          </span>
        </div>
      )}
    </button>
  );
};

export const LifecycleManagement: React.FC = () => {
  const [stats, setStats] = useState<ConsolidationStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [selectedLC, setSelectedLC] = useState<LifecycleKey>('RETAIN');
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');

  const { memories, total, pages, loading: memLoading } = useMemoryData(
    page, 15, undefined, search || undefined, undefined, selectedLC
  );

  useEffect(() => {
    api.getConsolidationStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setStatsLoading(false));
  }, []);

  const [selectedMemory, setSelectedMemory] = useState<MemoryRecord | null>(null);

  const lcStates = stats?.lifecycle_states;
  const cfg = LC_CONFIG[selectedLC];
  const Icon = cfg.icon;

  return (
    <div className="space-y-5">
      {/* ── Lifecycle explanation cards ────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {(Object.keys(LC_CONFIG) as LifecycleKey[]).map((lc) => (
          <LCCard
            key={lc}
            lifecycle={lc}
            count={lcStates ? lcStates[lc.toLowerCase() as keyof typeof lcStates]?.count ?? null : null}
            percentage={lcStates ? lcStates[lc.toLowerCase() as keyof typeof lcStates]?.percentage ?? null : null}
            loading={statsLoading}
            selected={selectedLC === lc}
            onClick={() => { setSelectedLC(lc); setPage(1); setSearch(''); }}
          />
        ))}
      </div>

      {/* ── Selected lifecycle explanation ──────────────────────────── */}
      <div className="card p-5 border-l-4" style={{ borderLeftColor: cfg.border }}>
        <div className="flex items-start gap-3">
          <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: cfg.textColor }} />
          <div>
            <h2 className="section-title mb-1.5">{cfg.label} — Lifecycle Decision</h2>
            <p className="text-[13px] text-slate-600 leading-relaxed">{cfg.longDesc}</p>
          </div>
        </div>
      </div>

      {/* ── Records for selected lifecycle ──────────────────────────── */}
      <div className="card">
        <div className="p-4 border-b flex flex-col sm:flex-row sm:items-center gap-3" style={{ borderColor: 'var(--line)' }}>
          <div className="flex items-center gap-2 flex-1">
            <Icon className="w-4 h-4" style={{ color: cfg.textColor }} />
            <h3 className="section-title">{selectedLC} Records</h3>
            <span className="badge" style={{ background: cfg.bg, color: cfg.textColor, borderColor: cfg.border }}>
              {total.toLocaleString()} total
            </span>
          </div>
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search records…"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              className="search-input text-[12px]"
              style={{ paddingLeft: '2rem' }}
            />
          </div>
        </div>

        {memLoading ? (
          <div className="p-8 flex items-center justify-center gap-2.5">
            <div className="spinner" />
            <span className="text-[13px] text-slate-500">Loading…</span>
          </div>
        ) : memories.length === 0 ? (
          <div className="empty-state">
            <Brain className="w-8 h-8" />
            <p className="font-medium text-slate-500 mt-2">No {selectedLC} records found.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="text-left">Memory ID</th>
                  <th className="text-left">Application</th>
                  <th className="text-left">Content</th>
                  <th className="text-center">Importance</th>
                  <th className="text-left">Recency</th>
                  <th className="text-left">Frequency</th>
                  <th className="text-left">Retrieval History</th>
                  <th className="text-center">Details</th>
                </tr>
              </thead>
              <tbody>
                {memories.map((mem) => (
                  <tr key={mem.memory_id} onClick={() => setSelectedMemory(mem)} style={{ cursor: 'pointer' }}>
                    <td>
                      <span className="font-mono text-[11px] text-slate-400 truncate block max-w-[120px]">
                        {mem.memory_id}
                      </span>
                    </td>
                    <td>
                      <span className="font-medium text-[12px] truncate block max-w-[110px]">
                        {mem.application || '—'}
                      </span>
                    </td>
                    <td>
                      <p className="text-[11px] text-slate-500 truncate max-w-[200px]">{mem.content || '—'}</p>
                    </td>
                    <td className="text-center">
                      <span className="font-mono font-bold text-[12px]" style={{ color: cfg.textColor }}>
                        {Number(mem.importance_score ?? 0.5).toFixed(3)}
                      </span>
                    </td>
                    <td><div className="w-14"><MemoryFeatureBar label="Recency" value={mem.recency ?? 0} isCalculated /></div></td>
                    <td><div className="w-14"><MemoryFeatureBar label="Frequency" value={mem.frequency ?? 0} isCalculated /></div></td>
                    <td><div className="w-14"><MemoryFeatureBar label="Retrieval History" value={mem.retrieval_history ?? 0} isCalculated /></div></td>
                    <td className="text-center">
                      <button
                        className="flex items-center gap-1 text-[11px] font-medium mx-auto"
                        style={{ color: 'var(--accent)' }}
                        onClick={(e) => { e.stopPropagation(); setSelectedMemory(mem); }}
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

        {/* Pagination */}
        {!memLoading && pages > 1 && (
          <div className="flex items-center justify-between p-3 px-4 border-t" style={{ borderColor: 'var(--line)' }}>
            <span className="text-[12px] text-slate-500">
              Page {page} of {pages}
            </span>
            <div className="flex items-center gap-1.5">
              <button onClick={() => setPage(page - 1)} disabled={page <= 1} className="btn btn-secondary h-8 px-2.5">
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-[12px] font-mono text-slate-600 px-1">{page}</span>
              <button onClick={() => setPage(page + 1)} disabled={page >= pages} className="btn btn-secondary h-8 px-2.5">
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>

      <MemoryDetailDrawer memory={selectedMemory} onClose={() => setSelectedMemory(null)} />
    </div>
  );
};

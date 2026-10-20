import React, { useEffect, useState } from 'react';
import {
  Eye, X, Layers
} from 'lucide-react';
import { ConsolidationStats, LongTermMemoryNode } from '../types/memory';
import { api } from '../services/api';

export const Consolidation: React.FC = () => {
  const [stats,       setStats]      = useState<ConsolidationStats | null>(null);
  const [ltmNodes,    setLtmNodes]   = useState<LongTermMemoryNode[]>([]);
  const [selectedLtm, setSelectedLtm] = useState<LongTermMemoryNode | null>(null);
  const [loading,     setLoading]    = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const [sRes, lRes] = await Promise.all([
        api.getConsolidationStats().catch(() => null),
        api.getLongTermMemories().catch(() => [])
      ]);
      setStats(sRes);
      setLtmNodes(lRes || []);
      setLoading(false);
    };
    load();
  }, []);

  const comp = stats?.compression_summary;

  return (
    <div className="space-y-5">
      {/* ── Intro ──────────────────────────────────────────────────── */}
      <div className="card p-5 border-l-4" style={{ borderLeftColor: 'var(--accent)' }}>
        <div className="section-eyebrow mb-1">DMCLM Consolidation Engine</div>
        <h1 className="page-title mb-1">Long-Term Memory Consolidation</h1>
        <p className="text-[13px] text-slate-500 leading-relaxed">
          DMCLM identifies repeated episodic activity patterns and consolidates them into durable Long-Term Memory (LTM) nodes.
          Multiple raw episodic records with similar application context, semantic content, and temporal proximity are merged into a single abstract LTM node,
          dramatically reducing storage while preserving knowledge.
        </p>
      </div>

      {/* ── Compression metrics ─────────────────────────────────────── */}
      <div className="card p-5">
        <h2 className="section-title mb-3">Consolidation Summary</h2>
        {loading ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="metric-card"><div className="skeleton h-12 w-full" /></div>
            ))}
          </div>
        ) : comp ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Raw Episodic Records', value: comp.original_episodic_records?.toLocaleString() ?? '—', sub: 'Input memory records', color: 'var(--accent)' },
              { label: 'LTM Nodes Synthesized', value: comp.consolidated_ltm_nodes?.toString() ?? '—', sub: 'Abstract knowledge nodes', color: '#16a34a' },
              { label: 'Records Reduced', value: comp.memory_reduction_count?.toLocaleString() ?? '—', sub: 'Episodic records consolidated', color: '#d97706' },
              { label: 'Compression Ratio', value: comp.compression_ratio_pct != null ? `${comp.compression_ratio_pct.toFixed(2)}%` : '—', sub: 'Storage efficiency gain', color: '#7c3aed' },
            ].map(({ label, value, sub, color }) => (
              <div key={label} className="metric-card">
                <div className="metric-card-accent" style={{ background: color }} />
                <div className="label-sm mb-1">{label}</div>
                <div className="text-xl font-bold font-mono leading-tight" style={{ color }}>{value}</div>
                <p className="text-[11px] text-slate-400 mt-0.5">{sub}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state py-6">
            <p>Consolidation statistics unavailable. Connect the backend.</p>
          </div>
        )}

        {/* Extended stats if available */}
        {comp && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            {[
              { label: 'Unconsolidated Memories', value: (comp as any).unconsolidated_memories?.toLocaleString() ?? '—', color: '#0891b2' },
              { label: 'Total Retained Nodes', value: (comp as any).total_retained_nodes?.toLocaleString() ?? '—', color: '#16a34a' },
              { label: 'Avg Group Size', value: (comp as any).average_group_size != null ? `${(comp as any).average_group_size.toFixed(1)} records/node` : '—', color: '#d97706' },
              { label: 'Group Compression Ratio', value: (comp as any).group_compression_ratio_pct != null ? `${(comp as any).group_compression_ratio_pct.toFixed(2)}%` : '—', color: '#7c3aed' },
            ].map(({ label, value, color }) => (
              <div key={label} className="card-flat p-3">
                <div className="label-sm mb-1">{label}</div>
                <div className="font-bold font-mono text-[14px]" style={{ color }}>{value}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── LTM Nodes ────────────────────────────────────────────────── */}
      <div className="card">
        <div className="p-5 border-b flex items-center justify-between" style={{ borderColor: 'var(--line)' }}>
          <div>
            <h2 className="section-title">Consolidated Long-Term Memory Nodes</h2>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Abstract knowledge nodes synthesized from real episodic activity. Click to inspect source records.
            </p>
          </div>
          <span className="badge badge-navy">{ltmNodes.length} nodes</span>
        </div>

        {loading ? (
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="rounded border p-4 space-y-2" style={{ borderColor: 'var(--line)' }}>
                <div className="flex justify-between"><div className="skeleton h-3 w-20" /><div className="skeleton h-3 w-12" /></div>
                <div className="skeleton h-3 w-3/4" />
                <div className="skeleton h-3 w-full" />
              </div>
            ))}
          </div>
        ) : ltmNodes.length === 0 ? (
          <div className="empty-state py-12">
            <Layers className="w-8 h-8" />
            <p className="font-medium text-slate-500 mt-2">No LTM nodes available.</p>
            <p className="text-[12px] text-slate-400 mt-1">Run the memory consolidation pipeline first.</p>
          </div>
        ) : (
          <div className="p-5 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {ltmNodes.slice(0, 18).map((node) => (
              <div
                key={node.ltm_id}
                onClick={() => setSelectedLtm(node)}
                className="rounded border p-4 cursor-pointer transition-colors flex flex-col justify-between"
                style={{
                  background: '#f8fafc',
                  borderColor: 'var(--line)',
                }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent-soft)')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--line)')}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] text-slate-400 truncate max-w-[110px]">
                      {node.ltm_id}
                    </span>
                    <span className="badge badge-emerald text-[9px]">
                      {(node.stability_score * 100).toFixed(0)}% stable
                    </span>
                  </div>
                  <h3 className="text-[12px] font-bold text-slate-900 leading-snug mb-1">{node.topic}</h3>
                  <p className="text-[11px] text-slate-500 line-clamp-3 leading-relaxed">{node.summary}</p>
                </div>
                <div className="pt-3 border-t flex items-center justify-between mt-2" style={{ borderColor: 'var(--line)' }}>
                  <span className="text-[10px] text-slate-400">
                    From <strong>{node.source_count}</strong> episodic records
                  </span>
                  <button
                    className="text-[11px] font-semibold flex items-center gap-1"
                    style={{ color: 'var(--accent)' }}
                    onClick={(e) => { e.stopPropagation(); setSelectedLtm(node); }}
                  >
                    <Eye className="w-3 h-3" /> Inspect
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
        {ltmNodes.length > 18 && (
          <div className="p-3 border-t text-center text-[12px] text-slate-500" style={{ borderColor: 'var(--line)' }}>
            Showing first 18 of {ltmNodes.length} nodes.
          </div>
        )}
      </div>

      {/* ── LTM Inspector Modal ───────────────────────────────────────── */}
      {selectedLtm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: 'rgba(15,23,42,0.4)' }}
          onClick={() => setSelectedLtm(null)}>
          <div
            className="bg-white rounded-xl max-w-2xl w-full p-6 space-y-5 border shadow-2xl max-h-[85vh] overflow-y-auto"
            style={{ borderColor: 'var(--line)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-4 border-b" style={{ borderColor: 'var(--line)' }}>
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded flex items-center justify-center text-white text-[10px] font-bold"
                  style={{ background: 'var(--accent)' }}>
                  LTM
                </div>
                <div>
                  <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Long-Term Memory Node</div>
                  <h3 className="text-[13px] font-bold text-slate-900 font-mono">{selectedLtm.ltm_id}</h3>
                </div>
              </div>
              <button onClick={() => setSelectedLtm(null)}
                className="w-8 h-8 rounded hover:bg-slate-100 flex items-center justify-center text-slate-400 hover:text-slate-700 cursor-pointer transition-colors">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-1.5">
              <div className="label-sm">Consolidated Summary</div>
              <p className="text-[12px] text-slate-700 p-3 rounded border leading-relaxed"
                style={{ background: '#f8fafc', borderColor: 'var(--line)' }}>
                {selectedLtm.summary}
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {[
                { label: 'Application', value: selectedLtm.application, color: '#334155' },
                { label: 'Contributing', value: `${selectedLtm.source_count} memories`, color: '#16a34a' },
                { label: 'Stability', value: `${(selectedLtm.stability_score * 100).toFixed(1)}%`, color: 'var(--accent)' },
                { label: 'Interactions', value: selectedLtm.total_interactions.toString(), color: '#334155' },
              ].map(({ label, value, color }) => (
                <div key={label} className="p-2.5 rounded border" style={{ background: '#f8fafc', borderColor: 'var(--line)' }}>
                  <div className="label-sm mb-1">{label}</div>
                  <div className="font-bold text-[12px]" style={{ color }}>{value}</div>
                </div>
              ))}
            </div>

            {selectedLtm.unique_windows_accessed.length > 0 && (
              <div>
                <div className="label-sm mb-1.5">Unique Window Contexts</div>
                <div className="flex flex-wrap gap-1">
                  {selectedLtm.unique_windows_accessed.map((w, i) => (
                    <span key={i} className="badge badge-slate">{w}</span>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <div className="label-sm">Contributing Memory IDs ({selectedLtm.source_memory_ids.length})</div>
              </div>
              <div
                className="max-h-32 overflow-y-auto p-3 rounded text-[10px] font-mono space-y-0.5"
                style={{ background: '#1a2236', color: '#8fa3bd', border: '1px solid #253555' }}
              >
                {selectedLtm.source_memory_ids.map((id, i) => (
                  <div key={i} className="truncate">
                    [{i + 1}] {id}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-end pt-1">
              <button onClick={() => setSelectedLtm(null)} className="btn btn-secondary">
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

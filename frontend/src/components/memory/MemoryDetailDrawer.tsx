import React, { useEffect, useState } from 'react';
import {
  X, Calendar, Clock, Monitor, Globe, Hash, Shield,
  Layers, ArrowUpRight, Sparkles, RefreshCw, CheckCircle2, Brain
} from 'lucide-react';
import { MemoryDetailResponse, MemoryRecord, AdaptiveRetrievalResponse } from '../../types/memory';
import { MemoryFeatureBar } from './MemoryFeatureBar';
import { api } from '../../services/api';

interface MemoryDetailDrawerProps {
  memory: MemoryRecord | null;
  onClose: () => void;
  onSelectRelated?: (memoryId: string) => void;
  onMemoryUpdated?: (updatedMem: MemoryRecord) => void;
}

const lcColors: Record<string, { bg: string; text: string; border: string }> = {
  RETAIN:   { bg: '#ecfdf5', text: '#065f46', border: '#6ee7b7' },
  COMPRESS: { bg: '#eff6ff', text: '#1d4ed8', border: '#93c5fd' },
  ARCHIVE:  { bg: '#fffbeb', text: '#92400e', border: '#fcd34d' },
  FORGET:   { bg: '#fff1f2', text: '#9f1239', border: '#fda4af' },
};

const Field: React.FC<{ label: string; icon?: React.ReactNode; children: React.ReactNode; span?: boolean }> = ({
  label, icon, children, span
}) => (
  <div className={span ? 'col-span-2' : ''}>
    <div className="flex items-center gap-1 mb-0.5" style={{ color: '#94a3b8' }}>
      {icon && <span className="w-3.5 h-3.5">{icon}</span>}
      <span className="text-[10px] font-semibold uppercase tracking-wider">{label}</span>
    </div>
    <div className="text-[12px] font-semibold text-slate-800">{children}</div>
  </div>
);

export const MemoryDetailDrawer: React.FC<MemoryDetailDrawerProps> = ({
  memory, onClose, onSelectRelated, onMemoryUpdated
}) => {
  const [detailData,     setDetailData]     = useState<MemoryDetailResponse | null>(null);
  const [loading,        setLoading]        = useState(false);
  const [retrieving,     setRetrieving]     = useState(false);
  const [retrievalResult, setRetrievalResult] = useState<AdaptiveRetrievalResponse | null>(null);

  useEffect(() => {
    if (!memory) { setDetailData(null); setRetrievalResult(null); return; }
    setLoading(true);
    api.getMemoryById(memory.memory_id)
      .then(setDetailData)
      .catch(() => setDetailData({
        memory,
        related_memories: [],
        lifecycle_state: {
          status:     memory.lifecycle_prediction || 'RETAIN',
          confidence: memory.lifecycle_confidence || 0.95,
          reason:     memory.lifecycle_explanation || 'DMCLM cognitive evaluation'
        }
      }))
      .finally(() => setLoading(false));
  }, [memory]);

  if (!memory) return null;

  const mem = detailData?.memory || memory;
  const lc  = mem.lifecycle_prediction || 'RETAIN';
  const lcStyle = lcColors[lc] || { bg: '#f1f5f9', text: '#334155', border: '#cbd5e1' };

  const handleRetrieval = async () => {
    setRetrieving(true);
    try {
      const res = await api.retrieveMemory(mem.memory_id);
      setRetrievalResult(res);
      const updated: MemoryRecord = {
        ...mem,
        lifecycle_prediction: res.new_lifecycle as any,
        lifecycle_confidence: res.confidence,
        lifecycle_explanation: res.explanation,
        recency: res.updated_features.recency,
        retrieval_history: res.updated_features.retrieval_history
      };
      if (detailData) setDetailData({ ...detailData, memory: updated, lifecycle_state: { status: res.new_lifecycle, confidence: res.confidence, reason: res.explanation } });
      onMemoryUpdated?.(updated);
    } catch (err: any) {
      alert(`Retrieval update failed: ${err.message}`);
    } finally {
      setRetrieving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end" style={{ background: 'rgba(15,23,42,0.45)', backdropFilter: 'blur(4px)' }}>
      <div
        className="w-full max-w-lg bg-white h-full shadow-2xl flex flex-col border-l border-slate-200 drawer-slide-in overflow-hidden"
        role="dialog"
        aria-label="Memory Detail"
      >
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="px-5 py-4 flex items-center justify-between border-b border-slate-100 flex-shrink-0"
          style={{ background: '#f8fafc' }}>
          <div className="flex items-center gap-3 min-w-0">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center text-white text-[11px] font-bold flex-shrink-0"
              style={{ background: '#0d9488' }}
            >
              MEM
            </div>
            <div className="min-w-0">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Memory Trace</div>
              <div className="text-[12px] font-bold text-slate-900 font-mono truncate">{mem.memory_id}</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg hover:bg-slate-200 flex items-center justify-center text-slate-400 hover:text-slate-700 transition-colors flex-shrink-0 cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* ── Body ───────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {loading ? (
            <div className="space-y-3">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="skeleton h-4 rounded" style={{ width: `${70 + (i % 3) * 10}%` }} />
              ))}
            </div>
          ) : (
            <>
              {/* Metadata grid */}
              <div className="grid grid-cols-2 gap-x-4 gap-y-3 p-4 rounded-xl border border-slate-100 bg-slate-50">
                <Field label="Application" icon={<Monitor className="w-3.5 h-3.5" />}>
                  {mem.application || 'Unknown'}
                </Field>
                <Field label="Source Modality" icon={<Globe className="w-3.5 h-3.5" />}>
                  <span className="capitalize">{mem.source_type}</span>
                </Field>
                <Field label="Timestamp" icon={<Calendar className="w-3.5 h-3.5" />}>
                  <span className="font-mono text-[11px]">{mem.timestamp}</span>
                </Field>
                <Field label="Duration" icon={<Clock className="w-3.5 h-3.5" />}>
                  {mem.duration_ms ? `${(mem.duration_ms / 1000).toFixed(1)}s` : 'Instantaneous'}
                </Field>
                {mem.domain && (
                  <Field label="Web Domain" span>
                    <span className="font-mono text-emerald-700">{mem.domain}</span>
                  </Field>
                )}
                <Field label="Session ID" icon={<Hash className="w-3.5 h-3.5" />} span>
                  <span className="font-mono text-[11px] text-slate-600 truncate block">{mem.session_id}</span>
                </Field>
              </div>

              {/* Content */}
              <div className="space-y-2">
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Extracted Content / Observation
                </div>
                <div
                  className="p-4 rounded-xl text-[11px] leading-relaxed whitespace-pre-wrap break-words"
                  style={{
                    fontFamily: 'var(--font-mono)',
                    background: '#0f172a',
                    color: '#e2e8f0',
                    border: '1px solid #1e293b'
                  }}
                >
                  {mem.content || 'No text extracted'}
                </div>
              </div>

              {/* DMCLM 5D Features */}
              <div className="p-4 rounded-xl border border-slate-200 bg-white space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-teal-600" />
                    <h3 className="text-[11px] font-bold uppercase tracking-wider text-slate-800">
                      DMCLM 5D Features
                    </h3>
                  </div>
                  <span className="badge badge-calc">CALCULATED</span>
                </div>
                <div className="space-y-2.5 pt-1">
                  <MemoryFeatureBar label="Recency (R)"              value={mem.recency           ?? (mem as any).recency_score           ?? 0} isCalculated />
                  <MemoryFeatureBar label="Frequency (F)"            value={mem.frequency         ?? (mem as any).frequency_score         ?? 0} isCalculated />
                  <MemoryFeatureBar label="Retrieval History (H)"    value={mem.retrieval_history ?? (mem as any).retrieval_history_score ?? 0} isCalculated />
                  <MemoryFeatureBar label="Semantic Relevance (S)"   value={mem.semantic_relevance  ?? 0} isCalculated />
                  <MemoryFeatureBar label="Relationship Strength (Rel)" value={mem.relationship_strength ?? 0} isCalculated />
                </div>
                <div
                  className="pt-2 mt-1 border-t border-slate-100 flex items-center justify-between"
                >
                  <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">
                    Importance Score
                  </span>
                  <span className="font-mono font-bold text-base text-teal-700">
                    {Number(mem.importance_score ?? (mem as any).baseline_score ?? 0.5).toFixed(4)}
                  </span>
                </div>
              </div>

              {/* Lifecycle prediction */}
              <div
                className="p-4 rounded-xl border space-y-2"
                style={{ background: lcStyle.bg, borderColor: lcStyle.border }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Shield className="w-4 h-4" style={{ color: lcStyle.text }} />
                    <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: lcStyle.text }}>
                      Lifecycle Prediction
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono" style={{ color: lcStyle.text, opacity: 0.7 }}>
                      {((mem.lifecycle_confidence || 0.95) * 100).toFixed(1)}% conf.
                    </span>
                    <span
                      className="badge text-[9px] tracking-widest font-bold"
                      style={{ background: lcStyle.bg, color: lcStyle.text, borderColor: lcStyle.border }}
                    >
                      {lc}
                    </span>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed p-2.5 rounded-lg bg-white/60"
                  style={{ color: '#475569' }}>
                  {mem.lifecycle_explanation || detailData?.lifecycle_state?.reason || 'DMCLM baseline evaluation.'}
                </p>
              </div>

              {/* Adaptive retrieval */}
              <div className="p-4 rounded-xl border border-teal-200 bg-teal-50/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-teal-600" />
                    <span className="text-[11px] font-bold uppercase tracking-wider text-teal-900">
                      Adaptive Retrieval
                    </span>
                  </div>
                  <span className="badge badge-indigo text-[9px]">Dynamic Loop</span>
                </div>
                <p className="text-[12px] text-teal-900/70 leading-relaxed">
                  Trigger active user retrieval: boosts recency and retrieval history, then re-evaluates lifecycle through DMCLM.
                </p>
                <button
                  onClick={handleRetrieval}
                  disabled={retrieving}
                  className="btn btn-primary w-full justify-center"
                  style={{ background: '#0d9488', borderColor: '#0d9488' }}
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${retrieving ? 'animate-spin' : ''}`} />
                  <span>{retrieving ? 'Updating Audit Log…' : 'Retrieve This Memory'}</span>
                </button>

                {retrievalResult && (
                  <div className="p-3 bg-white rounded-lg border border-indigo-200 space-y-1 fade-in">
                    <div className="flex items-center gap-1 text-emerald-700 font-bold text-[11px]">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Dynamic Transition Executed
                    </div>
                    <div className="text-[11px] text-slate-700 font-mono">
                      {retrievalResult.previous_lifecycle} → <strong className="text-teal-700">{retrievalResult.new_lifecycle}</strong>
                    </div>
                    <p className="text-[10px] text-slate-500 leading-relaxed">{retrievalResult.explanation}</p>
                  </div>
                )}
              </div>

              {/* Related memories */}
              {detailData?.related_memories && detailData.related_memories.length > 0 && (
                <div className="space-y-2">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                    Co-occurring Session Traces ({detailData.related_memories.length})
                  </div>
                  <div className="space-y-1.5">
                    {detailData.related_memories.map((rel) => (
                      <div
                        key={rel.memory_id}
                        onClick={() => onSelectRelated?.(rel.memory_id)}
                        className="flex items-center justify-between p-3 rounded-lg border border-slate-200 bg-slate-50 hover:bg-white hover:border-slate-300 cursor-pointer transition-colors"
                      >
                        <div className="min-w-0">
                          <div className="text-[12px] font-semibold text-slate-800">{rel.application}</div>
                          <p className="text-[10px] text-slate-500 truncate">{rel.content}</p>
                        </div>
                        <ArrowUpRight className="w-4 h-4 text-slate-400 flex-shrink-0 ml-2" />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

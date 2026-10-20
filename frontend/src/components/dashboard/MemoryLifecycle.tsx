import React from 'react';
import { Database, Filter, Cpu, GitMerge, ArrowRight, CheckCircle, Archive, Trash2, Layers, ChevronRight } from 'lucide-react';

const stages = [
  {
    step: '01',
    title: 'Raw Collector Data',
    subtitle: 'Desktop Traces',
    desc: 'Active windows, browser tabs, interactions & screenshots ingested from collector.db (read-only).',
    icon: Database,
    statusLabel: 'REAL DATA',
    statusClass: 'badge-blue'
  },
  {
    step: '02',
    title: 'Memory Extraction',
    subtitle: 'Episodic Normalization',
    desc: 'Unified schema: application, domain, duration, content & timestamp → memory_records.json.',
    icon: Filter,
    statusLabel: 'REAL DATA',
    statusClass: 'badge-blue'
  },
  {
    step: '03',
    title: 'Feature Analysis',
    subtitle: '5D Cognitive Factors',
    desc: 'Recency (R) · Frequency (F) · Retrieval (H) · Semantic (S) · Relationship (Rel)',
    icon: Cpu,
    statusLabel: 'ENGINE ON',
    statusClass: 'badge-indigo'
  },
  {
    step: '04',
    title: 'DMCLM Decision',
    subtitle: 'Lifecycle Gating',
    desc: 'Score: 0.20R + 0.20F + 0.25H + 0.20S + 0.15Rel → Consolidation & lifecycle prediction.',
    icon: GitMerge,
    statusLabel: 'ENGINE ON',
    statusClass: 'badge-indigo'
  }
];

const outcomes = [
  { name: 'RETAIN',   desc: 'High-value active traces',        icon: CheckCircle, cls: 'lc-retain',   iconColor: '#059669' },
  { name: 'COMPRESS', desc: 'Episodic → LTM abstraction',     icon: Layers,      cls: 'lc-compress', iconColor: '#2563eb' },
  { name: 'ARCHIVE',  desc: 'Decayed historical context',      icon: Archive,     cls: 'lc-archive',  iconColor: '#d97706' },
  { name: 'FORGET',   desc: 'Prune ephemeral noise',           icon: Trash2,      cls: 'lc-forget',   iconColor: '#e11d48' }
];

interface MemoryLifecycleProps {
  onSelectOutcome: (lifecycle: string) => void;
}

export const MemoryLifecycle: React.FC<MemoryLifecycleProps> = ({ onSelectOutcome }) => {
  return (
    <div className="card p-6">
      <div className="mb-5">
        <div className="section-eyebrow mb-1">Research Architecture</div>
        <h2 className="text-sm font-bold text-slate-900 tracking-tight">
          DMCLM Cognitive Memory Lifecycle
        </h2>
        <p className="text-[11px] text-slate-500 mt-0.5">
          End-to-end pipeline from raw desktop activity to durable consolidated knowledge representations.
        </p>
      </div>

      {/* Pipeline stages */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 relative mb-5">
        {stages.map((stage, idx) => {
          const Icon = stage.icon;
          return (
            <React.Fragment key={stage.step}>
              <div
                className="rounded-lg p-4 border relative"
                style={{
                  background: '#f8fafc',
                  borderColor: '#e2e8f0',
                  transition: 'border-color 150ms ease'
                }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = '#99f6e4')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = '#e6eaf0')}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[10px] font-bold text-slate-400 font-mono">STAGE {stage.step}</span>
                  <span className={`badge ${stage.statusClass}`}>{stage.statusLabel}</span>
                </div>

                <div className="flex items-center gap-2.5 mb-2">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: '#f0fdfa', border: '1px solid #99f6e4' }}
                  >
                    <Icon className="w-4 h-4" style={{ color: '#0d9488' }} />
                  </div>
                  <div>
                    <h3 className="text-[12px] font-bold text-slate-900 leading-tight">{stage.title}</h3>
                    <p className="text-[10px] text-slate-500">{stage.subtitle}</p>
                  </div>
                </div>

                <p className="text-[11px] text-slate-600 leading-relaxed">{stage.desc}</p>

                {/* Connector arrow — hidden on mobile, between cards on lg */}
                {idx < stages.length - 1 && (
                  <div className="hidden lg:flex absolute -right-4 top-1/2 -translate-y-1/2 z-10 w-8 h-8 rounded-full bg-white border border-slate-200 shadow-sm items-center justify-center">
                    <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                  </div>
                )}
              </div>
            </React.Fragment>
          );
        })}
      </div>

      {/* Outcome tier */}
      <div className="pt-4 border-t border-slate-100">
        <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-3">
          Lifecycle Decision Outcomes
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {outcomes.map((out) => {
            const Icon = out.icon;
            return (
              <button
                key={out.name}
                type="button"
                onClick={() => onSelectOutcome(out.name)}
                aria-label={`Show ${out.name.toLowerCase()} memories`}
                className={`flex w-full items-center gap-2.5 px-3 py-2.5 rounded-lg text-left transition hover:brightness-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-600 ${out.cls}`}
              >
                <Icon className="w-4 h-4 flex-shrink-0" style={{ color: out.iconColor }} />
                <div className="min-w-0 flex-1">
                  <div className="text-[11px] font-bold leading-tight">{out.name}</div>
                  <div className="text-[10px] opacity-75 leading-tight mt-0.5">{out.desc}</div>
                </div>
                <ArrowRight className="w-3.5 h-3.5 flex-shrink-0 opacity-60" aria-hidden="true" />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

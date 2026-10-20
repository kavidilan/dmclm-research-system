import React from 'react';
import { Database, Cpu, Activity, Network, Scale, Sparkles, ChevronRight } from 'lucide-react';

const steps = [
  {
    num: '1',
    title: 'Episodic Memories',
    desc: 'Raw ingested traces from desktop activity via collector.db',
    icon: Database,
    badge: 'REAL STORE',
    badgeClass: 'badge-blue'
  },
  {
    num: '2',
    title: 'Feature Extraction',
    desc: 'Time delta, access count & session frequency calculation',
    icon: Cpu,
    badge: 'ENGINE',
    badgeClass: 'badge-indigo'
  },
  {
    num: '3',
    title: 'Importance Analysis',
    desc: 'Ebbinghaus decay curve & weighted cognitive scoring',
    icon: Activity,
    badge: 'ENGINE',
    badgeClass: 'badge-indigo'
  },
  {
    num: '4',
    title: 'Semantic Linking',
    desc: 'Cross-app clustering & conceptual affinity mapping',
    icon: Network,
    badge: 'ENGINE',
    badgeClass: 'badge-indigo'
  },
  {
    num: '5',
    title: 'Lifecycle Decision',
    desc: 'Resource-aware threshold classification (RETAIN/COMPRESS/ARCHIVE/FORGET)',
    icon: Scale,
    badge: 'ENGINE',
    badgeClass: 'badge-indigo'
  },
  {
    num: '6',
    title: 'Long-Term Memory',
    desc: 'Consolidated LTM nodes ready for retrieval',
    icon: Sparkles,
    badge: 'LTM STORE',
    badgeClass: 'badge-emerald'
  }
];

export const ConsolidationPipeline: React.FC = () => {
  return (
    <div className="card p-6">
      <div className="mb-5">
        <div className="section-eyebrow mb-1">Theoretical Workflow</div>
        <h2 className="text-sm font-bold text-slate-900 tracking-tight">
          DMCLM Adaptive Consolidation Architecture
        </h2>
        <p className="text-[11px] text-slate-500 mt-0.5">
          Cognitive transformation from volatile episodic observation to durable long-term memory structures
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 relative">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={step.num} className="relative">
              <div
                className="rounded-xl p-3.5 flex flex-col h-full border border-slate-200 hover:border-teal-200 transition-colors"
                style={{ background: '#f8fafc' }}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[10px] font-bold font-mono text-slate-400">
                    {step.num.padStart(2, '0')}
                  </span>
                  <span className={`badge ${step.badgeClass} text-[8px]`}>{step.badge}</span>
                </div>

                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center mb-2.5 flex-shrink-0"
                  style={{ background: '#f0fdfa', border: '1px solid #99f6e4' }}
                >
                  <Icon className="w-4 h-4" style={{ color: '#0d9488' }} />
                </div>

                <h3 className="text-[11px] font-bold text-slate-900 leading-snug mb-1">{step.title}</h3>
                <p className="text-[10px] text-slate-500 leading-relaxed">{step.desc}</p>
              </div>

              {/* Connector */}
              {idx < steps.length - 1 && (
                <div className="hidden lg:flex absolute -right-4 top-1/2 -translate-y-1/2 z-10 w-8 h-8 items-center justify-center">
                  <ChevronRight className="w-4 h-4 text-slate-300" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

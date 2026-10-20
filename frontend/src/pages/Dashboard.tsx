import React from 'react';
import { DashboardStats, TimelineDataPoint } from '../types/collector';
import { MemoryRecord } from '../types/memory';
import { ActivityChart } from '../components/dashboard/ActivityChart';
import { RecentMemoryTable } from '../components/dashboard/RecentMemoryTable';

interface DashboardPageProps {
  stats: DashboardStats | null;
  timeline: TimelineDataPoint[];
  recentMemories: MemoryRecord[];
  loading: boolean;
  onSelectMemory: (mem: MemoryRecord) => void;
  onNavigateToMemory: () => void;
  onSelectLifecycle: (lifecycle: string) => void;
  onNavigateToLifecycle?: () => void;
}

const MetricItem: React.FC<{
  label: string;
  value: string | number | null;
  sub?: string;
  loading?: boolean;
  accentColor?: string;
}> = ({ label, value, sub, loading, accentColor = 'var(--accent)' }) => (
  <div className="metric-card">
    <div className="metric-card-accent" style={{ background: accentColor }} />
    <div className="label-sm mb-1">{label}</div>
    {loading ? (
      <div className="skeleton h-6 w-20 my-1" />
    ) : (
      <div className="text-xl font-bold font-mono leading-tight tracking-tight" style={{ color: accentColor }}>
        {value == null ? '—' : typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    )}
    {sub && <p className="text-[11px] text-slate-400 mt-0.5">{sub}</p>}
  </div>
);

export const Dashboard: React.FC<DashboardPageProps> = ({
  stats,
  timeline,
  recentMemories,
  loading,
  onSelectMemory,
  onNavigateToMemory,
  onSelectLifecycle,
  onNavigateToLifecycle
}) => {
  return (
    <div className="space-y-5">
      {/* ── Page intro ───────────────────────────────────────────────── */}
      <div className="card p-5 border-l-4" style={{ borderLeftColor: 'var(--accent)' }}>
        <div className="section-eyebrow mb-1">Member 4 — Adaptive Long-Term Memory Consolidation</div>
        <h1 className="page-title mb-1">Adaptive Memory Consolidation</h1>
        <p className="text-[13px] text-slate-500">
          DMCLM-based lifecycle management for personalized desktop intelligence.
          Evaluate memory importance, preserve useful information, and manage long-term storage through adaptive lifecycle decisions.
        </p>
      </div>

      {/* ── Summary metrics ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricItem
          label="Memory Records"
          value={stats?.memory_records ?? null}
          sub="Normalized episodic records"
          loading={loading}
          accentColor="var(--accent)"
        />
        <MetricItem
          label="Activity Events"
          value={stats?.activity_events ?? null}
          sub="Window & app focus"
          loading={loading}
          accentColor="#2563eb"
        />
        <MetricItem
          label="Browser Events"
          value={stats?.browser_events ?? null}
          sub="Web pages & URLs"
          loading={loading}
          accentColor="#059669"
        />
        <MetricItem
          label="Interactions"
          value={stats?.interaction_events ?? null}
          sub="Keystrokes & clicks"
          loading={loading}
          accentColor="#d97706"
        />
        <MetricItem
          label="Screenshots"
          value={stats?.screenshots ?? null}
          sub="Contextual captures"
          loading={loading}
          accentColor="#7c3aed"
        />
        <MetricItem
          label="Sessions"
          value={stats?.sessions ?? null}
          sub="Collection sessions"
          loading={loading}
          accentColor="#0891b2"
        />
      </div>

      {/* ── Activity chart ──────────────────────────────────────────── */}
      <ActivityChart data={timeline} loading={loading} />

      {/* ── Recent memory records ────────────────────────────────────── */}
      <RecentMemoryTable
        memories={recentMemories}
        loading={loading}
        onSelectMemory={onSelectMemory}
        onViewAll={onNavigateToMemory}
      />
    </div>
  );
};

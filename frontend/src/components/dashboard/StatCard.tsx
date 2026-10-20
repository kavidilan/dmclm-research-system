import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: number | string;
  description: string;
  icon: LucideIcon;
  badge?: string;
  badgeType?: 'real' | 'calculated' | 'pending';
  loading?: boolean;
  accent?: 'blue' | 'indigo' | 'emerald' | 'amber' | 'rose';
}

const accentConfig = {
  blue:    { bar: '#2563eb', iconBg: '#eff6ff', iconColor: '#2563eb' },
  indigo:  { bar: '#0d9488', iconBg: '#f0fdfa', iconColor: '#0d9488' },
  emerald: { bar: '#059669', iconBg: '#ecfdf5', iconColor: '#059669' },
  amber:   { bar: '#d97706', iconBg: '#fffbeb', iconColor: '#d97706' },
  rose:    { bar: '#e11d48', iconBg: '#fff1f2', iconColor: '#e11d48' },
};

const badgeStyles = {
  real:       'badge-blue',
  calculated: 'badge-emerald',
  pending:    'badge-amber',
};

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  description,
  icon: Icon,
  badge = 'REAL',
  badgeType = 'real',
  loading = false,
  accent = 'blue'
}) => {
  const cfg = accentConfig[accent];

  return (
    <div className="metric-card">
      <div className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-xl" style={{ background: cfg.bar }} />

      <div className="flex items-start justify-between mb-3">
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: cfg.iconBg }}
        >
          <Icon style={{ color: cfg.iconColor, width: 18, height: 18 }} />
        </div>
        <span className={`badge ${badgeStyles[badgeType]}`}>{badge}</span>
      </div>

      <div className="space-y-0.5">
        <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">{title}</p>
        {loading ? (
          <div className="skeleton h-7 w-20 my-1" />
        ) : (
          <div className="text-2xl font-extrabold tracking-tight leading-none text-slate-900">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </div>
        )}
        <p className="text-[12px] text-slate-400 leading-snug">{description}</p>
      </div>
    </div>
  );
};

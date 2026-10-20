import React from 'react';
import { LucideIcon } from 'lucide-react';

interface LifecycleCardProps {
  name: string;
  count: number;
  percentage: number;
  status: string;
  description: string;
  icon: LucideIcon;
  colorScheme: 'emerald' | 'blue' | 'amber' | 'rose';
}

const schemeMap = {
  emerald: {
    border: '#6ee7b7', bg: '#ecfdf5', iconBg: '#d1fae5',
    iconColor: '#059669', bar: '#10b981', text: '#065f46', badgeCls: 'lc-retain'
  },
  blue: {
    border: '#93c5fd', bg: '#eff6ff', iconBg: '#dbeafe',
    iconColor: '#2563eb', bar: '#3b82f6', text: '#1d4ed8', badgeCls: 'lc-compress'
  },
  amber: {
    border: '#fcd34d', bg: '#fffbeb', iconBg: '#fef3c7',
    iconColor: '#d97706', bar: '#f59e0b', text: '#92400e', badgeCls: 'lc-archive'
  },
  rose: {
    border: '#fda4af', bg: '#fff1f2', iconBg: '#ffe4e6',
    iconColor: '#e11d48', bar: '#f43f5e', text: '#9f1239', badgeCls: 'lc-forget'
  }
};

export const LifecycleCard: React.FC<LifecycleCardProps> = ({
  name, count, percentage, status, description, icon: Icon, colorScheme
}) => {
  const s = schemeMap[colorScheme];
  const isPending = count === 0 && percentage === 0;

  return (
    <div
      className="rounded-xl p-5 flex flex-col justify-between border"
      style={{ background: s.bg, borderColor: s.border }}
    >
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: s.iconBg }}
            >
              <Icon className="w-4 h-4" style={{ color: s.iconColor }} />
            </div>
            <h3 className="text-sm font-bold tracking-tight" style={{ color: '#0f172a' }}>{name}</h3>
          </div>
          <span className={`badge ${s.badgeCls} text-[9px] tracking-wider`}>{status}</span>
        </div>

        <p className="text-[11px] leading-relaxed mb-4" style={{ color: '#475569' }}>{description}</p>
      </div>

      {/* Stats + mini bar */}
      <div className="pt-3 border-t space-y-2" style={{ borderColor: s.border }}>
        <div className="flex items-baseline justify-between">
          <div>
            <div className="text-[9px] font-bold uppercase tracking-wider mb-0.5" style={{ color: s.text, opacity: 0.6 }}>
              Memory Count
            </div>
            {isPending ? (
              <span className="text-[12px] font-semibold italic" style={{ color: '#94a3b8' }}>Pending</span>
            ) : (
              <span className="text-xl font-bold font-mono" style={{ color: '#0f172a' }}>
                {count.toLocaleString()}
              </span>
            )}
          </div>
          <div className="text-right">
            <div className="text-[9px] font-bold uppercase tracking-wider mb-0.5" style={{ color: s.text, opacity: 0.6 }}>
              Share
            </div>
            {isPending ? (
              <span className="text-[11px]" style={{ color: '#94a3b8' }}>—</span>
            ) : (
              <span className="text-sm font-bold font-mono" style={{ color: s.text }}>
                {percentage.toFixed(1)}%
              </span>
            )}
          </div>
        </div>

        {/* Proportion bar */}
        {!isPending && (
          <div className="feature-bar-track">
            <div
              className="feature-bar-fill"
              style={{ width: `${Math.min(100, percentage)}%`, background: s.bar }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

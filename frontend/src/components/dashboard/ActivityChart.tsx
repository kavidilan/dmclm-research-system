import React, { useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { TimelineDataPoint } from '../../types/collector';

interface ActivityChartProps {
  data: TimelineDataPoint[];
  loading?: boolean;
}

type Metric = 'all' | 'activity' | 'browser' | 'interactions';

const tooltipStyle = {
  backgroundColor: '#ffffff',
  border: '1px solid var(--line)',
  borderRadius: '8px',
  fontSize: '12px',
  color: '#172033',
  fontFamily: 'Inter, sans-serif',
  boxShadow: '0 4px 12px rgba(15,23,42,0.07)'
};

export const ActivityChart: React.FC<ActivityChartProps> = ({ data, loading }) => {
  const [activeMetric, setActiveMetric] = useState<Metric>('all');

  const metrics: { key: Metric; label: string }[] = [
    { key: 'all',          label: 'All Sources' },
    { key: 'activity',     label: 'Activity' },
    { key: 'browser',      label: 'Browser' },
    { key: 'interactions', label: 'Interactions' }
  ];

  return (
    <div className="card p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-sm font-bold text-slate-900 tracking-tight">
              Data Collection Activity
            </h2>
            <span className="badge badge-real">REAL DATA</span>
          </div>
          <p className="text-[11px] text-slate-500">
            Timeline aggregation from DesktopResearchCollector · SQLite
          </p>
        </div>

        {/* Metric switcher */}
        <div className="seg-control flex-shrink-0">
          {metrics.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveMetric(key)}
              className={`seg-item ${activeMetric === key ? 'active' : ''}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="h-56 flex items-center justify-center">
          <div className="text-center">
            <div className="spinner mx-auto mb-2" />
            <p className="text-[11px] text-slate-400">Loading timeline from SQLite…</p>
          </div>
        </div>
      ) : data.length === 0 ? (
        <div className="empty-state h-56">
          <p className="font-medium text-slate-500">No timeline data yet</p>
          <p className="text-slate-400 mt-1">Events appear as the collector logs sessions.</p>
        </div>
      ) : (
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="gActivity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#315E9E" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#315E9E" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gBrowser" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#16a34a" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#16a34a" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gInteractions" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#d97706" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#d97706" stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="time"
                stroke="#cbd5e1"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                tick={{ fill: '#94a3b8' }}
              />
              <YAxis
                stroke="#cbd5e1"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                tick={{ fill: '#94a3b8' }}
              />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#94a3b8' }} />
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '12px', color: '#64748b' }}
              />

              {(activeMetric === 'all' || activeMetric === 'activity') && (
                <Area
                  type="monotone"
                  dataKey="activity"
                  name="Activity Events"
                  stroke="#315E9E"
                  strokeWidth={2}
                  fill="url(#gActivity)"
                  dot={false}
                  activeDot={{ r: 4, fill: '#315E9E', strokeWidth: 0 }}
                />
              )}
              {(activeMetric === 'all' || activeMetric === 'browser') && (
                <Area
                  type="monotone"
                  dataKey="browser"
                  name="Browser Events"
                  stroke="#16a34a"
                  strokeWidth={1.5}
                  fill="url(#gBrowser)"
                  dot={false}
                  activeDot={{ r: 4, fill: '#16a34a', strokeWidth: 0 }}
                />
              )}
              {(activeMetric === 'all' || activeMetric === 'interactions') && (
                <Area
                  type="monotone"
                  dataKey="interactions"
                  name="Interaction Events"
                  stroke="#d97706"
                  strokeWidth={1.5}
                  fill="url(#gInteractions)"
                  dot={false}
                  activeDot={{ r: 4, fill: '#d97706', strokeWidth: 0 }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

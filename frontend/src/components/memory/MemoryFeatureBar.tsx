import React from 'react';

interface MemoryFeatureBarProps {
  label: string;
  value?: number | null;
  isCalculated?: boolean;
  color?: string;
}

const featureColors: Record<string, string> = {
  'Recency':               '#0d9488',
  'Recency (R)':           '#0d9488',
  'Frequency':             '#3b82f6',
  'Frequency (F)':         '#3b82f6',
  'Retrieval History':     '#8b5cf6',
  'Retrieval History (H)': '#8b5cf6',
  'Semantic Relevance':    '#10b981',
  'Semantic Relevance (S)':'#10b981',
  'Relationship Strength': '#f59e0b',
  'Relationship Strength (Rel)': '#f59e0b',
};

export const MemoryFeatureBar: React.FC<MemoryFeatureBarProps> = ({
  label,
  value,
  isCalculated = false,
  color
}) => {
  const parsed = typeof value === 'number' ? value
    : typeof value === 'string' ? parseFloat(value)
    : NaN;

  const hasValue = !isNaN(parsed);
  const numValue = hasValue ? parsed : 0;
  const isPending = !isCalculated || !hasValue || numValue === 0;
  const barColor = color || featureColors[label] || '#0d9488';
  const pct = Math.min(100, Math.max(0, numValue * 100));

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-slate-600">{label}</span>
        {isPending ? (
          <span
            className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
            style={{
              background: '#fef9c3',
              color: '#92400e',
              border: '1px solid #fde68a'
            }}
          >
            {hasValue && numValue === 0 ? '0.00 · Cold Start' : 'N/A'}
          </span>
        ) : (
          <span
            className="text-[11px] font-bold font-mono"
            style={{ color: barColor }}
          >
            {numValue.toFixed(2)}
          </span>
        )}
      </div>

      <div className="feature-bar-track">
        {isPending ? (
          <div
            className="h-full rounded-full"
            style={{ width: '100%', background: 'linear-gradient(90deg, #f1f5f9, #e2e8f0, #f1f5f9)', backgroundSize: '200% 100%' }}
          />
        ) : (
          <div
            className="feature-bar-fill"
            style={{ width: `${pct}%`, background: barColor, opacity: 0.85 }}
          />
        )}
      </div>
    </div>
  );
};

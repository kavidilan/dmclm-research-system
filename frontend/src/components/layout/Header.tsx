import React from 'react';
import { RefreshCw } from 'lucide-react';

interface HeaderProps {
  title: string;
  subtitle?: string;
  onRefresh: () => void;
  isRefreshing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  onRefresh,
  isRefreshing = false
}) => {
  return (
    <header className="h-14 px-6 flex items-center justify-between sticky top-0 z-20"
      style={{ background: '#fff', borderBottom: '1px solid var(--line)' }}>
      <div className="min-w-0">
        <h1 className="text-[14px] font-bold text-slate-900 leading-tight tracking-tight truncate">
          {title}
        </h1>
        {subtitle && (
          <p className="text-[11px] text-slate-500 mt-0.5 truncate hidden md:block">{subtitle}</p>
        )}
      </div>

      <div className="flex items-center gap-2 flex-shrink-0 ml-4">
        {/* Refresh */}
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="btn btn-secondary h-8 px-3"
          title="Refresh data"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline text-[12px]">Refresh</span>
        </button>
      </div>
    </header>
  );
};

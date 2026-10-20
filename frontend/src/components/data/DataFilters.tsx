import React from 'react';
import { Search, Filter, RefreshCw, Database } from 'lucide-react';

interface DataFiltersProps {
  search: string;
  onSearchChange: (val: string) => void;
  appFilter?: string;
  onAppFilterChange?: (val: string) => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
  totalRecords: number;
}

export const DataFilters: React.FC<DataFiltersProps> = ({
  search,
  onSearchChange,
  appFilter,
  onAppFilterChange,
  onRefresh,
  isRefreshing = false,
  totalRecords
}) => {
  return (
    <div className="card p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
      <div className="flex flex-1 items-center gap-2.5">
        {/* Search */}
        <div className="relative flex-1 max-w-lg">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search by keyword, window title, URL, or application…"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="search-input"
            style={{ paddingLeft: '2.25rem' }}
          />
        </div>

        {/* Application filter */}
        {onAppFilterChange && (
          <div className="relative">
            <Filter className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Filter app…"
              value={appFilter || ''}
              onChange={(e) => onAppFilterChange(e.target.value)}
              className="search-input w-40"
              style={{ paddingLeft: '2rem' }}
            />
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 text-[11px]">
        <div className="flex items-center gap-1.5 text-slate-500">
          <Database className="w-3.5 h-3.5 text-slate-400" />
          <span>SQLite total:</span>
          <strong className="font-mono text-slate-900">{totalRecords.toLocaleString()}</strong>
        </div>

        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="btn btn-secondary h-8 px-3"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-teal-600' : ''}`} />
          <span>Sync</span>
        </button>
      </div>
    </div>
  );
};

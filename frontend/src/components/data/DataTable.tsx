import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Column<T> {
  key: string;
  header: string;
  render?: (item: T) => React.ReactNode;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  loading?: boolean;
  emptyMessage?: string;
  page: number;
  totalPages: number;
  totalItems: number;
  onPageChange: (newPage: number) => void;
  onRowClick?: (item: T) => void;
}

export function DataTable<T extends Record<string, any>>({
  columns,
  data,
  loading = false,
  emptyMessage = 'No records found matching query.',
  page,
  totalPages,
  totalItems,
  onPageChange,
  onRowClick
}: DataTableProps<T>) {
  const SKELETONS = 7;

  return (
    <div className="card overflow-hidden flex flex-col">
      <div className="overflow-x-auto flex-1">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`whitespace-nowrap ${
                    col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                  } ${col.width || ''}`}
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {loading ? (
              Array.from({ length: SKELETONS }).map((_, ri) => (
                <tr key={ri}>
                  {columns.map((col, ci) => (
                    <td key={col.key} className="py-3 px-4">
                      <div
                        className="skeleton h-3 rounded"
                        style={{ width: ci === 0 ? '90px' : ci === columns.length - 1 ? '60px' : '120px' }}
                      />
                    </td>
                  ))}
                </tr>
              ))
            ) : data.length === 0 ? (
              <tr>
                <td colSpan={columns.length}>
                  <div className="empty-state py-12">
                    <p className="font-medium text-slate-500">{emptyMessage}</p>
                    <p className="text-slate-400 mt-1 text-[11px]">Adjust search or filter criteria.</p>
                  </div>
                </td>
              </tr>
            ) : (
              data.map((item, index) => (
                <tr
                  key={
                    item.id ||
                    item.event_id ||
                    item.browser_event_id ||
                    item.interaction_id ||
                    item.screenshot_id ||
                    item.session_id ||
                    index
                  }
                  onClick={() => onRowClick && onRowClick(item)}
                  className={onRowClick ? 'cursor-pointer' : ''}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={
                        col.align === 'right'
                          ? 'text-right'
                          : col.align === 'center'
                          ? 'text-center'
                          : ''
                      }
                    >
                      {col.render ? col.render(item) : item[col.key] ?? '—'}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div
        className="flex items-center justify-between px-5 py-3 border-t border-slate-100 text-[11px] text-slate-500"
        style={{ background: '#f8fafc' }}
      >
        <div>
          Page <span className="font-bold font-mono text-slate-800">{page}</span> of{' '}
          <span className="font-bold font-mono text-slate-800">{Math.max(1, totalPages)}</span>
          <span className="ml-2 text-slate-400">({totalItems.toLocaleString()} total records)</span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page <= 1 || loading}
            className="btn btn-secondary h-7 px-2.5 text-[11px]"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
            <span>Prev</span>
          </button>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages || loading}
            className="btn btn-secondary h-7 px-2.5 text-[11px]"
          >
            <span>Next</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

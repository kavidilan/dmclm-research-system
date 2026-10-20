import React from 'react';

interface PageContainerProps {
  children: React.ReactNode;
}

export const PageContainer: React.FC<PageContainerProps> = ({ children }) => {
  return (
    <div className="flex-1 flex flex-col" style={{ background: 'var(--page)' }}>
      <main className="flex-1 p-6 max-w-[1440px] w-full mx-auto">
        <div className="space-y-5 fade-in">
          {children}
        </div>
      </main>

      <footer className="py-3 px-6 text-[11px] flex flex-col sm:flex-row items-center justify-between gap-1 text-slate-400"
        style={{ borderTop: '1px solid var(--line)' }}>
        <span>
          <span className="font-semibold text-slate-500">Project:</span>{' '}
          Adaptive Privacy-Preserving Cognitive Memory System for Personalized Desktop Intelligence
        </span>
        <span className="font-mono text-slate-400">
          Member 4 — DMCLM — Adaptive Long-Term Memory Consolidation
        </span>
      </footer>
    </div>
  );
};

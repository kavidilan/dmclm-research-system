import React, { useState, useEffect } from 'react';
import {
  Brain,
  Archive,
  GitMerge
} from 'lucide-react';
import { Memory } from './Memory';
import { LifecycleManagement } from './LifecycleManagement';
import { Consolidation } from './Consolidation';

export type MemorySubTab = 'records' | 'lifecycle' | 'consolidation';

interface MemoryHubProps {
  initialSubTab?: MemorySubTab;
  initialLifecycleFilter?: string;
}

export const MemoryHub: React.FC<MemoryHubProps> = ({
  initialSubTab = 'records',
  initialLifecycleFilter = ''
}) => {
  const [activeSubTab, setActiveSubTab] = useState<MemorySubTab>(initialSubTab);

  useEffect(() => {
    if (initialSubTab) {
      setActiveSubTab(initialSubTab);
    }
  }, [initialSubTab]);

  const tabs: { id: MemorySubTab; label: string; icon: React.ElementType; badge?: string }[] = [
    { id: 'records', label: 'Memory Records', icon: Brain },
    { id: 'lifecycle', label: 'Lifecycle Manager', icon: Archive },
    { id: 'consolidation', label: 'Consolidation & LTM', icon: GitMerge },
  ];

  return (
    <div className="space-y-5">
      {/* ── Sub-navigation Bar ─────────────────────────────────────────── */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-2 shadow-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeSubTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-[13px] font-medium transition-all ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 shadow-xs border border-blue-200/60 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
                {tab.badge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded font-mono font-bold ${
                      isActive ? 'bg-blue-200/80 text-blue-900' : 'bg-slate-200 text-slate-600'
                    }`}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        <div className="text-[12px] text-slate-400 px-3 hidden lg:block font-mono">
          DMCLM Memory Core
        </div>
      </div>

      {/* ── Active Sub-View ───────────────────────────────────────────── */}
      <div>
        {activeSubTab === 'records' && (
          <Memory initialLifecycleFilter={initialLifecycleFilter} />
        )}
        {activeSubTab === 'lifecycle' && <LifecycleManagement />}
        {activeSubTab === 'consolidation' && <Consolidation />}
      </div>
    </div>
  );
};

export default MemoryHub;

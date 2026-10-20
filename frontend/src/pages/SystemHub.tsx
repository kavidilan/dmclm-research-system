import React, { useState, useEffect } from 'react';
import {
  FlaskConical,
  Database
} from 'lucide-react';
import { Preprocessing } from './Preprocessing';
import { RawData } from './RawData';

export type SystemSubTab = 'preprocessing' | 'raw-data';

interface SystemHubProps {
  initialSubTab?: SystemSubTab;
}

export const SystemHub: React.FC<SystemHubProps> = ({
  initialSubTab = 'preprocessing'
}) => {
  const [activeSubTab, setActiveSubTab] = useState<SystemSubTab>(initialSubTab);

  useEffect(() => {
    if (initialSubTab) {
      setActiveSubTab(initialSubTab);
    }
  }, [initialSubTab]);

  const tabs: { id: SystemSubTab; label: string; icon: React.ElementType }[] = [
    { id: 'preprocessing', label: 'Data Preprocessing', icon: FlaskConical },
    { id: 'raw-data', label: 'Raw Collector Tables', icon: Database },
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
              </button>
            );
          })}
        </div>

        <div className="text-[12px] text-slate-400 px-3 hidden lg:block font-mono">Collector Data</div>
      </div>

      {/* ── Active Sub-View ───────────────────────────────────────────── */}
      <div>
        {activeSubTab === 'preprocessing' && <Preprocessing />}
        {activeSubTab === 'raw-data' && <RawData />}
      </div>
    </div>
  );
};

export default SystemHub;

import React from 'react';
import {
  LayoutDashboard,
  Brain,
  Server
} from 'lucide-react';

interface SidebarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
}

const navItems = [
  {
    id: 'dashboard',
    label: 'Overview',
    description: 'System overview & activity',
    icon: LayoutDashboard,
  },
  {
    id: 'memories',
    label: 'Memory Hub',
    description: 'Records, lifecycle & LTM',
    icon: Brain,
  },
  {
    id: 'system',
    label: 'Collector Data',
    description: 'Preprocessing & source tables',
    icon: Server,
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, onTabChange }) => {
  return (
    <aside
      className="w-64 flex flex-col h-screen fixed left-0 top-0 z-30 select-none shadow-lg"
      style={{
        background: 'var(--sidebar)',
        borderRight: '1px solid rgba(255,255,255,0.08)'
      }}
    >
      {/* Logo / project identity */}
      <div className="px-5 pt-6 pb-5" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center font-bold text-white shadow-xs flex-shrink-0"
            style={{ background: 'var(--accent)', fontSize: 13 }}
          >
            DM
          </div>
          <div>
            <div className="text-white font-bold text-[15px] leading-tight tracking-tight">DMCLM Core</div>
            <div className="text-[11px] mt-0.5 text-slate-400 font-medium">
              Member 4 Research System
            </div>
          </div>
        </div>
      </div>

      {/* Main Navigation (Clean 3-Tab Streamlined Nav) */}
      <nav className="flex-1 px-3 py-5 overflow-y-auto space-y-1.5">
        <div className="sidebar-section-label px-3 mb-2 text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
          Main Views
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`w-full text-left p-3 rounded-xl transition-all flex items-start gap-3.5 group ${
                isActive
                  ? 'bg-blue-600/20 text-white border border-blue-500/30 shadow-xs'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-white/5 border border-transparent'
              }`}
            >
              <div
                className={`p-2 rounded-lg flex-shrink-0 transition-colors ${
                  isActive
                    ? 'bg-blue-500 text-white shadow-xs'
                    : 'bg-white/5 text-slate-400 group-hover:text-slate-200 group-hover:bg-white/10'
                }`}
              >
                <Icon className="w-4 h-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div
                  className={`text-[13px] font-semibold leading-tight ${
                    isActive ? 'text-white' : 'text-slate-300 group-hover:text-white'
                  }`}
                >
                  {item.label}
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5 leading-snug">
                  {item.description}
                </div>
              </div>
            </button>
          );
        })}
      </nav>

    </aside>
  );
};

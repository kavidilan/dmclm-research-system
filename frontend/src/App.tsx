import React, { useState } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { PageContainer } from './components/layout/PageContainer';
import { Dashboard } from './pages/Dashboard';
import { MemoryHub, MemorySubTab } from './pages/MemoryHub';
import { SystemHub, SystemSubTab } from './pages/SystemHub';
import { useCollectorData } from './hooks/useCollectorData';
import { useMemoryData } from './hooks/useMemoryData';
import { MemoryRecord } from './types/memory';
import { MemoryDetailDrawer } from './components/memory/MemoryDetailDrawer';

export function App() {
  const [currentTab, setCurrentTab] = useState<string>('dashboard');
  const [memorySubTab, setMemorySubTab] = useState<MemorySubTab>('records');
  const [systemSubTab, setSystemSubTab] = useState<SystemSubTab>('preprocessing');
  const [selectedMemory, setSelectedMemory] = useState<MemoryRecord | null>(null);
  const [initialLifecycleFilter, setInitialLifecycleFilter] = useState<string>('');

  const { stats, timeline, loading: collectorLoading, refetch } = useCollectorData();
  const { memories } = useMemoryData(1, 10);

  const getPageTitle = (): { title: string; subtitle?: string } => {
    switch (currentTab) {
      case 'dashboard':
        return {
          title: 'System Overview',
          subtitle: 'DMCLM adaptive long-term memory consolidation dashboard.'
        };
      case 'memories':
        return {
          title: 'Memory Hub',
          subtitle: 'Unified episodic records, lifecycle actions & LTM consolidation.'
        };
      case 'system':
        return {
          title: 'Collector Data',
          subtitle: 'Data preprocessing and raw collector tables.'
        };
      default:
        return { title: 'DMCLM Research System' };
    }
  };

  const pageInfo = getPageTitle();

  return (
    <div className="min-h-screen flex flex-col font-sans" style={{ background: 'var(--page)' }}>
      {/* Fixed Left Sidebar */}
      <Sidebar
        currentTab={currentTab}
        onTabChange={(tab) => {
          setCurrentTab(tab);
          if (tab === 'memories') setMemorySubTab('records');
          if (tab === 'system') setSystemSubTab('preprocessing');
        }}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col pl-64">
        {/* Sticky Header */}
        <Header
          title={pageInfo.title}
          subtitle={pageInfo.subtitle}
          onRefresh={refetch}
          isRefreshing={collectorLoading}
        />

        {/* Page Content */}
        <PageContainer>
          {currentTab === 'dashboard' && (
            <Dashboard
              stats={stats}
              timeline={timeline}
              recentMemories={memories}
              loading={collectorLoading}
              onSelectMemory={setSelectedMemory}
              onNavigateToMemory={() => {
                setMemorySubTab('records');
                setCurrentTab('memories');
              }}
              onSelectLifecycle={(filter) => {
                setInitialLifecycleFilter(filter);
                setMemorySubTab('lifecycle');
                setCurrentTab('memories');
              }}
              onNavigateToLifecycle={() => {
                setMemorySubTab('lifecycle');
                setCurrentTab('memories');
              }}
            />
          )}

          {currentTab === 'memories' && (
            <MemoryHub
              initialSubTab={memorySubTab}
              initialLifecycleFilter={initialLifecycleFilter}
            />
          )}

          {currentTab === 'system' && (
            <SystemHub
              initialSubTab={systemSubTab}
            />
          )}
        </PageContainer>
      </div>

      {/* Global Memory Detail Drawer */}
      <MemoryDetailDrawer
        memory={selectedMemory}
        onClose={() => setSelectedMemory(null)}
      />
    </div>
  );
}

export default App;

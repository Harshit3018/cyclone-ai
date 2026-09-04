import { BrowserRouter, Routes, Route, NavLink, Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import {
  Activity, BarChart3, Cloud, Database, Eye, Globe2,
  Home, Info, Layers, LineChart, AlertTriangle, Radio, Settings
} from 'lucide-react';
import { getSystemStatus } from './api/client';
import Dashboard from './pages/Dashboard';
import CycloneDetail from './pages/CycloneDetail';
import Predictions from './pages/Predictions';
import HistoricalCyclones from './pages/HistoricalCyclones';
import AIAnalysis from './pages/AIAnalysis';
import DataExplorer from './pages/DataExplorer';
import ModelPerformance from './pages/ModelPerformance';
import DatasetsPage from './pages/DatasetsPage';
import About from './pages/About';
import type { SystemStatus } from './types';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/predictions', label: 'Predictions', icon: Activity },
  { path: '/historical', label: 'Historical', icon: Globe2 },
  { path: '/ai-analysis', label: 'AI Analysis', icon: Eye },
  { path: '/data-explorer', label: 'Data Explorer', icon: Database },
  { path: '/model-performance', label: 'Models', icon: BarChart3 },
  { path: '/datasets', label: 'Datasets', icon: Layers },
  { path: '/about', label: 'About', icon: Info },
];

export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    getSystemStatus().then(r => setStatus(r.data)).catch(() => {});
    const interval = setInterval(() => {
      getSystemStatus().then(r => setStatus(r.data)).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const dataMode = status?.data_mode || 'DEMO';

  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <aside className={`${sidebarOpen ? 'w-60' : 'w-16'} flex-shrink-0 bg-white border-r border-slate-200 flex flex-col transition-all duration-300 z-50`}>
          {/* Logo */}
          <div className="p-4 border-b border-slate-200">
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center">
                <Cloud className="w-5 h-5 text-brand-primary" />
              </div>
              {sidebarOpen && (
                <div>
                  <h1 className="text-sm font-bold text-slate-900 tracking-wide">CYCLONE-AI</h1>
                  <p className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">Intelligence Platform</p>
                </div>
              )}
            </Link>
          </div>

          {/* Data mode badge */}
          {sidebarOpen && (
            <div className="px-4 py-3">
              <div className={`text-[10px] font-mono font-medium px-2 py-1 rounded text-center ${
                dataMode === 'DEMO' ? 'bg-amber-50 text-amber-600 border border-amber-200' :
                dataMode === 'LIVE' ? 'bg-green-50 text-green-600 border border-green-200' :
                'bg-blue-50 text-brand-primary border border-blue-200'
              }`}>
                {dataMode} MODE
              </div>
            </div>
          )}

          {/* Navigation */}
          <nav className="flex-1 py-2 overflow-y-auto">
            {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-4 py-2 mx-2 rounded-md text-[13px] font-medium transition-all ${
                    isActive
                      ? 'bg-blue-50 text-brand-primary border-l-[3px] border-brand-primary'
                      : 'text-slate-500 hover:text-slate-900 hover:bg-slate-50 border-l-[3px] border-transparent'
                  }`
                }
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span>{label}</span>}
              </NavLink>
            ))}
          </nav>

          {/* Status footer */}
          {sidebarOpen && status && (
            <div className="p-4 border-t border-slate-200 text-[11px] text-slate-500">
              <div className="flex items-center gap-2 mb-1.5 font-medium">
                <div className={`w-1.5 h-1.5 rounded-full ${status.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`} />
                <span>System {status.status === 'ok' ? 'Online' : 'Offline'}</span>
              </div>
              <div className="font-mono">v{status.version} • {status.environment}</div>
            </div>
          )}

          {/* Toggle */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-3 border-t border-slate-200 text-slate-500 hover:text-slate-700 hover:bg-slate-50 text-[11px] font-medium text-left flex items-center justify-center transition-colors"
          >
            {sidebarOpen ? '◀ Collapse' : '▶'}
          </button>
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto bg-base text-slate-900">
          {/* Top bar */}
          <header className="sticky top-0 z-40 bg-white/95 backdrop-blur border-b border-slate-200 px-8 py-4 flex items-center justify-between">
            <div>
              <h2 className="text-[19px] font-semibold text-slate-900 tracking-tight">Tropical Cyclone Intelligence</h2>
              <p className="text-[11px] font-medium text-slate-500 uppercase tracking-wider mt-0.5">Multi-Source AI Decision Support Platform</p>
            </div>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 text-[11px] font-medium font-mono">
                <Radio className={`w-3.5 h-3.5 ${dataMode === 'LIVE' ? 'text-green-500 animate-pulse' : 'text-amber-500'}`} />
                <span className="text-slate-600 uppercase">
                  {dataMode === 'DEMO' ? 'Demo Data Active' : dataMode === 'LIVE' ? 'Live Data' : 'Historical Data'}
                </span>
              </div>
              {status && (
                <div className="text-[11px] text-slate-500 font-mono bg-slate-50 px-3 py-1 rounded border border-slate-200">
                  <span className="text-slate-700 font-semibold">{status.active_cyclones}</span> active • <span className="text-slate-700 font-semibold">{status.total_cyclones}</span> total
                </div>
              )}
            </div>
          </header>

          {/* Disclaimer banner */}
          <div className="mx-8 mt-6 mb-2">
            <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded-md px-4 py-2.5 flex items-start sm:items-center gap-3">
              <AlertTriangle className="w-4 h-4 text-[#92400E] flex-shrink-0 mt-0.5 sm:mt-0" />
              <p className="text-[12px] font-medium text-[#92400E] leading-relaxed">
                AI-generated predictions for research and decision-support purposes.
                This system is not an official warning system and must not replace
                official advisories issued by the India Meteorological Department or other authorized agencies.
              </p>
            </div>
          </div>

          {/* Page content */}
          <div className="p-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/cyclone/:id" element={<CycloneDetail />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/historical" element={<HistoricalCyclones />} />
              <Route path="/ai-analysis" element={<AIAnalysis />} />
              <Route path="/data-explorer" element={<DataExplorer />} />
              <Route path="/model-performance" element={<ModelPerformance />} />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

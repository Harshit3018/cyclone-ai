import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity, AlertTriangle, Cloud, Eye, MapPin, Wind,
  Gauge, TrendingUp, Clock, ChevronRight, Waves
} from 'lucide-react';
import { getCyclones, getAlerts, getSystemStatus } from '../api/client';
import CycloneMap from '../maps/CycloneMap';
import type { Cyclone, Alert, SystemStatus } from '../types';

export default function Dashboard() {
  const [cyclones, setCyclones] = useState<Cyclone[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getCyclones().then(r => setCyclones(r.data.cyclones || [])),
      getAlerts().then(r => setAlerts(r.data.alerts || [])),
      getSystemStatus().then(r => setStatus(r.data)),
    ]).catch(err => console.error('Dashboard load error:', err))
      .finally(() => setLoading(false));
  }, []);

  const activeCyclones = cyclones.filter(c => c.is_active);

  // Alert severity, ascending. The old "High Risk Systems" card counted only
  // HIGH/EXTREME RISK alerts, which the risk engine emits but the demo seeder never
  // does — so it always read 0 next to a sublabel saying "2 total alerts". Report the
  // alert count and the highest severity actually present instead.
  const SEVERITY = ['NORMAL', 'WATCH', 'ADVISORY', 'HIGH RISK', 'EXTREME RISK'];
  const topSeverityIdx = alerts.reduce((max, a) => {
    const i = SEVERITY.indexOf(a.alert_level);
    return i > max ? i : max;
  }, -1);
  const topSeverity = topSeverityIdx >= 0 ? SEVERITY[topSeverityIdx] : 'NONE';
  const severityColor = topSeverityIdx >= 3 ? 'red' : topSeverityIdx >= 2 ? 'amber' : 'green';

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Cloud className="w-12 h-12 text-brand-primary animate-pulse mx-auto mb-3 opacity-80" />
          <p className="text-[13px] text-slate-500 font-medium">INITIALIZING CYCLONE INTELLIGENCE...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in flex flex-col min-h-full">
      {/* Top KPI Ribbon - Flat, borderless, separated by dividers */}
      <div className="grid grid-cols-2 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-slate-200 border-b border-slate-200 bg-white">
        <StatItem label="Active Cyclones" value={activeCyclones.length} sub={`${status?.data_mode || 'DEMO'} DATA`} color="cyan" />
        <StatItem label="Cyclones Analyzed" value={cyclones.length} sub="IN DATABASE" color="blue" />
        <StatItem label="Active Alerts" value={alerts.length} sub={`HIGHEST: ${topSeverity}`} color={severityColor} />
        <StatItem label="System Uptime" value={status ? `${Math.round(status.uptime_seconds / 60)}m` : '--'} sub={`v${status?.version || '0.1.0'}`} color="purple" />
      </div>

      {/* Main Operational Canvas */}
      <div className="flex-1 grid grid-cols-1 xl:grid-cols-4 bg-base min-h-[600px]">
        {/* Map Area (3 columns) */}
        <div className="xl:col-span-3 border-b xl:border-b-0 xl:border-r border-slate-200 flex flex-col relative">
          <div className="px-6 py-3 border-b border-slate-200 flex items-center justify-between bg-white">
            <h3 className="text-[11px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2">
              <MapPin className="w-3.5 h-3.5 text-brand-primary" />
              North Indian Ocean Monitor
            </h3>
            <span className="text-[10px] text-amber-700 font-mono tracking-wider bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-sm">
              {status?.data_mode || 'DEMO'} TELEMETRY
            </span>
          </div>
          
          <div className="flex-1 relative bg-base">
            <CycloneMap cyclones={cyclones} />
            
            {/* Active System Briefing Overlay (The "Hero") */}
            {activeCyclones.length > 0 && (
              <div className="absolute top-6 right-6 z-[1000] bg-white/95 backdrop-blur-md border border-slate-200 p-5 w-72 shadow-lg rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-red-600 animate-pulse shadow-[0_0_8px_rgba(220,38,38,0.5)]" />
                  <span className="text-[10px] font-bold text-red-600 uppercase tracking-widest">Active System</span>
                </div>
                <h2 className="text-2xl font-black text-slate-900 tracking-tight leading-none mb-4 uppercase">
                  {activeCyclones[0].name.replace('DEMO_', '')}
                </h2>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">Wind Max</div>
                    <div className="text-xl font-mono text-slate-800 mt-0.5">
                      {activeCyclones[0].peak_wind_kph?.toFixed(0) || '--'} <span className="text-[10px] text-slate-500">km/h</span>
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">Pressure Min</div>
                    <div className="text-xl font-mono text-slate-800 mt-0.5">
                      {activeCyclones[0].min_pressure_hpa?.toFixed(0) || '--'} <span className="text-[10px] text-slate-500">hPa</span>
                    </div>
                  </div>
                </div>
                <div className="text-[11px] font-semibold text-amber-600 border-t border-slate-100 pt-3 uppercase tracking-wide">
                  {activeCyclones[0].peak_category}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Intelligence Data Sidebar (1 column) */}
        <div className="flex flex-col bg-white border-l border-slate-200 overflow-hidden">
          {/* Alerts Module */}
          <div className="px-5 py-3 border-b border-slate-200 bg-slate-50">
            <h3 className="text-[11px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              Intelligence Alerts
            </h3>
          </div>
          <div className="divide-y divide-slate-100 border-b border-slate-200 max-h-[300px] overflow-y-auto">
            {alerts.length === 0 ? (
              <div className="p-4 text-[11px] text-slate-500 font-mono">No active alerts.</div>
            ) : (
              alerts.map(a => (
                <div key={a.id} className="px-5 py-3 flex items-start gap-3 hover:bg-slate-50 transition-colors">
                  <div className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    a.alert_level === 'EXTREME RISK' ? 'bg-red-600' :
                    a.alert_level === 'HIGH RISK' ? 'bg-orange-500' :
                    a.alert_level === 'ADVISORY' ? 'bg-amber-500' : 'bg-brand-primary'
                  }`} />
                  <div>
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className={`text-[9px] font-bold tracking-widest uppercase ${
                        a.alert_level.includes('RISK') ? 'text-red-600' : 'text-amber-600'
                      }`}>{a.alert_level}</span>
                      <span className="text-[9px] font-mono text-slate-500 uppercase">{a.data_type}</span>
                    </div>
                    <div className="text-[12px] text-slate-700 leading-snug">{a.message}</div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Tracked Systems Module */}
          <div className="px-5 py-3 border-y border-slate-200 bg-slate-50 mt-4 xl:mt-0 xl:border-t-0">
            <h3 className="text-[11px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-brand-secondary" />
              Tracked Systems
            </h3>
          </div>
          <div className="divide-y divide-slate-100 overflow-y-auto flex-1 bg-white">
            {cyclones.length === 0 ? (
              <div className="p-4 text-[11px] text-slate-500 font-mono">Database empty.</div>
            ) : (
              cyclones.map(c => (
                <Link
                  key={c.id}
                  to={`/cyclone/${c.id}`}
                  className="flex flex-col px-5 py-3.5 hover:bg-slate-50 transition-colors group relative"
                >
                  {/* Subtle left border for active systems */}
                  {c.is_active && <div className="absolute left-0 top-0 bottom-0 w-1 bg-green-500" />}
                  
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[13px] font-bold text-slate-900 uppercase tracking-wide">
                      {c.name.replace('DEMO_', '')}
                    </span>
                    <span className={`text-[9px] font-bold tracking-widest px-1.5 py-0.5 rounded ${
                      c.is_active ? 'bg-green-50 text-green-700' : 'bg-slate-100 text-slate-600'
                    }`}>
                      {c.is_active ? 'ACTIVE' : 'HISTORICAL'}
                    </span>
                  </div>
                  
                  <div className="text-[10px] text-brand-primary font-medium uppercase tracking-wide mb-2">
                    {c.peak_category}
                  </div>

                  <div className="flex items-center gap-4 text-[10px] text-slate-500 font-mono">
                    <span className="flex items-center gap-1.5">
                      <Wind className="w-3 h-3 text-slate-400" />
                      {c.peak_wind_kph?.toFixed(0) || '--'} km/h
                    </span>
                    <span className="flex items-center gap-1.5">
                      <Gauge className="w-3 h-3 text-slate-400" />
                      {c.min_pressure_hpa?.toFixed(0) || '--'} hPa
                    </span>
                  </div>
                </Link>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Replaced StatCard with a flat, borderless StatItem for the KPI ribbon
function StatItem({ label, value, sub, color }: {
  label: string; value: string | number; sub: string; color: string;
}) {
  const colorClasses: Record<string, string> = {
    cyan: 'text-brand-secondary',
    blue: 'text-brand-primary',
    red: 'text-red-600',
    amber: 'text-amber-600',
    green: 'text-green-600',
    purple: 'text-brand-primary',
  };
  const valueColor = colorClasses[color] || colorClasses.blue;

  return (
    <div className="px-6 py-4 flex flex-col justify-center">
      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">{label}</span>
      <div className={`text-3xl font-black tracking-tighter leading-none mb-1 ${valueColor}`}>
        {value}
      </div>
      <div className="text-[9px] font-mono text-slate-500 uppercase tracking-wider">{sub}</div>
    </div>
  );
}

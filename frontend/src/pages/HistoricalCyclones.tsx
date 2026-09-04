import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Globe2, Wind, Gauge, ChevronRight, Filter } from 'lucide-react';
import { getCyclones } from '../api/client';
import type { Cyclone } from '../types';

export default function HistoricalCyclones() {
  const [cyclones, setCyclones] = useState<Cyclone[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    getCyclones({ limit: 100 }).then(r => setCyclones(r.data.cyclones || []))
      .catch(console.error).finally(() => setLoading(false));
  }, []);

  const filtered = cyclones.filter(c =>
    c.name.toLowerCase().includes(filter.toLowerCase()) ||
    c.peak_category?.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Historical Cyclones</h2>
        <p className="text-sm text-slate-600 mt-1">Browse and analyze past tropical cyclone records</p>
      </div>

      <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4 flex items-center gap-3">
        <Filter className="w-4 h-4 text-slate-600" />
        <input type="text" placeholder="Filter by name or category..." value={filter} onChange={e => setFilter(e.target.value)}
          className="flex-1 bg-transparent text-sm text-slate-900 outline-none placeholder-gray-500" />
        <span className="text-xs text-gray-500">{filtered.length} cyclones</span>
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-600">Loading...</div>
      ) : (
        <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-600 border-b border-slate-200">
                <th className="text-left py-3 px-4">Name</th>
                <th className="text-left py-3 px-4">Season</th>
                <th className="text-left py-3 px-4">Basin</th>
                <th className="text-left py-3 px-4">Category</th>
                <th className="text-right py-3 px-4">Peak Wind</th>
                <th className="text-right py-3 px-4">Min Pressure</th>
                <th className="text-center py-3 px-4">Data</th>
                <th className="py-3 px-4"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(c => (
                <tr key={c.id} className="border-b border-slate-200/50 hover:bg-white transition-colors">
                  <td className="py-3 px-4 font-medium text-slate-900">{c.name.replace('DEMO_', '')}</td>
                  <td className="py-3 px-4 text-slate-600">{c.season}</td>
                  <td className="py-3 px-4 text-slate-600">{c.sub_basin || c.basin}</td>
                  <td className="py-3 px-4"><span className="text-xs bg-brand-primary/100/10 text-blue-600 px-2 py-0.5 rounded">{c.peak_category}</span></td>
                  <td className="py-3 px-4 text-right text-slate-900 flex items-center justify-end gap-1">
                    <Wind className="w-3 h-3 text-cyan-600" />{c.peak_wind_kph?.toFixed(0)} km/h
                  </td>
                  <td className="py-3 px-4 text-right text-slate-900">{c.min_pressure_hpa?.toFixed(0)} hPa</td>
                  <td className="py-3 px-4 text-center">
                    <span className="text-[10px] bg-amber-500/20 text-amber-600 px-2 py-0.5 rounded font-mono">{c.data_type}</span>
                  </td>
                  <td className="py-3 px-4">
                    <Link to={`/cyclone/${c.id}`} className="text-blue-600 hover:text-blue-300">
                      <ChevronRight className="w-4 h-4" />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { Database, Clock, MapPin } from 'lucide-react';
import { getObservations, getCyclones } from '../api/client';

export default function DataExplorer() {
  const [observations, setObservations] = useState<any[]>([]);
  const [cyclones, setCyclones] = useState<any[]>([]);
  const [selectedCyclone, setSelectedCyclone] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getCyclones().then(r => setCyclones(r.data.cyclones || [])).catch(console.error);
    getObservations().then(r => setObservations(r.data.observations || []))
      .catch(console.error).finally(() => setLoading(false));
  }, []);

  const loadObs = (cid: string) => {
    setSelectedCyclone(cid);
    setLoading(true);
    getObservations(cid || undefined).then(r => setObservations(r.data.observations || []))
      .catch(console.error).finally(() => setLoading(false));
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Data Explorer</h2>
        <p className="text-sm text-slate-600 mt-1">Inspect satellite observations, track points, and environmental data</p>
      </div>

      <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4 flex items-center gap-4">
        <Database className="w-4 h-4 text-blue-600" />
        <select value={selectedCyclone} onChange={e => loadObs(e.target.value)}
          className="bg-white text-sm text-slate-900 border border-slate-200 rounded px-3 py-1.5 flex-1">
          <option value="">All Cyclones</option>
          {cyclones.map(c => <option key={c.id} value={c.id}>{c.name.replace('DEMO_', '')} ({c.season})</option>)}
        </select>
        <span className="text-xs text-gray-500">{observations.length} observations</span>
      </div>

      <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900">Track Point Observations</h3>
          <span className="text-[10px] bg-amber-500/20 text-amber-600 px-2 py-0.5 rounded font-mono">DEMO DATA</span>
        </div>
        {loading ? (
          <div className="p-8 text-center text-gray-500">Loading...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-600 border-b border-slate-200">
                  <th className="py-2 px-3 text-left">ID</th>
                  <th className="py-2 px-3 text-left">Cyclone</th>
                  <th className="py-2 px-3 text-left">Timestamp</th>
                  <th className="py-2 px-3 text-right">Lat</th>
                  <th className="py-2 px-3 text-right">Lon</th>
                  <th className="py-2 px-3 text-right">Wind (kt)</th>
                  <th className="py-2 px-3 text-right">Pressure</th>
                  <th className="py-2 px-3 text-left">Category</th>
                  <th className="py-2 px-3 text-center">Type</th>
                </tr>
              </thead>
              <tbody>
                {observations.map(o => (
                  <tr key={o.id} className="border-b border-slate-200/30 hover:bg-white">
                    <td className="py-2 px-3 font-mono text-gray-500">{o.id}</td>
                    <td className="py-2 px-3 text-blue-600">{o.cyclone_id?.replace('DEMO_NI_', '').slice(0, 12)}</td>
                    <td className="py-2 px-3 text-slate-500 flex items-center gap-1"><Clock className="w-3 h-3" />{o.timestamp?.slice(0, 16)}</td>
                    <td className="py-2 px-3 text-right text-slate-500">{o.latitude?.toFixed(2)}</td>
                    <td className="py-2 px-3 text-right text-slate-500">{o.longitude?.toFixed(2)}</td>
                    <td className="py-2 px-3 text-right text-slate-900">{o.wind_knots?.toFixed(1)}</td>
                    <td className="py-2 px-3 text-right text-slate-900">{o.pressure_hpa?.toFixed(1)}</td>
                    <td className="py-2 px-3 text-slate-600">{o.category}</td>
                    <td className="py-2 px-3 text-center"><span className="text-[9px] bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded">{o.data_type}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

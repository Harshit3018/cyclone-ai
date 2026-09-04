import { useEffect, useState } from 'react';
import { Layers, ExternalLink } from 'lucide-react';
import { getDatasets } from '../api/client';
import type { Dataset } from '../types';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDatasets().then(r => setDatasets(r.data.datasets || []))
      .catch(console.error).finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Datasets</h2>
        <p className="text-sm text-slate-600 mt-1">Data sources registered in the CYCLONE-AI system</p>
      </div>

      {loading ? <div className="text-center py-12 text-slate-600">Loading...</div> : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {datasets.map(d => (
            <div key={d.id} className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-blue-600" />
                  <h3 className="text-sm font-semibold text-slate-900">{d.name}</h3>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                  d.status === 'available' ? 'bg-green-500/20 text-green-600' : 'bg-gray-500/20 text-slate-600'
                }`}>{d.status?.toUpperCase()}</span>
              </div>
              <p className="text-xs text-slate-600 mb-3">{d.description}</p>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between"><span className="text-gray-500">Source</span><span className="text-slate-500">{d.source}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Type</span><span className="text-slate-500">{d.data_type}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">License</span><span className="text-slate-500">{d.license || 'See source'}</span></div>
                {d.source_url && (
                  <a href={d.source_url} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-1 text-blue-600 hover:text-blue-300 mt-2">
                    <ExternalLink className="w-3 h-3" /> Official Source
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

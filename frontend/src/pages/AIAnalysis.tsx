import { useState } from 'react';
import { Eye, Loader2, Zap, Brain } from 'lucide-react';
import { predictFull } from '../api/client';

export default function AIAnalysis() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [sampleIdx, setSampleIdx] = useState(0);

  const runAnalysis = async () => {
    setLoading(true);
    try { const r = await predictFull(sampleIdx); setResult(r.data); }
    catch(e) { console.error(e); }
    finally { setLoading(false); }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-slate-900">AI Analysis Pipeline</h2>
        <p className="text-sm text-slate-600 mt-1">Run the complete multi-task cyclone analysis on a satellite observation</p>
      </div>

      <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4 flex items-center gap-4">
        <span className="text-sm text-slate-600">Sample:</span>
        <input type="range" min={0} max={49} value={sampleIdx} onChange={e => setSampleIdx(Number(e.target.value))} className="flex-1 accent-blue-500" />
        <span className="text-sm font-mono text-slate-900">{sampleIdx}</span>
        <button onClick={runAnalysis} disabled={loading}
          className="bg-gradient-to-r from-blue-600 to-cyan-600 text-slate-900 px-6 py-2 rounded-lg text-sm font-medium hover:from-blue-500 hover:to-cyan-500 transition-all disabled:opacity-50 flex items-center gap-2">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
          Run Full Analysis
        </button>
      </div>

      {result && (
        <div className="space-y-4">
          <div className="text-xs text-gray-500">Total inference: {result.total_inference_time_ms?.toFixed(0)} ms</div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Detection */}
            <Section title="1. Detection" color="cyan">
              <KV label="Detected" value={result.detection?.cyclone_detected ? 'YES' : 'NO'} />
              <KV label="Confidence" value={`${(result.detection?.confidence * 100).toFixed(1)}%`} />
              <KV label="Center" value={`${result.detection?.center?.latitude}°N, ${result.detection?.center?.longitude}°E`} />
            </Section>

            {/* Classification */}
            <Section title="2. Classification" color="blue">
              <KV label="Class" value={result.classification?.class || '--'} />
              <KV label="Confidence" value={`${((result.classification?.confidence || 0) * 100).toFixed(1)}%`} />
              <div className="mt-1 space-y-0.5">
                {result.classification?.probabilities && Object.entries(result.classification.probabilities)
                  .sort(([,a]: any,[,b]: any) => b - a).slice(0, 3).map(([cls, p]: [string, any]) => (
                  <div key={cls} className="flex items-center gap-1">
                    <div className="flex-1 h-1 bg-slate-200 rounded"><div className="h-full bg-brand-primary/100 rounded" style={{width:`${p*100}%`}}/></div>
                    <span className="text-[9px] text-gray-500 w-24 truncate">{cls}</span>
                  </div>
                ))}
              </div>
            </Section>

            {/* Intensity */}
            <Section title="3. Intensity" color="amber">
              <KV label="Wind" value={`${result.intensity?.wind_kph?.toFixed(1)} km/h`} />
              <KV label="Pressure" value={`${result.intensity?.pressure_hpa?.toFixed(1)} hPa`} />
              <KV label="Category" value={result.intensity?.imd_category || '--'} />
            </Section>

            {/* Track */}
            <Section title="4. Track Forecast" color="green">
              {result.track_forecast?.forecast_points?.map((fp: any) => (
                <div key={fp.forecast_hour} className="flex gap-2 text-xs">
                  <span className="text-amber-600 font-mono">+{fp.forecast_hour}h</span>
                  <span className="text-slate-500">{fp.latitude.toFixed(2)}°N {fp.longitude.toFixed(2)}°E</span>
                </div>
              ))}
            </Section>

            {/* Rapid Intensification.
                `probability` is null: there is no trained intensity-forecast head. It used
                to be `(wind_kph - 100) / 150`, i.e. current intensity rescaled and labelled
                as the chance of future intensification. Rendering null as "0.0%" would claim
                a zero probability, so the unavailable reason is shown instead. */}
            <Section title="5. Rapid Intensification" color="red">
              <KV
                label="RI Probability"
                value={result.rapid_intensification?.probability !== null
                  && result.rapid_intensification?.probability !== undefined
                  ? `${(result.rapid_intensification.probability * 100).toFixed(1)}%`
                  : 'Not available'}
              />
              <KV label="Threshold" value={`${result.rapid_intensification?.threshold_knots_24h || 30} kt/24h`} />
              {result.rapid_intensification?.available === false && (
                <p className="text-[10px] text-slate-500 leading-snug mt-1">
                  {result.rapid_intensification.reason}
                </p>
              )}
            </Section>

            {/* Risk */}
            <Section title="6. Risk Assessment" color="orange">
              <KV label="Overall Risk" value={result.risk?.overall_risk != null
                ? `${(result.risk.overall_risk * 100).toFixed(0)}%` : 'Not available'} />
              <KV label="Alert Level" value={result.risk?.alert_level || '--'} />
              <KV label="Wind Risk" value={result.risk?.wind_risk != null
                ? `${(result.risk.wind_risk * 100).toFixed(0)}%` : 'Not available'} />
              <KV label="Coastal Risk" value={result.risk?.coastal_risk != null
                ? `${(result.risk.coastal_risk * 100).toFixed(0)}%` : 'Not available'} />
              {result.risk?.components_unavailable?.length > 0 && (
                <p className="text-[10px] text-slate-500 leading-snug mt-1">
                  Excluded from the score: {result.risk.components_unavailable.join(', ')} —
                  weights renormalised over the rest rather than defaulted.
                </p>
              )}
            </Section>
          </div>

          {/* Explanation. `available: false` means no attribution could be computed -- the
              backward pass produced no gradients at the target layer, or the model has no
              conv layer to attribute to. It used to return a uniform 0.5 heatmap, which
              rendered as a flat coloured square and read as "the model attends everywhere
              equally". Showing the reason instead. */}
          {result.explanation && (
            <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4">
              <h3 className="text-sm font-semibold text-slate-900 mb-2 flex items-center gap-2">
                <Zap className="w-4 h-4 text-purple-600" />
                AI Explanation (Grad-CAM)
                <span className="text-[10px] text-purple-600 bg-purple-400/10 px-2 py-0.5 rounded font-mono ml-auto">EXPLAINABLE AI</span>
              </h3>
              {result.explanation.available === false ? (
                <p className="text-xs text-slate-500 leading-snug">
                  No attribution available — {result.explanation.reason}
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <KV label="Method" value={result.explanation.explanation_method || 'Grad-CAM'} />
                    <KV label="Attention Area" value={result.explanation.attention_area_fraction != null
                      ? `${(result.explanation.attention_area_fraction * 100).toFixed(1)}%` : '--'} />
                    <KV label="Peak Location" value={result.explanation.peak_row != null
                      ? `(${result.explanation.peak_row}, ${result.explanation.peak_col})` : '--'} />
                    <KV label="Heatmap Mean" value={result.explanation.heatmap_mean?.toFixed(3) ?? '--'} />
                  </div>
                  {/* Heatmap visualization */}
                  {result.explanation.heatmap && (
                    <div className="mt-3">
                      <HeatmapCanvas heatmap={result.explanation.heatmap} size={64} />
                    </div>
                  )}
                </>
              )}
              {result.explanation.model_status && result.explanation.model_status !== 'trained' && (
                <p className="text-[10px] text-amber-700 leading-snug mt-2">
                  Model status: {result.explanation.model_status}. Saliency from an untrained
                  network shows where an unlearned filter responds, not where a cyclone is.
                </p>
              )}
            </div>
          )}

          {/* Disclaimer */}
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3">
            <p className="text-xs text-amber-300/80">{result.disclaimer}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function Section({ title, color, children }: { title: string; color: string; children: React.ReactNode }) {
  return (
    <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4">
      <h4 className="text-xs font-semibold text-slate-900 mb-2">{title}</h4>
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between"><span className="text-[11px] text-slate-600">{label}</span><span className="text-[11px] text-slate-900">{value}</span></div>
  );
}

function HeatmapCanvas({ heatmap, size }: { heatmap: number[][]; size: number }) {
  const canvasRef = (canvas: HTMLCanvasElement | null) => {
    if (!canvas || !heatmap || heatmap.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const h = heatmap.length;
    const w = heatmap[0]?.length || 0;
    canvas.width = w;
    canvas.height = h;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = heatmap[y]?.[x] || 0;
        const r = Math.round(v * 255);
        const g = Math.round((1 - v) * 100);
        const b = Math.round((1 - v) * 50);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  };
  return (
    <div>
      <canvas ref={canvasRef} className="w-32 h-32 rounded border border-slate-200" style={{ imageRendering: 'pixelated' }} />
      <p className="text-[10px] text-gray-500 mt-1">Grad-CAM activation heatmap — brighter = higher model attention</p>
    </div>
  );
}

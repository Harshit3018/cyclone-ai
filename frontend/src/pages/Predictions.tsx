import { useState, useEffect, useRef } from 'react';
import { Activity, Eye, Wind, Gauge, MapPin, Zap, Loader2, Play } from 'lucide-react';
import { predictDetect, predictClassification, predictIntensity, predictTrack } from '../api/client';
import type { DetectionResult, ClassificationResult, IntensityResult, TrackForecast, Plausibility } from '../types';

export default function Predictions() {
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [classification, setClassification] = useState<ClassificationResult | null>(null);
  const [intensity, setIntensity] = useState<IntensityResult | null>(null);
  const [track, setTrack] = useState<TrackForecast | null>(null);
  const [sampleIdx, setSampleIdx] = useState(0);
  const [loading, setLoading] = useState<string>('');
  const [error, setError] = useState<string>('');

  const runDetection = async () => {
    setLoading('detect'); try { const r = await predictDetect(sampleIdx); setDetection(r.data); } catch(e) { console.error(e); } finally { setLoading(''); }
  };
  const runClassification = async () => {
    setLoading('classify'); try { const r = await predictClassification(sampleIdx); setClassification(r.data); } catch(e) { console.error(e); } finally { setLoading(''); }
  };
  const runIntensity = async () => {
    setLoading('intensity'); try { const r = await predictIntensity(sampleIdx); setIntensity(r.data); } catch(e) { console.error(e); } finally { setLoading(''); }
  };
  const runTrack = async () => {
    setLoading('track'); try { const r = await predictTrack(sampleIdx); setTrack(r.data); } catch(e) { console.error(e); } finally { setLoading(''); }
  };

  const runAll = async (idx: number) => {
    setLoading('all');
    setError('');
    // allSettled, not all: one failing task should still show the other three
    // rather than blanking the page.
    const results = await Promise.allSettled([
      predictDetect(idx), predictClassification(idx), predictIntensity(idx), predictTrack(idx),
    ]);
    const setters = [setDetection, setClassification, setIntensity, setTrack] as const;
    const names = ['detection', 'classification', 'intensity', 'track'];
    const failed: string[] = [];
    results.forEach((r, i) => {
      if (r.status === 'fulfilled') (setters[i] as (v: unknown) => void)(r.value.data);
      else { failed.push(names[i]); console.error(names[i], r.reason); }
    });
    if (failed.length) setError(`Failed to run: ${failed.join(', ')}`);
    setLoading('');
  };

  // Run every task on mount, and again when the sample changes. Without this the
  // page rendered completely blank until four separate clicks -- the worst possible
  // first impression for a page whose whole purpose is showing that inference works.
  // Debounced because a range input fires onChange continuously while dragging, which
  // would otherwise launch a request per intermediate value.
  const debounce = useRef<number | undefined>(undefined);
  useEffect(() => {
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => runAll(sampleIdx), 350);
    return () => window.clearTimeout(debounce.current);
  }, [sampleIdx]);

  const busy = loading === 'all';

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Prediction Center</h2>
        <p className="text-sm text-slate-600 mt-1">Run real AI inference on demo satellite observations</p>
      </div>

      {/* Sample selector */}
      <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4">
        <div className="flex items-center gap-4">
          <span className="text-sm text-slate-600">Sample Index:</span>
          <input type="range" min={0} max={49} value={sampleIdx} onChange={e => setSampleIdx(Number(e.target.value))}
            className="flex-1 accent-blue-500" />
          <span className="text-sm font-mono text-slate-900 w-8">{sampleIdx}</span>
          <button onClick={() => runAll(sampleIdx)} disabled={busy}
            className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            {busy ? 'Running' : 'Run All'}
          </button>
        </div>
        <p className="text-[10px] text-gray-500 mt-2">
          Selecting synthetic demo satellite observation #{sampleIdx} from demo dataset (50 total)
          {' — all four models run automatically when the sample changes.'}
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-xs text-red-700">{error}</div>
      )}

      {/* Prediction buttons */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <PredButton icon={<Eye />} label="Detect" onClick={runDetection} loading={loading === 'detect' || busy} />
        <PredButton icon={<Activity />} label="Classify" onClick={runClassification} loading={loading === 'classify' || busy} />
        <PredButton icon={<Wind />} label="Intensity" onClick={runIntensity} loading={loading === 'intensity' || busy} />
        <PredButton icon={<MapPin />} label="Track" onClick={runTrack} loading={loading === 'track' || busy} />
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Detection */}
        {detection && (
          <ResultCard title="Detection Result" icon={<Eye className="w-4 h-4 text-cyan-600" />}>
            <KV label="Cyclone Detected" value={detection.cyclone_detected ? 'YES' : 'NO'} highlight={detection.cyclone_detected} />
            <KV label="Confidence" value={`${(detection.confidence * 100).toFixed(1)}%`} />
            <KV label="Center Lat" value={`${detection.center.latitude}°`} />
            <KV label="Center Lon" value={`${detection.center.longitude}°`} />
            <KV label="Inference Time" value={`${detection.inference_time_ms} ms`} />
            <ModelTag name={detection.model_name} version={detection.model_version} status={detection.model_status} />
          </ResultCard>
        )}

        {/* Classification */}
        {classification && (
          <ResultCard title="Classification Result" icon={<Activity className="w-4 h-4 text-blue-600" />}>
            <KV label="Predicted Class" value={classification.class} highlight />
            <KV label="Confidence" value={`${(classification.confidence * 100).toFixed(1)}%`} />
            <div className="mt-2">
              <span className="text-xs text-slate-600">Class Probabilities:</span>
              <div className="space-y-1 mt-1">
                {Object.entries(classification.probabilities).sort(([,a],[,b]) => b - a).slice(0, 5).map(([cls, prob]) => (
                  <div key={cls} className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-600 w-40 truncate">{cls}</span>
                    <div className="flex-1 h-1.5 bg-slate-200 rounded-full">
                      <div className="h-full bg-brand-primary/100 rounded-full" style={{ width: `${prob * 100}%` }} />
                    </div>
                    <span className="text-[10px] text-slate-500 w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
            <ModelTag name={classification.model_name} version={classification.model_version} status={classification.model_status} />
          </ResultCard>
        )}

        {/* Intensity */}
        {intensity && (
          <ResultCard title="Intensity Estimation" icon={<Wind className="w-4 h-4 text-amber-600" />}>
            <KV label="Wind Speed" value={`${intensity.wind_kph.toFixed(1)} km/h`} highlight />
            <KV label="Pressure" value={`${intensity.pressure_hpa.toFixed(1)} hPa`} />
            <KV label="IMD Category" value={intensity.imd_category || '--'} />
            {intensity.confidence_interval && (
              <>
                <KV label="Wind CI" value={`${intensity.confidence_interval.wind_low.toFixed(1)} - ${intensity.confidence_interval.wind_high.toFixed(1)} km/h`} />
                <KV label="CI Method" value={intensity.confidence_interval.method || 'MC Dropout'} />
              </>
            )}
            <ModelTag name={intensity.model_name} version={intensity.model_version} status={intensity.model_status} />
            <PlausibilityNote report={intensity.plausibility} />
          </ResultCard>
        )}

        {/* Track */}
        {track && (
          <ResultCard title="Track Forecast" icon={<MapPin className="w-4 h-4 text-green-600" />}>
            <div className="flex items-center gap-3 pb-1 text-[10px] font-semibold text-slate-500 uppercase tracking-wide">
              <span className="w-10">Lead</span>
              <span className="flex-1">Position</span>
              <span className="w-20 text-right">90% cone</span>
              <span className="w-16 text-right">Mean err</span>
            </div>
            {track.forecast_points.map(fp => (
              <div key={fp.forecast_hour} className="flex items-center gap-3 py-1 border-b border-slate-200/50">
                <span className="text-xs font-mono text-amber-600 w-10">+{fp.forecast_hour}h</span>
                <span className="text-xs text-slate-500 flex-1">
                  {fp.latitude.toFixed(2)}°N, {fp.longitude.toFixed(2)}°E
                </span>
                {/* Null when the error was never measured at this lead time. Rendering
                    "--" rather than a number keeps an unmeasured cone from reading as a
                    tight one; the previous `.toFixed(0)` also threw on null. */}
                <span className="w-20 text-right text-[10px] font-mono text-slate-600">
                  {fp.uncertainty_radius_km == null
                    ? <span className="text-slate-400" title="Not measured at this lead time">not measured</span>
                    : `±${fp.uncertainty_radius_km.toFixed(0)} km`}
                </span>
                <span className="w-16 text-right text-[10px] font-mono text-slate-400">
                  {fp.mean_error_km == null ? '--' : `${fp.mean_error_km.toFixed(0)} km`}
                </span>
              </div>
            ))}
            <KV label="Uncertainty" value={track.uncertainty_method} />
            {/* Deliberately reported as its own quantity, well away from the cone. The
                MC-dropout spread is the ensemble's internal disagreement and runs about an
                order of magnitude below the verified error; presenting it as a confidence
                radius would understate real uncertainty by ~15x. */}
            {track.forecast_points.some(fp => fp.mc_dropout_spread_km != null) && (
              <KV
                label="MC-dropout spread"
                value={
                  track.forecast_points
                    .filter(fp => fp.mc_dropout_spread_km != null)
                    .map(fp => `+${fp.forecast_hour}h ${fp.mc_dropout_spread_km!.toFixed(1)}`)
                    .join(', ') + ' km — model self-disagreement, not forecast error'
                }
              />
            )}
            {track.forecast_points.some(fp => fp.persistence_km != null) && (
              <KV
                label="Displacement vs persistence"
                value={
                  track.forecast_points
                    .filter(fp => fp.displacement_km != null && fp.persistence_km != null)
                    .map(fp => `+${fp.forecast_hour}h ${fp.displacement_km!.toFixed(0)}/${fp.persistence_km!.toFixed(0)}`)
                    .join(', ') + ' km'
                }
              />
            )}
            {track.track_skill?.verdict && (
              <div className="mt-2 rounded-md border border-blue-200 bg-blue-50 px-2.5 py-2">
                <p className="text-[10px] font-semibold text-blue-800">Verified skill</p>
                <p className="text-[10px] text-blue-700 leading-snug mt-0.5">
                  {track.track_skill.verdict}
                </p>
                <p className="text-[10px] text-blue-600/80 leading-snug mt-1">
                  Error above is {track.track_skill.error_source}.
                </p>
              </div>
            )}
            <ModelTag name={track.model_name} version={track.model_version} status={track.model_status} />
            <PlausibilityNote report={track.plausibility} />
          </ResultCard>
        )}
      </div>
    </div>
  );
}

/** Shows the physical-consistency verdict from ml/baselines/physics.py.
 *  The detail page demotes an implausible neural forecast to an analytic baseline;
 *  this page shows raw model output by design, so it must at least say so rather
 *  than present a physically impossible forecast without comment. */
function PlausibilityNote({ report }: { report?: Plausibility }) {
  if (!report || report.physically_plausible !== false) return null;
  return (
    <div className="mt-2 rounded-md border border-red-300 bg-red-50 px-2.5 py-2">
      <p className="text-[10px] font-semibold text-red-700">Fails physical consistency check</p>
      <ul className="mt-1 space-y-0.5">
        {(report.violations || []).map((v, i) => (
          <li key={i} className="text-[10px] text-red-700 leading-snug">• {v}</li>
        ))}
      </ul>
      <p className="text-[10px] text-red-600/80 mt-1">
        The cyclone detail page will not serve a forecast that fails these checks; it falls
        back to the best-scoring analytic baseline instead. This page shows raw model output
        by design, so the violation is displayed rather than hidden.
      </p>
    </div>
  );
}

function PredButton({ icon, label, onClick, loading }: { icon: React.ReactNode; label: string; onClick: () => void; loading: boolean }) {
  return (
    <button onClick={onClick} disabled={loading}
      className="bg-white shadow-sm border border-slate-200 rounded-xl p-4 hover:border-blue-500/50 hover:bg-white transition-all flex items-center gap-3 disabled:opacity-50">
      {loading ? <Loader2 className="w-5 h-5 text-blue-600 animate-spin" /> : <span className="text-blue-600">{icon}</span>}
      <span className="text-sm font-medium text-slate-900">{label}</span>
    </button>
  );
}

function ResultCard({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-white shadow-sm border-slate-200 rounded-xl border border-slate-200 p-4">
      <h3 className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
        {icon} {title}
        <span className="text-[10px] text-amber-600 bg-amber-400/10 px-1.5 py-0.5 rounded font-mono ml-auto">AI PREDICTION</span>
      </h3>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function KV({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-slate-600">{label}</span>
      <span className={`text-sm ${highlight ? 'font-semibold text-blue-700' : 'text-slate-900'}`}>{value}</span>
    </div>
  );
}

function ModelTag({ name, version, status }: { name: string; version: string; status: string }) {
  return (
    <div className="mt-3 pt-2 border-t border-slate-200">
      <p className="text-[10px] text-gray-500">Model: {name} v{version} • Status: {status}</p>
      <p className="text-[10px] text-amber-500/70 mt-0.5">AI-generated prediction for research purposes only</p>
    </div>
  );
}

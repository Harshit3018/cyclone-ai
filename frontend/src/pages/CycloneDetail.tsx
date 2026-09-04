import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Wind, Gauge, MapPin, Activity, TrendingUp, AlertTriangle,
  ArrowLeft, Cloud, BarChart3, GitCompare, Check, X
} from 'lucide-react';
import { getCyclone, getCycloneTrack, getCycloneForecast, getCycloneRisk } from '../api/client';
import CycloneMap from '../maps/CycloneMap';
import { CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, XAxis, YAxis } from 'recharts';
import type { TrackPoint, ForecastPoint, RiskAssessment, ForecastTrackRisk } from '../types';

const PANEL = 'bg-white border border-slate-200 rounded-xl shadow-sm';
const HEADING = 'text-[11px] font-bold text-slate-600 uppercase tracking-widest flex items-center gap-2';

export default function CycloneDetail() {
  const { id } = useParams<{ id: string }>();
  const [cyclone, setCyclone] = useState<any>(null);
  const [track, setTrack] = useState<TrackPoint[]>([]);
  const [forecast, setForecast] = useState<any>(null);
  const [risk, setRisk] = useState<RiskAssessment | null>(null);
  const [trackRisk, setTrackRisk] = useState<ForecastTrackRisk | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    Promise.all([
      getCyclone(id).then(r => setCyclone(r.data)),
      getCycloneTrack(id).then(r => setTrack(r.data.track_points || [])),
      getCycloneForecast(id).then(r => setForecast(r.data)).catch(() => {}),
      getCycloneRisk(id).then(r => {
        setRisk(r.data.risk);
        setTrackRisk(r.data.forecast_track_risk ?? null);
      }).catch(() => {}),
    ]).catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return (
    <div className="text-center py-12">
      <Cloud className="w-10 h-10 animate-pulse mx-auto mb-2 text-brand-primary" />
      <p className="text-[11px] text-slate-500 font-mono uppercase tracking-widest">Loading cyclone data...</p>
    </div>
  );
  if (error) return <div className="text-center py-12 text-red-600 text-sm">Error: {error}</div>;
  if (!cyclone) return <div className="text-center py-12 text-slate-500 text-sm">Cyclone not found</div>;

  const latestPoint = track.length > 0 ? track[track.length - 1] : null;

  // The API decides which forecast is trustworthy enough to headline: the neural
  // network only when it is trained AND its output passes physical plausibility
  // checks, otherwise the analytic baseline. Draw whatever it nominated rather than
  // always drawing the network.
  const primaryKey: string = forecast?.primary_forecast ?? 'neural_network';
  const primaryFc = primaryKey === 'neural_network'
    ? forecast?.track_forecast
    : forecast?.baseline_forecasts?.[primaryKey];
  const forecastPoints: ForecastPoint[] = primaryFc?.forecast_points || [];
  const primaryLabel: string = primaryFc?.model_name
    || (primaryKey === 'neural_network' ? 'Neural Track Model' : primaryKey);

  const windData = track.map((p, i) => ({ idx: i, wind: p.wind_kph, time: p.timestamp?.slice(5, 16) }));
  const pressureData = track.map((p, i) => ({ idx: i, pressure: p.pressure_hpa, time: p.timestamp?.slice(5, 16) }));

  return (
    <div className="space-y-6 animate-fade-in p-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/" className="text-slate-400 hover:text-brand-primary transition-colors">
          <ArrowLeft className="w-5 h-5" />
        </Link>
        <div className="flex-1">
          <h2 className="text-2xl font-black text-slate-900 tracking-tight uppercase leading-none">
            {cyclone.name?.replace('DEMO_', '')}
          </h2>
          <div className="flex items-center gap-3 mt-2">
            <span className="text-[10px] font-bold text-brand-primary uppercase tracking-wider">
              {cyclone.peak_category}
            </span>
            <span className="text-[10px] text-slate-500 font-mono uppercase tracking-wider">
              {cyclone.sub_basin} • Season {cyclone.season}
            </span>
            <span className="text-[10px] text-amber-700 font-mono tracking-wider bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-sm">
              {cyclone.data_type}
            </span>
          </div>
        </div>
      </div>

      {/* Map + stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={`lg:col-span-2 ${PANEL} overflow-hidden flex flex-col`}>
          <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
            <h3 className={HEADING}>
              <MapPin className="w-3.5 h-3.5 text-brand-primary" />
              Track &amp; Forecast Map
            </h3>
            <div className="flex gap-3 text-[10px] text-slate-500 font-medium">
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-brand-primary" />Observed</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500" />{primaryLabel}</span>
            </div>
          </div>
          {/* flex-1 so the map fills the panel instead of leaving dead white space
              when the stat column beside it is taller than 400px */}
          <div className="flex-1 min-h-[400px]">
            <CycloneMap cyclones={[cyclone]} selectedId={id} trackPoints={track} forecastPoints={forecastPoints} />
          </div>
        </div>

        <div className="space-y-4">
          <div className={`${PANEL} p-5`}>
            <h3 className={`${HEADING} mb-4`}>
              <Activity className="w-3.5 h-3.5 text-brand-secondary" />
              Current Conditions
            </h3>
            {latestPoint ? (
              <div className="space-y-3">
                <Row label="Wind Speed" value={`${latestPoint.wind_kph?.toFixed(0)} km/h`} />
                <Row label="Pressure" value={`${latestPoint.pressure_hpa?.toFixed(0)} hPa`} />
                <Row label="Position" value={`${latestPoint.latitude?.toFixed(2)}°N, ${latestPoint.longitude?.toFixed(2)}°E`} />
                <Row label="Category" value={latestPoint.category} accent />
                <Row label="Time" value={latestPoint.timestamp} dim />
              </div>
            ) : <p className="text-[11px] text-slate-500 font-mono">No data</p>}

            {forecast?.current_intensity_estimate && (
              <div className="mt-4 pt-3 border-t border-slate-100">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">
                    Model estimate from imagery
                  </span>
                  <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded-sm border ${
                    forecast.current_intensity_estimate.model_status === 'trained'
                      ? 'bg-green-50 text-green-700 border-green-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200'
                  }`}>
                    {forecast.current_intensity_estimate.model_status === 'baseline'
                      ? 'untrained' : forecast.current_intensity_estimate.model_status}
                  </span>
                </div>
                <Row
                  label="Est. Wind"
                  value={`${forecast.current_intensity_estimate.wind_kph?.toFixed(0)} km/h`}
                  dim
                />
                <Row
                  label="Est. Pressure"
                  value={`${forecast.current_intensity_estimate.pressure_hpa?.toFixed(0)} hPa`}
                  dim
                />
                {forecast.current_intensity_estimate.plausibility
                  && !forecast.current_intensity_estimate.plausibility.physically_plausible && (
                  <p className="text-[10px] text-red-700 mt-2 leading-snug">
                    Fails physical consistency: {forecast.current_intensity_estimate.plausibility.violations[0]}
                  </p>
                )}
              </div>
            )}
          </div>

          <div className={`${PANEL} p-5`}>
            <h3 className={`${HEADING} mb-4`}>
              <TrendingUp className="w-3.5 h-3.5 text-brand-primary" />
              Peak Statistics
            </h3>
            <div className="space-y-3">
              <Row label="Peak Wind" value={`${cyclone.peak_wind_kph?.toFixed(0)} km/h`} />
              <Row label="Min Pressure" value={`${cyclone.min_pressure_hpa?.toFixed(0)} hPa`} />
              <Row label="Observations" value={String(cyclone.num_observations)} />
            </div>
          </div>

          {risk && (
            <div className={`${PANEL} p-5`}>
              <h3 className={`${HEADING} mb-4`}>
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                Risk Assessment
              </h3>
              <div className="space-y-2.5">
                <RiskBar label="Overall" value={risk.overall_risk} />
                <RiskBar label="Wind" value={risk.wind_risk} />
                <RiskBar
                  label="Coastal"
                  value={risk.coastal_risk}
                  note={risk.coastal_risk_basis
                    ? `Scored at ${risk.coastal_risk_basis} — ${risk.distance_to_coast_km} km from land.`
                    : undefined}
                />
                <RiskBar
                  label="Rainfall"
                  value={risk.rainfall_risk}
                  note={risk.component_provenance?.rainfall}
                />
                <RiskBar
                  label="Rapid Int."
                  value={risk.rapid_intensification_risk}
                  note={risk.component_provenance?.rapid_intensification}
                />
              </div>

              {/* What the score actually averaged over. Without this the renormalised
                  weights are invisible and the number looks like it used all five. */}
              {risk.components_unavailable && risk.components_unavailable.length > 0 && (
                <p className="mt-3 text-[9px] text-slate-500 leading-snug">
                  {risk.components_unavailable.length} of{' '}
                  {(risk.components_used?.length || 0) + risk.components_unavailable.length}{' '}
                  components unavailable and excluded — weights renormalised over the rest,
                  not defaulted to a constant.
                </p>
              )}
              {risk.coastal_risk_basis && (
                <p className="mt-1.5 text-[9px] text-slate-500 leading-snug">
                  Coastal risk scored at {risk.coastal_risk_basis}
                  {risk.distance_to_coast_km !== undefined
                    ? ` (${risk.distance_to_coast_km} km)`
                    : ''}.
                </p>
              )}

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                <span className={`text-[10px] font-bold tracking-widest uppercase px-2 py-0.5 rounded ${
                  risk.alert_level.includes('EXTREME') ? 'bg-red-50 text-red-700' :
                  risk.alert_level.includes('HIGH') ? 'bg-orange-50 text-orange-700' :
                  risk.alert_level.includes('ADVISORY') ? 'bg-amber-50 text-amber-700' :
                  risk.alert_level.includes('UNAVAILABLE') ? 'bg-slate-100 text-slate-500' :
                  'bg-green-50 text-green-700'
                }`}>{risk.alert_level}</span>
                <span className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">Rules-based</span>
              </div>
            </div>
          )}

          {/* Coastal approach along the forecast. This is the panel that answers the
              question a district officer actually has -- when does it reach my coast --
              rather than restating how far offshore it is right now. */}
          {trackRisk && trackRisk.closest_approach && (
            <div className={`${PANEL} p-5`}>
              <h3 className={`${HEADING} mb-4`}>
                <MapPin className="w-3.5 h-3.5 text-brand-primary" />
                Coastal Approach
              </h3>

              <p className="text-[11px] text-slate-700 leading-relaxed mb-3">
                {trackRisk.landfall.verdict}
              </p>

              <div className="space-y-2 mb-3">
                <Row
                  label="Closest approach"
                  value={`${trackRisk.closest_approach.distance_to_land_km} km @ +${trackRisk.closest_approach.forecast_hour}h`}
                />
                <Row label="Coast" value={trackRisk.closest_approach.nearest_coast_label} />
                <Row
                  label="Landfall possible"
                  value={trackRisk.landfall.possible_from_hour !== null
                    ? `+${trackRisk.landfall.possible_from_hour}h`
                    : 'not within horizon'}
                />
              </div>

              <table className="w-full text-[10px] font-mono">
                <thead>
                  <tr className="text-slate-400 uppercase tracking-wider">
                    <th className="text-left font-semibold pb-1">Lead</th>
                    <th className="text-right font-semibold pb-1">To land</th>
                    <th className="text-right font-semibold pb-1">Cone</th>
                    <th className="text-right font-semibold pb-1">Reaches</th>
                  </tr>
                </thead>
                <tbody>
                  {trackRisk.coastal_profile.map(p => (
                    <tr key={p.forecast_hour} className="border-t border-slate-100">
                      <td className="py-1 text-slate-600">+{p.forecast_hour}h</td>
                      <td className="py-1 text-right text-slate-700">{p.distance_to_land_km}</td>
                      <td className="py-1 text-right text-slate-500">
                        {p.uncertainty_radius_km !== null ? p.uncertainty_radius_km.toFixed(0) : '--'}
                      </td>
                      <td className={`py-1 text-right font-semibold ${
                        p.cone_reaches_coast === true ? 'text-red-600'
                          : p.cone_reaches_coast === false ? 'text-green-600'
                          : 'text-slate-400'
                      }`}>
                        {/* null is "no cone was verified at this horizon", which is not "no". */}
                        {p.cone_reaches_coast === true ? 'YES'
                          : p.cone_reaches_coast === false ? 'no'
                          : 'n/a'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Observed intensity change: real, and the only RI signal in the system. */}
              <div className="mt-3 pt-3 border-t border-slate-100">
                {trackRisk.intensity_trend.available ? (
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">
                      Observed {trackRisk.intensity_trend.window_hours}h change
                    </span>
                    <span className={`text-[12px] font-mono font-semibold ${
                      trackRisk.intensity_trend.rapidly_intensifying ? 'text-red-600' : 'text-slate-700'
                    }`}>
                      {(trackRisk.intensity_trend.observed_change_kt ?? 0) > 0 ? '+' : ''}
                      {trackRisk.intensity_trend.observed_change_kt} kt
                      {trackRisk.intensity_trend.rapidly_intensifying ? ' · RI' : ''}
                    </span>
                  </div>
                ) : (
                  <p className="text-[9px] text-slate-400 leading-snug">
                    Intensity trend unavailable: {trackRisk.intensity_trend.reason}
                  </p>
                )}
                <p className="mt-1 text-[9px] text-slate-400 leading-snug">
                  Observed change against the {trackRisk.intensity_trend.threshold_kt_per_24h ?? 30} kt/24h
                  rapid-intensification threshold. This is what the storm has done, not a forecast —{' '}
                  {trackRisk.intensity_forecast_note}
                </p>
              </div>

              <p className="mt-3 pt-3 border-t border-slate-100 text-[9px] text-slate-400 leading-snug">
                {trackRisk.geometry.note}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className={`${PANEL} p-5`}>
          <h3 className={`${HEADING} mb-4`}>
            <Wind className="w-3.5 h-3.5 text-brand-secondary" />
            Wind Speed Evolution
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={windData}>
              <defs>
                <linearGradient id="windGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0891b2" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#0891b2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }} />
              <Area type="monotone" dataKey="wind" stroke="#0891b2" fill="url(#windGrad)" strokeWidth={2} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className={`${PANEL} p-5`}>
          <h3 className={`${HEADING} mb-4`}>
            <Gauge className="w-3.5 h-3.5 text-brand-primary" />
            Pressure Evolution
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={pressureData}>
              <defs>
                <linearGradient id="presGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} />
              <YAxis domain={['dataMin - 10', 'dataMax + 10']} tick={{ fontSize: 10, fill: '#64748b' }} />
              <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 12 }} />
              <Area type="monotone" dataKey="pressure" stroke="#2563eb" fill="url(#presGrad)" strokeWidth={2} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Forecast provenance & baseline comparison */}
      {forecast && (
        <div className={`${PANEL} p-5`}>
          <h3 className={`${HEADING} mb-1`}>
            <GitCompare className="w-3.5 h-3.5 text-brand-primary" />
            Track Forecast — Method &amp; Baselines
          </h3>
          <p className="text-[11px] text-slate-600 mb-4 max-w-3xl leading-relaxed">
            <span className="font-semibold text-slate-800">Showing: {primaryLabel}.</span>{' '}
            {forecast.primary_forecast_reason}
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[10px] text-slate-500 border-b border-slate-200 uppercase tracking-widest">
                  <th className="text-left py-2 px-3 font-bold">Method</th>
                  <th className="text-left py-2 px-3 font-bold">Status</th>
                  {(primaryFc?.forecast_points || []).map((p: any) => (
                    <th key={p.forecast_hour} className="text-right py-2 px-3 font-bold">+{p.forecast_hour}h</th>
                  ))}
                  {/* Where each method points is only half the story; how wrong it has been
                      on cases it never saw is the other half. Without this column the table
                      invites the reader to judge forecasts by how confident they look. */}
                  <th className="text-right py-2 px-3 font-bold whitespace-nowrap" title="Mean great-circle error at +24 h on held-out 2024-2025 seasons">
                    +24h err
                  </th>
                  <th className="text-center py-2 px-3 font-bold">Physical</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { key: 'neural_network', fc: forecast.track_forecast },
                  ...Object.entries(forecast.baseline_forecasts || {}).map(([k, v]) => ({ key: k, fc: v as any })),
                ].filter(r => r.fc).map(({ key, fc }) => {
                  const plausible = fc.plausibility?.physically_plausible;
                  return (
                    <tr key={key} className={`border-b border-slate-100 ${key === primaryKey ? 'bg-cyan-50/60' : ''}`}>
                      <td className="py-2 px-3 text-slate-800 font-medium whitespace-nowrap">
                        {fc.model_name || 'Neural Track Model'}
                        {key === primaryKey && (
                          <span className="ml-2 text-[9px] text-cyan-700 bg-cyan-100 px-1.5 py-0.5 rounded-sm font-bold tracking-wider">PRIMARY</span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        {/* Four statuses now reach this table: trained (the network),
                            fitted_baseline (CLIPER), analytic_baseline (persistence) and
                            analytic_baseline_superseded (CLIP-like, measured worse than
                            persistence). The last must not share the neutral grey of a
                            respectable baseline. */}
                        <span className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded-sm border ${
                          fc.model_status === 'trained' ? 'bg-green-50 text-green-700 border-green-200'
                          : fc.model_status === 'fitted_baseline' ? 'bg-blue-50 text-blue-700 border-blue-200'
                          : fc.model_status === 'analytic_baseline' ? 'bg-slate-100 text-slate-600 border-slate-200'
                          : fc.model_status === 'analytic_baseline_superseded' ? 'bg-orange-50 text-orange-700 border-orange-200'
                          : 'bg-amber-50 text-amber-700 border-amber-200'
                        }`}>{fc.model_status === 'baseline' ? 'untrained' : fc.model_status}</span>
                      </td>
                      {(fc.forecast_points || []).map((p: any) => (
                        <td key={p.forecast_hour} className="py-2 px-3 text-right text-slate-700 font-mono text-[11px] whitespace-nowrap">
                          {p.latitude.toFixed(1)}°N {p.longitude.toFixed(1)}°E
                          {/* Null means the error was never measured at this lead time.
                              Optional chaining alone rendered a bare "± km", which reads
                              as a missing digit rather than a missing measurement. */}
                          <span className="block text-[9px] text-slate-400">
                            {p.uncertainty_radius_km == null
                              ? 'cone not measured'
                              : `±${p.uncertainty_radius_km.toFixed(0)} km`}
                          </span>
                        </td>
                      ))}
                      <td className="py-2 px-3 text-right font-mono text-[11px] whitespace-nowrap">
                        {(() => {
                          // Baselines carry their own verified_skill block; the network's
                          // measured error sits at the top level of the forecast response,
                          // because it comes from the checkpoint rather than the baseline
                          // tables.
                          const err =
                            fc.verified_skill?.mean_error_km?.['24'] ??
                            (key === 'neural_network'
                              ? forecast.track_skill?.error_km?.['24']?.mean_km
                              : undefined);
                          if (err == null) return <span className="text-slate-300">—</span>;
                          const best = fc.model_status === 'trained';
                          return (
                            <span className={best ? 'text-green-700 font-semibold' : 'text-slate-600'}>
                              {err.toFixed(0)} km
                            </span>
                          );
                        })()}
                      </td>
                      <td className="py-2 px-3 text-center">
                        {plausible === undefined ? (
                          <span className="text-slate-300 text-[11px]">—</span>
                        ) : plausible ? (
                          <Check className="w-4 h-4 text-green-600 inline" />
                        ) : (
                          <X className="w-4 h-4 text-red-600 inline" />
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {forecast.track_forecast?.plausibility?.violations?.length > 0 && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-[10px] font-bold text-red-700 uppercase tracking-widest mb-1.5">
                Neural model failed physical checks
              </p>
              <ul className="space-y-1">
                {forecast.track_forecast.plausibility.violations.map((v: string, i: number) => (
                  <li key={i} className="text-[11px] text-red-800 leading-snug">• {v}</li>
                ))}
              </ul>
            </div>
          )}

          {primaryFc?.current_motion && (
            <p className="text-[9px] text-slate-500 mt-3 font-mono uppercase tracking-wider">
              Current motion: {primaryFc.current_motion.bearing_deg}° at {primaryFc.current_motion.speed_kph} km/h,
              derived over {primaryFc.current_motion.derived_over_hours} h from {primaryFc.current_motion.positions_used} positions
            </p>
          )}
          {primaryFc?.uncertainty_method && (
            <p className="text-[9px] text-slate-500 mt-1 font-mono">
              Uncertainty: {primaryFc.uncertainty_method}
            </p>
          )}
        </div>
      )}

      {/* Intensity forecast */}
      {forecast && (
        <div className={`${PANEL} p-5`}>
          <h3 className={`${HEADING} mb-4`}>
            <BarChart3 className="w-3.5 h-3.5 text-amber-600" />
            Intensity Forecast
            {forecast.intensity_forecast ? (
              <span className="text-[9px] text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-sm font-mono tracking-wider ml-1">
                AI PREDICTION
              </span>
            ) : (
              <span className="text-[9px] text-slate-600 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-sm font-mono tracking-wider ml-1">
                NOT AVAILABLE
              </span>
            )}
          </h3>

          {forecast.intensity_forecast ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[10px] text-slate-500 border-b border-slate-200 uppercase tracking-widest">
                    <th className="text-left py-2 px-3 font-bold">Horizon</th>
                    <th className="text-right py-2 px-3 font-bold">Wind (km/h)</th>
                    <th className="text-right py-2 px-3 font-bold">Pressure (hPa)</th>
                  </tr>
                </thead>
                <tbody>
                  {forecast.intensity_forecast.map((f: any) => (
                    <tr key={f.forecast_hour} className="border-b border-slate-100">
                      <td className="py-2 px-3 text-amber-700 font-mono font-semibold">+{f.forecast_hour}h</td>
                      <td className="py-2 px-3 text-right text-slate-800 font-mono">{f.wind_kph?.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right text-slate-800 font-mono">{f.pressure_hpa?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[12px] text-slate-600 leading-relaxed max-w-2xl">
              Intensity forecasting is not yet implemented — there is no trained
              intensity-forecast head in this build, so no values are reported rather than
              synthesised ones. The current intensity <em>estimate</em> from satellite
              imagery is available in the AI Analysis pipeline.
            </p>
          )}

          <p className="text-[9px] text-slate-500 mt-4 font-mono uppercase tracking-wider">
            Track model: {forecast.track_forecast?.model_name || 'Baseline'} v{forecast.track_forecast?.model_version || '0.1.0'} •
            Status: {forecast.track_forecast?.model_status || 'untrained'} • {cyclone.data_type}
          </p>
        </div>
      )}
    </div>
  );
}

function Row({ label, value, accent, dim }: {
  label: string; value?: string; accent?: boolean; dim?: boolean;
}) {
  const cls = accent ? 'text-amber-700' : dim ? 'text-slate-500' : 'text-slate-800';
  return (
    <div className="flex justify-between items-baseline gap-3">
      <span className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">{label}</span>
      <span className={`text-[13px] font-mono ${cls}`}>{value ?? '--'}</span>
    </div>
  );
}

/**
 * One risk component as a bar.
 *
 * A null value renders as "n/a" over an empty grey track, never as 0%. The backend returns
 * null for components it cannot compute — currently rapid intensification, which needs a
 * trained intensity-forecast head. Drawing that as a green 0% bar would tell the user there
 * is no rapid-intensification risk, which is the opposite of "we don't know".
 */
function RiskBar({ label, value, note }: { label: string; value: number | null | undefined; note?: string }) {
  if (value === null || value === undefined) {
    return (
      <div className="flex items-center gap-2" title={note || 'Not available — this component is not computed, so it was excluded from the score rather than defaulted.'}>
        <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider w-16">{label}</span>
        <div className="flex-1 h-1.5 rounded-full bg-[repeating-linear-gradient(45deg,#e2e8f0_0_3px,#f1f5f9_3px_6px)]" />
        <span className="text-[10px] text-slate-400 font-mono w-8 text-right">n/a</span>
      </div>
    );
  }
  const pct = Math.round(value * 100);
  const color = value >= 0.7 ? 'bg-red-600' : value >= 0.4 ? 'bg-amber-500' : 'bg-green-600';
  return (
    <div className="flex items-center gap-2" title={note}>
      <span className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider w-16">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-slate-600 font-mono w-8 text-right">{pct}%</span>
    </div>
  );
}

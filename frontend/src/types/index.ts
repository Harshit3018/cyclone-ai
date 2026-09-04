// CYCLONE-AI Type Definitions

export interface Cyclone {
  id: string;
  name: string;
  basin: string;
  sub_basin?: string;
  season: number;
  start_time?: string;
  end_time?: string;
  peak_wind_kt?: number;
  peak_wind_kph?: number;
  min_pressure_hpa?: number;
  peak_category?: string;
  num_observations?: number;
  data_type: string;
  data_source?: string;
  is_active: boolean;
}

export interface TrackPoint {
  timestamp: string;
  latitude: number;
  longitude: number;
  wind_knots?: number;
  wind_kph?: number;
  pressure_hpa?: number;
  category?: string;
  dist_to_coast_km?: number;
  data_type: string;
  point_index?: number;
}

export interface ForecastPoint {
  forecast_hour: number;
  latitude: number;
  longitude: number;
  lat_uncertainty?: number;
  lon_uncertainty?: number;
  /**
   * 90th-percentile verified error, in km. Null at lead times where error was never
   * measured — the API reports null rather than extrapolating a cone, so consumers
   * must not assume this is a number.
   */
  uncertainty_radius_km: number | null;
  /** Mean verified error at this lead time. Absent when unmeasured. */
  mean_error_km?: number;
  /** How many held-out cases the error at this lead time was measured over. */
  verified_cases?: number;
  /** Great-circle distance from the analysis position, in km. */
  displacement_km?: number;
  /** What plain persistence would have moved the storm, for comparison. */
  persistence_km?: number;
  /**
   * Spread of MC-dropout samples. This is the model's internal disagreement, NOT a
   * forecast error estimate — it is roughly an order of magnitude smaller than the
   * verified error and must never be displayed as a cone radius.
   */
  mc_dropout_spread_km?: number;
  heading_deg?: number;
  translation_speed_kph?: number;
}

/**
 * Risk score and its components.
 *
 * Every component is nullable because a component the backend could not compute is
 * returned as null, not as a default. `rapid_intensification_risk` and `track_confidence`
 * are null today: there is no trained intensity-forecast head and no per-storm track
 * confidence estimate. They were previously sent as the literals 0.1 and 0.5 while
 * carrying 30% of the score's weight, so anything rendering them was drawing a constant.
 *
 * Render a null component as "not available" — never as 0, and never as a full-width or
 * empty bar. A 0 bar reads as "no rapid-intensification risk", which is a claim the system
 * cannot make.
 */
export interface RiskAssessment {
  overall_risk: number | null;
  wind_risk: number | null;
  coastal_risk: number | null;
  rainfall_risk: number | null;
  rapid_intensification_risk: number | null;
  track_confidence: number | null;
  alert_level: string;
  /** Components that entered the score. */
  components_used?: string[];
  /** Components excluded, with their weight redistributed over the rest. */
  components_unavailable?: string[];
  /** Renormalised weights actually applied; sums to 1 over `components_used`. */
  weights_applied?: Record<string, number>;
  /** Where each component's value comes from, measured or proxy. */
  component_provenance?: Record<string, string>;
  /** Which position `coastal_risk` was scored at — forecast closest approach or current. */
  coastal_risk_basis?: string;
  distance_to_coast_km?: number;
  methodology?: string;
}

/** One forecast position with its distance to the nearest coast. */
export interface CoastalProfilePoint {
  forecast_hour: number;
  latitude: number;
  longitude: number;
  distance_to_land_km: number;
  nearest_coast: string;
  nearest_coast_label: string;
  uncertainty_radius_km: number | null;
  /**
   * True when the verified p90 error cone reaches land at this lead time. Null where no
   * cone was measured for that horizon — which is not the same as false.
   */
  cone_reaches_coast: boolean | null;
}

export interface ForecastTrackRisk {
  coastal_profile: CoastalProfilePoint[];
  closest_approach: {
    forecast_hour: number;
    distance_to_land_km: number;
    nearest_coast: string;
    nearest_coast_label: string;
    latitude: number;
    longitude: number;
  } | null;
  landfall: {
    /** First lead time at which landfall falls inside the verified error cone. */
    possible_from_hour: number | null;
    possible_coast: string | null;
    /** Later: first lead time the forecast centre itself reaches the coast. */
    track_arrives_hour: number | null;
    track_arrives_coast: string | null;
    geometry_resolution_km: number;
    verdict: string;
  };
  /** Observed 24 h intensity change. An observation, not a forecast. */
  intensity_trend: {
    available: boolean;
    reason?: string;
    observed_change_kt?: number;
    window_hours?: number;
    rate_kt_per_24h?: number;
    threshold_kt_per_24h?: number;
    rapidly_intensifying?: boolean;
    basis?: string;
  };
  intensity_at_forecast_positions: string;
  intensity_forecast_available: boolean;
  intensity_forecast_note: string;
  geometry: {
    note: string;
    mean_absolute_error_km: number;
    p90_absolute_error_km: number;
    reference: string;
    validated_on_positions: number;
  };
  forecast_source?: string;
  forecast_source_reason?: string;
}

export interface DetectionResult {
  cyclone_detected: boolean;
  confidence: number;
  center: { latitude: number; longitude: number };
  inference_time_ms: number;
  model_name: string;
  model_version: string;
  model_status: string;
}

export interface ClassificationResult {
  class: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  model_name: string;
  model_version: string;
  model_status: string;
}

/** Verdict from ml/baselines/physics.py -- whether a model's output is
 *  physically self-consistent (wind/pressure relation, translation speed, etc.). */
export interface Plausibility {
  physically_plausible: boolean;
  violations: string[];
}

export interface IntensityResult {
  wind_kph: number;
  pressure_hpa: number;
  imd_category?: string;
  confidence_interval?: {
    wind_low: number;
    wind_high: number;
    pressure_low: number;
    pressure_high: number;
    method: string;
  };
  plausibility?: Plausibility;
  model_name: string;
  model_version: string;
  model_status: string;
}

export interface TrackSkill {
  error_km: Record<string, HorizonError>;
  error_source: string;
  significance_vs_cliper?: Record<string, SignificanceEntry>;
  verdict?: string;
}

export interface TrackForecast {
  forecast_points: ForecastPoint[];
  horizons_hours: number[];
  uncertainty_method: string;
  plausibility?: Plausibility;
  model_name: string;
  model_version: string;
  model_status: string;
  /** Verified held-out accuracy of whichever model produced this forecast. */
  track_skill?: TrackSkill | null;
  method?: string;
  max_trained_horizon_h?: number;
  training_provenance?: Record<string, any>;
}

export interface Alert {
  id: number;
  cyclone_id: string;
  alert_level: string;
  alert_type?: string;
  message?: string;
  is_active: boolean;
  data_type: string;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  source?: string;
  source_url?: string;
  data_type?: string;
  license?: string;
  status: string;
}

export interface HorizonError {
  mean_km: number;
  p90_km: number;
  n: number;
}

export interface SignificanceEntry {
  n: number;
  mean_difference_km: number;
  ci95_km: [number, number];
  p_value: number;
  significant: boolean;
  favours: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  version?: string;
  task?: string;
  architecture?: string;
  dataset_id?: string;
  metrics?: Record<string, any>;
  status: string;
  /** Measured held-out error keyed by forecast hour. Absent for unevaluated models. */
  verified_error_km?: Record<string, HorizonError>;
  /** Positive means more accurate than persistence at that lead time. */
  skill_vs_persistence_pct?: Record<string, number>;
  /** Paired-bootstrap result against the fitted CLIPER, keyed by forecast hour. */
  significance_vs_cliper?: Record<string, SignificanceEntry>;
  is_primary?: boolean;
}

export interface SystemStatus {
  status: string;
  version: string;
  environment: string;
  database: string;
  models_loaded: Record<string, boolean>;
  data_mode: string;
  active_cyclones: number;
  total_cyclones: number;
  uptime_seconds: number;
  disclaimer: string;
}

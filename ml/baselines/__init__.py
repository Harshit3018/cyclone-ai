"""
Analytic baselines and physical constraints for CYCLONE-AI.

Kept separate from `ml/models` on purpose: nothing in here is learned, so nothing in
here can silently degrade when a checkpoint is missing. These are the reference
forecasts a learned model has to beat, and the physical limits its output is checked
against.
"""
from .physics import (
    ENV_PRESSURE_HPA,
    KPH_TO_KT,
    KT_TO_KPH,
    TRANSLATION_SPEED_MAX_KPH,
    TRANSLATION_SPEED_SUSPECT_KPH,
    WORLD_RECORD_MIN_PRESSURE_HPA,
    PlausibilityReport,
    check_intensity,
    check_track,
    haversine_km,
    initial_bearing_deg,
    destination_point,
    pressure_from_wind,
    wind_from_pressure,
)
from .track import DEFAULT_HORIZONS, clip_lite, estimate_motion, persistence

__all__ = [
    "ENV_PRESSURE_HPA",
    "KPH_TO_KT",
    "KT_TO_KPH",
    "TRANSLATION_SPEED_MAX_KPH",
    "TRANSLATION_SPEED_SUSPECT_KPH",
    "WORLD_RECORD_MIN_PRESSURE_HPA",
    "PlausibilityReport",
    "check_intensity",
    "check_track",
    "haversine_km",
    "initial_bearing_deg",
    "destination_point",
    "pressure_from_wind",
    "wind_from_pressure",
    "DEFAULT_HORIZONS",
    "clip_lite",
    "estimate_motion",
    "persistence",
]

"""
Windowed track-forecasting samples built from IBTrACS storm tracks.

Turns a list of ``StormTrack`` into supervised (history, target) pairs suitable for a
sequence model, and carries the normalisation and unit conventions that make the
learning problem well posed.

Three decisions here do most of the work
----------------------------------------
1. **Targets are displacements, not positions.** Regressing absolute lat/lon lets the
   model score well by emitting the basin centroid: North Indian Ocean storms live in a
   narrow box, so "always predict 17N 87E" already has a low MSE and zero forecast
   skill. Displacement relative to the current position removes that shortcut, and it
   is also the quantity operational verification actually measures.

2. **Displacements are in kilometres (east, north), not degrees.** One degree of
   longitude is 111 km at the equator but 95 km at 20N, so a degree-space loss silently
   weights southern storms more than northern ones. Converting to a local
   east/north frame makes one unit of loss mean the same thing everywhere, and the
   inverse transform back to lat/lon is exact enough over forecast distances.

3. **History features separate position from motion.** Absolute latitude is kept because
   the physics depends on it -- Coriolis scales with sin(lat) and the beta drift that
   pushes storms poleward-westward is latitude-dependent -- while the step-to-step
   displacement carries the persistence signal. A model given only absolute positions
   has to difference them itself, which a 2-layer LSTM on 1400 samples will not do well.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ibtracs import StormTrack, TrackPoint

# --- Conventions -----------------------------------------------------------------

KM_PER_DEG_LAT = 111.195           # mean meridional degree on a sphere of R=6371.0088
SIX_HOURLY_STEP_H = 6.0

# Forecast horizons in hours. These match IMD bulletin lead times and the four horizons
# the platform's API already reports, so a trained model drops into the existing
# response schema without changing the frontend.
DEFAULT_HORIZONS_H = (6, 12, 24, 48)

# Length of input history in 6-hourly steps. Four steps is 18 h of motion, enough to
# resolve a turn; longer histories cost samples (each extra step removes one sample per
# storm) and this basin cannot afford them.
DEFAULT_HISTORY_STEPS = 4

# Feature layout of one history step. Documented as a constant because the trainer, the
# predictor and any future ONNX export must all agree on it.
TRACK_FEATURE_NAMES = (
    "lat_norm",         # latitude / 30, retains the Coriolis/beta dependence
    "lon_norm",         # (longitude - 72.5) / 32.5, centred on the basin
    "sin_lat",          # explicit Coriolis proxy
    "u_norm",           # eastward motion over the previous step / 30 km/h
    "v_norm",           # northward motion over the previous step / 30 km/h
    "speed_norm",       # translation speed magnitude / 30 km/h
    "wind_norm",        # wind_kt / 140, IMD 3-minute sustained where available
    "pres_norm",        # (1010 - pres_hpa) / 100, a deficit so 0 means "no storm"
    "sin_doy",          # seasonality: NIO has a bimodal pre/post-monsoon season
    "cos_doy",
    "wind_present",     # 1 if a real wind observation backed wind_norm, else 0
)
TRACK_FEATURE_DIM = len(TRACK_FEATURE_NAMES)

# Every feature above is scaled to land within roughly [-2, 2]. This is not cosmetic:
# leaving the motion components in raw km/h put them at +-50 while the remaining nine
# features sat near 1, and a 50:1 scale disparity across inputs to a shared LSTM makes
# the motion channels dominate the input-gate gradients regardless of their information
# content. The same failure -- on a far worse 7-orders-of-magnitude spread -- is
# documented in ml/utils/env_normalization.py.
#
# 30 km/h is used rather than the observed maximum because it is just above the 95th
# percentile of real translation speed (27.1 km/h): scaling by the maximum would compress
# the bulk of the distribution into a narrow band around 0.4 to make room for a handful
# of recurving outliers.
MOTION_SCALE_KMH = 30.0

# Basin centring constants for lon_norm: the NIO domain is 40E-105E, so 72.5 is the
# midpoint and 32.5 the half-width, mapping the basin onto roughly [-1, 1].
LON_CENTRE = 72.5
LON_HALF_WIDTH = 32.5
LAT_SCALE = 30.0
WIND_SCALE = 140.0            # a little above the basin record (AMPHAN, 130 kt)
PRES_REFERENCE_HPA = 1010.0
PRES_DEFICIT_SCALE = 100.0

# Target scaling. Displacements at +48 h reach ~1000 km; dividing by 500 puts the
# targets near unit variance without clipping the tail, which keeps the regression head
# in a numerically comfortable range without distorting large-displacement cases.
DISPLACEMENT_SCALE_KM = 500.0


def km_per_deg_lon(lat_deg: float) -> float:
    """Length of one degree of longitude at a given latitude, in km."""
    return KM_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def displacement_km(lat0: float, lon0: float, lat1: float, lon1: float
                    ) -> tuple[float, float]:
    """(east, north) displacement in km from point 0 to point 1.

    A local tangent-plane approximation, evaluated at the mean latitude of the pair.
    Over the few hundred km of a 48 h forecast the error against the true geodesic is
    well under a kilometre -- far below the ~100 km scale of track-forecast error -- and
    unlike the geodesic it inverts in closed form.
    """
    mean_lat = 0.5 * (lat0 + lat1)
    east = (lon1 - lon0) * km_per_deg_lon(mean_lat)
    north = (lat1 - lat0) * KM_PER_DEG_LAT
    return east, north


def offset_from_km(lat0: float, lon0: float, east_km: float, north_km: float
                   ) -> tuple[float, float]:
    """Inverse of :func:`displacement_km`: apply an (east, north) km offset."""
    lat1 = lat0 + north_km / KM_PER_DEG_LAT
    mean_lat = 0.5 * (lat0 + lat1)
    denom = km_per_deg_lon(mean_lat)
    lon1 = lon0 + (east_km / denom if abs(denom) > 1e-6 else 0.0)
    return lat1, lon1


# --- Sample construction ---------------------------------------------------------

@dataclass
class TrackSample:
    """One supervised forecasting example.

    Attributes:
        history: [T, TRACK_FEATURE_DIM] normalised input sequence.
        target_km: [H, 2] (east, north) displacement in km from the origin, unscaled.
        target_scaled: [H, 2] the same divided by DISPLACEMENT_SCALE_KM -- what the
            network regresses.
        target_mask: [H] 1.0 where the horizon has a verifying observation. Storms
            dissipate, so a +48 h target frequently does not exist; masking lets the
            sample still train the shorter horizons instead of being discarded.
        origin_lat / origin_lon: analysis position the displacement is measured from.
        sid / season / name / subbasin: provenance, so any prediction can be traced
            back to a specific storm and verified against the best track by hand.
    """
    history: np.ndarray
    target_km: np.ndarray
    target_scaled: np.ndarray
    target_mask: np.ndarray
    origin_lat: float
    origin_lon: float
    origin_time_iso: str
    sid: str
    season: int
    name: str
    subbasin: str


def step_features(pt: TrackPoint, prev: TrackPoint | None) -> list[float]:
    """Build one history step's feature vector, in TRACK_FEATURE_NAMES order.

    Public because inference must call exactly this function. A serving-side
    reimplementation of the feature vector is the classic way a correctly-trained track
    model produces confident nonsense in production: the tensor has the right shape and
    the output looks like a track, so nothing errors and nothing is visibly wrong until
    someone verifies against a best track. See ml/inference/track_runtime.py.
    """
    if prev is not None:
        dt_h = (pt.time - prev.time).total_seconds() / 3600.0
        dt_h = dt_h if dt_h > 0 else SIX_HOURLY_STEP_H
        east, north = displacement_km(prev.lat, prev.lon, pt.lat, pt.lon)
        u, v = east / dt_h, north / dt_h
    else:
        # First step of the window has no predecessor inside the window. Zero motion is
        # the honest encoding: "unknown", and the LSTM sees the same value every time so
        # it can learn to discount that position.
        u = v = 0.0
    speed = math.hypot(u, v)

    wind_present = 1.0 if pt.wind_kt is not None else 0.0
    wind = (pt.wind_kt or 0.0) / WIND_SCALE
    # Pressure as a deficit below ambient rather than an absolute: 1002 hPa and 1004 hPa
    # differ by 0.2% as absolutes but by a factor of 1.25 as deficits, which is the
    # physically meaningful comparison.
    pres = ((PRES_REFERENCE_HPA - pt.pres_hpa) / PRES_DEFICIT_SCALE
            if pt.pres_hpa is not None else 0.0)

    doy = pt.time.timetuple().tm_yday
    angle = 2.0 * math.pi * doy / 365.25

    return [
        pt.lat / LAT_SCALE,
        (pt.lon - LON_CENTRE) / LON_HALF_WIDTH,
        math.sin(math.radians(pt.lat)),
        u / MOTION_SCALE_KMH, v / MOTION_SCALE_KMH, speed / MOTION_SCALE_KMH,
        wind, pres,
        math.sin(angle), math.cos(angle),
        wind_present,
    ]


def build_samples(
    storms: list[StormTrack],
    history_steps: int = DEFAULT_HISTORY_STEPS,
    horizons_h: tuple[int, ...] = DEFAULT_HORIZONS_H,
    require_first_horizon: bool = True,
) -> list[TrackSample]:
    """Slide a window over every storm to produce training samples.

    Args:
        history_steps: number of 6-hourly observations fed to the model.
        horizons_h: forecast lead times to predict.
        require_first_horizon: keep only samples that can verify at the shortest lead.
            A sample whose every target is masked carries no gradient at all.

    A target is located by timestamp rather than by index offset, so a gap in the best
    track produces a masked horizon instead of a mislabelled one. This matters: indexing
    ``points[i + 8]`` for +48 h silently uses a +54 h or +72 h observation whenever a
    synoptic time is missing, which corrupts the label without any error.
    """
    samples: list[TrackSample] = []

    for storm in storms:
        pts = storm.points
        by_time = {p.time: p for p in pts}

        for i in range(history_steps - 1, len(pts)):
            window = pts[i - history_steps + 1: i + 1]
            origin = window[-1]

            # Require the history window itself to be contiguous 6-hourly. A window
            # spanning a 24 h data gap is not an 18 h motion history.
            gaps = [(window[k + 1].time - window[k].time).total_seconds() / 3600.0
                    for k in range(len(window) - 1)]
            if any(abs(g - SIX_HOURLY_STEP_H) > 1e-6 for g in gaps):
                continue

            feats = []
            for k, p in enumerate(window):
                prev = window[k - 1] if k > 0 else (pts[i - history_steps]
                                                    if i - history_steps >= 0 else None)
                feats.append(step_features(p, prev))
            history = np.asarray(feats, dtype=np.float32)

            target_km = np.zeros((len(horizons_h), 2), dtype=np.float32)
            mask = np.zeros(len(horizons_h), dtype=np.float32)
            for hi, h in enumerate(horizons_h):
                want = origin.time + _timedelta_hours(h)
                nxt = by_time.get(want)
                if nxt is None:
                    continue
                east, north = displacement_km(origin.lat, origin.lon, nxt.lat, nxt.lon)
                target_km[hi] = (east, north)
                mask[hi] = 1.0

            if require_first_horizon and mask[0] < 1.0:
                continue
            if mask.sum() == 0:
                continue

            samples.append(TrackSample(
                history=history,
                target_km=target_km,
                target_scaled=target_km / DISPLACEMENT_SCALE_KM,
                target_mask=mask,
                origin_lat=origin.lat,
                origin_lon=origin.lon,
                origin_time_iso=origin.time.isoformat(),
                sid=storm.sid,
                season=storm.season,
                name=storm.name,
                subbasin=storm.subbasin,
            ))

    return samples


def _timedelta_hours(h: int):
    from datetime import timedelta
    return timedelta(hours=h)


def stack_samples(samples: list[TrackSample]) -> dict[str, np.ndarray]:
    """Collate samples into batched arrays for training."""
    if not samples:
        raise ValueError("cannot stack an empty sample list")
    return {
        "history": np.stack([s.history for s in samples]),
        "target_scaled": np.stack([s.target_scaled for s in samples]),
        "target_km": np.stack([s.target_km for s in samples]),
        "mask": np.stack([s.target_mask for s in samples]),
        "origin_lat": np.asarray([s.origin_lat for s in samples], dtype=np.float32),
        "origin_lon": np.asarray([s.origin_lon for s in samples], dtype=np.float32),
    }


def describe_samples(samples: list[TrackSample], horizons_h=DEFAULT_HORIZONS_H) -> str:
    """Human-readable audit of a sample set, including per-horizon coverage.

    Printed by the trainer so the effective sample count at +48 h -- usually far smaller
    than the nominal total -- is visible rather than assumed.
    """
    if not samples:
        return "0 samples"
    mask = np.stack([s.target_mask for s in samples])
    km = np.stack([s.target_km for s in samples])
    lines = [f"{len(samples)} samples from {len({s.sid for s in samples})} storms "
             f"(seasons {min(s.season for s in samples)}-{max(s.season for s in samples)})"]
    for hi, h in enumerate(horizons_h):
        n = int(mask[:, hi].sum())
        if n == 0:
            lines.append(f"  +{h:>3}h: no verifying observations")
            continue
        d = np.linalg.norm(km[mask[:, hi] > 0, hi, :], axis=1)
        lines.append(f"  +{h:>3}h: {n:>5} targets, mean displacement "
                     f"{d.mean():>6.1f} km, max {d.max():>6.1f} km")
    return "\n".join(lines)

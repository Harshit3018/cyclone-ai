"""
Non-learned baseline track forecasters.

Every operational track-forecast claim is judged against baselines, not against zero.
A neural network that cannot beat persistence has no demonstrated skill, so these
baselines are not filler -- they are the yardstick, and they are also the only
forecast this platform can honestly display before the network is trained.

Two are provided:

* ``persistence`` (PERS) -- hold the current motion vector constant. The standard
  zero-skill reference for short lead times, and genuinely hard to beat at +6/+12 h.
* ``clip_lite`` (CLIP-like) -- relax the current motion toward North Indian Ocean
  climatological motion as lead time grows.

MEASURED, 2024-2025 held-out seasons: ``clip_lite`` is WORSE than ``persistence`` at every
lead time (-10.5% at +6 h to -32.7% at +48 h). The intuition it was built on -- that
persistence degrades past 24 h because it cannot recurve, so blending toward basin-mean
motion should help -- does not survive contact with the data, because the *specified*
bearings below are not accurate enough for the blend to pay off. It is kept for
reproducibility and comparison, and must not be served as a preferred forecast. Use the
fitted CLIPER in ml/baselines/cliper.py, which does beat persistence (+8.5% at +24 h,
+17.1% at +48 h) because its coefficients come from this basin's own best track.

IMPORTANT -- on the climatology used by ``clip_lite``:
    The climatological bearings and speed in CLIMATOLOGY below are *specified priors*
    reflecting the well-documented qualitative behaviour of NIO systems (WNW/NW
    motion at low latitude, poleward recurvature above roughly 18N). They are NOT
    regression coefficients fitted to observations. A true CLIPER is a multiple
    regression on CLImatology and PERsistence predictors fitted to a historical best
    track archive; that now exists in ml/baselines/cliper.py, so ``clip_lite`` is
    superseded rather than merely provisional.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .physics import (
    TRANSLATION_SPEED_MAX_KPH,
    destination_point,
    haversine_km,
    initial_bearing_deg,
)

DEFAULT_HORIZONS = (6, 12, 24, 48, 72)

# Verified forecast error per lead time, in km, MEASURED by running these exact functions
# over the held-out 2024-2025 IBTrACS North Indian Ocean test cases. Regenerate with:
#
#     python -m ml.evaluation.eval_analytic_baselines
#
# These replace a hand-chosen linear cone (15 km + 2.2 km/h x lead) that was flagged as a
# placeholder and turned out to understate the real error by a factor of five at +24 h.
# That mattered as soon as the trained model began reporting its measured error in the same
# API response: the guessed number made persistence look four times more confident than a
# model that is in fact more accurate.
#
# 'p90_km' is the radius that actually contains 9 outcomes in 10, and is what the displayed
# cone uses. A cone drawn at the mean error excludes roughly 40% of outcomes while looking
# authoritative.
PERSISTENCE_ERROR_KM = {
    6: {"mean_km": 41.8, "p90_km": 81.8, "n": 200},
    12: {"mean_km": 83.9, "p90_km": 167.4, "n": 186},
    24: {"mean_km": 180.1, "p90_km": 358.8, "n": 159},
    48: {"mean_km": 410.7, "p90_km": 715.0, "n": 113},
}

# MEASURED RESULT WORTH READING BEFORE USING clip_lite:
# blending toward the specified climatology makes the forecast WORSE than plain
# persistence at every lead time -- -10.5% at +6 h, -20.2% at +12 h, -29.9% at +24 h and
# -32.7% at +48 h on the test split. Its cross-track bias reaches +185 km at +48 h and its
# speed bias +150 km, i.e. it systematically pushes storms right of track and too far.
# The module docstring's assumption that relaxing toward basin-mean motion would help past
# 24 h is simply not supported: the specified bearings in climatological_bearing_deg are
# not close enough to real NIO motion for the blend to pay off.
#
# Use the fitted CLIPER in ml/baselines/cliper.py instead -- a real regression on this
# basin's best track, which does beat persistence (8.5% at +24 h, 17.1% at +48 h).
# clip_lite is retained only so this measurement stays reproducible and so the comparison
# can be shown; it must not be served as a preferred forecast.
CLIP_LIKE_ERROR_KM = {
    6: {"mean_km": 46.2, "p90_km": 85.8, "n": 200},
    12: {"mean_km": 100.9, "p90_km": 186.4, "n": 186},
    24: {"mean_km": 234.0, "p90_km": 395.0, "n": 159},
    48: {"mean_km": 545.0, "p90_km": 831.9, "n": 113},
}

ERROR_SOURCE = ("measured great-circle error on held-out IBTrACS North Indian Ocean "
                "seasons 2024-2025")


def _radius_km(table: dict, horizon_h: float) -> float | None:
    """Verified 90th-percentile error at a lead time, or None if it was never measured.

    Returns None rather than extrapolating. The measurement covers +6 to +48 h; a +72 h
    number produced by extending a trend would carry the authority of a measurement
    without being one, and DEFAULT_HORIZONS includes 72 h.
    """
    entry = table.get(int(horizon_h))
    return entry["p90_km"] if entry else None


# e-folding lead time over which the persistence motion vector is relaxed toward
# climatology in clip_lite. At +24 h the forecast is roughly half persistence,
# half climatology.
CLIMATOLOGY_EFOLD_H = 24.0

CLIMATOLOGICAL_SPEED_KPH = 14.0


def climatological_bearing_deg(lat: float) -> float:
    """Mean NIO storm heading at a given latitude, degrees clockwise from north.

    Piecewise and deliberately coarse -- see the module docstring. Captures the
    qualitative pattern only: west-northwest at low latitude, turning poleward and
    then recurving northeast as the system approaches the subtropical ridge axis.
    """
    if lat < 12.0:
        return 295.0   # WNW
    if lat < 18.0:
        return 320.0   # NW
    if lat < 22.0:
        return 350.0   # NNW
    return 25.0        # NNE -- recurved


@dataclass
class MotionVector:
    """Current storm motion, derived from recent observed positions."""
    bearing_deg: float
    speed_kph: float
    hours_used: float
    points_used: int
    capped: bool = False


def _to_hours(ts: str | datetime | None) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.timestamp() / 3600.0


def estimate_motion(history: list[dict], window_hours: float = 12.0) -> MotionVector | None:
    """Derive the current motion vector from the most recent observed positions.

    Uses the displacement across roughly `window_hours` rather than the single last
    pair. Best-track positions are rounded to 0.1 degrees, so a single 3-hourly pair
    can carry a large relative error in bearing; averaging over 12 h suppresses that
    without smoothing away a genuine turn.

    Returns None if there is not enough positional history to define motion.
    """
    pts = [p for p in history if p.get("latitude") is not None and p.get("longitude") is not None]
    if len(pts) < 2:
        return None

    times = [_to_hours(p.get("timestamp")) for p in pts]
    if any(t is None for t in times):
        # No usable timestamps: assume the 3-hourly spacing of IMD best-track data.
        times = [3.0 * i for i in range(len(pts))]

    end = len(pts) - 1
    start = end - 1
    # Walk back until the window is covered or history runs out.
    while start > 0 and (times[end] - times[start]) < window_hours:
        start -= 1

    dt = times[end] - times[start]
    if dt <= 0:
        return None

    dist = haversine_km(pts[start]["latitude"], pts[start]["longitude"],
                        pts[end]["latitude"], pts[end]["longitude"])
    bearing = initial_bearing_deg(pts[start]["latitude"], pts[start]["longitude"],
                                 pts[end]["latitude"], pts[end]["longitude"])
    speed = dist / dt

    capped = False
    if speed > TRANSLATION_SPEED_MAX_KPH:
        # A speed this high in the NIO means bad or duplicated positions in the
        # history, not a real storm. Cap rather than propagate it for 72 h.
        speed = TRANSLATION_SPEED_MAX_KPH
        capped = True

    return MotionVector(
        bearing_deg=bearing, speed_kph=speed,
        hours_used=dt, points_used=end - start + 1, capped=capped,
    )


def _blend_bearing(b1: float, b2: float, w2: float) -> float:
    """Interpolate between two bearings the short way round the compass."""
    diff = ((b2 - b1 + 180.0) % 360.0) - 180.0
    return (b1 + diff * w2) % 360.0


def _forecast(history: list[dict], horizons, use_climatology: bool) -> dict | None:
    motion = estimate_motion(history)
    if motion is None:
        return None

    error_table = CLIP_LIKE_ERROR_KM if use_climatology else PERSISTENCE_ERROR_KM
    last = history[-1]
    lat, lon = last["latitude"], last["longitude"]

    points = []
    prev_lat, prev_lon, prev_h = lat, lon, 0.0
    unmeasured = []

    for h in horizons:
        step = h - prev_h
        if use_climatology:
            # Weight on climatology grows with total lead time, not with step size.
            w_clim = 1.0 - pow(2.718281828, -h / CLIMATOLOGY_EFOLD_H)
            bearing = _blend_bearing(
                motion.bearing_deg, climatological_bearing_deg(prev_lat), w_clim
            )
            speed = motion.speed_kph * (1 - w_clim) + CLIMATOLOGICAL_SPEED_KPH * w_clim
        else:
            bearing = motion.bearing_deg
            speed = motion.speed_kph

        nlat, nlon = destination_point(prev_lat, prev_lon, bearing, speed * step)
        radius = _radius_km(error_table, h)
        point = {
            "forecast_hour": h,
            "latitude": round(nlat, 2),
            "longitude": round(nlon, 2),
            "heading_deg": round(bearing, 1),
            "translation_speed_kph": round(speed, 1),
        }
        if radius is None:
            # Explicitly null, not absent and not guessed, so a consumer cannot mistake a
            # missing measurement for a small one.
            point["uncertainty_radius_km"] = None
            unmeasured.append(h)
        else:
            point["uncertainty_radius_km"] = round(radius, 1)
            point["mean_error_km"] = round(error_table[int(h)]["mean_km"], 1)
            point["verified_cases"] = error_table[int(h)]["n"]
        points.append(point)
        prev_lat, prev_lon, prev_h = nlat, nlon, h

    measured = [h for h in horizons if h not in unmeasured]
    uncertainty_method = (
        f"90th-percentile great-circle error per lead time, {ERROR_SOURCE}"
        + (f" ({error_table[int(measured[0])]['n']} verifying cases at +{measured[0]}h)"
           if measured else "")
        + (f". Not measured at +{', +'.join(str(h) for h in unmeasured)}h, reported as "
           f"null rather than extrapolated." if unmeasured else "")
    )

    return {
        "forecast_points": points,
        "horizons_hours": list(horizons),
        "current_motion": {
            "bearing_deg": round(motion.bearing_deg, 1),
            "speed_kph": round(motion.speed_kph, 1),
            "derived_over_hours": round(motion.hours_used, 1),
            "positions_used": motion.points_used,
            "speed_was_capped": motion.capped,
        },
        "uncertainty_method": uncertainty_method,
    }


def persistence(history: list[dict], horizons=DEFAULT_HORIZONS) -> dict | None:
    """Constant-motion (persistence) track forecast. The zero-skill reference."""
    out = _forecast(history, horizons, use_climatology=False)
    if out is not None:
        out["model_name"] = "Persistence (PERS)"
        out["model_status"] = "analytic_baseline"
        out["method"] = "Current motion vector held constant along a great circle."
        out["verified_skill"] = {
            "reference": "self (this is the zero-skill reference)",
            "mean_error_km": {str(h): v["mean_km"]
                              for h, v in PERSISTENCE_ERROR_KM.items()},
            "source": ERROR_SOURCE,
        }
    return out


def clip_lite(history: list[dict], horizons=DEFAULT_HORIZONS) -> dict | None:
    """Persistence relaxed toward NIO climatological motion as lead time grows.

    Measured worse than plain persistence at every lead time -- see the module docstring.
    Retained for reproducibility, not for serving.
    """
    out = _forecast(history, horizons, use_climatology=True)
    if out is not None:
        out["model_name"] = "Persistence + Climatology (CLIP-like)"
        out["model_status"] = "analytic_baseline_superseded"
        out["method"] = (
            "Current motion relaxed toward specified North Indian Ocean climatological "
            f"motion with a {CLIMATOLOGY_EFOLD_H:.0f} h e-folding lead time. "
            "Climatology is a specified prior, not a regression fitted to best-track "
            "data, so this is not a true CLIPER."
        )
        out["verified_skill"] = {
            "reference": "Persistence (PERS)",
            "mean_error_km": {str(h): v["mean_km"] for h, v in CLIP_LIKE_ERROR_KM.items()},
            "skill_vs_persistence_pct": {
                str(h): round(100.0 * (PERSISTENCE_ERROR_KM[h]["mean_km"]
                                       - v["mean_km"]) / PERSISTENCE_ERROR_KM[h]["mean_km"], 1)
                for h, v in CLIP_LIKE_ERROR_KM.items() if h in PERSISTENCE_ERROR_KM
            },
            "source": ERROR_SOURCE,
            "note": ("Negative skill at every lead time: this method is worse than "
                     "persistence. Superseded by the fitted CLIPER in "
                     "ml/baselines/cliper.py."),
        }
    return out

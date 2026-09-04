"""Risk along the forecast track, and the measured facts that support it.

The problem this fixes
---------------------
The platform's risk endpoint scored the storm's *current* position and nothing else. It read
``dist_to_coast_km`` off the latest observed track point and returned one alert level. But
"how far from land is it now" is a fact anyone can read off a map; the decision a district
officer has to make is "when does this reach my coast, and how sure are you". A risk engine
that ignores the forecast it just produced is not decision support.

Two of the five inputs to the old score were also invented at the call site --
``ri_probability=0.1`` and ``track_confidence=0.5``, together 30% of the weight. Neither was
measured, neither varied between storms, and neither was labelled as a constant. Everything
here is either derived from data or reported as null.

What is measured and what is not
--------------------------------
Measured:
  * distance to the nearest coast at every forecast position, from ``ml.geo.coastline``
    (mean absolute error 8.7 km against IBTrACS -- see ``MEASURED_ACCURACY``);
  * the observed 24-hour intensity change from the storm's own wind history, against the
    30 kt / 24 h threshold IMD and NHC both use to define rapid intensification;
  * whether the 90th-percentile track-error cone -- itself measured on held-out seasons --
    reaches land at each lead time.

Not measured, and therefore reported as null rather than guessed:
  * *forecast* rapid intensification. There is no trained intensity-forecast head, so the
    probability that this storm will rapidly intensify is unknown. The old code multiplied
    a number derived purely from current wind speed (``(wind - 100) / 150``) and called it
    an RI probability, which relabels intensity as intensification.
  * intensity at future positions. Wind is carried forward unchanged and labelled
    ``persisted``, because a constant is a defensible null hypothesis and a fabricated
    intensification curve is not.
"""
from __future__ import annotations

from datetime import datetime

from ..geo.coastline import GEOMETRY_NOTE, MEASURED_ACCURACY, nearest_land

KT_PER_KPH = 1.0 / 1.852

RI_THRESHOLD_KT_24H = 30.0
"""Rapid intensification: an increase of at least 30 kt in 24 h.

The operational definition used by IMD and by NHC, and the 95th percentile of observed
24-hour intensity change in the Atlantic. Stated as a constant here so the number in the
response can be traced to a definition rather than to a tuning pass.
"""

COAST_LABELS = {
    "mainland": "mainland coast",
    "arabia_horn": "Arabian peninsula / Horn of Africa",
    "sumatra": "Sumatra",
    "sumatra_islands": "Simeulue / Nias",
    "gulf_of_thailand": "Gulf of Thailand coast",
    "sri_lanka": "Sri Lanka",
    "andaman_nicobar": "Andaman and Nicobar Islands",
    "maldives": "Maldives",
    "lakshadweep": "Lakshadweep",
    "socotra": "Socotra",
}


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


def intensity_trend(history: list[dict], window_h: float = 24.0) -> dict:
    """Observed intensity change over the trailing window, in knots.

    This is an observation, not a forecast: it says what the storm has just done. Reported
    because it is the only rapid-intensification information in the system that is real,
    and because a storm that gained 35 kt in the last 24 h is a different decision problem
    from one that has been steady, even when both currently read 120 km/h.

    Returns ``available: False`` when the wind record is too short or too sparse to support
    the statement, rather than a change of 0.0 -- "no data" and "no change" are opposite
    conclusions for a forecaster.
    """
    pts = [p for p in history if p.get("wind_kph") is not None]
    if len(pts) < 2:
        return {"available": False,
                "reason": "fewer than two observations carry a wind value"}

    times = [_to_hours(p.get("timestamp")) for p in pts]
    if any(t is None for t in times):
        return {"available": False,
                "reason": "wind observations have no usable timestamps"}

    end = len(pts) - 1
    start = end
    while start > 0 and (times[end] - times[start]) < window_h:
        start -= 1

    span_h = times[end] - times[start]
    if span_h <= 0:
        return {"available": False, "reason": "observations span zero hours"}

    change_kt = (pts[end]["wind_kph"] - pts[start]["wind_kph"]) * KT_PER_KPH
    # Scale a short window up to 24 h so the threshold comparison is like-for-like, but
    # refuse to extrapolate from less than half a day: a 3-hour trend multiplied by eight
    # is an artefact, not a rate.
    if span_h < 12.0:
        return {"available": False,
                "reason": f"only {span_h:.0f} h of wind history; too short for a 24 h rate",
                "observed_change_kt": round(change_kt, 1),
                "window_hours": round(span_h, 1)}

    rate_24h = change_kt * (24.0 / span_h)
    return {
        "available": True,
        "observed_change_kt": round(change_kt, 1),
        "window_hours": round(span_h, 1),
        "rate_kt_per_24h": round(rate_24h, 1),
        "threshold_kt_per_24h": RI_THRESHOLD_KT_24H,
        "rapidly_intensifying": bool(rate_24h >= RI_THRESHOLD_KT_24H),
        "basis": "observed best-track wind history",
    }


def coastal_profile(forecast_points: list[dict]) -> list[dict]:
    """Distance to the nearest coast at each forecast position.

    ``cone_reaches_coast`` is the operationally useful column. It is true when the
    90th-percentile track error at that lead time is at least the distance to land, i.e.
    when landfall by that hour is inside the verified error of the forecast. It is null
    where no cone was measured -- at +72 h, for instance, which is in the horizon list but
    was never verified -- because "we cannot say" and "no" are not the same answer.
    """
    rows = []
    for p in forecast_points:
        lat, lon = p.get("latitude"), p.get("longitude")
        if lat is None or lon is None:
            continue
        group, dist = nearest_land(lat, lon)
        radius = p.get("uncertainty_radius_km")
        rows.append({
            "forecast_hour": p.get("forecast_hour"),
            "latitude": lat,
            "longitude": lon,
            "distance_to_land_km": round(dist, 1),
            "nearest_coast": group,
            "nearest_coast_label": COAST_LABELS.get(group, group),
            "uncertainty_radius_km": radius,
            "cone_reaches_coast": None if radius is None else bool(radius >= dist),
        })
    return rows


def closest_approach(profile: list[dict]) -> dict | None:
    """The forecast position that comes nearest to land."""
    if not profile:
        return None
    row = min(profile, key=lambda r: r["distance_to_land_km"])
    return {
        "forecast_hour": row["forecast_hour"],
        "distance_to_land_km": row["distance_to_land_km"],
        "nearest_coast": row["nearest_coast"],
        "nearest_coast_label": row["nearest_coast_label"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
    }


def landfall_window(profile: list[dict]) -> dict:
    """When landfall becomes possible, and when the forecast track itself reaches land.

    Deliberately two different answers to two different questions.

    ``possible_from_hour`` is the first lead time whose verified error cone reaches the
    coast: by then, landfall is within the forecast's demonstrated accuracy. This is the
    number an evacuation decision hangs on, and it is always earlier than the track's own
    arrival -- which is the point. Warning on the centre line alone is warning late.

    ``track_arrives_hour`` is the first lead time at which the forecast centre is within
    the coastline geometry's own resolution of land. That resolution is roughly 20 km at
    the 90th percentile, so anything tighter would be false precision.
    """
    resolution_km = float(MEASURED_ACCURACY["p90_abs_error_km"])

    # Sort by lead time rather than trusting the caller's ordering: "the first hour at
    # which" is only meaningful on a monotonic sequence, and an out-of-order list would
    # silently report a later landfall as the earliest one.
    ordered = sorted(profile, key=lambda r: (r["forecast_hour"] is None,
                                             r["forecast_hour"] or 0))

    # `cone_reaches_coast` is None where no cone was verified. None is falsy, so those
    # lead times are skipped rather than counted as "no" -- which is the intent.
    possible = next((r for r in ordered if r["cone_reaches_coast"]), None)
    arrives = next((r for r in ordered
                    if r["distance_to_land_km"] <= resolution_km), None)

    if possible is None and arrives is None:
        verdict = ("No landfall within the forecast horizon: the verified error cone does "
                   "not reach any coast at any lead time reported.")
    elif arrives is None:
        verdict = (f"Landfall is inside the verified error of the forecast from "
                   f"+{possible['forecast_hour']} h onward, on the "
                   f"{possible['nearest_coast_label']}, but the forecast centre track "
                   f"stays offshore through the horizon.")
    elif possible is None:
        verdict = (f"Forecast centre reaches the {arrives['nearest_coast_label']} at "
                   f"+{arrives['forecast_hour']} h. No cone was verified at the earlier "
                   f"lead times, so the earliest hour landfall becomes possible is unknown.")
    else:
        verdict = (f"Forecast centre reaches the {arrives['nearest_coast_label']} at "
                   f"+{arrives['forecast_hour']} h; landfall is already inside the "
                   f"verified error cone from +{possible['forecast_hour']} h onward.")

    return {
        "possible_from_hour": possible["forecast_hour"] if possible else None,
        "possible_coast": possible["nearest_coast_label"] if possible else None,
        "track_arrives_hour": arrives["forecast_hour"] if arrives else None,
        "track_arrives_coast": arrives["nearest_coast_label"] if arrives else None,
        "geometry_resolution_km": round(resolution_km, 1),
        "verdict": verdict,
    }


def build(forecast_points: list[dict], history: list[dict]) -> dict:
    """Assemble the full forecast-track risk context.

    Returns the geometry note alongside the numbers by design. A distance to coast quoted
    without its 8.7 km mean error invites the reader to treat 40 km and 48 km as different
    answers when the geometry cannot distinguish them.
    """
    profile = coastal_profile(forecast_points)
    return {
        "coastal_profile": profile,
        "closest_approach": closest_approach(profile),
        "landfall": landfall_window(profile),
        "intensity_trend": intensity_trend(history),
        "intensity_at_forecast_positions": "persisted",
        "intensity_forecast_available": False,
        "intensity_forecast_note": (
            "Wind is carried forward unchanged at every forecast position. There is no "
            "trained intensity-forecast head in this system, so any intensification curve "
            "shown here would be invented."),
        "geometry": {
            "note": GEOMETRY_NOTE,
            "mean_absolute_error_km": MEASURED_ACCURACY["mae_km"],
            "p90_absolute_error_km": MEASURED_ACCURACY["p90_abs_error_km"],
            "reference": MEASURED_ACCURACY["reference"],
            "validated_on_positions": MEASURED_ACCURACY["positions"],
        },
    }

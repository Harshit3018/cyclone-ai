"""
Physical relations and plausibility limits for tropical cyclones.

This module exists so that every number the platform reports can be checked against
something physical before it reaches a user. An untrained (or badly trained) network
will happily emit 885 hPa alongside a 1 km/h wind, or move a storm 1100 km in six
hours. Those outputs are not "uncertain" -- they are impossible, and shipping them
next to a real IMD category is what destroys credibility.

Nothing here is learned. These are published relations and documented sanity bounds,
which is exactly why they are useful as a check on a learned model.

References
----------
Atkinson, G.D. and Holliday, C.R. (1977). "Tropical cyclone minimum sea level
    pressure / maximum sustained wind relationship for the western North Pacific."
    Monthly Weather Review, 105, 421-427.
    The Dvorak-era wind-pressure relation, widely applied in the North Indian Ocean.
Koba, H. et al. (1990) provide the alternative table used operationally by RSMC
    New Delhi. The two agree closely over the Cyclonic Storm to VSCS range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- Unit conversion -------------------------------------------------------------

KT_TO_KPH = 1.852
KPH_TO_KT = 1.0 / KT_TO_KPH

# --- Physical bounds -------------------------------------------------------------

# Lowest central pressure ever measured in a tropical cyclone: 870 hPa
# (Typhoon Tip, 12 Oct 1979). Nothing below this is physical.
WORLD_RECORD_MIN_PRESSURE_HPA = 870.0

# Ambient/environmental pressure assumed by the wind-pressure relation. A single
# constant is a simplification: the true value varies with the monsoon trough and
# season, and using a constant biases weak systems. Adequate for a bounds check.
ENV_PRESSURE_HPA = 1010.0

# Translation speed limits, calibrated against the IBTrACS v04r01 North Indian Ocean
# record (1990-2025, 4249 consecutive 6-hourly segments over 229 quality-controlled
# storms). Measured distribution:
#
#     p50 12.0    p90 23.1    p95 27.1    p99 35.6    p99.9 51.9    max 57.3 km/h
#
# The fastest segments are real, named storms recurving into the mid-latitude
# westerlies at 19-25N: SITRANG (2022) 57.3 km/h and MOCHA (2023) 52.2 km/h.
# An earlier revision of this file asserted a 40 km/h "physical ceiling" from the
# literature's typical-speed range; that would have declared both of those storms
# impossible. Bounds that reject the best track are worse than no bounds, so the ceiling
# is now set above the observed maximum with headroom, and the suspect level at p99.
TRANSLATION_SPEED_SUSPECT_KPH = 36.0
TRANSLATION_SPEED_MAX_KPH = 65.0

# Lower bound, and the reason for it is resolution rather than dynamics. IBTrACS NI
# positions are archived on a 0.1 degree grid in every era (only ten distinct fractional
# values appear in 4520 points), so one grid cell is about 11 km and 2.52% of
# consecutive 6-hourly pairs are byte-identical. Motion slower than roughly 1.9 km/h
# over a 6 h step therefore cannot be represented in the source data at all.
#
# A forecast that moves a storm less than one grid cell in a full day is not describing a
# slow storm; it is claiming precision finer than the observations it was trained on,
# which is the signature of a collapsed regression head returning its mean output.
# 0.5 km/h over 24 h is 12 km, just over one grid cell. Against the real record this
# flags 0.5% of 24 h windows -- exactly the grid-quantised ones -- which is the accepted
# false-positive cost.
TRANSLATION_SPEED_MIN_KPH = 0.5
MIN_SPAN_FOR_MOTION_CHECK_H = 24.0

# Best-track position quantum, degrees. Used to express the motion floor in the same
# units the archive is stored in.
BEST_TRACK_GRID_DEG = 0.1

# Rapid intensification is defined by IMD/NHC as >= 30 kt increase in 24 h.
# Sustained rates far above this are not physical for a 6 h step.
MAX_INTENSIFICATION_KT_PER_24H = 60.0

EARTH_RADIUS_KM = 6371.0088

# One grid cell of latitude in km -- the smallest displacement the archive can express.
KM_PER_GRID_CELL = BEST_TRACK_GRID_DEG * math.pi * EARTH_RADIUS_KM / 180.0


# --- Wind <-> pressure -----------------------------------------------------------

def pressure_from_wind(wind_kt: float, env_pressure_hpa: float = ENV_PRESSURE_HPA) -> float:
    """Central pressure (hPa) implied by a maximum sustained wind, Atkinson-Holliday.

    Inverts ``V = 6.7 * (Penv - Pc) ** 0.644``.
    """
    if wind_kt <= 0:
        return env_pressure_hpa
    deficit = (wind_kt / 6.7) ** (1.0 / 0.644)
    return max(WORLD_RECORD_MIN_PRESSURE_HPA, env_pressure_hpa - deficit)


def wind_from_pressure(pressure_hpa: float, env_pressure_hpa: float = ENV_PRESSURE_HPA) -> float:
    """Maximum sustained wind (kt) implied by a central pressure, Atkinson-Holliday."""
    deficit = env_pressure_hpa - pressure_hpa
    if deficit <= 0:
        return 0.0
    return 6.7 * (deficit ** 0.644)


# --- Geodesy --------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from N."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float
                      ) -> tuple[float, float]:
    """Point reached by travelling `distance_km` along `bearing_deg` from (lat, lon).

    Great-circle, not a flat lat/lon offset. The flat version understates eastward
    displacement badly at higher latitudes and lets a forecast drift off a physical
    path over 72 h.
    """
    d = distance_km / EARTH_RADIUS_KM
    b = math.radians(bearing_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)

    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * math.sin(p2),
    )
    # Normalise longitude to [-180, 180)
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


# --- Plausibility checking -------------------------------------------------------

@dataclass
class PlausibilityReport:
    """Outcome of checking a forecast or estimate against physical limits."""
    ok: bool = True
    violations: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.violations.append(message)

    def as_dict(self) -> dict:
        return {"physically_plausible": self.ok, "violations": list(self.violations)}


def check_intensity(wind_kph: float, pressure_hpa: float,
                    tolerance_hpa: float = 25.0) -> PlausibilityReport:
    """Check a (wind, pressure) pair for internal physical consistency.

    `tolerance_hpa` is deliberately loose. The wind-pressure relation is itself an
    empirical fit with real scatter (size, latitude and environmental pressure all
    matter), so a tight tolerance would reject valid states. It is here to catch
    output that is wrong by a category or more, not to enforce the relation exactly.

    Checked against IMD best-track peaks, Atkinson-Holliday estimates NIO central
    pressures 6-22 hPa deeper than observed -- it is a western North Pacific fit and
    NIO storms are typically smaller:

        FANI 2019      127 kt   observed 932 hPa   relation 914 hPa
        AMPHAN 2020    140 kt   observed 920 hPa   relation 898 hPa
        TAUKTAE 2021   100 kt   observed 950 hPa   relation 944 hPa
        BIPARJOY 2023   90 kt   observed 966 hPa   relation 954 hPa

    All four fall inside the 25 hPa default, which is how that number was chosen.
    Swapping in the Koba et al. (1990) NIO table would allow a tighter bound.
    """
    report = PlausibilityReport()

    if pressure_hpa < WORLD_RECORD_MIN_PRESSURE_HPA:
        report.fail(
            f"central pressure {pressure_hpa:.1f} hPa is below the world-record "
            f"minimum of {WORLD_RECORD_MIN_PRESSURE_HPA:.0f} hPa"
        )
    if wind_kph < 0:
        report.fail(f"wind speed {wind_kph:.1f} km/h is negative")

    expected = pressure_from_wind(wind_kph * KPH_TO_KT)
    if abs(expected - pressure_hpa) > tolerance_hpa:
        report.fail(
            f"wind {wind_kph:.0f} km/h implies about {expected:.0f} hPa by the "
            f"Atkinson-Holliday relation, but pressure is {pressure_hpa:.0f} hPa "
            f"({abs(expected - pressure_hpa):.0f} hPa inconsistent)"
        )
    return report


def check_track(points: list[dict], current_lat: float, current_lon: float
                ) -> PlausibilityReport:
    """Check a forecast track for impossible motion.

    `points` must be ordered by increasing forecast_hour and carry
    'forecast_hour', 'latitude' and 'longitude'.
    """
    report = PlausibilityReport()
    prev_lat, prev_lon, prev_h = current_lat, current_lon, 0.0

    for p in points:
        h = float(p["forecast_hour"])
        dt = h - prev_h
        if dt <= 0:
            report.fail(f"forecast hours are not increasing at +{h:g}h")
            continue

        dist = haversine_km(prev_lat, prev_lon, p["latitude"], p["longitude"])
        speed = dist / dt
        if speed > TRANSLATION_SPEED_MAX_KPH:
            report.fail(
                f"+{prev_h:g}h to +{h:g}h implies a translation speed of "
                f"{speed:.0f} km/h ({dist:.0f} km in {dt:g} h); the physical ceiling "
                f"for this basin is about {TRANSLATION_SPEED_MAX_KPH:.0f} km/h"
            )
        prev_lat, prev_lon, prev_h = p["latitude"], p["longitude"], h

    # A lower bound as well as an upper one, justified by the resolution of the best
    # track rather than by a claim about how slowly storms can move. Storms do stall --
    # 2.4% of real 24 h windows in this basin average under 2 km/h, and 0.8% are exactly
    # zero because the archive quantises position to 0.1 degrees. What no real track does
    # is stay inside a single 11 km grid cell for a day while being reported to
    # four-decimal precision. Checking mean speed over the whole span rather than per
    # segment allows a genuine short stall while still rejecting a track that barely
    # moves, which is exactly what an untrained network produces: it regresses towards
    # its mean output and returns near-identical coordinates at every horizon. Without
    # this the most obvious failure mode passed the check.
    if points and prev_h >= MIN_SPAN_FOR_MOTION_CHECK_H:
        total = haversine_km(current_lat, current_lon, prev_lat, prev_lon)
        mean_speed = total / prev_h
        if mean_speed < TRANSLATION_SPEED_MIN_KPH:
            report.fail(
                f"track moves only {total:.0f} km in {prev_h:g} h "
                f"({mean_speed:.2f} km/h mean), less than the {BEST_TRACK_GRID_DEG:g} "
                f"degree ({KM_PER_GRID_CELL:.0f} km) quantum of the best-track archive; "
                f"the forecast is effectively stationary"
            )

    return report

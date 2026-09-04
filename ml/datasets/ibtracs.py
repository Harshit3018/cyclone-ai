"""
IBTrACS North Indian Ocean best-track loader.

Reads the raw IBTrACS v04r01 CSV and turns it into clean, per-storm tracks in real
physical units. This is the only place in the codebase that knows the IBTrACS column
layout or its sentinel conventions; everything downstream consumes ``StormTrack``.

Why this dataset
----------------
IBTrACS is the WMO-archived merge of every agency's best-track. For the North Indian
Ocean the authoritative agency is RSMC New Delhi (the IMD), whose columns are carried
verbatim as ``NEWDELHI_WIND`` / ``NEWDELHI_PRES``. Using IMD's own analysis rather than
a US re-analysis matters here: the intensity scales differ (IMD reports 3-minute
sustained wind, the JTWC 1-minute), so mixing them silently introduces a ~12% bias.
We prefer New Delhi, fall back to the WMO merge, and record which was used.

Data reality, stated plainly
----------------------------
This basin is data-poor. There are only ~370 storms since 1990, and a 6-hourly track
of a 5-day storm yields ~20 points, most of which are consumed by the input history and
the forecast lead. That is a few thousand training samples, not a few million. Any claim
that a deep network "learns cyclone dynamics" from this is not supportable; what it can
do is learn a basin-specific motion prior, which is precisely what CLIPER does and is a
fair thing to try to beat.

References
----------
Knapp, K.R. et al. (2010). "The International Best Track Archive for Climate
    Stewardship (IBTrACS)." Bull. Amer. Meteor. Soc., 91, 363-376.
Gahtan, J. et al. (2024). IBTrACS Project, Version 4r01. NOAA NCEI.
    https://doi.org/10.25921/82ty-9e16
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("cyclone_ai")

# --- IBTrACS conventions ---------------------------------------------------------

# IBTrACS writes missing numerics as an empty field or as a space-padded blank. Some
# derived columns additionally use -999 / -1. Treat all of them as absent rather than
# letting a -999 kt wind become a training target.
_MISSING = {"", "-999", "-999.0", "-1", "-1.0", "NOT_NAMED", "UNNAMED"}

# Only 'main' tracks are the real storm. 'spur-merge' rows are duplicate fragments from
# a merged track and would appear as teleporting positions if kept.
VALID_TRACK_TYPE = "main"

# NATURE codes: TS tropical storm, DS disturbance, ET extratropical, MX mixed,
# NR not reported. A tropical-cyclone motion model should not be trained on the
# extratropical phase, where the steering physics is entirely different.
TROPICAL_NATURES = frozenset({"TS", "DS"})

# Synoptic hours. IBTrACS carries 3-hourly rows for some agencies and eras; mixing
# 3-hourly and 6-hourly rows into one sequence makes the timestep meaningless, so we
# keep the 6-hourly synoptic times that every agency reports.
SYNOPTIC_HOURS = (0, 6, 12, 18)

# 1990 onward: geostationary coverage of the basin is continuous, IMD's own best-track
# is populated, and the satellite-era intensity record is comparable across years.
DEFAULT_MIN_SEASON = 1990

# North Indian Ocean domain. Anything outside is a parse error or a mislabelled basin.
LAT_BOUNDS = (-5.0, 32.0)
LON_BOUNDS = (40.0, 105.0)

# Largest credible 6-hourly displacement. 40 km/h over 6 h is 240 km; allow headroom
# for a genuinely fast recurving storm and reject anything beyond as a bad position.
MAX_STEP_DISPLACEMENT_KM = 350.0

EARTH_RADIUS_KM = 6371.0088


def _clean(value: str) -> str | None:
    v = value.strip()
    return None if v in _MISSING else v


def _to_float(value: str) -> float | None:
    v = _clean(value)
    if v is None:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if not math.isfinite(f) else f


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


# --- Records --------------------------------------------------------------------

@dataclass
class TrackPoint:
    """One best-track observation, in real physical units."""
    time: datetime
    lat: float                    # degrees north
    lon: float                    # degrees east
    wind_kt: float | None = None  # maximum sustained wind
    pres_hpa: float | None = None
    wind_source: str | None = None   # 'newdelhi' | 'wmo' | 'usa'
    dist2land_km: float | None = None
    nature: str = "TS"

    @property
    def hours_since_epoch(self) -> float:
        return self.time.replace(tzinfo=timezone.utc).timestamp() / 3600.0


@dataclass
class StormTrack:
    """A single storm's quality-controlled 6-hourly track."""
    sid: str
    name: str
    season: int
    subbasin: str                 # 'BB' Bay of Bengal, 'AS' Arabian Sea, ...
    points: list[TrackPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def peak_wind_kt(self) -> float | None:
        winds = [p.wind_kt for p in self.points if p.wind_kt is not None]
        return max(winds) if winds else None

    @property
    def min_pres_hpa(self) -> float | None:
        pres = [p.pres_hpa for p in self.points if p.pres_hpa is not None]
        return min(pres) if pres else None

    def __repr__(self) -> str:
        pk = f"{self.peak_wind_kt:.0f} kt" if self.peak_wind_kt is not None else "no wind"
        return (f"<StormTrack {self.name or self.sid} {self.season} {self.subbasin} "
                f"{len(self.points)} pts, peak {pk}>")


@dataclass
class LoadReport:
    """What the loader kept and what it threw away, and why.

    Emitted so that the size of the training set is never a mystery. A silent loader
    that returns 2000 samples instead of 8000 is indistinguishable from a working one
    until the model underperforms for reasons nobody can explain.
    """
    rows_read: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    storms_kept: int = 0
    storms_dropped_short: int = 0
    points_kept: int = 0
    wind_sources: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def summary(self) -> str:
        lines = [
            f"IBTrACS load: {self.rows_read} rows read -> "
            f"{self.points_kept} points in {self.storms_kept} storms",
        ]
        if self.storms_dropped_short:
            lines.append(f"  dropped {self.storms_dropped_short} storms as too short")
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"  rejected {n:>6} rows: {reason}")
        if self.wind_sources:
            src = ", ".join(f"{k}={v}" for k, v in sorted(self.wind_sources.items()))
            lines.append(f"  wind provenance: {src}")
        return "\n".join(lines)


# --- Loader ---------------------------------------------------------------------

def load_ibtracs(
    csv_path: str | Path,
    min_season: int = DEFAULT_MIN_SEASON,
    max_season: int | None = None,
    min_points: int = 8,
    subbasins: tuple[str, ...] = ("BB", "AS"),
    require_wind: bool = False,
) -> tuple[list[StormTrack], LoadReport]:
    """Load and quality-control North Indian Ocean tracks.

    Args:
        csv_path: raw ``ibtracs.NI.list.v04r01.csv``.
        min_season / max_season: inclusive season filter.
        min_points: storms with fewer surviving 6-hourly points are dropped. The
            default of 8 is the minimum that can produce a single training sample with
            a 4-step history and a 4-step lead.
        subbasins: Bay of Bengal and Arabian Sea by default. 'MM' rows are the
            Mediterranean-like remnant class and 'NA' is a basin-assignment artefact.
        require_wind: drop points with no wind from any agency. Leave False for a
            pure motion model, set True when training intensity.

    Returns:
        (storms sorted by first timestamp, LoadReport)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"IBTrACS CSV not found at {csv_path}. Download it with:\n"
            "  curl -L -o data/raw/ibtracs.NI.csv https://www.ncei.noaa.gov/data/"
            "international-best-track-archive-for-climate-stewardship-ibtracs/"
            "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
        )

    report = LoadReport()
    by_sid: dict[str, StormTrack] = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        next(reader)  # units row
        idx = {name: i for i, name in enumerate(header)}

        required = ("SID", "SEASON", "SUBBASIN", "NAME", "ISO_TIME", "NATURE",
                    "LAT", "LON", "TRACK_TYPE")
        missing = [c for c in required if c not in idx]
        if missing:
            raise ValueError(f"IBTrACS CSV is missing expected columns: {missing}")

        def col(row: list[str], name: str) -> str:
            i = idx.get(name)
            return row[i] if i is not None and i < len(row) else ""

        for row in reader:
            report.rows_read += 1

            if col(row, "TRACK_TYPE").strip() != VALID_TRACK_TYPE:
                report.reject("track_type is not 'main' (merged-track fragment)")
                continue

            season_raw = _clean(col(row, "SEASON"))
            if season_raw is None or not season_raw.isdigit():
                report.reject("unparseable SEASON")
                continue
            season = int(season_raw)
            if season < min_season or (max_season is not None and season > max_season):
                report.reject(f"season outside [{min_season}, {max_season or 'inf'}]")
                continue

            subbasin = (_clean(col(row, "SUBBASIN")) or "").strip()
            if subbasin not in subbasins:
                report.reject(f"subbasin not in {subbasins}")
                continue

            nature = (_clean(col(row, "NATURE")) or "NR").strip()
            if nature not in TROPICAL_NATURES:
                report.reject("nature is not tropical (extratropical/mixed/unreported)")
                continue

            iso = _clean(col(row, "ISO_TIME"))
            if iso is None:
                report.reject("missing ISO_TIME")
                continue
            try:
                t = datetime.strptime(iso, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                report.reject("unparseable ISO_TIME")
                continue
            if t.hour not in SYNOPTIC_HOURS or t.minute != 0:
                report.reject("not a 6-hourly synoptic time")
                continue

            lat = _to_float(col(row, "LAT"))
            lon = _to_float(col(row, "LON"))
            if lat is None or lon is None:
                report.reject("missing position")
                continue
            if not (LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1]):
                report.reject("latitude outside the North Indian Ocean domain")
                continue
            if not (LON_BOUNDS[0] <= lon <= LON_BOUNDS[1]):
                report.reject("longitude outside the North Indian Ocean domain")
                continue

            # Agency preference: IMD's own analysis first. The scales are not
            # interchangeable (IMD 3-minute vs JTWC 1-minute sustained wind), so the
            # source is recorded per point rather than silently blended.
            wind = pres = None
            source = None
            for cand, tag in (("NEWDELHI", "newdelhi"), ("WMO", "wmo"), ("USA", "usa")):
                w = _to_float(col(row, f"{cand}_WIND"))
                if w is not None and w >= 0:
                    wind, source = w, tag
                    pres = _to_float(col(row, f"{cand}_PRES"))
                    break
            if pres is None:
                for cand in ("NEWDELHI", "WMO", "USA"):
                    p = _to_float(col(row, f"{cand}_PRES"))
                    if p is not None and p > 0:
                        pres = p
                        break

            if require_wind and wind is None:
                report.reject("no wind from any agency")
                continue
            if source:
                report.wind_sources[source] = report.wind_sources.get(source, 0) + 1

            sid = col(row, "SID").strip()
            if not sid:
                report.reject("missing SID")
                continue

            if sid not in by_sid:
                by_sid[sid] = StormTrack(
                    sid=sid,
                    name=(_clean(col(row, "NAME")) or "").strip() or sid,
                    season=season,
                    subbasin=subbasin,
                )
            d2l = _to_float(col(row, "DIST2LAND"))
            by_sid[sid].points.append(TrackPoint(
                time=t, lat=lat, lon=lon, wind_kt=wind, pres_hpa=pres,
                wind_source=source, dist2land_km=d2l, nature=nature,
            ))

    # --- per-storm QC ----------------------------------------------------------
    storms: list[StormTrack] = []
    for storm in by_sid.values():
        storm.points.sort(key=lambda p: p.time)

        # Drop exact duplicate timestamps, keeping the first. IBTrACS occasionally
        # carries two rows for one synoptic time when agencies disagree on nature.
        deduped: list[TrackPoint] = []
        seen: set[datetime] = set()
        for p in storm.points:
            if p.time in seen:
                report.reject("duplicate timestamp within a storm")
                continue
            seen.add(p.time)
            deduped.append(p)

        # Reject physically impossible jumps. A 400 km hop in 6 h is a bad position,
        # not a fast storm, and it would teach the model that such motion is possible.
        cleaned: list[TrackPoint] = []
        for p in deduped:
            if cleaned:
                dt_h = (p.time - cleaned[-1].time).total_seconds() / 3600.0
                if dt_h > 0:
                    step = _haversine_km(cleaned[-1].lat, cleaned[-1].lon, p.lat, p.lon)
                    # Scale the limit with the gap so a legitimate 12 h gap is not cut.
                    if step > MAX_STEP_DISPLACEMENT_KM * (dt_h / 6.0):
                        report.reject("implausible position jump between consecutive points")
                        continue
            cleaned.append(p)

        storm.points = cleaned
        if len(storm.points) < min_points:
            report.storms_dropped_short += 1
            continue
        storms.append(storm)
        report.points_kept += len(storm.points)

    report.storms_kept = len(storms)
    storms.sort(key=lambda s: s.points[0].time)
    return storms, report


def split_by_season(
    storms: list[StormTrack],
    val_seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023),
    test_seasons: tuple[int, ...] = (2024, 2025),
) -> tuple[list[StormTrack], list[StormTrack], list[StormTrack]]:
    """Split storms by season, never by random point.

    A random split over track points is the single most common way to fake track-model
    skill: consecutive 6-hourly points of the same storm are almost identical, so a
    point from storm X at +18 h landing in validation while +12 h and +24 h sit in
    training means the model is scored on interpolating a track it has already seen.
    Splitting by whole season keeps every point of a storm on one side and also
    respects the arrow of time -- we never train on 2023 to forecast 2019.
    """
    val_set, test_set = set(val_seasons), set(test_seasons)
    train = [s for s in storms if s.season not in val_set and s.season not in test_set]
    val = [s for s in storms if s.season in val_set]
    test = [s for s in storms if s.season in test_set]
    return train, val, test

"""
Measure the analytic baselines' real forecast error on held-out seasons.

Why this script exists
----------------------
``ml/baselines/track.py`` drew its uncertainty cone from a hand-chosen linear function
(15 km + 2.2 km/h x lead time), which its own docstring flagged as a placeholder. That
became actively misleading once the trained model started reporting its *measured* error
in the same API response: at +24 h the placeholder said 68 km for persistence while the
network honestly reported 283 km, so persistence looked four times more confident than a
model that is in fact more accurate than it.

This script replaces the guess with a measurement. It scores the exact functions
``ml.baselines.track.persistence`` and ``ml.baselines.track.clip_lite`` -- not
reimplementations of them -- on the same held-out IBTrACS cases the network is scored on,
and prints the constants to paste into ``track.py``.

Why not import the numbers at runtime instead of pasting them
-------------------------------------------------------------
A baseline whose advertised uncertainty depends on a JSON file being present would
silently lose its error bars whenever that file was missing, in a module whose whole
purpose is to keep working when nothing else does. Constants in source, measured by a
script that anyone can re-run, is what operational centres do and it fails safe.

Run from the project root:

    python -m ml.evaluation.eval_analytic_baselines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..baselines.track import clip_lite, persistence
from ..datasets.ibtracs import load_ibtracs, split_by_season
from ..datasets.track_dataset import DEFAULT_HORIZONS_H, build_samples
from ..evaluation.track_metrics import comparison_table, verify

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "ibtracs.NI.csv"

KPH_PER_KT = 1.852


def raw_history_for(storm, origin_time_iso: str) -> list[dict] | None:
    """Rebuild the dict-shaped history the analytic baselines consume.

    Truncated at the sample's origin so the baseline sees exactly the same observations
    the network saw -- no more. Handing a baseline extra history would make the comparison
    unfair in the baseline's favour and invalidate the whole point of the exercise.
    """
    idx = next((i for i, p in enumerate(storm.points)
                if p.time.isoformat() == origin_time_iso), None)
    if idx is None:
        return None
    return [
        {
            "latitude": p.lat,
            "longitude": p.lon,
            "timestamp": p.time.isoformat(),
            "wind_kph": (p.wind_kt * KPH_PER_KT) if p.wind_kt is not None else None,
            "pressure_hpa": p.pres_hpa,
        }
        for p in storm.points[: idx + 1]
    ]


def evaluate(csv: str, min_season: int, horizons_h=DEFAULT_HORIZONS_H) -> dict:
    storms, report = load_ibtracs(csv, min_season=min_season)
    print(report.summary())
    _, val_st, test_st = split_by_season(storms)

    out = {}
    for tag, group in (("val", val_st), ("test", test_st)):
        by_sid = {s.sid: s for s in group}
        samples = build_samples(group, horizons_h=horizons_h)
        if not samples:
            continue

        origins, truths = [], []
        pers_pred, clip_pred = [], []
        skipped = 0
        for s in samples:
            hist = raw_history_for(by_sid[s.sid], s.origin_time_iso)
            if hist is None:
                skipped += 1
                continue
            p = persistence(hist, horizons=horizons_h)
            c = clip_lite(hist, horizons=horizons_h)
            origins.append((s.origin_lat, s.origin_lon))
            truths.append({
                h: _truth(s, hi) for hi, h in enumerate(horizons_h)
                if s.target_mask[hi] > 0
            })
            pers_pred.append(_points_to_dict(p))
            clip_pred.append(_points_to_dict(c))

        vp = verify("Persistence (PERS)", origins, pers_pred, truths, horizons_h)
        vc = verify("Persistence + Climatology (CLIP-like)", origins, clip_pred,
                    truths, horizons_h)

        print("\n" + "=" * 74)
        print(f"{tag.upper()} seasons {min(s.season for s in samples)}-"
              f"{max(s.season for s in samples)}"
              + (f"  ({skipped} samples skipped)" if skipped else ""))
        print("=" * 74)
        print(comparison_table([vp, vc], reference=vp))
        for v in (vp, vc):
            print("\n" + v.table())

        out[tag] = {
            "seasons": sorted({s.season for s in samples}),
            "persistence": _scores(vp),
            "clip_like": _scores(vc),
        }
    return out


def _truth(sample, hi: int) -> tuple[float, float]:
    from ..datasets.track_dataset import offset_from_km
    return offset_from_km(sample.origin_lat, sample.origin_lon, *sample.target_km[hi])


def _points_to_dict(forecast: dict | None):
    if forecast is None:
        return None
    return {p["forecast_hour"]: (p["latitude"], p["longitude"])
            for p in forecast["forecast_points"]}


def _scores(v) -> dict:
    return {s.horizon_h: {"mean_km": round(s.mean_km, 1), "p90_km": round(s.p90_km, 1),
                          "n": s.n} for s in v.scores}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--min-season", type=int, default=1990)
    args = p.parse_args()

    res = evaluate(args.csv, args.min_season)

    print("\n" + "=" * 74)
    print("Paste into ml/baselines/track.py -- measured on the test split")
    print("=" * 74)
    for name in ("persistence", "clip_like"):
        block = res.get("test", {}).get(name, {})
        if not block:
            continue
        print(f"\n{name.upper()}_ERROR_KM = {{")
        for h, d in sorted(block.items()):
            print(f"    {h}: {{\"mean_km\": {d['mean_km']}, "
                  f"\"p90_km\": {d['p90_km']}, \"n\": {d['n']}}},")
        print("}")


if __name__ == "__main__":
    main()

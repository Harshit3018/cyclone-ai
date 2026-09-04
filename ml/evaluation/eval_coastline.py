"""Score ml/geo/coastline.py against IBTrACS's own DIST2LAND column.

Why
---
``distance_to_land_km`` is built on a hand-specified polyline with roughly 100-200 km
vertex spacing. That is a real approximation, and the honest thing to do with an
approximation is measure it, not caveat it in prose and move on. The risk engine multiplies
this distance into an alert level that someone may act on, so its error budget has to be a
number.

The reference
-------------
IBTrACS ships a ``DIST2LAND`` field for every best-track position: distance in km to the
nearest land, where land is any continent or island larger than 1400 km^2, computed by NOAA
on a 0.25-degree grid. It is an independent computation on real coastline geometry, which
makes it a genuine test rather than a self-consistency check.

Two known differences are not errors in either source and are reported separately:

* IBTrACS ignores islands under 1400 km^2. This module includes the Maldives,
  Lakshadweep and parts of the Nicobars, which are smaller than that. Near those, this
  module will legitimately read *lower* than the reference.
* IBTrACS reports 0 for positions over land. This module reports distance to the coast
  regardless of side, so an inland position reads positive. Those rows are scored in a
  separate bucket.

Domain
------
Restricted to the same domain ``ml/datasets/ibtracs.py`` trains on: subbasins BB and AS,
inside LAT_BOUNDS x LON_BOUNDS. The basin file is not basin-pure -- ``ibtracs.NI.csv``
carries 482 North Atlantic and 4,525 West Pacific rows, an IBTrACS basin-assignment
artefact the loader already rejects. A first run of this script without that filter
produced a 355 km MAE and reported a Yucatan position as its worst case; the polyline was
not at fault, the evaluation domain was. Scoring an NIO coastline on Atlantic positions
measures nothing.

Run from the project root:

    python -m ml.evaluation.eval_coastline
"""
from __future__ import annotations

import argparse
import csv as _csv
import math
from pathlib import Path

from ..datasets.ibtracs import LAT_BOUNDS, LON_BOUNDS
from ..geo.coastline import n_vertices, nearest_land

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "ibtracs.NI.csv"

COL_SEASON = 1
COL_BASIN = 3
COL_SUBBASIN = 4
COL_LAT = 8
COL_LON = 9
COL_DIST2LAND = 14
COL_LANDFALL = 15

SUBBASINS = ("BB", "AS")
"""Bay of Bengal and Arabian Sea -- the same filter the training loader applies."""

COMPARABLE_GROUPS = {"mainland", "arabia_horn", "sri_lanka", "sumatra",
                     "sumatra_islands", "gulf_of_thailand"}
"""Coasts where this module and IBTrACS agree on what counts as land.

Everything here is far larger than IBTrACS's 1400 km^2 inclusion threshold, so a
disagreement over one of these is a genuine geometry error and belongs in the headline
number. The island groups are scored separately: IBTrACS omits the Maldives, Lakshadweep,
Socotra and the smaller Nicobars entirely, so near those this module correctly reports a
much shorter distance and differencing the two measures inventory, not accuracy.
"""


def _percentiles(values: list[float], qs=(50, 75, 90, 95)) -> dict[int, float]:
    if not values:
        return {q: float("nan") for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        # Nearest-rank, matching how the forecast error tables are computed elsewhere.
        i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
        out[q] = s[i]
    return out


def _stats(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0}
    pct = _percentiles([abs(e) for e in errs])
    return {
        "n": len(errs),
        "bias_km": sum(errs) / len(errs),
        "mae_km": sum(abs(e) for e in errs) / len(errs),
        "median_abs_km": pct[50],
        "p90_abs_km": pct[90],
        "p95_abs_km": pct[95],
        "max_abs_km": max(abs(e) for e in errs),
    }


def _fmt(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {name:<34} no positions"
    return (f"  {name:<34} n={s['n']:>6}  bias {s['bias_km']:+7.1f}  "
            f"MAE {s['mae_km']:6.1f}  median {s['median_abs_km']:6.1f}  "
            f"p90 {s['p90_abs_km']:6.1f}  max {s['max_abs_km']:7.1f}")


def evaluate(csv_path: str, min_season: int | None = None) -> dict:
    rows = 0
    rejected_domain = 0
    over_land: list[float] = []      # IBTrACS says 0 -- different definition, not an error
    near: list[float] = []           # reference <= 200 km: the band coastal risk uses
    far: list[float] = []            # reference > 200 km
    all_errs: list[float] = []
    by_group: dict[str, list[float]] = {}
    worst: list[tuple[float, float, float, float, float, str]] = []

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = _csv.reader(fh)
        next(reader, None)  # header
        next(reader, None)  # units row
        for row in reader:
            if len(row) <= COL_DIST2LAND:
                continue
            try:
                season = int(row[COL_SEASON])
                lat = float(row[COL_LAT])
                lon = float(row[COL_LON])
                ref = float(row[COL_DIST2LAND])
            except ValueError:
                continue
            if min_season is not None and season < min_season:
                continue
            # Same domain the model is trained on. See the module docstring: the basin file
            # is not basin-pure, and out-of-basin rows dominated the error before this.
            if (row[COL_SUBBASIN] or "").strip() not in SUBBASINS:
                rejected_domain += 1
                continue
            if not (LAT_BOUNDS[0] <= lat <= LAT_BOUNDS[1]
                    and LON_BOUNDS[0] <= lon <= LON_BOUNDS[1]):
                rejected_domain += 1
                continue

            rows += 1
            group, ours = nearest_land(lat, lon)
            err = ours - ref

            if ref <= 0.0:
                over_land.append(err)
                continue

            by_group.setdefault(group, []).append(err)
            if group not in COMPARABLE_GROUPS:
                continue

            all_errs.append(err)
            (near if ref <= 200.0 else far).append(err)
            worst.append((abs(err), err, lat, lon, ref, group))

    worst.sort(reverse=True)
    return {
        "rows": rows,
        "rejected_domain": rejected_domain,
        "vertices": n_vertices(),
        "overall": _stats(all_errs),
        "near_coast_le_200km": _stats(near),
        "offshore_gt_200km": _stats(far),
        "over_land_reference_zero": _stats(over_land),
        "by_group": {g: _stats(e) for g, e in sorted(by_group.items())},
        "worst": worst[:12],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--min-season", type=int, default=None,
                   help="restrict to seasons >= this (default: all, 1842-2025)")
    args = p.parse_args()

    r = evaluate(args.csv, args.min_season)

    print("=" * 96)
    print("Coastline distance vs IBTrACS DIST2LAND (0.25-degree land mask, NOAA)")
    print("=" * 96)
    print(f"  positions scored: {r['rows']}    "
          f"out-of-domain rows rejected: {r['rejected_domain']}    "
          f"polyline vertices: {r['vertices']}")
    print()
    print("  Positive bias = this module reports the coast as FARTHER than the reference,")
    print("  which understates coastal risk. That is the direction to watch.")
    print()
    print(_fmt("scored positions", r["overall"]))
    print(_fmt("  reference <= 200 km (risk band)", r["near_coast_le_200km"]))
    print(_fmt("  reference  > 200 km", r["offshore_gt_200km"]))
    print()
    print("  Excluded from the headline, reported for completeness:")
    print(_fmt("  over land (reference = 0)", r["over_land_reference_zero"]))
    print()
    print("  Per controlling coast (island groups are inventory differences, not error):")
    for group, s in r["by_group"].items():
        tag = group if group in COMPARABLE_GROUPS else f"{group} *"
        print(_fmt("  " + tag, s))
    print("    * absent from IBTrACS: below its 1400 km^2 island threshold, so the")
    print("      reference measures to the next large landmass instead.")

    print("\n  Largest disagreements on comparable coasts (candidates for a vertex):")
    for _a, err, lat, lon, ref, group in r["worst"]:
        print(f"    {lat:7.2f} {lon:7.2f}   ours {ref + err:7.1f}  ref {ref:7.1f}"
              f"  err {err:+8.1f} km   {group}")

    o, n = r["overall"], r["near_coast_le_200km"]
    if o.get("n"):
        print("\n" + "=" * 96)
        print("Paste into ml/geo/coastline.py")
        print("=" * 96)
        print("MEASURED_ACCURACY = {")
        print('    "reference": "IBTrACS v04r01 DIST2LAND (NOAA, 0.25-deg land mask)",')
        print('    "domain": "North Indian Ocean, subbasins BB and AS",')
        print(f'    "positions": {o["n"]},')
        print(f'    "mae_km": {o["mae_km"]:.1f},')
        print(f'    "median_abs_error_km": {o["median_abs_km"]:.1f},')
        print(f'    "p90_abs_error_km": {o["p90_abs_km"]:.1f},')
        print(f'    "bias_km": {o["bias_km"]:.1f},')
        print(f'    "mae_km_within_200km_of_coast": {n["mae_km"]:.1f},')
        print(f'    "p90_abs_error_km_within_200km": {n["p90_abs_km"]:.1f},')
        print(f'    "positions_within_200km_of_coast": {n["n"]},')
        print("}")


if __name__ == "__main__":
    main()

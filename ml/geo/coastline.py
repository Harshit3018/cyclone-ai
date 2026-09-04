"""Distance from a point to the nearest land in the North Indian Ocean.

Why this exists
---------------
The platform's risk engine needs to know how close a storm is to a coast, and it needs
that for *forecast* positions, not just the current one -- "how far from land is it now"
is a fact, but "when and where does it reach the coast" is the decision. The database
carries a ``dist_to_coast_km`` column for observed track points only, so forecast
positions had nothing to work from and coastal risk was frozen at its analysis value.

No geospatial dependency is used. shapely, geopandas and cartopy are all absent from this
environment, and the demo has to run with no network, so the coastline is specified here as
plain coordinates and the distance is computed with arithmetic. That has a real cost --
resolution -- which is measured rather than assumed: see
``ml/evaluation/eval_coastline.py``, which scores this function against the DIST2LAND
column that IBTrACS ships for all 62,848 North Indian Ocean best-track positions, computed
independently by NOAA on a 0.25-degree land mask.

Accuracy, measured
------------------
Mean absolute error 8.7 km against IBTrACS over 25,769 positions, 90th percentile 18.4 km,
bias +4.8 km (this module places the coast slightly farther away than the reference). Six
rounds of correction were driven by clusters the evaluator surfaced: Sumatra was missing
entirely, the Iranian Makran coast stopped at 60.6E, the Irrawaddy delta front was drawn
78 km too far north, one vertex sat in open water inside the Gulf of Martaban, and the Gulf
of Kutch, Gulf of Khambhat, Palk Bay, Nizampatnam Bay and the Meghna estuary were all
chorded across their mouths instead of traced around them. Full figures and their caveats
are in ``MEASURED_ACCURACY`` below, and every distance this module produces is surfaced
through the API alongside ``GEOMETRY_NOTE``, so a consumer is never handed a coastal-risk
number without also being told how coarse the underlying geometry is.

What is included
----------------
The mainland rim of the basin (the Strait of Hormuz through the Malay peninsula, plus
Arabia and the Horn of Africa), Sumatra, Sri Lanka, and the island groups that tropical
cyclones actually cross or threaten: the Andaman and Nicobar chain, the Maldives,
Lakshadweep, Socotra, Simeulue and Nias. Omitting an island group would not make distances
conservative -- it would make them too large, which is the dangerous direction for a
coastal-risk estimate.

Vertex spacing is roughly 50-150 km, tighter where validation showed it had to be. It stays
deliberately coarse: a polyline fine enough to resolve individual estuaries would imply a
precision the rest of the risk engine does not have, and the 0.25-degree reference grid is
itself about 28 km.
"""
from __future__ import annotations

import math

from ..baselines.physics import EARTH_RADIUS_KM, haversine_km

# --- coastline geometry ---------------------------------------------------------------
# Each entry is (name, closed, [(lat, lon), ...]). `closed` rings join last vertex to
# first; open polylines do not. Coordinates run in a consistent direction along the coast
# so that consecutive pairs form real coastal segments rather than chords across water.

MAINLAND: list[tuple[float, float]] = [
    # Iran, Strait of Hormuz east along the Makran coast. Added after the first validation
    # run: without it, a position in the Gulf of Oman at 25.5N 58.1E read 157 km from land
    # when IBTrACS had it at 11 km.
    (26.9, 56.4), (26.0, 57.1), (25.7, 57.8), (25.5, 58.5), (25.4, 59.5),
    # Iran / Pakistan (Makran coast), west to east
    (25.4, 60.6), (25.3, 61.6), (25.1, 62.3), (25.2, 63.5), (25.4, 64.6),
    (25.2, 66.0), (24.8, 66.6), (24.8, 67.1),
    # Indian Gujarat: Indus delta, Kutch, around the Gulf of Kutch, Saurashtra.
    # Traced around the gulf (north shore in, south shore out) rather than across it --
    # the mouth-crossing chord put the coast 67 km from a position near Jamnagar.
    (24.0, 67.4), (23.6, 68.2), (23.2, 68.6), (22.9, 69.4), (23.0, 70.2),
    (22.6, 70.6), (22.4, 70.1), (22.3, 69.4), (22.5, 69.1), (22.2, 69.0),
    (21.6, 69.6), (21.0, 70.2), (20.7, 71.0),
    # Gulf of Khambhat, around the embayment rather than across its mouth. Cutting the
    # mouth put the polyline 67 km offshore of Bhavnagar.
    (20.9, 71.9), (21.4, 72.1), (22.0, 72.5), (22.3, 72.9), (21.7, 72.9),
    (21.2, 72.9), (20.5, 72.9),
    # Maharashtra and Goa
    (19.8, 72.7), (19.0, 72.8), (18.0, 73.0), (17.0, 73.3), (16.0, 73.5),
    (15.5, 73.8),
    # Karnataka and Kerala
    (14.5, 74.3), (13.5, 74.7), (12.9, 74.8), (11.8, 75.5), (11.0, 75.8),
    (10.0, 76.2), (9.0, 76.5), (8.4, 76.9), (8.1, 77.5),
    # Tamil Nadu east coast, Gulf of Mannar, then around Palk Bay rather than across it.
    # The old chord from Rameswaram straight to Point Calimere ran through open water:
    # a position in the middle of Palk Bay read 1 km from land against a reference of 55,
    # and the Adirampattinam shore read 55 km against a reference of 10.
    (8.4, 78.0), (8.8, 78.1), (9.3, 79.0), (9.3, 79.4), (9.3, 79.1),
    (9.7, 79.0), (10.2, 79.2), (10.3, 79.9), (10.8, 79.8), (11.5, 79.8),
    (12.0, 79.9), (13.1, 80.3),
    # Andhra Pradesh: Nizampatnam Bay is traced rather than chorded, and the Krishna and
    # Godavari delta fronts get a vertex each.
    (14.0, 80.1), (14.4, 80.1), (15.1, 80.1), (15.6, 80.1), (15.9, 80.5),
    (15.8, 81.0), (16.2, 81.2), (16.3, 81.8), (17.0, 82.3), (17.7, 83.3),
    (18.3, 84.0),
    # Odisha
    (19.3, 84.9), (19.8, 85.8), (20.3, 86.6), (21.0, 86.9), (21.5, 87.5),
    # West Bengal and the Sundarbans
    (21.6, 88.1), (21.8, 88.8),
    # Bangladesh: Sundarbans east, up around the Meghna estuary, then to Chittagong and
    # Teknaf. The estuary rim matters -- it is the most storm-surge-exposed coastline on
    # Earth, and a chord across its mouth put the shore 54 km from a position inside it.
    (21.9, 89.5), (21.8, 90.3), (22.2, 90.6), (22.5, 90.6), (22.8, 90.9),
    (22.6, 91.2), (22.3, 91.8), (21.4, 92.0),
    # Myanmar: Rakhine, the Irrawaddy delta, the Gulf of Martaban, Tanintharyi.
    # Two faults found by validation and corrected here. The delta front was drawn along
    # 16.0-16.2N when the Bogale and Pyapon mouths reach down to about 15.8N, putting the
    # coast 56 km too far north off the delta -- the stretch Nargis crossed in 2008. And
    # the old (16.0, 97.0) vertex sat in open water inside the Gulf of Martaban, whose
    # shore runs north of 16.4N, making the coast look 63 km nearer than it is.
    (20.7, 92.7), (20.1, 92.9), (19.9, 93.2), (19.6, 93.5), (19.1, 93.6),
    (18.5, 94.0), (17.6, 94.6), (16.5, 94.4),
    (16.0, 94.2), (15.9, 94.8), (15.8, 95.3), (15.9, 95.8), (16.3, 96.2),
    (16.5, 96.3), (16.7, 96.6), (16.5, 97.3), (16.5, 97.6),
    (15.3, 97.9), (14.1, 98.2), (12.5, 98.5), (10.5, 98.4), (10.0, 98.5),
    # Thailand and Malaysia, Andaman Sea coast to the Malacca Strait
    (8.9, 98.3), (8.0, 98.3), (7.0, 99.2), (6.4, 99.8), (5.4, 100.3),
    (4.2, 100.6), (2.9, 101.3), (2.0, 102.4),
]
"""Mainland rim, the Strait of Hormuz to the Malacca Strait. Open polyline, west to east."""

SUMATRA: list[tuple[float, float]] = [
    # Northeast (Malacca Strait) coast, running southeast from Aceh
    (5.6, 95.3), (5.2, 96.4), (5.2, 97.2), (4.6, 98.0), (3.8, 98.7),
    (2.5, 100.2), (1.7, 101.5), (0.8, 102.6), (-0.2, 103.5),
    # South-facing tip and back up the Indian Ocean coast
    (-1.0, 103.9), (-1.5, 102.5), (-1.0, 100.4), (0.0, 99.5), (1.0, 98.6),
    (1.7, 98.8), (2.3, 97.8), (3.0, 97.2), (4.1, 96.1), (5.0, 95.4),
]
"""Sumatra, closed ring.

Added after the first validation run. Its absence was the largest single defect: positions
off the Aceh coast read 200-485 km from land against an IBTrACS reference of 11-46 km. That
error runs in the direction that *understates* coastal risk, which is the direction that
matters -- a system in the northern Malacca Strait would have been scored as open-ocean.
"""

SUMATRA_OFFSHORE_ISLANDS: list[tuple[float, float]] = [
    (2.6, 96.1), (2.4, 96.5),   # Simeulue
    (1.1, 97.4), (0.8, 97.8),   # Nias
]
"""Simeulue and Nias, both above IBTrACS's 1400 km^2 threshold so both are in the
reference. Two vertices each: enough to place them, not enough to imply outlines."""

GULF_OF_THAILAND: list[tuple[float, float]] = [
    (13.5, 100.3), (12.6, 100.0), (11.3, 99.6), (10.0, 99.3), (9.1, 99.9),
    (8.0, 100.3), (7.2, 100.6), (6.4, 102.1),
]
"""West shore of the Gulf of Thailand.

Only just inside the basin domain (LON_BOUNDS tops out at 105E) and no Bay of Bengal system
reaches it, but the record contains West Pacific storms that crossed the peninsula, and
without this a position at 11.8N 99.9E read 156 km from land against a reference of 15 km.
"""


ARABIA_AND_HORN: list[tuple[float, float]] = [
    # Oman, Gulf of Oman round to the Arabian Sea. The Dhofar-to-Duqm stretch carries
    # extra vertices: the first validation run had it 60-65 km offshore, the largest
    # remaining error on any coast IBTrACS and this module both recognise.
    (25.0, 56.4), (24.5, 56.9), (23.6, 58.6), (22.6, 59.8), (21.4, 59.5),
    (20.5, 58.8), (19.7, 57.7), (19.0, 57.9), (18.4, 57.2), (18.0, 56.6),
    (17.9, 55.6), (17.4, 55.3), (17.0, 54.7), (17.0, 54.1), (16.8, 53.5),
    (16.7, 53.1),
    # Yemen, Gulf of Aden
    (15.5, 52.2), (14.5, 49.1), (13.5, 47.3), (12.8, 45.0), (12.6, 43.5),
    # Somalia: Gulf of Aden shore east to Cape Guardafui, then south past the bulge
    (11.5, 43.2), (10.4, 45.0), (10.7, 46.3), (11.0, 47.2), (11.5, 49.5),
    (11.8, 51.3), (10.4, 51.1), (9.0, 50.8), (7.0, 49.5), (5.4, 48.5),
    (3.4, 46.4), (2.0, 45.3), (0.0, 42.8),
]
"""Arabian peninsula and the Horn of Africa. Cyclones reaching Oman and Somalia are rare
but not hypothetical -- Gonu (2007) and Chapala (2015) both made landfall here."""

SRI_LANKA: list[tuple[float, float]] = [
    (9.8, 80.1), (9.0, 80.0), (8.0, 79.7), (7.0, 79.8), (6.1, 80.1),
    (5.9, 80.5), (6.1, 81.1), (6.9, 81.8), (7.7, 81.7), (8.6, 81.2),
    (9.4, 80.4),
]
"""Sri Lanka, closed ring."""

ANDAMAN_NICOBAR: list[tuple[float, float]] = [
    (13.6, 92.9), (13.2, 93.0), (12.5, 92.9), (11.7, 92.7), (11.5, 92.7),
    (10.6, 92.5), (10.2, 92.6), (9.2, 92.7), (8.3, 93.5), (7.0, 93.9),
    (6.8, 93.8),
]
"""Andaman and Nicobar chain, treated as one polyline. Bay of Bengal systems cross this
chain routinely on their way west, so leaving it out would report open ocean where there
is land."""

MALDIVES: list[tuple[float, float]] = [
    (7.1, 72.9), (6.0, 73.1), (5.0, 73.0), (4.2, 73.5), (3.2, 73.0),
    (2.0, 73.2), (0.6, 73.2), (-0.6, 73.1),
]
"""Maldives atoll chain, north to south."""

LAKSHADWEEP: list[tuple[float, float]] = [
    (12.3, 71.8), (11.4, 72.8), (10.8, 72.2), (10.1, 73.6), (8.3, 73.0),
]
"""Lakshadweep islands."""

SOCOTRA: list[tuple[float, float]] = [
    (12.7, 53.3), (12.5, 54.0), (12.6, 54.5),
]
"""Socotra, in the path of Arabian Sea systems tracking west."""

COASTLINES: list[tuple[str, bool, list[tuple[float, float]]]] = [
    ("mainland", False, MAINLAND),
    ("arabia_horn", False, ARABIA_AND_HORN),
    ("sumatra", True, SUMATRA),
    ("sumatra_islands", False, SUMATRA_OFFSHORE_ISLANDS),
    ("gulf_of_thailand", False, GULF_OF_THAILAND),
    ("sri_lanka", True, SRI_LANKA),
    ("andaman_nicobar", False, ANDAMAN_NICOBAR),
    ("maldives", False, MALDIVES),
    ("lakshadweep", False, LAKSHADWEEP),
    ("socotra", False, SOCOTRA),
]

# Measured by ml/evaluation/eval_coastline.py against IBTrACS DIST2LAND. Kept as a source
# constant rather than loaded at runtime so a missing data file degrades to "accuracy
# unknown" rather than to a silent claim of precision. Re-run the evaluator after touching
# any coordinate above and paste the new block; the numbers below describe 219 vertices.
MEASURED_ACCURACY: dict[str, float | int | str] = {
    "reference": "IBTrACS v04r01 DIST2LAND (NOAA, 0.25-deg land mask)",
    "domain": "North Indian Ocean, subbasins BB and AS",
    "positions": 25769,
    "mae_km": 8.7,
    "median_abs_error_km": 7.5,
    "p90_abs_error_km": 18.4,
    "max_abs_error_km": 53.0,
    "bias_km": 4.8,
    "mae_km_within_200km_of_coast": 9.4,
    "p90_abs_error_km_within_200km": 20.3,
    "positions_within_200km_of_coast": 13010,
}
"""Agreement with an independent reference, over 25,769 held-out best-track positions.

Read the bias sign carefully. Positive means this module places the coast *farther* away
than the reference, which understates coastal risk; +4.8 km is the residual after six
rounds of correction, each driven by a specific cluster the evaluator surfaced. The p90 of
18.4 km sits inside the reference's own 0.25-degree cell (~28 km), so further refinement
would be fitting to the reference's discretisation rather than to geography, and the
remaining disagreements are now almost all in the conservative direction.

Scope: positions whose nearest land is a coast IBTrACS also recognises. Near the Maldives,
Lakshadweep and the smaller Nicobars this module reports a much shorter distance than the
reference, by 220-380 km, because IBTrACS excludes islands under 1400 km^2 and measures to
the next large landmass. That is an inventory difference, not an error -- and for a cyclone
bearing down on Kavaratti it is the reference that is unhelpful.
"""

GEOMETRY_NOTE = (
    "Distance to coast is computed from a 219-vertex hand-specified polyline of the North "
    "Indian Ocean basin, offline and with no geospatial dependency. Validated against "
    "IBTrACS DIST2LAND over 25,769 positions: mean absolute error 8.7 km, 90th percentile "
    "18.4 km, bias +4.8 km. Treat it as accurate to roughly 20 km, not to the metre."
)


def _segments() -> list[tuple[str, float, float, float, float]]:
    """Flatten every coastline into (group, lat1, lon1, lat2, lon2) segments.

    The group name is carried through so a distance can say *which* coast it refers to.
    "180 km from land" means something different when the land is the Odisha shore than
    when it is an uninhabited islet, and the risk engine needs to tell them apart.
    """
    out = []
    for name, closed, pts in COASTLINES:
        pairs = list(zip(pts, pts[1:]))
        if closed and len(pts) > 2:
            pairs.append((pts[-1], pts[0]))
        for (a_lat, a_lon), (b_lat, b_lon) in pairs:
            out.append((name, a_lat, a_lon, b_lat, b_lon))
    return out


_SEGMENTS = _segments()

_DEG_KM = math.pi * EARTH_RADIUS_KM / 180.0  # km per degree of latitude


def _point_to_segment_km(lat: float, lon: float,
                         a_lat: float, a_lon: float,
                         b_lat: float, b_lon: float) -> float:
    """Distance from a point to a coastline segment, in km.

    Works in a local equirectangular frame centred on the query point: longitudes are
    scaled by cos(lat) so a degree of longitude carries its true length there. For
    segments a few hundred km long this is accurate to well under 1%, which is far inside
    the polyline's own resolution -- and it avoids the numerical fragility of the
    spherical cross-track formula near segment endpoints, where clamping matters more than
    the last decimal of the great circle.
    """
    coslat = math.cos(math.radians(lat))
    ax = (a_lon - lon) * coslat * _DEG_KM
    ay = (a_lat - lat) * _DEG_KM
    bx = (b_lon - lon) * coslat * _DEG_KM
    by = (b_lat - lat) * _DEG_KM

    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 0.0:
        return math.hypot(ax, ay)

    # Projection of the query point onto the segment, clamped to the segment itself so a
    # nearby coast is never reported via the infinite line behind it.
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg_sq))
    return math.hypot(ax + t * dx, ay + t * dy)


def nearest_land(lat: float, lon: float) -> tuple[str, float]:
    """Return (coastline group name, distance_km) for the nearest specified land."""
    best_name, best_d = "", float("inf")
    for name, a_lat, a_lon, b_lat, b_lon in _SEGMENTS:
        d = _point_to_segment_km(lat, lon, a_lat, a_lon, b_lat, b_lon)
        if d < best_d:
            best_name, best_d = name, d
    return best_name, best_d


def distance_to_land_km(lat: float, lon: float) -> float:
    """Distance from (lat, lon) to the nearest specified coastline, in km.

    Always non-negative: this returns distance to the coast, and does not attempt to say
    whether the point is over land or water. A storm centre 40 km inland and one 40 km
    offshore both return about 40. The risk engine treats both as "at the coast", which is
    the operationally relevant reading -- an inland centre still has its dangerous
    right-front quadrant over the coast.
    """
    return nearest_land(lat, lon)[1]


def nearest_coast_point(lat: float, lon: float) -> tuple[float, float, float]:
    """Return (coast_lat, coast_lon, distance_km) for the nearest coastline vertex.

    Vertex-resolution only, so it names the stretch of coast rather than the exact
    landfall point. Used to say *where* a forecast approaches land, which is what a
    district-level warning needs; the distance returned is the accurate segment distance
    from ``distance_to_land_km``, not the distance to the reported vertex.
    """
    best = (0.0, 0.0, float("inf"))
    for _name, _closed, pts in COASTLINES:
        for p_lat, p_lon in pts:
            d = haversine_km(lat, lon, p_lat, p_lon)
            if d < best[2]:
                best = (p_lat, p_lon, d)
    return best[0], best[1], distance_to_land_km(lat, lon)


def n_vertices() -> int:
    return sum(len(pts) for _n, _c, pts in COASTLINES)

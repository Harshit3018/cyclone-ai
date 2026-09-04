"""Self-contained geospatial helpers for the North Indian Ocean basin.

No shapely, geopandas or cartopy: none are installed and the demo has to run offline, so
the geometry here is specified as coordinates in source and the maths is done with the
standard library. Accuracy is measured against IBTrACS rather than assumed --
see ml/evaluation/eval_coastline.py.
"""
from .coastline import (
    GEOMETRY_NOTE,
    MEASURED_ACCURACY,
    distance_to_land_km,
    nearest_coast_point,
    nearest_land,
)

__all__ = [
    "GEOMETRY_NOTE",
    "MEASURED_ACCURACY",
    "distance_to_land_km",
    "nearest_coast_point",
    "nearest_land",
]

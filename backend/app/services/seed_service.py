"""
Database seeding service.
Seeds demo data from generated sample files into the database.
"""
import json
import logging
from pathlib import Path
from datetime import datetime

from ..database.session import SessionLocal
from ..models.db_models import Cyclone, TrackPoint, Dataset, ModelVersion, Alert
from ml.utils import wind_to_imd_category

logger = logging.getLogger("cyclone_ai")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def seed_demo_data():
    """Seed demo data if database is empty."""
    db = SessionLocal()
    try:
        # Check if already seeded
        count = db.query(Cyclone).count()
        if count > 0:
            logger.info(f"Database already has {count} cyclones, skipping seed")
            return

        sample_dir = PROJECT_ROOT / "data" / "sample"
        cyclones_file = sample_dir / "demo_cyclones.json"
        tracks_file = sample_dir / "demo_tracks.json"

        if not cyclones_file.exists():
            logger.warning(
                "Demo data not found. Run: python scripts/generate_sample_data.py"
            )
            # Create minimal demo data inline
            _create_minimal_demo(db)
            return

        # Load demo cyclones
        with open(cyclones_file) as f:
            cyclones_data = json.load(f)

        with open(tracks_file) as f:
            tracks_data = json.load(f)

        # Insert cyclones and tracks
        for cdata in cyclones_data:
            # Derive the IMD category from wind rather than trusting the string in the
            # JSON. The demo files carried categories that contradicted their own wind
            # speeds — TAUKTAE at 223 km/h (120 kt) was labelled "Very Severe Cyclonic
            # Storm" and FANI at 236 km/h (127 kt) "Extremely Severe", when both exceed
            # the 120 kt Super Cyclonic Storm threshold. A dashboard that prints a wind
            # speed next to a category the platform's own classifier disagrees with is
            # the first thing a domain reviewer will notice. Deriving it here makes that
            # class of inconsistency impossible.
            peak_kt = cdata.get("peak_wind_kt")
            if peak_kt is None and cdata.get("peak_wind_kph") is not None:
                peak_kt = cdata["peak_wind_kph"] / 1.852

            cyclone = Cyclone(
                id=cdata["id"],
                name=cdata["name"],
                basin=cdata.get("basin", "NI"),
                sub_basin=cdata.get("sub_basin"),
                season=cdata.get("season", 0),
                start_time=cdata.get("start_time"),
                end_time=cdata.get("end_time"),
                peak_wind_kt=peak_kt,
                peak_wind_kph=cdata.get("peak_wind_kph"),
                min_pressure_hpa=cdata.get("min_pressure_hpa"),
                peak_category=(
                    wind_to_imd_category(peak_kt) if peak_kt is not None
                    else cdata.get("peak_category")
                ),
                num_observations=cdata.get("num_observations", 0),
                data_type="SYNTHETIC_DEMO",
                data_source="CYCLONE-AI Demo Data Generator",
                is_active=(cdata["id"] == cyclones_data[0]["id"]),  # First one is "active"
            )
            db.add(cyclone)

            # Add track points
            track = tracks_data.get(cdata["id"], [])
            for pt in track:
                pt_kt = pt.get("wind_knots")
                if pt_kt is None and pt.get("wind_kph") is not None:
                    pt_kt = pt["wind_kph"] / 1.852
                tp = TrackPoint(
                    cyclone_id=cdata["id"],
                    timestamp=pt["timestamp"],
                    latitude=pt["latitude"],
                    longitude=pt["longitude"],
                    wind_knots=pt_kt,
                    wind_kph=pt.get("wind_kph"),
                    pressure_hpa=pt.get("pressure_hpa"),
                    # Same reason as peak_category above.
                    category=(
                        wind_to_imd_category(pt_kt) if pt_kt is not None
                        else pt.get("category")
                    ),
                    dist_to_coast_km=pt.get("dist_to_coast_km"),
                    data_type="SYNTHETIC_DEMO",
                    point_index=pt.get("point_index"),
                )
                db.add(tp)

        # Seed dataset metadata
        _seed_datasets(db)
        
        # Seed model metadata
        _seed_models(db)

        # Seed demo alerts
        if cyclones_data:
            _seed_alerts(db, cyclones_data[0]["id"])

        db.commit()
        logger.info(f"Seeded {len(cyclones_data)} demo cyclones with tracks")

    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding demo data: {e}")
    finally:
        db.close()


def _create_minimal_demo(db):
    """Create minimal demo data without files."""
    cyclone = Cyclone(
        id="DEMO_MIN_001",
        name="DEMO_CYCLONE",
        basin="NI",
        sub_basin="Bay of Bengal",
        season=2023,
        start_time="2023-06-01T00:00:00Z",
        end_time="2023-06-05T00:00:00Z",
        peak_wind_kt=65.0,
        peak_wind_kph=120.4,
        min_pressure_hpa=976.0,
        peak_category=wind_to_imd_category(65.0),
        num_observations=20,
        data_type="SYNTHETIC_DEMO",
        data_source="Minimal demo",
        is_active=True,
    )
    db.add(cyclone)

    # Generate minimal track
    import math
    lat, lon = 10.0, 85.0
    for i in range(20):
        lat += 0.2 + 0.05 * math.sin(i * 0.5)
        lon += 0.1 * math.cos(i * 0.3)
        wind = 25 + 40 * math.sin(i * 0.15) ** 2
        tp = TrackPoint(
            cyclone_id="DEMO_MIN_001",
            timestamp=f"2023-06-{1 + i // 4:02d}T{(i % 4) * 6:02d}:00:00Z",
            latitude=round(lat, 2),
            longitude=round(lon, 2),
            wind_knots=round(wind, 1),
            wind_kph=round(wind * 1.852, 1),
            pressure_hpa=round(1010 - wind * 0.5, 1),
            category=wind_to_imd_category(wind),
            data_type="SYNTHETIC_DEMO",
            point_index=i,
        )
        db.add(tp)

    _seed_datasets(db)
    _seed_models(db)
    db.commit()
    logger.info("Created minimal demo data")


def _seed_datasets(db):
    """Seed dataset metadata records."""
    datasets = [
        Dataset(
            id="demo_synthetic",
            name="CYCLONE-AI Demo Dataset",
            description="Synthetically generated demonstration data",
            source="CYCLONE-AI Generator",
            data_type="SYNTHETIC_DEMO",
            license="Project license",
            status="available",
        ),
        Dataset(
            id="ibtracs_ni",
            name="IBTrACS North Indian Ocean",
            description="International Best Track Archive - North Indian Ocean basin",
            source="NOAA NCEI",
            source_url="https://www.ncei.noaa.gov/products/international-best-track-archive",
            data_type="BEST_TRACK",
            license="Public domain (US Government)",
            status="not_downloaded",
        ),
        Dataset(
            id="tcir",
            name="TCIR Satellite Imagery",
            description="Tropical Cyclone Intensity estimation from IR satellite imagery",
            source="Academic dataset",
            data_type="SATELLITE_IMAGERY",
            status="not_downloaded",
        ),
        Dataset(
            id="era5",
            name="ERA5 Reanalysis",
            description="ECMWF Reanalysis v5 environmental variables",
            source="Copernicus Climate Data Store",
            source_url="https://cds.climate.copernicus.eu/",
            data_type="ENVIRONMENTAL",
            license="Copernicus License",
            status="not_downloaded",
        ),
        Dataset(
            id="gpm_imerg",
            name="NASA GPM IMERG",
            description="Global Precipitation Measurement - rainfall data",
            source="NASA",
            source_url="https://gpm.nasa.gov/data/imerg",
            data_type="RAINFALL",
            license="NASA Data Policy",
            status="not_downloaded",
        ),
    ]
    for ds in datasets:
        db.add(ds)


def _seed_models(db):
    """Seed model version metadata."""
    models = [
        ModelVersion(
            id="detector_v0.1",
            name="Cyclone Detector",
            version="0.1.0",
            task="detection",
            architecture="CNN + Classification + Localization heads",
            dataset_id="demo_synthetic",
            metrics={"status": "Not yet evaluated - untrained weights"},
            status="untrained",
        ),
        ModelVersion(
            id="classifier_v0.1",
            name="Cyclone Classifier",
            version="0.1.0",
            task="classification",
            architecture="CNN + 9-class IMD classification head",
            dataset_id="demo_synthetic",
            metrics={"status": "Not yet evaluated - untrained weights"},
            status="untrained",
        ),
        ModelVersion(
            id="intensity_v0.1",
            name="Intensity Estimator",
            version="0.1.0",
            task="intensity",
            architecture="CNN + MLP fusion, wind/pressure regression",
            dataset_id="demo_synthetic",
            metrics={"status": "Not yet evaluated - untrained weights"},
            status="untrained",
        ),
        ModelVersion(
            id="track_v0.1",
            name="Track Predictor",
            version="0.1.0",
            task="track",
            architecture="LSTM + CNN + MLP fusion, multi-horizon trajectory",
            dataset_id="demo_synthetic",
            metrics={"status": "Not yet evaluated - untrained weights"},
            status="untrained",
        ),
    ]
    for m in models:
        db.add(m)


def _seed_alerts(db, cyclone_id: str):
    """Seed demo alert records."""
    alerts = [
        Alert(
            cyclone_id=cyclone_id,
            alert_level="ADVISORY",
            alert_type="intensity",
            message="[DEMO] Cyclone intensifying - AI model indicates potential category upgrade",
            data_type="SYNTHETIC_DEMO",
            is_active=True,
        ),
        Alert(
            cyclone_id=cyclone_id,
            alert_level="WATCH",
            alert_type="track",
            message="[DEMO] Predicted track approaching coastal region within 48 hours",
            data_type="SYNTHETIC_DEMO",
            is_active=True,
        ),
    ]
    for a in alerts:
        db.add(a)

"""
API Routes for CYCLONE-AI.

All prediction endpoints clearly indicate:
- Model version and status
- Data mode (DEMO/LIVE/HISTORICAL)
- Disclaimer about AI predictions
"""
import time
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..models.db_models import Cyclone, TrackPoint, Prediction, Alert, Dataset, ModelVersion
from ..schemas.schemas import (
    HealthResponse, SystemStatus, CycloneBase, CycloneDetail,
    TrackPointSchema, RiskAssessment, AlertSchema,
    DatasetSchema, ModelVersionSchema,
)
from ..core.config import settings
from ..services.ml_service import (
    get_predictor, get_sample_satellite_image,
    get_sample_env_features, get_sample_track_history, get_sample_track_origin,
    get_sample_track_raw,
)
from ml.baselines import (
    DEFAULT_HORIZONS, check_intensity, check_track, clip_lite, persistence,
)
from ml.baselines.track import (
    CLIP_LIKE_ERROR_KM, ERROR_SOURCE as BASELINE_ERROR_SOURCE, PERSISTENCE_ERROR_KM,
)
from ml.geo.coastline import distance_to_land_km
from ml.inference import track_risk

logger = logging.getLogger("cyclone_ai")

router = APIRouter()

_start_time = time.time()

DISCLAIMER = (
    "AI-generated prediction for research and decision-support purposes. "
    "This system is not an official warning system and must not replace "
    "official advisories issued by the India Meteorological Department "
    "or other authorized agencies."
)


# ============================================================
# Health & System
# ============================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    # Both of these used to be hardcoded strings. A health check that cannot
    # report a failure is decoration: it said "connected" with the database down,
    # and "baseline" would have kept saying "baseline" after the models were
    # trained. Both are now derived from the live state.
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
        status = "ok"
    except Exception as e:
        logger.error(f"Health check database probe failed: {e}")
        db_status = "unavailable"
        status = "degraded"

    try:
        predictor = get_predictor()
        trained = sum(1 for v in predictor.models_loaded.values() if v)
        total = len(predictor.models_loaded)
        if trained == 0:
            model_status = "untrained"
        elif trained < total:
            model_status = f"partially_trained ({trained}/{total})"
        else:
            model_status = "trained"
    except Exception as e:
        logger.error(f"Health check predictor probe failed: {e}")
        model_status = "unavailable"
        status = "degraded"

    return HealthResponse(
        status=status,
        version="0.1.0",
        environment=settings.APP_ENV,
        database=db_status,
        model_status=model_status,
        data_mode=settings.APP_ENV.upper(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


@router.get("/system/status", response_model=SystemStatus, tags=["System"])
def system_status(db: Session = Depends(get_db)):
    """Detailed system status."""
    total = db.query(Cyclone).count()
    active = db.query(Cyclone).filter(Cyclone.is_active == True).count()

    predictor = get_predictor()
    models_loaded = {}
    if predictor:
        models_loaded = predictor.models_loaded

    return SystemStatus(
        status="ok",
        version="0.1.0",
        environment=settings.APP_ENV,
        database="connected",
        models_loaded=models_loaded,
        data_mode=settings.APP_ENV.upper(),
        active_cyclones=active,
        total_cyclones=total,
        uptime_seconds=round(time.time() - _start_time, 1),
        disclaimer=DISCLAIMER,
    )


# ============================================================
# Cyclones
# ============================================================

@router.get("/cyclones", tags=["Cyclones"])
def list_cyclones(
    basin: Optional[str] = None,
    season: Optional[int] = None,
    active: Optional[bool] = None,
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    """List all cyclones with optional filters."""
    query = db.query(Cyclone)
    if basin:
        query = query.filter(Cyclone.basin == basin)
    if season:
        query = query.filter(Cyclone.season == season)
    if active is not None:
        query = query.filter(Cyclone.is_active == active)

    cyclones = query.order_by(Cyclone.season.desc()).limit(limit).all()

    return {
        "cyclones": [
            {
                "id": c.id,
                "name": c.name,
                "basin": c.basin,
                "sub_basin": c.sub_basin,
                "season": c.season,
                "peak_wind_kt": c.peak_wind_kt,
                "peak_wind_kph": c.peak_wind_kph,
                "min_pressure_hpa": c.min_pressure_hpa,
                "peak_category": c.peak_category,
                "num_observations": c.num_observations,
                "data_type": c.data_type,
                "is_active": c.is_active,
            }
            for c in cyclones
        ],
        "count": len(cyclones),
        "data_mode": settings.APP_ENV.upper(),
    }


@router.get("/cyclones/{cyclone_id}", tags=["Cyclones"])
def get_cyclone(cyclone_id: str, db: Session = Depends(get_db)):
    """Get detailed cyclone information."""
    cyclone = db.query(Cyclone).filter(Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")

    return {
        "id": cyclone.id,
        "name": cyclone.name,
        "basin": cyclone.basin,
        "sub_basin": cyclone.sub_basin,
        "season": cyclone.season,
        "start_time": cyclone.start_time,
        "end_time": cyclone.end_time,
        "peak_wind_kt": cyclone.peak_wind_kt,
        "peak_wind_kph": cyclone.peak_wind_kph,
        "min_pressure_hpa": cyclone.min_pressure_hpa,
        "peak_category": cyclone.peak_category,
        "num_observations": cyclone.num_observations,
        "data_type": cyclone.data_type,
        "data_source": cyclone.data_source,
        "is_active": cyclone.is_active,
        "data_mode": settings.APP_ENV.upper(),
    }


@router.get("/cyclones/{cyclone_id}/track", tags=["Cyclones"])
def get_cyclone_track(cyclone_id: str, db: Session = Depends(get_db)):
    """Get cyclone track points."""
    cyclone = db.query(Cyclone).filter(Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")

    points = (
        db.query(TrackPoint)
        .filter(TrackPoint.cyclone_id == cyclone_id)
        .order_by(TrackPoint.point_index)
        .all()
    )

    return {
        "cyclone_id": cyclone_id,
        "cyclone_name": cyclone.name,
        "track_points": [
            {
                "timestamp": p.timestamp,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "wind_knots": p.wind_knots,
                "wind_kph": p.wind_kph,
                "pressure_hpa": p.pressure_hpa,
                "category": p.category,
                "dist_to_coast_km": p.dist_to_coast_km,
                "data_type": p.data_type,
                "point_index": p.point_index,
            }
            for p in points
        ],
        "count": len(points),
        "data_mode": settings.APP_ENV.upper(),
    }


# ============================================================
# Predictions - Real ML Inference
# ============================================================

def _forecast_bundle(cyclone_id: str, db: Session, predictor, latest_point) -> dict:
    """Build the track forecast, its baselines, and the arbitration between them.

    Shared by ``/forecast`` and ``/risk`` so both score the same track. When the risk
    endpoint ran its own logic it read only the latest observed position, which meant the
    alert level and the forecast the map drew could disagree about where the storm was
    going — the sort of inconsistency that destroys trust in a decision-support tool much
    faster than a wide error bar does.
    """
    # Observed positions, oldest first — the input the analytic baselines need.
    history_points = (
        db.query(TrackPoint)
        .filter(TrackPoint.cyclone_id == cyclone_id)
        .order_by(TrackPoint.point_index.asc())
        .all()
    )
    history = [
        {
            "latitude": p.latitude,
            "longitude": p.longitude,
            "timestamp": p.timestamp,
            "wind_kph": p.wind_kph,
            "pressure_hpa": p.pressure_hpa,
        }
        for p in history_points
    ]

    # Load demo data for inference
    sat_image = get_sample_satellite_image(0)
    env_features = get_sample_env_features(0)
    track_history = get_sample_track_history(cyclone_id)

    # Run track prediction. The trained track-only model is tried first: it is the only
    # track model in the project with fitted weights and a season-held-out evaluation,
    # and it forecasts from position history alone. It returns None if it cannot honestly
    # forecast this history (too few points, or not on the 6-hourly grid it was trained
    # on), in which case we fall back to the multimodal net, which has no trained track
    # weights and will be caught by the primary-forecast arbitration below.
    track_forecast = predictor.predict_track_from_history(history)
    if track_forecast is None:
        track_forecast = predictor.predict_track(
            track_history, sat_image, env_features,
            current_lat=latest_point.latitude,
            current_lon=latest_point.longitude,
        )

    # Analytic and fitted baselines. These need no imagery, so they are the only forecast
    # that is meaningful while the network is untrained — and they stay in the response
    # afterwards as the skill reference the network has to beat.
    baselines = {}
    horizons = track_forecast.get("horizons_hours") or DEFAULT_HORIZONS
    for name, fn in (("persistence", persistence), ("clip_like", clip_lite)):
        result = fn(history, horizons=horizons)
        if result:
            baselines[name] = result
    # The fitted CLIPER is a real regression on this basin's best track and measurably
    # beats persistence (+8.5% at +24 h, +17.1% at +48 h), unlike clip_like, which
    # measured WORSE than persistence at every lead time. It is the reference the network
    # was scored against, so it belongs in every response.
    cliper_result = predictor.cliper_forecast_from_history(history)
    if cliper_result:
        baselines["cliper_fitted"] = cliper_result

    # Check the network's forecast against physical limits before reporting it.
    nn_plausibility = check_track(
        track_forecast.get("forecast_points", []),
        latest_point.latitude, latest_point.longitude,
    )
    track_forecast["plausibility"] = nn_plausibility.as_dict()

    # Decide what the UI should draw as the headline forecast. An untrained network,
    # or one whose output violates physical limits, must not be the primary number
    # shown next to an IMD category.
    nn_is_trained = track_forecast.get("model_status") == "trained"
    if nn_is_trained and nn_plausibility.ok:
        primary_forecast = "neural_network"
        primary_reason = (
            "Trained track model, output within physical limits. "
            + (track_forecast.get("uncertainty_method") or "")
        ).strip()
    else:
        # Fall back in measured order of skill, NOT in the order the methods were written.
        # clip_like used to be the fallback; measured on held-out seasons it is the worst
        # of the three (-10.5% to -32.7% against plain persistence), so preferring it was
        # actively choosing the least accurate available forecast.
        why = ("Neural track model is untrained or declined to forecast this history"
               if not nn_is_trained
               else "Neural track model output failed physical plausibility checks")
        for candidate, label in (
            ("cliper_fitted", "fitted CLIPER (beats persistence on held-out seasons)"),
            ("persistence", "persistence"),
            ("clip_like", "CLIP-like (measured worse than persistence; last resort)"),
        ):
            if candidate in baselines:
                primary_forecast = candidate
                primary_reason = f"{why}; showing {label} instead."
                break
        else:
            primary_forecast = "neural_network"
            primary_reason = "No baseline available (insufficient position history)."

    # Verified held-out accuracy, and the significance verdict against a fitted CLIPER.
    # Attached to every forecast response so the accuracy claim travels with the forecast
    # rather than living only on a separate page nobody opens.
    track_skill = predictor.track_skill_summary()

    # The positions risk is actually scored on: whichever forecast arbitration chose, not
    # whichever one happens to be the neural network.
    source = (track_forecast if primary_forecast == "neural_network"
              else baselines.get(primary_forecast, track_forecast))
    primary_points = source.get("forecast_points") or []

    return {
        "history": history,
        "sat_image": sat_image,
        "env_features": env_features,
        "track_forecast": track_forecast,
        "baselines": baselines,
        "primary_forecast": primary_forecast,
        "primary_reason": primary_reason,
        "primary_points": primary_points,
        "track_skill": track_skill,
    }


@router.get("/cyclones/{cyclone_id}/forecast", tags=["Predictions"])
def get_cyclone_forecast(cyclone_id: str, db: Session = Depends(get_db)):
    """Generate AI forecast for a cyclone using actual ML models."""
    cyclone = db.query(Cyclone).filter(Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")

    # Get latest track point for current position
    latest_point = (
        db.query(TrackPoint)
        .filter(TrackPoint.cyclone_id == cyclone_id)
        .order_by(TrackPoint.point_index.desc())
        .first()
    )

    if not latest_point:
        raise HTTPException(status_code=404, detail="No track data available")

    predictor = get_predictor()
    if not predictor:
        raise HTTPException(status_code=503, detail="ML models not available")

    bundle = _forecast_bundle(cyclone_id, db, predictor, latest_point)
    track_forecast = bundle["track_forecast"]
    baselines = bundle["baselines"]
    primary_forecast = bundle["primary_forecast"]
    primary_reason = bundle["primary_reason"]
    track_skill = bundle["track_skill"]
    sat_image = bundle["sat_image"]
    env_features = bundle["env_features"]

    # Run intensity estimation
    intensity = predictor.estimate_intensity(sat_image, env_features)
    intensity_plausibility = check_intensity(
        intensity["wind_kph"], intensity["pressure_hpa"]
    )

    # NOTE: intensity *forecasting* is deliberately not reported.
    # There is no trained intensity-forecast head in the project yet. This endpoint
    # used to synthesise one by scaling the track model's latitude uncertainty
    # (wind_delta = lat_uncertainty * 20 - 5), which is not a prediction at all and
    # produced unphysical output — e.g. 244 km/h at 827 hPa six hours after a
    # Depression, below the 870 hPa world-record minimum. Reporting nothing is more
    # defensible than reporting that. Restore this once a real head is trained.
    intensity_forecast = None
    intensity_forecast_status = "not_implemented"

    return {
        "cyclone_id": cyclone_id,
        "current_position": {
            "latitude": latest_point.latitude,
            "longitude": latest_point.longitude,
            "wind_kph": latest_point.wind_kph,
            "pressure_hpa": latest_point.pressure_hpa,
            "category": latest_point.category,
        },
        "current_intensity_estimate": {
            "wind_kph": intensity["wind_kph"],
            "pressure_hpa": intensity["pressure_hpa"],
            "imd_category": intensity.get("imd_category"),
            "model_status": intensity.get("model_status"),
            "confidence_interval": intensity.get("confidence_interval"),
            "plausibility": intensity_plausibility.as_dict(),
        },
        "track_forecast": track_forecast,
        "track_skill": track_skill,
        "baseline_forecasts": baselines,
        "primary_forecast": primary_forecast,
        "primary_forecast_reason": primary_reason,
        "intensity_forecast": intensity_forecast,
        "intensity_forecast_status": intensity_forecast_status,
        "data_mode": settings.APP_ENV.upper(),
        "disclaimer": DISCLAIMER,
    }


@router.get("/cyclones/{cyclone_id}/risk", tags=["Risk"])
def get_cyclone_risk(cyclone_id: str, db: Session = Depends(get_db)):
    """Risk assessment along the forecast track, not just at the current position.

    What changed and why
    --------------------
    This endpoint used to score the latest observed position and stop there. "How far from
    land is it now" is something anyone can read off a map; what a district officer needs is
    "when does this reach my coast, and how sure are you". So the response now carries a
    per-lead-time coastal profile, the closest approach along the forecast, and a landfall
    window split into two distinct answers — the hour landfall first falls inside the
    verified error cone, and the later hour the forecast centre itself arrives.

    It also passed ``ri_probability=0.1`` and ``track_confidence=0.5`` as literals, which
    together drove 30% of ``overall_risk`` without varying between storms. Both are now
    omitted: ``assess_risk`` renormalises over the components that exist and reports the
    rest as null. In their place the response carries observed 24-hour intensity change,
    which is measured from best-track history and is a real rapid-intensification signal.
    """
    cyclone = db.query(Cyclone).filter(Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")

    latest_point = (
        db.query(TrackPoint)
        .filter(TrackPoint.cyclone_id == cyclone_id)
        .order_by(TrackPoint.point_index.desc())
        .first()
    )

    predictor = get_predictor()
    if not predictor:
        # Models unavailable. Report that, rather than a plausible-looking score: a
        # hardcoded 0.3 / "WATCH" is indistinguishable downstream from a computed one, and
        # the previous version of this block emitted exactly that.
        risk = {
            "overall_risk": None,
            "wind_risk": None,
            "coastal_risk": None,
            "rainfall_risk": None,
            "rapid_intensification_risk": None,
            "track_confidence": None,
            "alert_level": "UNAVAILABLE",
            "components_used": [],
            "components_unavailable": ["wind", "coastal", "rainfall",
                                       "rapid_intensification", "track_uncertainty"],
            "methodology": (
                "No risk assessment: ML models are unavailable, so no component could be "
                "computed. Previously this path returned a fixed 0.3 / WATCH, which was "
                "indistinguishable from a real assessment."
            ),
        }
        return {"cyclone_id": cyclone_id, "risk": risk, "disclaimer": DISCLAIMER}

    wind_kph = latest_point.wind_kph if latest_point else 50
    pressure = latest_point.pressure_hpa if latest_point else 1000

    # Coastal risk is scored at the closest approach along the forecast, not at the current
    # position. A storm 400 km offshore whose forecast puts it 30 km from the Odisha coast
    # in 24 h is a coastal emergency now; scoring it at 400 km called it a low concern.
    track_context = None
    dist_coast = None
    if latest_point is not None:
        bundle = _forecast_bundle(cyclone_id, db, predictor, latest_point)
        track_context = track_risk.build(bundle["primary_points"], bundle["history"])
        track_context["forecast_source"] = bundle["primary_forecast"]
        track_context["forecast_source_reason"] = bundle["primary_reason"]
        closest = track_context.get("closest_approach")
        if closest:
            dist_coast = closest["distance_to_land_km"]

    if dist_coast is None:
        # No forecast available: fall back to the current position, measured with the same
        # geometry rather than read from the DB column, so one number does not silently come
        # from a different source than the rest.
        if latest_point is not None:
            dist_coast = distance_to_land_km(latest_point.latitude, latest_point.longitude)
            coastal_basis = "current observed position (no forecast track available)"
        else:
            dist_coast = 500.0
            coastal_basis = "no position data; nominal 500 km, coastal risk is not meaningful"
    else:
        coastal_basis = (
            f"closest approach along the {track_context['forecast_source']} forecast, "
            f"at +{track_context['closest_approach']['forecast_hour']} h"
        )

    risk = predictor.assess_risk(
        wind_kph=wind_kph,
        pressure_hpa=pressure,
        dist_to_coast_km=dist_coast,
        # ri_probability and track_confidence deliberately omitted. See assess_risk: they
        # were literals here, contributing 30% of the score while varying between no two
        # storms. Observed intensity change is reported separately, as an observation.
    )
    risk["coastal_risk_basis"] = coastal_basis
    risk["distance_to_coast_km"] = round(dist_coast, 1)

    return {
        "cyclone_id": cyclone_id,
        "risk": risk,
        "forecast_track_risk": track_context,
        "data_mode": settings.APP_ENV.upper(),
        "disclaimer": DISCLAIMER,
    }


# ============================================================
# Prediction Endpoints
# ============================================================

@router.post("/predict/detect", tags=["Predictions"])
def predict_detect(sample_index: int = 0):
    """Run cyclone detection on a sample satellite image."""
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(status_code=503, detail="ML models not available")

    sat_image = get_sample_satellite_image(sample_index)
    result = predictor.detect(sat_image)
    result["disclaimer"] = DISCLAIMER
    result["data_mode"] = settings.APP_ENV.upper()
    return result


@router.post("/predict/classification", tags=["Predictions"])
def predict_classification(sample_index: int = 0):
    """Run cyclone classification on a sample."""
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(status_code=503, detail="ML models not available")

    sat_image = get_sample_satellite_image(sample_index)
    env_features = get_sample_env_features(sample_index)
    result = predictor.classify(sat_image, env_features)
    result["disclaimer"] = DISCLAIMER
    result["data_mode"] = settings.APP_ENV.upper()
    return result


@router.post("/predict/intensity", tags=["Predictions"])
def predict_intensity(sample_index: int = 0):
    """Run intensity estimation."""
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(status_code=503, detail="ML models not available")

    sat_image = get_sample_satellite_image(sample_index)
    env_features = get_sample_env_features(sample_index)
    result = predictor.estimate_intensity(sat_image, env_features)
    # Same physical check the forecast endpoint applies. Without it this endpoint
    # reported a wind/pressure pair that the platform's own physics module rejects,
    # while /cyclones/{id}/forecast flagged it -- two screens disagreeing about the
    # same number is worse than either verdict alone.
    result["plausibility"] = check_intensity(
        result.get("wind_kph") or 0.0, result.get("pressure_hpa") or 0.0
    ).as_dict()
    result["disclaimer"] = DISCLAIMER
    result["data_mode"] = settings.APP_ENV.upper()
    return result


@router.post("/predict/track", tags=["Predictions"])
def predict_track(sample_index: int = 0, cyclone_id: str = None):
    """Run track prediction."""
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(status_code=503, detail="ML models not available")

    sat_image = get_sample_satellite_image(sample_index)
    env_features = get_sample_env_features(sample_index)
    track_history = get_sample_track_history(cyclone_id)
    # Anchor the forecast to the storm's real position. predict_track defaults to
    # current_lat=15.0, current_lon=85.0 (mid Bay of Bengal), so omitting these
    # returned a Bay of Bengal forecast for an Arabian Sea cyclone -- the requested
    # cyclone_id was accepted, used to build the history, and then silently ignored
    # for positioning, putting the track about 2200 km from the actual storm.
    origin = get_sample_track_origin(cyclone_id)

    # Trained track-only model first; it derives its own origin from the history, so it
    # cannot be mis-anchored the way the multimodal path could.
    raw_history = get_sample_track_raw(cyclone_id)
    result = (predictor.predict_track_from_history(raw_history)
              if raw_history else None)
    if result is None:
        result = predictor.predict_track(
            track_history, sat_image, env_features,
            **({"current_lat": origin[0], "current_lon": origin[1]} if origin else {}),
        )
    else:
        result["track_skill"] = predictor.track_skill_summary()
    points = result.get("forecast_points") or []
    # Measure displacement from the last OBSERVED position, not from the first
    # forecast point -- using a forecast point as its own origin makes the check
    # partly self-referential and hides any offset between analysis and forecast.
    if origin is None:
        origin = ((points[0]["latitude"], points[0]["longitude"]) if points else (0.0, 0.0))
    result["plausibility"] = check_track(points, origin[0], origin[1]).as_dict()
    result["disclaimer"] = DISCLAIMER
    result["data_mode"] = settings.APP_ENV.upper()
    return result


@router.post("/predict", tags=["Predictions"])
def predict_full(sample_index: int = 0, cyclone_id: str = None):
    """Run full analysis pipeline."""
    predictor = get_predictor()
    if not predictor:
        raise HTTPException(status_code=503, detail="ML models not available")

    sat_image = get_sample_satellite_image(sample_index)
    env_features = get_sample_env_features(sample_index)
    track_history = get_sample_track_history(cyclone_id)
    # Same anchoring as /predict/track: full_analysis also defaults to 15N 85E.
    origin = get_sample_track_origin(cyclone_id)
    result = predictor.full_analysis(
        sat_image, env_features, track_history,
        **({"current_lat": origin[0], "current_lon": origin[1]} if origin else {}),
    )
    result["data_mode"] = settings.APP_ENV.upper()
    return result


# ============================================================
# Datasets
# ============================================================

@router.get("/datasets", tags=["Datasets"])
def list_datasets(db: Session = Depends(get_db)):
    """List all registered datasets."""
    datasets = db.query(Dataset).all()
    return {
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "source": d.source,
                "source_url": d.source_url,
                "data_type": d.data_type,
                "license": d.license,
                "status": d.status,
            }
            for d in datasets
        ],
        "count": len(datasets),
    }


@router.get("/datasets/{dataset_id}", tags=["Datasets"])
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "source": ds.source,
        "source_url": ds.source_url,
        "data_type": ds.data_type,
        "license": ds.license,
        "status": ds.status,
    }


# ============================================================
# Models
# ============================================================

def _measured_rows() -> List[dict]:
    """Registry rows for every forecaster whose accuracy has actually been measured.

    Built at request time from the loaded checkpoint and from the baseline error tables,
    NOT read from the ModelVersion table. The seeded registry is written once when the
    database is created and is guarded against re-seeding, so a metric stored there would
    keep reporting the accuracy of whatever was trained the day the database was made.
    Deriving these rows from the artefact that is actually being served means the page
    cannot claim a number the weights do not support.

    Every row here is a *forecaster the API will serve*. The untrained multimodal models
    seeded in the database are reported separately and still say untrained, because they
    are.
    """
    rows: List[dict] = []

    def error_table(errs: dict, key=str) -> dict:
        return {
            f"+{h}h mean error (km)": round(errs[key(h)]["mean_km"], 1)
            for h in sorted(int(k) for k in errs)
        } | {
            f"+{h}h 90th pct (km)": round(errs[key(h)]["p90_km"], 1)
            for h in sorted(int(k) for k in errs)
        } | {
            "verifying cases": errs[key(min(int(k) for k in errs))]["n"],
        }

    def structured(errs: dict, key=str) -> dict:
        """The same numbers as machine-readable fields.

        Given alongside the flat display metrics so the UI can draw a comparison table
        without parsing label strings like "+24h mean error (km)" -- a renaming of which
        would otherwise silently blank a column.
        """
        return {
            str(h): {"mean_km": round(errs[key(h)]["mean_km"], 1),
                     "p90_km": round(errs[key(h)]["p90_km"], 1),
                     "n": errs[key(h)]["n"]}
            for h in sorted(int(k) for k in errs)
        }

    def skill_vs_pers(errs: dict, key=str) -> dict:
        return {
            str(h): round(100.0 * (PERSISTENCE_ERROR_KM[h]["mean_km"]
                                   - errs[key(h)]["mean_km"])
                          / PERSISTENCE_ERROR_KM[h]["mean_km"], 1)
            for h in sorted(int(k) for k in errs) if h in PERSISTENCE_ERROR_KM
        }

    predictor = get_predictor()
    runtime = getattr(predictor, "track_runtime", None) if predictor else None

    if runtime is not None:
        meta = runtime.meta
        skill = runtime.skill_summary()
        seasons = meta.get("test_seasons") or meta.get("val_seasons") or []
        metrics = error_table(skill["error_km"])
        metrics["evaluated on"] = (
            f"held-out seasons {min(seasons)}-{max(seasons)}" if seasons else "unrecorded")
        metrics["verdict vs fitted CLIPER"] = skill["verdict"]
        # Skill against persistence is the claim that is actually defensible, so state it
        # rather than leaving the reader to divide the two tables in their head.
        metrics["skill vs persistence (%)"] = ", ".join(
            f"+{h}h {v:+.1f}" for h, v in
            sorted((int(k), v) for k, v in skill_vs_pers(skill["error_km"]).items()))
        rows.append({
            "id": "track_only_v1",
            "name": "CycloneAI TrackNet (track-only)",
            "version": "1.0.0",
            "task": "track",
            "architecture": (
                f"GRU encoder + multi-horizon displacement head, "
                f"{meta.get('n_parameters', 0):,} parameters, residual-persistence "
                f"parameterisation"
            ),
            "dataset_id": meta.get("dataset") or "ibtracs_ni_v04r01",
            "metrics": metrics,
            "status": "trained",
            "verified_error_km": structured(skill["error_km"]),
            "skill_vs_persistence_pct": skill_vs_pers(skill["error_km"]),
            "significance_vs_cliper": skill.get("significance_vs_cliper") or {},
            "is_primary": True,
        })

    if runtime is not None and runtime.cliper is not None and runtime.cliper_error_km:
        rows.append({
            "id": "cliper_ni_v1",
            "name": "CLIPER (fitted, North Indian Ocean)",
            "version": "1.0.0",
            "task": "track",
            "architecture": (
                "Least-squares multiple regression on 12 climatology-and-persistence "
                "predictors, one fit per horizon and component"
            ),
            "dataset_id": "ibtracs_ni_v04r01",
            "metrics": error_table(runtime.cliper_error_km, key=int) | {
                "role": "the baseline an operational track forecast must beat",
                "skill vs persistence (%)": ", ".join(
                    f"+{h}h {v:+.1f}" for h, v in
                    sorted((int(k), v) for k, v in
                           skill_vs_pers(runtime.cliper_error_km, key=int).items())),
            },
            "status": "fitted",
            "verified_error_km": structured(runtime.cliper_error_km, key=int),
            "skill_vs_persistence_pct": skill_vs_pers(runtime.cliper_error_km, key=int),
        })

    rows.append({
        "id": "persistence_v1",
        "name": "Persistence (PERS)",
        "version": "1.0.0",
        "task": "track",
        "architecture": "Current motion vector held constant along a great circle (no fit)",
        "dataset_id": "ibtracs_ni_v04r01",
        "metrics": error_table(PERSISTENCE_ERROR_KM, key=int) | {
            "role": "zero-skill reference every other forecast is scored against",
            "source": BASELINE_ERROR_SOURCE,
        },
        "status": "analytic",
        "verified_error_km": structured(PERSISTENCE_ERROR_KM, key=int),
        "skill_vs_persistence_pct": {},
    })
    rows.append({
        "id": "clip_like_v1",
        "name": "Persistence + Climatology (CLIP-like)",
        "version": "1.0.0",
        "task": "track",
        "architecture": "Persistence relaxed toward specified NIO climatology, 24 h e-folding",
        "dataset_id": "ibtracs_ni_v04r01",
        "metrics": error_table(CLIP_LIKE_ERROR_KM, key=int) | {
            "skill vs persistence (%)": ", ".join(
                f"+{h}h {v:+.1f}" for h, v in
                sorted((int(k), v) for k, v in
                       skill_vs_pers(CLIP_LIKE_ERROR_KM, key=int).items())),
            "note": ("Worse than persistence at every lead time. Kept for comparison and "
                     "superseded by the fitted CLIPER; never served as the primary."),
        },
        "status": "superseded",
        "verified_error_km": structured(CLIP_LIKE_ERROR_KM, key=int),
        "skill_vs_persistence_pct": skill_vs_pers(CLIP_LIKE_ERROR_KM, key=int),
    })
    return rows


@router.get("/models", tags=["Models"])
def list_models(db: Session = Depends(get_db)):
    """List every forecaster, measured ones first, then the untrained seeded models."""
    measured = _measured_rows()
    seeded = [
        {
            "id": m.id,
            "name": m.name,
            "version": m.version,
            "task": m.task,
            "architecture": m.architecture,
            "dataset_id": m.dataset_id,
            "metrics": m.metrics,
            "status": m.status,
        }
        for m in db.query(ModelVersion).all()
        # The seeded track_v0.1 row describes the multimodal TrackPredictor, which is a
        # different (still untrained) model from the track-only network above. Keeping
        # both would read as one model reported twice with contradictory status.
        if m.id != "track_v0.1"
    ]
    return {
        "models": measured + seeded,
        "count": len(measured) + len(seeded),
        "note": (
            f"{len(measured)} track forecasters have measured held-out accuracy "
            f"({BASELINE_ERROR_SOURCE}); metrics for them are read from the artefact "
            f"being served, not from the registry table. The remaining "
            f"{len(seeded)} satellite models are untrained and report no metrics."
        ),
    }


@router.get("/models/{model_id}", tags=["Models"])
def get_model(model_id: str, db: Session = Depends(get_db)):
    m = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not m:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {
        "id": m.id,
        "name": m.name,
        "version": m.version,
        "task": m.task,
        "architecture": m.architecture,
        "metrics": m.metrics,
        "status": m.status,
    }


# ============================================================
# Alerts
# ============================================================

@router.get("/alerts", tags=["Alerts"])
def list_alerts(active: bool = True, db: Session = Depends(get_db)):
    """List alerts."""
    query = db.query(Alert)
    if active:
        query = query.filter(Alert.is_active == True)
    alerts = query.order_by(Alert.id.desc()).limit(50).all()
    return {
        "alerts": [
            {
                "id": a.id,
                "cyclone_id": a.cyclone_id,
                "alert_level": a.alert_level,
                "alert_type": a.alert_type,
                "message": a.message,
                "is_active": a.is_active,
                "data_type": a.data_type,
            }
            for a in alerts
        ],
        "count": len(alerts),
        "data_mode": settings.APP_ENV.upper(),
    }


@router.post("/alerts/evaluate", tags=["Alerts"])
def evaluate_alerts(cyclone_id: str, db: Session = Depends(get_db)):
    """Evaluate alert rules for a cyclone."""
    cyclone = db.query(Cyclone).filter(Cyclone.id == cyclone_id).first()
    if not cyclone:
        raise HTTPException(status_code=404, detail=f"Cyclone {cyclone_id} not found")

    # Get risk assessment
    predictor = get_predictor()
    if not predictor:
        return {"alerts": [], "message": "ML models not available for alert evaluation"}

    latest = (
        db.query(TrackPoint)
        .filter(TrackPoint.cyclone_id == cyclone_id)
        .order_by(TrackPoint.point_index.desc())
        .first()
    )

    wind = latest.wind_kph if latest else 50
    pressure = latest.pressure_hpa if latest else 1000

    # Score the closest approach along the forecast, not the current position. This
    # endpoint writes Alert rows, so the distinction is not cosmetic: a storm still 400 km
    # offshore whose forecast puts it 30 km from the coast in 24 h needs the alert now, and
    # scoring it where it currently sits raised nothing until it was too late to act on.
    dist = None
    basis = None
    if latest is not None:
        bundle = _forecast_bundle(cyclone_id, db, predictor, latest)
        ctx = track_risk.build(bundle["primary_points"], bundle["history"])
        closest = ctx.get("closest_approach")
        if closest:
            dist = closest["distance_to_land_km"]
            basis = f"closest forecast approach, +{closest['forecast_hour']} h"
    if dist is None:
        # Prefer the observed DIST2LAND the best track ships; fall back to measuring the
        # current position with the same geometry rather than to a magic 500.
        if latest is not None and latest.dist_to_coast_km:
            dist = latest.dist_to_coast_km
            basis = "observed best-track distance to land, current position"
        elif latest is not None:
            dist = distance_to_land_km(latest.latitude, latest.longitude)
            basis = "measured distance to land, current position"
        else:
            dist = 500.0
            basis = "no position data"

    risk = predictor.assess_risk(wind, pressure, dist)
    risk["coastal_risk_basis"] = basis

    new_alerts = []
    if risk["alert_level"] in ["HIGH RISK", "EXTREME RISK"]:
        alert = Alert(
            cyclone_id=cyclone_id,
            alert_level=risk["alert_level"],
            alert_type="risk_assessment",
            message=f"[{settings.APP_ENV.upper()}] AI risk assessment: {risk['alert_level']} - Wind: {wind} km/h",
            data_type="AI_GENERATED",
            is_active=True,
        )
        db.add(alert)
        db.commit()
        new_alerts.append({
            "alert_level": risk["alert_level"],
            "message": alert.message,
        })

    return {
        "cyclone_id": cyclone_id,
        "risk": risk,
        "new_alerts": new_alerts,
        "data_mode": settings.APP_ENV.upper(),
        "disclaimer": DISCLAIMER,
    }


# ============================================================
# Metrics & Observations
# ============================================================

@router.get("/metrics", tags=["Metrics"])
def get_metrics(db: Session = Depends(get_db)):
    """Get model performance metrics."""
    measured = _measured_rows()
    return {
        "models": [
            {"id": r["id"], "name": r["name"], "task": r["task"],
             "status": r["status"], "metrics": r["metrics"]}
            for r in measured
        ] + [
            {
                "id": m.id,
                "name": m.name,
                "task": m.task,
                "status": m.status,
                # Says untrained because it is. Previously read "not yet evaluated -
                # baseline weights", which invites the reading that evaluation is merely
                # pending on a model that works.
                "metrics": m.metrics or {
                    "status": "Untrained weights - no metrics, and none will be shown"},
            }
            for m in db.query(ModelVersion).all()
            if m.id != "track_v0.1"
        ],
        "note": (
            "Track-forecast metrics are measured great-circle error on held-out IBTrACS "
            "North Indian Ocean seasons, read from the artefact being served. The "
            "satellite models are untrained and report no metrics."
        ),
        "data_mode": settings.APP_ENV.upper(),
    }


@router.get("/observations", tags=["Observations"])
def list_observations(
    cyclone_id: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """List track observations."""
    query = db.query(TrackPoint)
    if cyclone_id:
        query = query.filter(TrackPoint.cyclone_id == cyclone_id)
    points = query.order_by(TrackPoint.id.desc()).limit(limit).all()
    return {
        "observations": [
            {
                "id": p.id,
                "cyclone_id": p.cyclone_id,
                "timestamp": p.timestamp,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "wind_knots": p.wind_knots,
                "pressure_hpa": p.pressure_hpa,
                "category": p.category,
                "data_type": p.data_type,
            }
            for p in points
        ],
        "count": len(points),
        "data_mode": settings.APP_ENV.upper(),
    }


@router.get("/observations/{observation_id}", tags=["Observations"])
def get_observation(observation_id: int, db: Session = Depends(get_db)):
    obs = db.query(TrackPoint).filter(TrackPoint.id == observation_id).first()
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found")
    return {
        "id": obs.id,
        "cyclone_id": obs.cyclone_id,
        "timestamp": obs.timestamp,
        "latitude": obs.latitude,
        "longitude": obs.longitude,
        "wind_knots": obs.wind_knots,
        "pressure_hpa": obs.pressure_hpa,
        "category": obs.category,
        "data_type": obs.data_type,
    }

# ============================================================
# Live External API Integrations (IMD)
# ============================================================

from ..services.imd_service import fetch_live_cyclone_tracks

@router.get("/imd/live-tracks", tags=["Live Data"])
def get_live_imd_tracks():
    """
    Fetch live cyclone tracks from the official IMD API.
    Requires IMD_API_KEY to be configured.
    """
    result = fetch_live_cyclone_tracks()
    if not result["status"]:
        # If API key is missing or failed, return 503 or 401 depending on the case
        status_code = 401 if "Authorization" in result["message"] else 503
        raise HTTPException(status_code=status_code, detail=result["message"])
        
    return {
        "source": "IMD (India Meteorological Department)",
        "data": result["data"],
        "message": result["message"]
    }

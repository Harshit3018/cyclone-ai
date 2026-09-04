"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


# === Health ===
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    environment: str = "demo"
    database: str = "connected"
    model_status: str = "untrained"
    data_mode: str = "DEMO"
    timestamp: str = ""


class SystemStatus(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    models_loaded: Dict[str, bool] = {}
    data_mode: str
    active_cyclones: int = 0
    total_cyclones: int = 0
    uptime_seconds: float = 0
    disclaimer: str = (
        "AI-generated predictions for research and decision-support purposes. "
        "Not an official warning system."
    )


# === Cyclone ===
class CycloneBase(BaseModel):
    id: str
    name: str
    basin: str = "NI"
    sub_basin: Optional[str] = None
    season: int = 0
    peak_wind_kt: Optional[float] = None
    peak_wind_kph: Optional[float] = None
    min_pressure_hpa: Optional[float] = None
    peak_category: Optional[str] = None
    data_type: str = "SYNTHETIC_DEMO"
    is_active: bool = False


class CycloneDetail(CycloneBase):
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    num_observations: int = 0
    data_source: Optional[str] = None


class TrackPointSchema(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    wind_knots: Optional[float] = None
    wind_kph: Optional[float] = None
    pressure_hpa: Optional[float] = None
    category: Optional[str] = None
    dist_to_coast_km: Optional[float] = None
    data_type: str = "SYNTHETIC_DEMO"
    point_index: Optional[int] = None


# === Predictions ===
class DetectionResult(BaseModel):
    cyclone_detected: bool
    confidence: float
    center: Dict[str, float]
    inference_time_ms: float = 0
    model_name: str = ""
    model_version: str = ""
    model_status: str = "untrained"
    timestamp: str = ""


class ClassificationResult(BaseModel):
    classification: str = Field(alias="class", default="")
    class_index: int = 0
    confidence: float = 0
    probabilities: Dict[str, float] = {}
    inference_time_ms: float = 0
    model_name: str = ""
    model_version: str = ""
    model_status: str = "untrained"

    class Config:
        populate_by_name = True


class IntensityResult(BaseModel):
    wind_kph: float
    pressure_hpa: float
    imd_category: Optional[str] = None
    confidence_interval: Optional[Dict[str, Any]] = None
    inference_time_ms: float = 0
    model_name: str = ""
    model_version: str = ""
    model_status: str = "untrained"


class ForecastPoint(BaseModel):
    forecast_hour: int
    latitude: float
    longitude: float
    lat_uncertainty: float = 0
    lon_uncertainty: float = 0
    uncertainty_radius_km: float = 0


class TrackForecast(BaseModel):
    forecast_points: List[ForecastPoint] = []
    horizons_hours: List[int] = [6, 12, 24, 48]
    uncertainty_method: str = ""
    inference_time_ms: float = 0
    model_name: str = ""
    model_version: str = ""
    model_status: str = "untrained"


class RiskAssessment(BaseModel):
    """Risk score with its components.

    Every component is Optional because a component that could not be computed is reported
    as null rather than defaulted. ``rapid_intensification_risk`` and ``track_confidence``
    are null in practice today: there is no trained intensity-forecast head and no per-storm
    track confidence estimate, and they were previously filled with the literals 0.1 and 0.5
    at the call site while carrying 30% of the score's weight. ``weights_applied`` shows how
    the remaining weights were renormalised, so a consumer can see what the number averaged
    over.
    """
    overall_risk: Optional[float] = None
    wind_risk: Optional[float] = None
    coastal_risk: Optional[float] = None
    rainfall_risk: Optional[float] = None
    rapid_intensification_risk: Optional[float] = None
    track_confidence: Optional[float] = None
    alert_level: str
    components_used: List[str] = []
    components_unavailable: List[str] = []
    weights_applied: Dict[str, float] = {}
    component_provenance: Dict[str, str] = {}
    coastal_risk_basis: Optional[str] = None
    distance_to_coast_km: Optional[float] = None
    methodology: str = ""


class FullAnalysis(BaseModel):
    detection: Dict[str, Any]
    classification: Dict[str, Any]
    intensity: Dict[str, Any]
    track_forecast: Dict[str, Any]
    rapid_intensification: Dict[str, Any]
    risk: Dict[str, Any]
    explanation: Dict[str, Any]
    total_inference_time_ms: float
    disclaimer: str


# === Datasets ===
class DatasetSchema(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    version: Optional[str] = None
    data_type: Optional[str] = None
    license: Optional[str] = None
    status: str = "not_downloaded"


class ModelVersionSchema(BaseModel):
    id: str
    name: str
    version: Optional[str] = None
    task: Optional[str] = None
    architecture: Optional[str] = None
    dataset_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str = "untrained"


# === Alerts ===
class AlertSchema(BaseModel):
    id: int = 0
    cyclone_id: str
    alert_level: str
    alert_type: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    is_active: bool = True
    data_type: str = "SYNTHETIC_DEMO"

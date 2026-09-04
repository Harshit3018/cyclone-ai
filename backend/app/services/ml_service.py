"""
ML Inference Service - Bridge between FastAPI and ML models.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cyclone_ai")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Singleton predictor
_predictor = None


def get_predictor():
    """Lazy-load the ML predictor singleton."""
    global _predictor
    if _predictor is None:
        try:
            from ml.inference.predictor import CyclonePredictor
            from ml.utils import load_model_config

            config = load_model_config()
            model_dir = str(PROJECT_ROOT / "models" / "checkpoints")
            _predictor = CyclonePredictor(model_dir=model_dir, config=config)
            logger.info("ML Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML predictor: {e}")
            _predictor = None
    return _predictor


def get_sample_satellite_image(index: int = 0) -> Optional[np.ndarray]:
    """Load a sample satellite tensor from demo data."""
    try:
        path = PROJECT_ROOT / "data" / "sample" / "demo_satellite_tensors.npy"
        if path.exists():
            tensors = np.load(str(path))
            idx = index % len(tensors)
            return tensors[idx]
        return None
    except Exception as e:
        logger.error(f"Error loading satellite tensor: {e}")
        return None


def get_sample_env_features(index: int = 0) -> Optional[np.ndarray]:
    """Load sample environmental features from demo data."""
    try:
        path = PROJECT_ROOT / "data" / "sample" / "demo_env_features.npy"
        if path.exists():
            features = np.load(str(path))
            idx = index % len(features)
            return features[idx]
        return None
    except Exception as e:
        logger.error(f"Error loading env features: {e}")
        return None


def get_sample_track_history(cyclone_id: str = None) -> Optional[np.ndarray]:
    """Generate track history array from demo track data."""
    import json
    try:
        path = PROJECT_ROOT / "data" / "sample" / "demo_tracks.json"
        if not path.exists():
            return None
        
        with open(path) as f:
            tracks = json.load(f)
        
        if cyclone_id and cyclone_id in tracks:
            track = tracks[cyclone_id]
        else:
            track = list(tracks.values())[0]
        
        # Convert to feature array [T, features]
        # Features: lat, lon, wind_kt, wind_kph, pressure, dlat, dlon, dt, sin_dir, cos_dir
        history = []
        for i, pt in enumerate(track[-8:]):  # last 8 points
            lat = pt["latitude"]
            lon = pt["longitude"]
            wind = pt.get("wind_knots", 0)
            wind_kph = pt.get("wind_kph", 0)
            pres = pt.get("pressure_hpa", 1010)
            
            dlat = 0 if i == 0 else lat - track[max(0, len(track)-8+i-1)]["latitude"]
            dlon = 0 if i == 0 else lon - track[max(0, len(track)-8+i-1)]["longitude"]
            
            import math
            direction = math.atan2(dlat, dlon)
            
            history.append([
                lat / 30.0,  # normalize
                lon / 100.0,
                wind / 150.0,
                wind_kph / 280.0,
                (pres - 900) / 120.0,
                dlat,
                dlon,
                0.25,  # 6-hour timestep normalized
                math.sin(direction),
                math.cos(direction),
            ])
        
        return np.array(history, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error loading track history: {e}")
        return None


def get_sample_track_raw(cyclone_id: str = None) -> Optional[list]:
    """Demo track history in raw physical units, as a list of dicts.

    This is what the trained track-only model consumes. It is deliberately *unnormalised*:
    the feature vector is built by ml.datasets.track_dataset.step_features, the same
    function used during training, so any scaling applied here would be applied twice.

    Kept separate from get_sample_track_history, which pre-scales ten columns for the
    untrained multimodal TrackPredictor under a different and incompatible convention.
    """
    import json
    try:
        path = PROJECT_ROOT / "data" / "sample" / "demo_tracks.json"
        if not path.exists():
            return None
        with open(path) as f:
            tracks = json.load(f)
        track = tracks.get(cyclone_id) if cyclone_id else None
        if not track:
            track = next(iter(tracks.values()), None)
        if not track:
            return None
        return [
            {
                "latitude": pt["latitude"],
                "longitude": pt["longitude"],
                "timestamp": pt["timestamp"],
                "wind_kph": pt.get("wind_kph"),
                "pressure_hpa": pt.get("pressure_hpa"),
            }
            for pt in track
        ]
    except Exception as e:
        logger.error(f"Error loading raw track history: {e}")
        return None


def get_sample_track_origin(cyclone_id: str = None) -> Optional[tuple]:
    """Last observed (latitude, longitude) in real degrees.

    Callers need the true analysis position to measure a forecast's displacement
    against. They must not read it out of get_sample_track_history: the columns there
    are scaled for the network (lat/30, lon/100), so index 0 of the last row is 0.5,
    not 15.0 degrees. Keeping the descaling out of the API layer means the scaling
    constants stay in one place.
    """
    import json
    try:
        path = PROJECT_ROOT / "data" / "sample" / "demo_tracks.json"
        if not path.exists():
            return None
        with open(path) as f:
            tracks = json.load(f)
        track = tracks.get(cyclone_id) if cyclone_id else None
        if not track:
            track = next(iter(tracks.values()), None)
        if not track:
            return None
        last = track[-1]
        return float(last["latitude"]), float(last["longitude"])
    except Exception as e:
        logger.error(f"Error loading track origin: {e}")
        return None

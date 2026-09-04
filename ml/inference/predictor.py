"""
Unified Inference Pipeline for CYCLONE-AI.

Provides a single entry point for all prediction tasks:
- Detection
- Classification
- Intensity estimation
- Track prediction
- Rapid intensification
- Uncertainty estimation
- Explainability
- Risk assessment
"""
import torch
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from ..models import (
    CycloneDetector,
    CycloneClassifier,
    IntensityModel,
    TrackPredictor,
    MultimodalCycloneModel,
)
from ..explainability.gradcam import GradCAM, compute_feature_importance
from .track_runtime import TrackOnlyRuntime
from .track_risk import RI_THRESHOLD_KT_24H
from ..utils import (
    get_device,
    load_model_config,
    IMD_CATEGORIES,
    knots_to_kmh,
    wind_to_imd_category,
)
from ..utils.env_normalization import EnvNormalizer

logger = logging.getLogger(__name__)


class CyclonePredictor:
    """
    Unified prediction pipeline for all cyclone analysis tasks.
    
    Loads models, runs inference, computes uncertainty, and generates explanations.
    All predictions include model version metadata.
    """

    def __init__(
        self,
        model_dir: str = "models/checkpoints",
        device: str = "auto",
        config: dict = None,
    ):
        self.device = get_device(device)
        self.config = config or load_model_config()
        self.model_dir = Path(model_dir)
        self.model_version = self.config.get("version", {})

        # Standardises environmental features before they reach any model. Starts from
        # climatological priors; _load_checkpoint replaces it with the statistics saved
        # in a checkpoint, so inference always uses the same transform training used.
        self.env_normalizer = EnvNormalizer()

        # Initialize models
        self._init_models()
        
        # Track model status
        self.models_loaded = {
            "detector": False,
            "classifier": False,
            "intensity": False,
            "track": False,
            "multimodal": False,
        }

        # Try loading checkpoints
        self._load_checkpoints()

        # The track-only forecaster is loaded separately from the multimodal checkpoints
        # above because it is a different model with a different input contract: it needs
        # only position history, where TrackPredictor requires satellite imagery and
        # reanalysis fields that no dataset in this project supplies. It is the only track
        # model with trained weights and a held-out evaluation, so it is what actually
        # serves forecasts; TrackPredictor stays wired up for when imagery is assembled.
        self.track_runtime = TrackOnlyRuntime.load(
            self.model_dir / "track_only.pt", device=self.device)
        self.models_loaded["track_only"] = self.track_runtime is not None
        if self.track_runtime is not None:
            # The fitted CLIPER is the reference the network is scored against, so it is
            # served from the same runtime and the same feature path. That keeps the
            # baseline shown in the UI byte-identical to the one in the evaluation table.
            self.models_loaded["cliper"] = self.track_runtime.attach_cliper(
                self.model_dir / "cliper_ni.json")

        logger.info(f"CyclonePredictor initialized on {self.device}")
        logger.info(f"Models loaded: {self.models_loaded}")

    def _init_models(self):
        """Initialize all model architectures."""
        cfg = self.config

        sat_cfg = cfg.get("satellite_cnn", {})
        in_ch = sat_cfg.get("input_channels", 4)
        feat_dim = sat_cfg.get("feature_dim", 128)

        self.detector = CycloneDetector(in_ch, feat_dim).to(self.device)
        
        env_cfg = cfg.get("environmental_mlp", {})
        env_dim = env_cfg.get("input_dim", 20)
        
        self.classifier = CycloneClassifier(
            in_ch, cfg.get("classifier", {}).get("num_classes", 9), feat_dim,
            env_dim=env_dim,
        ).to(self.device)
        
        self.intensity_model = IntensityModel(
            in_ch, feat_dim,
            env_cfg.get("input_dim", 20),
            env_cfg.get("output_dim", 64),
        ).to(self.device)
        
        track_cfg = cfg.get("temporal_model", {})
        self.track_model = TrackPredictor(
            track_input_dim=track_cfg.get("input_dim", 10),
            track_hidden_dim=track_cfg.get("hidden_dim", 64),
            sat_channels=in_ch,
            sat_feature_dim=feat_dim,
            env_input_dim=env_cfg.get("input_dim", 20),
            env_feature_dim=env_cfg.get("output_dim", 64),
            forecast_horizons=len(cfg.get("track", {}).get("forecast_horizons", [6, 12, 24, 48])),
        ).to(self.device)

        # Set to eval mode
        self.detector.eval()
        self.classifier.eval()
        self.intensity_model.eval()
        self.track_model.eval()

    def _load_checkpoints(self):
        """Try loading saved model weights."""
        checkpoint_map = {
            "detector": ("detector.pt", self.detector),
            "classifier": ("classifier.pt", self.classifier),
            "intensity": ("intensity.pt", self.intensity_model),
            "track": ("track.pt", self.track_model),
        }

        for name, (filename, model) in checkpoint_map.items():
            path = self.model_dir / filename
            if path.exists():
                try:
                    state = torch.load(path, map_location=self.device, weights_only=True)
                    # Two accepted formats: a bare state_dict, or a wrapper carrying the
                    # env normalisation statistics alongside the weights. The wrapper is
                    # preferred -- applying a different feature transform at inference
                    # than at training silently destroys accuracy, with no error to
                    # trace, so the transform belongs with the weights.
                    if isinstance(state, dict) and "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"])
                        self._adopt_env_normalizer(state.get("env_normalizer"), name)
                    else:
                        model.load_state_dict(state)
                    self.models_loaded[name] = True
                    logger.info(f"Loaded {name} model from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
                    self.models_loaded[name] = False
            else:
                # No trained weights: the model runs on its random initialisation.
                self.models_loaded[name] = False
                logger.info(f"No checkpoint for {name} - using untrained weights")

    def _adopt_env_normalizer(self, stats: dict | None, source: str):
        """Adopt env normalisation statistics saved with a checkpoint."""
        if not stats:
            logger.warning(
                f"Checkpoint '{source}' carries no env_normalizer; falling back to "
                f"climatological priors. If it was trained on fitted statistics its "
                f"accuracy will be degraded."
            )
            return
        try:
            self.env_normalizer = EnvNormalizer.from_dict(stats)
            logger.info(f"Adopted env normalisation from '{source}': {self.env_normalizer}")
        except Exception as e:
            logger.warning(f"Bad env_normalizer in '{source}' ({e}); keeping priors")

    def _model_metadata(self, task: str) -> dict:
        """Return model version metadata for every prediction."""
        return {
            "model_name": self.model_version.get("name", "CycloneAI Multimodal Net"),
            "model_version": self.model_version.get("version", "0.1.0"),
            # "untrained" rather than "baseline": ml/baselines holds the analytic
            # baselines (persistence, CLIP-like), and conflating "no checkpoint" with
            # "reference forecast" hides exactly the distinction that matters.
            "model_status": "trained" if self.models_loaded.get(task, False) else "untrained",
            "device": str(self.device),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def detect(self, satellite_image: np.ndarray, center_lat: float = 15.0, center_lon: float = 85.0) -> Dict[str, Any]:
        """
        Detect cyclone presence and localize center.
        
        Args:
            satellite_image: [C, H, W] or [H, W, C] numpy array
            center_lat: reference latitude for the image center
            center_lon: reference longitude for the image center
        """
        start = time.time()
        tensor = self._prepare_satellite(satellite_image)

        with torch.no_grad():
            output = self.detector(tensor)

        prob = float(output["detection_prob"][0])
        offset = output["center_offset"][0].cpu().numpy()

        # Convert normalized offset to lat/lon
        pred_lat = center_lat + float(offset[0]) * 5.0  # scale factor
        pred_lon = center_lon + float(offset[1]) * 5.0

        elapsed = time.time() - start

        return {
            "cyclone_detected": prob > 0.5,
            "confidence": round(prob, 4),
            "center": {
                "latitude": round(pred_lat, 2),
                "longitude": round(pred_lon, 2),
            },
            "inference_time_ms": round(elapsed * 1000, 1),
            **self._model_metadata("detector"),
        }

    def classify(self, satellite_image: np.ndarray, env_features: np.ndarray = None) -> Dict[str, Any]:
        """
        Classify cyclone stage using IMD categories.
        """
        start = time.time()
        tensor = self._prepare_satellite(satellite_image)
        # Always supply env features. The classifier head is sized for
        # feature_dim + env_dim, so passing None used to raise a shape error
        # (1x128 against 148x128) and return HTTP 500 for any classification
        # request without environmental data. _prepare_env(None) substitutes
        # climatological means, which is the meteorologically honest reading of
        # "no environmental data available" anyway.
        env_tensor = self._prepare_env(env_features)

        with torch.no_grad():
            output = self.classifier(tensor, env_tensor)

        probs = output["probabilities"][0].cpu().numpy()
        pred_idx = int(output["predicted_class"][0])

        prob_dict = {
            IMD_CATEGORIES[i]: round(float(probs[i]), 4)
            for i in range(len(IMD_CATEGORIES))
        }

        elapsed = time.time() - start

        return {
            "class": IMD_CATEGORIES[pred_idx],
            "class_index": pred_idx,
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": prob_dict,
            "inference_time_ms": round(elapsed * 1000, 1),
            **self._model_metadata("classifier"),
        }

    def estimate_intensity(
        self, satellite_image: np.ndarray, env_features: np.ndarray,
        with_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate cyclone intensity (wind speed and pressure).
        """
        start = time.time()
        sat_tensor = self._prepare_satellite(satellite_image)
        env_tensor = self._prepare_env(env_features)

        if with_uncertainty:
            output = self.intensity_model.predict_with_uncertainty(
                sat_tensor, env_tensor, n_samples=self.config.get("intensity", {}).get("mc_samples", 20)
            )
            result = {
                "wind_kph": round(float(output["wind_kph"][0]), 1),
                "pressure_hpa": round(float(output["pressure_hpa"][0]), 1),
                "confidence_interval": {
                    "wind_low": round(float(output["wind_low"][0]), 1),
                    "wind_high": round(float(output["wind_high"][0]), 1),
                    "pressure_low": round(float(output["pressure_low"][0]), 1),
                    "pressure_high": round(float(output["pressure_high"][0]), 1),
                    "method": "Monte Carlo Dropout (model-derived, not scientifically validated)",
                },
            }
        else:
            with torch.no_grad():
                output = self.intensity_model(sat_tensor, env_tensor)
            result = {
                "wind_kph": round(float(output["wind_kph"][0]), 1),
                "pressure_hpa": round(float(output["pressure_hpa"][0]), 1),
            }

        elapsed = time.time() - start
        result["imd_category"] = wind_to_imd_category(result["wind_kph"] / 1.852)  # convert to knots
        result["inference_time_ms"] = round(elapsed * 1000, 1)
        result.update(self._model_metadata("intensity"))
        return result

    def predict_track(
        self,
        track_history: np.ndarray,
        satellite_image: np.ndarray,
        env_features: np.ndarray,
        current_lat: float = 15.0,
        current_lon: float = 85.0,
        with_uncertainty: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict future cyclone track at multiple horizons.
        
        Args:
            track_history: [T, features] historical track data
            satellite_image: current satellite observation
            env_features: current environmental conditions
            current_lat/lon: current cyclone position
        """
        start = time.time()
        sat_tensor = self._prepare_satellite(satellite_image)
        env_tensor = self._prepare_env(env_features)
        track_tensor = self._prepare_track(track_history)

        horizons = self.config.get("track", {}).get("forecast_horizons", [6, 12, 24, 48])

        if with_uncertainty:
            output = self.track_model.predict_with_uncertainty(
                track_tensor, sat_tensor, env_tensor,
                n_samples=self.config.get("track", {}).get("mc_samples", 20),
            )
            positions = output["predicted_positions"][0].cpu().numpy()
            stds = output["position_std"][0].cpu().numpy()
        else:
            with torch.no_grad():
                output = self.track_model(track_tensor, sat_tensor, env_tensor)
            positions = output["predicted_positions"][0].cpu().numpy()
            stds = np.zeros_like(positions)

        forecast_points = []
        for i, h in enumerate(horizons):
            lat = current_lat + float(positions[i, 0])
            lon = current_lon + float(positions[i, 1])
            forecast_points.append({
                "forecast_hour": h,
                "latitude": round(lat, 2),
                "longitude": round(lon, 2),
                "lat_uncertainty": round(float(stds[i, 0]), 3),
                "lon_uncertainty": round(float(stds[i, 1]), 3),
                "uncertainty_radius_km": round(float(np.sqrt(stds[i, 0]**2 + stds[i, 1]**2) * 111), 1),
            })

        elapsed = time.time() - start

        return {
            "forecast_points": forecast_points,
            "horizons_hours": horizons,
            "uncertainty_method": "Monte Carlo Dropout (model-derived estimate)" if with_uncertainty else "none",
            "inference_time_ms": round(elapsed * 1000, 1),
            **self._model_metadata("track"),
        }

    def predict_track_from_history(
        self,
        history: list,
        with_uncertainty: bool = True,
    ) -> Dict[str, Any] | None:
        """Forecast a track from position history alone, using the trained model.

        This is the track forecast the platform should serve. Unlike
        :meth:`predict_track` it uses ``TrackOnlyForecaster`` -- the model that has
        trained weights and a season-held-out evaluation -- and it needs no satellite
        imagery or reanalysis fields, so nothing has to be fabricated to call it.

        Args:
            history: chronological list of dicts with ``latitude``, ``longitude``,
                ``timestamp``, and optionally ``wind_kph`` / ``pressure_hpa``.
            with_uncertainty: also report MC-dropout spread alongside the verified
                error radius.

        Returns:
            The forecast dict, or **None** when no trained checkpoint is loaded or the
            history is unusable (too short, or not on the 6-hourly grid the model was
            trained on). None means "fall back to the analytic baseline" -- the runtime
            declines rather than extrapolating off its training distribution, because a
            plausible-looking forecast from out-of-distribution input is worse than no
            forecast at all.
        """
        if self.track_runtime is None:
            return None
        forecast = self.track_runtime.forecast(
            history,
            with_uncertainty=with_uncertainty,
            mc_samples=self.config.get("track", {}).get("mc_samples", 20),
        )
        if forecast is not None:
            forecast["model_version"] = self.model_version.get("version", "0.1.0")
            forecast["device"] = str(self.device)
            forecast["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return forecast

    def cliper_forecast_from_history(self, history: list) -> Dict[str, Any] | None:
        """Fitted-CLIPER track forecast from position history, or None if unavailable.

        This is the baseline the network must beat, and the one it *was* scored against.
        Serving it alongside every forecast means the platform's accuracy claim can be
        checked against a real reference in the same response, rather than asking anyone
        to trust a number on a separate page.
        """
        if self.track_runtime is None:
            return None
        return self.track_runtime.cliper_forecast(history)

    def track_skill_summary(self) -> Dict[str, Any] | None:
        """Verified track accuracy and the honest verdict against CLIPER, or None."""
        if self.track_runtime is None:
            return None
        return self.track_runtime.skill_summary()

    def explain(self, satellite_image: np.ndarray, task: str = "detection") -> Dict[str, Any]:
        """Generate Grad-CAM explanation for satellite image analysis.

        The `with` matters: self.detector and self.classifier are long-lived and shared with
        every other inference route. Without it each call left a hook pair attached to them
        permanently -- measured at 64 KiB retained per request, unbounded, with the stale
        hooks still firing during unrelated predictions.

        May return {"available": False, "reason": ...} when there is no attribution to show.
        """
        tensor = self._prepare_satellite(satellite_image)

        if task == "detection":
            model = self.detector
        elif task == "classification":
            model = self.classifier
        else:
            model = self.detector

        with GradCAM(model) as gradcam:
            result = gradcam.generate_overlay(tensor, task=task)
        result["task"] = task
        result["explanation_method"] = "Grad-CAM"
        result.update(self._model_metadata(task.split("_")[0] if "_" in task else "detector"))
        return result

    RISK_WEIGHTS = {
        "wind": 0.35,
        "coastal": 0.25,
        "rapid_intensification": 0.20,
        "rainfall": 0.10,
        "track_uncertainty": 0.10,
    }
    """Component weights, renormalised over whatever is actually available.

    The weights themselves are a judgement call, not a fitted result, and are named here so
    they can be argued with instead of being buried in an expression.
    """

    def assess_risk(
        self,
        wind_kph: float,
        pressure_hpa: float,
        dist_to_coast_km: float = 500.0,
        ri_probability: Optional[float] = None,
        track_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Rules-based risk score. Missing inputs drop out; they are never defaulted.

        This is a rules-based engine, not a learned model: the thresholds below encode
        operational convention, and no part of this score was fitted to outcome data.

        Why the signature changed
        -------------------------
        ``ri_probability`` and ``track_confidence`` used to default to ``0.0`` and ``0.5``,
        and both call sites passed literals -- ``0.1`` and ``0.5``. Together those two
        carried 30% of ``overall_risk``. A constant contributing a third of an alert level
        is not a risk model; it is a number that looks like one, and it moved no storm's
        score relative to any other's.

        They are now ``None`` by default. A component that is ``None`` is dropped and the
        remaining weights are renormalised, so the score is an average over the evidence
        that exists rather than over evidence plus filler. The response says which
        components were used, and reports the missing ones as ``null``. When every
        component is supplied the arithmetic is identical to before, so this does not
        silently re-scale existing alert levels.

        Pass ``ri_probability`` only from a real intensity forecast. Observed intensity
        change belongs in ``ml.inference.track_risk.intensity_trend``, which reports it as
        the observation it is -- a storm that has already gained 35 kt is a fact, not a
        probability about the future.

        Known limitations, stated rather than hidden
        -------------------------------------------
        * ``rainfall_risk`` is ``wind_kph / 250`` -- a monotone function of wind, so it
          double-counts intensity and is not a rainfall forecast. It is kept at its
          existing weight to avoid re-scaling live alert levels, and flagged in
          ``component_provenance`` as wind-derived.
        * ``pressure_hpa`` is accepted for interface stability but does not enter the
          score. Pressure deficit would be a genuinely independent intensity measure;
          folding it in without validating the combination would just move numbers around.
        """
        # Wind risk (0-1)
        if wind_kph >= 220:
            wind_risk = 1.0
        elif wind_kph >= 170:
            wind_risk = 0.9
        elif wind_kph >= 120:
            wind_risk = 0.7
        elif wind_kph >= 80:
            wind_risk = 0.5
        elif wind_kph >= 50:
            wind_risk = 0.3
        else:
            wind_risk = 0.1

        # Coastal risk
        if dist_to_coast_km < 50:
            coastal_risk = 1.0
        elif dist_to_coast_km < 100:
            coastal_risk = 0.8
        elif dist_to_coast_km < 200:
            coastal_risk = 0.6
        elif dist_to_coast_km < 500:
            coastal_risk = 0.3
        else:
            coastal_risk = 0.1

        # Rainfall proxy. See the docstring: derived from wind, not forecast.
        rainfall_risk = min(1.0, wind_kph / 250.0)

        ri_risk = None if ri_probability is None else min(1.0, max(0.0, ri_probability))
        track_uncertainty = (
            None if track_confidence is None
            else 1.0 - min(1.0, max(0.0, track_confidence))
        )

        components: Dict[str, Optional[float]] = {
            "wind": wind_risk,
            "coastal": coastal_risk,
            "rapid_intensification": ri_risk,
            "rainfall": rainfall_risk,
            "track_uncertainty": track_uncertainty,
        }
        available = {k: v for k, v in components.items() if v is not None}
        missing = sorted(k for k, v in components.items() if v is None)

        # Renormalise over the components present. Every component is in [0, 1] and the
        # weights sum to 1 after renormalisation, so overall_risk stays in [0, 1].
        total_weight = sum(self.RISK_WEIGHTS[k] for k in available)
        overall_risk = sum(
            self.RISK_WEIGHTS[k] * v for k, v in available.items()
        ) / total_weight

        # Alert level
        if overall_risk >= 0.8:
            alert_level = "EXTREME RISK"
        elif overall_risk >= 0.6:
            alert_level = "HIGH RISK"
        elif overall_risk >= 0.4:
            alert_level = "ADVISORY"
        elif overall_risk >= 0.2:
            alert_level = "WATCH"
        else:
            alert_level = "NORMAL"

        methodology = "Rules-based risk engine with configurable thresholds"
        if missing:
            methodology += (
                f"; {len(missing)} of {len(components)} components unavailable "
                f"({', '.join(missing)}) and excluded by renormalising the remaining "
                f"weights rather than substituting a default"
            )

        return {
            "overall_risk": round(overall_risk, 3),
            "wind_risk": round(wind_risk, 3),
            "coastal_risk": round(coastal_risk, 3),
            "rainfall_risk": round(rainfall_risk, 3),
            "rapid_intensification_risk": None if ri_risk is None else round(ri_risk, 3),
            "track_confidence": (None if track_confidence is None
                                 else round(min(1.0, max(0.0, track_confidence)), 3)),
            "alert_level": alert_level,
            "components_used": sorted(available),
            "components_unavailable": missing,
            "weights_applied": {k: round(self.RISK_WEIGHTS[k] / total_weight, 3)
                                for k in sorted(available)},
            "component_provenance": {
                "wind": "observed or estimated intensity, threshold ladder",
                "coastal": "measured distance to coast (ml.geo.coastline, MAE 8.7 km)",
                "rainfall": "proxy: wind_kph / 250, monotone in wind, not a rainfall forecast",
                "rapid_intensification": "requires a trained intensity-forecast head",
                "track_uncertainty": "requires a per-storm track confidence estimate",
            },
            "methodology": methodology,
        }


    def full_analysis(
        self,
        satellite_image: np.ndarray,
        env_features: np.ndarray,
        track_history: np.ndarray,
        current_lat: float = 15.0,
        current_lon: float = 85.0,
        dist_to_coast_km: float = 500.0,
    ) -> Dict[str, Any]:
        """
        Run complete cyclone analysis pipeline.
        """
        start = time.time()

        # 1. Detection
        detection = self.detect(satellite_image, current_lat, current_lon)

        # 2. Classification
        classification = self.classify(satellite_image, env_features)

        # 3. Intensity
        intensity = self.estimate_intensity(satellite_image, env_features)

        # 4. Track prediction
        track = self.predict_track(
            track_history, satellite_image, env_features,
            current_lat, current_lon,
        )

        # 5. Rapid intensification: not forecast, and no longer faked.
        #
        # This used to be `min(1.0, (wind_kph - 100) / 150)` -- a rescaling of current wind
        # speed, presented as the probability of future intensification. It could not have
        # been right: it is a function of how strong the storm is now, and says nothing
        # about whether it is strengthening. A steady 160 km/h system and one that gained
        # 40 kt in the last day scored identically.
        #
        # There is no trained intensity-forecast head, so the honest value is None. Observed
        # 24-hour intensity change -- which is real, and is the RI signal a forecaster would
        # actually look at -- is computed from best-track history in
        # ml.inference.track_risk.intensity_trend and surfaced by the risk endpoint.

        # 6. Risk assessment. ri_probability and track_confidence are omitted rather than
        # defaulted; assess_risk renormalises over the components it does have.
        risk = self.assess_risk(
            intensity["wind_kph"],
            intensity["pressure_hpa"],
            dist_to_coast_km,
        )

        # 7. Explainability
        explanation = self.explain(satellite_image, "detection")

        elapsed = time.time() - start

        return {
            "detection": detection,
            "classification": classification,
            "intensity": intensity,
            "track_forecast": track,
            "rapid_intensification": {
                "probability": None,
                "available": False,
                "threshold_knots_24h": RI_THRESHOLD_KT_24H,
                "reason": (
                    "No trained intensity-forecast head. Reporting null rather than a "
                    "number derived from current wind speed, which would describe how "
                    "strong this storm is, not whether it is intensifying."
                ),
                "observed_alternative": (
                    "Observed 24 h intensity change is available from the risk endpoint "
                    "(intensity_trend), measured from best-track wind history."
                ),
                **self._model_metadata("intensity"),
            },
            "risk": risk,
            "explanation": explanation,
            "total_inference_time_ms": round(elapsed * 1000, 1),
            "disclaimer": (
                "AI-generated prediction for research and decision-support purposes. "
                "This system is not an official warning system and must not replace "
                "official advisories issued by the India Meteorological Department "
                "or other authorized agencies."
            ),
        }

    def _prepare_satellite(self, image: np.ndarray) -> torch.Tensor:
        """Prepare satellite image tensor [1, C, H, W]."""
        if image is None:
            # Generate dummy input
            size = self.config.get("satellite_cnn", {}).get("image_size", 64) or 64
            channels = self.config.get("satellite_cnn", {}).get("input_channels", 4) or 4
            image = np.random.randn(channels, size, size).astype(np.float32)

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[np.newaxis, :, :]  # [1, H, W]
            elif image.ndim == 3 and image.shape[2] in [1, 2, 3, 4]:
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            tensor = torch.from_numpy(image.astype(np.float32))
        else:
            tensor = image

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Resize if needed
        target_size = self.config.get("satellite_cnn", {}).get("image_size", 64) or 64
        if tensor.shape[2] != target_size or tensor.shape[3] != target_size:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_size, target_size), mode='bilinear', align_corners=False
            )

        # Match expected channels
        expected_ch = self.config.get("satellite_cnn", {}).get("input_channels", 4) or 4
        if tensor.shape[1] < expected_ch:
            # Pad missing channels with zeros
            pad = torch.zeros(tensor.shape[0], expected_ch - tensor.shape[1], tensor.shape[2], tensor.shape[3])
            tensor = torch.cat([tensor, pad], dim=1)
        elif tensor.shape[1] > expected_ch:
            tensor = tensor[:, :expected_ch]

        return tensor.to(self.device)

    def _prepare_env(self, features: np.ndarray) -> torch.Tensor:
        """Prepare a standardised environmental features tensor [1, D].

        Features arrive in raw physical units whose magnitudes span roughly seven
        orders of magnitude (vorticity ~1e-3 against geopotential ~5.5e3). They are
        standardised here, at the single point every model's env input passes through,
        so no caller can accidentally skip it. See ml/utils/env_normalization.py for
        why this matters and where the statistics come from.
        """
        expected = self.config.get("environmental_mlp", {}).get("input_dim", 20) or 20
        means = self.env_normalizer.means

        if features is None:
            # Climatological mean, not zeros. Zeros are only "neutral" after
            # standardisation -- as raw physical units they would mean 0 degC SST and
            # 0 hPa MSLP, which is not a missing value but an impossible atmosphere,
            # and would clip to -8 sigma on almost every feature.
            arr = means.copy()
        elif isinstance(features, torch.Tensor):
            arr = features.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(features, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]

        # Pad/trim in raw units, filling any missing feature with its climatological
        # mean so it standardises to zero.
        n = arr.shape[1]
        if n < expected:
            fill = np.broadcast_to(means[n:expected], (arr.shape[0], expected - n))
            arr = np.concatenate([arr, fill], axis=1)
        elif n > expected:
            arr = arr[:, :expected]

        standardised = self.env_normalizer.transform(arr)
        return torch.from_numpy(standardised).to(self.device)

    def _prepare_track(self, history: np.ndarray) -> torch.Tensor:
        """Prepare track history tensor [1, T, D]."""
        if history is None:
            seq_len = self.config.get("temporal_model", {}).get("sequence_length", 8) or 8
            dim = self.config.get("temporal_model", {}).get("input_dim", 10) or 10
            history = np.zeros((seq_len, dim), dtype=np.float32)

        if isinstance(history, np.ndarray):
            tensor = torch.from_numpy(history.astype(np.float32))
        else:
            tensor = history

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Adjust sequence length
        seq_len = self.config.get("temporal_model", {}).get("sequence_length", 8) or 8
        if tensor.shape[1] < seq_len:
            pad = torch.zeros(tensor.shape[0], seq_len - tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([pad, tensor], dim=1)
        elif tensor.shape[1] > seq_len:
            tensor = tensor[:, -seq_len:]

        # Adjust feature dim
        expected_dim = self.config.get("temporal_model", {}).get("input_dim", 10) or 10
        if tensor.shape[2] < expected_dim:
            pad = torch.zeros(tensor.shape[0], tensor.shape[1], expected_dim - tensor.shape[2])
            tensor = torch.cat([tensor, pad], dim=2)
        elif tensor.shape[2] > expected_dim:
            tensor = tensor[:, :, :expected_dim]

        return tensor.to(self.device)

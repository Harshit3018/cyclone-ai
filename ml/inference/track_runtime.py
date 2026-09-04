"""
Serving wrapper for the trained track forecaster.

This is the bridge between the API's track history (a list of dicts out of the database)
and ``TrackOnlyForecaster``, and it exists as its own module for one reason: the features
the model is fed at serving time must be built by *the same code* that built them at
training time. A serving-side reimplementation of the feature vector is the single most
common way a correctly-trained track model produces confident nonsense in production, and
it fails silently -- the tensor has the right shape, the forecast looks like a track, and
the error is only visible if someone verifies against a best track months later.

So ``_step_features`` is imported from ml/datasets/track_dataset.py rather than copied, and
the checkpoint's recorded ``feature_names`` is compared against the current
``TRACK_FEATURE_NAMES`` on load. A mismatch refuses the checkpoint instead of serving it.

Two further guards
------------------
**Contiguity.** The model was trained exclusively on strictly 6-hourly contiguous history
windows. If the supplied history is not on that grid the runtime declines to forecast
(returns None) and the caller falls back to the analytic baseline. Silently feeding a 3-hourly
or gap-ridden window would put the input off the training distribution while still
producing a plausible-looking answer.

**Horizons.** Only the lead times the checkpoint was trained and verified for are reported.
The API's baselines run to +72 h; this model was fitted to +48 h and is not extrapolated
past it, because a 72 h number carrying a 48 h model's authority is worse than no number.

Uncertainty
-----------
The forecast radius is the model's own **measured** held-out error at that lead time, read
from the evaluation table embedded in the checkpoint -- not a hand-chosen linear function
of lead time (which is what ml/baselines/track.py still uses, and which its own docstring
flags as a placeholder). Two radii are available:

* ``mean_km``  -- the average error; roughly a 50-60% containment radius.
* ``p90_km``   -- the 90th percentile; the radius that actually contains 9 cases in 10.

``p90`` is used for the displayed cone, because a cone drawn at the mean error excludes
40% of outcomes while looking authoritative. MC-dropout spread is reported alongside it as
a separate, clearly-labelled quantity: it measures the model's internal disagreement, which
is not the same thing as its error and is empirically much smaller.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..baselines.cliper import CliperModel
from ..datasets.ibtracs import TrackPoint
from ..datasets.track_dataset import (
    DISPLACEMENT_SCALE_KM,
    SIX_HOURLY_STEP_H,
    TRACK_FEATURE_NAMES,
    displacement_km,
    offset_from_km,
    step_features,
)
from ..models.track_only import TrackOnlyForecaster

logger = logging.getLogger(__name__)

KT_PER_KPH = 1.0 / 1.852

# Tolerance on the 6-hourly grid, in hours. Best-track timestamps are on exact synoptic
# hours, so this only absorbs float round-off in timestamp parsing, not real irregularity.
GRID_TOLERANCE_H = 0.05


def _parse_time(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_track_points(history: list[dict]) -> list[TrackPoint]:
    """Convert API/database history dicts into the dataclass the feature builder expects.

    Wind arrives in km/h from the database and is converted to knots, because the trained
    feature is ``wind_kt / 140``. Feeding km/h here would inflate that feature by 1.85 and
    is exactly the kind of unit slip this module exists to prevent.
    """
    pts: list[TrackPoint] = []
    for row in history:
        t = _parse_time(row.get("timestamp"))
        lat, lon = row.get("latitude"), row.get("longitude")
        if t is None or lat is None or lon is None:
            continue
        wind_kph = row.get("wind_kph")
        pts.append(TrackPoint(
            time=t,
            lat=float(lat),
            lon=float(lon),
            wind_kt=(float(wind_kph) * KT_PER_KPH) if wind_kph is not None else None,
            pres_hpa=(float(row["pressure_hpa"])
                      if row.get("pressure_hpa") is not None else None),
        ))
    pts.sort(key=lambda p: p.time)
    return pts


class TrackOnlyRuntime:
    """A loaded track checkpoint, ready to forecast from a position history."""

    def __init__(self, model: TrackOnlyForecaster, meta: dict):
        self.model = model
        self.meta = meta
        self.horizons_h: tuple[int, ...] = tuple(meta["horizons_h"])
        self.history_steps: int = int(meta["history_steps"])
        self.displacement_scale_km: float = float(
            meta.get("displacement_scale_km", DISPLACEMENT_SCALE_KM))
        # Verified error per horizon, from the split the checkpoint records as untouched.
        self.error_km: dict[int, dict[str, float]] = meta.get("error_km", {})
        self.error_source: str = meta.get("error_source", "unavailable")
        # Verified error of the fitted CLIPER on the same cases, so the baseline shown
        # beside the network carries an equally real cone.
        self.cliper_error_km: dict[int, dict[str, float]] = meta.get("cliper_error_km", {})
        self.cliper: CliperModel | None = None

    # --- loading -----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, device=None) -> "TrackOnlyRuntime | None":
        """Load a checkpoint, or return None with a logged reason.

        Returning None rather than raising is deliberate: a missing or incompatible track
        checkpoint must degrade the platform to its analytic baseline, not take the API
        down. Every rejection path logs why, so a silently-baseline deployment is
        diagnosable from the logs.
        """
        path = Path(path)
        if not path.exists():
            logger.info(f"no track-only checkpoint at {path}; analytic baseline will serve")
            return None
        try:
            ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"could not read track checkpoint {path}: {e}")
            return None

        saved_features = tuple(ckpt.get("feature_names", ()))
        if saved_features != tuple(TRACK_FEATURE_NAMES):
            # The whole point of storing the layout with the weights. A checkpoint trained
            # on a different feature order would still load and still emit tracks.
            logger.error(
                "track checkpoint feature layout does not match the current code and is "
                f"refused. checkpoint={saved_features} current={tuple(TRACK_FEATURE_NAMES)}"
            )
            return None

        arch = ckpt.get("architecture")
        if not arch:
            logger.error("track checkpoint carries no architecture block; refused")
            return None
        try:
            model = TrackOnlyForecaster.from_architecture(arch)
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception as e:
            logger.error(f"track checkpoint weights do not fit its architecture: {e}")
            return None
        model.eval()
        if device is not None:
            model.to(device)

        error_km, error_source = cls._extract_errors(ckpt)
        cliper_error_km, _ = cls._extract_errors(ckpt, match="CLIPER")
        meta = {
            "horizons_h": ckpt.get("horizons_h", list(model.horizons_h)),
            "history_steps": ckpt.get("history_steps", 4),
            "displacement_scale_km": ckpt.get("displacement_scale_km",
                                              DISPLACEMENT_SCALE_KM),
            "train_seasons": ckpt.get("train_seasons", []),
            "val_seasons": ckpt.get("val_seasons", []),
            "test_seasons": ckpt.get("test_seasons", []),
            "best_epoch": ckpt.get("best_epoch"),
            "trained_at": ckpt.get("trained_at"),
            "dataset": ckpt.get("dataset"),
            "n_train_samples": ckpt.get("n_train_samples"),
            "n_parameters": model.n_parameters(),
            "error_km": error_km,
            "error_source": error_source,
            "cliper_error_km": cliper_error_km,
            "significance_vs_cliper": (ckpt.get("evaluation", {})
                                       .get("test_significance_vs_cliper", {})),
        }
        logger.info(
            f"loaded track model: {model.n_parameters()} params, horizons "
            f"{meta['horizons_h']}, error source {error_source}"
        )
        return cls(model, meta)

    @staticmethod
    def _extract_errors(ckpt: dict, match: str = "TrackNet"
                        ) -> tuple[dict[int, dict[str, float]], str]:
        """Pull one method's per-horizon verified error out of the evaluation table.

        Prefers the test split, which the trainer scores exactly once and never uses for
        any decision, over validation, which selected the checkpoint and is therefore
        mildly optimistic.
        """
        ev = ckpt.get("evaluation") or {}
        for split in ("test", "val"):
            block = ev.get(split) or {}
            for name, payload in block.items():
                if match not in name:
                    continue
                out = {}
                for s in payload.get("scores", []):
                    out[int(s["horizon_h"])] = {
                        "mean_km": float(s["mean_km"]),
                        "p90_km": float(s["p90_km"]),
                        "n": int(s["n"]),
                    }
                if out:
                    seasons = ckpt.get(f"{split}_seasons", [])
                    label = (f"measured held-out error on {split} seasons "
                             f"{min(seasons)}-{max(seasons)}" if seasons
                             else f"measured held-out error ({split})")
                    return out, label
        return {}, "unavailable"

    # --- forecasting -------------------------------------------------------------

    def _build_history(self, pts: list[TrackPoint]) -> np.ndarray | None:
        """Assemble the [1, T, F] input tensor, or None if the history is unusable."""
        if len(pts) < self.history_steps:
            logger.debug(f"track model needs {self.history_steps} points, got {len(pts)}")
            return None
        window = pts[-self.history_steps:]
        for a, b in zip(window, window[1:]):
            gap = (b.time - a.time).total_seconds() / 3600.0
            if abs(gap - SIX_HOURLY_STEP_H) > GRID_TOLERANCE_H:
                logger.info(
                    f"track history is not on the 6-hourly grid the model was trained on "
                    f"({gap:.2f} h step); declining to forecast so the caller falls back"
                )
                return None

        # The step before the window supplies the first step's motion, matching
        # build_samples. Without it the first row would carry zero motion here but real
        # motion in training -- a subtle train/serve skew.
        before = pts[-self.history_steps - 1] if len(pts) > self.history_steps else None
        feats = []
        for k, p in enumerate(window):
            prev = window[k - 1] if k > 0 else before
            feats.append(step_features(p, prev))
        return np.asarray([feats], dtype=np.float32)

    def forecast(self, history: list[dict], with_uncertainty: bool = True,
                 mc_samples: int = 20) -> dict | None:
        """Forecast from a position history, in the API's response shape.

        Returns None when the model cannot honestly forecast this history, so the caller
        can fall back to the analytic baseline. It never guesses.
        """
        pts = _to_track_points(history)
        arr = self._build_history(pts)
        if arr is None:
            return None

        origin = pts[-1]
        tensor = torch.from_numpy(arr)

        with torch.no_grad():
            out = self.model(tensor)
        disp = out["displacement_scaled"][0].numpy() * self.displacement_scale_km
        prior = out["persistence_prior"][0].numpy() * self.displacement_scale_km

        spread_km = None
        if with_uncertainty:
            unc = self.model.predict_with_uncertainty(tensor, n_samples=mc_samples)
            spread_km = unc["displacement_std"][0].numpy() * self.displacement_scale_km

        points = []
        for i, h in enumerate(self.horizons_h):
            lat, lon = offset_from_km(origin.lat, origin.lon,
                                      float(disp[i, 0]), float(disp[i, 1]))
            err = self.error_km.get(h, {})
            # Heading and speed of the forecast leg, for the bulletin-style readout the
            # baselines already provide, so the UI can render every method identically.
            east, north = float(disp[i, 0]), float(disp[i, 1])
            leg = math.hypot(east, north)
            point = {
                "forecast_hour": h,
                "latitude": round(lat, 2),
                "longitude": round(lon, 2),
                "heading_deg": round(math.degrees(math.atan2(east, north)) % 360.0, 1),
                "translation_speed_kph": round(leg / h, 1) if h else 0.0,
                "displacement_km": round(leg, 1),
                # Persistence contribution, so the learned correction is inspectable in
                # the response rather than hidden inside one number.
                "persistence_km": round(
                    math.hypot(float(prior[i, 0]), float(prior[i, 1])), 1),
            }
            if err:
                point["uncertainty_radius_km"] = round(err["p90_km"], 1)
                point["mean_error_km"] = round(err["mean_km"], 1)
                point["verified_cases"] = err["n"]
            if spread_km is not None:
                point["mc_dropout_spread_km"] = round(
                    float(math.hypot(spread_km[i, 0], spread_km[i, 1])), 1)
            points.append(point)

        return {
            "forecast_points": points,
            "horizons_hours": list(self.horizons_h),
            "model_name": "CycloneAI TrackNet (track-only LSTM)",
            "model_status": "trained",
            "method": (
                "LSTM over 6-hourly motion history predicting a correction to the "
                "persistence forecast. Trained on IBTrACS North Indian Ocean best track "
                f"seasons {self._season_span('train_seasons')}, verified on held-out "
                f"seasons {self._season_span('test_seasons')}."
            ),
            "uncertainty_method": (
                f"90th-percentile great-circle error per lead time, {self.error_source} "
                f"({self.error_km.get(self.horizons_h[0], {}).get('n', 0)} verifying cases "
                f"at +{self.horizons_h[0]}h). This is an observed error statistic, not a "
                f"model-derived confidence interval."
                if self.error_km else
                "unavailable -- checkpoint carries no verified error statistics"
            ),
            "training_provenance": {
                "dataset": self.meta.get("dataset"),
                "train_seasons": self._season_span("train_seasons"),
                "test_seasons": self._season_span("test_seasons"),
                "n_train_samples": self.meta.get("n_train_samples"),
                "n_parameters": self.meta.get("n_parameters"),
                "trained_at": self.meta.get("trained_at"),
            },
            "max_trained_horizon_h": max(self.horizons_h),
        }

    def _season_span(self, key: str) -> str:
        seasons = self.meta.get(key) or []
        if not seasons:
            return "unrecorded"
        return f"{min(seasons)}-{max(seasons)}" if len(seasons) > 1 else str(seasons[0])

    def skill_summary(self) -> dict:
        """Verified accuracy and the honest verdict, for the API's /models endpoint."""
        sig = self.meta.get("significance_vs_cliper") or {}
        n_sig = sum(1 for d in sig.values() if d.get("significant"))
        return {
            "error_km": {str(h): v for h, v in self.error_km.items()},
            "error_source": self.error_source,
            "significance_vs_cliper": sig,
            "verdict": (
                f"Statistically distinguishable from a fitted CLIPER at {n_sig} of "
                f"{len(sig)} lead times on the untouched test split."
                if sig else "No significance testing recorded in the checkpoint."
            ),
        }

    # --- fitted CLIPER, served through the same feature path ----------------------

    def attach_cliper(self, path: str | Path) -> bool:
        """Load the fitted CLIPER saved beside the checkpoint. Returns success.

        CLIPER is served from the *same* normalised history array the network consumes,
        via ``CliperModel.predict_latlon_from_history``, so the baseline shown in the UI is
        the identical regression that was scored against the network during training. A
        separately-implemented serving CLIPER could drift from the one in the evaluation
        table and quietly invalidate every skill number the platform reports.
        """
        path = Path(path)
        if not path.exists():
            logger.info(f"no fitted CLIPER at {path}; CLIPER baseline unavailable")
            self.cliper = None
            return False
        try:
            self.cliper = CliperModel.load(path)
        except Exception as e:
            logger.warning(f"could not load fitted CLIPER from {path}: {e}")
            self.cliper = None
            return False
        logger.info(f"loaded fitted CLIPER, horizons {self.cliper.horizons_h}")
        return True

    def cliper_forecast(self, history: list[dict],
                        error_km: dict[int, dict] | None = None) -> dict | None:
        """Fitted-CLIPER forecast in the API's response shape, or None."""
        if getattr(self, "cliper", None) is None:
            return None
        pts = _to_track_points(history)
        arr = self._build_history(pts)
        if arr is None:
            return None
        origin = pts[-1]
        latlon = self.cliper.predict_latlon_from_history(
            arr[0], origin.lat, origin.lon)
        if not latlon:
            return None

        error_km = error_km if error_km is not None else self.cliper_error_km
        points = []
        for h in sorted(latlon):
            lat, lon = latlon[h]
            east, north = displacement_km(origin.lat, origin.lon, lat, lon)
            leg = math.hypot(east, north)
            point = {
                "forecast_hour": h,
                "latitude": round(lat, 2),
                "longitude": round(lon, 2),
                "heading_deg": round(math.degrees(math.atan2(east, north)) % 360.0, 1),
                "translation_speed_kph": round(leg / h, 1) if h else 0.0,
                "displacement_km": round(leg, 1),
            }
            err = error_km.get(h)
            if err:
                point["uncertainty_radius_km"] = round(err["p90_km"], 1)
                point["mean_error_km"] = round(err["mean_km"], 1)
                point["verified_cases"] = err["n"]
            points.append(point)

        span = (f"{min(self.cliper.train_seasons)}-{max(self.cliper.train_seasons)}"
                if self.cliper.train_seasons else "unrecorded")
        return {
            "forecast_points": points,
            "horizons_hours": sorted(latlon),
            "model_name": "CLIPER (fitted)",
            "model_status": "fitted_baseline",
            "method": (
                "Multiple linear regression of displacement on climatology and "
                "persistence predictors, fitted by least squares to IBTrACS North Indian "
                f"Ocean best track seasons {span}. This is a real CLIPER in the sense of "
                "Neumann (1972), not a specified climatological prior."
            ),
            "uncertainty_method": (
                "90th-percentile great-circle error per lead time, measured on the same "
                "held-out seasons as the network."
                if error_km else "not measured for this method"
            ),
            "train_seasons": span,
        }

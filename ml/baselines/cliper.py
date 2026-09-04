"""
CLIPER -- CLImatology and PERsistence track forecast, fitted by least squares.

This replaces the specified-prior ``clip_lite`` in ml/baselines/track.py with the real
thing: a multiple linear regression of forecast displacement on climatology and
persistence predictors, fitted to the IBTrACS North Indian Ocean best track. That is what
CLIPER has meant since Neumann (1972), and it is the baseline every operational track
forecast is judged against, so it is worth having a real one rather than a placeholder.

Why bother, when a neural network is the headline
-------------------------------------------------
Because CLIPER is hard to beat and cheap to fit, and a network that loses to it has no
demonstrated skill. Reporting "our model beat persistence" while quietly omitting CLIPER
is the standard way track-forecast results are oversold. Fitting CLIPER properly means the
comparison cannot be gamed: same samples, same held-out seasons, same metric.

Design
------
One independent regression per (horizon, component). The target is displacement in km
east and north of the analysis position -- the same target the network regresses, so the
two are directly comparable. Predictors follow the classic CLIPER set: current position,
current motion, current intensity, seasonality, and a few interaction terms that let the
steering depend on latitude.

Fitted by ``numpy.linalg.lstsq``, which returns the minimum-norm solution when the design
matrix is rank-deficient. With ~2500 samples and 12 predictors that will not happen, but
the condition number is reported so a future predictor addition that does collapse the
design is visible rather than silent.

References
----------
Neumann, C.J. (1972). "An alternate to the HURRAN tropical cyclone forecast system."
    NOAA Tech. Memo. NWS SR-62. The original CLIPER.
Aberson, S.D. (1998). "Five-day tropical cyclone track forecasts in the North Atlantic
    basin." Wea. Forecasting, 13, 1005-1015.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..datasets.track_dataset import (
    DEFAULT_HORIZONS_H,
    LAT_SCALE,
    LON_CENTRE,
    LON_HALF_WIDTH,
    MOTION_SCALE_KMH,
    WIND_SCALE,
    TrackSample,
    offset_from_km,
)

PREDICTOR_NAMES = (
    "intercept",
    "lat",              # deg N
    "lon_centred",      # deg E relative to basin centre
    "u",                # current eastward motion, km/h
    "v",                # current northward motion, km/h
    "wind_kt",
    "sin_doy",
    "cos_doy",
    "lat_x_u",          # steering strength varies with latitude
    "lat_x_v",
    "u_sq",             # mild nonlinearity in persistence
    "v_sq",
)


def predictors_from_history(history: np.ndarray) -> np.ndarray:
    """Recover raw physical predictors from a normalised track-history array.

    Takes the same ``[T, TRACK_FEATURE_DIM]`` array the network is given, so CLIPER cannot
    be accused of seeing information the network does not. The un-normalisation constants
    come from ml/datasets/track_dataset.py rather than being restated here.

    Accepting the array rather than a ``TrackSample`` is what lets CLIPER be *served* from
    live position history through the same feature builder used in training -- see
    ml/inference/track_runtime.py. A serving-side copy of this un-normalisation would be
    free to drift from the fit.
    """
    last = history[-1]
    lat = float(last[0]) * LAT_SCALE
    lon_c = float(last[1]) * LON_HALF_WIDTH          # already centred on LON_CENTRE
    u = float(last[3]) * MOTION_SCALE_KMH
    v = float(last[4]) * MOTION_SCALE_KMH
    wind = float(last[6]) * WIND_SCALE
    sin_doy, cos_doy = float(last[8]), float(last[9])
    return np.array([
        1.0, lat, lon_c, u, v, wind, sin_doy, cos_doy,
        lat * u, lat * v, u * u, v * v,
    ], dtype=np.float64)


def _predictors_from_sample(s: TrackSample) -> np.ndarray:
    """Predictor vector for a training sample."""
    return predictors_from_history(s.history)


@dataclass
class CliperModel:
    """Fitted CLIPER coefficients for every horizon and both components."""
    horizons_h: tuple[int, ...]
    # coefficients[horizon] = [2, n_predictors] -- row 0 east, row 1 north
    coefficients: dict[int, np.ndarray] = field(default_factory=dict)
    n_train: dict[int, int] = field(default_factory=dict)
    condition_number: dict[int, float] = field(default_factory=dict)
    train_seasons: tuple[int, ...] = ()

    def predict_km(self, sample: TrackSample) -> dict[int, tuple[float, float]]:
        """Forecast (east, north) displacement in km for each horizon."""
        return self.predict_km_from_history(sample.history)

    def predict_km_from_history(self, history: np.ndarray
                                ) -> dict[int, tuple[float, float]]:
        """Forecast (east, north) displacement in km from a normalised history array."""
        x = predictors_from_history(history)
        out = {}
        for h in self.horizons_h:
            c = self.coefficients.get(h)
            if c is None:
                continue
            out[h] = (float(c[0] @ x), float(c[1] @ x))
        return out

    def predict_latlon_from_history(self, history: np.ndarray,
                                    origin_lat: float, origin_lon: float
                                    ) -> dict[int, tuple[float, float]]:
        """Forecast absolute (lat, lon) from a normalised history array and an origin."""
        return {
            h: offset_from_km(origin_lat, origin_lon, e, n)
            for h, (e, n) in self.predict_km_from_history(history).items()
        }

    def predict_latlon(self, sample: TrackSample) -> dict[int, tuple[float, float]]:
        """Forecast absolute (lat, lon) for each horizon."""
        return {
            h: offset_from_km(sample.origin_lat, sample.origin_lon, e, n)
            for h, (e, n) in self.predict_km(sample).items()
        }

    def to_dict(self) -> dict:
        return {
            "horizons_h": list(self.horizons_h),
            "predictor_names": list(PREDICTOR_NAMES),
            "coefficients": {str(h): c.tolist() for h, c in self.coefficients.items()},
            "n_train": {str(h): n for h, n in self.n_train.items()},
            "condition_number": {str(h): v for h, v in self.condition_number.items()},
            "train_seasons": list(self.train_seasons),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CliperModel:
        m = cls(horizons_h=tuple(int(h) for h in d["horizons_h"]))
        m.coefficients = {int(h): np.asarray(c, dtype=np.float64)
                          for h, c in d["coefficients"].items()}
        m.n_train = {int(h): int(n) for h, n in d.get("n_train", {}).items()}
        m.condition_number = {int(h): float(v)
                              for h, v in d.get("condition_number", {}).items()}
        m.train_seasons = tuple(d.get("train_seasons", ()))
        return m

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CliperModel:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [f"CLIPER fitted on seasons "
                 f"{min(self.train_seasons)}-{max(self.train_seasons)}"
                 if self.train_seasons else "CLIPER (training seasons unrecorded)"]
        for h in self.horizons_h:
            if h not in self.coefficients:
                continue
            lines.append(f"  +{h:>3}h  n={self.n_train.get(h, 0):>5}  "
                         f"cond={self.condition_number.get(h, float('nan')):>9.1f}")
        return "\n".join(lines)

    def report_coefficients(self, horizon_h: int) -> str:
        """Print one horizon's coefficients against their predictor names.

        Useful as a sanity check: the coefficient on `u` at +6 h should be close to 6
        (six hours of the current eastward speed) if persistence dominates at short lead,
        and should shrink as the horizon grows and climatology takes over. If it does not,
        the fit is wrong in a way no aggregate metric will reveal.
        """
        c = self.coefficients.get(horizon_h)
        if c is None:
            return f"no coefficients for +{horizon_h}h"
        lines = [f"+{horizon_h}h coefficients      east         north",
                 "-" * 44]
        for i, name in enumerate(PREDICTOR_NAMES):
            lines.append(f"  {name:<16} {c[0][i]:>10.4f}   {c[1][i]:>11.4f}")
        return "\n".join(lines)


def fit_cliper(train_samples: list[TrackSample],
               horizons_h: tuple[int, ...] = DEFAULT_HORIZONS_H) -> CliperModel:
    """Fit CLIPER by least squares on the training split only.

    Each horizon is fitted on the subset of samples that verify at that horizon, so the
    +48 h regression uses fewer samples than the +6 h one. That is correct -- and the
    counts are recorded so the difference is visible in the summary.
    """
    if not train_samples:
        raise ValueError("cannot fit CLIPER on an empty training set")

    model = CliperModel(
        horizons_h=tuple(horizons_h),
        train_seasons=tuple(sorted({s.season for s in train_samples})),
    )
    X_all = np.stack([_predictors_from_sample(s) for s in train_samples])

    for hi, h in enumerate(horizons_h):
        keep = np.array([s.target_mask[hi] > 0 for s in train_samples])
        if keep.sum() < len(PREDICTOR_NAMES) * 3:
            # Refuse to fit a regression that has barely more samples than parameters;
            # it would produce coefficients dominated by noise and a flattering
            # in-sample fit. Better to have no CLIPER at that horizon than a fake one.
            continue
        X = X_all[keep]
        Y = np.stack([s.target_km[hi] for s in train_samples])[keep]  # [n, 2] east,north

        coef = np.zeros((2, len(PREDICTOR_NAMES)), dtype=np.float64)
        for comp in (0, 1):
            sol, *_ = np.linalg.lstsq(X, Y[:, comp], rcond=None)
            coef[comp] = sol
        model.coefficients[h] = coef
        model.n_train[h] = int(keep.sum())
        model.condition_number[h] = float(np.linalg.cond(X))

    if not model.coefficients:
        raise ValueError("CLIPER could not be fitted at any horizon (too few samples)")
    return model

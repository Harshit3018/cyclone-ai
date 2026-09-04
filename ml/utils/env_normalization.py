"""
Standardisation of environmental predictor features.

Why this module exists
----------------------
The 20 environmental features are raw physical quantities whose magnitudes span
about seven orders of magnitude -- 850 hPa relative vorticity is O(1e-3) while 500 hPa
geopotential is O(5.5e3). Fed to a linear layer unscaled, the largest-magnitude
features dominate every dot product. Two things then go wrong:

1. At inference the classifier logits spread over ~127 units, so softmax saturates to
   exactly 1.0 / 0.0. Every prediction is reported at 100% confidence regardless of
   input, which makes the confidence figure worthless and the uncertainty estimates
   meaningless.
2. At training time the gradient is dominated by geopotential and MSLP. SST, shear and
   mid-level moisture -- the features that actually govern intensification -- would
   contribute almost nothing. The model would train, report a falling loss, and have
   learned essentially one predictor.

The second point is why this had to be fixed before training rather than after.

On the statistics below
-----------------------
These are *specified climatological priors* for the North Indian Ocean, not values
fitted to a dataset. They only need to be right to within a factor of a few to put
every feature on a comparable scale, which is all standardisation requires.

Once real ERA5 predictors are assembled, replace them with statistics fitted on the
training split only -- never on the full dataset, and never recomputed at inference,
or information leaks from validation into training. `fit_statistics` is provided for
exactly that, and `EnvNormalizer.to_dict` / `from_dict` let the fitted values be
saved alongside the model checkpoint so inference uses the same transform training did.
"""
from __future__ import annotations

import numpy as np

# Feature order must match scripts/generate_sample_data.py and the ERA5 extraction.
ENV_FEATURE_NAMES = (
    "sst_celsius", "wind_shear_200_850_ms", "rh_700_pct",
    "temp_200_K", "temp_500_K", "temp_850_K",
    "u_wind_200_ms", "v_wind_200_ms", "u_wind_850_ms", "v_wind_850_ms",
    "geopotential_500_m2s2", "mslp_hpa", "vorticity_850",
    "divergence_200", "specific_humidity_700", "theta_e_850",
    "vertical_wind_shear_dir", "ocean_heat_content",
    "mid_level_moisture", "upper_level_divergence",
)

# (mean, std) per feature, in the feature's own physical units.
ENV_FEATURE_STATS: dict[str, tuple[float, float]] = {
    "sst_celsius":             (28.0, 2.0),
    "wind_shear_200_850_ms":   (8.0, 5.0),
    "rh_700_pct":              (70.0, 10.0),
    "temp_200_K":              (220.0, 5.0),
    "temp_500_K":              (260.0, 3.0),
    "temp_850_K":              (290.0, 2.0),
    "u_wind_200_ms":           (0.0, 10.0),
    "v_wind_200_ms":           (0.0, 10.0),
    "u_wind_850_ms":           (0.0, 5.0),
    "v_wind_850_ms":           (0.0, 5.0),
    "geopotential_500_m2s2":   (5500.0, 100.0),
    "mslp_hpa":                (1000.0, 10.0),
    "vorticity_850":           (0.0, 1e-3),
    "divergence_200":          (0.0, 1e-4),
    "specific_humidity_700":   (0.01, 3e-3),
    "theta_e_850":             (340.0, 5.0),
    # NOTE: shear direction is a circular variable in degrees. Standardising it
    # linearly is wrong -- 359 deg and 1 deg are adjacent physically but maximally far
    # apart numerically, so the model cannot learn a smooth response to direction.
    # The correct encoding is a (sin, cos) pair, which needs env_dim to go from 20 to
    # 21. No checkpoint has been trained yet, so that change is still free to make;
    # doing it later means retraining. Until then this at least puts the feature on
    # the same scale as its neighbours. std is that of a uniform [0, 360] variable.
    "vertical_wind_shear_dir": (180.0, 103.9),
    "ocean_heat_content":      (60.0, 20.0),
    "mid_level_moisture":      (0.6, 0.17),
    "upper_level_divergence":  (0.0, 1e-4),
}

# Standardised features beyond this many sigma are clipped. Guards against a single
# corrupt or missing-data-sentinel value (ERA5 fill values are often -9999) blowing
# up an entire forward pass.
CLIP_SIGMA = 8.0


def _default_arrays() -> tuple[np.ndarray, np.ndarray]:
    means = np.array([ENV_FEATURE_STATS[n][0] for n in ENV_FEATURE_NAMES], dtype=np.float32)
    stds = np.array([ENV_FEATURE_STATS[n][1] for n in ENV_FEATURE_NAMES], dtype=np.float32)
    return means, stds


class EnvNormalizer:
    """Standardise environmental features to roughly zero mean, unit variance."""

    def __init__(self, means: np.ndarray | None = None, stds: np.ndarray | None = None,
                 fitted: bool = False):
        d_means, d_stds = _default_arrays()
        self.means = d_means if means is None else np.asarray(means, dtype=np.float32)
        self.stds = d_stds if stds is None else np.asarray(stds, dtype=np.float32)
        # A zero or negative std would divide by zero; a constant feature carries no
        # information anyway, so map it to zero by giving it unit scale.
        self.stds = np.where(self.stds > 0, self.stds, 1.0).astype(np.float32)
        self.fitted = fitted

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardise a [D] or [N, D] array of raw physical features."""
        x = np.asarray(features, dtype=np.float32)
        n_expected = self.means.shape[0]
        if x.shape[-1] != n_expected:
            raise ValueError(
                f"expected {n_expected} environmental features, got {x.shape[-1]}"
            )
        # Non-finite values (ERA5 gaps, division artefacts) become the feature mean,
        # i.e. zero after standardisation -- a neutral value rather than a NaN that
        # would propagate through the whole network.
        x = np.where(np.isfinite(x), x, self.means)
        z = (x - self.means) / self.stds
        return np.clip(z, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) * self.stds + self.means).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(ENV_FEATURE_NAMES),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EnvNormalizer":
        return cls(np.array(d["means"], dtype=np.float32),
                   np.array(d["stds"], dtype=np.float32),
                   bool(d.get("fitted", False)))

    def __repr__(self) -> str:
        kind = "fitted" if self.fitted else "climatological prior"
        return f"EnvNormalizer({self.means.shape[0]} features, {kind})"


def fit_statistics(train_features: np.ndarray) -> EnvNormalizer:
    """Fit standardisation statistics on the TRAINING SPLIT ONLY.

    Fitting on all data before splitting leaks validation and test distribution
    information into training and inflates reported skill.
    """
    x = np.asarray(train_features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a [N, D] array, got shape {x.shape}")
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("some features have no finite values in the training split")
    means = np.where(finite, x, 0.0).sum(axis=0) / counts
    var = np.where(finite, (x - means) ** 2, 0.0).sum(axis=0) / counts
    return EnvNormalizer(means.astype(np.float32), np.sqrt(var).astype(np.float32),
                         fitted=True)


# Shared default instance. Replace via EnvNormalizer.from_dict once fitted statistics
# are saved with a checkpoint.
DEFAULT_ENV_NORMALIZER = EnvNormalizer()

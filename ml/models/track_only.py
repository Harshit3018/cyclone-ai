"""
Track-only displacement forecaster -- the model that can actually be trained today.

Why this exists alongside TrackPredictor
----------------------------------------
``TrackPredictor`` in track_model.py is a three-branch multimodal network: track LSTM,
satellite CNN and environmental MLP. It is the right long-term architecture, but it cannot
be trained on the only real dataset this project has. IBTrACS is a best-track archive --
positions, wind and pressure. It contains no imagery and no reanalysis fields, so two of
the three branches would have to be fed fabricated input.

Rather than invent satellite tensors for 1990s storms and call the result a trained
multimodal model, this module trains what the data supports: motion history in,
displacement out. That is an honest model with a real evaluation, and it is directly
comparable to CLIPER and persistence, which take the same inputs.

The design that matters
-----------------------
**Residual around persistence.** The head predicts a *correction* to the persistence
forecast rather than the displacement itself. Persistence already explains most of the
variance at short lead -- the +6 h CLIPER fit recovers a coefficient of 6.36 on current
eastward speed, essentially exactly the 6 h that physics demands -- so asking the network
to rediscover it from 2549 samples wastes most of its capacity. Predicting the residual
means an untrained network (all weights near zero) emits *persistence*, which is a
respectable forecast, instead of a stationary storm. The model degrades to a sensible
baseline rather than to nonsense, which is what the physical plausibility checker kept
catching before.

**Small on purpose.** 2549 training samples with a +48 h target count of 1432 supports a
model with thousands of parameters, not millions. The default configuration is about 30k
parameters. A larger network fits the training seasons better and the held-out seasons
worse; that was measured, not assumed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import MC_DROPOUT_SEED, mc_dropout


class TrackOnlyForecaster(nn.Module):
    """LSTM over track history predicting multi-horizon displacement.

    Args:
        input_dim: per-step feature width; must equal ``TRACK_FEATURE_DIM``.
        hidden_dim: LSTM hidden width.
        num_layers: LSTM depth. Two layers on ~2500 samples is already generous.
        n_horizons: number of forecast lead times.
        dropout: applied between LSTM layers and in the head; also the source of the
            MC-dropout uncertainty estimate at inference.
        residual_persistence: predict a correction to the persistence forecast rather
            than the raw displacement. Strongly recommended -- see the module docstring.
        u_index / v_index / motion_scale / horizons_h: needed to reconstruct the
            persistence forecast inside ``forward`` when ``residual_persistence`` is set.
            Defaults match ml/datasets/track_dataset.py; they are stored on the module so
            a checkpoint fully determines the model's behaviour.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 48,
        num_layers: int = 2,
        n_horizons: int = 4,
        dropout: float = 0.2,
        residual_persistence: bool = True,
        u_index: int = 3,
        v_index: int = 4,
        motion_scale_kmh: float = 30.0,
        horizons_h: tuple[int, ...] = (6, 12, 24, 48),
        displacement_scale_km: float = 500.0,
    ):
        super().__init__()
        if len(horizons_h) != n_horizons:
            raise ValueError(
                f"n_horizons={n_horizons} but horizons_h has {len(horizons_h)} entries; "
                "these must agree or the persistence prior is applied to the wrong leads"
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_horizons = n_horizons
        self.residual_persistence = residual_persistence
        self.u_index = u_index
        self.v_index = v_index
        self.motion_scale_kmh = motion_scale_kmh
        self.horizons_h = tuple(horizons_h)
        self.displacement_scale_km = displacement_scale_km

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_horizons * 2),
        )

        # Persistence lead times in units of DISPLACEMENT_SCALE_KM per (km/h), so that
        # persistence_scaled = u_kmh * hours / displacement_scale_km is a plain multiply.
        self.register_buffer(
            "_lead_factor",
            torch.tensor([h / displacement_scale_km for h in horizons_h],
                         dtype=torch.float32),
            persistent=False,
        )
        self._init_head()

    def _init_head(self) -> None:
        """Start the residual head at zero so the model begins exactly at persistence.

        With a zero final layer the initial forecast *is* the persistence forecast, which
        already scores ~35 km at +6 h. Training then only has to improve on that. A
        randomly initialised head instead starts by adding noise to a good forecast, and
        on a dataset this small it can spend most of its epochs recovering.
        """
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def persistence_scaled(self, history: torch.Tensor) -> torch.Tensor:
        """Persistence displacement for each horizon, in target (scaled) units.

        Returns [B, n_horizons, 2] with the last axis (east, north).
        """
        last = history[:, -1, :]
        u = last[:, self.u_index] * self.motion_scale_kmh   # km/h
        v = last[:, self.v_index] * self.motion_scale_kmh
        # [B, 1] * [n_horizons] -> [B, n_horizons]
        east = u.unsqueeze(1) * self._lead_factor.unsqueeze(0)
        north = v.unsqueeze(1) * self._lead_factor.unsqueeze(0)
        return torch.stack([east, north], dim=-1)

    def forward(self, history: torch.Tensor) -> dict:
        """
        Args:
            history: [B, T, input_dim] normalised track history.

        Returns:
            dict with 'displacement_scaled' [B, n_horizons, 2], plus the persistence
            prior and the learned correction separately so the contribution of each is
            inspectable rather than entangled in one number.
        """
        if history.dim() != 3:
            raise ValueError(f"history must be [B, T, F], got shape {tuple(history.shape)}")
        if history.shape[-1] != self.input_dim:
            raise ValueError(
                f"history has {history.shape[-1]} features but this model was built for "
                f"{self.input_dim}. A feature-layout mismatch produces plausible-looking "
                f"but meaningless forecasts, so it is refused here."
            )

        out, (h_n, _) = self.lstm(history)
        correction = self.head(h_n[-1]).view(-1, self.n_horizons, 2)

        if self.residual_persistence:
            prior = self.persistence_scaled(history)
            total = prior + correction
        else:
            prior = torch.zeros_like(correction)
            total = correction

        return {
            "displacement_scaled": total,
            "persistence_prior": prior,
            "correction": correction,
        }

    def predict_with_uncertainty(self, history: torch.Tensor, n_samples: int = 20,
                                 seed: int | None = MC_DROPOUT_SEED) -> dict:
        """MC-dropout forecast spread.

        Reproducible by default so the same storm yields the same cone on every call;
        pass ``seed=None`` for an independent draw. The spread is a measure of model
        uncertainty only -- it does not include observation error or the uncertainty in
        the input analysis position, so it should not be presented as a total error cone
        without saying so.
        """
        samples = []
        with mc_dropout(self, seed=seed), torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.forward(history)["displacement_scaled"])
        stack = torch.stack(samples)
        return {
            "displacement_scaled": stack.mean(dim=0),
            "displacement_std": stack.std(dim=0),
            "n_samples": n_samples,
        }

    def architecture(self) -> dict:
        """Everything needed to rebuild this model, for storage in a checkpoint."""
        return {
            "class": type(self).__name__,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "n_horizons": self.n_horizons,
            "residual_persistence": self.residual_persistence,
            "u_index": self.u_index,
            "v_index": self.v_index,
            "motion_scale_kmh": self.motion_scale_kmh,
            "horizons_h": list(self.horizons_h),
            "displacement_scale_km": self.displacement_scale_km,
        }

    @classmethod
    def from_architecture(cls, arch: dict, dropout: float = 0.2) -> "TrackOnlyForecaster":
        """Rebuild from :meth:`architecture` output, ignoring unknown extra keys."""
        return cls(
            input_dim=arch["input_dim"],
            hidden_dim=arch["hidden_dim"],
            num_layers=arch["num_layers"],
            n_horizons=arch["n_horizons"],
            dropout=dropout,
            residual_persistence=arch.get("residual_persistence", True),
            u_index=arch.get("u_index", 3),
            v_index=arch.get("v_index", 4),
            motion_scale_kmh=arch.get("motion_scale_kmh", 30.0),
            horizons_h=tuple(arch.get("horizons_h", (6, 12, 24, 48))),
            displacement_scale_km=arch.get("displacement_scale_km", 500.0),
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

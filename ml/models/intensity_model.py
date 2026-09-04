"""
Intensity Estimation Model - Predicts max sustained wind and min central pressure.

Combines satellite imagery features with environmental variables
for cyclone intensity regression with uncertainty estimation.
"""
import torch
import torch.nn as nn
from .cyclone_detector import SatelliteCNNBackbone
from ..utils import MC_DROPOUT_SEED, mc_dropout

# Physical bounds on min central pressure (hPa). The record low in any tropical
# cyclone is 870 hPa (Typhoon Tip, 1979), so anything below that is unphysical.
PRESSURE_MIN_HPA = 870.0
PRESSURE_MAX_HPA = 1020.0


class EnvironmentalMLP(nn.Module):
    """MLP branch for environmental/numerical features."""

    def __init__(self, input_dim: int = 20, hidden_dims: list = None, output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class IntensityModel(nn.Module):
    """
    Cyclone intensity regression model.
    
    Predicts:
    - Maximum sustained wind speed (km/h)
    - Minimum central pressure (hPa)
    
    Supports Monte Carlo dropout for uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        sat_feature_dim: int = 128,
        env_input_dim: int = 20,
        env_feature_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.satellite_branch = SatelliteCNNBackbone(in_channels, sat_feature_dim)
        self.env_branch = EnvironmentalMLP(env_input_dim, output_dim=env_feature_dim, dropout=dropout)

        combined_dim = sat_feature_dim + env_feature_dim

        # Wind speed regression head
        self.wind_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            # Softplus, not ReLU: wind speed is always positive, but a ReLU output
            # unit has zero gradient once it saturates at 0 and can never recover.
            nn.Softplus(),
        )

        # Pressure regression head
        self.pressure_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, satellite: torch.Tensor, env_features: torch.Tensor) -> dict:
        sat_feat = self.satellite_branch(satellite)
        env_feat = self.env_branch(env_features)

        combined = torch.cat([sat_feat, env_feat], dim=-1)

        wind = self.wind_head(combined).squeeze(-1)
        pressure_raw = self.pressure_head(combined).squeeze(-1)

        # Map the raw head output into the physical pressure range with a sigmoid
        # rather than a bare additive offset. A raw "+900.0" offset is unbounded and
        # happily emits values like 827 hPa, below the world-record minimum.
        # This is differentiable everywhere and cannot leave [870, 1020].
        pressure = PRESSURE_MIN_HPA + (PRESSURE_MAX_HPA - PRESSURE_MIN_HPA) * torch.sigmoid(pressure_raw)

        return {
            "wind_kph": wind,
            "pressure_hpa": pressure,
            "features": combined,
        }

    def predict_with_uncertainty(
        self,
        satellite: torch.Tensor,
        env_features: torch.Tensor,
        n_samples: int = 20,
        seed: int | None = MC_DROPOUT_SEED,
    ) -> dict:
        """Monte Carlo dropout uncertainty estimation.

        Reproducible by default: the same input returns the same estimate and the same
        interval every call. Pass ``seed=None`` for a fresh stochastic draw.
        """
        wind_samples = []
        pressure_samples = []

        with mc_dropout(self, seed=seed), torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(satellite, env_features)
                wind_samples.append(out["wind_kph"])
                pressure_samples.append(out["pressure_hpa"])

        wind_stack = torch.stack(wind_samples)
        pressure_stack = torch.stack(pressure_samples)

        return {
            "wind_kph": wind_stack.mean(dim=0),
            "wind_std": wind_stack.std(dim=0),
            "wind_low": wind_stack.quantile(0.1, dim=0),
            "wind_high": wind_stack.quantile(0.9, dim=0),
            "pressure_hpa": pressure_stack.mean(dim=0),
            "pressure_std": pressure_stack.std(dim=0),
            "pressure_low": pressure_stack.quantile(0.1, dim=0),
            "pressure_high": pressure_stack.quantile(0.9, dim=0),
        }

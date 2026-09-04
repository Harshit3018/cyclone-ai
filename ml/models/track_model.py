"""
Track Prediction Model - LSTM-based cyclone trajectory forecasting.

Predicts future cyclone positions at +6h, +12h, +24h, +48h horizons
using historical track data, satellite features, and environmental conditions.
"""
import torch
import torch.nn as nn
from .cyclone_detector import SatelliteCNNBackbone
from .intensity_model import EnvironmentalMLP
from ..utils import MC_DROPOUT_SEED, mc_dropout


class TrackLSTM(nn.Module):
    """LSTM for temporal sequence modeling of cyclone tracks."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim] - sequence of track features
        Returns: [B, hidden_dim] - last hidden state
        """
        output, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from final layer
        return h_n[-1]


class TrackPredictor(nn.Module):
    """
    Multi-horizon cyclone track prediction model.
    
    Combines:
    - Historical track sequence (LSTM)
    - Current satellite imagery (CNN)
    - Environmental conditions (MLP)
    
    Predicts lat/lon at multiple forecast horizons.
    """

    def __init__(
        self,
        track_input_dim: int = 10,
        track_hidden_dim: int = 64,
        track_num_layers: int = 2,
        sat_channels: int = 4,
        sat_feature_dim: int = 128,
        env_input_dim: int = 20,
        env_feature_dim: int = 64,
        forecast_horizons: int = 4,  # 6h, 12h, 24h, 48h
        dropout: float = 0.3,
    ):
        super().__init__()
        self.forecast_horizons = forecast_horizons

        # Track temporal branch
        self.track_branch = TrackLSTM(track_input_dim, track_hidden_dim, track_num_layers, dropout)

        # Satellite branch
        self.satellite_branch = SatelliteCNNBackbone(sat_channels, sat_feature_dim)

        # Environmental branch
        self.env_branch = EnvironmentalMLP(env_input_dim, output_dim=env_feature_dim, dropout=dropout)

        # Fusion
        combined_dim = track_hidden_dim + sat_feature_dim + env_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Trajectory decoder - predicts (lat, lon) for each horizon
        self.trajectory_head = nn.Linear(128, forecast_horizons * 2)

    def forward(
        self,
        track_sequence: torch.Tensor,
        satellite: torch.Tensor,
        env_features: torch.Tensor,
    ) -> dict:
        """
        Args:
            track_sequence: [B, T, track_input_dim] - historical positions + features
            satellite: [B, C, H, W] - current satellite image
            env_features: [B, env_input_dim] - environmental variables
        
        Returns:
            dict with predicted positions per horizon
        """
        track_feat = self.track_branch(track_sequence)
        sat_feat = self.satellite_branch(satellite)
        env_feat = self.env_branch(env_features)

        combined = torch.cat([track_feat, sat_feat, env_feat], dim=-1)
        fused = self.fusion(combined)

        trajectory = self.trajectory_head(fused)
        # Reshape to [B, horizons, 2]
        trajectory = trajectory.view(-1, self.forecast_horizons, 2)

        return {
            "predicted_positions": trajectory,  # [B, H, 2] lat/lon offsets
            "features": fused,
        }

    def predict_with_uncertainty(
        self,
        track_sequence: torch.Tensor,
        satellite: torch.Tensor,
        env_features: torch.Tensor,
        n_samples: int = 20,
        seed: int | None = MC_DROPOUT_SEED,
    ) -> dict:
        """Monte Carlo dropout uncertainty for track prediction.

        Reproducible by default: the same input returns the same forecast and the same
        uncertainty every call. Pass ``seed=None`` for a fresh stochastic draw.
        """
        all_trajectories = []
        with mc_dropout(self, seed=seed), torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(track_sequence, satellite, env_features)
                all_trajectories.append(out["predicted_positions"])

        traj_stack = torch.stack(all_trajectories)  # [S, B, H, 2]

        return {
            "predicted_positions": traj_stack.mean(dim=0),
            "position_std": traj_stack.std(dim=0),
            "all_samples": traj_stack,
        }

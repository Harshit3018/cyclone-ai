"""
Multimodal Cyclone Model - Unified model combining all prediction heads.

This model integrates:
- Satellite CNN branch
- Environmental MLP branch
- Temporal LSTM branch
- Detection, Classification, Intensity, Track, and RI heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cyclone_detector import SatelliteCNNBackbone
from .intensity_model import EnvironmentalMLP
from .track_model import TrackLSTM


class MultimodalCycloneModel(nn.Module):
    """
    Unified multimodal model for all cyclone prediction tasks.
    
    Supports selective head activation for inference efficiency.
    """

    def __init__(
        self,
        sat_channels: int = 4,
        sat_feature_dim: int = 128,
        env_input_dim: int = 20,
        env_feature_dim: int = 64,
        track_input_dim: int = 10,
        track_hidden_dim: int = 64,
        num_classes: int = 9,
        forecast_horizons: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.forecast_horizons = forecast_horizons

        # === Encoder branches ===
        self.satellite_encoder = SatelliteCNNBackbone(sat_channels, sat_feature_dim)
        self.env_encoder = EnvironmentalMLP(env_input_dim, output_dim=env_feature_dim, dropout=dropout)
        self.track_encoder = TrackLSTM(track_input_dim, track_hidden_dim, dropout=dropout)

        # === Fusion ===
        fused_dim = sat_feature_dim + env_feature_dim + track_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # === Prediction heads ===
        # Detection
        self.detection_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Localization
        self.localization_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        # Classification
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Intensity - wind
        self.wind_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # Intensity - pressure
        self.pressure_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Track prediction
        self.track_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizons * 2),
        )

        # Rapid intensification
        self.ri_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        env_features: torch.Tensor,
        track_sequence: torch.Tensor,
        active_heads: list = None,
    ) -> dict:
        """
        Full forward pass through all or selected heads.
        
        Args:
            satellite: [B, C, H, W]
            env_features: [B, env_dim]
            track_sequence: [B, T, track_dim]
            active_heads: list of head names to activate (None = all)
        """
        if active_heads is None:
            active_heads = ["detection", "localization", "classification",
                          "intensity", "track", "ri"]

        # Encode
        sat_feat = self.satellite_encoder(satellite)
        env_feat = self.env_encoder(env_features)
        track_feat = self.track_encoder(track_sequence)

        # Fuse
        combined = torch.cat([sat_feat, env_feat, track_feat], dim=-1)
        fused = self.fusion(combined)

        result = {"features": fused}

        if "detection" in active_heads:
            det_logit = self.detection_head(fused)
            result["detection_logit"] = det_logit.squeeze(-1)
            result["detection_prob"] = torch.sigmoid(det_logit).squeeze(-1)

        if "localization" in active_heads:
            result["center_offset"] = self.localization_head(fused)

        if "classification" in active_heads:
            logits = self.classification_head(fused)
            result["class_logits"] = logits
            result["class_probs"] = F.softmax(logits, dim=-1)
            result["predicted_class"] = torch.argmax(logits, dim=-1)

        if "intensity" in active_heads:
            result["wind_kph"] = self.wind_head(fused).squeeze(-1)
            result["pressure_hpa"] = self.pressure_head(fused).squeeze(-1) + 900.0

        if "track" in active_heads:
            track_pred = self.track_head(fused)
            result["predicted_positions"] = track_pred.view(-1, self.forecast_horizons, 2)

        if "ri" in active_heads:
            ri_logit = self.ri_head(fused)
            result["ri_logit"] = ri_logit.squeeze(-1)
            result["ri_prob"] = torch.sigmoid(ri_logit).squeeze(-1)

        return result

"""
Cyclone Detection Model - CNN-based cyclone detection and center localization.

Architecture:
- Simple CNN backbone for satellite image feature extraction
- Binary classification head (cyclone/no-cyclone)
- Center localization regression head (lat/lon offset)

Input: [B, C, H, W] satellite imagery tensor
Output: detection probability, center coordinates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SatelliteCNNBackbone(nn.Module):
    """
    Lightweight CNN backbone for satellite image feature extraction.
    Designed to work with variable number of input channels.
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.projection(features)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return last conv feature maps for Grad-CAM."""
        return self.features(x)


class CycloneDetector(nn.Module):
    """
    Cyclone detection model with:
    - Binary detection head
    - Center localization head (predicts lat/lon)
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 128):
        super().__init__()
        self.backbone = SatelliteCNNBackbone(in_channels, feature_dim)

        # Detection head - binary classification
        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Center localization head - regression (lat_offset, lon_offset)
        self.localization_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),  # (lat_offset, lon_offset) normalized
            nn.Tanh(),  # bound to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)

        detection_logit = self.detection_head(features)
        detection_prob = torch.sigmoid(detection_logit)

        center_offset = self.localization_head(features)

        return {
            "detection_logit": detection_logit.squeeze(-1),
            "detection_prob": detection_prob.squeeze(-1),
            "center_offset": center_offset,  # [B, 2] normalized offsets
            "features": features,
        }

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_feature_maps(x)

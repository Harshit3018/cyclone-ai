"""
Cyclone Classifier - Multi-class classification into IMD cyclone categories.

Uses satellite CNN features + optional environmental features to classify
cyclone stage according to IMD terminology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cyclone_detector import SatelliteCNNBackbone


class CycloneClassifier(nn.Module):
    """
    Multi-class cyclone stage classifier.
    
    Classes (IMD terminology):
    0: Low Pressure Area
    1: Well Marked Low
    2: Depression
    3: Deep Depression
    4: Cyclonic Storm
    5: Severe Cyclonic Storm
    6: Very Severe Cyclonic Storm
    7: Extremely Severe Cyclonic Storm
    8: Super Cyclonic Storm
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 9,
        feature_dim: int = 128,
        env_dim: int = 0,  # optional environmental features
    ):
        super().__init__()
        self.num_classes = num_classes
        self.env_dim = env_dim
        self.backbone = SatelliteCNNBackbone(in_channels, feature_dim)

        combined_dim = feature_dim + env_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, satellite: torch.Tensor, env_features: torch.Tensor = None) -> dict:
        sat_features = self.backbone(satellite)

        if env_features is None:
            # The head is sized for feature_dim + env_dim, so there is no valid
            # env-free path here. Silently concatenating nothing produced an opaque
            # "mat1 and mat2 shapes cannot be multiplied" deep inside nn.Linear;
            # callers with no environmental data should pass climatological means
            # (CyclonePredictor._prepare_env does this for None).
            raise ValueError(
                f"{type(self).__name__} requires env_features of width "
                f"{self.env_dim}; pass climatological means rather than None."
            )
        combined = torch.cat([sat_features, env_features], dim=-1)

        logits = self.classifier(combined)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "features": sat_features,
        }

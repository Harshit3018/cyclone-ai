"""
ML utility functions for CYCLONE-AI.
"""
import os
import random
import numpy as np
import torch
import yaml
from contextlib import contextmanager
from pathlib import Path

from .env_normalization import (
    DEFAULT_ENV_NORMALIZER,
    ENV_FEATURE_NAMES,
    ENV_FEATURE_STATS,
    EnvNormalizer,
    fit_statistics,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


_DROPOUT_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
)


def enable_mc_dropout(module: torch.nn.Module) -> None:
    """Put `module` in eval mode, then re-enable only its dropout layers.

    Monte Carlo dropout needs stochastic dropout at inference time. Calling
    ``module.train()`` to get it is a trap: it also switches every BatchNorm layer
    into batch-statistics mode, so each forward pass overwrites that layer's running
    mean/var from a single sample. The model then degrades a little on every
    prediction and repeated calls on identical input return different answers.
    """
    module.eval()
    for m in module.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()


# Fixed seed for served MC-dropout inference. Any value works; it only has to be stable.
MC_DROPOUT_SEED = 1234


@contextmanager
def mc_dropout(module: torch.nn.Module, seed: int | None = MC_DROPOUT_SEED):
    """Run a block with dropout active, BatchNorm frozen, and results reproducible.

    Two properties have to hold for MC dropout to be usable behind an API:

    1. Dropout on, BatchNorm off. ``module.train()`` enables both, which corrupts
       BatchNorm running statistics one sample at a time -- see enable_mc_dropout().
    2. The same input must give the same answer twice. MC dropout is stochastic by
       construction, so unseeded it returns a different intensity for the same
       satellite image on every request. That reads as a bug to anyone evaluating the
       system, and makes a reported number impossible to cite or reproduce. Seeding
       inside ``torch.random.fork_rng`` fixes the sample draw without perturbing the
       caller's RNG stream, which matters if this is ever called mid-training.

    Pass ``seed=None`` to draw a genuinely fresh sample, e.g. when deliberately
    accumulating an ensemble across repeated calls.
    """
    was_training = module.training
    enable_mc_dropout(module)
    try:
        if seed is None:
            yield
        else:
            try:
                device = next(module.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            # fork_rng defaults to forking every visible CUDA device; only fork the
            # one the module actually lives on.
            devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                yield
    finally:
        module.train(was_training)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get the project root directory (cyclone-ai/)."""
    return Path(__file__).parent.parent.parent


def get_config_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


def load_model_config() -> dict:
    return load_config(str(get_config_dir() / "model.yaml"))


def load_data_config() -> dict:
    return load_config(str(get_config_dir() / "data.yaml"))


def load_training_config() -> dict:
    return load_config(str(get_config_dir() / "training.yaml"))


# IMD Classification mapping
IMD_CATEGORIES = [
    "Low Pressure Area",
    "Well Marked Low",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]

# Lower-bound wind thresholds (knots, 3-min sustained) for each IMD_CATEGORIES entry.
# Official IMD/RSMC New Delhi bands: Depression 17-27, Deep Depression 28-33,
# Cyclonic Storm 34-47, Severe CS 48-63, Very Severe CS 64-89,
# Extremely Severe CS 90-119, Super Cyclonic Storm >= 120.
# NOTE: IMD defines "Low Pressure Area" and "Well Marked Low" by pressure gradient,
# not by wind speed. The 0 / 8 kt bounds below are an approximation used only so the
# 9-class classifier head has a monotonic wind mapping for every output class.
IMD_WIND_THRESHOLDS_KT = [0, 8, 17, 28, 34, 48, 64, 90, 120]

# Saffir-Simpson scale (for reference/comparison)
SAFFIR_SIMPSON = {
    "Tropical Depression": (0, 33),
    "Tropical Storm": (34, 63),
    "Category 1": (64, 82),
    "Category 2": (83, 95),
    "Category 3": (96, 112),
    "Category 4": (113, 136),
    "Category 5": (137, 999),
}


def wind_to_imd_category(wind_knots: float) -> str:
    """Convert wind speed (knots) to IMD cyclone category."""
    for i in range(len(IMD_WIND_THRESHOLDS_KT) - 1, -1, -1):
        if wind_knots >= IMD_WIND_THRESHOLDS_KT[i]:
            return IMD_CATEGORIES[i]
    return IMD_CATEGORIES[0]


def knots_to_kmh(knots: float) -> float:
    """Convert knots to km/h."""
    return knots * 1.852


def kmh_to_knots(kmh: float) -> float:
    """Convert km/h to knots."""
    return kmh / 1.852

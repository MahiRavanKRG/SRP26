"""Backward-compatible facade re-exporting standardized preprocessing from ``srp26.data``."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from srp26.data import (
    StandardTrainScaler,
    TimeSeriesSlidingWindowDataset,
    build_dataset_dataloaders,
    build_etth1_dataloaders,
)

__all__ = [
    "StandardTrainScaler",
    "TimeSeriesSlidingWindowDataset",
    "build_dataset_dataloaders",
    "build_etth1_dataloaders",
]

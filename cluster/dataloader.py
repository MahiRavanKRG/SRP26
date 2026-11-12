"""Cluster runner facade re-exporting dataset helpers from ``srp26.data``."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from srp26.data import MTSDataset, create_dataloader

__all__ = ["MTSDataset", "create_dataloader"]

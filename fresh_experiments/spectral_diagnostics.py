"""Backward-compatible facade re-exporting Parseval spectral diagnostics from ``srp26.spectral``."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from srp26.spectral import (
    compute_log_spectral_distance,
    evaluate_all_spectral_diagnostics,
    parseval_band_decomposition,
    rolling_window_diagnostics,
)

__all__ = [
    "compute_log_spectral_distance",
    "evaluate_all_spectral_diagnostics",
    "parseval_band_decomposition",
    "rolling_window_diagnostics",
]

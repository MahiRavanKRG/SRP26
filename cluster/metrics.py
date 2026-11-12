"""Cluster runner facade re-exporting spectral metrics from ``srp26.spectral``."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from srp26.spectral import (
    band_wise_error_decomposition,
    fft_band_wise_error_decomposition,
    filter_band_wise_error_decomposition,
    psd_comparison,
    spectral_coherence_analysis,
)

__all__ = [
    "band_wise_error_decomposition",
    "fft_band_wise_error_decomposition",
    "filter_band_wise_error_decomposition",
    "psd_comparison",
    "spectral_coherence_analysis",
]

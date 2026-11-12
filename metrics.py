"""Backward-compatible facade re-exporting spectral metrics from ``srp26.spectral``."""

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

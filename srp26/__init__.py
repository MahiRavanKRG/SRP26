"""
SRP26 — Shared Scientific Utilities, Spectral Diagnostics, and Lightweight MTS Models.

Consolidates common data preprocessing, Fourier extrapolation, Parseval spectral
diagnostics, and lightweight baseline architectures so all experiment scripts share
one verified source of truth without hardcoded paths or duplicated helpers.
"""

from srp26.utils import ensure_hf_cache, get_device, get_repo_root, resolve_data_path, set_seed
from srp26.data import (
    MTSDataset,
    StandardScaler,
    StandardTrainScaler,
    TimeSeriesSlidingWindowDataset,
    build_dataset_dataloaders,
    build_etth1_dataloaders,
    create_dataloader,
    get_canonical_splits,
    load_eval_windows,
    load_raw_dataframe,
)
from srp26.fourier import (
    extract_and_extrapolate_fourier_base_np,
    extract_and_extrapolate_fourier_base_torch,
    extract_fourier_base_1d,
)
from srp26.spectral import (
    band_masks_rfft_torch,
    band_wise_error_decomposition,
    compute_log_spectral_distance,
    compute_spectral_diagnostics,
    dsof_band_metrics_torch,
    evaluate_all_spectral_diagnostics,
    fft_band_wise_error_decomposition,
    filter_band_wise_error_decomposition,
    parseval_band_decomposition,
    psd_comparison,
    rolling_error_tracking,
    rolling_window_diagnostics,
    spectral_coherence_analysis,
)
from srp26.models_lite import (
    CycleNet,
    DLinear,
    MovingAverageBlock,
    RecurrentCycleQueue,
    RevIN,
    train_and_evaluate_model,
)

__all__ = [
    "CycleNet",
    "DLinear",
    "MTSDataset",
    "MovingAverageBlock",
    "RecurrentCycleQueue",
    "RevIN",
    "StandardScaler",
    "StandardTrainScaler",
    "TimeSeriesSlidingWindowDataset",
    "band_masks_rfft_torch",
    "band_wise_error_decomposition",
    "build_dataset_dataloaders",
    "build_etth1_dataloaders",
    "compute_log_spectral_distance",
    "compute_spectral_diagnostics",
    "create_dataloader",
    "dsof_band_metrics_torch",
    "ensure_hf_cache",
    "evaluate_all_spectral_diagnostics",
    "extract_and_extrapolate_fourier_base_np",
    "extract_and_extrapolate_fourier_base_torch",
    "extract_fourier_base_1d",
    "fft_band_wise_error_decomposition",
    "filter_band_wise_error_decomposition",
    "get_canonical_splits",
    "get_device",
    "get_repo_root",
    "load_eval_windows",
    "load_raw_dataframe",
    "parseval_band_decomposition",
    "psd_comparison",
    "resolve_data_path",
    "rolling_error_tracking",
    "rolling_window_diagnostics",
    "set_seed",
    "spectral_coherence_analysis",
    "train_and_evaluate_model",
]


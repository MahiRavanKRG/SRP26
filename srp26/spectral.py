"""Parseval-exact spectral decomposition, PSD distortion, and rolling-window diagnostics."""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import signal


def _to_numpy_3d(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert Tensor or ndarray of shape ``[B, L]`` or ``[B, L, C]`` to float NumPy ``[B, L, C]``."""
    out = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    if out.ndim == 2:
        out = out[:, :, None]
    return out


def compute_spectral_diagnostics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Compute subband Fraction of Variance Unexplained (FVU) and Log-Spectral Distance (LSD).

    Partitions the rFFT frequency axis into three physical subbands:
      - Low band:  ``[0, F//4)``
      - Mid band:  ``[F//4, 3F//4)``
      - High band: ``[3F//4, F]``
    """
    y_t = y_true if y_true.ndim == 3 else y_true.unsqueeze(-1)
    y_p = y_pred if y_pred.ndim == 3 else y_pred.unsqueeze(-1)

    y_true_fft = torch.fft.rfft(y_t.permute(0, 2, 1).contiguous(), dim=-1)
    y_pred_fft = torch.fft.rfft(y_p.permute(0, 2, 1).contiguous(), dim=-1)

    power_error = torch.abs(y_true_fft - y_pred_fft) ** 2
    power_true = torch.abs(y_true_fft) ** 2

    f_bins = y_true_fft.shape[-1]
    b1 = max(1, f_bins // 4)
    b2 = min(f_bins - 1, 3 * f_bins // 4)
    if b2 <= b1:
        b2 = min(f_bins, b1 + 1)

    err_low = power_error[:, :, :b1].sum().item()
    err_mid = power_error[:, :, b1:b2].sum().item()
    err_high = power_error[:, :, b2:].sum().item()

    true_low = max(1e-6, power_true[:, :, :b1].sum().item())
    true_mid = max(1e-6, power_true[:, :, b1:b2].sum().item())
    true_high = max(1e-6, power_true[:, :, b2:].sum().item())

    eps = 1e-8
    spec_true_db = 10.0 * torch.log10(power_true + eps)
    spec_pred_db = 10.0 * torch.log10(torch.abs(y_pred_fft) ** 2 + eps)
    lsd = torch.sqrt(torch.mean((spec_true_db - spec_pred_db) ** 2)).item()

    return {
        "fvu_low": err_low / true_low,
        "fvu_mid": err_mid / true_mid,
        "fvu_high": err_high / true_high,
        "lsd_db": lsd,
    }


def parseval_band_decomposition(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    fs: float = 1.0,
    low_cutoff: float = 0.1,
    high_cutoff: float = 0.4,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Parseval-exact frequency-domain error decomposition using real FFT.

    Guarantees ``mse_low + mse_mid + mse_high == total_mse`` without filter ripple
    or boundary truncation artifacts.
    """
    y_t = _to_numpy_3d(y_true)
    y_p = _to_numpy_3d(y_pred)

    horizon = y_t.shape[1]
    freqs = np.fft.rfftfreq(horizon, d=1.0 / fs)

    weights = np.full(freqs.shape[0], 2.0)
    weights[0] = 1.0
    if horizon % 2 == 0:
        weights[-1] = 1.0
    weights = weights[None, :, None]

    mask_low = freqs < low_cutoff
    mask_mid = (freqs >= low_cutoff) & (freqs < high_cutoff)
    mask_high = freqs >= high_cutoff

    diff = y_t - y_p
    total_mse = float(np.mean(diff ** 2))

    diff_power = weights * (np.abs(np.fft.rfft(diff, axis=1)) ** 2) / (horizon ** 2)
    true_power = weights * (np.abs(np.fft.rfft(y_t, axis=1)) ** 2) / (horizon ** 2)
    pred_power = weights * (np.abs(np.fft.rfft(y_p, axis=1)) ** 2) / (horizon ** 2)

    def _calc_band(mask: np.ndarray, is_low: bool = False):
        if not np.any(mask):
            return 0.0, 0.0, 0.0
        mse_b = float(np.mean(np.sum(diff_power[:, mask, :], axis=1)))
        p_true_b = float(np.mean(np.sum(true_power[:, mask, :], axis=1)))
        p_pred_b = float(np.mean(np.sum(pred_power[:, mask, :], axis=1)))

        if is_low:
            dc_mean_sq = float(np.mean(np.mean(y_t, axis=1) ** 2))
            var_true_b = max(p_true_b - dc_mean_sq, eps)
        else:
            var_true_b = max(p_true_b, eps)

        fvu_b = float(mse_b / var_true_b)
        energy_ratio_b = float(p_pred_b / max(p_true_b, eps))
        return mse_b, fvu_b, energy_ratio_b

    mse_low, fvu_low, er_low = _calc_band(mask_low, is_low=True)
    mse_mid, fvu_mid, er_mid = _calc_band(mask_mid, is_low=False)
    mse_high, fvu_high, er_high = _calc_band(mask_high, is_low=False)

    return {
        "total_mse": total_mse,
        "mse_low": mse_low,
        "mse_mid": mse_mid,
        "mse_high": mse_high,
        "fvu_low": fvu_low,
        "fvu_mid": fvu_mid,
        "fvu_high": fvu_high,
        "energy_ratio_low": er_low,
        "energy_ratio_mid": er_mid,
        "energy_ratio_high": er_high,
        "parseval_additive_error": float(abs(total_mse - (mse_low + mse_mid + mse_high))),
    }


def fft_band_wise_error_decomposition(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    fs: float = 1.0,
    low_cutoff: float = 0.1,
    high_cutoff: float = 0.4,
    eps: float = 1e-8,
) -> Dict:
    """Wrapper matching the legacy ``metrics.py`` dictionary layout with ``method='fft'``."""
    out = parseval_band_decomposition(
        y_true, y_pred, fs=fs, low_cutoff=low_cutoff, high_cutoff=high_cutoff, eps=eps
    )
    out.pop("parseval_additive_error", None)
    out["method"] = "fft"
    return out


def filter_band_wise_error_decomposition(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    fs: float = 1.0,
    low_cutoff: float = 0.1,
    high_cutoff: float = 0.4,
    eps: float = 1e-8,
) -> Dict:
    """Time-domain zero-phase Butterworth IIR band decomposition."""
    y_t = _to_numpy_3d(y_true)
    y_p = _to_numpy_3d(y_pred)

    seq_len = y_t.shape[1]
    b_low, a_low = signal.butter(4, low_cutoff, btype="low", fs=fs)
    b_high, a_high = signal.butter(4, high_cutoff, btype="high", fs=fs)
    padlen = min(3 * max(len(a_low), len(b_low)), seq_len - 1)

    def _bands(data: np.ndarray):
        low = signal.filtfilt(b_low, a_low, data, axis=1, padlen=padlen)
        high = signal.filtfilt(b_high, a_high, data, axis=1, padlen=padlen)
        return low, data - low - high, high

    t_low, t_mid, t_high = _bands(y_t)
    p_low, p_mid, p_high = _bands(y_p)

    def _metrics(t_b: np.ndarray, p_b: np.ndarray):
        mse_b = float(np.mean((t_b - p_b) ** 2))
        var_t = max(float(np.var(t_b)), eps)
        var_p = float(np.var(p_b))
        return mse_b, float(mse_b / var_t), float(var_p / var_t)

    mse_low, fvu_low, er_low = _metrics(t_low, p_low)
    mse_mid, fvu_mid, er_mid = _metrics(t_mid, p_mid)
    mse_high, fvu_high, er_high = _metrics(t_high, p_high)

    return {
        "mse_low": mse_low,
        "mse_mid": mse_mid,
        "mse_high": mse_high,
        "fvu_low": fvu_low,
        "fvu_mid": fvu_mid,
        "fvu_high": fvu_high,
        "energy_ratio_low": er_low,
        "energy_ratio_mid": er_mid,
        "energy_ratio_high": er_high,
        "total_mse": float(np.mean((y_t - y_p) ** 2)),
        "method": "butterworth",
    }


def band_wise_error_decomposition(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    method: str = "fft",
    fs: float = 1.0,
    low_cutoff: float = 0.1,
    high_cutoff: float = 0.4,
) -> Dict:
    """Unified band-wise error decomposition interface ('fft' or 'butterworth')."""
    if method == "fft":
        return fft_band_wise_error_decomposition(
            y_true, y_pred, fs=fs, low_cutoff=low_cutoff, high_cutoff=high_cutoff
        )
    if method in ("butterworth", "filter"):
        return filter_band_wise_error_decomposition(
            y_true, y_pred, fs=fs, low_cutoff=low_cutoff, high_cutoff=high_cutoff
        )
    raise ValueError(f"Unknown method '{method}'. Choose 'fft' or 'butterworth'.")


def spectral_coherence_analysis(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    fs: float = 1.0,
) -> Dict[str, float]:
    """Compute mean magnitude-squared coherence between ground truth and predictions."""
    y_t = _to_numpy_3d(y_true)
    y_p = _to_numpy_3d(y_pred)
    nperseg = min(256, y_t.shape[1])

    coherence_vals: List[float] = []
    for b in range(y_t.shape[0]):
        for c in range(y_t.shape[2]):
            _, cxy = signal.coherence(y_t[b, :, c], y_p[b, :, c], fs=fs, nperseg=nperseg)
            coherence_vals.append(float(np.mean(np.nan_to_num(cxy, nan=0.0))))
    return {"mean_spectral_coherence": float(np.mean(coherence_vals))}


def compute_log_spectral_distance(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    fs: float = 1.0,
    nperseg: int = 64,
) -> float:
    """Compute mean Log-Spectral Distance (dB) via Welch PSD estimation."""
    y_t = _to_numpy_3d(y_true)
    y_p = _to_numpy_3d(y_pred)

    seg = min(nperseg, y_t.shape[1])
    eps = 1e-10
    _, p_true = signal.welch(y_t, fs=fs, nperseg=seg, axis=1)
    _, p_pred = signal.welch(y_p, fs=fs, nperseg=seg, axis=1)
    lsd_per_series = np.sqrt(
        np.mean((10.0 * np.log10(p_true + eps) - 10.0 * np.log10(p_pred + eps)) ** 2, axis=1)
    )
    return float(np.mean(lsd_per_series))


def psd_comparison(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    fs: float = 1.0,
) -> Dict[str, float]:
    """Legacy dictionary wrapper around Welch PSD Log-Spectral Distance."""
    return {"mean_lsd": compute_log_spectral_distance(y_true, y_pred, fs=fs, nperseg=256)}


def rolling_window_diagnostics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    window_size: int = 24,
    threshold_sigma: float = 2.0,
) -> Dict[str, float]:
    """Sliding-window failure localization identifying transient regime-shift spikes."""
    y_t = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    y_p = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.asarray(y_pred)

    sq_error = np.mean((y_t - y_p) ** 2, axis=-1)
    total_samples = sq_error.shape[0]
    flagged_samples = 0
    max_rolling_errs = []

    for b in range(total_samples):
        series = pd.Series(sq_error[b])
        rolling_mse = series.rolling(window=min(window_size, len(series)), min_periods=1).mean()
        mean_err = rolling_mse.mean()
        std_err = rolling_mse.std() if len(rolling_mse) > 1 else 0.0
        thresh = mean_err + threshold_sigma * std_err
        if np.any(rolling_mse > thresh):
            flagged_samples += 1
        max_rolling_errs.append(float(rolling_mse.max()))

    return {
        "failure_rate_pct": float((flagged_samples / max(1, total_samples)) * 100.0),
        "mean_peak_rolling_mse": float(np.mean(max_rolling_errs)) if max_rolling_errs else 0.0,
    }


def rolling_error_tracking(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    window_size: int = 24,
    threshold_sigma: float = 2.0,
) -> List[Dict]:
    """Per-batch rolling MSE series and dynamic anomaly threshold flags."""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    sq_error = np.mean((y_true_np - y_pred_np) ** 2, axis=-1)

    results = []
    for b in range(sq_error.shape[0]):
        rolling_mse = pd.Series(sq_error[b]).rolling(window=window_size).mean()
        threshold = rolling_mse.mean() + (threshold_sigma * rolling_mse.std())
        flags = rolling_mse > threshold
        results.append(
            {
                "rolling_mse": rolling_mse.values,
                "flags": flags.values,
                "threshold": threshold,
            }
        )
    return results


def evaluate_all_spectral_diagnostics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> Dict[str, float]:
    """Compute Parseval subband decomposition, Welch LSD, and rolling-window diagnostics."""
    parseval_dict = parseval_band_decomposition(y_true, y_pred)
    lsd_val = compute_log_spectral_distance(y_true, y_pred)
    rolling_dict = rolling_window_diagnostics(y_true, y_pred)
    return {
        **parseval_dict,
        "log_spectral_distance": lsd_val,
        **rolling_dict,
    }

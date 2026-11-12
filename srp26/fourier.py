"""Analytical Fourier decomposition and closed-form sinusoidal extrapolation."""

import math
from typing import Tuple

import numpy as np
import torch


def extract_and_extrapolate_fourier_base_torch(
    x: torch.Tensor,
    pred_len: int,
    top_k: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized analytical Fourier decomposition and sinusoidal extrapolation.

    Args:
        x: Lookback tensor of shape ``[B, L, C]``.
        pred_len: Forecast horizon ``H``.
        top_k: Number of dominant non-DC Fourier harmonics to retain.

    Returns:
        Tuple ``(x_base, y_base, x_res)`` with shapes ``[B, L, C]``,
        ``[B, pred_len, C]``, and ``[B, L, C]`` respectively.
    """
    _, seq_len, _ = x.shape
    x_perm = x.permute(0, 2, 1).contiguous()  # [B, C, L] (contiguous required for MKL rfft)
    fft_vals = torch.fft.rfft(x_perm, dim=-1)
    amps = torch.abs(fft_vals)
    amps[..., 0] = 0.0  # Exclude DC bin from top-k harmonic selection

    k = min(top_k, amps.shape[-1] - 1)
    _, topk_indices = torch.topk(amps, k=k, dim=-1)  # [B, C, k]

    filtered_fft = torch.zeros_like(fft_vals)
    filtered_fft.scatter_(-1, topk_indices, torch.gather(fft_vals, -1, topk_indices))
    x_base = torch.fft.irfft(filtered_fft, n=seq_len, dim=-1)

    t_future = torch.arange(seq_len, seq_len + pred_len, device=x.device, dtype=torch.float32).view(1, 1, 1, pred_len)
    freqs = (topk_indices.float() / float(seq_len)).unsqueeze(-1)
    topk_fft = torch.gather(fft_vals, -1, topk_indices)
    topk_amps = (torch.abs(topk_fft) / (float(seq_len) / 2.0)).unsqueeze(-1)
    topk_phases = torch.angle(topk_fft).unsqueeze(-1)

    cos_terms = topk_amps * torch.cos(2.0 * math.pi * freqs * t_future + topk_phases)
    y_base = torch.sum(cos_terms, dim=2)

    x_base = x_base.permute(0, 2, 1)
    y_base = y_base.permute(0, 2, 1)
    x_res = x - x_base
    return x_base, y_base, x_res


def extract_fourier_base_1d(
    series: np.ndarray,
    pred_len: int,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-channel linear-detrended Fourier decomposition and future extrapolation.

    Returns ``(x_base, y_base)`` for a 1D NumPy array ``series``.
    """
    seq_len = len(series)
    t_in = np.arange(seq_len)
    t_future = np.arange(seq_len, seq_len + pred_len)

    poly = np.polyfit(t_in, series, deg=1)
    trend_in = np.polyval(poly, t_in)
    trend_future = np.polyval(poly, t_future)

    detrended = series - trend_in
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(seq_len)
    amps = np.abs(fft_vals)

    k = min(top_k, len(amps) - 1)
    top_indices = np.argsort(amps[1:])[-k:] + 1 if k > 0 else np.array([], dtype=int)

    filtered_fft = np.zeros_like(fft_vals)
    if len(top_indices) > 0:
        filtered_fft[top_indices] = fft_vals[top_indices]
    harmonics_in = np.fft.irfft(filtered_fft, n=seq_len)
    x_base = (trend_in + harmonics_in).astype(series.dtype)

    harmonics_future = np.zeros(pred_len, dtype=np.float64)
    for idx in top_indices:
        amp = amps[idx] / (seq_len / 2.0)
        phase = np.angle(fft_vals[idx])
        freq = freqs[idx]
        harmonics_future += amp * np.cos(2.0 * np.pi * freq * t_future + phase)

    y_base = (trend_future + harmonics_future).astype(series.dtype)
    return x_base, y_base


def extract_and_extrapolate_fourier_base_np(
    x: np.ndarray,
    pred_len: int,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy convenience wrapper over ``extract_and_extrapolate_fourier_base_torch``."""
    x_t = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        x_base, y_base, x_res = extract_and_extrapolate_fourier_base_torch(x_t, pred_len=pred_len, top_k=top_k)
    return x_base.numpy(), y_base.numpy(), x_res.numpy()

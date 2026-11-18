"""Transform registry for the SRP26 transform-ablation study.

Each transform exposes a single method ``encode`` that maps a lookback window
``x`` of shape ``[B, L, C]`` to a **per-channel feature vector** ``z`` of shape
``[B, C, F]``.  The forecasting head is *not* part of the transform: a shared
linear/MLP map ``F -> H`` is applied channel-independently by
``srp26.transform_ablation.TransformForecaster``.  This keeps the transform the
single varying factor in the ablation.

Energy normalization
--------------------
Orthonormal bases (identity, DCT-II, orthonormal wavelets, and the
unitary-scaled rFFT) preserve Parseval energy, so features stay in the same
numeric range as the RevIN-standardized input.  Non-orthonormal transforms
(STFT with an overlapping window) are rescaled by a fixed, data-independent
constant.  Any residual scale difference is absorbed by the learned linear
head, which is scale-invariant under Adam; the normalization's job is
optimization stability and comparable weight magnitudes, not to change the
final learned mapping.
"""

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import ptwt


def _moving_average_1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Moving average along the last dim of ``x`` (shape ``[B, C, L]``)."""
    front = x[:, :, 0:1].repeat(1, 1, (kernel_size - 1) // 2)
    back = x[:, :, -1:].repeat(1, 1, kernel_size // 2)
    xp = torch.cat([front, x, back], dim=-1)
    return F.avg_pool1d(xp, kernel_size=kernel_size, stride=1)


class BaseTransform(nn.Module):
    """Interface shared by every transform in the registry."""

    name: str = "base"

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def feature_dim(self) -> int:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` of shape ``[B, L, C]`` to features of shape ``[B, C, F]``."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class IdentityTransform(BaseTransform):
    """Time domain: raw lookback, per channel. Orthonormal (trivial)."""

    name = "identity"

    def feature_dim(self) -> int:
        return self.seq_len

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).contiguous()  # [B, C, L]


class DLinearDecompTransform(BaseTransform):
    """Trend/seasonal moving-average decomposition (DLinear's basis).

    Equivalent to DLinear's two linears when paired with a linear head:
    a single ``Linear(2L -> H)`` on ``[seasonal, trend]`` is ``W_s·seasonal +
    W_t·trend``.  Feature dim ``2L``.
    """

    name = "dlinear"

    def __init__(self, seq_len: int, kernel_size: int = 25) -> None:
        super().__init__(seq_len)
        self.kernel_size = kernel_size

    def feature_dim(self) -> int:
        return 2 * self.seq_len

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        trend = _moving_average_1d(xc, self.kernel_size)
        seasonal = xc - trend
        return torch.cat([seasonal, trend], dim=-1)  # [B, C, 2L]


class RFFTTransform(BaseTransform):
    """Global Fourier: real + imaginary of the real FFT, unitary-scaled.

    ``torch.fft.rfft`` yields ``L//2 + 1`` complex bins; stacking real and
    imaginary parts gives feature dim ``2 * (L//2 + 1)``.  Divided by
    ``sqrt(L)`` so features have the unitary-DFT energy scale.
    """

    name = "rfft"

    def feature_dim(self) -> int:
        return 2 * (self.seq_len // 2 + 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        spec = torch.fft.rfft(xc, dim=-1) / math.sqrt(self.seq_len)  # [B, C, L//2+1]
        return torch.cat([spec.real, spec.imag], dim=-1)  # [B, C, 2*(L//2+1)]


class DCTTransform(BaseTransform):
    """Orthonormal DCT-II (real cosine basis), decorrelates autocorrelated input."""

    name = "dct"

    def __init__(self, seq_len: int) -> None:
        super().__init__(seq_len)
        self.register_buffer("dct", torch.tensor(_dct2_matrix(seq_len), dtype=torch.float32))

    def feature_dim(self) -> int:
        return self.seq_len

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        return torch.matmul(xc, self.dct.t())  # [B, C, L] (coeffs X = D @ x)


class STFTTransform(BaseTransform):
    """Windowed Fourier (fixed Hann window), real+imag stacked.

    Implemented with ``unfold`` + rFFT for unambiguous frame/freq axes.  The
    window is unit-energy normalized; the transform is redundant (non-unitary),
    so the linear head absorbs the residual scale.
    """

    name = "stft"

    def __init__(self, seq_len: int, window_size: int = 64, hop: int = 32) -> None:
        super().__init__(seq_len)
        self.window_size = window_size
        self.hop = hop
        self.frames = 1 + (seq_len - window_size) // hop
        self.freqs = window_size // 2 + 1
        win = torch.hann_window(window_size)
        self.register_buffer("window", win / win.norm())

    def feature_dim(self) -> int:
        return 2 * self.frames * self.freqs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        frames = xc.unfold(-1, self.window_size, self.hop)  # [B, C, frames, W]
        frames = frames * self.window  # [B, C, frames, W]
        spec = torch.fft.rfft(frames, dim=-1)  # [B, C, frames, freqs]
        z = torch.cat([spec.real, spec.imag], dim=-1)  # [B, C, frames, 2*freqs]
        return z.flatten(start_dim=2)  # [B, C, F]


class DWTTransform(BaseTransform):
    """Discrete Wavelet Transform (orthonormal, e.g. Haar / Daubechies).

    Coefficients are concatenated across subbands and padded/truncated to
    ``seq_len`` so every wavelet variant shares the same feature dim.
    """

    name = "dwt"

    def __init__(self, seq_len: int, wavelet: str = "haar", level: int = 2) -> None:
        super().__init__(seq_len)
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level

    def feature_dim(self) -> int:
        return self.seq_len

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        coeffs = ptwt.wavedec(xc, self.wavelet, level=self.level)  # list of [B, C, len_k]
        z = torch.cat(coeffs, dim=-1)  # [B, C, total]
        total = z.shape[-1]
        if total >= self.seq_len:
            return z[..., : self.seq_len]
        pad = torch.zeros(*z.shape[:-1], self.seq_len - total, device=z.device, dtype=z.dtype)
        return torch.cat([z, pad], dim=-1)


def _dct2_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix ``D`` of shape ``[n, n]`` (``X = D @ x``)."""
    k = np.arange(n, dtype=np.float64)[:, None]  # [n, 1]
    m = np.arange(n, dtype=np.float64)[None, :]  # [1, n]
    d = np.sqrt(2.0 / n) * np.cos((np.pi / n) * (m + 0.5) * k)
    d[0, :] = np.sqrt(1.0 / n)
    return d


def _dwt_haar(seq_len: int) -> DWTTransform:
    return DWTTransform(seq_len, wavelet="haar", level=2)


def _dwt_db4(seq_len: int) -> DWTTransform:
    return DWTTransform(seq_len, wavelet="db4", level=2)


TRANSFORMS: Dict[str, BaseTransform] = {
    "identity": IdentityTransform,
    "dlinear": DLinearDecompTransform,
    "rfft": RFFTTransform,
    "dct": DCTTransform,
    "stft": STFTTransform,
    "dwt_haar": _dwt_haar,
    "dwt_db4": _dwt_db4,
}

ALL_TRANSFORMS = list(TRANSFORMS.keys())


def build_transform(name: str, seq_len: int) -> BaseTransform:
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform '{name}'. Choose from {ALL_TRANSFORMS}.")
    return TRANSFORMS[name](seq_len)

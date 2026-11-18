"""v2 metric set (protocol §3a), computed in one streaming pass over the test set.

Primary: MSE, MAE, skill vs seasonal naive.
Diagnostic: Parseval band MSE / FVU / energy ratio over *physical* bands
(slow > 36 h, daily 8-36 h, fast < 8 h), error vs lead time, per-window error.
Secondary (appendix): per-band log-spectral distance ("spectral realism").

All quantities are additive over windows, so they are accumulated batch by
batch and never require the full ``[N, H, C]`` prediction tensor in memory.
"""

from typing import Dict, Optional

import numpy as np
import torch

SLOW_MIN_HOURS = 36.0
FAST_MAX_HOURS = 8.0
BANDS = ("level", "slow", "daily", "fast")


def band_masks(horizon: int, dt_hours: float) -> Dict[str, torch.Tensor]:
    """Boolean masks over the ``horizon//2+1`` rFFT bins, by period in hours.

    ``level`` is the DC bin alone (error in the window mean); ``slow`` starts
    at the first non-zero frequency, so level + slow + daily + fast == MSE.
    """
    freqs = torch.fft.rfftfreq(horizon)  # cycles per step
    period_h = torch.full_like(freqs, float("inf"))
    period_h[1:] = dt_hours / freqs[1:]
    level = torch.zeros_like(freqs, dtype=torch.bool)
    level[0] = True
    return {
        "level": level,
        "slow": (period_h > SLOW_MIN_HOURS) & ~level,
        "daily": (period_h >= FAST_MAX_HOURS) & (period_h <= SLOW_MIN_HOURS),
        "fast": period_h < FAST_MAX_HOURS,
    }


def parseval_weights(horizon: int) -> torch.Tensor:
    """One-sided rFFT weights so that sum_k w_k |X_k|^2 / H^2 == mean_t x_t^2."""
    n_bins = horizon // 2 + 1
    w = torch.full((n_bins,), 2.0, dtype=torch.float64)
    w[0] = 1.0
    if horizon % 2 == 0:
        w[-1] = 1.0
    return w


class StreamingEvaluator:
    """Accumulates every v2 metric over batches of ``[B, H, C]`` predictions."""

    def __init__(self, horizon: int, dt_hours: float, device: torch.device, eps: float = 1e-8):
        self.h = horizon
        self.eps = eps
        self.device = device
        self.masks = {k: v.to(device) for k, v in band_masks(horizon, dt_hours).items()}
        self.w = parseval_weights(horizon).to(device)
        self.n_elem = 0  # windows * channels
        self.sq_sum = 0.0
        self.abs_sum = 0.0
        self.snaive_sq_sum = 0.0
        self.snaive_abs_sum = 0.0
        self.step_sq = torch.zeros(horizon, dtype=torch.float64, device=device)
        self.band = {k: {"err": 0.0, "true": 0.0, "pred": 0.0, "lsd_sq": 0.0, "lsd_n": 0} for k in BANDS}
        self.window_mse = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor, snaive: Optional[torch.Tensor] = None) -> None:
        pred = pred.double()
        true = true.double()
        b, h, c = true.shape
        diff = pred - true
        self.n_elem += b * c
        self.sq_sum += diff.pow(2).sum().item()
        self.abs_sum += diff.abs().sum().item()
        self.step_sq += diff.pow(2).sum(dim=(0, 2))
        self.window_mse.append(diff.pow(2).mean(dim=(1, 2)).float().cpu())
        if snaive is not None:
            sd = snaive.double() - true
            self.snaive_sq_sum += sd.pow(2).sum().item()
            self.snaive_abs_sum += sd.abs().sum().item()

        scale = self.w / float(h * h)  # [F]
        f_diff = torch.fft.rfft(diff, dim=1)  # [B, F, C]
        f_true = torch.fft.rfft(true, dim=1)
        f_pred = torch.fft.rfft(pred, dim=1)
        p_diff = f_diff.abs().pow(2) * scale[None, :, None]
        p_true = f_true.abs().pow(2) * scale[None, :, None]
        p_pred = f_pred.abs().pow(2) * scale[None, :, None]
        db_true = 10.0 * torch.log10(f_true.abs().pow(2) + self.eps)
        db_pred = 10.0 * torch.log10(f_pred.abs().pow(2) + self.eps)
        lsd_sq = (db_true - db_pred).pow(2)
        for name, mask in self.masks.items():
            if not bool(mask.any()):
                continue
            acc = self.band[name]
            acc["err"] += p_diff[:, mask, :].sum().item()
            acc["true"] += p_true[:, mask, :].sum().item()
            acc["pred"] += p_pred[:, mask, :].sum().item()
            acc["lsd_sq"] += lsd_sq[:, mask, :].sum().item()
            acc["lsd_n"] += int(mask.sum().item()) * b * c

    def result(self) -> Dict[str, object]:
        n = self.n_elem * self.h
        mse = self.sq_sum / n
        mae = self.abs_sum / n
        out: Dict[str, object] = {"mse": mse, "mae": mae}
        if self.snaive_sq_sum > 0:
            out["snaive_mse"] = self.snaive_sq_sum / n
            out["snaive_mae"] = self.snaive_abs_sum / n
            out["skill_mse"] = 1.0 - mse / out["snaive_mse"]
            out["skill_mae"] = 1.0 - mae / out["snaive_mae"]
        band_total = 0.0
        for name in BANDS:
            acc = self.band[name]
            if acc["lsd_n"] == 0:  # band not resolvable at this horizon (e.g. daily at H=96, 10-min data)
                out[f"{name}_mse"] = None
                out[f"{name}_fvu"] = None
                out[f"{name}_energy_ratio"] = None
                out[f"{name}_lsd_db"] = None
                continue
            band_mse = acc["err"] / self.n_elem
            band_total += band_mse
            out[f"{name}_mse"] = band_mse
            # FVU/energy ratio are undefined for the level term (true "power" there is the squared mean)
            out[f"{name}_fvu"] = None if name == "level" else acc["err"] / max(acc["true"], self.eps)
            out[f"{name}_energy_ratio"] = None if name == "level" else acc["pred"] / max(acc["true"], self.eps)
            out[f"{name}_lsd_db"] = float(np.sqrt(acc["lsd_sq"] / acc["lsd_n"]))
        out["parseval_abs_error"] = abs(band_total - mse)
        out["mse_by_step"] = (self.step_sq / self.n_elem).cpu().numpy().round(6).tolist()
        return out

    def window_errors(self) -> np.ndarray:
        return torch.cat(self.window_mse).numpy() if self.window_mse else np.zeros(0, dtype=np.float32)


def seasonal_naive(x: torch.Tensor, pred_len: int, period: int) -> torch.Tensor:
    """Repeat the last full ``period`` of the lookback ``x`` ([B, L, C]) over the horizon."""
    last = x[:, -period:, :]
    reps = -(-pred_len // period)
    return last.repeat(1, reps, 1)[:, :pred_len, :]

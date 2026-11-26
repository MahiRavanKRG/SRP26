"""Transform-ablation forecaster and trainer (Tier-2 scheme).

A single minimal backbone — a shared linear (or small MLP) head — is applied
to the per-channel feature vector produced by a transform, so the transform is
the only varying factor across runs.  RevIN is applied around the whole
pipeline per the standardization protocol (``L=336``, RevIN on).
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from srp26.models_lite import RevIN
from srp26.spectral import evaluate_all_spectral_diagnostics
from srp26.transforms import build_transform


class TransformForecaster(nn.Module):
    """RevIN -> transform -> shared head -> RevIN⁻¹, mapping ``[B, L, C]`` to ``[B, H, C]``."""

    def __init__(
        self,
        transform_name: str,
        seq_len: int,
        pred_len: int,
        channels: int,
        head: str = "linear",
        hidden: int = 128,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.revin = RevIN() if revin else None
        self.transform = build_transform(transform_name, seq_len)
        feature_dim = self.transform.feature_dim()
        if head == "linear":
            self.head = nn.Linear(feature_dim, pred_len)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, pred_len)
            )
        else:
            raise ValueError(f"Unknown head '{head}'. Choose 'linear' or 'mlp'.")
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        z = self.transform.encode(x)  # [B, C, F]
        y = self.head(z)  # [B, C, H]
        y = y.permute(0, 2, 1)  # [B, H, C]
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


def train_transform_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
) -> Tuple[Dict, np.ndarray]:
    """Train with AdamW + cosine annealing + AMP + early stopping; evaluate with DSOF.

    Returns ``(record, per_window_mse)`` where ``record`` holds scalar metrics
    (MSE/MAE, Parseval subband FVU and energy ratios, LSD, rolling failure rate)
    and ``per_window_mse`` is the per-test-window MSE vector (mean over horizon
    and channels) used for paired significance testing.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_weights = None
    best_epoch = 1
    patience_ctr = 0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                val_loss += criterion(model(xv), yv).item() * len(xv)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_ctr = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    train_time = time.time() - t0
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xt, yt in test_loader:
            preds.append(model(xt.to(device)).cpu())
            trues.append(yt)
    preds_t = torch.cat(preds, 0)
    trues_t = torch.cat(trues, 0)

    mse = float(nn.functional.mse_loss(preds_t, trues_t).item())
    mae = float(nn.functional.l1_loss(preds_t, trues_t).item())
    spec = evaluate_all_spectral_diagnostics(trues_t, preds_t)
    per_window_mse = (preds_t - trues_t).pow(2).mean(dim=(1, 2)).numpy()

    record = {
        "params": params,
        "train_time_sec": round(train_time, 2),
        "best_epoch": best_epoch,
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "fvu_low": round(spec["fvu_low"], 4),
        "fvu_mid": round(spec["fvu_mid"], 4),
        "fvu_high": round(spec["fvu_high"], 4),
        "energy_ratio_low": round(spec["energy_ratio_low"], 4),
        "energy_ratio_mid": round(spec["energy_ratio_mid"], 4),
        "energy_ratio_high": round(spec["energy_ratio_high"], 4),
        "log_spectral_distance": round(spec["log_spectral_distance"], 4),
        "failure_rate_pct": round(spec["failure_rate_pct"], 3),
        "mean_peak_rolling_mse": round(spec["mean_peak_rolling_mse"], 5),
    }
    return record, per_window_mse

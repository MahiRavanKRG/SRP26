"""Lightweight benchmark architectures (RevIN, CycleNet, DLinear) and shared trainer."""

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from srp26.spectral import compute_spectral_diagnostics


class RevIN(nn.Module):
    """Parameter-free Reversible Instance Normalization (Kim et al., ICLR 2022)."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        if mode == "denorm":
            if self.mean is None or self.stdev is None:
                raise RuntimeError("RevIN 'norm' must be called before 'denorm'.")
            return x * self.stdev + self.mean
        raise ValueError(f"Unknown RevIN mode: {mode!r}")


class RecurrentCycleQueue(nn.Module):
    """Learnable recurrent cycle state queue (Lin et al., NeurIPS 2024)."""

    def __init__(self, cycle_len: int, channels: int) -> None:
        super().__init__()
        self.cycle_len = cycle_len
        self.channels = channels
        self.cycle_queue = nn.Parameter(torch.zeros(cycle_len, channels))
        nn.init.normal_(self.cycle_queue, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        repeats = (seq_len + self.cycle_len - 1) // self.cycle_len
        full_cycle = self.cycle_queue.repeat(repeats, 1)[:seq_len, :]
        batch_cycle = full_cycle.unsqueeze(0).expand(batch_size, -1, -1)
        return x - batch_cycle, batch_cycle


class CycleNet(nn.Module):
    """Lightweight CycleNet baseline combining RevIN, RecurrentCycleQueue, and linear residual head."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, cycle_len: int = 24) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN()
        self.cycle_module = RecurrentCycleQueue(cycle_len=cycle_len, channels=channels)
        self.linear_backbone = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin(x, "norm")
        x_res, _ = self.cycle_module(x_norm)
        y_res = self.linear_backbone(x_res.permute(0, 2, 1)).permute(0, 2, 1)

        repeats = (self.pred_len + self.cycle_module.cycle_len - 1) // self.cycle_module.cycle_len
        y_cycle = (
            self.cycle_module.cycle_queue.repeat(repeats, 1)[: self.pred_len, :]
            .unsqueeze(0)
            .expand(x.shape[0], -1, -1)
        )
        return self.revin(y_res + y_cycle, "denorm")


class MovingAverageBlock(nn.Module):
    """Moving-average trend/seasonal decomposition block (Zeng et al., AAAI 2023)."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        pad_back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([pad_front, x, pad_back], dim=1)
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """DLinear time-domain decomposition baseline (Zeng et al., AAAI 2023)."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.decom = MovingAverageBlock(kernel_size=kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decom(x)
        s_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        t_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        return s_out + t_out


def train_and_evaluate_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 30,
    patience: int = 7,
    lr: float = 3e-4,
) -> Dict:
    """Train with AdamW, cosine annealing, AMP on CUDA, early stopping, and evaluate."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    use_cuda_amp = device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    last_epoch = 0
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    start_time = time.time()

    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                y_hat = model(x_batch)
                loss = criterion(y_hat, y_batch)

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_loss += criterion(model(x_val), y_val).item() * len(x_val)
        val_loss /= max(1, len(val_loader.dataset))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            test_preds.append(model(x_test.to(device)).cpu())
            test_trues.append(y_test.cpu())

    preds_t = torch.cat(test_preds, dim=0)
    trues_t = torch.cat(test_trues, dim=0)
    spectral = compute_spectral_diagnostics(trues_t, preds_t)

    return {
        "model": model_name,
        "params": param_count,
        "train_time_sec": round(train_time, 2),
        "best_epoch": max(1, last_epoch - patience_counter + 1),
        "test_mse": round(F.mse_loss(preds_t, trues_t).item(), 4),
        "test_mae": round(F.l1_loss(preds_t, trues_t).item(), 4),
        "fvu_low": round(spectral["fvu_low"], 4),
        "fvu_mid": round(spectral["fvu_mid"], 4),
        "fvu_high": round(spectral["fvu_high"], 4),
        "lsd_db": round(spectral["lsd_db"], 2),
    }


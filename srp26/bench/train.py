"""The single v2 trainer/evaluator used by every model (protocol §2.1).

Recipe: Adam, cosine-annealed LR, FP32, MSE loss, early stopping on
validation MSE with best-checkpoint restore. The test set is only touched in
``evaluate_test``, after training has finished.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from srp26.bench.metrics import StreamingEvaluator, seasonal_naive


@torch.no_grad()
def mean_mse(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y, cyc in loader:
        x, y, cyc = x.to(device), y.to(device), cyc.to(device)
        total += (model(x, cyc) - y).double().pow(2).sum().item()
        count += y.numel()
    return total / max(count, 1)


def fit(model: nn.Module, train_loader, val_loader, device: torch.device, lr: float,
        max_epochs: int, patience: int) -> Dict[str, object]:
    model.to(device)
    if not getattr(model, "trainable", True):
        return {"epochs_run": 0, "best_epoch": 0, "train_time_sec": 0.0,
                "val_mse": mean_mse(model, val_loader, device), "val_curve": []}

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    curve = []
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y, cyc in train_loader:
            x, y, cyc = x.to(device), y.to(device), cyc.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (model(x, cyc) - y).pow(2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            opt.step()
        sched.step()
        val = mean_mse(model, val_loader, device)
        curve.append(round(val, 6))
        if val < best_val:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": epoch, "best_epoch": best_epoch, "train_time_sec": round(time.time() - t0, 2),
            "val_mse": best_val, "val_curve": curve}


@torch.no_grad()
def evaluate_test(model: nn.Module, test_loader, device: torch.device, pred_len: int, dt_hours: float,
                  snaive_period: int) -> Tuple[Dict[str, object], np.ndarray]:
    model.eval()
    ev = StreamingEvaluator(pred_len, dt_hours, device)
    for x, y, cyc in test_loader:
        x, y, cyc = x.to(device), y.to(device), cyc.to(device)
        ev.update(model(x, cyc), y, snaive=seasonal_naive(x, pred_len, snaive_period))
    return ev.result(), ev.window_errors()

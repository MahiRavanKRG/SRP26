"""Model registry for the v2 harness.

Every model exposes ``forward(x, cycle_index) -> y`` with ``x: [B, L, C]`` and
``y: [B, H, C]`` so the trainer and evaluator are identical for all of them.

CycleNet-Pro is imported from ``models/cyclenet/cyclenet_pro_Full.py`` (the
four-backbone version) so v2 runs the proposed model's own code, including its
automatic period discovery (mean-of-channels FFT + ACF) and channel-dependence
diagnostic.
"""

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.cyclenet.cyclenet_pro_Full import (
    CycleNetPro,
    compute_correlation_metrics,
    decide_channel_dependence,
    decide_multi_or_single_cycle,
    filter_true_periods_acf,
    find_top_periods_fft,
)
from srp26.bench.data import Bundle
from srp26.bench.metrics import seasonal_naive
from srp26.transform_ablation import TransformForecaster
from srp26.transforms import ALL_TRANSFORMS

# --------------------------------------------------------------------------
# Untrained references
# --------------------------------------------------------------------------


class RepeatLast(nn.Module):
    trainable = False

    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x, cycle_index=None):
        return x[:, -1:, :].repeat(1, self.pred_len, 1)


class SeasonalNaive(nn.Module):
    trainable = False

    def __init__(self, pred_len: int, period: int):
        super().__init__()
        self.pred_len = pred_len
        self.period = period

    def forward(self, x, cycle_index=None):
        return seasonal_naive(x, self.pred_len, self.period)


# --------------------------------------------------------------------------
# Wrappers giving every trained model the (x, cycle_index) signature
# --------------------------------------------------------------------------


class _NoCycle(nn.Module):
    trainable = True

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, cycle_index=None):
        return self.net(x)


class _WithCycle(nn.Module):
    trainable = True

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, cycle_index):
        return self.net(x, cycle_index)


# --------------------------------------------------------------------------
# PatchTST (Nie et al., ICLR 2023), supervised PatchTST/42 configuration
# --------------------------------------------------------------------------


class _BatchNormTokens(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):  # [N, P, D]
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class _PatchTSTLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = _BatchNormTokens(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = _BatchNormTokens(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop1(a))
        return self.norm2(x + self.drop2(self.ff(x)))


class PatchTST(nn.Module):
    """Channel-independent PatchTST: RevIN, end-padded patches (16/8 -> 42 for L=336),
    learnable positions, BatchNorm Transformer encoder, flatten head.

    Re-implementation; differs from the official code only in not using
    residual attention scores. Recorded in the result's ``model_notes``.
    """

    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8, n_layers=3,
                 d_model=128, n_heads=16, d_ff=256, dropout=0.2, head_dropout=0.0):
        super().__init__()
        self.revin_weight = nn.Parameter(torch.ones(enc_in))
        self.revin_bias = nn.Parameter(torch.zeros(enc_in))
        self.patch_len, self.stride = patch_len, stride
        self.pad = nn.ReplicationPad1d((0, stride))
        n_patches = (seq_len - patch_len) // stride + 2
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.empty(n_patches, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_PatchTSTLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Dropout(head_dropout),
                                  nn.Linear(n_patches * d_model, pred_len))

    def forward(self, x, cycle_index=None):
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - mean) / std * self.revin_weight + self.revin_bias
        b, l, c = x.shape
        z = self.pad(x.permute(0, 2, 1))  # [B, C, L+S]
        z = z.unfold(-1, self.patch_len, self.stride)  # [B, C, P, patch]
        z = z.reshape(b * c, z.shape[2], self.patch_len)
        z = self.drop(self.embed(z) + self.pos)
        for layer in self.layers:
            z = layer(z)
        y = self.head(z).reshape(b, c, -1).permute(0, 2, 1)  # [B, H, C]
        y = (y - self.revin_bias) / (self.revin_weight + 1e-10)
        return y * std + mean


PATCHTST_SMALL = dict(d_model=16, n_heads=4, d_ff=128, dropout=0.3)  # ETTh1/ETTh2 official scripts
PATCHTST_LARGE = dict(d_model=128, n_heads=16, d_ff=256, dropout=0.2)  # all other datasets


# --------------------------------------------------------------------------
# CycleNet-Pro decisions (taken from cyclenet_pro_Full.main, unchanged)
# --------------------------------------------------------------------------


def cyclenet_pro_decisions(bundle: Bundle, seq_len: int) -> Dict[str, object]:
    train_raw = bundle.train_raw
    mean_series = train_raw.mean(axis=1)
    max_period = min(seq_len, max(len(train_raw) // 2, 4))
    candidates = find_top_periods_fft(mean_series, n_periods=5, min_period=2, max_period=max_period)
    true_periods, noise_periods, _ = filter_true_periods_acf(mean_series, candidates)
    fallback = False
    if not true_periods:
        true_periods = candidates[:1] if candidates else [24]
        fallback = not candidates
    periods = decide_multi_or_single_cycle(mean_series, true_periods)["periods_to_use"]
    if not periods:
        periods, fallback = [24], True
    cd_auto, _ = decide_channel_dependence(compute_correlation_metrics(train_raw))
    return {"candidates": [int(p) for p in candidates], "true_periods": [int(p) for p in true_periods],
            "periods": [int(p) for p in periods], "period_fallback": fallback, "channel_dependence_auto": bool(cd_auto)}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CYCLENET_BACKBONES = ("linear", "mlp", "cnn", "sor")
TIER1_TRAINED = ["linear", "dlinear", "cyclenet_fixed"] + \
    [f"cyclenet_pro_{b}" for b in CYCLENET_BACKBONES] + [f"cyclenet_pro_{b}_ci" for b in CYCLENET_BACKBONES]
UNTRAINED = ["repeat_last", "seasonal_naive"]
TRANSFORM_MODELS = [f"tf_{t}" for t in ALL_TRANSFORMS if t not in ("identity", "dlinear")]
ALL_MODELS = UNTRAINED + TIER1_TRAINED + ["patchtst"] + TRANSFORM_MODELS


def model_defaults(name: str, dataset: str) -> Dict[str, object]:
    """Fixed (non-tuned) training settings per model; LR is tuned unless fixed here."""
    if name == "patchtst":
        big = dataset in ("electricity", "traffic", "solar")
        return {"lr": 1e-4, "batch_size": 32 if big else 128, "max_epochs": 100, "patience": 10, "tune_lr": False}
    return {"batch_size": 32, "max_epochs": 30, "patience": 5, "tune_lr": True}


def build_model(name: str, bundle: Bundle, seq_len: int, pred_len: int) -> Tuple[nn.Module, Dict[str, object]]:
    spec = bundle.spec
    c = spec.channels
    info: Dict[str, object] = {}
    if name == "repeat_last":
        return RepeatLast(pred_len), info
    if name == "seasonal_naive":
        info["period"] = spec.daily_period
        return SeasonalNaive(pred_len, spec.daily_period), info
    if name == "linear":
        return _NoCycle(TransformForecaster("identity", seq_len, pred_len, c)), info
    if name == "dlinear":
        return _NoCycle(TransformForecaster("dlinear", seq_len, pred_len, c)), info
    if name.startswith("tf_"):
        return _NoCycle(TransformForecaster(name[3:], seq_len, pred_len, c)), info
    if name == "cyclenet_fixed":
        info.update(periods=[spec.cycle_len], channel_dependence=False,
                    model_notes="CycleNet-Pro code with the CycleNet-paper cycle length, linear backbone, no channel mixing")
        return _WithCycle(CycleNetPro(seq_len, pred_len, c, [spec.cycle_len], use_channel_dependence=False,
                                      model_type="linear")), info
    if name.startswith("cyclenet_pro_"):
        rest = name[len("cyclenet_pro_"):]
        force_ci = rest.endswith("_ci")
        backbone = rest[:-3] if force_ci else rest
        if backbone not in CYCLENET_BACKBONES:
            raise ValueError(f"Unknown CycleNet-Pro backbone in {name}")
        dec = cyclenet_pro_decisions(bundle, seq_len)
        cd = False if force_ci else dec["channel_dependence_auto"]
        info.update(dec, channel_dependence=cd)
        return _WithCycle(CycleNetPro(seq_len, pred_len, c, dec["periods"], use_channel_dependence=cd,
                                      model_type=backbone)), info
    if name == "patchtst":
        cfg = PATCHTST_SMALL if spec.name in ("ETTh1", "ETTh2") else PATCHTST_LARGE
        info.update(patchtst_config=cfg, model_notes="PatchTST/42 re-implementation without residual attention")
        m = PatchTST(seq_len, pred_len, c, **cfg)
        m.trainable = True
        return m, info
    raise ValueError(f"Unknown model '{name}'. Known: {ALL_MODELS}")


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

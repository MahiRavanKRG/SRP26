"""P1 gate tests for the v2 harness (protocol §3a, §5)."""

import math

import numpy as np
import pytest
import torch

from srp26.bench.data import DATASETS, build_bundle
from srp26.bench.env import seed_everything
from srp26.bench.metrics import StreamingEvaluator, band_masks, seasonal_naive


def _evaluate(pred, true, dt_hours, snaive=None):
    ev = StreamingEvaluator(true.shape[1], dt_hours, torch.device("cpu"))
    for i in range(0, true.shape[0], 7):  # odd batch size exercises streaming
        ev.update(pred[i:i + 7], true[i:i + 7], None if snaive is None else snaive[i:i + 7])
    return ev.result()


# ---------------------------------------------------------------- metrics

def test_parseval_bands_sum_to_mse():
    g = torch.Generator().manual_seed(0)
    true = torch.randn(50, 96, 3, generator=g)
    pred = torch.randn(50, 96, 3, generator=g)
    r = _evaluate(pred, true, 1.0)
    assert math.isclose(r["mse"], torch.mean((pred.double() - true.double()) ** 2).item(), rel_tol=1e-12)
    total = sum(r[f"{b}_mse"] for b in ("level", "slow", "daily", "fast"))
    assert abs(total - r["mse"]) < 1e-8
    assert r["parseval_abs_error"] < 1e-8


def test_perfect_forecast_has_zero_band_error():
    true = torch.randn(10, 192, 2)
    r = _evaluate(true.clone(), true, 0.25)
    assert r["mse"] == 0.0
    assert all(r[f"{b}_mse"] == 0.0 for b in ("level", "slow", "daily", "fast"))


def test_mse_optimal_smooth_forecast_has_fast_fvu_one():
    t = torch.arange(96, dtype=torch.float64)
    signal = torch.sin(2 * math.pi * t / 24)[None, :, None].repeat(400, 1, 1)
    g = torch.Generator().manual_seed(1)
    true = signal + 0.3 * torch.randn(signal.shape, generator=g, dtype=torch.float64)
    r = _evaluate(signal, true, 1.0)
    assert abs(r["fast_fvu"] - 1.0) < 0.02  # predicting nothing in an unpredictable band -> FVU = 1


@pytest.mark.parametrize("name", list(DATASETS))
def test_24h_cycle_lands_in_daily_band(name):
    spec = DATASETS[name]
    horizon = 720
    if horizon * spec.dt_hours < 24:
        pytest.skip("window shorter than one day")
    freqs = torch.fft.rfftfreq(horizon)
    k = int(torch.argmin(torch.abs(freqs - spec.dt_hours / 24.0)))
    assert band_masks(horizon, spec.dt_hours)["daily"][k]


def test_seasonal_naive_has_zero_skill_against_itself():
    x = torch.randn(20, 336, 2)
    true = torch.randn(20, 96, 2)
    sn = seasonal_naive(x, 96, 24)
    r = _evaluate(sn, true, 1.0, snaive=sn)
    assert abs(r["skill_mse"]) < 1e-12


def test_seasonal_naive_repeats_last_period():
    x = torch.arange(48, dtype=torch.float32).view(1, 48, 1)
    out = seasonal_naive(x, 30, 24)
    assert torch.equal(out[0, :24, 0], x[0, 24:, 0])
    assert torch.equal(out[0, 24:, 0], x[0, 24:30, 0])


@pytest.mark.parametrize("wavelet", ["haar", "db4"])
def test_chunked_dwt_matches_unchunked(wavelet):
    import ptwt
    from srp26.transforms import DWTTransform

    x = torch.randn(9, 336, 7, generator=torch.Generator().manual_seed(3))
    t = DWTTransform(336, wavelet=wavelet, level=2)
    ref = torch.cat(ptwt.wavedec(x.permute(0, 2, 1).contiguous(), t.wavelet, level=2), dim=-1)[..., :336]
    t.max_rows = 5  # force several uneven chunks
    assert torch.equal(t.encode(x), ref)


# ---------------------------------------------------------------- data

def test_etth1_split_sizes_and_no_leakage():
    b = build_bundle("ETTh1", 336, 96, 32, generator=seed_everything(0))
    assert (b.split["train_end"], b.split["val_end"], b.split["test_end"]) == (8640, 11520, 14400)
    # first test target starts exactly at val_end: the lookback prefix comes from validation, targets do not
    _, _, cyc = b.test_loader.dataset[0]
    assert int(cyc) + 336 == b.split["val_end"]
    # scaler statistics are those of the training split only
    assert np.allclose(b.train_scaled.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(b.train_scaled.std(axis=0), 1.0, atol=1e-4)


def test_test_loader_keeps_last_partial_batch():
    b = build_bundle("ETTh1", 336, 96, 32, generator=seed_everything(0))
    assert b.test_loader.drop_last is False
    assert sum(y.shape[0] for _, y, _ in b.test_loader) == 2880 - 96 + 1


# ---------------------------------------------------------------- reproducibility (sanity, CPU)

def test_same_seed_same_result_cpu():
    from srp26.bench.models import build_model
    from srp26.bench.train import fit

    def once():
        g = seed_everything(2024)
        b = build_bundle("ETTh1", 336, 96, 32, generator=g)
        seed_everything(2024)
        b.train_loader.generator.manual_seed(2024)
        m, _ = build_model("cyclenet_pro_linear", b, 336, 96)
        return fit(m, b.train_loader, b.val_loader, torch.device("cpu"), lr=1e-3, max_epochs=1, patience=1)["val_mse"]

    assert once() == once()

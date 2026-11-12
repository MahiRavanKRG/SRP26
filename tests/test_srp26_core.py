"""Core unit tests for the ``srp26`` package, lightweight baselines, and cluster config hygiene."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from srp26 import (
    CycleNet,
    DLinear,
    StandardTrainScaler,
    TimeSeriesSlidingWindowDataset,
    compute_spectral_diagnostics,
    ensure_hf_cache,
    evaluate_all_spectral_diagnostics,
    extract_and_extrapolate_fourier_base_torch,
    extract_fourier_base_1d,
    get_canonical_splits,
    parseval_band_decomposition,
    set_seed,
)
from models.fcmath import FCMath


def test_parseval_band_decomposition_exact_identity():
    """Verify MSE_total == MSE_low + MSE_mid + MSE_high up to floating-point precision."""
    set_seed(2024)
    y_true = torch.randn(4, 96, 7)
    y_pred = y_true + 0.25 * torch.randn(4, 96, 7)

    result = parseval_band_decomposition(y_true, y_pred)
    reconstructed_mse = result["mse_low"] + result["mse_mid"] + result["mse_high"]
    assert abs(result["total_mse"] - reconstructed_mse) < 1e-6
    assert result["parseval_additive_error"] < 1e-6

    full = evaluate_all_spectral_diagnostics(y_true, y_pred)
    assert full["log_spectral_distance"] >= 0.0
    assert 0.0 <= full["failure_rate_pct"] <= 100.0


def test_compute_spectral_diagnostics_shapes():
    """Ensure rFFT subband diagnostics return valid positive FVU/LSD for 2D and 3D inputs."""
    set_seed(2024)
    y_true = torch.randn(2, 96, 3)
    y_pred = torch.randn(2, 96, 3)
    diag = compute_spectral_diagnostics(y_true, y_pred)
    for key in ("fvu_low", "fvu_mid", "fvu_high", "lsd_db"):
        assert key in diag
        assert np.isfinite(diag[key])


def test_fourier_base_torch_exact_sinusoid_recovery():
    """Top-k Fourier extrapolation should nearly zero out residual for pure periodic signals."""
    seq_len, pred_len = 96, 48
    t_all = torch.arange(seq_len + pred_len, dtype=torch.float32)
    freq = 4.0 / seq_len  # Exact integer cycle within lookback window
    signal = torch.sin(2.0 * np.pi * freq * t_all).view(1, -1, 1).expand(2, -1, 3)

    x = signal[:, :seq_len, :]
    y_true = signal[:, seq_len:, :]

    x_base, y_base, x_res = extract_and_extrapolate_fourier_base_torch(x, pred_len=pred_len, top_k=1)
    assert x_base.shape == (2, seq_len, 3)
    assert y_base.shape == (2, pred_len, 3)
    assert x_res.abs().max().item() < 1e-5
    assert torch.mean((y_base - y_true) ** 2).item() < 1e-5


def test_fourier_base_1d_and_fcmath():
    """Check 1D linear-detrended Fourier base helper and FCMath fallback baseline."""
    t = np.arange(144, dtype=np.float32)
    series = 0.02 * t[:96] + np.cos(2.0 * np.pi * 3.0 * t[:96] / 96.0)
    x_base, y_base = extract_fourier_base_1d(series, pred_len=48, top_k=2)
    assert x_base.shape == (96,)
    assert y_base.shape == (48,)
    assert np.mean((series - x_base) ** 2) < 5e-4

    fc = FCMath(top_k=2)
    x_b, y_b = fc.extract_and_extrapolate_fourier_base(series[:, None], pred_len=48)
    assert x_b.shape == (96, 1)
    assert y_b.shape == (48, 1)


def test_legacy_facade_imports():
    """Verify root and cluster compatibility facades resolve cleanly to ``srp26``."""
    import dataloader
    import diagnostics
    import metrics

    assert callable(metrics.band_wise_error_decomposition)
    assert callable(metrics.psd_comparison)
    assert callable(dataloader.create_dataloader)
    assert callable(diagnostics.rolling_error_tracking)


def test_standard_scaler_strict_fit_and_canonical_splits():
    """Verify zero-variance channel protection, inverse transform, and canonical splits."""
    train = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    scaler = StandardTrainScaler().fit(train)
    assert scaler.std[0, 1] == 1.0  # Constant channel does not divide by zero
    scaled = scaler.transform(train)
    assert np.allclose(scaler.inverse_transform(scaled), train, atol=1e-5)

    t_end, v_end, test_end, _ = get_canonical_splits("ETTh1.csv", 17420)
    assert (t_end, v_end, test_end) == (8640, 11520, 14400)

    t2, v2, test2, _ = get_canonical_splits("traffic.csv", 1000)
    assert (t2, v2, test2) == (700, 800, 1000)

    ds = TimeSeriesSlidingWindowDataset(np.zeros((200, 4), dtype=np.float32), seq_len=96, pred_len=24, return_cycle=True)
    x, y, idx = ds[3]
    assert x.shape == (96, 4)
    assert y.shape == (24, 4)
    assert int(idx.item()) == 3


def test_cyclenet_and_dlinear_forward_shapes():
    """Verify forward pass shapes of lightweight CycleNet and DLinear models."""
    set_seed(2024)
    x = torch.randn(4, 96, 7)
    for model in (CycleNet(96, 24, 7, cycle_len=24), DLinear(96, 24, 7)):
        y = model(x)
        assert y.shape == (4, 24, 7)
        assert torch.isfinite(y).all()


def test_ensure_hf_cache_respects_env(monkeypatch, tmp_path):
    """Ensure ``ensure_hf_cache`` never overwrites user-configured HF_HOME."""
    custom = str(tmp_path / "hf")
    monkeypatch.setenv("HF_HOME", custom)
    assert ensure_hf_cache() == custom


def test_bigmama_remote_requires_env_without_hardcoded_secrets(monkeypatch, tmp_path):
    """Cluster SSH helper must fail cleanly without fallback credentials when env/.env are unset."""
    monkeypatch.delenv("BIGMAMA_HOST", raising=False)
    monkeypatch.delenv("BIGMAMA_USER", raising=False)
    monkeypatch.delenv("BIGMAMA_PASS", raising=False)

    # Point module .env lookup temporarily away from any developer .env
    from bigmama_deployment import remote_exec

    monkeypatch.setattr(remote_exec, "load_env_config", lambda env_path=None: {})
    with pytest.raises(RuntimeError, match="Missing cluster configuration"):
        remote_exec.resolve_ssh_credentials(interactive=False)


def test_sync_and_remote_cli_syntax():
    """Compile-check core deployment scripts cleanly without invoking SSH."""
    for script in ("bigmama_deployment/remote_exec.py", "bigmama_deployment/sync_to_bigmama.py"):
        res = subprocess.run([sys.executable, "-m", "py_compile", script], check=False)
        assert res.returncode == 0

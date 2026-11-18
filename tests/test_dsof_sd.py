"""Unit and integration tests for DSOF-SD (models/dsof_sd.py and unified rFFT LSD metrics)."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dsof_sd import (
    DSOFSDConfig,
    DSOFSDTrainer,
    DSOFSpectralLoss,
    FourierBaseTeacher,
    TeacherTargetCache,
    TeacherTargetDataset,
    build_dsof_student,
    check_hypothesis,
    discover_dataset_periods,
)
from srp26.data import TimeSeriesSlidingWindowDataset
from srp26.spectral import dsof_band_metrics_torch, parseval_band_decomposition
from srp26.utils import set_seed


def test_unified_dsof_metrics_match_numpy_parseval_exactness():
    """PyTorch ``dsof_band_metrics_torch`` must satisfy Parseval identity and match NumPy."""
    set_seed(2024)
    y_true = torch.randn(4, 96, 5)
    y_pred = y_true + 0.3 * torch.randn(4, 96, 5)

    np_diag = parseval_band_decomposition(y_true, y_pred)
    pt_diag = dsof_band_metrics_torch(y_true, y_pred)

    band_sum = pt_diag["mse"]["low"] + pt_diag["mse"]["mid"] + pt_diag["mse"]["high"]
    assert torch.allclose(pt_diag["total_mse"], band_sum, atol=1e-6)
    assert abs(float(pt_diag["mse"]["low"].item()) - np_diag["mse_low"]) < 1e-5
    assert abs(float(pt_diag["mse"]["mid"].item()) - np_diag["mse_mid"]) < 1e-5
    assert abs(float(pt_diag["mse"]["high"].item()) - np_diag["mse_high"]) < 1e-5
    assert abs(float(pt_diag["fvu"]["mid"].item()) - np_diag["fvu_mid"]) < 1e-4


def test_dsof_loss_is_literally_gated_squared_rfft_lsd():
    """Verify ``L_spec`` identically equals sum_b g_b * LSD_b(student, teacher)^2."""
    set_seed(2024)
    cfg = DSOFSDConfig(gate_mode="hard")
    loss_fn = DSOFSpectralLoss(cfg)
    loss_fn.gate_vector.copy_(torch.tensor([1.0, 0.5, 0.0]))

    y_true = torch.randn(2, 48, 3)
    y_teacher = torch.randn(2, 48, 3)
    y_student = torch.randn(2, 48, 3, requires_grad=True)

    total, info = loss_fn(y_student, y_true, y_teacher, lam_spec=0.25)
    metrics_st = dsof_band_metrics_torch(y_teacher, y_student, eps=cfg.eps)
    expected_lspec = (
        1.0 * metrics_st["lsd"]["low"] ** 2
        + 0.5 * metrics_st["lsd"]["mid"] ** 2
        + 0.0 * metrics_st["lsd"]["high"] ** 2
    )
    assert abs(info["loss_spec"] - float(expected_lspec.item())) < 1e-5
    total.backward()
    assert y_student.grad is not None and torch.isfinite(y_student.grad).all()


def test_calibration_gate_rejects_blurry_teacher_high_band():
    """Hard trust gate must disable high-band distillation when teacher has high-freq error."""
    set_seed(2024)
    horizon = 96
    t = torch.arange(horizon, dtype=torch.float32).view(1, horizon, 1)
    low_sig = torch.sin(2.0 * np.pi * 2.0 * t / horizon)
    high_sig = 0.8 * torch.sin(2.0 * np.pi * 43.0 * t / horizon)  # freq ~ 0.448 >= 0.4 (high band)
    y_true = (low_sig + high_sig).expand(6, -1, 4).clone()

    # Teacher nails low band cleanly, blurs out high band completely
    y_teacher = low_sig.expand(6, -1, 4).clone()
    # Warm-start student has noisy low band, good high band
    y_student_warm = (0.3 * low_sig + high_sig).expand(6, -1, 4).clone()

    loss_fn = DSOFSpectralLoss(DSOFSDConfig(gate_mode="hard"))
    gates = loss_fn.calibrate_band_gates(y_true, y_teacher, y_student_warm)

    assert gates["low"] == 1.0
    assert gates["high"] == 0.0


def test_teacher_cache_roundtrip(tmp_path):
    """Verify Phase 0 offline cache generation, save, and load preserve exact tensors."""
    set_seed(2024)
    raw = np.sin(np.linspace(0, 20 * np.pi, 300, dtype=np.float32))[:, None] * np.ones((1, 3), dtype=np.float32)
    ds = TimeSeriesSlidingWindowDataset(raw, seq_len=96, pred_len=48, start_idx_in_full=0, return_cycle=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    teacher = FourierBaseTeacher(pred_len=48, top_k=2)
    cached = TeacherTargetCache.build_from_loader(loader, teacher)
    assert len(cached) == len(ds)
    assert cached.y_teacher.shape == cached.y.shape

    cache_file = tmp_path / "cache.pt"
    TeacherTargetCache.save(cache_file, cached, cached, cached, metadata={"teacher": "fourier"})
    tr, va, te, meta = TeacherTargetCache.load(cache_file)
    assert meta["teacher"] == "fourier"
    assert torch.allclose(tr.y_teacher, cached.y_teacher)
    assert torch.equal(te.cycle_index, cached.cycle_index)


def test_end_to_end_dsof_sd_trainer_with_sor_student():
    """Run Phases 1-3 end-to-end on synthetic periodic+residual signal with CycleNetPro(sor)."""
    set_seed(2024)
    n_total, seq_len, pred_len, channels = 360, 96, 48, 3
    t = torch.arange(n_total, dtype=torch.float32).unsqueeze(-1)
    data = (torch.sin(2.0 * np.pi * t / 24.0) + 0.1 * torch.randn(n_total, channels)).numpy()

    def make_loader(sl: np.ndarray, start_idx: int, shuffle: bool) -> DataLoader:
        ds_base = TimeSeriesSlidingWindowDataset(sl, seq_len=seq_len, pred_len=pred_len, start_idx_in_full=start_idx, return_cycle=True)
        base_loader = DataLoader(ds_base, batch_size=16, shuffle=False)
        cached = TeacherTargetCache.build_from_loader(base_loader, FourierBaseTeacher(pred_len=pred_len, top_k=2))
        return DataLoader(cached, batch_size=16, shuffle=shuffle)

    train_loader = make_loader(data[:200], 0, shuffle=True)
    val_loader = make_loader(data[200 - seq_len : 280], 200 - seq_len, shuffle=False)
    test_loader = make_loader(data[280 - seq_len :], 280 - seq_len, shuffle=False)

    student = build_dsof_student(seq_len=seq_len, pred_len=pred_len, channels=channels, student_type="sor", periods=[24])
    cfg = DSOFSDConfig(warm_epochs=2, distill_epochs=3, patience=3, lr=1e-3, lambda_max=0.2)
    trainer = DSOFSDTrainer(student=student, config=cfg, device=torch.device("cpu"))
    report = trainer.fit_and_evaluate(train_loader, val_loader, test_loader)

    for key in ("teacher", "warm_student", "distilled_student", "gates", "hypothesis"):
        assert key in report
    assert np.isfinite(report["distilled_student"]["mse"])
    assert report["hypothesis"]["status"] in {"supported", "refuted", "no_transfer"}


def test_check_hypothesis_verdicts():
    """Verify supported, refuted, and negative-gate hypothesis verdicts."""
    base = {
        "gates": {"low": 1.0, "mid": 0.0, "high": 0.0},
        "warm_student": {"mse": 0.50, "bands": {"low": {"lsd": 0.80}, "mid": {"lsd": 0.5}, "high": {"lsd": 0.4}}},
        "distilled_student": {"mse": 0.48, "bands": {"low": {"lsd": 0.65}, "mid": {"lsd": 0.5}, "high": {"lsd": 0.4}}},
    }
    assert check_hypothesis(base)["passed"] is True
    assert check_hypothesis(base)["status"] == "supported"

    base["distilled_student"]["mse"] = 0.55
    assert check_hypothesis(base)["passed"] is False
    assert check_hypothesis(base)["status"] == "refuted"

    base["gates"] = {"low": 0.0, "mid": 0.0, "high": 0.0}
    assert check_hypothesis(base)["status"] == "no_transfer"


def test_discover_dataset_periods_multichannel():
    """Automated period discovery + channel-dependence wiring on 2D array."""
    t = np.arange(600, dtype=np.float32)[:, None]
    sig = np.sin(2.0 * np.pi * t / 24.0)
    data = np.concatenate([sig, sig * 0.9, sig * -0.8], axis=1)
    periods, use_cd = discover_dataset_periods(data, seq_len=336)
    assert isinstance(periods, list) and len(periods) >= 1
    assert isinstance(use_cd, bool) and use_cd is True

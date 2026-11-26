"""Stage-2 tests: foundation-model evaluation windows, Fourier references, teacher cache
alignment and the DSOF-SD lambda=0 control (CPU, no model downloads)."""

import json
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from models.cyclenet.cyclenet_pro_Full import CycleNetPro
from srp26.bench import dsof, foundation, grid, report
from srp26.bench.data import DATASETS, build_bundle
from srp26.bench.env import DATA_DIR, seed_everything
from srp26.fourier import extract_and_extrapolate_fourier_base_torch

needs_data = pytest.mark.skipif(not (DATA_DIR / DATASETS["ETTh1"].filename).exists(),
                                reason="ETTh1.csv not present (set SRP26_DATA_DIR)")


# ---------------------------------------------------------------- evaluation windows

def test_stride_h_windows_are_non_overlapping_and_cover_test():
    idx = foundation.stride_h_indices(2785, 96)
    assert idx[0] == 0 and np.all(np.diff(idx) == 96)
    assert idx[-1] < 2785 <= idx[-1] + 96  # last start is the last one that fits; nothing left uncovered but a tail < H
    assert len(idx) == math.ceil(2785 / 96)
    assert foundation.stride_h_indices(2785, 96, max_windows=3).tolist() == [0, 96, 192]
    assert foundation.stride_h_indices(10, 3).tolist() == [0, 3, 6, 9]


# ---------------------------------------------------------------- Fourier references

def _toy(b=4, l=336, c=3, level=5.0):
    t = torch.arange(l, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    x = level + torch.sin(2 * math.pi * t / 24)[None, :, None] + 0.1 * torch.randn(b, l, c, generator=g)
    return x


def test_fourier_base_includes_lookback_mean():
    x = _toy()
    pred = foundation.fourier_base_forecast(x, 96)
    _, y_base, _ = extract_and_extrapolate_fourier_base_torch(x, 96, top_k=3)
    assert torch.allclose(pred, x.mean(1, keepdim=True) + y_base)
    assert abs(pred.mean().item() - 5.0) < 0.05  # y_base alone would be centred on 0
    assert abs(y_base.mean().item()) < 0.05


def test_fourier_decoupling_adds_y_base_back():
    x = _toy()
    seen = {}

    def model(inp):
        seen["x"] = inp.clone()
        return inp[:, -1:, :].repeat(1, 96, 1)  # repeat-last on whatever it is given

    pred = foundation.fourier_decoupled(model, x, 96)
    _, y_base, x_res = extract_and_extrapolate_fourier_base_torch(x, 96, top_k=3)
    assert torch.allclose(seen["x"], x_res)  # the model only sees the residual
    assert torch.allclose(pred, x_res[:, -1:, :].repeat(1, 96, 1) + y_base)
    zero = foundation.fourier_decoupled(lambda inp: torch.zeros(inp.shape[0], 96, inp.shape[2]), x, 96)
    assert torch.allclose(zero, y_base)


def test_series_reshape_roundtrip():
    x = torch.randn(5, 7, 3)
    s = foundation._to_series(x)
    assert torch.equal(s[1 * 3 + 2], x[1, :, 2])  # window-major, channel-minor
    assert torch.equal(foundation._from_series(s, 5, 3), x)
    assert torch.equal(foundation._from_series(s[:, :4], 5, 3), x[:, :4, :])


# ---------------------------------------------------------------- teacher cache alignment

@needs_data
def test_teacher_cache_is_aligned_by_index(tmp_path):
    bundle = build_bundle("ETTh1", 336, 96, 32, generator=seed_everything(0))
    payload = dsof.build_teacher_cache(bundle, "fourier_base", 96, batch=512)
    path = dsof.cache_path("fourier_base", "ETTh1", 96, tmp_path)
    dsof.save_cache(payload, path)
    loaded = dsof.load_or_build_cache(bundle, "fourier_base", 96, cache_dir=tmp_path)  # verifies alignment
    for split, ds in dsof._split_datasets(bundle).items():
        assert loaded["metadata"]["n_windows"][split] == len(ds)
        aligned = dsof.TeacherAlignedDataset(ds, loaded[split]["y_teacher"])
        for i in (0, 1, len(ds) // 3, len(ds) - 1):
            x, y, yt, cyc = aligned[i]
            x0, y0, c0 = ds[i]
            assert torch.equal(x, x0) and torch.equal(y, y0) and int(cyc) == int(c0)
            assert torch.allclose(yt, foundation.fourier_base_forecast(x0[None], 96)[0], atol=1e-5)
    # a shifted cache must be refused
    bad = {k: (dict(v) if isinstance(v, dict) else v) for k, v in loaded.items()}
    bad["test"]["last_obs"] = torch.roll(bad["test"]["last_obs"], 1, dims=0)
    with pytest.raises(RuntimeError, match="misaligned"):
        dsof.verify_cache(bad, bundle, "fourier_base", 96)


# ---------------------------------------------------------------- DSOF control

def _tiny_loaders(seed, garbage_teacher=True, n=96, l=48, h=12, c=2):
    g = torch.Generator().manual_seed(123)
    t = torch.arange(n + l + h, dtype=torch.float32)
    series = torch.sin(2 * math.pi * t / 12)[:, None].repeat(1, c) + 0.1 * torch.randn(len(t), c, generator=g)
    x = torch.stack([series[i:i + l] for i in range(n)])
    y = torch.stack([series[i + l:i + l + h] for i in range(n)])
    yt = 10 * torch.randn(y.shape, generator=g) if garbage_teacher else y.clone()
    cyc = torch.arange(n)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return x[i], y[i], yt[i], cyc[i]

    gen = seed_everything(seed)
    return (DataLoader(DS(), batch_size=16, shuffle=True, drop_last=True, generator=gen),
            DataLoader(DS(), batch_size=32), DataLoader(DS(), batch_size=32))


def _fit(gate, seed=7):
    train, val, test = _tiny_loaders(seed)
    seed_everything(seed)
    net = CycleNetPro(48, 12, 2, [12], use_channel_dependence=False, model_type="linear")
    cfg = dsof.make_config(gate, epochs_override=(1, 2))
    cfg.patience = 100
    trainer = dsof._trainer_cls()(net, cfg, torch.device("cpu"))
    return trainer.fit_and_evaluate(train, val, test), net, cfg


def test_dsof_control_uses_lambda_zero_and_full_schedule():
    rep, _, cfg = _fit("control")
    assert cfg.lambda_max == 0.0 and cfg.skip_degenerate_phase2 is False
    assert cfg.warm_epochs == 1 and cfg.distill_epochs == 2
    assert rep["phase2_degenerate"] is False
    assert len(rep["history"]) == 2  # Phase 2 ran the full schedule
    assert all(e["lam_spec"] == 0.0 for e in rep["history"])


def test_zero_gate_run_trains_like_control_instead_of_skipping():
    rep_c, net_c, _ = _fit("control")
    rep_h, net_h, cfg_h = _fit("hard")  # garbage teacher -> every hard gate closes
    assert cfg_h.lambda_max == 0.2
    assert all(g == 0.0 for g in rep_h["gates"].values())
    assert len(rep_h["history"]) == 2 and rep_h["phase2_degenerate"] is False
    for a, b in zip(net_c.state_dict().values(), net_h.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)  # same data order, zero-weighted spectral term


def test_make_config_defaults_match_method():
    from models.dsof_sd import DSOFSDConfig
    d = DSOFSDConfig()
    cfg = dsof.make_config("benefit")
    assert (cfg.lr, cfg.warm_epochs, cfg.distill_epochs, cfg.patience, cfg.lambda_max) == \
        (d.lr, d.warm_epochs, d.distill_epochs, d.patience, d.lambda_max) == (3e-4, 5, 25, 7, 0.2)
    assert cfg.gate_mode == "benefit"
    with pytest.raises(ValueError):
        dsof.make_config("nope")


# ---------------------------------------------------------------- end-to-end (no downloads)

@needs_data
def test_foundation_fourier_base_run_and_rescore(tmp_path, monkeypatch):
    out = foundation.run_one("t", "ETTh1", 96, "fourier_base", seed=2024, max_windows=3, out_root=tmp_path)
    r = json.loads(out.read_text())
    assert r["phase"] == "foundation" and r["model"] == "fourier_base"
    assert r["eval_windows"]["indices"] == [0, 96, 192] and r["eval_windows"]["stride"] == 96
    wmse = np.load(out.with_suffix(".wmse.npy"))
    wmae = np.load(out.with_suffix(".wmae.npy"))
    assert len(wmse) == len(wmae) == 3
    assert math.isclose(float(wmse.astype(np.float64).mean()), r["metrics"]["mse"], rel_tol=1e-5)
    assert "skill_mse" in r["metrics"] and r["provenance"]["commit"]

    # re-scoring a trained run on the same windows uses its stride-1 per-window losses
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    run_dir = tmp_path / "tr" / "final" / "ETTh1" / "H96" / "linear"
    run_dir.mkdir(parents=True)
    n_test = r["eval_windows"]["n_test_stride1"]
    w = np.arange(n_test, dtype=np.float32)
    np.save(run_dir / "s2024.wmse.npy", w)
    np.save(run_dir / "s2024.wmae.npy", w / 10)
    (run_dir / "s2024.json").write_text(json.dumps({"model": "linear", "seed": 2024, "metrics": {"mse": 1.0}}))
    df = report.rescore_on_windows("tr", "ETTh1", 96, [0, 96, 192], n_test)
    assert df.loc[0, "mse"] == pytest.approx(96.0) and df.loc[0, "mae"] == pytest.approx(9.6)


def test_stage2_job_lists():
    f = grid.stage2_foundation_jobs()
    assert len(f) == 8 * 4 * 3 * 2 + 8 * 2 + 8 * 4
    assert all(j["horizon"] == 96 for j in f if j["model"] == "ttm")
    assert not any(j["fourier"] for j in f if j["model"] == "fourier_base")
    d = grid.stage2_dsof_jobs()
    assert len(d) == 2 * 4 * 9 * 3
    assert {(j["student"], j["gate"]) for j in d} == set(grid.STAGE2_DSOF_VARIANTS)


def test_ttm_refuses_other_horizons():
    with pytest.raises(ValueError, match="H=96"):
        foundation.run_one("t", "ETTh1", 192, "ttm")
    with pytest.raises(ValueError):
        foundation.run_one("t", "ETTh1", 96, "fourier_base", fourier=True)

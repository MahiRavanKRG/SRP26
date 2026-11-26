"""Stage 2a: zero-shot foundation models on the v2 bench (protocol §4 Tier 2).

Single run:
    uv run python -m srp26.bench.foundation --set local --dataset ETTh1 --horizon 96 --model timesfm [--fourier]

Job list (also reachable through ``srp26.bench.run --jobs``, which dispatches on ``phase``):
    uv run python -m srp26.bench.foundation --set bigmama --jobs configs/v2/jobs_stage2_foundation.jsonl --shard 0 --n-shards 8

Rules (protocol §8 "Stage 2 implementation record"):
- Data comes from ``build_bundle`` (train-only scaler, canonical splits, checksum
  verification); inputs are the scaled L=336 lookbacks; metrics in scaled space.
- Evaluation windows are the NON-OVERLAPPING test windows 0, H, 2H, ... of the
  v2 test dataset (stride = H), which covers the whole test period once. The
  indices are stored in the result so trained models can be re-scored on the
  same windows (``srp26.bench.report --foundation-compare``).
- Metrics come from ``StreamingEvaluator`` with the seasonal naive of the
  dataset's daily period: the identical metric set used for trained models.
- ``--fourier`` = Fourier decoupling (FCMath): the model forecasts the residual
  ``x_res`` of the top-3 rFFT base, and the extrapolated base ``y_base`` is added.

Results are immutable: an existing result file is skipped, never overwritten.
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from srp26.bench import HARNESS_VERSION
from srp26.bench.data import build_bundle
from srp26.bench.env import REPO_ROOT, get_device, provenance, seed_everything
from srp26.bench.metrics import StreamingEvaluator, seasonal_naive
from srp26.fourier import extract_and_extrapolate_fourier_base_torch

SEQ_LEN = 336
FOURIER_TOP_K = 3
CHRONOS_T5_SAMPLES = 20
TTM_HORIZON = 96
FOUNDATION_MODELS = ("timesfm", "chronos_bolt", "chronos_t5", "ttm", "fourier_base")
DECOUPLABLE = ("timesfm", "chronos_bolt", "chronos_t5", "ttm")
HF_IDS = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "chronos_bolt": "amazon/chronos-bolt-base",
    "chronos_t5": "amazon/chronos-t5-tiny",
    "ttm": "ibm-granite/granite-timeseries-ttm-r1",
}
# Series (= window x channel) per model call; tuned for a 6 GB laptop GPU, larger is fine on the cluster.
DEFAULT_SERIES_BATCH = {"timesfm": 256, "chronos_bolt": 128, "chronos_t5": 64, "ttm": 256, "fourier_base": 4096}
MAX_SERIES_PER_CHUNK = 4096  # windows are processed in chunks of at most this many series


# --------------------------------------------------------------------------
# Window selection and Fourier helpers (pure functions, unit-tested)
# --------------------------------------------------------------------------


def stride_h_indices(n_test: int, horizon: int, max_windows: Optional[int] = None) -> np.ndarray:
    """Non-overlapping test windows: dataset indices 0, H, 2H, ... (< n_test)."""
    idx = np.arange(0, n_test, horizon, dtype=np.int64)
    return idx[:max_windows] if max_windows else idx


def fourier_base_forecast(x: torch.Tensor, pred_len: int, top_k: int = FOURIER_TOP_K) -> torch.Tensor:
    """Zero-parameter reference: lookback mean + top-k rFFT harmonic extrapolation.

    ``extract_and_extrapolate_fourier_base_torch`` drops the DC bin, so ``y_base``
    has zero mean; the old benchmark scored ``y_base`` alone (mean omitted).
    """
    _, y_base, _ = extract_and_extrapolate_fourier_base_torch(x, pred_len=pred_len, top_k=top_k)
    return x.mean(dim=1, keepdim=True) + y_base


def fourier_decoupled(forecast_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, pred_len: int,
                      top_k: int = FOURIER_TOP_K) -> torch.Tensor:
    """FCMath decoupling: the model forecasts ``x_res`` (which keeps the mean), ``y_base`` is added back."""
    _, y_base, x_res = extract_and_extrapolate_fourier_base_torch(x, pred_len=pred_len, top_k=top_k)
    return forecast_fn(x_res) + y_base


def _to_series(x: torch.Tensor) -> torch.Tensor:
    """[B, L, C] -> [B*C, L] (window-major, channel-minor)."""
    b, l, c = x.shape
    return x.permute(0, 2, 1).reshape(b * c, l)


def _from_series(s: torch.Tensor, b: int, c: int) -> torch.Tensor:
    """[B*C, H] -> [B, H, C]."""
    return s.reshape(b, c, -1).permute(0, 2, 1).contiguous()


# --------------------------------------------------------------------------
# Forecasters: callable [B, L, C] (CPU float32) -> [B, H, C] (CPU float32)
# --------------------------------------------------------------------------


class Forecaster:
    def __init__(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor], info: Dict[str, object],
                 notes: List[str]):
        self.name, self.fn, self.info, self.notes = name, fn, info, notes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def _load_timesfm(horizon: int, series_batch: int) -> Forecaster:
    from models.dsof_sd import TimesFMTeacher

    teacher = TimesFMTeacher(seq_len=SEQ_LEN, pred_len=horizon, model_id=HF_IDS["timesfm"],
                             per_core_batch_size=series_batch)
    teacher._ensure_loaded()  # fail here (not mid-evaluation) if the model cannot be built
    fc = teacher._model.forecast_config
    info = {"hf_id": HF_IDS["timesfm"], "precision": "fp32", "series_batch": series_batch,
            "forecast_config": {"max_context": fc.max_context, "max_horizon": fc.max_horizon,
                                "normalize_inputs": fc.normalize_inputs,
                                "use_continuous_quantile_head": fc.use_continuous_quantile_head,
                                "force_flip_invariance": fc.force_flip_invariance,
                                "infer_is_positive": fc.infer_is_positive,
                                "fix_quantile_crossing": fc.fix_quantile_crossing},
            "point": "point output (median quantile, index 5)"}
    notes = [f"ForecastConfig of models/dsof_sd.py TimesFMTeacher; compiled max_context={fc.max_context} "
             f"(336 rounded up to the 32-step patch, left-padded and masked), max_horizon={fc.max_horizon}"]
    return Forecaster("timesfm", teacher, info, notes)


def _load_chronos_bolt(device: torch.device, series_batch: int, hf_id: str) -> Forecaster:
    from chronos import ChronosBoltPipeline

    pipe = ChronosBoltPipeline.from_pretrained(hf_id, device_map=str(device),
                                               torch_dtype=torch.bfloat16)
    quantiles = [float(q) for q in pipe.quantiles]
    if 0.5 not in quantiles:
        raise RuntimeError(f"Chronos-Bolt quantiles {quantiles} contain no 0.5 level.")
    q_med = quantiles.index(0.5)

    def fn(x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        series = _to_series(x)
        horizon = fn.horizon
        outs = []
        for i in range(0, series.shape[0], series_batch):
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "prediction length > 64" advisory; long horizons are intended
                q = pipe.predict(series[i:i + series_batch], prediction_length=horizon, limit_prediction_length=False)
            outs.append(q[:, q_med, :].float().cpu())
        return _from_series(torch.cat(outs), b, c)

    info = {"hf_id": hf_id, "precision": "bf16", "series_batch": series_batch,
            "quantiles": quantiles, "median_index": q_med, "native_prediction_length": int(pipe.model_prediction_length)}
    notes = [f"point = 0.5 quantile (index {q_med} of {quantiles}); horizons > "
             f"{int(pipe.model_prediction_length)} use the pipeline's own autoregressive quantile roll-out"]
    return Forecaster("chronos_bolt", fn, info, notes)


def _load_chronos_t5(device: torch.device, series_batch: int, seed: int) -> Forecaster:
    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained(HF_IDS["chronos_t5"], device_map=str(device), torch_dtype=torch.bfloat16)
    logging.getLogger("chronos").setLevel(logging.ERROR)  # silence the per-call "prediction length > 64" warning

    def fn(x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        series = _to_series(x)
        outs = []
        for i in range(0, series.shape[0], series_batch):
            with torch.no_grad():
                samples = pipe.predict(series[i:i + series_batch], prediction_length=fn.horizon,
                                       num_samples=CHRONOS_T5_SAMPLES, limit_prediction_length=False)
            outs.append(torch.quantile(samples.float(), 0.5, dim=1))  # [N, S, H] -> [N, H]
        return _from_series(torch.cat(outs), b, c)

    info = {"hf_id": HF_IDS["chronos_t5"], "precision": "bf16", "series_batch": series_batch,
            "num_samples": CHRONOS_T5_SAMPLES, "torch_seed": seed,
            "native_prediction_length": int(pipe.model.config.prediction_length)}
    notes = [f"point = median over {CHRONOS_T5_SAMPLES} samples (mean of the two middle samples); the old suite "
             "used num_samples=1 (a single random draw), not replicated",
             f"torch seeded with {seed} immediately before inference; sampling also depends on batch composition",
             "horizons > 64 use the pipeline's autoregressive roll-out (median sample appended to the context)"]
    return Forecaster("chronos_t5", fn, info, notes)


def _load_ttm(device: torch.device, horizon: int) -> Forecaster:
    if horizon != TTM_HORIZON:
        raise ValueError(f"TTM-R1 ({HF_IDS['ttm']}) only supports H={TTM_HORIZON}; refusing H={horizon}.")
    try:
        from tsfm_public import TinyTimeMixerForPrediction
    except ImportError as exc:
        raise RuntimeError("TTM unavailable: run TTM jobs with the separate env in envs/ttm "
                           "(granite-tsfm 0.3.12 needs torch>=2.10; the main v2 env pins torch 2.8).") from exc
    model = TinyTimeMixerForPrediction.from_pretrained(HF_IDS["ttm"]).to(device).eval()
    ctx = int(model.config.context_length)

    def fn(x: torch.Tensor) -> torch.Tensor:
        pad = ctx - x.shape[1]
        if pad < 0:
            raise ValueError(f"lookback {x.shape[1]} longer than TTM context {ctx}")
        xp = torch.cat([x[:, :1, :].repeat(1, pad, 1), x], dim=1)
        with torch.no_grad():
            out = model(past_values=xp.to(device)).prediction_outputs
        return out[:, :horizon, :].float().cpu()

    info = {"hf_id": HF_IDS["ttm"], "precision": "fp32", "context_length": ctx}
    notes = [f"context {ctx}: the 336-step lookback is left-padded by repeating its first value "
             f"({ctx - SEQ_LEN} steps), as in the old suite"]
    return Forecaster("ttm", fn, info, notes)


def _fourier_base(horizon: int) -> Forecaster:
    info = {"top_k": FOURIER_TOP_K, "params": 0}
    notes = ["zero-parameter reference: lookback mean + top-3 rFFT harmonic extrapolation "
             "(srp26.fourier.extract_and_extrapolate_fourier_base_torch); the old suite scored y_base without the mean"]
    return Forecaster("fourier_base", lambda x: fourier_base_forecast(x, horizon), info, notes)


_CACHE: Dict[str, object] = {}


def load_forecaster(model: str, horizon: int, device: torch.device, seed: int,
                    series_batch: Optional[int] = None, hf_id: Optional[str] = None) -> Forecaster:
    """Build (or reuse) a forecaster. At most one foundation model is kept in memory."""
    if model not in FOUNDATION_MODELS:
        raise ValueError(f"Unknown foundation model '{model}'. Known: {FOUNDATION_MODELS}")
    series_batch = series_batch or DEFAULT_SERIES_BATCH[model]
    # TimesFM is compiled for one max_horizon; TTM is H-specific; Chronos is horizon-agnostic.
    key = (model, horizon if model in ("timesfm", "ttm", "fourier_base") else None, series_batch, seed, hf_id)
    if _CACHE.get("key") == key:
        fc = _CACHE["fc"]
    else:
        _CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if model == "timesfm":
            fc = _load_timesfm(horizon, series_batch)
        elif model == "chronos_bolt":
            fc = _load_chronos_bolt(device, series_batch, hf_id or HF_IDS["chronos_bolt"])
        elif model == "chronos_t5":
            fc = _load_chronos_t5(device, series_batch, seed)
        elif model == "ttm":
            fc = _load_ttm(device, horizon)
        else:
            fc = _fourier_base(horizon)
        _CACHE.update(key=key, fc=fc)
    if hasattr(fc.fn, "__dict__"):
        fc.fn.horizon = horizon  # Chronos closures read the horizon at call time
    return fc


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def result_name(model: str, fourier: bool) -> str:
    return model + ("_fourier" if fourier else "")


def result_path(root: Path, dataset: str, horizon: int, model: str, fourier: bool, seed: int) -> Path:
    return root / "foundation" / dataset / f"H{horizon}" / result_name(model, fourier) / f"s{seed}.json"


def evaluate_forecaster(forecast: Callable[[torch.Tensor], torch.Tensor], test_ds, indices: np.ndarray,
                        horizon: int, dt_hours: float, snaive_period: int, channels: int):
    """Score ``forecast`` on the selected test windows; returns (metrics, wmse, wmae)."""
    ev = StreamingEvaluator(horizon, dt_hours, torch.device("cpu"))
    chunk = max(1, MAX_SERIES_PER_CHUNK // channels)
    for start in range(0, len(indices), chunk):
        items = [test_ds[int(i)] for i in indices[start:start + chunk]]
        x = torch.stack([it[0] for it in items])
        y = torch.stack([it[1] for it in items])
        pred = forecast(x).float().cpu()
        if pred.shape != y.shape:
            raise RuntimeError(f"forecast shape {tuple(pred.shape)} != target shape {tuple(y.shape)}")
        if not torch.isfinite(pred).all():
            raise FloatingPointError("non-finite forecast")
        ev.update(pred, y, snaive=seasonal_naive(x, horizon, snaive_period))
    return ev.result(), ev.window_errors(), ev.window_abs_errors()


def run_one(set_label: str, dataset: str, horizon: int, model: str, fourier: bool = False, seed: int = 2024,
            max_windows: Optional[int] = None, series_batch: Optional[int] = None,
            out_root: Optional[Path] = None, hf_id: Optional[str] = None) -> Optional[Path]:
    if fourier and model not in DECOUPLABLE:
        raise ValueError(f"--fourier applies to foundation models {DECOUPLABLE}, not '{model}'.")
    if hf_id is not None and (model != "chronos_bolt" or max_windows is None):
        raise ValueError("--hf-id-override is a smoke-test aid for chronos_bolt only and requires --max-windows.")
    if model == "ttm" and horizon != TTM_HORIZON:
        raise ValueError(f"TTM-R1 only supports H={TTM_HORIZON}; refusing H={horizon}.")
    out_root = out_root or REPO_ROOT / "results" / "v2" / set_label
    path = result_path(out_root, dataset, horizon, model, fourier, seed)
    if path.exists():
        print(f"[skip] {path.relative_to(out_root)} exists")
        return path

    t0 = time.time()
    device = get_device()
    bundle = build_bundle(dataset, SEQ_LEN, horizon, batch_size=32, generator=seed_everything(seed))
    test_ds = bundle.test_loader.dataset
    indices = stride_h_indices(len(test_ds), horizon, max_windows)
    fc = load_forecaster(model, horizon, device, seed, series_batch, hf_id)
    forecast = (lambda x: fourier_decoupled(fc, x, horizon)) if fourier else fc

    seed_everything(seed)  # Chronos-T5 sampling: seeded immediately before inference
    t_inf = time.time()
    metrics, wmse, wmae = evaluate_forecaster(forecast, test_ds, indices, horizon, bundle.spec.dt_hours,
                                              bundle.spec.daily_period, bundle.spec.channels)
    inference_sec = time.time() - t_inf

    notes = list(fc.notes)
    if fourier:
        notes.append(f"Fourier decoupling: model forecasts x_res = x - x_base (top-{FOURIER_TOP_K} non-DC rFFT "
                     "harmonics, mean kept in x_res); prediction = model(x_res) + y_base")
    record = {
        "harness_version": HARNESS_VERSION,
        "set": set_label, "phase": "foundation", "dataset": dataset, "pred_len": horizon, "seq_len": SEQ_LEN,
        "model": result_name(model, fourier), "base_model": model, "fourier": fourier, "seed": seed,
        "model_info": {**fc.info, "hf_id_override": hf_id, "fourier_decoupling": fourier, "fourier_top_k": FOURIER_TOP_K if fourier else None},
        "model_notes": notes,
        "eval_windows": {"rule": "non-overlapping test windows, stride = H, from test window 0",
                         "stride": horizon, "n": int(len(indices)), "n_test_stride1": len(test_ds),
                         "max_windows": max_windows, "indices": indices.tolist()},
        "split": bundle.split, "dataset_sha256": bundle.sha256,
        "metrics": metrics, "seasonal_naive_period": bundle.spec.daily_period,
        "window_hours": horizon * bundle.spec.dt_hours,
        "daily_cycle_resolvable": horizon * bundle.spec.dt_hours >= 24.0,
        "inference_time_sec": round(inference_sec, 2),
        "wall_time_sec": round(time.time() - t0, 2),
        "provenance": provenance(device),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".wmse.npy"), wmse.astype(np.float32))
    np.save(path.with_suffix(".wmae.npy"), wmae.astype(np.float32))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    print(f"[done] foundation {dataset} H{horizon} {record['model']} s{seed} windows={len(indices)} "
          f"mse={metrics['mse']:.4f} mae={metrics['mae']:.4f} ({record['wall_time_sec']}s)")
    return path


def run_job(set_label: str, job: Dict) -> Optional[Path]:
    return run_one(set_label, job["dataset"], job["horizon"], job["model"], bool(job.get("fourier", False)),
                   job.get("seed", 2024), job.get("max_windows"), job.get("series_batch"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True)
    p.add_argument("--dataset")
    p.add_argument("--horizon", type=int)
    p.add_argument("--model", choices=FOUNDATION_MODELS)
    p.add_argument("--fourier", action="store_true", help="Fourier decoupling (suffix _fourier).")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--max-windows", type=int, help="Smoke tests only: evaluate the first N stride-H windows.")
    p.add_argument("--series-batch", type=int, help="Series per model call (default per model).")
    p.add_argument("--hf-id-override", help="Smoke tests only (chronos_bolt + --max-windows): load another "
                   "checkpoint, e.g. amazon/chronos-bolt-tiny when the base weights are not cached. Recorded.")
    p.add_argument("--jobs", help="JSONL job list (phase=foundation entries).")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args()
    if a.jobs:
        jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
        mine = jobs[a.shard::a.n_shards]
        print(f"shard {a.shard}/{a.n_shards}: {len(mine)} of {len(jobs)} jobs")
        failures = 0
        for j in mine:
            try:
                run_job(a.set, j)
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {j}: {type(exc).__name__}: {exc}")
        print(f"shard {a.shard} finished, failures={failures}")
        raise SystemExit(1 if failures else 0)
    if not (a.dataset and a.horizon and a.model):
        p.error("--dataset, --horizon and --model are required without --jobs")
    run_one(a.set, a.dataset, a.horizon, a.model, a.fourier, a.seed, a.max_windows, a.series_batch,
            hf_id=a.hf_id_override)


if __name__ == "__main__":
    main()

"""Stage 2b: DSOF-SD spectral distillation with a matched control (protocol §4 Tier 2).

    uv run python -m srp26.bench.dsof cache-teacher --dataset ETTh1 --horizon 96 [--teacher timesfm]
    uv run python -m srp26.bench.dsof run --set local --dataset ETTh1 --horizon 96 --student sor --gate hard --seed 2024
    uv run python -m srp26.bench.dsof jobs --set bigmama --jobs configs/v2/jobs_stage2_dsof.jsonl --shard 0 --n-shards 8

Confound fixed here: ``models/dsof_sd.py`` compares the distilled student (5 warm
+ up to 25 distillation epochs) with the warm student (5 epochs), mixing the
effect of distillation with the effect of 25 extra epochs. Every gate is now
paired with ``--gate control``: the identical trainer, schedule, optimizer and
data order with ``lambda_max = 0`` (pure time-domain training for the same
epochs). Distillation must beat the control, not the warm student.

The student is the v2 CycleNet-Pro network (``build_model("cyclenet_pro_<b>")``,
inner ``CycleNetPro``); the trainer is ``DSOFSDTrainer`` unchanged except that
Phase 2 is never skipped (``skip_degenerate_phase2=False``). Final, warm and
teacher forecasts are all scored on the full stride-1 test set with the v2
``StreamingEvaluator``.
"""

import argparse
import copy
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from srp26.bench import HARNESS_VERSION
from srp26.bench.data import Bundle, build_bundle
from srp26.bench.env import REPO_ROOT, get_device, provenance, seed_everything
from srp26.bench.metrics import StreamingEvaluator, seasonal_naive
from srp26.bench.models import build_model, count_params
from srp26.bench.train import evaluate_test_windows

SEQ_LEN = 336
BATCH_SIZE = 32
EVAL_BATCH = 256
STUDENTS = ("linear", "sor", "cnn", "mlp")
GATES = ("hard", "soft", "benefit", "consensus", "control")
TEACHERS = ("timesfm", "fourier_base")
CACHE_DIR = REPO_ROOT / "cache" / "teacher"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# Teacher cache (Phase 0): one forecast per dataset window, in dataset order
# --------------------------------------------------------------------------


def cache_path(teacher: str, dataset: str, horizon: int, cache_dir: Optional[Path] = None) -> Path:
    return (cache_dir or CACHE_DIR) / f"{teacher}_{dataset}_H{horizon}.pt"


def make_teacher(teacher: str, horizon: int):
    if teacher == "timesfm":
        from models.dsof_sd import TimesFMTeacher
        return TimesFMTeacher(seq_len=SEQ_LEN, pred_len=horizon)
    if teacher == "fourier_base":
        from srp26.bench.foundation import fourier_base_forecast
        return lambda x: fourier_base_forecast(x, horizon)
    raise ValueError(f"Unknown teacher '{teacher}'. Known: {TEACHERS}")


def _split_datasets(bundle: Bundle) -> Dict[str, Dataset]:
    return {"train": bundle.train_loader.dataset, "val": bundle.val_loader.dataset, "test": bundle.test_loader.dataset}


@torch.no_grad()
def teacher_forecasts(teacher_fn, ds: Dataset, batch: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher forecast for every window of ``ds`` in index order (non-shuffled).

    Returns ``(y_teacher [N, H, C], last_obs [N, C])``; ``last_obs[i]`` is the
    final lookback value of window ``i``, kept as an alignment fingerprint.
    """
    preds, last = [], []
    for x, _, _ in DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False):
        preds.append(teacher_fn(x).float().cpu())
        last.append(x[:, -1, :].clone())
    return torch.cat(preds), torch.cat(last)


def build_teacher_cache(bundle: Bundle, teacher: str, horizon: int, teacher_fn=None, batch: int = 128) -> Dict:
    teacher_fn = teacher_fn or make_teacher(teacher, horizon)
    payload: Dict[str, object] = {}
    t0 = time.time()
    for name, ds in _split_datasets(bundle).items():
        y_t, last = teacher_forecasts(teacher_fn, ds, batch)
        if y_t.shape[0] != len(ds):
            raise RuntimeError(f"teacher produced {y_t.shape[0]} forecasts for {len(ds)} {name} windows")
        payload[name] = {"y_teacher": y_t, "last_obs": last, "start_idx_in_full": int(ds.start_idx_in_full)}
        print(f"  [teacher {teacher}] {name}: {len(ds)} windows ({time.time() - t0:.1f}s)", flush=True)
    payload["metadata"] = {
        "teacher": teacher, "dataset": bundle.spec.name, "dataset_sha256": bundle.sha256,
        "pred_len": horizon, "seq_len": SEQ_LEN, "n_windows": {s: len(d) for s, d in _split_datasets(bundle).items()},
        "build_time_sec": round(time.time() - t0, 2),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return payload


def save_cache(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def verify_cache(payload: Dict, bundle: Bundle, teacher: str, horizon: int) -> None:
    """Refuse a cache that does not belong to this bundle window-by-window."""
    meta = payload["metadata"]
    expect = {"teacher": teacher, "dataset_sha256": bundle.sha256, "pred_len": horizon, "seq_len": SEQ_LEN}
    for k, v in expect.items():
        if meta.get(k) != v:
            raise RuntimeError(f"teacher cache mismatch on {k}: cache={meta.get(k)!r}, expected={v!r}")
    for name, ds in _split_datasets(bundle).items():
        part = payload[name]
        if part["y_teacher"].shape[0] != len(ds) or part["start_idx_in_full"] != ds.start_idx_in_full:
            raise RuntimeError(f"teacher cache {name}: window count/offset differs from the bundle")
        n = len(ds)
        probe = sorted({0, n // 2, n - 1})
        for i in probe:  # window i of the cache must be window i of the dataset
            if not torch.equal(part["last_obs"][i], ds[i][0][-1]):
                raise RuntimeError(f"teacher cache {name}: window {i} is misaligned")


def load_or_build_cache(bundle: Bundle, teacher: str, horizon: int, cache_dir: Optional[Path] = None) -> Dict:
    path = cache_path(teacher, bundle.spec.name, horizon, cache_dir)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
    else:
        print(f"[teacher] building {path.name}", flush=True)
        payload = build_teacher_cache(bundle, teacher, horizon)
        save_cache(payload, path)
    verify_cache(payload, bundle, teacher, horizon)
    return payload


class TeacherAlignedDataset(Dataset):
    """Yields ``(x, y, y_teacher, cycle_index)`` with the teacher forecast of window ``i`` at index ``i``."""

    def __init__(self, base: Dataset, y_teacher: torch.Tensor):
        if len(base) != y_teacher.shape[0]:
            raise ValueError(f"{len(base)} windows vs {y_teacher.shape[0]} teacher forecasts")
        self.base, self.y_teacher = base, y_teacher

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y, cyc = self.base[i]
        return x, y, self.y_teacher[i], cyc


# --------------------------------------------------------------------------
# Trainer wrapper
# --------------------------------------------------------------------------


def _trainer_cls():
    from models.dsof_sd import DSOFSDTrainer

    class V2DSOFSDTrainer(DSOFSDTrainer):
        """DSOFSDTrainer whose benefit-gate probes leave every RNG untouched, so the
        Phase 2 data order and dropout stream equal those of the control run."""

        def _calibrate_benefit_gates(self, train_loader, val_loader, warm_val_metrics):
            gen = getattr(train_loader, "generator", None)
            gen_state = gen.get_state() if gen is not None else None
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            out = super()._calibrate_benefit_gates(train_loader, val_loader, warm_val_metrics)
            if gen is not None:
                gen.set_state(gen_state)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            return out

    return V2DSOFSDTrainer


def make_config(gate: str, epochs_override: Optional[Tuple[int, int]] = None):
    """Method defaults (lr 3e-4, warm 5, distill 25, patience 7, lambda_max 0.2).

    ``control`` = the same trainer with ``lambda_max = 0``; its gates are still
    calibrated with the hard rule (diagnostic only, they multiply a zero weight).
    """
    from models.dsof_sd import DSOFSDConfig

    if gate not in GATES:
        raise ValueError(f"Unknown gate '{gate}'. Known: {GATES}")
    cfg = DSOFSDConfig(gate_mode="hard" if gate == "control" else gate, skip_degenerate_phase2=False)
    if gate == "control":
        cfg.lambda_max = 0.0
    if epochs_override:
        cfg.warm_epochs, cfg.distill_epochs = epochs_override
        cfg.benefit_probe_epochs = min(cfg.benefit_probe_epochs, max(1, epochs_override[1]))
    return cfg


@torch.no_grad()
def evaluate_teacher(y_teacher: torch.Tensor, test_ds: Dataset, horizon: int, dt_hours: float, snaive_period: int):
    ev = StreamingEvaluator(horizon, dt_hours, torch.device("cpu"))
    off = 0
    for x, y, _ in DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False):
        ev.update(y_teacher[off:off + len(y)], y, snaive=seasonal_naive(x, horizon, snaive_period))
        off += len(y)
    return ev.result()


def run_name(student: str, gate: str, teacher: str = "timesfm") -> str:
    """``<student>_<gate>``; a non-default teacher is appended so it can never mix with TimesFM runs."""
    return f"{student}_{gate}" + ("" if teacher == "timesfm" else f"_teacher-{teacher}")


def result_path(root: Path, dataset: str, horizon: int, student: str, gate: str, seed: int,
                teacher: str = "timesfm") -> Path:
    return root / "dsof" / dataset / f"H{horizon}" / run_name(student, gate, teacher) / f"s{seed}.json"


def run_one(set_label: str, dataset: str, horizon: int, student: str, gate: str, seed: int,
            teacher: str = "timesfm", epochs_override: Optional[Tuple[int, int]] = None,
            out_root: Optional[Path] = None, cache_dir: Optional[Path] = None) -> Optional[Path]:
    if student not in STUDENTS:
        raise ValueError(f"Unknown student '{student}'. Known: {STUDENTS}")
    out_root = out_root or REPO_ROOT / "results" / "v2" / set_label
    path = result_path(out_root, dataset, horizon, student, gate, seed, teacher)
    if path.exists():
        print(f"[skip] {path.relative_to(out_root)} exists")
        return path

    t0 = time.time()
    device = get_device()
    gen = seed_everything(seed)
    bundle = build_bundle(dataset, SEQ_LEN, horizon, BATCH_SIZE, generator=gen, eval_batch_size=EVAL_BATCH)
    cache = load_or_build_cache(bundle, teacher, horizon, cache_dir)
    dss = _split_datasets(bundle)
    aligned = {s: TeacherAlignedDataset(dss[s], cache[s]["y_teacher"]) for s in SPLITS}

    seed_everything(seed)  # identical student init and data order for every gate of a (dataset, H, student, seed)
    gen.manual_seed(seed)
    train_loader = DataLoader(aligned["train"], batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=gen)
    val_loader = DataLoader(aligned["val"], batch_size=EVAL_BATCH, shuffle=False)
    test_loader = DataLoader(aligned["test"], batch_size=EVAL_BATCH, shuffle=False)

    wrapped, info = build_model(f"cyclenet_pro_{student}", bundle, SEQ_LEN, horizon)
    net = wrapped.net  # inner CycleNetPro (DSOFSDTrainer calls it with (x, cycle_index))
    cfg = make_config(gate, epochs_override)
    trainer = _trainer_cls()(net, cfg, device)
    report = trainer.fit_and_evaluate(train_loader, val_loader, test_loader)

    # v2 metrics on the full stride-1 test set
    spec = bundle.spec
    metrics, wmse, wmae = evaluate_test_windows(wrapped, bundle.test_loader, device, horizon, spec.dt_hours,
                                                spec.daily_period)
    warm = copy.deepcopy(wrapped)
    warm.net.load_state_dict({k: v.to(device) for k, v in trainer.warm_state.items()})
    metrics_warm, _, _ = evaluate_test_windows(warm, bundle.test_loader, device, horizon, spec.dt_hours,
                                               spec.daily_period)
    metrics_teacher = evaluate_teacher(cache["test"]["y_teacher"], dss["test"], horizon, spec.dt_hours,
                                       spec.daily_period)

    gates = report["gates"]
    all_gates_zero = all(float(g) == 0.0 for g in gates.values())
    hypothesis = report["hypothesis"] if gate != "control" else {
        "status": "not_applicable", "reason": "control run (lambda_max = 0)"}
    record = {
        "harness_version": HARNESS_VERSION,
        "set": set_label, "phase": "dsof", "dataset": dataset, "pred_len": horizon, "seq_len": SEQ_LEN,
        "model": run_name(student, gate, teacher), "student": student, "gate": gate, "seed": seed,
        "teacher": teacher, "teacher_cache": {k: v for k, v in cache["metadata"].items()},
        "hparams": {"dsof_config": asdict(cfg), "batch_size": BATCH_SIZE, "optimizer": "AdamW (DSOFSDTrainer)",
                    "scheduler": "cosine over distill_epochs (eta_min 1e-5)", "grad_clip": 1.0, "precision": "fp32",
                    "epochs_override": list(epochs_override) if epochs_override else None},
        "model_info": {**info, "student_model": f"cyclenet_pro_{student}"},
        "model_notes": ["control = identical trainer/schedule/data order with lambda_max = 0" if gate == "control"
                        else f"gate={gate}, lambda annealed 0 -> {cfg.lambda_max}",
                        "Phase 2 always runs the full schedule (skip_degenerate_phase2=False); "
                        "an all-zero-gate run therefore equals the control",
                        "trainer_report.* uses the method's own fixed-cutoff DSOF bands; metrics* use v2 metrics"],
        "params": count_params(wrapped), "split": bundle.split, "dataset_sha256": bundle.sha256,
        "gates": gates, "all_gates_zero": all_gates_zero,
        "metrics": metrics, "metrics_warm": metrics_warm, "metrics_teacher": metrics_teacher,
        "seasonal_naive_period": spec.daily_period,
        "window_hours": horizon * spec.dt_hours, "daily_cycle_resolvable": horizon * spec.dt_hours >= 24.0,
        "trainer_report": {"gates": gates, "phase2_degenerate": report["phase2_degenerate"],
                           "calibration": report["calibration"], "history": report["history"],
                           "hypothesis": hypothesis, "teacher": report["teacher"],
                           "warm_student": report["warm_student"], "distilled_student": report["distilled_student"],
                           "runtime_sec": report["runtime_sec"]},
        "distill_epochs_run": len(report["history"]),
        "wall_time_sec": round(time.time() - t0, 2),
        "provenance": provenance(device),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".wmse.npy"), wmse.astype(np.float32))
    np.save(path.with_suffix(".wmae.npy"), wmae.astype(np.float32))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=float))
    tmp.replace(path)
    print(f"[done] dsof {dataset} H{horizon} {student}_{gate} s{seed} teacher={teacher} gates={gates} "
          f"final={metrics['mse']:.4f} warm={metrics_warm['mse']:.4f} teacher={metrics_teacher['mse']:.4f} "
          f"({record['wall_time_sec']}s)")
    return path


def cache_teacher(dataset: str, horizon: int, teacher: str = "timesfm", cache_dir: Optional[Path] = None) -> Path:
    path = cache_path(teacher, dataset, horizon, cache_dir)
    if path.exists():
        print(f"[skip] {path} exists")
        return path
    bundle = build_bundle(dataset, SEQ_LEN, horizon, BATCH_SIZE, generator=torch.Generator())
    payload = build_teacher_cache(bundle, teacher, horizon)
    save_cache(payload, path)
    print(f"[done] teacher cache {path} {payload['metadata']['n_windows']}")
    return path


def _parse_epochs(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    w, d = (int(v) for v in s.split(","))
    return w, d


def run_job(set_label: str, job: Dict) -> Optional[Path]:
    if job["phase"] == "dsof_teacher":
        return cache_teacher(job["dataset"], job["horizon"], job.get("teacher", "timesfm"))
    return run_one(set_label, job["dataset"], job["horizon"], job["student"], job["gate"], job["seed"],
                   job.get("teacher", "timesfm"), _parse_epochs(job.get("epochs_override")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache-teacher")
    c.add_argument("--dataset", required=True)
    c.add_argument("--horizon", type=int, required=True)
    c.add_argument("--teacher", choices=TEACHERS, default="timesfm")
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--horizon", type=int, required=True)
    r.add_argument("--student", choices=STUDENTS, required=True)
    r.add_argument("--gate", choices=GATES, required=True)
    r.add_argument("--seed", type=int, default=2024)
    r.add_argument("--teacher", choices=TEACHERS, default="timesfm")
    r.add_argument("--epochs-override", help="SMOKE ONLY: 'warm,distill' epochs, e.g. 1,1 (recorded in the JSON).")
    j = sub.add_parser("jobs")
    j.add_argument("--set", required=True)
    j.add_argument("--jobs", required=True)
    j.add_argument("--shard", type=int, default=0)
    j.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "cache-teacher":
        cache_teacher(a.dataset, a.horizon, a.teacher)
    elif a.cmd == "run":
        run_one(a.set, a.dataset, a.horizon, a.student, a.gate, a.seed, a.teacher, _parse_epochs(a.epochs_override))
    else:
        jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
        mine = jobs[a.shard::a.n_shards]
        print(f"shard {a.shard}/{a.n_shards}: {len(mine)} of {len(jobs)} jobs")
        failures = 0
        for job in mine:
            try:
                run_job(a.set, job)
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {job}: {type(exc).__name__}: {exc}")
        print(f"shard {a.shard} finished, failures={failures}")
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

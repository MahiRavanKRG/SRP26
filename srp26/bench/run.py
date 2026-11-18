"""v2 run entry point.

Single run:
    uv run python -m srp26.bench.run --set local --phase final --dataset ETTh1 --horizon 96 --model linear --seed 2024

Job list (one JSON object per line; used by Slurm arrays and local queues):
    uv run python -m srp26.bench.run --set bigmama --jobs configs/v2/jobs_final.jsonl --shard 0 --n-shards 8

Phases:
    lrsel  validation-only LR selection (seed 2024); the test set is never evaluated.
    final  the reported runs; LR comes from configs/v2/selected_lr.json unless fixed per model.

Results are immutable: an existing result file is skipped, never overwritten.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from srp26.bench import HARNESS_VERSION
from srp26.bench.data import Bundle, build_bundle
from srp26.bench.env import REPO_ROOT, get_device, provenance, seed_everything
from srp26.bench.models import build_model, count_params, model_defaults
from srp26.bench.train import evaluate_test, fit

SEQ_LEN = 336
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3]
SELECTED_LR_FILE = REPO_ROOT / "configs" / "v2" / "selected_lr.json"


def result_path(root: Path, phase: str, dataset: str, horizon: int, model: str, seed: int,
                lr: Optional[float]) -> Path:
    tag = f"s{seed}" + (f"_lr{lr:g}" if phase == "lrsel" else "")
    return root / phase / dataset / f"H{horizon}" / model / f"{tag}.json"


def selected_lr(dataset: str, horizon: int, model: str) -> float:
    table = json.loads(SELECTED_LR_FILE.read_text())
    key = f"{dataset}|{horizon}|{model}"
    if key not in table:
        raise KeyError(f"No selected LR for {key}; run the lrsel phase and select_lr first.")
    return float(table[key])


_BUNDLES: Dict[tuple, Bundle] = {}


def get_bundle(dataset: str, horizon: int, batch_size: int, seed: int) -> Bundle:
    key = (dataset, horizon, batch_size)
    if key not in _BUNDLES:
        _BUNDLES.clear()  # keep at most one dataset in memory
        _BUNDLES[key] = build_bundle(dataset, SEQ_LEN, horizon, batch_size, generator=seed_everything(seed))
    bundle = _BUNDLES[key]
    bundle.train_loader.generator.manual_seed(seed)  # identical shuffle order for a given seed
    return bundle


def run_one(set_label: str, phase: str, dataset: str, horizon: int, model_name: str, seed: int,
            lr: Optional[float] = None, out_root: Optional[Path] = None) -> Optional[Path]:
    out_root = out_root or REPO_ROOT / "results" / "v2" / set_label
    defaults = model_defaults(model_name, dataset)
    if phase == "lrsel":
        if not defaults["tune_lr"]:
            return None
        assert lr is not None, "lrsel requires --lr"
    elif lr is None:
        lr = defaults["lr"] if not defaults["tune_lr"] else (
            selected_lr(dataset, horizon, model_name) if model_name not in ("repeat_last", "seasonal_naive") else 0.0)

    path = result_path(out_root, phase, dataset, horizon, model_name, seed, lr)
    if path.exists():
        print(f"[skip] {path.relative_to(out_root)} exists")
        return path

    device = get_device()
    bundle = get_bundle(dataset, horizon, defaults["batch_size"], seed)
    seed_everything(seed)
    bundle.train_loader.generator.manual_seed(seed)
    model, info = build_model(model_name, bundle, SEQ_LEN, horizon)
    params = count_params(model)
    t0 = time.time()
    train = fit(model, bundle.train_loader, bundle.val_loader, device, lr=lr,
                max_epochs=defaults["max_epochs"], patience=defaults["patience"])

    record = {
        "harness_version": HARNESS_VERSION,
        "set": set_label, "phase": phase, "dataset": dataset, "pred_len": horizon, "seq_len": SEQ_LEN,
        "model": model_name, "seed": seed,
        "hparams": {"lr": lr, "batch_size": defaults["batch_size"], "max_epochs": defaults["max_epochs"],
                    "patience": defaults["patience"], "optimizer": "Adam", "scheduler": "cosine", "precision": "fp32"},
        "model_info": info, "params": params, "split": bundle.split, "dataset_sha256": bundle.sha256,
        **{k: v for k, v in train.items()},
        "window_hours": horizon * bundle.spec.dt_hours,
        "daily_cycle_resolvable": horizon * bundle.spec.dt_hours >= 24.0,
    }
    if phase == "final":
        metrics, window_err = evaluate_test(model, bundle.test_loader, device, horizon, bundle.spec.dt_hours,
                                            bundle.spec.daily_period)
        record["metrics"] = metrics
        record["seasonal_naive_period"] = bundle.spec.daily_period
    record["wall_time_sec"] = round(time.time() - t0, 2)
    record["provenance"] = provenance(device)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    path.parent.mkdir(parents=True, exist_ok=True)
    if phase == "final":
        np.save(path.with_suffix(".wmse.npy"), window_err.astype(np.float32))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)  # atomic: a crashed run never leaves a half-written result
    m = record.get("metrics", {})
    print(f"[done] {phase} {dataset} H{horizon} {model_name} s{seed} lr={lr:g} "
          f"val={train['val_mse']:.4f} test={m.get('mse', float('nan')):.4f} ({record['wall_time_sec']}s)")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, help="Result-set label, e.g. 'local' or 'bigmama'.")
    p.add_argument("--phase", choices=["lrsel", "final"], default="final")
    p.add_argument("--dataset")
    p.add_argument("--horizon", type=int)
    p.add_argument("--model")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--lr", type=float)
    p.add_argument("--jobs", help="JSONL job list (overrides single-run args).")
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
                run_one(a.set, j["phase"], j["dataset"], j["horizon"], j["model"], j["seed"], j.get("lr"))
            except Exception as exc:  # record and continue; completeness check reports the gap
                failures += 1
                print(f"[FAIL] {j}: {type(exc).__name__}: {exc}")
        print(f"shard {a.shard} finished, failures={failures}")
        raise SystemExit(1 if failures else 0)
    run_one(a.set, a.phase, a.dataset, a.horizon, a.model, a.seed, a.lr)


if __name__ == "__main__":
    main()

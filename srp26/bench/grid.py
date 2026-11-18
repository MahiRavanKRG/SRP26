"""Pre-registered v2 grid (protocol §4): job lists, LR selection, completeness check.

    uv run python -m srp26.bench.grid jobs                 # write configs/v2/jobs_*.jsonl
    uv run python -m srp26.bench.grid select-lr --set bigmama
    uv run python -m srp26.bench.grid check --set bigmama --phase final
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from srp26.bench.env import REPO_ROOT
from srp26.bench.models import TIER1_TRAINED, TRANSFORM_MODELS, UNTRAINED
from srp26.bench.run import LR_GRID, SELECTED_LR_FILE, result_path

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic", "solar"]
HORIZONS = [96, 192, 336, 720]
SEEDS = [2024, 2025, 2026]
LRSEL_SEED = 2024
TUNED = TIER1_TRAINED + TRANSFORM_MODELS
CONFIG_DIR = REPO_ROOT / "configs" / "v2"
BIG = {"electricity", "traffic", "solar"}


def lrsel_jobs() -> List[Dict]:
    return [{"phase": "lrsel", "dataset": d, "horizon": h, "model": m, "seed": LRSEL_SEED, "lr": lr}
            for d in DATASETS for h in HORIZONS for m in TUNED for lr in LR_GRID]


def final_jobs(include_patchtst_big: bool = True) -> List[Dict]:
    jobs = []
    for d in DATASETS:
        for h in HORIZONS:
            for m in UNTRAINED:  # deterministic: one seed
                jobs.append({"phase": "final", "dataset": d, "horizon": h, "model": m, "seed": LRSEL_SEED})
            models = TUNED + (["patchtst"] if (include_patchtst_big or d not in BIG) else [])
            for m in models:
                for s in SEEDS:
                    jobs.append({"phase": "final", "dataset": d, "horizon": h, "model": m, "seed": s})
    return jobs


def _sort_heavy_last(jobs: List[Dict]) -> List[Dict]:
    """Small datasets first so results arrive early; PatchTST on big data last."""
    order = {d: i for i, d in enumerate(DATASETS)}
    return sorted(jobs, key=lambda j: (j["model"] == "patchtst", order[j["dataset"]], j["horizon"]))


def write_jobs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = {
        "jobs_lrsel.jsonl": lrsel_jobs(),
        "jobs_final_bigmama.jsonl": final_jobs(include_patchtst_big=True),
        "jobs_final_local.jsonl": final_jobs(include_patchtst_big=False),
    }
    for name, jobs in sets.items():
        (CONFIG_DIR / name).write_text("\n".join(json.dumps(j) for j in _sort_heavy_last(jobs)) + "\n")
        print(f"{name}: {len(jobs)} jobs")


def select_lr(set_label: str) -> None:
    root = REPO_ROOT / "results" / "v2" / set_label
    table, missing = {}, []
    for d in DATASETS:
        for h in HORIZONS:
            for m in TUNED:
                vals = []
                for lr in LR_GRID:
                    p = result_path(root, "lrsel", d, h, m, LRSEL_SEED, lr)
                    if p.exists():
                        vals.append((json.loads(p.read_text())["val_mse"], lr))
                    else:
                        missing.append(str(p.relative_to(root)))
                if len(vals) == len(LR_GRID):
                    table[f"{d}|{h}|{m}"] = min(vals)[1]  # lowest validation MSE; ties -> smaller LR
    if missing:
        raise SystemExit(f"LR selection incomplete: {len(missing)} lrsel results missing, e.g. {missing[:3]}")
    SELECTED_LR_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_LR_FILE.write_text(json.dumps(table, indent=1, sort_keys=True))
    counts = defaultdict(int)
    for v in table.values():
        counts[v] += 1
    print(f"selected LR for {len(table)} cells -> {SELECTED_LR_FILE}; distribution {dict(sorted(counts.items()))}")


def check(set_label: str, phase: str, jobs_file: str) -> None:
    root = REPO_ROOT / "results" / "v2" / set_label
    jobs = [json.loads(l) for l in (CONFIG_DIR / jobs_file).read_text().splitlines() if l.strip()]
    missing = [j for j in jobs if j["phase"] == phase and not result_path(
        root, phase, j["dataset"], j["horizon"], j["model"], j["seed"], j.get("lr")).exists()]
    n = sum(j["phase"] == phase for j in jobs)
    print(f"{set_label}/{phase}: {n - len(missing)}/{n} complete")
    for j in missing[:20]:
        print("  missing:", j)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    s = sub.add_parser("select-lr")
    s.add_argument("--set", required=True)
    c = sub.add_parser("check")
    c.add_argument("--set", required=True)
    c.add_argument("--phase", default="final")
    c.add_argument("--jobs", default="jobs_final_bigmama.jsonl")
    a = p.parse_args()
    if a.cmd == "jobs":
        write_jobs()
    elif a.cmd == "select-lr":
        select_lr(a.set)
    else:
        check(a.set, a.phase, a.jobs)


if __name__ == "__main__":
    main()

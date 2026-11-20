"""Pre-registered v2 grid (protocol §4): job lists, LR selection, completeness check.

    uv run python -m srp26.bench.grid jobs                 # write configs/v2/jobs_*.jsonl
    uv run python -m srp26.bench.grid select-lr --set bigmama
    uv run python -m srp26.bench.grid check --set bigmama --phase final
    uv run python -m srp26.bench.grid stage2-jobs          # configs/v2/jobs_stage2_{foundation,teacher,dsof}.jsonl
    uv run python -m srp26.bench.grid stage2-check --set bigmama

Stage-2 job lists run through the same dispatcher as stage 1 (it switches on ``phase``):
    uv run python -m srp26.bench.run --set bigmama --jobs configs/v2/jobs_stage2_foundation.jsonl --shard $I --n-shards $N
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


# --------------------------------------------------------------------------
# Stage 2 (Tier 2): zero-shot foundation models and DSOF-SD distillation
# --------------------------------------------------------------------------

STAGE2_SEED = 2024
STAGE2_FOUNDATION = ["timesfm", "chronos_bolt", "chronos_t5"]
STAGE2_DSOF_DATASETS = ["ETTh1", "ETTh2"]
STAGE2_DSOF_VARIANTS = [("linear", "hard"), ("linear", "control"),
                        ("sor", "hard"), ("sor", "soft"), ("sor", "benefit"), ("sor", "consensus"), ("sor", "control"),
                        ("cnn", "benefit"), ("cnn", "control")]


def stage2_foundation_jobs() -> List[Dict]:
    """Model-major order, so a strided shard mostly reuses the loaded model."""
    def job(d, h, m, f):
        return {"phase": "foundation", "dataset": d, "horizon": h, "model": m, "fourier": f, "seed": STAGE2_SEED}
    jobs = []
    for m in STAGE2_FOUNDATION:
        jobs += [job(d, h, m, f) for f in (False, True) for d in DATASETS for h in HORIZONS]
    jobs += [job(d, 96, "ttm", f) for f in (False, True) for d in DATASETS]  # TTM-R1 supports H=96 only
    jobs += [job(d, h, "fourier_base", False) for d in DATASETS for h in HORIZONS]
    return jobs


def stage2_teacher_jobs() -> List[Dict]:
    return [{"phase": "dsof_teacher", "dataset": d, "horizon": h, "teacher": "timesfm"}
            for d in STAGE2_DSOF_DATASETS for h in HORIZONS]


def stage2_dsof_jobs() -> List[Dict]:
    return [{"phase": "dsof", "dataset": d, "horizon": h, "student": st, "gate": g, "seed": s, "teacher": "timesfm"}
            for d in STAGE2_DSOF_DATASETS for h in HORIZONS for st, g in STAGE2_DSOF_VARIANTS for s in SEEDS]


def write_stage2_jobs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = {
        "jobs_stage2_foundation.jsonl": stage2_foundation_jobs(),
        "jobs_stage2_teacher.jsonl": stage2_teacher_jobs(),  # run before jobs_stage2_dsof.jsonl
        "jobs_stage2_dsof.jsonl": stage2_dsof_jobs(),
    }
    for name, jobs in sets.items():
        (CONFIG_DIR / name).write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
        print(f"{name}: {len(jobs)} jobs")


def stage2_check(set_label: str) -> None:
    from srp26.bench import dsof, foundation

    root = REPO_ROOT / "results" / "v2" / set_label
    for name, jobs, path_of in (
        ("foundation", stage2_foundation_jobs(),
         lambda j: foundation.result_path(root, j["dataset"], j["horizon"], j["model"], j["fourier"], j["seed"])),
        ("dsof", stage2_dsof_jobs(),
         lambda j: dsof.result_path(root, j["dataset"], j["horizon"], j["student"], j["gate"], j["seed"]))):
        missing = [j for j in jobs if not path_of(j).exists()]
        print(f"{set_label}/{name}: {len(jobs) - len(missing)}/{len(jobs)} complete")
        for j in missing[:20]:
            print("  missing:", j)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    sub.add_parser("stage2-jobs")
    s2 = sub.add_parser("stage2-check")
    s2.add_argument("--set", required=True)
    s = sub.add_parser("select-lr")
    s.add_argument("--set", required=True)
    c = sub.add_parser("check")
    c.add_argument("--set", required=True)
    c.add_argument("--phase", default="final")
    c.add_argument("--jobs", default="jobs_final_bigmama.jsonl")
    a = p.parse_args()
    if a.cmd == "jobs":
        write_jobs()
    elif a.cmd == "stage2-jobs":
        write_stage2_jobs()
    elif a.cmd == "stage2-check":
        stage2_check(a.set)
    elif a.cmd == "select-lr":
        select_lr(a.set)
    else:
        check(a.set, a.phase, a.jobs)


if __name__ == "__main__":
    main()

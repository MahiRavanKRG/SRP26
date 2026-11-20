"""Aggregate v2 results into tables (protocol §3, §5 P5).

    uv run python -m srp26.bench.report --set bigmama            # runs.csv + summary.md
    uv run python -m srp26.bench.report --compare local bigmama  # cross-machine agreement
    uv run python -m srp26.bench.report --foundation-compare bigmama  # zero-shot vs trained, same windows
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from srp26.bench.env import REPO_ROOT

RESULTS = REPO_ROOT / "results" / "v2"
BAND_COLS = [f"{b}_{k}" for b in ("level", "slow", "daily", "fast") for k in ("mse", "fvu", "energy_ratio", "lsd_db")]


def load_runs(set_label: str) -> pd.DataFrame:
    rows = []
    for p in (RESULTS / set_label / "final").rglob("*.json"):
        r = json.loads(p.read_text())
        m = r.get("metrics", {})
        rows.append({
            "dataset": r["dataset"], "pred_len": r["pred_len"], "model": r["model"], "seed": r["seed"],
            "lr": r["hparams"]["lr"], "params": r["params"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "train_time_sec": r["train_time_sec"], "val_mse": r["val_mse"],
            "mse": m.get("mse"), "mae": m.get("mae"), "skill_mse": m.get("skill_mse"), "skill_mae": m.get("skill_mae"),
            **{c: m.get(c) for c in BAND_COLS},
            "periods": r["model_info"].get("periods"), "channel_dependence": r["model_info"].get("channel_dependence"),
            "daily_cycle_resolvable": r["daily_cycle_resolvable"],
            "commit": r["provenance"]["commit"][:10], "gpu": r["provenance"]["gpu"], "torch": r["provenance"]["torch"],
        })
    if not rows:
        raise SystemExit(f"No final results under {RESULTS / set_label}")
    return pd.DataFrame(rows).sort_values(["dataset", "pred_len", "model", "seed"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "pred_len", "model"])
    out = g.agg(n=("seed", "size"), mse_mean=("mse", "mean"), mse_std=("mse", "std"),
                mae_mean=("mae", "mean"), mae_std=("mae", "std"), skill_mse=("skill_mse", "mean"),
                params=("params", "first")).reset_index()
    out["mse_std"] = out["mse_std"].fillna(0.0)
    out["mae_std"] = out["mae_std"].fillna(0.0)
    return out


def wide_table(summary: pd.DataFrame, metric: str = "mse") -> pd.DataFrame:
    s = summary.copy()
    s["cell"] = s.apply(lambda r: f"{r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}", axis=1)
    return s.pivot_table(index=["dataset", "pred_len"], columns="model", values="cell", aggfunc="first")


def write_set_report(set_label: str) -> None:
    df = load_runs(set_label)
    out_dir = RESULTS / set_label
    df.to_csv(out_dir / "runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    commits = sorted(df["commit"].unique())
    lines = [f"# v2 results: `{set_label}`", "",
             f"- Runs: {len(df)} | commits: {commits} | GPUs: {sorted(df['gpu'].unique())} | torch: {sorted(df['torch'].unique())}",
             "- Values: test MSE mean ± std over seeds (untrained references: one deterministic run).", ""]
    for metric in ("mse", "mae"):
        lines += [f"## {metric.upper()}", "", wide_table(summary, metric).to_markdown(), ""]
    skill = summary.pivot_table(index=["dataset", "pred_len"], columns="model", values="skill_mse").round(3)
    lines += ["## Skill vs seasonal naive (1 − MSE/MSE_snaive; > 0 beats naive)", "", skill.to_markdown(), ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{set_label}: {len(df)} runs -> {out_dir / 'summary.md'}")


def compare(a: str, b: str) -> None:
    sa, sb = summarize(load_runs(a)), summarize(load_runs(b))
    m = sa.merge(sb, on=["dataset", "pred_len", "model"], suffixes=(f"_{a}", f"_{b}"))
    m["diff"] = m[f"mse_mean_{a}"] - m[f"mse_mean_{b}"]
    m["pooled_std"] = np.sqrt((m[f"mse_std_{a}"] ** 2 + m[f"mse_std_{b}"] ** 2) / 2)
    m["rel_diff_pct"] = 100 * m["diff"] / m[f"mse_mean_{b}"]
    # agreement rule (protocol P2): means within the seed spread; deterministic references must match to 1e-4
    tol = np.maximum(2 * m["pooled_std"], 1e-4)
    m["agree"] = m["diff"].abs() <= tol
    out = RESULTS / f"compare_{a}_vs_{b}.csv"
    m.to_csv(out, index=False)
    print(f"{len(m)} cells compared; agree within seed spread: {int(m['agree'].sum())}/{len(m)}; "
          f"median |rel diff| {m['rel_diff_pct'].abs().median():.3f}%  -> {out.relative_to(REPO_ROOT)}")
    bad = m[~m["agree"]].sort_values("rel_diff_pct", key=lambda s: s.abs(), ascending=False)
    if len(bad):
        print(bad[["dataset", "pred_len", "model", f"mse_mean_{a}", f"mse_mean_{b}", "pooled_std", "rel_diff_pct"]]
              .head(20).to_string(index=False))


def rescore_on_windows(trained_set: str, dataset: str, horizon: int, indices, n_test: int) -> pd.DataFrame:
    """Re-score every trained final run of (dataset, H) on the given test-window indices.

    Uses the saved stride-1 per-window losses: MSE from ``*.wmse.npy`` and MAE
    from ``*.wmae.npy`` (MAE is NaN for runs saved before ``.wmae.npy`` existed).
    All windows have the same size, so the mean of per-window MSE over the
    subset equals the MSE over those windows.
    """
    idx = np.asarray(indices, dtype=np.int64)
    rows = []
    for p in sorted((RESULTS / trained_set / "final" / dataset / f"H{horizon}").glob("*/s*.json")):
        wmse_p = p.with_suffix(".wmse.npy")
        if not wmse_p.exists():
            continue
        wmse = np.load(wmse_p)
        if len(wmse) != n_test:
            print(f"[warn] {p.relative_to(RESULTS)}: {len(wmse)} windows, expected {n_test}; skipped")
            continue
        wmae_p = p.with_suffix(".wmae.npy")
        mae = float(np.load(wmae_p)[idx].astype(np.float64).mean()) if wmae_p.exists() else float("nan")
        r = json.loads(p.read_text())
        rows.append({"dataset": dataset, "pred_len": horizon, "model": r["model"], "seed": r["seed"],
                     "kind": "trained", "n_windows": len(idx), "mse": float(wmse[idx].astype(np.float64).mean()),
                     "mae": mae, "mse_full_test": r["metrics"]["mse"]})
    return pd.DataFrame(rows)


def foundation_compare(foundation_set: str, trained_set: str = None) -> pd.DataFrame:
    """Foundation (zero-shot) and trained models on identical stride-H test windows."""
    trained_set = trained_set or foundation_set
    rows, done = [], set()
    for p in sorted((RESULTS / foundation_set / "foundation").rglob("s*.json")):
        r = json.loads(p.read_text())
        ew = r["eval_windows"]
        rows.append({"dataset": r["dataset"], "pred_len": r["pred_len"], "model": r["model"], "seed": r["seed"],
                     "kind": "foundation", "n_windows": ew["n"], "mse": r["metrics"]["mse"],
                     "mae": r["metrics"]["mae"], "mse_full_test": None})
        key = (r["dataset"], r["pred_len"], tuple(ew["indices"]))
        if ew.get("max_windows") is None and key not in done:  # full stride-H set only
            done.add(key)
            rows += rescore_on_windows(trained_set, r["dataset"], r["pred_len"], ew["indices"],
                                       ew["n_test_stride1"]).to_dict("records")
    if not rows:
        raise SystemExit(f"No foundation results under {RESULTS / foundation_set / 'foundation'}")
    df = pd.DataFrame(rows).sort_values(["dataset", "pred_len", "kind", "model", "seed"])
    out = RESULTS / foundation_set / f"foundation_vs_trained_{trained_set}.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} rows -> {out.relative_to(REPO_ROOT)}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set")
    p.add_argument("--compare", nargs=2)
    p.add_argument("--foundation-compare", metavar="SET",
                   help="Score trained final runs on the foundation models' stride-H windows.")
    p.add_argument("--trained-set", help="Set holding the trained final runs (default: same as --foundation-compare).")
    a = p.parse_args()
    if a.foundation_compare:
        foundation_compare(a.foundation_compare, a.trained_set)
    elif a.compare:
        compare(*a.compare)
    else:
        write_set_report(a.set)


if __name__ == "__main__":
    main()

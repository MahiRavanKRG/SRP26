"""Aggregate transform-ablation results into comparison tables + significance.

Reads ``transform_ablation_summary.csv`` and the ``raw/*_wmse.npy`` per-window
MSE vectors, then writes ``transform_ablation_report.md`` with:

1. Mean ± std MSE per (dataset, horizon), ranked by mean.
2. A paired Wilcoxon signed-rank test (per test window) of each transform vs
   the identity transform, which is the statistically valid way to claim "X
   beats the time domain" despite only 3 training seeds.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def _load_wmse(raw_dir: str, ds: str, h: int, transform: str, seeds) -> np.ndarray:
    chunks = []
    for s in seeds:
        p = os.path.join(raw_dir, f"{ds}_H{h}_{transform}_s{s}_wmse.npy")
        if os.path.exists(p):
            chunks.append(np.load(p))
    if not chunks:
        return np.array([])
    return np.concatenate(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="results/transform_ablation")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    save_dir = args.save_dir
    csv_path = args.csv or os.path.join(save_dir, "transform_ablation_summary.csv")
    raw_dir = os.path.join(save_dir, "raw")
    df = pd.read_csv(csv_path)
    seeds = sorted(df["seed"].unique().tolist())

    lines = ["# Transform-Ablation Summary\n"]
    lines.append(f"- Runs: {len(df)} | Seeds: {seeds} | Transforms: {sorted(df['transform'].unique())}\n")

    # 1. Mean ± std MSE, ranked, per (dataset, horizon).
    for ds, g_ds in df.groupby("dataset"):
        lines.append(f"\n## Dataset: `{ds}`\n")
        for h, g in g_ds.groupby("pred_len"):
            agg = g.groupby("transform")["mse"].agg(["mean", "std"]).sort_values("mean")
            agg.columns = ["mean_mse", "std_mse"]
            lines.append(f"\n### H={h} (ranked by mean MSE)\n")
            lines.append(agg.round(6).to_markdown() + "\n")
            best = agg.index[0]
            lines.append(f"**Best:** `{best}` — mean MSE {agg.iloc[0]['mean_mse']:.6f} ± {agg.iloc[0]['std_mse']:.6f}\n")

    # 2. Paired Wilcoxon vs identity (per-window MSE).
    lines.append("\n## Paired Wilcoxon signed-rank vs identity (per-window MSE)\n")
    lines.append("p < 0.05 = transform differs from the time-domain (identity) baseline.\n")
    sig_rows = []
    for ds in sorted(df["dataset"].unique()):
        for h in sorted(df[df["dataset"] == ds]["pred_len"].unique()):
            id_ref = _load_wmse(raw_dir, ds, h, "identity", seeds)
            if id_ref.size == 0:
                continue
            for t in sorted(df[df["dataset"] == ds]["transform"].unique()):
                if t == "identity":
                    continue
                t_wmse = _load_wmse(raw_dir, ds, h, t, seeds)
                if t_wmse.size == 0:
                    continue
                n = min(id_ref.size, t_wmse.size)
                a, b = id_ref[:n], t_wmse[:n]
                # Remove exact ties (Wilcoxon requires non-zero differences).
                diff = b - a
                diff = diff[np.abs(diff) > 1e-12]
                if diff.size < 5:
                    continue
                try:
                    stat, p = stats.wilcoxon(b[: diff.size], a[: diff.size], alternative="two-sided")
                except ValueError:
                    continue
                sig_rows.append(
                    {
                        "dataset": ds,
                        "pred_len": h,
                        "transform": t,
                        "n_windows": int(diff.size),
                        "p_value": round(float(p), 6),
                        "significant": bool(p < 0.05),
                    }
                )
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        lines.append(sig_df.to_markdown(index=False) + "\n")
    else:
        lines.append("(no identity baseline present — run the `identity` transform to enable significance.)\n")

    out_path = os.path.join(save_dir, "transform_ablation_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()

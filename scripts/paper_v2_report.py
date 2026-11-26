"""Build every v2 number used in ``paper/main.tex`` (protocol docs/controlled_rerun_protocol.md, P5/P6).

    uv run --no-sync python scripts/paper_v2_report.py --results D:/uni/SRP26/results/v2

Reads the immutable v2 result files and writes

* ``results/v2_paper/*.csv``     machine-readable aggregates (committed, small)
* ``paper/tables/v2_*.tex``      generated LaTeX tables (never edit by hand)
* ``paper/tables/v2_numbers.tex`` ``\\newcommand`` macros for every number quoted in the prose
* ``paper/figures/data/*.csv``   series plotted by pgfplots

Nothing in the paper's results sections is typed by hand: rerun this script after new
results arrive and recompile.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_CSV = REPO / "results" / "v2_paper"
OUT_TEX = REPO / "paper" / "tables"
OUT_FIG = REPO / "paper" / "figures" / "data"
PUBLISHED = REPO / "results" / "published" / "hdmixer_aaai24_table1.csv"

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic", "solar"]
DNAME = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2", "weather": "Weather",
         "electricity": "Electricity", "traffic": "Traffic", "solar": "Solar"}
HORIZONS = [96, 192, 336, 720]
SEEDS = [2024, 2025, 2026]
BANDS = ["level", "slow", "daily", "fast"]
MAIN_MODELS = ["seasonal_naive", "linear", "dlinear", "cyclenet_fixed", "cyclenet_pro_linear",
               "cyclenet_pro_mlp", "cyclenet_pro_cnn", "cyclenet_pro_sor", "patchtst"]
MNAME = {"seasonal_naive": "S-Naive", "repeat_last": "Repeat", "linear": "Linear", "dlinear": "DLinear",
         "cyclenet_fixed": "CycleNet$_{\\text{fix}}$", "cyclenet_pro_linear": "Pro-Lin", "cyclenet_pro_mlp": "Pro-MLP",
         "cyclenet_pro_cnn": "Pro-CNN", "cyclenet_pro_sor": "Pro-SOR", "cyclenet_pro_linear_ci": "Pro-Lin$_{\\text{CI}}$",
         "cyclenet_pro_mlp_ci": "Pro-MLP$_{\\text{CI}}$", "cyclenet_pro_cnn_ci": "Pro-CNN$_{\\text{CI}}$",
         "cyclenet_pro_sor_ci": "Pro-SOR$_{\\text{CI}}$", "patchtst": "PatchTST", "tf_rfft": "rFFT", "tf_dct": "DCT",
         "tf_stft": "STFT", "tf_dwt_haar": "DWT-Haar", "tf_dwt_db4": "DWT-db4"}
TRANSFORMS = ["linear", "dlinear", "tf_rfft", "tf_dct", "tf_dwt_haar", "tf_dwt_db4", "tf_stft"]
BACKBONES = ["linear", "mlp", "cnn", "sor"]
MACROS: dict = {}


# ----------------------------------------------------------------------------- helpers

def macro(name: str, value, fmt: str = "{}") -> None:
    assert name.isalpha(), name
    text = fmt.format(value) if not isinstance(value, str) else value
    if not isinstance(value, str) and text.startswith("-"):
        text = "\\ensuremath{-}" + text[1:]  # typographic minus, not a hyphen
    MACROS[name] = text


def dkey(d: str) -> str:
    """Dataset name usable inside a macro name (digits spelled out): ETTh1 -> ETThOne."""
    return DNAME[d].replace("1", "One").replace("2", "Two")


def f4(x) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def pct(x, digits=1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{digits}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- loading

def load_final(root: Path, set_label: str) -> pd.DataFrame:
    rows = []
    for p in (root / set_label / "final").rglob("s*.json"):
        r = json.loads(p.read_text())
        m = r["metrics"]
        row = {"dataset": r["dataset"], "H": r["pred_len"], "model": r["model"], "seed": r["seed"],
               "mse": m["mse"], "mae": m["mae"], "skill_mse": m.get("skill_mse"), "snaive_mse": m.get("snaive_mse"),
               "params": r["params"], "lr": r["hparams"]["lr"], "epochs_run": r["epochs_run"],
               "periods": "/".join(map(str, r["model_info"].get("periods") or [])),
               "channel_dependence": r["model_info"].get("channel_dependence"),
               "daily_resolvable": r["daily_cycle_resolvable"], "parseval_abs_error": m["parseval_abs_error"],
               "gpu": r["provenance"]["gpu"], "commit": r["provenance"]["commit"][:7], "dirty": r["provenance"]["dirty"],
               "n_test": r["split"]["n_test"], "path": str(p)}
        for b in BANDS:
            for k in ("mse", "fvu", "energy_ratio"):
                row[f"{b}_{k}"] = m.get(f"{b}_{k}")
        row["mse_by_step"] = m["mse_by_step"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"no final results under {root / set_label}")
    return pd.DataFrame(rows).sort_values(["dataset", "H", "model", "seed"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "H", "model"])
    s = g.agg(n=("seed", "size"), mse=("mse", "mean"), mse_std=("mse", "std"), mae=("mae", "mean"),
              mae_std=("mae", "std"), skill=("skill_mse", "mean"), params=("params", "first")).reset_index()
    return s.fillna({"mse_std": 0.0, "mae_std": 0.0})


def window_losses(df: pd.DataFrame, dataset: str, H: int, model: str) -> dict:
    """seed -> per-window MSE vector (stride-1 test windows)."""
    out = {}
    for _, r in df[(df.dataset == dataset) & (df.H == H) & (df.model == model)].iterrows():
        p = Path(r.path).with_suffix(".wmse.npy")
        if p.exists():
            out[int(r.seed)] = np.load(p).astype(np.float64)
    return out


# ----------------------------------------------------------------------------- statistics

def dm_test(d: np.ndarray, h: int) -> tuple:
    """Test of equal expected loss (Diebold-Mariano type) with a gapped batch-means variance.

    Stride-1 windows of length h overlap, so per-window losses are strongly autocorrelated
    and the effective sample size is about n/h. A Newey-West DM statistic with h-1 lags was
    found to over-reject badly in simulation (17% at nominal 5% for h=96, worse for larger h).
    Instead the loss-difference series is cut into blocks of h consecutive windows separated
    by gaps of h windows, so no two blocks share a forecast target step; the block means are
    compared with a one-sample t-test (B-1 df). Simulated size on MA(h-1) noise: 4-5%.
    Returns (mean difference, t statistic, two-sided p, number of blocks); p = 1 if B < 2.
    """
    from scipy import stats
    n_blocks = len(d) // (2 * h)
    if n_blocks < 2:
        return d.mean(), float("nan"), 1.0, n_blocks
    m = d[: n_blocks * 2 * h].reshape(n_blocks, 2 * h)[:, :h].mean(axis=1)
    sd = m.std(ddof=1)
    if sd == 0:
        return d.mean(), float("nan"), 1.0, n_blocks
    t = m.mean() / (sd / math.sqrt(n_blocks))
    return d.mean(), t, float(2 * stats.t.sf(abs(t), n_blocks - 1)), n_blocks


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_test(df: pd.DataFrame, s: pd.DataFrame, family: str, a: str, b: str, dataset: str, H: int) -> dict:
    la, lb = window_losses(df, dataset, H, a), window_losses(df, dataset, H, b)
    seeds = sorted(set(la) & set(lb))
    row = {"family": family, "a": a, "b": b, "dataset": dataset, "H": H, "n_seeds": len(seeds)}
    if not seeds:
        return {**row, "mse_a": np.nan, "mse_b": np.nan, "rel_pct": np.nan, "t_stat": np.nan, "p": np.nan,
                "n_blocks": 0, "seed_wins_a": np.nan}
    d = np.mean([la[k] - lb[k] for k in seeds], axis=0)  # seed-averaged per-window loss difference
    mean_d, stat, p, n_blocks = dm_test(d, H)
    sa = s[(s.dataset == dataset) & (s.H == H) & (s.model == a)].iloc[0]
    sb = s[(s.dataset == dataset) & (s.H == H) & (s.model == b)].iloc[0]
    per_seed = df[(df.dataset == dataset) & (df.H == H)].pivot_table(index="seed", columns="model", values="mse")
    wins = int(sum(per_seed.loc[k, a] < per_seed.loc[k, b] for k in seeds))
    return {**row, "mse_a": sa.mse, "mse_b": sb.mse, "rel_pct": 100 * (sa.mse - sb.mse) / sb.mse,
            "t_stat": stat, "p": p, "n_blocks": n_blocks, "seed_wins_a": wins}


# ----------------------------------------------------------------------------- LaTeX tables

def mark_row(vals: list) -> list:
    """Return 'bold' / 'under' / '' per value (lowest MSE bold, second underlined)."""
    finite = sorted({v for v in vals if v is not None and not math.isnan(v)})
    best = finite[0] if finite else None
    second = finite[1] if len(finite) > 1 else None
    return ["bold" if v == best else "under" if v == second else "" for v in vals]


def cell(mean, std, n, style, std_digits=4, deterministic=False) -> str:
    if mean is None or math.isnan(mean):
        return "--"
    core = f"{mean:.4f}"
    if style == "bold":
        core = f"\\textbf{{{core}}}"
    elif style == "under":
        core = f"\\underline{{{core}}}"
    if deterministic:  # untrained reference: one seed-free run, no spread
        return core
    dag ="$^{\\dagger}$" if n < 3 else ""
    return f"{core}{dag}\\,{{\\tiny$\\pm${std:.{std_digits}f}}}"


def accuracy_table(s: pd.DataFrame, models: list, metric: str = "mse") -> str:
    cols = "ll" + "c" * len(models)
    head = " & ".join(["\\textbf{Data}", "$H$"] + [f"\\textbf{{{MNAME[m]}}}" for m in models])
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head + " \\\\", "\\midrule"]
    for d in DATASETS:
        for i, H in enumerate(HORIZONS):
            sub = s[(s.dataset == d) & (s.H == H)].set_index("model")
            vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in models]
            styles = mark_row(vals)
            cells = []
            for m, v, st in zip(models, vals, styles):
                if m in sub.index:
                    cells.append(cell(v, sub.loc[m, f"{metric}_std"], sub.loc[m, "n"], st,
                                      deterministic=m in ("seasonal_naive", "repeat_last")))
                else:
                    cells.append("--")
            label = f"\\multirow{{4}}{{*}}{{{DNAME[d]}}}" if i == 0 else ""
            lines.append(" & ".join([label, str(H)] + cells) + " \\\\")
        lines.append("\\midrule" if d != DATASETS[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main analysis

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "v2"), help="root holding <set>/final, foundation, dsof")
    ap.add_argument("--set", default="bigmama")
    ap.add_argument("--local-set", default="local")
    a = ap.parse_args()
    root = Path(a.results)
    for d in (OUT_CSV, OUT_TEX, OUT_FIG):
        d.mkdir(parents=True, exist_ok=True)

    df = load_final(root, a.set)
    s = summarize(df)
    df.drop(columns=["mse_by_step", "path"]).to_csv(OUT_CSV / "runs_final.csv", index=False)
    s.to_csv(OUT_CSV / "summary_final.csv", index=False)

    # ---------------- completeness and provenance
    jobs = [json.loads(l) for l in (REPO / "configs" / "v2" / "jobs_final_bigmama.jsonl").read_text().splitlines() if l.strip()]
    expected = {(j["dataset"].replace(".csv", ""), j["horizon"], j["model"], j["seed"]) for j in jobs}
    have = set(zip(df.dataset, df.H, df.model, df.seed))
    missing = sorted(expected - have)
    pd.DataFrame(missing, columns=["dataset", "H", "model", "seed"]).to_csv(OUT_CSV / "missing_final.csv", index=False)
    macro("vRunsFinal", len(have & expected))
    macro("vRunsExpected", len(expected))
    macro("vRunsMissing", len(missing))
    macro("vMissingModels", ", ".join(sorted({MNAME[m[2]] for m in missing})) if missing else "none")
    inc = {}
    for d_, h_, m_, _ in missing:
        inc.setdefault(m_, {}).setdefault(d_, set()).add(h_)
    parts = []
    for m_ in sorted(inc, key=lambda k: MNAME[k]):
        cells = "; ".join(f"{DNAME[d_]} $H\\in\\{{{','.join(map(str, sorted(hs)))}\\}}$" for d_, hs in sorted(inc[m_].items()))
        parts.append(f"{MNAME[m_]} ({cells})")
    macro("vIncompleteCells", ", ".join(parts) if parts else "none")
    complete = lambda models: not any(m[2] in models for m in missing)
    macro("vCyclePeriodComplete", "complete" if complete({"linear", "cyclenet_fixed", "cyclenet_pro_linear_ci"}) else "not complete")
    macro("vMaxParseval", f"{df.parseval_abs_error.max():.1e}".replace("e-", "\\times 10^{-") + "}")
    macro("vDirtyRuns", int(df.dirty.sum()))
    gpu_counts = df.gpu.value_counts()
    macro("vGpuList", "; ".join(f"{g.replace('NVIDIA ', '')} ({c})" for g, c in gpu_counts.items()))
    lrsel = list((root / a.set / "lrsel").rglob("*.json"))
    macro("vRunsLrsel", len(lrsel))
    lr = json.loads((REPO / "configs" / "v2" / "selected_lr.json").read_text())
    macro("vLrCells", len(lr) if isinstance(lr, (list, dict)) else 0)

    # ---------------- main tables
    write(OUT_TEX / "v2_main_mse.tex", accuracy_table(s, MAIN_MODELS, "mse"))
    write(OUT_TEX / "v2_main_mae.tex", accuracy_table(s, MAIN_MODELS, "mae"))
    ci_models = ["cyclenet_fixed"] + [f"cyclenet_pro_{b}_ci" for b in BACKBONES] + ["patchtst"]
    write(OUT_TEX / "v2_ci_mse.tex", accuracy_table(s, ci_models, "mse"))
    write(OUT_TEX / "v2_transform_mse.tex", accuracy_table(s, TRANSFORMS, "mse"))

    P = s.pivot_table(index=["dataset", "H"], columns="model", values="mse")
    trained = [m for m in P.columns if m not in ("seasonal_naive", "repeat_last")]
    ranks = P[trained].rank(axis=1).mean().sort_values()
    best = P[trained].idxmin(axis=1)
    ranks.rename("mean_rank").to_csv(OUT_CSV / "mean_rank_trained.csv")
    best.rename("best_model").to_csv(OUT_CSV / "best_model_per_cell.csv")
    macro("vNumTrainedModels", len(trained))
    macro("vRankPatch", ranks["patchtst"], "{:.2f}")
    macro("vRankFixed", ranks["cyclenet_fixed"], "{:.2f}")
    macro("vRankProLin", ranks["cyclenet_pro_linear"], "{:.2f}")
    macro("vRankProLinCI", ranks["cyclenet_pro_linear_ci"], "{:.2f}")
    macro("vRankProMlpCI", ranks["cyclenet_pro_mlp_ci"], "{:.2f}")
    macro("vRankLinear", ranks["linear"], "{:.2f}")
    macro("vRankDLinear", ranks["dlinear"], "{:.2f}")
    macro("vRankCNN", ranks["cyclenet_pro_cnn"], "{:.2f}")
    macro("vBestPatch", int((best == "patchtst").sum()))
    macro("vBestCycleFamily", int(best.str.startswith("cyclenet").sum()))
    macro("vBestFixed", int((best == "cyclenet_fixed").sum()))
    macro("vBestTransform", int(best.str.startswith("tf_").sum()))
    skill_all = s[s.model.isin(trained)]
    macro("vSkillNeg", int((skill_all.skill <= 0).sum()))
    neg = skill_all[skill_all.skill <= 0]
    macro("vSkillNegWhere", "; ".join(f"{MNAME[r.model]} on {DNAME[r.dataset]}, $H={r.H}$" for _, r in neg.iterrows()) or "none")
    dl720 = s[(s.dataset == "ETTh1") & (s.H == 720) & (s.model == "dlinear")].iloc[0]
    macro("vDlinETThOneSevenTwentyStd", dl720.mse_std, "{:.4f}")
    macro("vDlinETThOneSevenTwenty", dl720.mse, "{:.4f}")
    macro("vSkillCells", len(skill_all))
    macro("vSkillMinFixed", s[s.model == "cyclenet_fixed"].skill.min(), "{:.2f}")
    macro("vSkillMaxFixed", s[s.model == "cyclenet_fixed"].skill.max(), "{:.2f}")
    for d in DATASETS:
        sub = s[(s.dataset == d) & (s.model == "cyclenet_fixed")]
        macro(f"vSkillFixed{dkey(d)}",
              sub.skill.mean(), "{:.2f}")

    # ---------------- contrasts with DM tests
    tests = []
    for d in DATASETS:
        for H in HORIZONS:
            tests.append(paired_test(df, s, "cycle", "cyclenet_fixed", "linear", d, H))
            tests.append(paired_test(df, s, "transform", "tf_rfft", "linear", d, H))
            tests.append(paired_test(df, s, "transform", "tf_dct", "linear", d, H))
            tests.append(paired_test(df, s, "patchtst", "patchtst", "cyclenet_fixed", d, H))
            fixed_p = df[(df.dataset == d) & (df.model == "cyclenet_fixed")].periods.iloc[0]
            auto_p = df[(df.dataset == d) & (df.model == "cyclenet_pro_linear_ci")].periods.iloc[0]
            if fixed_p != auto_p:
                tests.append(paired_test(df, s, "period", "cyclenet_pro_linear_ci", "cyclenet_fixed", d, H))
            cd_on = bool(df[(df.dataset == d) & (df.model == "cyclenet_pro_linear")].channel_dependence.iloc[0])
            if cd_on:
                for b in BACKBONES:
                    tests.append(paired_test(df, s, "mixing", f"cyclenet_pro_{b}", f"cyclenet_pro_{b}_ci", d, H))
    T = pd.DataFrame(tests)
    ok = T.p.notna()
    T.loc[ok, "p_holm"] = holm(T.loc[ok, "p"].to_numpy())
    T["sig"] = (T.p_holm < 0.05) & (T.n_seeds >= 2)
    T["consistent"] = T.sig & (((T.rel_pct < 0) & (T.seed_wins_a == T.n_seeds)) | ((T.rel_pct > 0) & (T.seed_wins_a == 0)))
    T.to_csv(OUT_CSV / "paired_tests.csv", index=False)
    macro("vNumTests", int(ok.sum()))
    macro("vNumUntestable", int((T.n_blocks < 2).sum()))
    macro("vUntestableWhere", ", ".join(sorted({f"{DNAME[r.dataset]} $H={r.H}$" for _, r in T[T.n_blocks < 2].iterrows()})) or "none")
    macro("vMinBlocks", int(T[T.n_blocks >= 2].n_blocks.min()))
    macro("vMaxBlocks", int(T.n_blocks.max()))

    def fam(name):
        return T[T.family == name]

    c = fam("cycle")
    macro("vCycleWins", int((c.rel_pct < 0).sum()))
    macro("vCycleCells", len(c))
    lost = c[c.rel_pct > 0]
    macro("vCycleLosses", len(lost))
    macro("vCycleLossWhere", ", ".join(f"{DNAME[r.dataset]} $H={r.H}$" for _, r in lost.iterrows()) or "none")
    macro("vCycleSigBetter", int((c.consistent & (c.rel_pct < 0)).sum()))
    macro("vCycleSigWorse", int((c.consistent & (c.rel_pct > 0)).sum()))
    cyc_by_d = c.groupby("dataset").rel_pct.mean()
    for d in DATASETS:
        macro(f"vCycle{dkey(d)}",
              cyc_by_d[d], "{:+.2f}")
    macro("vCycleETThTwoWorstH", c[(c.dataset == "ETTh2")].sort_values("rel_pct").iloc[-1].H)

    per = fam("period")
    per.to_csv(OUT_CSV / "period_discovery.csv", index=False)
    pe = per[per.dataset == "electricity"]
    macro("vPeriodElecMin", pe.rel_pct.min(), "{:.1f}")
    macro("vPeriodElecMax", pe.rel_pct.max(), "{:.1f}")
    macro("vPeriodElecSig", int((pe.consistent & (pe.rel_pct > 0)).sum()))
    macro("vPeriodSigBetter", int((per.consistent & (per.rel_pct < 0)).sum()))
    macro("vPeriodTests", int(per.p.notna().sum()))
    pt = per[(per.dataset == "traffic") & (per.H == 720)].iloc[0]
    macro("vPeriodTrafficSevenTwenty", pt.rel_pct, "{:+.2f}")
    macro("vPeriodTrafficSevenTwentySig", "significant" if pt.consistent else "not significant")
    pm = per[per.dataset == "ETTm2"]
    macro("vPeriodETTmTwoMaxAbs", pm.rel_pct.abs().max(), "{:.2f}")
    same = [d for d in DATASETS if d not in set(per.dataset)]
    macro("vPeriodSameCount", len(same))
    macro("vPeriodSameList", ", ".join(DNAME[d] for d in same))
    identical = all(
        np.allclose(s[(s.dataset == d) & (s.model == "cyclenet_fixed")].mse.to_numpy(),
                    s[(s.dataset == d) & (s.model == "cyclenet_pro_linear_ci")].mse.to_numpy(), rtol=0, atol=1e-6)
        for d in same)
    macro("vPeriodIdentical", "identical to within $10^{-6}$" if identical else "close but not identical")

    mix = fam("mixing")
    mix_rows = []
    for d in DATASETS:
        for b in BACKBONES:
            sub = mix[(mix.dataset == d) & (mix.a == f"cyclenet_pro_{b}")]
            if len(sub):
                mix_rows.append({"dataset": d, "backbone": b, "mean_rel_pct": sub.rel_pct.mean(),
                                 "wins_mixing": int((sub.rel_pct < 0).sum()), "cells": int(sub.rel_pct.notna().sum()),
                                 "sig_better": int((sub.consistent & (sub.rel_pct < 0)).sum()),
                                 "sig_worse": int((sub.consistent & (sub.rel_pct > 0)).sum()),
                                 "min_seeds": int(sub.n_seeds.min())})
    MX = pd.DataFrame(mix_rows)
    MX.to_csv(OUT_CSV / "channel_mixing.csv", index=False)
    macro("vMixDatasets", MX.dataset.nunique())
    macro("vMixCells", int(mix.rel_pct.notna().sum()))
    macro("vMixWins", int((mix.rel_pct < 0).sum()))
    macro("vMixSigBetter", int((mix.consistent & (mix.rel_pct < 0)).sum()))
    macro("vMixSigWorse", int((mix.consistent & (mix.rel_pct > 0)).sum()))
    worst = MX.sort_values("mean_rel_pct").iloc[-1]
    bestm = MX.sort_values("mean_rel_pct").iloc[0]
    macro("vMixWorstVal", worst.mean_rel_pct, "{:+.1f}")
    macro("vMixWorstWhere", f"{DNAME[worst.dataset]} with the {worst.backbone.upper() if worst.backbone != 'linear' else 'Linear'} backbone")
    macro("vMixBestVal", bestm.mean_rel_pct, "{:+.1f}")
    macro("vMixBestWhere", f"{DNAME[bestm.dataset]} with the {bestm.backbone.upper() if bestm.backbone != 'linear' else 'Linear'} backbone")

    # mixing table
    lines = ["\\begin{tabular}{l" + "c" * len(BACKBONES) + "}", "\\toprule",
             "\\textbf{Dataset} & " + " & ".join(f"\\textbf{{{b.upper() if b != 'linear' else 'Linear'}}}" for b in BACKBONES) + " \\\\",
             "\\midrule"]
    for d in DATASETS:
        sub = MX[MX.dataset == d].set_index("backbone")
        if not len(sub):
            continue
        cells = []
        for b in BACKBONES:
            r = sub.loc[b]
            mark = "$^{\\dagger}$" if r.min_seeds < 3 else ""
            cells.append(f"{r.mean_rel_pct:+.1f}\\% ({int(r.wins_mixing)}/{int(r.cells)}; "
                         f"{int(r.sig_better)}$\\uparrow$\\,{int(r.sig_worse)}$\\downarrow$){mark}")
        lines.append(DNAME[d] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(OUT_TEX / "v2_mixing.tex", "\n".join(lines) + "\n")

    # period table
    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "\\textbf{Dataset} & $H$ & \\textbf{Auto periods} & \\textbf{Known} & \\textbf{MSE auto} & \\textbf{MSE known} & \\textbf{$\\Delta$ (\\%)} \\\\",
             "\\midrule"]
    for d in [x for x in DATASETS if x in set(per.dataset)]:
        for _, r in per[per.dataset == d].iterrows():
            ap_ = df[(df.dataset == d) & (df.model == "cyclenet_pro_linear_ci")].periods.iloc[0].replace("/", ", ")
            kp = df[(df.dataset == d) & (df.model == "cyclenet_fixed")].periods.iloc[0]
            star = "$^{*}$" if r.consistent else ""
            lines.append(f"{DNAME[d]} & {r.H} & \\{{{ap_}\\}} & {kp} & {r.mse_a:.4f} & {r.mse_b:.4f} & {r.rel_pct:+.2f}{star} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    write(OUT_TEX / "v2_period.tex", "\n".join(lines) + "\n")

    # transforms
    tr_rows = []
    for m in TRANSFORMS[1:]:
        rel = 100 * (P[m] - P["linear"]) / P["linear"]
        tr_rows.append({"model": m, "mean_rel_pct": rel.mean(), "median_rel_pct": rel.median(),
                        "wins_vs_linear": int((rel < 0).sum()), "cells": int(rel.notna().sum()),
                        "mean_rank": P[TRANSFORMS].rank(axis=1)[m].mean(),
                        **{f"rel_{d}": rel.xs(d, level="dataset").mean() for d in DATASETS}})
    TR = pd.DataFrame(tr_rows)
    TR.to_csv(OUT_CSV / "transform_ablation.csv", index=False)
    macro("vRankTransformLinear", P[TRANSFORMS].rank(axis=1)["linear"].mean(), "{:.2f}")
    for m, nm in [("tf_rfft", "Rfft"), ("tf_dct", "Dct"), ("tf_stft", "Stft"), ("dlinear", "Dlin"),
                  ("tf_dwt_haar", "Haar"), ("tf_dwt_db4", "Dbfour")]:
        r = TR[TR.model == m].iloc[0]
        macro(f"vTr{nm}Mean", r.mean_rel_pct, "{:+.2f}")
        macro(f"vTr{nm}Median", r.median_rel_pct, "{:+.2f}")
        macro(f"vTr{nm}Wins", int(r.wins_vs_linear))
        macro(f"vTr{nm}Rank", r.mean_rank, "{:.2f}")
    trt = T[T.family == "transform"]
    for m, nm in [("tf_rfft", "Rfft"), ("tf_dct", "Dct")]:
        sub = trt[trt.a == m]
        macro(f"vTr{nm}SigBetter", int((sub.consistent & (sub.rel_pct < 0)).sum()))
        macro(f"vTr{nm}SigWorse", int((sub.consistent & (sub.rel_pct > 0)).sum()))
    rf_rel = 100 * (P["tf_rfft"] - P["linear"]) / P["linear"]
    macro("vTrRfftBestGain", -rf_rel.min(), "{:.1f}")
    macro("vTrRfftBestCell", f"{DNAME[rf_rel.idxmin()[0]]}, $H={rf_rel.idxmin()[1]}$")
    lin_sd = s[(s.model == "linear")].sort_values("mse_std").iloc[-1]
    macro("vLinOutlierCell", f"{DNAME[lin_sd.dataset]}, $H={lin_sd.H}$")
    macro("vLinOutlierStd", lin_sd.mse_std, "{:.4f}")
    lines = ["\\begin{tabular}{l" + "c" * len(DATASETS) + "ccc}", "\\toprule",
             "\\textbf{Transform} & " + " & ".join(f"\\textbf{{{DNAME[d]}}}" for d in DATASETS)
             + " & \\textbf{Mean} & \\textbf{Wins} & \\textbf{Rank} \\\\", "\\midrule",
             "Linear (identity) & " + " & ".join("0.0" for _ in DATASETS)
             + f" & 0.00 & -- & {P[TRANSFORMS].rank(axis=1)['linear'].mean():.2f} \\\\"]
    for _, r in TR.iterrows():
        lines.append(f"{MNAME[r.model]} & " + " & ".join(f"{r[f'rel_{d}']:+.1f}" for d in DATASETS)
                     + f" & {r.mean_rel_pct:+.2f} & {int(r.wins_vs_linear)}/{int(r.cells)} & {r.mean_rank:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(OUT_TEX / "v2_transform_summary.tex", "\n".join(lines) + "\n")

    pat = fam("patchtst")
    macro("vPatchWins", int((pat.rel_pct < 0).sum()))
    macro("vPatchSigBetter", int((pat.consistent & (pat.rel_pct < 0)).sum()))
    macro("vPatchSigWorse", int((pat.consistent & (pat.rel_pct > 0)).sum()))
    pbd = pat.groupby("dataset").rel_pct.mean()
    macro("vPatchSolar", pbd["solar"], "{:+.1f}")
    macro("vPatchWeather", pbd["weather"], "{:+.1f}")
    macro("vPatchElec", pbd["electricity"], "{:+.1f}")

    # ---------------- backbones (CI variants) and parameters
    bb_rows = []
    for d in DATASETS:
        for m in ["cyclenet_fixed"] + [f"cyclenet_pro_{b}_ci" for b in BACKBONES] + ["patchtst"]:
            sub = s[(s.dataset == d) & (s.model == m)]
            bb_rows.append({"dataset": d, "model": m, "avg_mse": sub.mse.mean(), "min_n": int(sub.n.min()),
                            "params_H96": int(sub[sub.H == 96].params.iloc[0]) if len(sub[sub.H == 96]) else np.nan,
                            "params_H720": int(sub[sub.H == 720].params.iloc[0]) if len(sub[sub.H == 720]) else np.nan})
    BB = pd.DataFrame(bb_rows)
    BB.to_csv(OUT_CSV / "backbones.csv", index=False)
    lines = ["\\begin{tabular}{l" + "r" * 6 + "}", "\\toprule",
             "\\textbf{Dataset} & " + " & ".join(f"\\textbf{{{MNAME[m]}}}" for m in
                                                ["cyclenet_fixed"] + [f"cyclenet_pro_{b}_ci" for b in BACKBONES] + ["patchtst"]) + " \\\\",
             "\\midrule", "\\multicolumn{7}{l}{\\textit{Average test MSE over $H\\in\\{96,192,336,720\\}$}} \\\\"]
    for d in DATASETS:
        sub = BB[BB.dataset == d].set_index("model")
        order = ["cyclenet_fixed"] + [f"cyclenet_pro_{b}_ci" for b in BACKBONES] + ["patchtst"]
        styles = mark_row([sub.loc[m, "avg_mse"] for m in order])
        cells = []
        for m, st in zip(order, styles):
            v = f"{sub.loc[m, 'avg_mse']:.4f}"
            v = f"\\textbf{{{v}}}" if st == "bold" else f"\\underline{{{v}}}" if st == "under" else v
            cells.append(v + ("$^{\\dagger}$" if sub.loc[m, "min_n"] < 3 else ""))
        lines.append(DNAME[d] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{7}{l}{\\textit{Trainable parameters at $H=96$ / $H=720$}} \\\\"]
    for d in ["ETTh1", "weather", "electricity", "traffic"]:
        sub = BB[BB.dataset == d].set_index("model")
        order = ["cyclenet_fixed"] + [f"cyclenet_pro_{b}_ci" for b in BACKBONES] + ["patchtst"]
        lines.append(DNAME[d] + " & " + " & ".join(f"{int(sub.loc[m, 'params_H96']):,} / {int(sub.loc[m, 'params_H720']):,}"
                                                  .replace(",", "{,}") for m in order) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(OUT_TEX / "v2_backbones.tex", "\n".join(lines) + "\n")
    bb_best = BB.loc[BB.groupby("dataset").avg_mse.idxmin()]
    macro("vBbPatchBestCount", int((bb_best.model == "patchtst").sum()))
    pe1 = BB[BB.dataset == "ETTh1"].set_index("model")
    macro("vParamCnnETThOne", int(pe1.loc["cyclenet_pro_cnn_ci", "params_H96"]), "{:,}")
    macro("vParamLinETThOne", int(pe1.loc["cyclenet_fixed", "params_H96"]), "{:,}")
    macro("vParamLinETThOneL", int(pe1.loc["cyclenet_fixed", "params_H720"]), "{:,}")
    macro("vParamMlpETThOne", int(pe1.loc["cyclenet_pro_mlp_ci", "params_H96"]), "{:,}")
    macro("vParamSorETThOne", int(pe1.loc["cyclenet_pro_sor_ci", "params_H96"]), "{:,}")
    macro("vParamPatchETThOne", int(pe1.loc["patchtst", "params_H96"]), "{:,}")
    pt_ = BB[BB.dataset == "traffic"].set_index("model")
    macro("vParamPatchTraffic", int(pt_.loc["patchtst", "params_H96"]), "{:,}")
    macro("vParamCnnTraffic", int(pt_.loc["cyclenet_pro_cnn_ci", "params_H96"]), "{:,}")
    for MM in ["cyclenet_pro_mlp_ci", "cyclenet_pro_sor_ci", "cyclenet_pro_cnn_ci"]:
        rel = 100 * (BB[BB.model == MM].set_index("dataset").avg_mse / BB[BB.model == "cyclenet_fixed"].set_index("dataset").avg_mse - 1)
        nm = {"cyclenet_pro_mlp_ci": "Mlp", "cyclenet_pro_sor_ci": "Sor", "cyclenet_pro_cnn_ci": "Cnn"}[MM]
        for d in DATASETS:
            macro(f"vBb{nm}{dkey(d)}",
                  rel[d], "{:+.1f}")

    # ---------------- band diagnostics
    bdf = df.copy()
    gb = bdf.groupby(["dataset", "H", "model"])[[f"{b}_{k}" for b in BANDS for k in ("mse", "fvu", "energy_ratio")] + ["mse"]].mean()
    gb.to_csv(OUT_CSV / "bands_mean.csv")
    band_models = ["seasonal_naive", "linear", "cyclenet_fixed", "cyclenet_pro_mlp_ci", "patchtst"]
    Hb = 336
    lines = ["\\begin{tabular}{ll" + "c" * 10 + "}", "\\toprule",
             "& & \\multicolumn{4}{c}{\\textbf{Share of MSE}} & \\multicolumn{3}{c}{\\textbf{FVU}} & \\multicolumn{3}{c}{\\textbf{Energy ratio}} \\\\",
             "\\cmidrule(lr){3-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-12}",
             "\\textbf{Data} & \\textbf{Model} & level & slow & daily & fast & slow & daily & fast & slow & daily & fast \\\\", "\\midrule"]
    for d in DATASETS:
        for i, m in enumerate(band_models):
            r = gb.loc[(d, Hb, m)]
            sh = [r[f"{b}_mse"] / r["mse"] if not pd.isna(r[f"{b}_mse"]) else np.nan for b in BANDS]
            fv = [r[f"{b}_fvu"] for b in ("slow", "daily", "fast")]
            er = [r[f"{b}_energy_ratio"] for b in ("slow", "daily", "fast")]
            fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
            label = f"\\multirow{{{len(band_models)}}}{{*}}{{{DNAME[d]}}}" if i == 0 else ""
            lines.append(" & ".join([label, MNAME[m]] + [fmt(v) for v in sh + fv + er]) + " \\\\")
        lines.append("\\midrule" if d != DATASETS[-1] else "\\bottomrule")
    lines.append("\\end{tabular}")
    write(OUT_TEX / "v2_bands.tex", "\n".join(lines) + "\n")

    tr_mask = gb.index.get_level_values("model").isin([m for m in trained])
    gtr = gb[tr_mask]
    level_share = (gtr["level_mse"] / gtr["mse"])
    macro("vLevelShareMin", level_share.min() * 100, "{:.0f}")
    macro("vLevelShareMax", level_share.max() * 100, "{:.0f}")
    for d in ["ETTh2", "ETTm2", "ETTh1", "electricity", "traffic"]:
        v = (gb.loc[(d, slice(None), "cyclenet_fixed"), "level_mse"] / gb.loc[(d, slice(None), "cyclenet_fixed"), "mse"]).mean()
        macro(f"vLevelShare{dkey(d)}", v * 100, "{:.0f}")
    fast_fvu_tr = gtr["fast_fvu"]
    macro("vFastFvuMin", fast_fvu_tr.min(), "{:.2f}")
    macro("vFastFvuMax", fast_fvu_tr.max(), "{:.2f}")
    macro("vFastFvuMedian", fast_fvu_tr.median(), "{:.2f}")
    sn = gb.xs("seasonal_naive", level="model")
    macro("vFastFvuSnaiveMin", sn.fast_fvu.min(), "{:.2f}")
    macro("vFastFvuSnaiveMax", sn.fast_fvu.max(), "{:.2f}")
    macro("vFastErMedian", gtr["fast_energy_ratio"].median(), "{:.2f}")
    macro("vFastEnergyPatchMax", gb.xs("patchtst", level="model").fast_energy_ratio.max(), "{:.2f}")
    macro("vFastEnergyPatchMin", gb.xs("patchtst", level="model").fast_energy_ratio.min(), "{:.2f}")
    fx = gb.xs("cyclenet_fixed", level="model")
    macro("vDailyFvuFixedETThOne", fx.loc["ETTh1"].daily_fvu.mean(), "{:.2f}")
    macro("vDailyFvuFixedETThTwo", fx.loc["ETTh2"].daily_fvu.mean(), "{:.2f}")
    macro("vDailyFvuFixedETTmOne", fx.loc["ETTm1"].daily_fvu.mean(), "{:.2f}")
    macro("vDailyErFixedMin", fx.daily_energy_ratio.min(), "{:.2f}")
    macro("vDailyErFixedMax", fx.daily_energy_ratio.max(), "{:.2f}")
    macro("vSlowFvuFixedMin", fx.slow_fvu.min(), "{:.2f}")
    macro("vSlowFvuFixedMax", fx.slow_fvu.max(), "{:.2f}")
    fvu_gt1 = gtr[gtr.fast_fvu > 1.0]
    macro("vFastFvuGtOne", len(fvu_gt1))
    macro("vFastFvuCells", int(gtr.fast_fvu.notna().sum()))
    pfast = gb.xs("patchtst", level="model").fast_fvu
    macro("vFastFvuPatchGtOne", int((pfast > 1).sum()))
    ffast = fx.fast_fvu
    macro("vFastFvuFixedGtOne", int((ffast > 1).sum()))

    # band decomposition of contrasts, summed over horizons
    def band_delta(ma, mb):
        A, B = gb.xs(ma, level="model").fillna(0.0), gb.xs(mb, level="model").fillna(0.0)
        out = pd.DataFrame({b: A[f"{b}_mse"] - B[f"{b}_mse"] for b in BANDS})
        out["total"] = A["mse"] - B["mse"]
        return out

    deltas = {"cycle": band_delta("cyclenet_fixed", "linear"),
              "period": band_delta("cyclenet_pro_linear_ci", "cyclenet_fixed"),
              "patchtst": band_delta("patchtst", "cyclenet_fixed")}
    for k, v in deltas.items():
        v.to_csv(OUT_CSV / f"band_delta_{k}.csv")
    lines = ["\\begin{tabular}{l" + "rrrrr" * 2 + "}", "\\toprule",
             "& \\multicolumn{5}{c}{\\textbf{CycleNet$_{\\text{fix}}$ $-$ Linear}} & \\multicolumn{5}{c}{\\textbf{PatchTST $-$ CycleNet$_{\\text{fix}}$}} \\\\",
             "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}",
             "\\textbf{Dataset} & level & slow & daily & fast & total & level & slow & daily & fast & total \\\\", "\\midrule"]
    for d in DATASETS:
        a1 = deltas["cycle"].loc[d].sum() * 1e3
        a2 = deltas["patchtst"].loc[d].sum() * 1e3
        lines.append(DNAME[d] + " & " + " & ".join(f"{a1[b]:+.1f}" for b in BANDS + ["total"]) + " & "
                     + " & ".join(f"{a2[b]:+.1f}" for b in BANDS + ["total"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(OUT_TEX / "v2_band_delta.tex", "\n".join(lines) + "\n")
    cyc = deltas["cycle"].groupby(level=0).sum()
    gain = cyc[cyc.total < 0]
    neg = gain[BANDS].clip(upper=0)
    share_daily_slow = (neg["daily"] + neg["slow"]).sum() / neg.sum(axis=1).sum()
    macro("vCycleGainDailySlowShare", share_daily_slow * 100, "{:.0f}")
    macro("vCycleETThTwoLevelShare", cyc.loc["ETTh2", "level"] / cyc.loc["ETTh2", "total"] * 100, "{:.0f}")
    el = deltas["period"].loc["electricity"].sum()
    macro("vPeriodElecSlowDailyShare", (el["slow"] + el["daily"]) / el["total"] * 100, "{:.0f}")
    macro("vPeriodElecSlowShare", el["slow"] / el["total"] * 100, "{:.0f}")
    ps = deltas["patchtst"].groupby(level=0).sum()
    macro("vPatchSolarLevelShare", ps.loc["solar", "level"] / ps.loc["solar", "total"] * 100, "{:.0f}")
    macro("vPatchSolarDailyShare", ps.loc["solar", "daily"] / ps.loc["solar", "total"] * 100, "{:.0f}")
    macro("vPatchWeatherLevelShare", ps.loc["weather", "level"] / ps.loc["weather", "total"] * 100, "{:.0f}")
    macro("vPatchETThOneFast", ps.loc["ETTh1", "fast"] * 1e3, "{:+.1f}")
    macro("vPatchETThOneTotal", ps.loc["ETTh1", "total"] * 1e3, "{:+.1f}")

    # ---------------- error vs lead time (figure data)
    lt_models = ["seasonal_naive", "linear", "cyclenet_fixed", "patchtst"]
    lead_rows = []
    for d in DATASETS:
        for m in lt_models:
            sub = df[(df.dataset == d) & (df.H == 720) & (df.model == m)]
            if not len(sub):
                continue
            curve = np.mean(np.stack(sub.mse_by_step.map(np.asarray).to_list()), axis=0)
            lead_rows.append({"dataset": d, "model": m, "step1": curve[0], "step720": curve[-1],
                              "ratio": curve[-1] / curve[0], "mean": curve.mean()})
            out = pd.DataFrame({"step": np.arange(1, 721), "mse": curve})
            out[(out.step - 1) % 10 == 0].to_csv(OUT_FIG / f"lead_{d}_{m}.csv", index=False)
    LT = pd.DataFrame(lead_rows)
    LT.to_csv(OUT_CSV / "lead_time_H720.csv", index=False)
    lfx = LT[LT.model == "cyclenet_fixed"].set_index("dataset")
    macro("vLeadRatioMin", lfx.ratio.min(), "{:.1f}")
    macro("vLeadRatioMax", lfx.ratio.max(), "{:.1f}")
    macro("vLeadRatioMinWhere", DNAME[lfx.ratio.idxmin()])
    macro("vLeadRatioMaxWhere", DNAME[lfx.ratio.idxmax()])
    lsn = LT[LT.model == "seasonal_naive"].set_index("dataset")
    cross = []
    for d in DATASETS:
        a_ = pd.read_csv(OUT_FIG / f"lead_{d}_cyclenet_fixed.csv")
        b_ = pd.read_csv(OUT_FIG / f"lead_{d}_seasonal_naive.csv")
        cross.append(int((a_.mse > b_.mse).sum()))
    macro("vLeadNaiveNeverBeaten", "never" if sum(cross) == 0 else f"in {sum(cross)} plotted steps")

    # ---------------- published references (separate table; not comparable as controlled results)
    pub = pd.read_csv(PUBLISHED, comment="#")
    pub = pub.rename(columns={"pred_len": "H"})
    ours = s[s.model.isin(["patchtst", "dlinear"])].pivot_table(index=["dataset", "H"], columns="model", values="mse").reset_index()
    pc = pub.merge(ours, on=["dataset", "H"], how="left")
    pc["patch_rel_pct"] = 100 * (pc["patchtst"] - pc["patchtst_mse"]) / pc["patchtst_mse"]
    pc["dlin_rel_pct"] = 100 * (pc["dlinear"] - pc["dlinear_mse"]) / pc["dlinear_mse"]
    pc.to_csv(OUT_CSV / "published_vs_ours.csv", index=False)
    macro("vPubPatchMedAbs", pc.patch_rel_pct.abs().median(), "{:.1f}")
    macro("vPubPatchWithinFive", int((pc.patch_rel_pct.abs() <= 5).sum()))
    macro("vPubCells", len(pc))
    e2 = pc[(pc.dataset == "ETTh2") & (pc.H == 720)].iloc[0]
    macro("vPubDlinETThTwo", e2.dlinear_mse, "{:.3f}")
    macro("vOurDlinETThTwo", e2.dlinear, "{:.4f}")
    lines = ["\\begin{tabular}{llcccccc}", "\\toprule",
             "& & \\multicolumn{3}{c}{\\textbf{Published (HDMixer AAAI-24, Tab.~1)}} & \\multicolumn{2}{c}{\\textbf{This work (v2)}} & \\\\",
             "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7}",
             "\\textbf{Data} & $H$ & HDMixer & PatchTST & DLinear & PatchTST & DLinear & $\\Delta_{\\text{PatchTST}}$ (\\%) \\\\", "\\midrule"]
    for d in [x for x in DATASETS if x in set(pc.dataset)]:
        for i, (_, r) in enumerate(pc[pc.dataset == d].iterrows()):
            label = f"\\multirow{{4}}{{*}}{{{DNAME[d]}}}" if i == 0 else ""
            lines.append(f"{label} & {r.H} & {r.hdmixer_mse:.3f} & {r.patchtst_mse:.3f} & {r.dlinear_mse:.3f} & "
                         f"{f4(r.patchtst)} & {f4(r.dlinear)} & {pct(r.patch_rel_pct)} \\\\")
        lines.append("\\midrule" if d != "traffic" else "\\bottomrule")
    if lines[-1] != "\\bottomrule":
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    write(OUT_TEX / "v2_published.tex", "\n".join(lines) + "\n")

    # ---------------- foundation models (zero-shot) on stride-H windows
    frows = []
    for p in (root / a.set / "foundation").rglob("s*.json"):
        r = json.loads(p.read_text())
        ew = r["eval_windows"]
        frows.append({"dataset": r["dataset"], "H": r["pred_len"], "model": r["model"], "kind": "foundation",
                      "mse": r["metrics"]["mse"], "mae": r["metrics"]["mae"], "n_windows": ew["n"],
                      "max_windows": ew.get("max_windows"), "indices": ew["indices"], "n_test": ew["n_test_stride1"]})
    F = pd.DataFrame(frows)
    ref_models = ["seasonal_naive", "linear", "cyclenet_fixed", "patchtst"]
    rs = []
    for (d, H), g in F.groupby(["dataset", "H"]):
        idx = np.asarray(g.indices.iloc[0], dtype=np.int64)
        for m in ref_models:
            wl = window_losses(df, d, H, m)
            vals = [v[idx].mean() for v in wl.values() if len(v) == g.n_test.iloc[0]]
            if vals:
                rs.append({"dataset": d, "H": H, "model": m, "kind": "trained", "mse": float(np.mean(vals)),
                           "n_windows": len(idx), "n_seeds": len(vals)})
    FR = pd.concat([F.drop(columns=["indices"]), pd.DataFrame(rs)], ignore_index=True)
    FR.to_csv(OUT_CSV / "foundation_vs_trained_strideH.csv", index=False)
    FP = FR.pivot_table(index=["dataset", "H"], columns="model", values="mse")
    nwin = F.groupby(["dataset", "H"]).n_windows.first()
    fm = ["timesfm", "chronos_bolt", "chronos_t5", "ttm", "fourier_base"]
    fcols = fm + ref_models
    fname = {"timesfm": "TimesFM", "chronos_bolt": "Chr.-Bolt", "chronos_t5": "Chr.-T5", "ttm": "TTM",
             "fourier_base": "Fourier", **{m: MNAME[m] for m in ref_models}}
    lines = ["\\begin{tabular}{llc" + "c" * len(fcols) + "}", "\\toprule",
             "& & & \\multicolumn{5}{c}{\\textbf{Zero-shot (no training)}} & \\multicolumn{4}{c}{\\textbf{Trained / naive, same windows}} \\\\",
             "\\cmidrule(lr){4-8} \\cmidrule(lr){9-12}",
             "\\textbf{Data} & $H$ & $n_w$ & " + " & ".join(f"\\textbf{{{fname[m]}}}" for m in fcols) + " \\\\", "\\midrule"]
    for d in DATASETS:
        for H in [96, 720]:
            row = FP.loc[(d, H)]
            vals = [row.get(m, np.nan) for m in fcols]
            styles = mark_row(vals)
            cells = []
            for v, st in zip(vals, styles):
                t = f4(v)
                if t != "--":
                    t = f"\\textbf{{{t}}}" if st == "bold" else f"\\underline{{{t}}}" if st == "under" else t
                cells.append(t)
            lines.append(f"{DNAME[d]} & {H} & {int(nwin.loc[(d, H)])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(OUT_TEX / "v2_foundation.tex", "\n".join(lines) + "\n")
    # who wins per cell among zero-shot+trained (excluding naive and fourier base)
    contenders = ["timesfm", "chronos_bolt", "chronos_t5", "linear", "cyclenet_fixed", "patchtst"]
    wins_tfm = int((FP[contenders].idxmin(axis=1) == "timesfm").sum())
    wins_trained = int(FP[contenders].idxmin(axis=1).isin(["linear", "cyclenet_fixed", "patchtst"]).sum())
    macro("vFoundCells", len(FP))
    macro("vFoundTimesfmBest", wins_tfm)
    macro("vFoundTrainedBest", wins_trained)
    macro("vFoundBoltBest", int((FP[contenders].idxmin(axis=1) == "chronos_bolt").sum()))
    macro("vFoundMinWindows", int(nwin.min()))
    macro("vFoundMaxWindows", int(nwin.max()))
    tfm_vs_fixed = 100 * (FP["timesfm"] - FP["cyclenet_fixed"]) / FP["cyclenet_fixed"]
    macro("vFoundTfmBeatsFixed", int((tfm_vs_fixed < 0).sum()))
    for base, nm in [("timesfm", "Tfm"), ("chronos_bolt", "Bolt"), ("chronos_t5", "Tfive"), ("ttm", "Ttm")]:
        rel = 100 * (FP[f"{base}_fourier"] - FP[base]) / FP[base]
        rel = rel.dropna()
        macro(f"vFourier{nm}Worse", int((rel > 0).sum()))
        macro(f"vFourier{nm}Cells", len(rel))
        macro(f"vFourier{nm}Median", rel.median(), "{:+.0f}")
    fb_vs_sn = (FP["fourier_base"] < FP["seasonal_naive"]).sum()
    macro("vFourierBaseBeatsSnaive", int(fb_vs_sn))

    # ---------------- DSOF-SD distillation
    drows = []
    for p in (root / a.set / "dsof").rglob("s*.json"):
        r = json.loads(p.read_text())
        drows.append({"dataset": r["dataset"], "H": r["pred_len"], "student": r["student"], "gate": r["gate"],
                      "seed": r["seed"], "mse": r["metrics"]["mse"], "warm": r["metrics_warm"]["mse"],
                      "teacher": r["metrics_teacher"]["mse"], "all_gates_zero": r["all_gates_zero"]})
    D = pd.DataFrame(drows)
    D.to_csv(OUT_CSV / "dsof_runs.csv", index=False)
    DS = D.groupby(["dataset", "H", "student", "gate"]).agg(mse=("mse", "mean"), sd=("mse", "std"), n=("seed", "size"),
                                                           zero=("all_gates_zero", "mean")).reset_index()
    DS.to_csv(OUT_CSV / "dsof_summary.csv", index=False)
    teacher = D.groupby(["dataset", "H"]).teacher.mean()
    gate_cols = [("linear", "control"), ("linear", "hard"), ("sor", "control"), ("sor", "hard"), ("sor", "soft"),
                 ("sor", "benefit"), ("sor", "consensus"), ("cnn", "control"), ("cnn", "benefit")]
    lines = ["\\begin{tabular}{ll" + "c" * (len(gate_cols) + 3) + "}", "\\toprule",
             "& & \\multicolumn{2}{c}{\\textbf{Linear student}} & \\multicolumn{5}{c}{\\textbf{SOR student}} & \\multicolumn{2}{c}{\\textbf{CNN student}} & & & \\\\",
             "\\cmidrule(lr){3-4} \\cmidrule(lr){5-9} \\cmidrule(lr){10-11}",
             "\\textbf{Data} & $H$ & ctrl & hard & ctrl & hard & soft & benefit & cons. & ctrl & benefit & \\textbf{TimesFM} & \\textbf{CycleNet$_{\\text{fix}}$} & \\textbf{Pro-SOR} \\\\",
             "\\midrule"]
    for d in ["ETTh1", "ETTh2"]:
        for H in HORIZONS:
            sub = DS[(DS.dataset == d) & (DS.H == H)].set_index(["student", "gate"])
            cells = []
            for st, gt in gate_cols:
                v = sub.loc[(st, gt), "mse"]
                ctrl = sub.loc[(st, "control"), "mse"]
                t = f"{v:.4f}"
                if gt != "control" and v < ctrl:
                    t = f"\\textbf{{{t}}}"
                cells.append(t)
            ref_fix = s[(s.dataset == d) & (s.H == H) & (s.model == "cyclenet_fixed")].mse.iloc[0]
            ref_sor = s[(s.dataset == d) & (s.H == H) & (s.model == "cyclenet_pro_sor")].mse.iloc[0]
            lines.append(f"{d} & {H} & " + " & ".join(cells) + f" & {teacher.loc[(d, H)]:.4f} & {ref_fix:.4f} & {ref_sor:.4f} \\\\")
        lines.append("\\midrule" if d == "ETTh1" else "\\bottomrule")
    lines.append("\\end{tabular}")
    write(OUT_TEX / "v2_dsof.tex", "\n".join(lines) + "\n")
    piv = DS.pivot_table(index=["dataset", "H", "student"], columns="gate", values="mse")
    sor = piv.xs("sor", level="student")
    rel_hard = 100 * (sor["hard"] - sor["control"]) / sor["control"]
    macro("vDsofSorHardEOneMin", -rel_hard.loc["ETTh1"].max(), "{:.1f}")
    macro("vDsofSorHardEOneMax", -rel_hard.loc["ETTh1"].min(), "{:.1f}")
    macro("vDsofSorHardETwoMin", rel_hard.loc["ETTh2"].min(), "{:+.1f}")
    macro("vDsofSorHardETwoMax", rel_hard.loc["ETTh2"].max(), "{:+.1f}")
    lin = piv.xs("linear", level="student")
    rel_lin = 100 * (lin["hard"] - lin["control"]) / lin["control"]
    macro("vDsofLinMaxAbs", rel_lin.abs().max(), "{:.1f}")
    sor_main = s[s.model == "cyclenet_pro_sor"].set_index(["dataset", "H"]).mse
    beats_main = int(sum(sor.loc[(d, H), "hard"] < sor_main.loc[(d, H)] for d in ["ETTh1", "ETTh2"] for H in HORIZONS))
    macro("vDsofSorHardBeatsMain", beats_main)
    ctrl_worse_main = int(sum(sor.loc[(d, H), "control"] > sor_main.loc[(d, H)] for d in ["ETTh1", "ETTh2"] for H in HORIZONS))
    macro("vDsofSorCtrlWorseMain", ctrl_worse_main)
    best_student = DS.groupby(["dataset", "H"]).mse.min()
    stud_beats_fixed = [(d, H) for d in ["ETTh1", "ETTh2"] for H in HORIZONS
                        if best_student.loc[(d, H)] < s[(s.dataset == d) & (s.H == H) & (s.model == "cyclenet_fixed")].mse.iloc[0]]
    macro("vDsofStudentBeatsFixed", len(stud_beats_fixed))
    macro("vDsofStudentBeatsFixedWhere", "; ".join(f"{d}, $H={H}$" for d, H in stud_beats_fixed) or "none")
    zero = D[D.gate.isin(["benefit", "consensus"])].groupby(["student", "gate"]).all_gates_zero.mean()
    macro("vDsofZeroGateMin", zero.min() * 100, "{:.0f}")
    macro("vDsofZeroGateMax", zero.max() * 100, "{:.0f}")
    fixed_main = s[s.model == "cyclenet_fixed"].set_index(["dataset", "H"]).mse
    tfm_full_worse = int(sum(teacher.loc[(d, H)] > fixed_main.loc[(d, H)] for d in ["ETTh1", "ETTh2"] for H in HORIZONS))
    macro("vTeacherWorseThanFixed", tfm_full_worse)
    macro("vTeacherETThOneNinetySix", teacher.loc[("ETTh1", 96)], "{:.4f}")
    macro("vFixedETThOneNinetySix", fixed_main.loc[("ETTh1", 96)], "{:.4f}")

    # ---------------- cross-machine reproduction (laptop vs cluster, ETT only)
    try:
        dl = load_final(root, a.local_set)
        sl = summarize(dl)
        m = sl.merge(s, on=["dataset", "H", "model"], suffixes=("_l", "_b"))
        m["diff"] = m.mse_l - m.mse_b
        m["pooled"] = np.sqrt((m.mse_std_l ** 2 + m.mse_std_b ** 2) / 2)
        m["agree"] = m["diff"].abs() <= np.maximum(2 * m.pooled, 1e-4)
        m["rel_pct"] = 100 * m["diff"] / m.mse_b
        m.to_csv(OUT_CSV / "cross_machine.csv", index=False)
        macro("vXmCells", len(m))
        macro("vXmAgree", int(m.agree.sum()))
        macro("vXmMedRel", m.rel_pct.abs().median(), "{:.3f}")
        macro("vXmMaxRel", m.rel_pct.abs().max(), "{:.2f}")
        macro("vXmRuns", len(dl))
        macro("vXmGpu", ", ".join(sorted(g.replace("NVIDIA ", "") for g in dl.gpu.unique())))
    except SystemExit:
        macro("vXmCells", 0)

    # ---------------- macros file
    lines = ["% Generated by scripts/paper_v2_report.py -- do not edit by hand."]
    for k in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
    write(OUT_TEX / "v2_numbers.tex", "\n".join(lines) + "\n")
    pd.Series(MACROS).to_csv(OUT_CSV / "paper_numbers.csv", header=["value"])
    print(f"final runs {len(df)}/{len(expected)}; {len(MACROS)} macros; tables -> {OUT_TEX}")


if __name__ == "__main__":
    main()

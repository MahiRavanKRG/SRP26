"""Generate every table and every number quoted in main_learnings.tex from the v2 results.

Usage (from the repo root):
    uv run python paper/v3_learnings/make_tables.py --v2 results/v2
Writes paper/v3_learnings/tables/*.tex and tables/numbers.tex (LaTeX macros).

Inputs (all produced by srp26.bench, harness v2):
    results/v2/bigmama/summary.csv                     seed mean/std per cell
    results/v2/bigmama/runs.csv                        per-seed runs incl. band MSE
    results/v2/bigmama/foundation_vs_trained_bigmama.csv  zero-shot models, stride-H windows
    results/v2/bigmama/dsof_runs_agg.csv               DSOF-SD runs (aggregated from dsof/*/*.json)
    results/v2/compare_local_vs_bigmama.csv            cross-machine check
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DS_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "solar", "traffic", "weather"]
DS_NAME = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2",
           "electricity": "Electricity", "solar": "Solar", "traffic": "Traffic", "weather": "Weather"}
MODEL_NAME = {
    "seasonal_naive": "S-Naive", "repeat_last": "Repeat", "linear": "Linear", "dlinear": "DLinear",
    "tf_rfft": "rFFT-Lin", "tf_dct": "DCT-Lin", "tf_dwt_haar": "Haar-Lin", "tf_dwt_db4": "db4-Lin",
    "tf_stft": "STFT-Lin", "cyclenet_fixed": "Cyc-Fixed", "cyclenet_pro_linear": "Cyc-Lin",
    "cyclenet_pro_linear_ci": "Cyc-Lin-CI", "cyclenet_pro_mlp": "Cyc-MLP", "cyclenet_pro_mlp_ci": "Cyc-MLP-CI",
    "cyclenet_pro_sor": "Cyc-SOR", "cyclenet_pro_sor_ci": "Cyc-SOR-CI", "cyclenet_pro_cnn": "Cyc-CNN",
    "cyclenet_pro_cnn_ci": "Cyc-CNN-CI", "patchtst": "PatchTST",
    "timesfm": "TimesFM-2.5", "chronos_bolt": "Chronos-Bolt", "chronos_t5": "Chronos-T5", "ttm": "TTM-R1",
    "fourier_base": "Fourier top-3",
}
MAIN_MODELS = ["seasonal_naive", "linear", "dlinear", "tf_rfft", "tf_dct", "tf_dwt_haar", "tf_stft",
               "cyclenet_fixed", "cyclenet_pro_linear_ci", "cyclenet_pro_mlp_ci", "cyclenet_pro_sor_ci",
               "cyclenet_pro_cnn_ci", "patchtst"]
BANDS = ["level_mse", "slow_mse", "daily_mse", "fast_mse"]


def ds_key(idx):
    return [DS_ORDER.index(d) for d in idx]


def fmt(v, nd=3):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def pct(v, nd=1):
    return "--" if pd.isna(v) else f"{v:+.{nd}f}"


def bold_best(row_vals, nd=3, second=True):
    """Return formatted cells with best bold and second-best underlined (lower is better)."""
    v = pd.Series(row_vals, dtype=float)
    order = v.dropna().sort_values().index.tolist()
    out = []
    for k, x in v.items():
        s = fmt(x, nd)
        if order and k == order[0]:
            s = r"\textbf{" + s + "}"
        elif second and len(order) > 1 and k == order[1]:
            s = r"\underline{" + s + "}"
        out.append(s)
    return out


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="results/v2")
    ap.add_argument("--out", default=str(Path(__file__).parent / "tables"))
    a = ap.parse_args()
    v2, out = Path(a.v2), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    macros = {}

    s = pd.read_csv(v2 / "bigmama/summary.csv")
    r = pd.read_csv(v2 / "bigmama/runs.csv")
    P = s.pivot_table(index=["dataset", "pred_len"], columns="model", values="mse_mean")
    Pstd = s.pivot_table(index=["dataset", "pred_len"], columns="model", values="mse_std")
    trained = [m for m in P.columns if m not in ("seasonal_naive", "repeat_last")]

    macros["nRuns"] = str(len(r))
    macros["nModels"] = str(len(P.columns))
    macros["nTrained"] = str(len(trained))

    # ---------------- Table: full MSE (appendix) -------------------------
    cols = MAIN_MODELS
    lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             "Data & $H$ & " + " & ".join(MODEL_NAME[c] for c in cols) + r" \\", r"\midrule"]
    for d in DS_ORDER:
        for i, h in enumerate([96, 192, 336, 720]):
            vals = {c: P.loc[(d, h), c] for c in cols}
            trained_vals = {c: v for c, v in vals.items() if c != "seasonal_naive"}
            cells = bold_best(trained_vals)
            row = [fmt(vals["seasonal_naive"])] + cells
            lines.append((DS_NAME[d] if i == 0 else "") + f" & {h} & " + " & ".join(row) + r" \\")
        lines.append(r"\midrule" if d != DS_ORDER[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write(out / "tab_full_mse.tex", "\n".join(lines))

    # ---------------- Table: dataset-averaged MSE + avg rank ---------------
    T = s[s.model.isin(trained)].copy()
    T["rank"] = T.groupby(["dataset", "pred_len"]).mse_mean.rank()
    avg_rank = T.groupby("model")["rank"].mean()
    Pd = P.groupby(level=0).mean().loc[DS_ORDER]
    cols2 = MAIN_MODELS
    lines = [r"\begin{tabular}{l" + "c" * len(cols2) + "}", r"\toprule",
             "Dataset & " + " & ".join(MODEL_NAME[c] for c in cols2) + r" \\", r"\midrule"]
    for d in DS_ORDER:
        vals = {c: Pd.loc[d, c] for c in cols2}
        cells = bold_best({c: v for c, v in vals.items() if c != "seasonal_naive"})
        lines.append(DS_NAME[d] + " & " + fmt(vals["seasonal_naive"]) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append("Avg.\\ rank (of 17) & -- & " + " & ".join(f"{avg_rank[c]:.1f}" for c in cols2[1:]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_dataset_avg.tex", "\n".join(lines))
    for c in ["patchtst", "cyclenet_fixed", "cyclenet_pro_mlp_ci", "cyclenet_pro_linear_ci", "tf_rfft", "linear",
              "dlinear", "cyclenet_pro_cnn_ci", "tf_stft"]:
        macros["rank" + MODEL_NAME[c].replace("-", "").replace(" ", "")] = f"{avg_rank[c]:.1f}"

    # ---------------- Spread among strong models ---------------------------
    def gaps(df):
        v = np.sort(df.values)
        return pd.Series({"g2": v[1] / v[0] - 1, "g5": v[4] / v[0] - 1, "w1": (v <= v[0] * 1.01).sum(),
                          "w3": (v <= v[0] * 1.03).sum()})
    G = P[trained].apply(gaps, axis=1)
    macros["medGapSecond"] = f"{G.g2.median() * 100:.1f}"
    macros["medGapFifth"] = f"{G.g5.median() * 100:.1f}"
    macros["medWithinThree"] = f"{G.w3.median():.1f}"
    winners = P[trained].idxmin(axis=1)
    macros["nCells"] = str(len(P))
    wc = winners.value_counts()
    macros["winsPatchTST"] = str(wc.get("patchtst", 0))
    macros["winsCycFixed"] = str(wc.get("cyclenet_fixed", 0))
    macros["winsCycMLPCI"] = str(wc.get("cyclenet_pro_mlp_ci", 0))
    macros["winsTF"] = str(sum(wc.get(m, 0) for m in wc.index if m.startswith("tf_")))
    macros["nDistinctWinners"] = str(len(wc))
    cv = (s[s.n == 3].mse_std / s[s.n == 3].mse_mean)
    macros["medSeedCV"] = f"{cv.median() * 100:.2f}"
    macros["nUnstable"] = str(int((cv > 0.05).sum()))
    macros["nSeedCells"] = str(int(len(cv)))

    # ---------------- Paired comparisons -----------------------------------
    noetth1 = P.index.get_level_values(0) != "ETTh1"
    comps = [
        ("Fixed linear transform vs.\\ raw lookback", "tf_rfft", "linear", None),
        ("", "tf_dct", "linear", None),
        ("", "tf_dwt_haar", "linear", None),
        ("", "dlinear", "linear", None),
        ("", "tf_stft", "linear", None),
        ("Cycle queue (auto period) vs.\\ none", "cyclenet_pro_linear_ci", "linear", None),
        ("Cycle queue (dataset period) vs.\\ none", "cyclenet_fixed", "linear", None),
        ("Nonlinear residual (MLP) vs.\\ linear", "cyclenet_pro_mlp_ci", "cyclenet_pro_linear_ci", None),
        ("Patch mixer (SOR) vs.\\ linear", "cyclenet_pro_sor_ci", "cyclenet_pro_linear_ci", None),
        ("1D-CNN vs.\\ linear", "cyclenet_pro_cnn_ci", "cyclenet_pro_linear_ci", None),
        ("Channel independent vs.\\ auto-mixing", "cyclenet_pro_linear_ci", "cyclenet_pro_linear", noetth1),
        ("", "cyclenet_pro_mlp_ci", "cyclenet_pro_mlp", noetth1),
        ("", "cyclenet_pro_sor_ci", "cyclenet_pro_sor", noetth1),
        ("", "cyclenet_pro_cnn_ci", "cyclenet_pro_cnn", noetth1),
        ("PatchTST vs.\\ best CycleNet variant", "patchtst", "cyclenet_pro_mlp_ci", None),
        ("", "patchtst", "cyclenet_fixed", None),
    ]
    lines = [r"\begin{tabular}{lllcccc}", r"\toprule",
             r"Question & A & B & cells & median $\Delta$ (\%) & A better & Wilcoxon $p$ \\", r"\midrule"]
    for q, A, B, mask in comps:
        dlt = P[A] / P[B] - 1
        if mask is not None:
            dlt = dlt[mask]
        w = stats.wilcoxon(dlt)
        lines.append(f"{q} & {MODEL_NAME[A]} & {MODEL_NAME[B]} & {len(dlt)} & {dlt.median() * 100:+.2f} & "
                     f"{(dlt < 0).sum()}/{len(dlt)} & {w.pvalue:.1g}" + r" \\")
        key = "cmp" + MODEL_NAME[A].replace("-", "").replace(".", "") + "vs" + MODEL_NAME[B].replace("-", "")
        macros[key] = f"{dlt.median() * 100:+.2f}"
        macros[key + "wins"] = f"{(dlt < 0).sum()}/{len(dlt)}"
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_paired.tex", "\n".join(lines))

    fam = ["linear", "dlinear", "tf_rfft", "tf_dct", "tf_dwt_haar", "tf_dwt_db4"]
    spread = (P[fam].max(axis=1) / P[fam].min(axis=1) - 1) * 100
    macros["linFamSpreadMed"] = f"{spread.median():.1f}"
    macros["linFamSpreadBigData"] = f"{spread[spread.index.get_level_values(0).isin(['electricity', 'solar', 'traffic'])].max():.1f}"

    # ---------------- Per-dataset effects table -----------------------------
    effects = [("Cycle queue", "cyclenet_pro_linear_ci", "linear"),
               ("Weekly vs auto period", "cyclenet_fixed", "cyclenet_pro_linear_ci"),
               ("MLP residual", "cyclenet_pro_mlp_ci", "cyclenet_pro_linear_ci"),
               ("Channel mixing (auto)", "cyclenet_pro_mlp", "cyclenet_pro_mlp_ci"),
               ("PatchTST", "patchtst", "cyclenet_pro_linear_ci")]
    chans = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "electricity": 321, "solar": 137, "traffic": 862, "weather": 21}
    periods = r.groupby(["dataset", "model"]).periods.first().unstack(1)
    lines = [r"\begin{tabular}{lcll" + "c" * len(effects) + "}", r"\toprule",
             r"Dataset & $C$ & auto $p$ & fixed $p$ & " + " & ".join(e[0] for e in effects) + r" \\", r"\midrule"]
    for d in DS_ORDER:
        row = []
        for _, A, B in effects:
            dd = ((P.loc[d, A] / P.loc[d, B] - 1) * 100).mean()
            row.append(pct(dd))
        ap_ = periods.loc[d, "cyclenet_pro_linear"].strip("[]")
        fp = periods.loc[d, "cyclenet_fixed"].strip("[]")
        lines.append(f"{DS_NAME[d]} & {chans[d]} & {ap_} & {fp} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_effects_by_dataset.tex", "\n".join(lines))

    # ---------------- Band decomposition -----------------------------------
    m = r.groupby(["dataset", "pred_len", "model"])[BANDS + ["mse", "slow_fvu", "daily_fvu", "fast_fvu"]].mean()
    macros["parsevalMaxErr"] = f"{(m[BANDS].sum(axis=1) - m.mse).abs().max():.0e}"
    ref = m.xs("cyclenet_pro_linear_ci", level=2)
    share = ref[BANDS].div(ref.mse, axis=0).groupby(level=0).mean() * 100
    fvu = ref[["slow_fvu", "daily_fvu", "fast_fvu"]].groupby(level=0).mean()
    snv = m.xs("seasonal_naive", level=2)
    sk = (1 - ref.mse / snv.mse).groupby(level=0).mean() * 100
    lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
             r"& \multicolumn{4}{c}{Share of MSE (\%)} & \multicolumn{3}{c}{FVU} & Skill vs. \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r"Dataset & level & slow & daily & fast & slow & daily & fast & S-Naive (\%) \\", r"\midrule"]
    for d in DS_ORDER:
        lines.append(f"{DS_NAME[d]} & " + " & ".join(f"{share.loc[d, b]:.0f}" for b in BANDS) + " & " +
                     " & ".join(fmt(fvu.loc[d, c], 2) for c in ["slow_fvu", "daily_fvu", "fast_fvu"]) +
                     f" & {sk.loc[d]:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_bands.tex", "\n".join(lines))
    for d in ["ETTh2", "ETTm2", "weather", "ETTh1", "solar", "traffic"]:
        macros["levelShare" + DS_NAME[d].replace("ETT", "E")] = f"{share.loc[d, 'level_mse']:.0f}"
        macros["levslowShare" + DS_NAME[d].replace("ETT", "E")] = f"{share.loc[d, ['level_mse', 'slow_mse']].sum():.0f}"
        macros["dailyShare" + DS_NAME[d].replace("ETT", "E")] = f"{share.loc[d, 'daily_mse']:.0f}"
        macros["dailyFVU" + DS_NAME[d].replace("ETT", "E")] = f"{fvu.loc[d, 'daily_fvu']:.2f}"

    # where gains come from
    pairs = [("Cycle queue", "cyclenet_pro_linear_ci", "linear"),
             ("MLP residual", "cyclenet_pro_mlp_ci", "cyclenet_pro_linear_ci"),
             ("PatchTST", "patchtst", "cyclenet_pro_linear_ci")]
    lines = [r"\begin{tabular}{l" + "cccc" * len(pairs) + "}", r"\toprule",
             "& " + " & ".join(r"\multicolumn{4}{c}{" + p[0] + "}" for p in pairs) + r" \\",
             "".join(fr"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(pairs))),
             "Dataset & " + " & ".join(["lev & slow & day & fast"] * len(pairs)) + r" \\", r"\midrule"]
    gain = {}
    for name, A, B in pairs:
        Am, Bm = m.xs(A, level=2), m.xs(B, level=2)
        gain[name] = ((Am[BANDS] - Bm[BANDS]).div(Bm.mse, axis=0) * 100).groupby(level=0).mean()
    for d in DS_ORDER:
        cells = []
        for name, _, _ in pairs:
            cells += [pct(gain[name].loc[d, b]) for b in BANDS]
        lines.append(DS_NAME[d] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_band_gains.tex", "\n".join(lines))
    macros["patchLevelGainSolar"] = f"{gain['PatchTST'].loc['solar', 'level_mse']:+.1f}"
    macros["patchDailyGainSolar"] = f"{gain['PatchTST'].loc['solar', 'daily_mse']:+.1f}"
    macros["patchLevelGainWeather"] = f"{gain['PatchTST'].loc['weather', 'level_mse']:+.1f}"
    macros["mlpLevelGainWeather"] = f"{gain['MLP residual'].loc['weather', 'level_mse']:+.1f}"
    macros["mlpLevelGainSolar"] = f"{gain['MLP residual'].loc['solar', 'level_mse']:+.1f}"
    macros["cycDailyGainSolar"] = f"{gain['Cycle queue'].loc['solar', 'daily_mse']:+.1f}"

    # ---------------- CNN params ------------------------------------------
    prm = r.groupby(["dataset", "model"]).params.first().unstack(0)
    macros["cnnParamsETT"] = f"{prm.loc['cyclenet_pro_cnn_ci', 'ETTh1']:,}".replace(",", "{,}")
    macros["linParamsETT"] = f"{prm.loc['cyclenet_pro_linear_ci', 'ETTh1']:,}".replace(",", "{,}")
    macros["mlpParamsETT"] = f"{prm.loc['cyclenet_pro_mlp_ci', 'ETTh1']:,}".replace(",", "{,}")
    lines = [r"\begin{tabular}{l" + "r" * len(DS_ORDER) + "}", r"\toprule",
             "Model & " + " & ".join(DS_NAME[d] for d in DS_ORDER) + r" \\", r"\midrule"]
    for c in ["linear", "dlinear", "tf_stft", "cyclenet_fixed", "cyclenet_pro_linear_ci", "cyclenet_pro_linear",
              "cyclenet_pro_mlp_ci", "cyclenet_pro_sor_ci", "cyclenet_pro_cnn_ci", "cyclenet_pro_cnn", "patchtst"]:
        lines.append(MODEL_NAME[c] + " & " + " & ".join(f"{int(prm.loc[c, d]):,}" for d in DS_ORDER) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_params.tex", "\n".join(lines))

    # ---------------- Foundation models ------------------------------------
    f = pd.read_csv(v2 / "bigmama/foundation_vs_trained_bigmama.csv")
    fm = f.groupby(["dataset", "pred_len", "model"]).mse.mean().unstack("model")
    tr_cols = [c for c in fm.columns if c in set(f[f.kind == "trained"].model) and c not in ("repeat_last", "seasonal_naive")]
    fm["best_trained"] = fm[tr_cols].min(axis=1)
    fcols = ["seasonal_naive", "fourier_base", "chronos_t5", "chronos_bolt", "timesfm", "linear", "cyclenet_pro_linear_ci", "patchtst", "best_trained"]
    names = dict(MODEL_NAME, best_trained="Best trained$^\\dagger$")
    fd = fm[fcols].groupby(level=0).mean().loc[DS_ORDER]
    lines = [r"\begin{tabular}{l" + "c" * len(fcols) + "}", r"\toprule",
             r"& \multicolumn{5}{c}{Untrained / zero-shot} & \multicolumn{4}{c}{Trained on the target dataset} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
             "Dataset & " + " & ".join(names[c] for c in fcols) + r" \\", r"\midrule"]
    for d in DS_ORDER:
        cells = bold_best({c: fd.loc[d, c] for c in fcols[:-1]})
        lines.append(DS_NAME[d] + " & " + " & ".join(cells) + " & " + fmt(fd.loc[d, "best_trained"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_foundation.tex", "\n".join(lines))
    for c in ["timesfm", "chronos_bolt", "chronos_t5"]:
        dl = fm[c] / fm["linear"] - 1
        key = MODEL_NAME[c].replace("-", "").replace(".", "")
        macros["fm" + key + "VsLinMed"] = f"{dl.median() * 100:+.1f}"
        macros["fm" + key + "VsLinWins"] = f"{(dl < 0).sum()}/{dl.notna().sum()}"
        ds = fm[c] / fm["seasonal_naive"] - 1
        macros["fm" + key + "VsSnWins"] = f"{(ds < 0).sum()}/{ds.notna().sum()}"
    tfm_traffic = (fm.loc["traffic", "timesfm"] < fm.loc["traffic", "best_trained"]).sum()
    macros["timesfmTrafficWins"] = str(int(tfm_traffic))
    macros["ttmVsLinWins"] = f"{int((fm['ttm'] < fm['linear']).sum())}/{int(fm['ttm'].notna().sum())}"

    # Fourier decoupling
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Dataset & TimesFM-2.5 & Chronos-Bolt & Chronos-T5 & TTM-R1 ($H{=}96$) \\", r"\midrule"]
    fdec = {}
    for c in ["timesfm", "chronos_bolt", "chronos_t5", "ttm"]:
        fdec[c] = ((fm[c + "_fourier"] / fm[c] - 1) * 100)
    for d in DS_ORDER:
        lines.append(DS_NAME[d] + " & " + " & ".join(pct(fdec[c].loc[d].mean(), 0) for c in ["timesfm", "chronos_bolt", "chronos_t5", "ttm"]) + r" \\")
    lines.append(r"\midrule")
    lines.append("cells improved & " + " & ".join(f"{int((fdec[c] < 0).sum())}/{int(fdec[c].notna().sum())}" for c in ["timesfm", "chronos_bolt", "chronos_t5", "ttm"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(out / "tab_fourier_decoupling.tex", "\n".join(lines))
    for c in ["timesfm", "chronos_bolt", "chronos_t5"]:
        key = MODEL_NAME[c].replace("-", "").replace(".", "")
        macros["fdec" + key + "Improved"] = f"{int((fdec[c] < 0).sum())}/{int(fdec[c].notna().sum())}"
    macros["fdecTTMImproved"] = f"{int((fdec['ttm'] < 0).sum())}/{int(fdec['ttm'].notna().sum())}"
    fb = fm["fourier_base"] / fm["seasonal_naive"] - 1
    macros["fourierBaseWorseThanSnaive"] = f"{int((fb > 0).sum())}/{int(fb.notna().sum())}"

    # ---------------- DSOF-SD ---------------------------------------------
    dd = pd.read_csv(v2 / "bigmama/dsof_runs_agg.csv")
    fin = dd[dd.kind == "metrics"].groupby(["dataset", "pred_len", "config"]).mse.mean().unstack("config")
    warm = dd[dd.kind == "metrics_warm"].groupby(["dataset", "pred_len", "config"]).mse.mean().unstack("config")
    teach = dd[dd.kind == "metrics_teacher"].groupby(["dataset", "pred_len"]).mse.mean()
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"& & Teacher & \multicolumn{2}{c}{Linear student} & \multicolumn{4}{c}{SOR student} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-9}",
             r"Data & $H$ & TimesFM & control & hard & warm & control & hard & soft \\", r"\midrule"]
    for d in ["ETTh1", "ETTh2"]:
        for i, h in enumerate([96, 192, 336, 720]):
            row = [teach.loc[(d, h)], fin.loc[(d, h), "linear_control"], fin.loc[(d, h), "linear_hard"],
                   warm.loc[(d, h), "sor_control"], fin.loc[(d, h), "sor_control"], fin.loc[(d, h), "sor_hard"],
                   fin.loc[(d, h), "sor_soft"]]
            cells = bold_best(dict(enumerate(row)))
            lines.append((d if i == 0 else "") + f" & {h} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if d == "ETTh1" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write(out / "tab_dsof.tex", "\n".join(lines))
    for st, g in [("sor", "hard"), ("sor", "soft"), ("linear", "hard"), ("sor", "benefit"), ("sor", "consensus"), ("cnn", "benefit")]:
        rel = (fin[f"{st}_{g}"] / fin[f"{st}_control"] - 1) * 100
        macros[f"dsof{st.upper()}{g}Med"] = f"{rel.median():+.1f}"
        macros[f"dsof{st.upper()}{g}Wins"] = f"{int((rel < 0).sum())}/{len(rel)}"
    rel = (fin["sor_control"] / warm["sor_control"] - 1) * 100
    macros["sorControlVsWarmMed"] = f"{rel.median():+.1f}"
    rel = (fin["sor_hard"] / fin["linear_control"] - 1) * 100
    macros["sorHardVsLinCtrlMed"] = f"{rel.median():+.1f}"
    zg = dd[dd.kind == "metrics"].groupby("config").all_gates_zero.mean()
    macros["benefitAllZeroPct"] = f"{zg['sor_benefit'] * 100:.0f}"
    macros["consensusAllZeroPct"] = f"{zg['sor_consensus'] * 100:.0f}"

    # ---------------- Cross-machine ----------------------------------------
    c = pd.read_csv(v2 / "compare_local_vs_bigmama.csv")
    macros["xmAgree"] = f"{int(c.agree.sum())}/{len(c)}"
    macros["xmMedAbs"] = f"{c.rel_diff_pct.abs().median():.3f}"

    # ---------------- key cells quoted in text ------------------------------
    def cell(model, d, h):
        return f"{P.loc[(d, h), model]:.4f}"
    macros["dlinearEtthTwoSevenTwenty"] = cell("dlinear", "ETTh2", 720)
    macros["cycLinEtthTwoSevenTwenty"] = cell("cyclenet_pro_linear", "ETTh2", 720)
    macros["rfftEtthTwoSevenTwenty"] = cell("tf_rfft", "ETTh2", 720)
    macros["trafficSevenTwentyCycLin"] = f"{P.loc[('traffic', 720), 'cyclenet_pro_linear']:.4f}"
    macros["trafficSevenTwentyCycLinStd"] = f"{Pstd.loc[('traffic', 720), 'cyclenet_pro_linear']:.4f}"
    macros["elecCycFixed"] = f"{Pd.loc['electricity', 'cyclenet_fixed']:.4f}"
    macros["elecCycLinCI"] = f"{Pd.loc['electricity', 'cyclenet_pro_linear_ci']:.4f}"

    words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

    def safe(k):  # LaTeX macro names may not contain digits
        return "".join(words[int(ch)] if ch.isdigit() else ch for ch in k)
    write(out / "numbers.tex", "\n".join(f"\\newcommand{{\\{safe(k)}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")


if __name__ == "__main__":
    main()

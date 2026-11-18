"""Generate exact, un-editorialized diagnostic tables across all datasets and horizons."""

import pandas as pd
import numpy as np

df = pd.read_csv("results/transform_ablation/transform_ablation_summary.csv")

def get_table(dataset, horizon):
    sub = df[(df["dataset"] == dataset) & (df["pred_len"] == horizon)]
    if len(sub) == 0:
        return None
    records = []
    # Rank by mean MSE ascending
    transforms = sub.groupby("transform")["mse"].mean().sort_values().index.tolist()
    for t in transforms:
        g = sub[sub["transform"] == t]
        mse_mean, mse_std = g["mse"].mean(), g["mse"].std(ddof=1) if len(g) > 1 else 0.0
        mae_mean, mae_std = g["mae"].mean(), g["mae"].std(ddof=1) if len(g) > 1 else 0.0
        fvu_l_m, fvu_l_s = g["fvu_low"].mean(), g["fvu_low"].std(ddof=1) if len(g) > 1 else 0.0
        fvu_m_m, fvu_m_s = g["fvu_mid"].mean(), g["fvu_mid"].std(ddof=1) if len(g) > 1 else 0.0
        fvu_h_m, fvu_h_s = g["fvu_high"].mean(), g["fvu_high"].std(ddof=1) if len(g) > 1 else 0.0
        er_l = g["energy_ratio_low"].mean()
        er_m = g["energy_ratio_mid"].mean()
        er_h = g["energy_ratio_high"].mean()
        lsd_m, lsd_s = g["log_spectral_distance"].mean(), g["log_spectral_distance"].std(ddof=1) if len(g) > 1 else 0.0
        fail_p = g["failure_rate_pct"].mean()

        row = {
            "Transform": t,
            "Params": int(g["params"].iloc[0]),
            "MSE": f"{mse_mean:.4f} ± {mse_std:.4f}",
            "MAE": f"{mae_mean:.4f} ± {mae_std:.4f}",
            "FVU_low": f"{fvu_l_m:.4f} ± {fvu_l_s:.4f}",
            "FVU_mid": f"{fvu_m_m:.4f} ± {fvu_m_s:.4f}",
            "FVU_high": f"{fvu_h_m:.4f} ± {fvu_h_s:.4f}",
            "ER_low": f"{er_l:.3f}",
            "ER_mid": f"{er_m:.3f}",
            "ER_high": f"{er_h:.3f}",
            "LSD (dB)": f"{lsd_m:.2f} ± {lsd_s:.2f}",
            "Fail (%)": f"{fail_p:.1f}%",
        }
        records.append(row)
    return pd.DataFrame(records)

datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"]
horizons = [96, 192, 336, 720]

for ds in datasets:
    for h in horizons:
        tbl = get_table(ds, h)
        if tbl is not None:
            print(f"\n### {ds} (H={h})")
            print(tbl.to_markdown(index=False))

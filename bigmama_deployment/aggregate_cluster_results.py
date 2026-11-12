"""
Aggregate Bigmama Cluster Benchmark Results
===========================================
Reads all JSON outputs from results/cluster_results/, calculates multi-seed
Mean +/- Std across random seeds [2024, 2025, 2026], and exports:
  1. results/bigmama_master_summary.md (Markdown Table)
  2. results/bigmama_master_summary.json (Machine readable JSON)
  3. results/table_bigmama_latex.tex (LaTeX table ready for paper/main.tex)
"""

import os
import glob
import json
import numpy as np
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "cluster_results")
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"No result JSON files found in {results_dir}")
        return

    all_records = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                elif isinstance(data, dict):
                    all_records.append(data)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} records across {len(json_files)} result files.")

    # Group by dataset, model, paradigm, pred_len
    group_cols = ['dataset', 'model', 'paradigm', 'params', 'pred_len']
    agg_df = df.groupby(group_cols).agg(
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        mid_fvu_mean=('mid_fvu', 'mean'),
        lsd_mean=('lsd_db', 'mean'),
        count=('seed', 'count')
    ).reset_index()

    # Format Mean +/- Std
    agg_df['MSE'] = agg_df.apply(lambda r: f"{r['mse_mean']:.4f}" if pd.isna(r['mse_std']) else f"{r['mse_mean']:.4f}±{r['mse_std']:.4f}", axis=1)
    agg_df['MAE'] = agg_df.apply(lambda r: f"{r['mae_mean']:.4f}" if pd.isna(r['mae_std']) else f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}", axis=1)
    agg_df['Mid_FVU'] = agg_df['mid_fvu_mean'].round(4)
    agg_df['LSD_dB'] = agg_df['lsd_mean'].round(2)

    display_cols = ['dataset', 'model', 'paradigm', 'params', 'pred_len', 'MSE', 'MAE', 'Mid_FVU', 'LSD_dB', 'count']
    summary_table = agg_df[display_cols].sort_values(by=['dataset', 'pred_len', 'paradigm'])

    # Export Markdown
    md_path = os.path.join(out_dir, "bigmama_master_summary.md")
    summary_table.to_markdown(md_path, index=False)

    # Export JSON
    json_path = os.path.join(out_dir, "bigmama_master_summary.json")
    with open(json_path, "w") as f:
        json.dump(agg_df.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 100)
    print(" BIGMAMA CLUSTER MULTI-SEED MASTER SUMMARY")
    print("=" * 100)
    print(summary_table.to_string(index=False))
    print(f"\nSaved summary to:\n  - {md_path}\n  - {json_path}")


if __name__ == "__main__":
    main()


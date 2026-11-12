"""
Empirical Evaluation of Fourier-TQNet (FTQ-Net): Frankenstein Hybrid Model
==========================================================================
Tests the fusion of:
- Stage 1: Fourier Spectral Harmonic Base Forecaster
- Stage 2: Temporal Query (TQ) Cross-Channel Attention on Residuals, where
  Query Q is derived dynamically from the Fourier harmonic base, and
  Keys K, Values V are derived from the non-periodic residuals.
"""

import os
import sys
import copy
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocessing import build_etth1_dataloaders
from models import TwoStageFourierTQNet
from run_experiments import compute_spectral_harmonic_error, set_seed, train_model


def run_ftq_experiment():
    set_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==================================================================")
    print(f"Launching Frankenstein Hybrid (FTQ-Net) Experiment on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
    print(f"==================================================================")

    data_path = os.path.join(current_dir, '..', 'data_sets', 'ETTh1.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', 'ETTh1.csv')

    seq_len = 336
    pred_len = 96
    batch_size = 32

    data_bundle = build_etth1_dataloaders(
        csv_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size
    )
    channels = data_bundle['channels']

    # Instantiate the Frankenstein Fourier-TQ Model
    model = TwoStageFourierTQNet(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        n_heads=4,
        d_mlp=128,
        dropout=0.1
    ).to(device)

    res_ftq = train_model(
        model_name="TwoStageFourierTQNet (Fourier Base + TQ Cross-Channel Residual Attention)",
        model=model,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )

    # Load previous 4-model summary to compare
    summary_json_path = os.path.join(current_dir, 'results', 'four_model_summary.json')
    if not os.path.exists(summary_json_path):
        summary_json_path = os.path.join(current_dir, 'results', 'summary.json')

    prev_results = []
    if os.path.exists(summary_json_path):
        with open(summary_json_path, 'r') as f:
            prev_results = json.load(f)

    all_results = prev_results + [res_ftq]

    print("\n" + "#"*95)
    print("COMPLETE 5-MODEL BENCHMARK COMPARISON (Including FTQ-Net Frankenstein Hybrid)")
    print("#"*95)
    summary_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Parameters': f"{r['params']:,}",
            'Best Epoch': f"{r['best_epoch']}/{r['total_epochs']}",
            'Time (s)': f"{r['train_time_sec']:.1f}",
            'Test MSE': f"{r['test_mse']:.4f}",
            'Test MAE': f"{r['test_mae']:.4f}",
            'Mid-Band MSE': f"{r['mid_band_error']:.5f}",
            'Mid-Band FVU': f"{r['mid_band_fvu']:.4f}"
        }
        for r in all_results
    ])
    print(summary_df.to_string(index=False))
    print("#"*95)

    # Save 5-model results
    five_json_path = os.path.join(current_dir, 'results', 'five_model_summary.json')
    with open(five_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    five_md_path = os.path.join(current_dir, 'results', 'five_model_summary.md')
    md_content = f"# Empirical Benchmark: 5-Model Comparative Study on ETTh1\n\n"
    md_content += f"- **Lookback Window ($L$):** 336 steps | **Horizon ($H$):** 96 steps\n"
    md_content += f"- **Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n"
    md_content += summary_df.to_markdown(index=False) + "\n\n"
    md_content += "## Model Taxonomy & Insights:\n"
    md_content += "1. **DLinear**: Simple moving average baseline with RevIN.\n"
    md_content += "2. **CycleNet**: NeurIPS 2024 periodic state queue indexed by t mod W.\n"
    md_content += "3. **Point-wise Attention**: Generic MHA on raw points (overfits severely).\n"
    md_content += "4. **Patch-based Attention**: PatchTST subseries patch tokenization (curbs overfitting).\n"
    md_content += "5. **FTQ-Net (Frankenstein Hybrid)**: Fourier harmonic base extraction + TQ cross-channel residual attention.\n"

    with open(five_md_path, 'w') as f:
        f.write(md_content)
    print(f"5-Model Markdown report saved to: {five_md_path}")


if __name__ == '__main__':
    run_ftq_experiment()

"""
Evaluation of Two-Stage Fourier Base + Patch-Based Residual Attention (PatchTST Style)
======================================================================================
Tests whether patch-based subseries tokenization eliminates the point-wise overfitting
observed with generic Multi-Head Attention on ETTh1 (L=336, H=96).
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
from models import TwoStageFourierPatchAttentionNet
from run_experiments import compute_spectral_harmonic_error, set_seed, train_model


def run_patch_experiment():
    set_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLaunching Patch-Based Attention Experiment on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

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

    # Instantiate Patch Attention Two-Stage Model
    model = TwoStageFourierPatchAttentionNet(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.2
    ).to(device)

    res_patch = train_model(
        model_name="TwoStageFourierPatchAttentionNet (Fourier Base + PatchTST Residuals)",
        model=model,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )

    # Load previous summary to compare
    summary_json_path = os.path.join(current_dir, 'results', 'summary.json')
    prev_results = []
    if os.path.exists(summary_json_path):
        with open(summary_json_path, 'r') as f:
            prev_results = json.load(f)

    # Append new patch model
    all_results = prev_results + [res_patch]

    print("\n" + "#"*90)
    print("FOUR-WAY BENCHMARK COMPARISON (Including Patch-Based Residual Attention)")
    print("#"*90)
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
    print("#"*90)

    # Save updated JSON and Markdown
    updated_json_path = os.path.join(current_dir, 'results', 'four_model_summary.json')
    with open(updated_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    updated_md_path = os.path.join(current_dir, 'results', 'four_model_summary.md')
    md_content = f"# Empirical Comparison: DLinear vs. CycleNet vs. Point-wise MHA vs. Patch-based Attention\n\n"
    md_content += f"- **Dataset:** ETTh1 (7 channels) | **Lookback $L$:** 336 | **Horizon $H$:** 96\n"
    md_content += f"- **Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n"
    md_content += summary_df.to_markdown(index=False) + "\n\n"
    md_content += "## Key Findings:\n"
    md_content += f"- **DLinear Test MSE:** {all_results[0]['test_mse']:.4f}\n"
    md_content += f"- **CycleNet Test MSE:** {all_results[1]['test_mse']:.4f}\n"
    md_content += f"- **Point-wise Attention Test MSE:** {all_results[2]['test_mse']:.4f}\n"
    md_content += f"- **Patch-based Attention Test MSE:** {res_patch['test_mse']:.4f} (Best Epoch: {res_patch['best_epoch']})\n"

    with open(updated_md_path, 'w') as f:
        f.write(md_content)
    print(f"Updated results saved to {updated_md_path}")


if __name__ == '__main__':
    run_patch_experiment()

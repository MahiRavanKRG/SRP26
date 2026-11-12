"""
Complete Comparative Experiment Suite on ETTh2 (Lookback L=336, Horizon H=96)
==============================================================================
Addresses user observation: "them getting trained only to one epoch is questionable."
Key Fixes & Upgrades:
1. Proper Transformer Learning Rate: Uses lr=3e-4 with weight_decay=1e-4 for attention models
   (instead of oversized lr=1e-3), preventing immediate gradient overshooting.
2. Learning Rate Scheduler: Integrates CosineAnnealingLR for stable multi-epoch convergence.
3. Dataset: Evaluates on ETTh2, known for severe non-stationary regime shifts and drift.
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
from torch.optim.lr_scheduler import CosineAnnealingLR

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocessing import build_dataset_dataloaders
from models import (
    DLinear,
    CycleNet,
    TwoStageFourierPatchAttentionNet,
    TwoStageFourierTQNet
)
from run_experiments import compute_spectral_harmonic_error, set_seed


def train_model_with_scheduler(model_name: str, model: nn.Module, data_bundle: dict, device: torch.device,
                               max_epochs: int = 40, patience: int = 7, lr: float = 1e-3,
                               weight_decay: float = 1e-4, use_scheduler: bool = True):
    print(f"\n" + "="*75)
    print(f"EXPERIMENT: {model_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | Trainable Parameters: {total_params:,}")
    print(f"Max Epochs: {max_epochs} | Patience: {patience} | Base LR: {lr} | Weight Decay: {weight_decay}")
    print("="*75)

    train_loader = data_bundle['train_loader']
    val_loader = data_bundle['val_loader']
    test_loader = data_bundle['test_loader']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5) if use_scheduler else None
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    epoch_logs = []

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        # 1. Training Pass
        model.train()
        train_loss = 0.0
        for x, y, cyc in train_loader:
            x, y, cyc = x.to(device), y.to(device), cyc.to(device)
            optimizer.zero_grad()
            out = model(x, cyc)
            loss = criterion(out, y)
            loss.backward()
            # Gradient clipping for attention models
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # 2. Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, cyc in val_loader:
                x, y, cyc = x.to(device), y.to(device), cyc.to(device)
                out = model(x, cyc)
                loss = criterion(out, y)
                val_loss += loss.item() * len(x)
        val_loss /= len(val_loader.dataset)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            tag = "  <-- *BEST*"
        else:
            patience_counter += 1
            tag = f"  (patience {patience_counter}/{patience})"

        epoch_logs.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'val_loss': round(val_loss, 5),
            'lr': round(current_lr, 6),
            'is_best': is_best
        })

        print(f"Epoch {epoch:02d}/{max_epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{tag}")

        if patience_counter >= patience:
            print(f">> Early stopping triggered at epoch {epoch}! Best checkpoint was at Epoch {best_epoch} (Val Loss: {best_val_loss:.5f})")
            break

    elapsed_time = time.time() - t_start

    # Restore best checkpoint
    assert best_weights is not None, "Error: No best checkpoint found."
    model.load_state_dict(best_weights)

    # 3. Test Evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y, cyc in test_loader:
            x, y, cyc = x.to(device), y.to(device), cyc.to(device)
            out = model(x, cyc)
            all_preds.append(out.cpu())
            all_trues.append(y.cpu())

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_trues, dim=0)

    test_mse = float(nn.MSELoss()(y_pred, y_true).item())
    test_mae = float(nn.L1Loss()(y_pred, y_true).item())
    mid_error, mid_fvu = compute_spectral_harmonic_error(y_true, y_pred)

    print(f"\n--- [TEST EVALUATION: {model_name}] ---")
    print(f"Restored from Epoch: {best_epoch}")
    print(f"Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f}")
    print(f"Mid-Band Harmonic Error: {mid_error:.5f} | Mid-Band FVU: {mid_fvu:.4f}")
    print(f"Total Time: {elapsed_time:.1f}s ({elapsed_time/len(epoch_logs):.2f}s/epoch)")

    return {
        'model_name': model_name,
        'params': total_params,
        'best_epoch': best_epoch,
        'total_epochs': len(epoch_logs),
        'train_time_sec': round(elapsed_time, 2),
        'test_mse': round(test_mse, 4),
        'test_mae': round(test_mae, 4),
        'mid_band_error': round(mid_error, 5),
        'mid_band_fvu': round(mid_fvu, 4),
        'epoch_logs': epoch_logs
    }


def run_all_etth2_experiments():
    set_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "#"*75)
    print(f"STARTING ETTh2 BENCHMARK EXPERIMENTS (L=336, H=96)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("#"*75)

    data_path = os.path.join(current_dir, '..', 'data_sets', 'ETTh2.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', 'ETTh2.csv')

    seq_len = 336
    pred_len = 96
    batch_size = 32

    data_bundle = build_dataset_dataloaders(
        csv_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size
    )
    channels = data_bundle['channels']

    results = []

    # 1. DLinear Baseline
    set_seed(2024)
    m_dlinear = DLinear(seq_len=seq_len, pred_len=pred_len, channels=channels).to(device)
    res_dlinear = train_model_with_scheduler(
        model_name="DLinear (AAAI 2023)",
        model=m_dlinear,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3,
        weight_decay=1e-4,
        use_scheduler=False
    )
    results.append(res_dlinear)

    # 2. CycleNet Baseline
    set_seed(2024)
    m_cyclenet = CycleNet(seq_len=seq_len, pred_len=pred_len, channels=channels, period_w=24).to(device)
    res_cyclenet = train_model_with_scheduler(
        model_name="CycleNet (NeurIPS 2024, W=24)",
        model=m_cyclenet,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3,
        weight_decay=1e-4,
        use_scheduler=False
    )
    results.append(res_cyclenet)

    # 3. TwoStageFourierPatchAttentionNet (PatchTST style with SOTA tuned LR)
    set_seed(2024)
    m_patch = TwoStageFourierPatchAttentionNet(
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
    res_patch = train_model_with_scheduler(
        model_name="TwoStageFourierPatchAttentionNet (Fourier Base + PatchTST Residuals)",
        model=m_patch,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=3e-4,          # SOTA tuned LR for time-series transformer
        weight_decay=1e-4,
        use_scheduler=True
    )
    results.append(res_patch)

    # 4. TwoStageFourierTQNet (Frankenstein Hybrid with SOTA tuned LR)
    set_seed(2024)
    m_ftq = TwoStageFourierTQNet(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        n_heads=4,
        d_mlp=128,
        dropout=0.1
    ).to(device)
    res_ftq = train_model_with_scheduler(
        model_name="TwoStageFourierTQNet (Fourier Base + TQ Cross-Channel Attention)",
        model=m_ftq,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=3e-4,          # SOTA tuned LR for cross-attention
        weight_decay=1e-4,
        use_scheduler=True
    )
    results.append(res_ftq)

    # Summary
    print("\n" + "#"*90)
    print("FINAL ETTh2 BENCHMARK COMPARISON TABLE (L=336, H=96)")
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
        for r in results
    ])
    print(summary_df.to_string(index=False))
    print("#"*90)

    # Save to JSON & Markdown
    res_dir = os.path.join(current_dir, 'results')
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, 'etth2_summary.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(res_dir, 'etth2_summary.md')
    md_content = f"# Empirical Comparison on ETTh2 (L=336, H=96)\n\n"
    md_content += f"- **Dataset:** ETTh2 (Non-stationary regime shift benchmark)\n"
    md_content += f"- **Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n"
    md_content += summary_df.to_markdown(index=False) + "\n\n"

    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"ETTh2 summary saved to: {md_path}")


if __name__ == '__main__':
    run_all_etth2_experiments()

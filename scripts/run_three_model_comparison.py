"""
Three-Model Benchmark Comparison on ETTh1 (Lookback L=336, Horizon H=96)
=========================================================================
Compares:
1. DLinear (Zeng et al., AAAI 2023 baseline with RevIN)
2. CycleNet (Lin et al., NeurIPS 2024 architecture with RecurrentCycle queue)
3. TwoStageFourierAttentionNet (Our Proposed Fourier-Harmonic Base + Residual Attention)

Strict Protocols Applied:
- Canonical 12/4/4 month splits on ETTh1.
- Lookback prefix overlap on Val and Test partitions to prevent boundary truncation.
- Scaler fitted strictly on the Training split.
- Early stopping on Validation MSE with patience=7 (restoring best checkpoint).
- Full Parseval subband spectral decomposition on Test set.
"""

import os
import sys
import time
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'models', 'cyclenet'))

from models.cyclenet.cyclenet_pro import (
    load_raw_dataframe,
    get_splits,
    Scaler,
    WindowDataset,
    CycleNetPro,
    RevIN as CycleRevIN
)
from models.baselines import DLinear
from models.fourier_attention_net import TwoStageFourierAttentionNet
from metrics import fft_band_wise_error_decomposition, psd_comparison


class DLinearWithRevIN(nn.Module):
    """DLinear wrapped with Reversible Instance Normalization for fair benchmarking."""
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.revin = CycleRevIN(channels, affine=True)
        self.dlinear = DLinear(seq_len, pred_len, channels)

    def forward(self, x, *args):
        x_norm = self.revin(x, 'norm')
        y_pred = self.dlinear(x_norm)
        y_denorm = self.revin(y_pred, 'denorm')
        return y_denorm


class CycleNetWrapper(nn.Module):
    """CycleNet wrapper ensuring uniform calling signature (x, cyc)."""
    def __init__(self, seq_len, pred_len, channels, period=24):
        super().__init__()
        self.model = CycleNetPro(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=channels,
            periods=[period],
            use_channel_dependence=False,
            model_type='linear',
            use_revin=True
        )

    def forward(self, x, cyc):
        return self.model(x, cyc)


class FourierAttentionWrapper(nn.Module):
    """TwoStageFourierAttentionNet wrapper conforming to uniform (x, cyc) signature."""
    def __init__(self, seq_len, pred_len, channels, d_model=64, n_heads=4, num_layers=2):
        super().__init__()
        self.model = TwoStageFourierAttentionNet(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=0.1,
            use_revin=True
        )

    def forward(self, x, cyc=None):
        y_hat, _ = self.model(x)
        return y_hat


def set_seed(seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate_model(model_name, model, train_loader, val_loader, test_loader, 
                             device, max_epochs=40, patience=7, lr=1e-3):
    print(f"\n==================================================================")
    print(f"Training: {model_name} on {device}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")
    print(f"Max Epochs: {max_epochs} | Early Stopping Patience: {patience} | LR: {lr}")
    print(f"==================================================================")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    epochs_no_improve = 0

    history = []
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        for x, y, cyc in train_loader:
            x, y, cyc = x.to(device), y.to(device), cyc.to(device)
            optimizer.zero_grad()
            out = model(x, cyc)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        # 2. Validation
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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 5),
            'val_loss': round(val_loss, 5),
            'is_best': is_best
        })

        best_flag = " *** BEST ***" if is_best else f" (no improve: {epochs_no_improve}/{patience})"
        print(f"Epoch {epoch:02d}/{max_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{best_flag}")

        # Stop only when overfitting / plateauing past patience
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}! Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.5f}")
            break

    elapsed_time = time.time() - t_start

    # Restore best checkpoint
    assert best_weights is not None, "No best weights saved!"
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

    # Frequency-domain spectral decomposition
    bw_errors = fft_band_wise_error_decomposition(y_true, y_pred, low_cutoff=0.1, high_cutoff=0.4)
    psd_res = psd_comparison(y_true, y_pred)

    print(f"\n>>> TEST RESULTS for {model_name} (Restored from Epoch {best_epoch}) <<<")
    print(f"MSE: {test_mse:.4f} | MAE: {test_mae:.4f} | Training Time: {elapsed_time:.1f}s")
    print(f"Mid-Band MSE: {bw_errors['mse_mid']:.5f} | FVU-Mid: {bw_errors['fvu_mid']:.4f}")
    print(f"High-Band FVU: {bw_errors['fvu_high']:.4f} | Log-Spectral Distance: {psd_res['mean_lsd']:.4f} dB\n")

    return {
        'model_name': model_name,
        'params': total_params,
        'best_epoch': best_epoch,
        'total_epochs': len(history),
        'training_time_sec': round(elapsed_time, 2),
        'test_mse': round(test_mse, 4),
        'test_mae': round(test_mae, 4),
        'band_mse_mid': round(bw_errors['mse_mid'], 5),
        'fvu_mid': round(bw_errors['fvu_mid'], 4),
        'fvu_high': round(bw_errors['fvu_high'], 4),
        'fvu_low': round(bw_errors['fvu_low'], 4),
        'lsd_db': round(psd_res['mean_lsd'], 4),
        'history': history
    }


def main():
    set_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing 3-Model Benchmark Experiment on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Loading
    data_path = os.path.join(root_dir, 'data_sets', 'ETTh1.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(root_dir, 'ETTh1.csv')
    raw_data = load_raw_dataframe(data_path)
    n_samples, n_channels = raw_data.shape
    print(f"Loaded ETTh1: {n_samples} timesteps, {n_channels} channels.")

    # 2. Split Setup: Canonical 12/4/4 split with Lookback Prefix Overlap
    seq_len = 336
    pred_len = 96
    batch_size = 32

    (s_tr, e_tr), (s_va, e_va), (s_te, e_te) = get_splits(n_samples, 'ETTh1')
    print(f"Splits -> Train: [{s_tr}:{e_tr}], Val: [{s_va}:{e_va}], Test: [{s_te}:{e_te}]")

    # Scaler fit strictly on training set
    scaler = Scaler().fit(raw_data[s_tr:e_tr])
    scaled_data = scaler.transform(raw_data)

    val_start_offset = max(0, s_va - seq_len)
    test_start_offset = max(0, s_te - seq_len)

    train_dataset = WindowDataset(scaled_data[s_tr:e_tr], seq_len, pred_len, start_offset=s_tr)
    val_dataset = WindowDataset(scaled_data[val_start_offset:e_va], seq_len, pred_len, start_offset=val_start_offset)
    test_dataset = WindowDataset(scaled_data[test_start_offset:e_te], seq_len, pred_len, start_offset=test_start_offset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Windows -> Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    results = []

    # -------------------------------------------------------------
    # Experiment 1: DLinear (Baseline 1 - Moving Average Decomposition)
    # -------------------------------------------------------------
    set_seed(2024)
    dlinear_model = DLinearWithRevIN(seq_len=seq_len, pred_len=pred_len, channels=n_channels).to(device)
    res_dlinear = train_and_evaluate_model(
        model_name="DLinear (RevIN)",
        model=dlinear_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )
    results.append(res_dlinear)

    # -------------------------------------------------------------
    # Experiment 2: CycleNet (Proposed in Paper - Recurrent Periodic Queue)
    # -------------------------------------------------------------
    set_seed(2024)
    cyclenet_model = CycleNetWrapper(seq_len=seq_len, pred_len=pred_len, channels=n_channels, period=24).to(device)
    res_cyclenet = train_and_evaluate_model(
        model_name="CycleNet (Paper Architecture, Period=24)",
        model=cyclenet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )
    results.append(res_cyclenet)

    # -------------------------------------------------------------
    # Experiment 3: TwoStageFourierAttentionNet (Proposed Two-Stage Architecture)
    # -------------------------------------------------------------
    set_seed(2024)
    fourier_attn_model = FourierAttentionWrapper(
        seq_len=seq_len, 
        pred_len=pred_len, 
        channels=n_channels,
        d_model=64,
        n_heads=4,
        num_layers=2
    ).to(device)
    res_fourier = train_and_evaluate_model(
        model_name="TwoStageFourierAttentionNet (Fourier Base + Residual Attention)",
        model=fourier_attn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )
    results.append(res_fourier)

    # -------------------------------------------------------------
    # Final Comparison Table & Report Generation
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print("FINAL 3-MODEL COMPARATIVE BENCHMARK REPORT (ETTh1, L=336, H=96)")
    print("="*85)
    df_summary = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Params': f"{r['params']:,}",
            'Best Epoch': f"{r['best_epoch']}/{r['total_epochs']}",
            'Time (s)': f"{r['training_time_sec']:.1f}",
            'Test MSE': f"{r['test_mse']:.4f}",
            'Test MAE': f"{r['test_mae']:.4f}",
            'Mid-Band MSE': f"{r['band_mse_mid']:.5f}",
            'FVU Mid': f"{r['fvu_mid']:.4f}",
            'FVU High': f"{r['fvu_high']:.4f}",
            'LSD (dB)': f"{r['lsd_db']:.4f}"
        }
        for r in results
    ])
    print(df_summary.to_string(index=False))
    print("="*85)

    # Save to JSON
    os.makedirs(os.path.join(root_dir, 'results'), exist_ok=True)
    json_path = os.path.join(root_dir, 'results', 'three_model_comparison_etth1.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed metrics saved to: {json_path}")

    # Save to Markdown
    md_path = os.path.join(root_dir, 'results', 'three_model_comparison_etth1.md')
    md_content = f"# Empirical Comparison: DLinear vs. CycleNet vs. Fourier-Residual Attention\n\n"
    md_content += f"**Dataset:** ETTh1 | **Lookback Window ($L$):** 336 | **Horizon ($H$):** 96\n"
    md_content += f"**Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | **Seed:** 2024\n\n"
    md_content += "## 1. Benchmark Results Summary\n\n"
    md_content += df_summary.to_markdown(index=False) + "\n\n"
    md_content += "## 2. Analysis & Insights\n\n"
    md_content += f"- **DLinear Test MSE:** {res_dlinear['test_mse']:.4f} (Best Epoch: {res_dlinear['best_epoch']})\n"
    md_content += f"- **CycleNet Test MSE:** {res_cyclenet['test_mse']:.4f} (Best Epoch: {res_cyclenet['best_epoch']})\n"
    md_content += f"- **Fourier-Residual Attention Test MSE:** {res_fourier['test_mse']:.4f} (Best Epoch: {res_fourier['best_epoch']})\n\n"
    md_content += "All models were stopped strictly on validation loss stagnation (patience=7) with full checkpoint restoration.\n"

    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown report compiled to: {md_path}")


if __name__ == '__main__':
    main()

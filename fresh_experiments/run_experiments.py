"""
Fresh 3-Model Experiment Suite on ETTh1 (Lookback L=336, Horizon H=96)
======================================================================
Autonomous execution of 3 models:
1. DLinear (Moving Average Decomposition + RevIN)
2. CycleNet (Paper Architecture: Recurrent Periodic Queue at period W=24 + RevIN)
3. TwoStageFourierAttentionNet (Proposed: Fourier Base Forecaster + Residual Multi-Head Attention)

Protocol:
- Early stopping triggered ONLY when validation loss stops improving / overfits (patience=7).
- Best checkpoint strictly restored for test set evaluation.
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

# Include fresh_experiments directory in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocessing import build_etth1_dataloaders
from models import DLinear, CycleNet, TwoStageFourierAttentionNet


def set_seed(seed: int = 2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_spectral_harmonic_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Computes Mid-Band (Periodic Harmonic) MSE and FVU using 1D real FFT.
    """
    y_t = y_true.detach().cpu().numpy()
    y_p = y_pred.detach().cpu().numpy()
    
    # 1D FFT along the prediction horizon (axis 1)
    fft_true = np.fft.rfft(y_t, axis=1)
    fft_pred = np.fft.rfft(y_p, axis=1)
    
    freq_len = fft_true.shape[1]
    # Mid-band corresponds to 0.1 to 0.4 normalized Nyquist frequency
    low_idx = int(0.1 * freq_len)
    high_idx = int(0.4 * freq_len)
    
    mid_true = fft_true[:, low_idx:high_idx, :]
    mid_pred = fft_pred[:, low_idx:high_idx, :]
    
    # Parseval energy
    mid_error = np.mean(np.abs(mid_true - mid_pred) ** 2)
    mid_var = np.var(np.abs(mid_true))
    mid_fvu = mid_error / (mid_var + 1e-8)
    
    return float(mid_error), float(mid_fvu)


def train_model(model_name: str, model: nn.Module, data_bundle: dict, device: torch.device,
                max_epochs: int = 40, patience: int = 7, lr: float = 1e-3):
    print(f"\n" + "="*70)
    print(f"EXPERIMENT: {model_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | Trainable Parameters: {total_params:,}")
    print(f"Max Epochs: {max_epochs} | Early Stopping Patience: {patience} | Learning Rate: {lr}")
    print("="*70)

    train_loader = data_bundle['train_loader']
    val_loader = data_bundle['val_loader']
    test_loader = data_bundle['test_loader']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

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
            'is_best': is_best
        })

        print(f"Epoch {epoch:02d}/{max_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{tag}")

        # Stop only when overfitting/plateauing past patience
        if patience_counter >= patience:
            print(f">> Early stopping triggered at epoch {epoch}! Overfitting detected.")
            print(f">> Best validation checkpoint was at Epoch {best_epoch} with Val Loss: {best_val_loss:.5f}")
            break

    elapsed_time = time.time() - t_start

    # Restore best checkpoint
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

    print(f"\n--- [TEST EVALUATION for {model_name}] ---")
    print(f"Restored from Epoch: {best_epoch}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Mid-Band (Harmonic) Error: {mid_error:.5f} | Mid-Band FVU: {mid_fvu:.4f}")
    print(f"Training Time: {elapsed_time:.1f}s ({elapsed_time/len(epoch_logs):.2f}s/epoch)")

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


def run_all_three_experiments():
    set_seed(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initialising Fresh Experiments on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Preprocessing
    data_path = os.path.join(current_dir, '..', 'data_sets', 'ETTh1.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', 'ETTh1.csv')

    data_bundle = build_etth1_dataloaders(
        csv_path=data_path,
        seq_len=336,
        pred_len=96,
        batch_size=32
    )
    channels = data_bundle['channels']
    seq_len = data_bundle['seq_len']
    pred_len = data_bundle['pred_len']

    results = []

    # -------------------------------------------------------------
    # Experiment 1: DLinear (Baseline 1)
    # -------------------------------------------------------------
    set_seed(2024)
    m1 = DLinear(seq_len=seq_len, pred_len=pred_len, channels=channels).to(device)
    res_dlinear = train_model(
        model_name="DLinear (AAAI 2023)",
        model=m1,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )
    results.append(res_dlinear)

    # -------------------------------------------------------------
    # Experiment 2: CycleNet (NeurIPS 2024 Paper Architecture)
    # -------------------------------------------------------------
    set_seed(2024)
    m2 = CycleNet(seq_len=seq_len, pred_len=pred_len, channels=channels, period_w=24).to(device)
    res_cyclenet = train_model(
        model_name="CycleNet (NeurIPS 2024, W=24)",
        model=m2,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )
    results.append(res_cyclenet)

    # -------------------------------------------------------------
    # Experiment 3: TwoStageFourierAttentionNet (Proposed Architecture)
    # -------------------------------------------------------------
    set_seed(2024)
    m3 = TwoStageFourierAttentionNet(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1
    ).to(device)
    res_fourier = train_model(
        model_name="TwoStageFourierAttentionNet (Fourier Base + Residual Attention)",
        model=m3,
        data_bundle=data_bundle,
        device=device,
        max_epochs=40,
        patience=7,
        lr=1e-3
    )
    results.append(res_fourier)

    # -------------------------------------------------------------
    # Summary & Reporting
    # -------------------------------------------------------------
    print("\n" + "#"*80)
    print("ALL 3 EXPERIMENTS CONCLUDED. FINAL COMPARATIVE BENCHMARK")
    print("#"*80)

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
    print("#"*80)

    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(results_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON saved to: {json_path}")

    # Save Markdown
    md_path = os.path.join(results_dir, 'summary.md')
    md_text = f"# Fresh 3-Model Empirical Benchmark on ETTh1\n\n"
    md_text += f"- **Dataset:** ETTh1 (Hourly Electricity Transformer Temperature, 7 channels)\n"
    md_text += f"- **Lookback Window ($L$):** 336 steps (14 days)\n"
    md_text += f"- **Forecasting Horizon ($H$):** 96 steps (4 days)\n"
    md_text += f"- **Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
    md_text += f"- **Random Seed:** 2024\n\n"
    md_text += "## 1. Comparative Results Table\n\n"
    md_text += summary_df.to_markdown(index=False) + "\n\n"
    md_text += "## 2. Key Observations\n\n"
    md_text += f"1. **DLinear**: Test MSE = **{res_dlinear['test_mse']:.4f}**, MAE = **{res_dlinear['test_mae']:.4f}** (converged at epoch {res_dlinear['best_epoch']})\n"
    md_text += f"2. **CycleNet**: Test MSE = **{res_cyclenet['test_mse']:.4f}**, MAE = **{res_cyclenet['test_mae']:.4f}** (converged at epoch {res_cyclenet['best_epoch']})\n"
    md_text += f"3. **TwoStageFourierAttentionNet**: Test MSE = **{res_fourier['test_mse']:.4f}**, MAE = **{res_fourier['test_mae']:.4f}** (converged at epoch {res_fourier['best_epoch']})\n\n"
    md_text += "All models stopped automatically when validation loss ceased to improve, ensuring no overfitted checkpoints were evaluated.\n"

    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f"Results Markdown report saved to: {md_path}")


if __name__ == '__main__':
    run_all_three_experiments()

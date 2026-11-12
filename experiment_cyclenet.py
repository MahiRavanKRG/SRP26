import os
import sys
import argparse
import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'cyclenet'))
from cyclenet_pro import (
    find_top_periods_fft,
    filter_true_periods_acf,
    decide_multi_or_single_cycle,
    decide_channel_dependence,
    compute_correlation_metrics,
    load_raw_dataframe,
    get_splits,
    Scaler,
    WindowDataset,
    CycleNetPro,
    get_device,
    EarlyStopping
)

from metrics import fft_band_wise_error_decomposition, psd_comparison
from diagnostics import rolling_error_tracking

def parse_args():
    parser = argparse.ArgumentParser(description='Automated CycleNet Benchmarking with Spectral Diagnostics')
    parser.add_argument('--dataset', type=str, default='ETTh1.csv')
    parser.add_argument('--model_type', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--save_dir', type=str, default='results/cyclenet')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    
    dataset_name = os.path.basename(args.dataset).split('.')[0]
    print(f"\n==================================================================")
    print(f"CycleNet-Pro Experiment: {dataset_name} | H={args.pred_len} | Seed={args.seed} | Device={device}")
    print(f"==================================================================")
    
    # 1. Locate Dataset
    data_path = args.dataset
    if not os.path.exists(data_path):
        alt_path = os.path.join('data', args.dataset)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset {args.dataset} not found in . or data/.")
            
    raw_data = load_raw_dataframe(data_path)
    n_samples, n_channels = raw_data.shape
    print(f"Loaded {n_samples} timesteps, {n_channels} channels from {data_path}")
    
    # 2. Canonical Splitting with historical lookback prefix
    (s_tr, e_tr), (s_va, e_va), (s_te, e_te) = get_splits(n_samples, dataset_name)
    print(f"Splits -> Train: [{s_tr}:{e_tr}], Val: [{s_va}:{e_va}], Test: [{s_te}:{e_te}]")
    
    # 3. Fit scaler strictly on Train split
    scaler = Scaler().fit(raw_data[s_tr:e_tr])
    scaled_data = scaler.transform(raw_data)
    
    # 4. Period & Channel Diagnostics on Training Split
    target_series = raw_data[s_tr:e_tr, -1]  # primary target channel
    candidates = find_top_periods_fft(target_series, n_periods=5, max_period=min(args.seq_len, e_tr // 2))
    true_periods, noise_periods, acf_scores = filter_true_periods_acf(target_series, candidates)
    multi_dec = decide_multi_or_single_cycle(target_series, true_periods)
    periods_to_use = multi_dec['periods_to_use']
    if not periods_to_use:
        periods_to_use = [24]
    
    corr_metrics = compute_correlation_metrics(scaled_data[s_tr:e_tr])
    use_cd, cd_msg = decide_channel_dependence(corr_metrics)
    
    print(f"Periods Detected (FFT+ACF): {true_periods} -> Active: {periods_to_use}")
    print(f"Channel Dependence: {use_cd} ({cd_msg})")
    
    # 5. Datasets with continuous cycle offsets
    # Lookback prefix: val and test include args.seq_len from previous partition
    val_start_offset = max(0, s_va - args.seq_len)
    test_start_offset = max(0, s_te - args.seq_len)
    
    train_dataset = WindowDataset(scaled_data[s_tr:e_tr], args.seq_len, args.pred_len, start_offset=s_tr)
    val_dataset = WindowDataset(scaled_data[val_start_offset:e_va], args.seq_len, args.pred_len, start_offset=val_start_offset)
    test_dataset = WindowDataset(scaled_data[test_start_offset:e_te], args.seq_len, args.pred_len, start_offset=test_start_offset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 6. Instantiate CycleNetPro
    model = CycleNetPro(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=n_channels,
        periods=periods_to_use,
        use_channel_dependence=use_cd,
        model_type=args.model_type,
        d_model=args.d_model,
        use_revin=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience)
    
    # 7. Training Loop with Early Stopping Checkpointing
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
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
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, cyc in val_loader:
                x, y, cyc = x.to(device), y.to(device), cyc.to(device)
                out = model(x, cyc)
                loss = criterion(out, y)
                val_loss += loss.item() * len(x)
        val_loss /= len(val_loader.dataset)
        
        early_stopping.step(val_loss, model)
        if epoch % 5 == 0 or early_stopping.best_loss == val_loss:
            print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} {'*Best*' if early_stopping.best_loss == val_loss else ''}")
            
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
            
    early_stopping.restore(model)
    elapsed_time = time.time() - start_time
    
    # 8. Evaluation on Test Set
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
    
    # Compute Standard Time-Domain Metrics
    mse = float(nn.MSELoss()(y_pred, y_true).item())
    mae = float(nn.L1Loss()(y_pred, y_true).item())
    
    # Compute Frequency-Domain Parseval Diagnostics
    bw_errors = fft_band_wise_error_decomposition(y_true, y_pred, low_cutoff=0.1, high_cutoff=0.4)
    psd_metrics = psd_comparison(y_true, y_pred)
    
    # Rolling Error Anomaly Diagnostics
    err_tracking = rolling_error_tracking(y_true, y_pred, window_size=min(12, args.pred_len))
    total_flags = int(sum([np.sum(res['flags']) for res in err_tracking]))
    
    print(f"\n--- Results for {dataset_name} H={args.pred_len} ---")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | Time: {elapsed_time:.1f}s")
    print(f"Mid-Band MSE: {bw_errors['mse_mid']:.5f} | FVU-mid: {bw_errors['fvu_mid']:.4f} | Energy-Ratio-mid: {bw_errors['energy_ratio_mid']:.4f}")
    print(f"High-Band MSE: {bw_errors['mse_high']:.5f} | FVU-high: {bw_errors['fvu_high']:.4f}")
    print(f"Log-Spectral Distance (LSD): {psd_metrics['mean_lsd']:.4f} dB | Flagged Windows: {total_flags}")
    
    # 9. Save Results
    os.makedirs(args.save_dir, exist_ok=True)
    res_record = {
        'model': f"cyclenet_{args.model_type}",
        'dataset': dataset_name,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'seed': args.seed,
        'mse': round(mse, 4),
        'mae': round(mae, 4),
        'band_mse_low': round(bw_errors['mse_low'], 5),
        'band_mse_mid': round(bw_errors['mse_mid'], 5),
        'band_mse_high': round(bw_errors['mse_high'], 5),
        'band_fvu_low': round(bw_errors['fvu_low'], 4),
        'band_fvu_mid': round(bw_errors['fvu_mid'], 4),
        'band_fvu_high': round(bw_errors['fvu_high'], 4),
        'energy_ratio_low': round(bw_errors['energy_ratio_low'], 4),
        'energy_ratio_mid': round(bw_errors['energy_ratio_mid'], 4),
        'energy_ratio_high': round(bw_errors['energy_ratio_high'], 4),
        'lsd': round(psd_metrics['mean_lsd'], 4),
        'periods_used': periods_to_use,
        'channel_dependence': use_cd,
        'failure_windows_flagged': total_flags,
        'training_time_sec': round(elapsed_time, 2)
    }
    
    res_path = os.path.join(args.save_dir, f"cyclenet_{args.model_type}_{dataset_name}_H{args.pred_len}_seed{args.seed}.json")
    with open(res_path, 'w') as f:
        json.dump(res_record, f, indent=4)
    print(f"Detailed metrics saved to: {res_path}")
    
    # Append to master summary CSV
    summary_csv = os.path.join(args.save_dir, "cyclenet_benchmark_summary.csv")
    df_row = pd.DataFrame([res_record])
    if os.path.exists(summary_csv):
        df_existing = pd.read_csv(summary_csv)
        # remove duplicate run if already present
        mask = (
            (df_existing['model'] == res_record['model']) &
            (df_existing['dataset'] == res_record['dataset']) &
            (df_existing['pred_len'] == res_record['pred_len']) &
            (df_existing['seed'] == res_record['seed'])
        )
        df_existing = df_existing[~mask]
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
        df_combined.to_csv(summary_csv, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)
    print(f"Updated summary CSV at: {summary_csv}")

if __name__ == '__main__':
    main()

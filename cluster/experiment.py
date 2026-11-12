import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataloader import create_dataloader
from models import DLinear, FreqEnhancedDLinear, WaveletEnhancedDLinear, train_one_epoch
from metrics import band_wise_error_decomposition, spectral_coherence_analysis, psd_comparison
from diagnostics import rolling_error_tracking

def parse_args():
    parser = argparse.ArgumentParser(description='MTS Forecasting')
    parser.add_argument('--dataset', type=str, default='ETTh1.csv')
    parser.add_argument('--model', type=str, default='dlinear', choices=['dlinear', 'freq', 'wavelet'])
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}, Dataset: {args.dataset}, Model: {args.model}")
    
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        alt_path = os.path.join("data", args.dataset)
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            print(f"Dataset {args.dataset} not found. Ensure it is downloaded.")
            return
        
    df = pd.read_csv(dataset_path)
    # Exclude date column if present
    data = df.iloc[:, 1:].values.astype(np.float32)
    
    scaler = StandardScaler()
    train_split = int(len(data) * 0.7)
    val_split = int(len(data) * 0.8)
    
    scaler.fit(data[:train_split])
    scaled_data = scaler.transform(data)
    
    channels = scaled_data.shape[1]
    
    train_loader = create_dataloader(scaled_data[:train_split], batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len, shuffle=True)
    test_loader = create_dataloader(scaled_data[val_split:], batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len, shuffle=False)
    
    if args.model == 'dlinear':
        model = DLinear(seq_len=args.seq_len, pred_len=args.pred_len, channels=channels).to(device)
    elif args.model == 'freq':
        model = FreqEnhancedDLinear(seq_len=args.seq_len, pred_len=args.pred_len, channels=channels).to(device)
    elif args.model == 'wavelet':
        model = WaveletEnhancedDLinear(seq_len=args.seq_len, pred_len=args.pred_len, channels=channels).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print("Training...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch} Loss: {loss:.6f}")
        
    print("Evaluating...")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            p = model(batch_x)
            preds.append(p)
            trues.append(batch_y)
            
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)
    
    # --------------------------------
    # 1. Standard Metrics
    # --------------------------------
    mse = nn.MSELoss()(y_pred, y_true).item()
    mae = nn.L1Loss()(y_pred, y_true).item()
    
    # --------------------------------
    # 2. Spectral Diagnostics
    # --------------------------------
    bw_errors = band_wise_error_decomposition(y_true, y_pred)
    coherence = spectral_coherence_analysis(y_true, y_pred)
    psd_dist = psd_comparison(y_true, y_pred)
    
    alpha_val = None
    if hasattr(model, 'alpha_raw'):
        alpha_val = torch.sigmoid(model.alpha_raw).item()
        
    # --------------------------------
    # 3. Failure Mode Extraction
    # --------------------------------
    error_tracking = rolling_error_tracking(y_true, y_pred, window_size=min(12, args.pred_len))
    
    # Count how many windows were flagged as anomalies
    total_flags = sum([np.sum(res['flags']) for res in error_tracking])
    
    # Dump metrics to JSON
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'mse': float(mse),
        'mae': float(mae),
        'band_mse_low': float(bw_errors.get('mse_low', 0)),
        'band_mse_mid': float(bw_errors.get('mse_mid', 0)),
        'band_mse_high': float(bw_errors.get('mse_high', 0)),
        'band_fvu_low': float(bw_errors.get('fvu_low', 0)),
        'band_fvu_mid': float(bw_errors.get('fvu_mid', 0)),
        'band_fvu_high': float(bw_errors.get('fvu_high', 0)),
        'energy_ratio_low': float(bw_errors.get('energy_ratio_low', 0)),
        'energy_ratio_mid': float(bw_errors.get('energy_ratio_mid', 0)),
        'energy_ratio_high': float(bw_errors.get('energy_ratio_high', 0)),
        'mean_coherence': float(coherence.get('mean_spectral_coherence', 0)),
        'lsd': float(psd_dist.get('mean_lsd', 0)),
        'alpha': float(alpha_val) if alpha_val is not None else None,
        'failure_windows_flagged': int(total_flags)
    }
    
    os.makedirs('results', exist_ok=True)
    res_file = f"results/{args.model}_{os.path.basename(args.dataset).split('.')[0]}_H{args.pred_len}.json"
    with open(res_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {res_file}")

if __name__ == '__main__':
    main()

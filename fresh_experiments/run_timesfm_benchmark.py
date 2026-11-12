"""
Comprehensive Foundation Model Benchmark:
Google TimesFM 2.5 (200M) vs. Amazon Chronos-T5 (8M) vs. Two-Stage Hybrids vs. CycleNet
========================================================================================
Benchmarking:
  1. Google TimesFM-2.5-200M (Zero-Shot) - Native patched 96-step forecasting.
  2. FTimesFM (Fourier Base + TimesFM on Residuals) - Decoupled two-stage hybrid.
  3. Amazon Chronos-T5-Tiny (Zero-Shot) - Discrete tokenized autoregressive model (64-step native).
  4. FCMath (Fourier Base + Chronos on Residuals) - Two-stage hybrid.
  5. CycleNet (W=24) - Ultra-lightweight periodic recurrent queue (32k params).

Protocol:
  - Lookback L = 336, Forecast Horizon H = 96.
  - Canonical chronological splits (ETT: 12/4/4 months, Non-ETT: 70/10/20).
  - Train-only Z-score standardization.
  - Multi-band Parseval spectral diagnostics: Mid-Band FVU & Log-Spectral Distance (LSD in dB).
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "models"))

from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
from chronos import ChronosPipeline
from srp26 import (
    CycleNet,
    StandardScaler,
    compute_spectral_diagnostics,
    extract_fourier_base_1d,
    set_seed,
)


def load_dataset(dataset_name: str, seq_len: int = 336, pred_len: int = 96, num_eval_windows: int = 15):
    path = os.path.join(parent_dir, "data_sets", dataset_name)
    if not os.path.exists(path):
        path = os.path.join(parent_dir, "data", dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {path}")

    if dataset_name.endswith('.txt'):
        df = pd.read_csv(path, header=None)
        raw_data = df.values.astype(np.float32)
    else:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        raw_data = df.values.astype(np.float32)

    total_len = len(raw_data)
    name_lower = dataset_name.lower()
    if name_lower.startswith('etth'):
        n_train = 12 * 30 * 24
        n_val = 4 * 30 * 24
        n_test = 4 * 30 * 24
    elif name_lower.startswith('ettm'):
        n_train = 12 * 30 * 24 * 4
        n_val = 4 * 30 * 24 * 4
        n_test = 4 * 30 * 24 * 4
    else:
        n_train = int(total_len * 0.70)
        n_val = int(total_len * 0.10)
        n_test = total_len - n_train - n_val

    train_raw = raw_data[:n_train]
    test_raw = raw_data[n_train + n_val - seq_len : min(total_len, n_train + n_val + n_test)]

    scaler = StandardScaler()
    scaler.fit(train_raw)
    train_norm = scaler.transform(train_raw)
    test_norm = scaler.transform(test_raw)

    total_valid = len(test_norm) - seq_len - pred_len + 1
    stride = max(1, total_valid // num_eval_windows)
    eval_indices = np.arange(0, total_valid, stride)[:num_eval_windows]

    x_list, y_list = [], []
    for idx in eval_indices:
        x_list.append(test_norm[idx : idx + seq_len])
        y_list.append(test_norm[idx + seq_len : idx + seq_len + pred_len])

    x_eval = np.array(x_list, dtype=np.float32)  # [B, L, C]
    y_eval = np.array(y_list, dtype=np.float32)  # [B, H, C]
    return x_eval, y_eval, train_norm, scaler, raw_data.shape[1]


def run_benchmark_on_dataset(
    dataset_name: str,
    tfm_model,
    chronos_pipeline,
    pred_len: int = 96,
    batch_size: int = 64,
):
    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print(f"BENCHMARK SUITE: {dataset_name} (H={pred_len})")
    print("=" * 90)

    seq_len = 336
    x_eval, y_eval, train_norm, scaler, C = load_dataset(dataset_name, seq_len=seq_len, pred_len=pred_len)
    B, L, _ = x_eval.shape
    total_series = B * C
    print(f"Dataset: {dataset_name:12s} | Channels: {C:4d} | Windows: {B:2d} | Sequences: {total_series:5d}")

    dataset_results = []

    # Precompute Fourier Base and Residuals for hybrid models
    print("Precomputing Fourier base and residuals ...", end=" ", flush=True)
    t_base_0 = time.time()
    x_base = np.zeros_like(x_eval)
    y_base = np.zeros_like(y_eval)
    for b in range(B):
        for c in range(C):
            xb, yb = extract_fourier_base_1d(x_eval[b, :, c], pred_len, top_k=3)
            x_base[b, :, c] = xb
            y_base[b, :, c] = yb
    x_res = x_eval - x_base
    print(f"Done in {time.time()-t_base_0:.2f}s!")

    # -------------------------------------------------------------
    # 1. Google TimesFM-Raw (Zero-Shot, 200M)
    # -------------------------------------------------------------
    print("  [1/5] Running Google TimesFM-Raw (Zero-Shot) ...", end=" ", flush=True)
    flat_x = [x_eval[b, :, c] for b in range(B) for c in range(C)]
    t0 = time.time()
    tfm_point_raw, _ = tfm_model.forecast(horizon=pred_len, inputs=flat_x)
    dt_tfm_raw = time.time() - t0
    y_pred_tfm_raw = tfm_point_raw.reshape(B, C, pred_len).transpose(0, 2, 1)

    mse_tfm_raw = float(np.mean((y_pred_tfm_raw - y_eval)**2))
    mae_tfm_raw = float(np.mean(np.abs(y_pred_tfm_raw - y_eval)))
    spec_tfm_raw = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_tfm_raw))
    print(f"Done in {dt_tfm_raw:.2f}s | MSE: {mse_tfm_raw:.4f} | MAE: {mae_tfm_raw:.4f} | LSD: {spec_tfm_raw['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'TimesFM-Raw (Zero-Shot)',
        'Params': '200M',
        'MSE': round(mse_tfm_raw, 4),
        'MAE': round(mae_tfm_raw, 4),
        'Mid_FVU': round(spec_tfm_raw['fvu_mid'], 4),
        'LSD_dB': round(spec_tfm_raw['lsd_db'], 2),
        'Runtime_Sec': round(dt_tfm_raw, 2),
    })

    # -------------------------------------------------------------
    # 2. Google FTimesFM (Fourier Base + TimesFM on Residuals, 200M)
    # -------------------------------------------------------------
    print("  [2/5] Running FTimesFM (Fourier + TimesFM) ...", end=" ", flush=True)
    flat_res = [x_res[b, :, c] for b in range(B) for c in range(C)]
    t0 = time.time()
    tfm_point_res, _ = tfm_model.forecast(horizon=pred_len, inputs=flat_res)
    dt_ftfm = time.time() - t0
    y_pred_res_tfm = tfm_point_res.reshape(B, C, pred_len).transpose(0, 2, 1)
    y_pred_ftfm = y_base + y_pred_res_tfm

    mse_ftfm = float(np.mean((y_pred_ftfm - y_eval)**2))
    mae_ftfm = float(np.mean(np.abs(y_pred_ftfm - y_eval)))
    spec_ftfm = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_ftfm))
    print(f"Done in {dt_ftfm:.2f}s | MSE: {mse_ftfm:.4f} | MAE: {mae_ftfm:.4f} | LSD: {spec_ftfm['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'FTimesFM (Fourier+TimesFM)',
        'Params': '200M',
        'MSE': round(mse_ftfm, 4),
        'MAE': round(mae_ftfm, 4),
        'Mid_FVU': round(spec_ftfm['fvu_mid'], 4),
        'LSD_dB': round(spec_ftfm['lsd_db'], 2),
        'Runtime_Sec': round(dt_ftfm, 2),
    })

    # -------------------------------------------------------------
    # 3. Amazon Chronos-Raw (Zero-Shot, 8M)
    # -------------------------------------------------------------
    print("  [3/5] Running Chronos-Raw (Zero-Shot) ...", end=" ", flush=True)
    t0 = time.time()
    res_flat_raw = x_eval.transpose(0, 2, 1).reshape(B * C, L)
    ctx_raw = torch.tensor(res_flat_raw, dtype=torch.float32)
    chronos_preds_raw = []
    ch_bs = 256
    for i in range(0, len(ctx_raw), ch_bs):
        with torch.no_grad():
            f_dist = chronos_pipeline.predict(ctx_raw[i : i + ch_bs], prediction_length=pred_len, num_samples=1)
            chronos_preds_raw.append(f_dist.squeeze(1).cpu().numpy())
    y_pred_ch_raw = np.concatenate(chronos_preds_raw, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
    dt_ch_raw = time.time() - t0

    mse_ch_raw = float(np.mean((y_pred_ch_raw - y_eval)**2))
    mae_ch_raw = float(np.mean(np.abs(y_pred_ch_raw - y_eval)))
    spec_ch_raw = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_ch_raw))
    print(f"Done in {dt_ch_raw:.2f}s | MSE: {mse_ch_raw:.4f} | MAE: {mae_ch_raw:.4f} | LSD: {spec_ch_raw['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'Chronos-Raw (Zero-Shot)',
        'Params': '8M',
        'MSE': round(mse_ch_raw, 4),
        'MAE': round(mae_ch_raw, 4),
        'Mid_FVU': round(spec_ch_raw['fvu_mid'], 4),
        'LSD_dB': round(spec_ch_raw['lsd_db'], 2),
        'Runtime_Sec': round(dt_ch_raw, 2),
    })

    # -------------------------------------------------------------
    # 4. Amazon FCMath (Fourier Base + Chronos on Residuals, 8M)
    # -------------------------------------------------------------
    print("  [4/5] Running FCMath (Fourier+Chronos) ...", end=" ", flush=True)
    t0 = time.time()
    res_flat = x_res.transpose(0, 2, 1).reshape(B * C, L)
    ctx_res = torch.tensor(res_flat, dtype=torch.float32)
    fcmath_res_preds = []
    for i in range(0, len(ctx_res), ch_bs):
        with torch.no_grad():
            f_dist = chronos_pipeline.predict(ctx_res[i : i + ch_bs], prediction_length=pred_len, num_samples=1)
            fcmath_res_preds.append(f_dist.squeeze(1).cpu().numpy())
    y_pred_fcmath_res = np.concatenate(fcmath_res_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
    y_pred_fcmath = y_base + y_pred_fcmath_res
    dt_fcmath = time.time() - t0

    mse_fcmath = float(np.mean((y_pred_fcmath - y_eval)**2))
    mae_fcmath = float(np.mean(np.abs(y_pred_fcmath - y_eval)))
    spec_fcmath = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_fcmath))
    print(f"Done in {dt_fcmath:.2f}s | MSE: {mse_fcmath:.4f} | MAE: {mae_fcmath:.4f} | LSD: {spec_fcmath['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'FCMath (Fourier+Chronos)',
        'Params': '8M',
        'MSE': round(mse_fcmath, 4),
        'MAE': round(mae_fcmath, 4),
        'Mid_FVU': round(spec_fcmath['fvu_mid'], 4),
        'LSD_dB': round(spec_fcmath['lsd_db'], 2),
        'Runtime_Sec': round(dt_fcmath, 2),
    })

    # -------------------------------------------------------------
    # 5. CycleNet (W=24, 32k parameters)
    # -------------------------------------------------------------
    print("  [5/5] Running CycleNet (W=24) ...", end=" ", flush=True)
    t0 = time.time()
    cyclenet = CycleNet(seq_len=seq_len, pred_len=pred_len, channels=C, cycle_len=24).to(device)
    optimizer = torch.optim.AdamW(cyclenet.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()
    cyclenet.train()
    
    # 40 fast steps on train_norm
    batch_sz = 32
    max_train_steps = min(40, (len(train_norm) - seq_len - pred_len) // batch_sz)
    for step in range(max_train_steps):
        idx = step * batch_sz
        xb = torch.tensor(np.array([train_norm[idx + b : idx + b + seq_len] for b in range(batch_sz)]), device=device)
        yb = torch.tensor(np.array([train_norm[idx + b + seq_len : idx + b + seq_len + pred_len] for b in range(batch_sz)]), device=device)
        optimizer.zero_grad()
        loss = criterion(cyclenet(xb), yb)
        loss.backward()
        optimizer.step()

    cyclenet.eval()
    with torch.no_grad():
        y_pred_cyc = cyclenet(torch.tensor(x_eval, device=device)).cpu().numpy()
    dt_cyc = time.time() - t0

    mse_cyc = float(np.mean((y_pred_cyc - y_eval)**2))
    mae_cyc = float(np.mean(np.abs(y_pred_cyc - y_eval)))
    spec_cyc = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_cyc))
    print(f"Done in {dt_cyc:.2f}s | MSE: {mse_cyc:.4f} | MAE: {mae_cyc:.4f} | LSD: {spec_cyc['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'CycleNet (W=24)',
        'Params': '32K',
        'MSE': round(mse_cyc, 4),
        'MAE': round(mae_cyc, 4),
        'Mid_FVU': round(spec_cyc['fvu_mid'], 4),
        'LSD_dB': round(spec_cyc['lsd_db'], 2),
        'Runtime_Sec': round(dt_cyc, 2),
    })

    df = pd.DataFrame(dataset_results)
    print("\n" + df.to_string(index=False) + "\n")
    return dataset_results


def main():
    parser = argparse.ArgumentParser(description="Google TimesFM vs. Chronos vs. CycleNet Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1.csv", "weather.csv", "traffic.csv"], help="Datasets to evaluate")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecast horizon")
    parser.add_argument("--batch_size", type=int, default=128, help="TimesFM batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on device: {device} ...")

    # Load Google TimesFM 2.5 (200M)
    print("Loading Google TimesFM 2.5 (200M) ...", flush=True)
    tfm_model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    forecast_config = ForecastConfig(
        max_context=352,
        max_horizon=128,
        per_core_batch_size=args.batch_size,
        normalize_inputs=True,
        infer_is_positive=False,
        force_flip_invariance=True,
    )
    tfm_model.compile(forecast_config)

    # Load Amazon Chronos-T5 (Tiny, 8M)
    print("Loading Amazon Chronos-T5-Tiny (8M) ...", flush=True)
    chronos_pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map=device,
        torch_dtype=torch.float32,
    )
    chronos_pipeline.tokenizer.config.prediction_length = args.pred_len
    chronos_pipeline.model.config.prediction_length = args.pred_len

    all_results = []
    for ds in args.datasets:
        res = run_benchmark_on_dataset(
            ds,
            tfm_model=tfm_model,
            chronos_pipeline=chronos_pipeline,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
        )
        all_results.extend(res)

    # Save summary
    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"timesfm_vs_chronos_benchmark_H{args.pred_len}.json")
    md_path = os.path.join(out_dir, f"timesfm_vs_chronos_benchmark_H{args.pred_len}.md")

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    master_df = pd.DataFrame(all_results)
    with open(md_path, 'w') as f:
        f.write(f"# Google TimesFM 2.5 (200M) vs. Amazon Chronos vs. CycleNet (H={args.pred_len})\n\n")
        f.write(master_df.to_markdown(index=False))
        f.write("\n")

    print(f"\nSaved comprehensive benchmark to:\n  - {json_path}\n  - {md_path}\n")


if __name__ == "__main__":
    main()

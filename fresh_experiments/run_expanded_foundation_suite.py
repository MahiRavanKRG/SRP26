"""
Expanded Time-Series Foundation Model Suite:
Google TimesFM (200M) vs. Amazon Chronos-Bolt-Base (200M) vs. IBM Granite TTM (805K) vs. Chronos-T5 (8M) vs. CycleNet
======================================================================================================================
Benchmarking:
  1. Google TimesFM 2.5 (200M) - Continuous patch decoder transformer (out patch 128)
  2. Amazon Chronos-Bolt-Base (200M) - Non-autoregressive patch-based foundation model
  3. IBM Granite TinyTimeMixer TTM-R1 (805K) - Ultra-lightweight MLP-Mixer foundation model
  4. Amazon Chronos-T5 (8M) - Classic discrete autoregressive baseline (64-step native)
  5. CycleNet (W=24, 32K) - Ultra-lightweight recurrent periodic queue champion

Protocol:
  - Lookback L = 336, Forecast Horizon H = 96
  - Standardized chronological splits (ETT: 12/4/4, Non-ETT: 70/10/20)
  - Train-only Z-score standardization
  - Parseval Mid-Band FVU & Log-Spectral Distance (LSD in dB)
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "models"))

from srp26.utils import ensure_hf_cache

ensure_hf_cache()

import time
import json
import argparse
from typing import Dict, List, Tuple

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
from chronos import ChronosPipeline, ChronosBoltPipeline
from tsfm_public import TinyTimeMixerForPrediction
from srp26 import CycleNet, StandardScaler, compute_spectral_diagnostics, set_seed


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


def run_suite_on_dataset(
    dataset_name: str,
    tfm_model,
    bolt_pipeline,
    ttm_model,
    chronos_t5_pipeline,
    pred_len: int = 96,
):
    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 95)
    print(f"BENCHMARKING FOUNDATION SUITE ON: {dataset_name} (H={pred_len})")
    print("=" * 95)

    seq_len = 336
    x_eval, y_eval, train_norm, scaler, C = load_dataset(dataset_name, seq_len=seq_len, pred_len=pred_len)
    B, L, _ = x_eval.shape
    total_series = B * C
    print(f"Dataset: {dataset_name:12s} | Channels: {C:4d} | Windows: {B:2d} | Sequences: {total_series:5d}")

    dataset_results = []

    # -------------------------------------------------------------
    # 1. Google TimesFM 2.5 (200M)
    # -------------------------------------------------------------
    print("  [1/5] Running Google TimesFM 2.5 (200M) ...", end=" ", flush=True)
    flat_x = [x_eval[b, :, c] for b in range(B) for c in range(C)]
    t0 = time.time()
    tfm_point_raw, _ = tfm_model.forecast(horizon=pred_len, inputs=flat_x)
    dt_tfm = time.time() - t0
    y_pred_tfm = tfm_point_raw.reshape(B, C, pred_len).transpose(0, 2, 1)

    mse_tfm = float(np.mean((y_pred_tfm - y_eval)**2))
    mae_tfm = float(np.mean(np.abs(y_pred_tfm - y_eval)))
    spec_tfm = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_tfm))
    print(f"Done in {dt_tfm:.2f}s | MSE: {mse_tfm:.4f} | MAE: {mae_tfm:.4f} | LSD: {spec_tfm['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'Google TimesFM 2.5',
        'Architecture': 'Patched Transformer (Non-AR)',
        'Params': '200M',
        'MSE': round(mse_tfm, 4),
        'MAE': round(mae_tfm, 4),
        'Mid_FVU': round(spec_tfm['fvu_mid'], 4),
        'LSD_dB': round(spec_tfm['lsd_db'], 2),
        'Runtime_Sec': round(dt_tfm, 2),
    })

    # -------------------------------------------------------------
    # 2. Amazon Chronos-Bolt-Base (200M)
    # -------------------------------------------------------------
    torch.cuda.empty_cache()
    print("  [2/5] Running Amazon Chronos-Bolt-Base (200M) ...", end=" ", flush=True)
    t0 = time.time()
    res_flat = x_eval.transpose(0, 2, 1).reshape(B * C, L)
    ctx_tensor = torch.tensor(res_flat, dtype=torch.float32).to(bolt_pipeline.model.device)
    bolt_preds = []
    bs = 64
    for i in range(0, len(ctx_tensor), bs):
        with torch.no_grad():
            f_dist = bolt_pipeline.predict(ctx_tensor[i : i + bs], prediction_length=pred_len)
            # Index 4 is median (50% quantile)
            point_pred = f_dist[:, 4, :].cpu().numpy()
            bolt_preds.append(point_pred)
    y_pred_bolt = np.concatenate(bolt_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
    dt_bolt = time.time() - t0

    mse_bolt = float(np.mean((y_pred_bolt - y_eval)**2))
    mae_bolt = float(np.mean(np.abs(y_pred_bolt - y_eval)))
    spec_bolt = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_bolt))
    print(f"Done in {dt_bolt:.2f}s | MSE: {mse_bolt:.4f} | MAE: {mae_bolt:.4f} | LSD: {spec_bolt['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'Amazon Chronos-Bolt-Base',
        'Architecture': 'Patch Encoder-Decoder (Non-AR)',
        'Params': '200M',
        'MSE': round(mse_bolt, 4),
        'MAE': round(mae_bolt, 4),
        'Mid_FVU': round(spec_bolt['fvu_mid'], 4),
        'LSD_dB': round(spec_bolt['lsd_db'], 2),
        'Runtime_Sec': round(dt_bolt, 2),
    })

    # -------------------------------------------------------------
    # 3. IBM Granite TinyTimeMixer TTM-R1 (805K)
    # -------------------------------------------------------------
    torch.cuda.empty_cache()
    print("  [3/5] Running IBM Granite TTM-R1 (805K) ...", end=" ", flush=True)
    t0 = time.time()
    x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(ttm_model.device)
    pad_len = ttm_model.config.context_length - L
    x_padded = torch.cat([x_tensor[:, :1, :].repeat(1, pad_len, 1), x_tensor], dim=1)
    with torch.no_grad():
        out = ttm_model(past_values=x_padded)
        y_pred_ttm = out.prediction_outputs.cpu().numpy()
    dt_ttm = time.time() - t0

    mse_ttm = float(np.mean((y_pred_ttm - y_eval)**2))
    mae_ttm = float(np.mean(np.abs(y_pred_ttm - y_eval)))
    spec_ttm = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_ttm))
    print(f"Done in {dt_ttm:.2f}s | MSE: {mse_ttm:.4f} | MAE: {mae_ttm:.4f} | LSD: {spec_ttm['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'IBM Granite TTM-R1',
        'Architecture': 'MLP-Mixer Foundation Model',
        'Params': '805K',
        'MSE': round(mse_ttm, 4),
        'MAE': round(mae_ttm, 4),
        'Mid_FVU': round(spec_ttm['fvu_mid'], 4),
        'LSD_dB': round(spec_ttm['lsd_db'], 2),
        'Runtime_Sec': round(dt_ttm, 2),
    })

    # -------------------------------------------------------------
    # 4. Amazon Chronos-T5 (Tiny, 8M)
    # -------------------------------------------------------------
    torch.cuda.empty_cache()
    print("  [4/5] Running Amazon Chronos-T5 (Tiny, 8M) ...", end=" ", flush=True)
    t0 = time.time()
    ctx_cpu = torch.tensor(res_flat, dtype=torch.float32)
    ch_preds = []
    for i in range(0, len(ctx_cpu), bs):
        with torch.no_grad():
            f_dist = chronos_t5_pipeline.predict(ctx_cpu[i : i + bs], prediction_length=pred_len, num_samples=1)
            ch_preds.append(f_dist.squeeze(1).cpu().numpy())
    y_pred_ch = np.concatenate(ch_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
    dt_ch = time.time() - t0

    mse_ch = float(np.mean((y_pred_ch - y_eval)**2))
    mae_ch = float(np.mean(np.abs(y_pred_ch - y_eval)))
    spec_ch = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_ch))
    print(f"Done in {dt_ch:.2f}s | MSE: {mse_ch:.4f} | MAE: {mae_ch:.4f} | LSD: {spec_ch['lsd_db']:.2f} dB")
    dataset_results.append({
        'Dataset': dataset_name,
        'Model': 'Amazon Chronos-T5',
        'Architecture': 'Discrete Tokenized T5 (AR)',
        'Params': '8M',
        'MSE': round(mse_ch, 4),
        'MAE': round(mae_ch, 4),
        'Mid_FVU': round(spec_ch['fvu_mid'], 4),
        'LSD_dB': round(spec_ch['lsd_db'], 2),
        'Runtime_Sec': round(dt_ch, 2),
    })

    # -------------------------------------------------------------
    # 5. CycleNet (W=24, 32K)
    # -------------------------------------------------------------
    torch.cuda.empty_cache()
    print("  [5/5] Running CycleNet (W=24, 32K) ...", end=" ", flush=True)
    t0 = time.time()
    cyclenet = CycleNet(seq_len=seq_len, pred_len=pred_len, channels=C, cycle_len=24).to(device)
    optimizer = torch.optim.AdamW(cyclenet.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()
    cyclenet.train()
    
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
        'Architecture': 'Recurrent Periodic Queue',
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
    parser = argparse.ArgumentParser(description="Expanded Foundation Model Benchmark Suite")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1.csv", "weather.csv", "traffic.csv"], help="Datasets to evaluate")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecast horizon")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for TimesFM")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on device: {device} (VRAM: RTX 4050 6GB) ...")

    # 1. Google TimesFM 2.5 (200M)
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

    # 2. Amazon Chronos-Bolt-Base (200M)
    print("Loading Amazon Chronos-Bolt-Base (200M) ...", flush=True)
    bolt_pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # 3. IBM Granite TTM-R1 (805K)
    print("Loading IBM Granite TinyTimeMixer TTM-R1 (805K) ...", flush=True)
    ttm_model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r1").to(device)
    ttm_model.eval()

    # 4. Amazon Chronos-T5 (Tiny, 8M)
    print("Loading Amazon Chronos-T5 (8M) ...", flush=True)
    chronos_t5_pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    chronos_t5_pipeline.tokenizer.config.prediction_length = args.pred_len
    chronos_t5_pipeline.model.config.prediction_length = args.pred_len

    all_results = []
    for ds in args.datasets:
        res = run_suite_on_dataset(
            ds,
            tfm_model=tfm_model,
            bolt_pipeline=bolt_pipeline,
            ttm_model=ttm_model,
            chronos_t5_pipeline=chronos_t5_pipeline,
            pred_len=args.pred_len,
        )
        all_results.extend(res)

    # Save summary
    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"expanded_foundation_suite_H{args.pred_len}.json")
    md_path = os.path.join(out_dir, f"expanded_foundation_suite_H{args.pred_len}.md")

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    master_df = pd.DataFrame(all_results)
    with open(md_path, 'w') as f:
        f.write(f"# Expanded Time-Series Foundation Model Suite Benchmark (H={args.pred_len})\n\n")
        f.write(master_df.to_markdown(index=False))
        f.write("\n")

    print(f"\nSaved master suite results to:\n  - {json_path}\n  - {md_path}\n")


if __name__ == "__main__":
    main()

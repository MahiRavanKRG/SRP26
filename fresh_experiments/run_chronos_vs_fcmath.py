"""
Multi-Dataset Benchmark: Chronos-Raw vs. FCMath vs. FCMath-Tuned vs. CycleNet
=============================================================================
Evaluates across the canonical LTSF benchmark datasets:
  - ETTh1.csv (Hourly, Diurnal Periodicity)
  - ETTh2.csv (Hourly, Non-Stationary Drift / Sensor Faults)
  - weather.csv (10-min, Dynamic Climatology)
  - electricity.csv (Hourly, Multi-Scale Power Grid)
  - traffic.csv (Hourly, High-Dimensional Spatial Network)
  - ETTm1.csv (15-min, High-Frequency Harmonics)

Protocol:
  - Standardized Chronological Splitting (ETT 12/4/4 vs. Non-ETT 70/10/20).
  - Lookback L=336, Horizon H=96, Train-only Z-score standardization.
  - Parseval subband spectral decomposition & Log-Spectral Distance.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "models"))

from fcmath import FCMath
from srp26 import (
    CycleNet,
    StandardScaler,
    compute_spectral_diagnostics,
    set_seed,
    train_and_evaluate_model,
)


def load_dataset_standardized(dataset_name: str, seq_len: int = 336, pred_len: int = 96):
    path = os.path.join(parent_dir, "data_sets", dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    if dataset_name.endswith('.txt'):
        df = pd.read_csv(path, header=None)
        raw_data = df.values.astype(np.float32)
    else:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        raw_data = df.values.astype(np.float32)

    total_len = len(raw_data)

    if 'ETTh' in dataset_name:
        n_train = 8640
        n_val = 2880
        n_test = 2880
    elif 'ETTm' in dataset_name:
        n_train = 34560
        n_val = 11520
        n_test = 11520
    else:
        n_train = int(total_len * 0.70)
        n_val = int(total_len * 0.10)
        n_test = total_len - n_train - n_val

    # Slices with lookback prefix overlap
    train_raw = raw_data[:n_train]
    val_raw = raw_data[n_train - seq_len : n_train + n_val]
    test_raw = raw_data[n_train + n_val - seq_len : n_train + n_val + n_test]

    scaler = StandardScaler()
    scaler.fit(train_raw)

    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw)
    test_norm = scaler.transform(test_raw)

    return train_norm, val_norm, test_norm, raw_data.shape[1]


def extract_evaluation_windows(test_norm: np.ndarray, seq_len: int = 336, pred_len: int = 96,
                               num_eval_windows: int = 25, stride: int = 24):
    total_valid = len(test_norm) - seq_len - pred_len + 1
    indices = np.arange(0, total_valid, stride)[:num_eval_windows]

    x_list, y_list = [], []
    for idx in indices:
        x_list.append(test_norm[idx : idx + seq_len])
        y_list.append(test_norm[idx + seq_len : idx + seq_len + pred_len])

    return np.array(x_list), np.array(y_list)


# =====================================================================
# 2. FCMath-Tuned: Trend-Aware + Fourier Harmonic Forecaster
# =====================================================================
class FCMathTuned:
    def __init__(self, top_k: int = 3, model_name: str = "amazon/chronos-t5-tiny", device: str = None):
        self.top_k = top_k
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def _lazy_load_chronos(self):
        if self.pipeline is None:
            from chronos import ChronosPipeline
            dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=dtype
            )

    def extract_trend_and_fourier_base(self, x: np.ndarray, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        L, C = x.shape
        x_base = np.zeros_like(x)
        y_base = np.zeros((pred_len, C), dtype=x.dtype)

        t_in = np.arange(L)
        t_future = np.arange(L, L + pred_len)

        for c in range(C):
            series = x[:, c]
            poly = np.polyfit(t_in, series, deg=1)
            trend_in = np.polyval(poly, t_in)
            trend_future = np.polyval(poly, t_future)

            detrended = series - trend_in
            fft_vals = np.fft.rfft(detrended)
            freqs = np.fft.rfftfreq(L)
            amps = np.abs(fft_vals)

            if len(amps) > 1:
                k = min(self.top_k, len(amps) - 1)
                top_indices = np.argsort(amps[1:])[-k:] + 1
            else:
                top_indices = np.array([0])

            filtered_fft = np.zeros_like(fft_vals)
            filtered_fft[top_indices] = fft_vals[top_indices]
            harmonics_in = np.fft.irfft(filtered_fft, n=L)
            x_base[:, c] = trend_in + harmonics_in

            harmonics_future = np.zeros(pred_len, dtype=x.dtype)
            for idx in top_indices:
                amp = amps[idx] / (L / 2.0)
                phase = np.angle(fft_vals[idx])
                freq = freqs[idx]
                harmonics_future += amp * np.cos(2.0 * np.pi * freq * t_future + phase)

            y_base[:, c] = trend_future + harmonics_future

        return x_base, y_base

    def forecast_batch(self, x_batch: np.ndarray, pred_len: int, batch_size: int = 128) -> np.ndarray:
        self._lazy_load_chronos()
        B, L, C = x_batch.shape

        x_base_all = np.zeros_like(x_batch)
        y_base_all = np.zeros((B, pred_len, C), dtype=x_batch.dtype)

        for b in range(B):
            xb, yb = self.extract_trend_and_fourier_base(x_batch[b], pred_len)
            x_base_all[b] = xb
            y_base_all[b] = yb

        x_res_all = x_batch - x_base_all
        res_flattened = x_res_all.transpose(0, 2, 1).reshape(B * C, L)
        context_tensor = torch.tensor(res_flattened, dtype=torch.float32)

        all_res_preds = []
        for i in range(0, len(context_tensor), batch_size):
            batch_ctx = context_tensor[i : i + batch_size]
            with torch.no_grad():
                forecast_dist = self.pipeline.predict(batch_ctx, prediction_length=pred_len, num_samples=15)
                median_pred = np.quantile(forecast_dist.cpu().numpy(), q=0.5, axis=1)
                all_res_preds.append(median_pred)

        y_res_flat = np.concatenate(all_res_preds, axis=0)
        y_res_all = y_res_flat.reshape(B, C, pred_len).transpose(0, 2, 1)

        return y_base_all + y_res_all


# =====================================================================
# 3. Raw Chronos Forecaster
# =====================================================================
class ChronosRawForecaster:
    def __init__(self, model_name: str = "amazon/chronos-t5-tiny", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def _lazy_load(self):
        if self.pipeline is None:
            from chronos import ChronosPipeline
            dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=dtype
            )

    def forecast_batch(self, x_batch: np.ndarray, pred_len: int, batch_size: int = 128) -> np.ndarray:
        self._lazy_load()
        B, L, C = x_batch.shape
        x_flattened = x_batch.transpose(0, 2, 1).reshape(B * C, L)
        context_tensor = torch.tensor(x_flattened, dtype=torch.float32)

        all_preds = []
        for i in range(0, len(context_tensor), batch_size):
            batch_ctx = context_tensor[i : i + batch_size]
            with torch.no_grad():
                forecast_dist = self.pipeline.predict(batch_ctx, prediction_length=pred_len, num_samples=15)
                median_pred = np.quantile(forecast_dist.cpu().numpy(), q=0.5, axis=1)
                all_preds.append(median_pred)

        y_flat = np.concatenate(all_preds, axis=0)
        return y_flat.reshape(B, C, pred_len).transpose(0, 2, 1)


# =====================================================================
# 4. Multi-Dataset Evaluation Orchestrator
# =====================================================================
def run_dataset_evaluation(
    dataset_name: str,
    pred_len: int = 96,
    num_eval_windows: int = 20,
    stride: int = 24,
    chronos_model: str = "amazon/chronos-t5-tiny",
    chronos_raw: ChronosRawForecaster = None,
    fcmath: FCMath = None,
    fcmath_tuned: FCMathTuned = None,
) -> List[Dict]:
    print(f"\n" + "=" * 80)
    print(f"BENCHMARKING DATASET: {dataset_name} (H={pred_len}, Eval Windows={num_eval_windows})")
    print("=" * 80)

    train_norm, val_norm, test_norm, channels = load_dataset_standardized(dataset_name, seq_len=336, pred_len=pred_len)
    x_eval, y_eval = extract_evaluation_windows(test_norm, seq_len=336, pred_len=pred_len,
                                                num_eval_windows=num_eval_windows, stride=stride)

    print(f"Loaded {channels} channels. Evaluation sample shape: x={x_eval.shape}, y={y_eval.shape}")

    dataset_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Chronos-Raw
    print("  -> Running Chronos-Raw ...", end=" ", flush=True)
    t0 = time.time()
    y_raw = chronos_raw.forecast_batch(x_eval, pred_len=pred_len, batch_size=128)
    dt_raw = time.time() - t0
    mse_raw = np.mean((y_raw - y_eval)**2)
    mae_raw = np.mean(np.abs(y_raw - y_eval))
    spec_raw = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_raw))
    dataset_results.append({
        'Dataset': dataset_name,
        'Channels': channels,
        'Model': "Chronos-Raw",
        'MSE': round(float(mse_raw), 4),
        'MAE': round(float(mae_raw), 4),
        'Mid_FVU': round(float(spec_raw['fvu_mid']), 4),
        'LSD_dB': round(float(spec_raw['lsd_db']), 2),
        'Time_s': round(dt_raw, 2)
    })
    print(f"Done in {dt_raw:.1f}s | MSE: {mse_raw:.4f} | Mid-FVU: {spec_raw['fvu_mid']:.4f}")

    # 2. FCMath (Standard)
    print("  -> Running FCMath (Standard) ...", end=" ", flush=True)
    t0 = time.time()
    y_fc, _ = fcmath.forecast(x_eval, prediction_length=pred_len, batch_size=128, num_samples=15)
    dt_fc = time.time() - t0
    mse_fc = np.mean((y_fc - y_eval)**2)
    mae_fc = np.mean(np.abs(y_fc - y_eval))
    spec_fc = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_fc))
    dataset_results.append({
        'Dataset': dataset_name,
        'Channels': channels,
        'Model': "FCMath (Standard)",
        'MSE': round(float(mse_fc), 4),
        'MAE': round(float(mae_fc), 4),
        'Mid_FVU': round(float(spec_fc['fvu_mid']), 4),
        'LSD_dB': round(float(spec_fc['lsd_db']), 2),
        'Time_s': round(dt_fc, 2)
    })
    print(f"Done in {dt_fc:.1f}s | MSE: {mse_fc:.4f} | Mid-FVU: {spec_fc['fvu_mid']:.4f}")

    # 3. FCMath-Tuned
    print("  -> Running FCMath-Tuned ...", end=" ", flush=True)
    t0 = time.time()
    y_tuned = fcmath_tuned.forecast_batch(x_eval, pred_len=pred_len, batch_size=128)
    dt_tuned = time.time() - t0
    mse_tuned = np.mean((y_tuned - y_eval)**2)
    mae_tuned = np.mean(np.abs(y_tuned - y_eval))
    spec_tuned = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_tuned))
    dataset_results.append({
        'Dataset': dataset_name,
        'Channels': channels,
        'Model': "FCMath-Tuned",
        'MSE': round(float(mse_tuned), 4),
        'MAE': round(float(mae_tuned), 4),
        'Mid_FVU': round(float(spec_tuned['fvu_mid']), 4),
        'LSD_dB': round(float(spec_tuned['lsd_db']), 2),
        'Time_s': round(dt_tuned, 2)
    })
    print(f"Done in {dt_tuned:.1f}s | MSE: {mse_tuned:.4f} | Mid-FVU: {spec_tuned['fvu_mid']:.4f}")

    # 4. CycleNet (W=24)
    print("  -> Running CycleNet (W=24) ...", end=" ", flush=True)
    t0 = time.time()
    cyclenet = CycleNet(seq_len=336, pred_len=pred_len, channels=channels, cycle_len=24).to(device)
    # Quick optimizer warmup for CycleNet
    optimizer = torch.optim.AdamW(cyclenet.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.MSELoss()
    cyclenet.train()
    # Train on 20 batches of train_norm
    batch_sz = 32
    max_train_steps = min(50, (len(train_norm) - 336 - pred_len) // batch_sz)
    for step in range(max_train_steps):
        idx = step * batch_sz
        xb = torch.tensor(np.array([train_norm[idx + b : idx + b + 336] for b in range(batch_sz)]), device=device)
        yb = torch.tensor(np.array([train_norm[idx + b + 336 : idx + b + 336 + pred_len] for b in range(batch_sz)]), device=device)
        optimizer.zero_grad()
        loss = criterion(cyclenet(xb), yb)
        loss.backward()
        optimizer.step()

    cyclenet.eval()
    with torch.no_grad():
        y_cyc = cyclenet(torch.tensor(x_eval, device=device)).cpu().numpy()
    dt_cyc = time.time() - t0
    mse_cyc = np.mean((y_cyc - y_eval)**2)
    mae_cyc = np.mean(np.abs(y_cyc - y_eval))
    spec_cyc = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_cyc))
    dataset_results.append({
        'Dataset': dataset_name,
        'Channels': channels,
        'Model': "CycleNet (W=24)",
        'MSE': round(float(mse_cyc), 4),
        'MAE': round(float(mae_cyc), 4),
        'Mid_FVU': round(float(spec_cyc['fvu_mid']), 4),
        'LSD_dB': round(float(spec_cyc['lsd_db']), 2),
        'Time_s': round(dt_cyc, 2)
    })
    print(f"Done in {dt_cyc:.1f}s | MSE: {mse_cyc:.4f} | Mid-FVU: {spec_cyc['fvu_mid']:.4f}")

    return dataset_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["ETTh1.csv", "ETTh2.csv", "weather.csv", "electricity.csv", "traffic.csv", "ETTm1.csv"],
                        help="Datasets to evaluate")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--num_eval_windows", type=int, default=20)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--chronos_model", type=str, default="amazon/chronos-t5-tiny")
    args = parser.parse_args()

    set_seed(2024)
    print("\n" + "=" * 90)
    print("SRP26 MASTER BENCHMARK SUITE: CHRONOS vs. FCMATH vs. CYCLENET ACROSS DATASETS")
    print(f"Evaluation Horizon: H={args.pred_len} | Windows per Dataset: {args.num_eval_windows}")
    print(f"Datasets: {args.datasets}")
    print("=" * 90)

    # Initialize shared models once on GPU
    chronos_raw = ChronosRawForecaster(model_name=args.chronos_model)
    fcmath = FCMath(top_k=3, model_name=args.chronos_model)
    fcmath_tuned = FCMathTuned(top_k=3, model_name=args.chronos_model)

    all_results = []
    for ds in args.datasets:
        try:
            ds_res = run_dataset_evaluation(
                dataset_name=ds,
                pred_len=args.pred_len,
                num_eval_windows=args.num_eval_windows,
                stride=args.stride,
                chronos_model=args.chronos_model,
                chronos_raw=chronos_raw,
                fcmath=fcmath,
                fcmath_tuned=fcmath_tuned,
            )
            all_results.extend(ds_res)
        except Exception as e:
            print(f"Error evaluating {ds}: {e}")

    # Master Table Output
    master_df = pd.DataFrame(all_results)
    print("\n" + "=" * 95)
    print("MASTER BENCHMARK COMPARISON TABLE (ALL DATASETS)")
    print("=" * 95)
    print(master_df.to_string(index=False))
    print("=" * 95 + "\n")

    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"chronos_vs_fcmath_all_datasets_H{args.pred_len}.json")
    md_path = os.path.join(out_dir, f"chronos_vs_fcmath_all_datasets_H{args.pred_len}.md")

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    with open(md_path, 'w') as f:
        f.write(f"# Master Benchmark: Chronos Base vs. FCMath vs. CycleNet across Datasets (H={args.pred_len})\n\n")
        f.write(master_df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved master reports to:\n  - {json_path}\n  - {md_path}\n")


if __name__ == "__main__":
    main()

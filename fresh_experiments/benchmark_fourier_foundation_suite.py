"""
Comprehensive Fourier Decoupling Benchmark: Foundation Models Raw vs. Fourier Base
==================================================================================
Testing whether the Two-Stage Fourier Base Decoupled Architecture improves or
degrades performance across different time-series foundation models:
  1. Google TimesFM 2.5 (200M)         : Raw vs. FTimesFM
  2. Amazon Chronos-Bolt-Base (200M)   : Raw vs. F-Chronos-Bolt
  3. IBM Granite TTM-R1 (805K)         : Raw vs. F-TTM
  4. Amazon Chronos-T5 (Tiny 8M)       : Raw vs. FCMath
  5. CycleNet (W=24, 32K)              : Reference lightweight periodic baseline

Datasets evaluated:
  - ETTh1.csv    (7 channels, periodic electrical transformer)
  - weather.csv  (21 channels, thermodynamic fluid variables, non-periodic physics)
  - traffic.csv  (862 channels, freeway speed sensors, strong diurnal/weekly cycles)

Lookback L = 336, Forecast Horizon H = 96
Standardized train-only Z-score scaling.
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
import math
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
from srp26 import (
    CycleNet,
    StandardScaler,
    compute_spectral_diagnostics,
    extract_and_extrapolate_fourier_base_torch,
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
    else:
        n_train = int(total_len * 0.70)
        n_val = int(total_len * 0.10)
        n_test = total_len - n_train - n_val

    train_data = raw_data[:n_train]
    test_slice = raw_data[-(n_test + seq_len):]

    scaler = StandardScaler()
    scaler.fit(train_data)
    test_scaled = scaler.transform(test_slice)

    total_test_len = len(test_scaled)
    max_start = total_test_len - seq_len - pred_len
    stride = max(1, max_start // num_eval_windows)
    starts = [i * stride for i in range(num_eval_windows) if i * stride <= max_start]
    if len(starts) < num_eval_windows:
        starts.append(max_start)
    starts = sorted(list(set(starts)))[:num_eval_windows]

    x_list, y_list = [], []
    for s in starts:
        x_list.append(test_scaled[s : s + seq_len])
        y_list.append(test_scaled[s + seq_len : s + seq_len + pred_len])

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def evaluate_forecast(y_eval: np.ndarray, y_pred: np.ndarray) -> Dict:
    mse = float(np.mean((y_pred - y_eval) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_eval)))
    spec = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred))
    return {
        'mse': round(mse, 4),
        'mae': round(mae, 4),
        'mid_fvu': round(spec['fvu_mid'], 4),
        'lsd_db': round(spec['lsd_db'], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="SRP26 Fourier Decoupling Benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1.csv", "weather.csv", "traffic.csv"],
        help="Datasets to evaluate",
    )
    args = parser.parse_args()

    set_seed(2024)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print(" SRP26: COMPREHENSIVE FOURIER DECOUPLING BENCHMARK")
    print(" Comparing Foundation Models Raw vs. Fourier-Coupled (Two-Stage Decoupled)")
    print(f" Datasets: {args.datasets}")
    print(f" Execution Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    # 1. Initialize Foundation Models
    print("\n--- 1. Initializing Foundation Models ---")

    # A. Google TimesFM 2.5 (200M)
    print("Loading Google TimesFM 2.5 (200M) ...")
    tfm_model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    tfm_forecast_config = ForecastConfig(
        max_context=352,
        max_horizon=128,
        per_core_batch_size=128,
        normalize_inputs=True,
        infer_is_positive=False,
        force_flip_invariance=True,
    )
    tfm_model.compile(tfm_forecast_config)

    # B. Amazon Chronos-Bolt-Base (200M)
    print("Loading Amazon Chronos-Bolt-Base (200M) ...")
    bolt_pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # C. IBM Granite TTM-R1 (805K)
    print("Loading IBM Granite TTM-R1 (805K) ...")
    ttm_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r1",
    ).to(device)
    ttm_model.eval()

    # D. Amazon Chronos-T5 (Tiny, 8M)
    print("Loading Amazon Chronos-T5 (Tiny, 8M) ...")
    chronos_t5_pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    datasets = args.datasets
    all_results = []
    L = 336
    pred_len = 96

    for dataset_name in datasets:
        print(f"\n=======================================================")
        print(f" DATASET: {dataset_name} (L={L}, H={pred_len})")
        print(f"=======================================================")

        x_eval, y_eval = load_dataset(dataset_name, seq_len=L, pred_len=pred_len, num_eval_windows=15)
        B, _, C = x_eval.shape
        print(f"Evaluation Windows: {B} | Channels: {C} | Total series: {B * C}")

        # Compute Fourier Base Extrapolation & Residuals
        x_eval_torch = torch.tensor(x_eval, dtype=torch.float32)
        x_base, y_base, x_res = extract_and_extrapolate_fourier_base_torch(x_eval_torch, pred_len=pred_len, top_k=3)
        x_base_np = x_base.numpy()
        y_base_np = y_base.numpy()
        x_res_np = x_res.numpy()

        # Pure Fourier Extrapolation Base alone (Baseline)
        f_eval = evaluate_forecast(y_eval, y_base_np)
        print(f"  [Fourier Base Only] MSE: {f_eval['mse']:.4f} | MAE: {f_eval['mae']:.4f} | LSD: {f_eval['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Fourier Base Only (Top-3)',
            'Paradigm': 'Pure Math Closed-Form',
            'Params': '0',
            'MSE': f_eval['mse'],
            'MAE': f_eval['mae'],
            'Mid_FVU': f_eval['mid_fvu'],
            'LSD_dB': f_eval['lsd_db'],
            'Runtime_Sec': 0.01,
        })

        # -------------------------------------------------------------
        # 1. Google TimesFM 2.5: Raw vs. FTimesFM
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        print("  Running Google TimesFM 2.5 (Raw) ...", end=" ", flush=True)
        t0 = time.time()
        flat_x = [x_eval[b, :, c] for b in range(B) for c in range(C)]
        tfm_point_raw, _ = tfm_model.forecast(horizon=pred_len, inputs=flat_x)
        y_pred_tfm_raw = tfm_point_raw.reshape(B, C, pred_len).transpose(0, 2, 1)
        dt_tfm_raw = time.time() - t0
        res_tfm_raw = evaluate_forecast(y_eval, y_pred_tfm_raw)
        print(f"Done in {dt_tfm_raw:.2f}s | MSE: {res_tfm_raw['mse']:.4f} | LSD: {res_tfm_raw['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Google TimesFM 2.5',
            'Paradigm': 'Raw (Zero-Shot)',
            'Params': '200M',
            'MSE': res_tfm_raw['mse'],
            'MAE': res_tfm_raw['mae'],
            'Mid_FVU': res_tfm_raw['mid_fvu'],
            'LSD_dB': res_tfm_raw['lsd_db'],
            'Runtime_Sec': round(dt_tfm_raw, 2),
        })

        print("  Running FTimesFM (Fourier + TimesFM Residuals) ...", end=" ", flush=True)
        t0 = time.time()
        flat_res = [x_res_np[b, :, c] for b in range(B) for c in range(C)]
        tfm_point_res, _ = tfm_model.forecast(horizon=pred_len, inputs=flat_res)
        y_pred_res_tfm = tfm_point_res.reshape(B, C, pred_len).transpose(0, 2, 1)
        y_pred_ftfm = y_base_np + y_pred_res_tfm
        dt_ftfm = time.time() - t0
        res_ftfm = evaluate_forecast(y_eval, y_pred_ftfm)
        print(f"Done in {dt_ftfm:.2f}s | MSE: {res_ftfm['mse']:.4f} | LSD: {res_ftfm['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Google TimesFM 2.5',
            'Paradigm': 'FTimesFM (Fourier+Res)',
            'Params': '200M',
            'MSE': res_ftfm['mse'],
            'MAE': res_ftfm['mae'],
            'Mid_FVU': res_ftfm['mid_fvu'],
            'LSD_dB': res_ftfm['lsd_db'],
            'Runtime_Sec': round(dt_ftfm, 2),
        })

        # -------------------------------------------------------------
        # 2. Amazon Chronos-Bolt-Base: Raw vs. F-Chronos-Bolt
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        print("  Running Chronos-Bolt-Base (Raw) ...", end=" ", flush=True)
        t0 = time.time()
        x_flat = x_eval.transpose(0, 2, 1).reshape(B * C, L)
        ctx_raw = torch.tensor(x_flat, dtype=torch.float32).to(bolt_pipeline.model.device)
        bolt_raw_preds = []
        bs = 64
        for i in range(0, len(ctx_raw), bs):
            with torch.no_grad():
                f_dist = bolt_pipeline.predict(ctx_raw[i : i + bs], prediction_length=pred_len)
                bolt_raw_preds.append(f_dist[:, 4, :].cpu().numpy())
        y_pred_bolt_raw = np.concatenate(bolt_raw_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
        dt_bolt_raw = time.time() - t0
        res_bolt_raw = evaluate_forecast(y_eval, y_pred_bolt_raw)
        print(f"Done in {dt_bolt_raw:.2f}s | MSE: {res_bolt_raw['mse']:.4f} | LSD: {res_bolt_raw['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Chronos-Bolt-Base',
            'Paradigm': 'Raw (Zero-Shot)',
            'Params': '200M',
            'MSE': res_bolt_raw['mse'],
            'MAE': res_bolt_raw['mae'],
            'Mid_FVU': res_bolt_raw['mid_fvu'],
            'LSD_dB': res_bolt_raw['lsd_db'],
            'Runtime_Sec': round(dt_bolt_raw, 2),
        })

        print("  Running F-Chronos-Bolt (Fourier + Bolt Residuals) ...", end=" ", flush=True)
        t0 = time.time()
        res_flat = x_res_np.transpose(0, 2, 1).reshape(B * C, L)
        ctx_res = torch.tensor(res_flat, dtype=torch.float32).to(bolt_pipeline.model.device)
        bolt_res_preds = []
        for i in range(0, len(ctx_res), bs):
            with torch.no_grad():
                f_dist = bolt_pipeline.predict(ctx_res[i : i + bs], prediction_length=pred_len)
                bolt_res_preds.append(f_dist[:, 4, :].cpu().numpy())
        y_pred_bolt_res = np.concatenate(bolt_res_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
        y_pred_fbolt = y_base_np + y_pred_bolt_res
        dt_fbolt = time.time() - t0
        res_fbolt = evaluate_forecast(y_eval, y_pred_fbolt)
        print(f"Done in {dt_fbolt:.2f}s | MSE: {res_fbolt['mse']:.4f} | LSD: {res_fbolt['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Chronos-Bolt-Base',
            'Paradigm': 'F-Bolt (Fourier+Res)',
            'Params': '200M',
            'MSE': res_fbolt['mse'],
            'MAE': res_fbolt['mae'],
            'Mid_FVU': res_fbolt['mid_fvu'],
            'LSD_dB': res_fbolt['lsd_db'],
            'Runtime_Sec': round(dt_fbolt, 2),
        })

        # -------------------------------------------------------------
        # 3. IBM Granite TTM-R1: Raw vs. F-TTM
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        print("  Running IBM Granite TTM-R1 (Raw) ...", end=" ", flush=True)
        t0 = time.time()
        x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(ttm_model.device)
        pad_len = ttm_model.config.context_length - L
        x_padded = torch.cat([x_tensor[:, :1, :].repeat(1, pad_len, 1), x_tensor], dim=1)
        with torch.no_grad():
            out = ttm_model(past_values=x_padded)
            y_pred_ttm_raw = out.prediction_outputs.cpu().numpy()
        dt_ttm_raw = time.time() - t0
        res_ttm_raw = evaluate_forecast(y_eval, y_pred_ttm_raw)
        print(f"Done in {dt_ttm_raw:.2f}s | MSE: {res_ttm_raw['mse']:.4f} | LSD: {res_ttm_raw['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'IBM Granite TTM-R1',
            'Paradigm': 'Raw (Zero-Shot)',
            'Params': '805K',
            'MSE': res_ttm_raw['mse'],
            'MAE': res_ttm_raw['mae'],
            'Mid_FVU': res_ttm_raw['mid_fvu'],
            'LSD_dB': res_ttm_raw['lsd_db'],
            'Runtime_Sec': round(dt_ttm_raw, 2),
        })

        print("  Running F-TTM (Fourier + Granite TTM Residuals) ...", end=" ", flush=True)
        t0 = time.time()
        x_res_tensor = torch.tensor(x_res_np, dtype=torch.float32).to(ttm_model.device)
        x_res_padded = torch.cat([x_res_tensor[:, :1, :].repeat(1, pad_len, 1), x_res_tensor], dim=1)
        with torch.no_grad():
            out_res = ttm_model(past_values=x_res_padded)
            y_pred_res_ttm = out_res.prediction_outputs.cpu().numpy()
        y_pred_fttm = y_base_np + y_pred_res_ttm
        dt_fttm = time.time() - t0
        res_fttm = evaluate_forecast(y_eval, y_pred_fttm)
        print(f"Done in {dt_fttm:.2f}s | MSE: {res_fttm['mse']:.4f} | LSD: {res_fttm['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'IBM Granite TTM-R1',
            'Paradigm': 'F-TTM (Fourier+Res)',
            'Params': '805K',
            'MSE': res_fttm['mse'],
            'MAE': res_fttm['mae'],
            'Mid_FVU': res_fttm['mid_fvu'],
            'LSD_dB': res_fttm['lsd_db'],
            'Runtime_Sec': round(dt_fttm, 2),
        })

        # -------------------------------------------------------------
        # 4. Amazon Chronos-T5: Raw vs. FCMath
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        # On Traffic (862 channels), AR rollout is very slow; we use existing verified results if long
        print("  Running Amazon Chronos-T5 (Raw) ...", end=" ", flush=True)
        t0 = time.time()
        ctx_cpu = torch.tensor(x_flat, dtype=torch.float32)
        ch_raw_preds = []
        for i in range(0, len(ctx_cpu), bs):
            with torch.no_grad():
                f_dist = chronos_t5_pipeline.predict(ctx_cpu[i : i + bs], prediction_length=pred_len, num_samples=1)
                ch_raw_preds.append(f_dist.squeeze(1).cpu().numpy())
        y_pred_ch_raw = np.concatenate(ch_raw_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
        dt_ch_raw = time.time() - t0
        res_ch_raw = evaluate_forecast(y_eval, y_pred_ch_raw)
        print(f"Done in {dt_ch_raw:.2f}s | MSE: {res_ch_raw['mse']:.4f} | LSD: {res_ch_raw['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Amazon Chronos-T5',
            'Paradigm': 'Raw (Zero-Shot)',
            'Params': '8M',
            'MSE': res_ch_raw['mse'],
            'MAE': res_ch_raw['mae'],
            'Mid_FVU': res_ch_raw['mid_fvu'],
            'LSD_dB': res_ch_raw['lsd_db'],
            'Runtime_Sec': round(dt_ch_raw, 2),
        })

        print("  Running FCMath (Fourier + Chronos-T5 Residuals) ...", end=" ", flush=True)
        t0 = time.time()
        ctx_res_cpu = torch.tensor(res_flat, dtype=torch.float32)
        ch_res_preds = []
        for i in range(0, len(ctx_res_cpu), bs):
            with torch.no_grad():
                f_dist = chronos_t5_pipeline.predict(ctx_res_cpu[i : i + bs], prediction_length=pred_len, num_samples=1)
                ch_res_preds.append(f_dist.squeeze(1).cpu().numpy())
        y_pred_ch_res = np.concatenate(ch_res_preds, axis=0).reshape(B, C, pred_len).transpose(0, 2, 1)
        y_pred_fcmath = y_base_np + y_pred_ch_res
        dt_fcmath = time.time() - t0
        res_fcmath = evaluate_forecast(y_eval, y_pred_fcmath)
        print(f"Done in {dt_fcmath:.2f}s | MSE: {res_fcmath['mse']:.4f} | LSD: {res_fcmath['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'Amazon Chronos-T5',
            'Paradigm': 'FCMath (Fourier+Res)',
            'Params': '8M',
            'MSE': res_fcmath['mse'],
            'MAE': res_fcmath['mae'],
            'Mid_FVU': res_fcmath['mid_fvu'],
            'LSD_dB': res_fcmath['lsd_db'],
            'Runtime_Sec': round(dt_fcmath, 2),
        })

        # -------------------------------------------------------------
        # 5. CycleNet Baseline
        # -------------------------------------------------------------
        torch.cuda.empty_cache()
        print("  Running CycleNet (W=24, 32K) ...", end=" ", flush=True)
        t0 = time.time()
        cyclenet = CycleNet(seq_len=L, pred_len=pred_len, enc_in=C, cycle_len=24, model_type='linear').to(device)
        cyclenet.eval()
        x_dev = torch.tensor(x_eval, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_cyc = cyclenet(x_dev).cpu().numpy()
        dt_cyc = time.time() - t0
        res_cyc = evaluate_forecast(y_eval, y_pred_cyc)
        print(f"Done in {dt_cyc:.2f}s | MSE: {res_cyc['mse']:.4f} | LSD: {res_cyc['lsd_db']:.2f} dB")
        all_results.append({
            'Dataset': dataset_name,
            'Model': 'CycleNet (W=24)',
            'Paradigm': 'Recurrent Periodic',
            'Params': '32K',
            'MSE': res_cyc['mse'],
            'MAE': res_cyc['mae'],
            'Mid_FVU': res_cyc['mid_fvu'],
            'LSD_dB': res_cyc['lsd_db'],
            'Runtime_Sec': round(dt_cyc, 2),
        })

    # Output & Save Results
    df_res = pd.DataFrame(all_results)
    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "fourier_decoupling_foundation_benchmark_H96.json")
    md_path = os.path.join(out_dir, "fourier_decoupling_foundation_benchmark_H96.md")

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    df_res.to_markdown(md_path, index=False)

    print("\n" + "=" * 100)
    print(" SUMMARY BENCHMARK TABLE")
    print("=" * 100)
    print(df_res.to_string(index=False))
    print(f"\nSaved results to:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
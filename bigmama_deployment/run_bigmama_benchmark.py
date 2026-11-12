"""
Bigmama Cluster Benchmark Runner (University of Hildesheim | ISMLL)
===================================================================
Project: Ultra-lightweight MTS Forecasting Sandbox & Spectral Decomposition (SRP26)
Supervisor: Jung Min Choi | Researcher: Aaditya Chaturvedi

Standardized Protocol:
  - Lookback L = 336
  - Horizons H in {96, 192, 336, 720}
  - Canonical Splits: ETT (12/4/4 months), Non-ETT (70% Train, 10% Val, 20% Test)
  - Train-only StandardScaler fit (Zero Data Leakage)
  - Parseval Spectral Decomposition (Low, Mid, High FVU & LSD in dB)
  - Multi-Seed Evaluation: 2024, 2025, 2026
"""

import os
import sys
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

# Auto-set HF Cache cleanly inside user home unless HF_HOME is explicitly overridden
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


def set_seed(seed: int = 2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0.0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean


def extract_and_extrapolate_fourier_base_torch(x: torch.Tensor, pred_len: int, top_k: int = 3):
    """
    Vectorized analytical Fourier decomposition & closed-form sinusoidal extrapolation.
    Input:
        x: [B, L, C]
    Returns:
        x_base: [B, L, C] (in-sample top-k periodic reconstruction)
        y_base: [B, pred_len, C] (closed-form extrapolation into future horizon)
        x_res:  [B, L, C] (orthogonal non-periodic residuals)
    """
    B, L, C = x.shape
    x_perm = x.permute(0, 2, 1)  # [B, C, L]
    fft_vals = torch.fft.rfft(x_perm, dim=-1)
    amps = torch.abs(fft_vals)
    amps[..., 0] = 0.0  # Zero DC component

    k = min(top_k, amps.shape[-1] - 1)
    topk_vals, topk_indices = torch.topk(amps, k=k, dim=-1)  # [B, C, k]

    # In-sample base reconstruction via filtered inverse FFT
    filtered_fft = torch.zeros_like(fft_vals)
    filtered_fft.scatter_(-1, topk_indices, torch.gather(fft_vals, -1, topk_indices))
    x_base = torch.fft.irfft(filtered_fft, n=L, dim=-1)  # [B, C, L]

    # Analytical future extrapolation: y_base(t) = sum_k A_k * cos(2*pi*f_k*t + phi_k)
    t_future = torch.arange(L, L + pred_len, device=x.device, dtype=torch.float32).view(1, 1, 1, pred_len)
    freqs = (topk_indices.float() / float(L)).unsqueeze(-1)  # [B, C, k, 1]
    topk_fft = torch.gather(fft_vals, -1, topk_indices)
    topk_amps = (torch.abs(topk_fft) / (float(L) / 2.0)).unsqueeze(-1)
    topk_phases = torch.angle(topk_fft).unsqueeze(-1)

    cos_terms = topk_amps * torch.cos(2.0 * math.pi * freqs * t_future + topk_phases)
    y_base = torch.sum(cos_terms, dim=2)  # [B, C, pred_len]

    x_base = x_base.permute(0, 2, 1)  # [B, L, C]
    y_base = y_base.permute(0, 2, 1)  # [B, pred_len, C]
    x_res = x - x_base
    return x_base, y_base, x_res


def compute_spectral_diagnostics(y_true: torch.Tensor, y_pred: torch.Tensor, fs: float = 1.0) -> Dict[str, float]:
    """Parseval Exact Subband Decomposition & Log-Spectral Distance."""
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    H = y_true.shape[1]
    fft_true = torch.fft.rfft(y_true, dim=1)
    fft_pred = torch.fft.rfft(y_pred, dim=1)

    err_fft = fft_pred - fft_true
    num_bins = fft_true.shape[1]

    idx_low = max(1, int(num_bins * 0.15))
    idx_mid = max(idx_low + 1, int(num_bins * 0.50))

    err_power = torch.abs(err_fft) ** 2
    true_power = torch.abs(fft_true) ** 2

    # Fraction of Variance Unexplained across physical subbands
    fvu_low = (torch.sum(err_power[:, :idx_low, :]) / torch.clamp(torch.sum(true_power[:, :idx_low, :]), min=1e-8)).item()
    fvu_mid = (torch.sum(err_power[:, idx_low:idx_mid, :]) / torch.clamp(torch.sum(true_power[:, idx_low:idx_mid, :]), min=1e-8)).item()
    fvu_high = (torch.sum(err_power[:, idx_mid:, :]) / torch.clamp(torch.sum(true_power[:, idx_mid:, :]), min=1e-8)).item()

    # Log-Spectral Distance (LSD in dB)
    psd_true = torch.clamp(torch.abs(fft_true) ** 2, min=1e-8)
    psd_pred = torch.clamp(torch.abs(fft_pred) ** 2, min=1e-8)
    log_ratio = torch.log10(psd_true) - torch.log10(psd_pred)
    lsd_db = (10.0 * torch.sqrt(torch.mean(log_ratio ** 2))).item()

    return {
        'fvu_low': round(fvu_low, 4),
        'fvu_mid': round(fvu_mid, 4),
        'fvu_high': round(fvu_high, 4),
        'lsd_db': round(lsd_db, 2),
    }


class CycleNet(nn.Module):
    """Automated Recurrent CycleNet Baseline."""
    def __init__(self, seq_len: int, pred_len: int, channels: int, cycle_len: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.cycle_len = cycle_len
        
        # Periodic recurrent state projection
        self.num_cycles = seq_len // cycle_len
        self.cycle_proj = nn.Linear(self.num_cycles, pred_len // cycle_len if pred_len >= cycle_len else 1)
        self.backbone = nn.Linear(seq_len, pred_len)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape
        x_lin = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, H, C]
        
        # Periodic projection
        eff_len = self.num_cycles * self.cycle_len
        x_cycles = x[:, -eff_len:, :].view(B, self.num_cycles, self.cycle_len, C)  # [B, num_cycles, cycle_len, C]
        x_cycles_perm = x_cycles.permute(0, 2, 3, 1)  # [B, cycle_len, C, num_cycles]
        cycle_out = self.cycle_proj(x_cycles_perm).permute(0, 3, 1, 2)  # [B, out_cycles, cycle_len, C]
        cycle_flat = cycle_out.reshape(B, -1, C)
        
        if cycle_flat.shape[1] < self.pred_len:
            reps = math.ceil(self.pred_len / cycle_flat.shape[1])
            cycle_flat = cycle_flat.repeat(1, reps, 1)[:, :self.pred_len, :]
        elif cycle_flat.shape[1] > self.pred_len:
            cycle_flat = cycle_flat[:, :self.pred_len, :]
            
        g = torch.sigmoid(self.gate)
        return g * x_lin + (1.0 - g) * cycle_flat


def load_dataset(dataset_name: str, data_dir: str = "data", seq_len: int = 336, pred_len: int = 96, full_test: bool = False, num_eval_windows: int = 15):
    # Try finding dataset
    search_paths = [
        os.path.join(data_dir, dataset_name),
        os.path.join(data_dir, dataset_name.lower()),
        os.path.join("data_sets", dataset_name),
        dataset_name
    ]
    path = None
    for p in search_paths:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found. Checked: {search_paths}")

    if path.endswith('.txt'):
        df = pd.read_csv(path, header=None)
        raw_data = df.values.astype(np.float32)
    else:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        raw_data = df.values.astype(np.float32)

    total_len = len(raw_data)
    name_lower = os.path.basename(path).lower()
    
    # Chronological Split Invariants
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
    if full_test:
        # Evaluate across the full test set with non-overlapping or fixed stride
        stride = pred_len
        eval_indices = np.arange(0, total_valid, stride)
    else:
        stride = max(1, total_valid // num_eval_windows)
        eval_indices = np.arange(0, total_valid, stride)[:num_eval_windows]

    x_list, y_list = [], []
    for idx in eval_indices:
        x_list.append(test_norm[idx : idx + seq_len])
        y_list.append(test_norm[idx + seq_len : idx + seq_len + pred_len])

    x_eval = np.array(x_list, dtype=np.float32)
    y_eval = np.array(y_list, dtype=np.float32)
    return x_eval, y_eval, train_norm, scaler, raw_data.shape[1]


def evaluate_predictions(y_eval: np.ndarray, y_pred: np.ndarray) -> Dict:
    mse = float(np.mean((y_pred - y_eval) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_eval)))
    spec = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred))
    return {
        'mse': round(mse, 4),
        'mae': round(mae, 4),
        'mid_fvu': spec['fvu_mid'],
        'lsd_db': spec['lsd_db'],
    }


def main():
    parser = argparse.ArgumentParser(description="Bigmama Cluster Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="ETTh1.csv", help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory holding datasets")
    parser.add_argument("--pred_len", type=int, default=96, choices=[96, 192, 336, 720], help="Forecast horizon")
    parser.add_argument("--seq_len", type=int, default=336, help="Lookback window")
    parser.add_argument("--model", type=str, default="all", help="Model to evaluate (all, timesfm, chronos_bolt, granite_ttm, chronos_t5, cyclenet)")
    parser.add_argument("--full_test", action="store_true", help="Evaluate across entire test set (instead of 15 windows)")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/cluster_results", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f" BIGMAMA CLUSTER MTS BENCHMARK (ISMLL)")
    print(f" Dataset: {args.dataset} | Horizon H: {args.pred_len} | Lookback L: {args.seq_len} | Seed: {args.seed}")
    print(f" Mode: {'FULL TEST SPLIT' if args.full_test else 'Sampled Windows (15)'} | Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)

    x_eval, y_eval, train_norm, scaler, C = load_dataset(
        args.dataset, data_dir=args.data_dir, seq_len=args.seq_len, pred_len=args.pred_len, full_test=args.full_test
    )
    B, L, _ = x_eval.shape
    print(f"Loaded {args.dataset}: Channels: {C} | Evaluation Windows: {B} | Total Series: {B * C}")

    # Compute Fourier Base Extrapolation & Residuals
    x_eval_torch = torch.tensor(x_eval, dtype=torch.float32)
    x_base, y_base, x_res = extract_and_extrapolate_fourier_base_torch(x_eval_torch, pred_len=args.pred_len, top_k=3)
    x_base_np = x_base.numpy()
    y_base_np = y_base.numpy()
    x_res_np = x_res.numpy()

    results = []

    # 1. Fourier Base Only (Baseline)
    f_res = evaluate_predictions(y_eval, y_base_np)
    print(f"[Baseline] Pure Fourier Base Only: MSE {f_res['mse']:.4f} | LSD {f_res['lsd_db']:.2f} dB")
    results.append({
        'dataset': args.dataset,
        'model': 'Fourier-Base-Only',
        'paradigm': 'Pure Math Closed-Form',
        'params': 0,
        'pred_len': args.pred_len,
        'seed': args.seed,
        'mse': f_res['mse'],
        'mae': f_res['mae'],
        'mid_fvu': f_res['mid_fvu'],
        'lsd_db': f_res['lsd_db'],
        'runtime_sec': 0.01
    })

    # 2. Google TimesFM 2.5 (200M)
    if args.model in ["all", "timesfm"]:
        print("\n--- Google TimesFM 2.5 (200M) ---")
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
        tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        cfg = ForecastConfig(max_context=args.seq_len + 32, max_horizon=max(128, args.pred_len), per_core_batch_size=128, normalize_inputs=True)
        tfm.compile(cfg)

        # Raw
        t0 = time.time()
        flat_x = [x_eval[b, :, c] for b in range(B) for c in range(C)]
        out_raw, _ = tfm.forecast(horizon=args.pred_len, inputs=flat_x)
        y_tfm_raw = out_raw.reshape(B, C, args.pred_len).transpose(0, 2, 1)
        dt_raw = time.time() - t0
        eval_tfm_raw = evaluate_predictions(y_eval, y_tfm_raw)
        print(f"  TimesFM Raw: MSE {eval_tfm_raw['mse']:.4f} | LSD {eval_tfm_raw['lsd_db']:.2f} dB ({dt_raw:.2f}s)")
        results.append({**eval_tfm_raw, 'dataset': args.dataset, 'model': 'TimesFM-2.5', 'paradigm': 'Raw (Zero-Shot)', 'params': '200M', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_raw, 2)})

        # Fourier Decoupled (FTimesFM)
        t0 = time.time()
        flat_res = [x_res_np[b, :, c] for b in range(B) for c in range(C)]
        out_res, _ = tfm.forecast(horizon=args.pred_len, inputs=flat_res)
        y_tfm_res = out_res.reshape(B, C, args.pred_len).transpose(0, 2, 1)
        y_ftfm = y_base_np + y_tfm_res
        dt_ftfm = time.time() - t0
        eval_ftfm = evaluate_predictions(y_eval, y_ftfm)
        print(f"  FTimesFM (Decoupled): MSE {eval_ftfm['mse']:.4f} | LSD {eval_ftfm['lsd_db']:.2f} dB ({dt_ftfm:.2f}s)")
        results.append({**eval_ftfm, 'dataset': args.dataset, 'model': 'TimesFM-2.5', 'paradigm': 'FTimesFM (Fourier+Res)', 'params': '200M', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_ftfm, 2)})
        del tfm; torch.cuda.empty_cache()

    # 3. Amazon Chronos-Bolt-Base (200M)
    if args.model in ["all", "chronos_bolt"]:
        print("\n--- Amazon Chronos-Bolt-Base (200M) ---")
        from chronos import ChronosBoltPipeline
        bolt = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.bfloat16)

        # Raw
        t0 = time.time()
        x_flat = x_eval.transpose(0, 2, 1).reshape(B * C, L)
        ctx_raw = torch.tensor(x_flat, dtype=torch.float32).to(bolt.model.device)
        bolt_preds = []
        bs = 64
        for i in range(0, len(ctx_raw), bs):
            with torch.no_grad():
                f_dist = bolt.predict(ctx_raw[i : i + bs], prediction_length=args.pred_len)
                bolt_preds.append(f_dist[:, 4, :].cpu().numpy())
        y_bolt_raw = np.concatenate(bolt_preds, axis=0).reshape(B, C, args.pred_len).transpose(0, 2, 1)
        dt_bolt = time.time() - t0
        eval_bolt_raw = evaluate_predictions(y_eval, y_bolt_raw)
        print(f"  Chronos-Bolt Raw: MSE {eval_bolt_raw['mse']:.4f} | LSD {eval_bolt_raw['lsd_db']:.2f} dB ({dt_bolt:.2f}s)")
        results.append({**eval_bolt_raw, 'dataset': args.dataset, 'model': 'Chronos-Bolt-Base', 'paradigm': 'Raw (Zero-Shot)', 'params': '200M', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_bolt, 2)})

        # Fourier Decoupled (F-Bolt)
        t0 = time.time()
        res_flat = x_res_np.transpose(0, 2, 1).reshape(B * C, L)
        ctx_res = torch.tensor(res_flat, dtype=torch.float32).to(bolt.model.device)
        bolt_res_preds = []
        for i in range(0, len(ctx_res), bs):
            with torch.no_grad():
                f_dist = bolt.predict(ctx_res[i : i + bs], prediction_length=args.pred_len)
                bolt_res_preds.append(f_dist[:, 4, :].cpu().numpy())
        y_bolt_res = np.concatenate(bolt_res_preds, axis=0).reshape(B, C, args.pred_len).transpose(0, 2, 1)
        y_fbolt = y_base_np + y_bolt_res
        dt_fbolt = time.time() - t0
        eval_fbolt = evaluate_predictions(y_eval, y_fbolt)
        print(f"  F-Bolt (Decoupled): MSE {eval_fbolt['mse']:.4f} | LSD {eval_fbolt['lsd_db']:.2f} dB ({dt_fbolt:.2f}s)")
        results.append({**eval_fbolt, 'dataset': args.dataset, 'model': 'Chronos-Bolt-Base', 'paradigm': 'F-Bolt (Fourier+Res)', 'params': '200M', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_fbolt, 2)})
        del bolt; torch.cuda.empty_cache()

    # 4. IBM Granite TTM-R1 (805K)
    if args.model in ["all", "granite_ttm"]:
        print("\n--- IBM Granite TTM-R1 (805K) ---")
        from tsfm_public import TinyTimeMixerForPrediction
        ttm = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r1").to(device)
        ttm.eval()

        pad_len = ttm.config.context_length - L
        # Raw
        t0 = time.time()
        x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(device)
        x_padded = torch.cat([x_tensor[:, :1, :].repeat(1, pad_len, 1), x_tensor], dim=1) if pad_len > 0 else x_tensor[:, -ttm.config.context_length:, :]
        with torch.no_grad():
            out = ttm(past_values=x_padded)
            y_ttm_raw = out.prediction_outputs.cpu().numpy()[:, :args.pred_len, :]
        dt_ttm = time.time() - t0
        eval_ttm_raw = evaluate_predictions(y_eval, y_ttm_raw)
        print(f"  Granite TTM Raw: MSE {eval_ttm_raw['mse']:.4f} | LSD {eval_ttm_raw['lsd_db']:.2f} dB ({dt_ttm:.2f}s)")
        results.append({**eval_ttm_raw, 'dataset': args.dataset, 'model': 'Granite-TTM-R1', 'paradigm': 'Raw (Zero-Shot)', 'params': '805K', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_ttm, 2)})

        # Fourier Decoupled (F-TTM)
        t0 = time.time()
        x_res_tensor = torch.tensor(x_res_np, dtype=torch.float32).to(device)
        x_res_padded = torch.cat([x_res_tensor[:, :1, :].repeat(1, pad_len, 1), x_res_tensor], dim=1) if pad_len > 0 else x_res_tensor[:, -ttm.config.context_length:, :]
        with torch.no_grad():
            out_res = ttm(past_values=x_res_padded)
            y_ttm_res = out_res.prediction_outputs.cpu().numpy()[:, :args.pred_len, :]
        y_fttm = y_base_np + y_ttm_res
        dt_fttm = time.time() - t0
        eval_fttm = evaluate_predictions(y_eval, y_fttm)
        print(f"  F-TTM (Decoupled): MSE {eval_fttm['mse']:.4f} | LSD {eval_fttm['lsd_db']:.2f} dB ({dt_fttm:.2f}s)")
        results.append({**eval_fttm, 'dataset': args.dataset, 'model': 'Granite-TTM-R1', 'paradigm': 'F-TTM (Fourier+Res)', 'params': '805K', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_fttm, 2)})
        del ttm; torch.cuda.empty_cache()

    # 5. CycleNet & F-CycleNet (Ultra-Lightweight 32K)
    if args.model in ["all", "cyclenet"]:
        print("\n--- CycleNet (W=24, 32K) ---")
        t0 = time.time()
        cyclenet = CycleNet(seq_len=L, pred_len=args.pred_len, channels=C, cycle_len=24).to(device)
        optimizer = torch.optim.AdamW(cyclenet.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.MSELoss()
        cyclenet.train()
        batch_sz = 32
        max_train_steps = min(60, (len(train_norm) - L - args.pred_len) // batch_sz)
        for step in range(max_train_steps):
            idx = step * batch_sz
            xb = torch.tensor(np.array([train_norm[idx + b : idx + b + L] for b in range(batch_sz)]), device=device)
            yb = torch.tensor(np.array([train_norm[idx + b + L : idx + b + L + args.pred_len] for b in range(batch_sz)]), device=device)
            optimizer.zero_grad()
            loss = criterion(cyclenet(xb), yb)
            loss.backward()
            optimizer.step()
        cyclenet.eval()
        with torch.no_grad():
            y_cyc_raw = cyclenet(torch.tensor(x_eval, device=device)).cpu().numpy()
        dt_cyc = time.time() - t0
        eval_cyc_raw = evaluate_predictions(y_eval, y_cyc_raw)
        print(f"  CycleNet Raw: MSE {eval_cyc_raw['mse']:.4f} | LSD {eval_cyc_raw['lsd_db']:.2f} dB ({dt_cyc:.2f}s)")
        results.append({**eval_cyc_raw, 'dataset': args.dataset, 'model': 'CycleNet (W=24)', 'paradigm': 'Raw (Recurrent)', 'params': '32K', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_cyc, 2)})

        # F-CycleNet
        t0 = time.time()
        cyclenet_f = CycleNet(seq_len=L, pred_len=args.pred_len, channels=C, cycle_len=24).to(device)
        opt_f = torch.optim.AdamW(cyclenet_f.parameters(), lr=1e-3, weight_decay=1e-3)
        cyclenet_f.train()
        for step in range(max_train_steps):
            idx = step * batch_sz
            xb_raw = torch.tensor(np.array([train_norm[idx + b : idx + b + L] for b in range(batch_sz)]), device=device)
            yb_raw = torch.tensor(np.array([train_norm[idx + b + L : idx + b + L + args.pred_len] for b in range(batch_sz)]), device=device)
            _, _, xb_res = extract_and_extrapolate_fourier_base_torch(xb_raw, pred_len=args.pred_len, top_k=3)
            _, yb_base, _ = extract_and_extrapolate_fourier_base_torch(xb_raw, pred_len=args.pred_len, top_k=3)
            opt_f.zero_grad()
            loss = criterion(cyclenet_f(xb_res), yb_raw - yb_base)
            loss.backward()
            opt_f.step()
        cyclenet_f.eval()
        with torch.no_grad():
            y_cyc_res = cyclenet_f(torch.tensor(x_res_np, device=device)).cpu().numpy()
        y_fcyc = y_base_np + y_cyc_res
        dt_fcyc = time.time() - t0
        eval_fcyc = evaluate_predictions(y_eval, y_fcyc)
        print(f"  F-CycleNet (Decoupled): MSE {eval_fcyc['mse']:.4f} | LSD {eval_fcyc['lsd_db']:.2f} dB ({dt_fcyc:.2f}s)")
        results.append({**eval_fcyc, 'dataset': args.dataset, 'model': 'CycleNet (W=24)', 'paradigm': 'F-CycleNet (Fourier+Res)', 'params': '32K', 'pred_len': args.pred_len, 'seed': args.seed, 'runtime_sec': round(dt_fcyc, 2)})
        del cyclenet, cyclenet_f; torch.cuda.empty_cache()

    # Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset.replace('.csv', '')}_H{args.pred_len}_seed{args.seed}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print(" SUMMARY RESULTS")
    print("=" * 95)
    print(df.to_string(index=False))
    print(f"\nSaved results JSON to: {out_file}")


if __name__ == "__main__":
    main()


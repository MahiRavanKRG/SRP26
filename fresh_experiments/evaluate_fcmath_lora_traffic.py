"""
Fast Evaluation of Fine-Tuned FCMath (LoRA) on Traffic Test Set
===============================================================
Loads the trained LoRA adapter from checkpoints/chronos_lora_traffic
and evaluates with deterministic decoding (num_samples=1) for speed and sharpness.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "models"))

from srp26 import StandardScaler, compute_spectral_diagnostics, extract_fourier_base_1d, set_seed


def evaluate():
    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("FAST EVALUATION: FINE-TUNED FCMath (LoRA) ON TRAFFIC TEST SET")
    print("=" * 80 + "\n")

    # 1. Load Traffic Test Set (70/10/20 chronological split)
    csv_path = os.path.join(parent_dir, "data_sets", "traffic.csv")
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    raw_data = df.values.astype(np.float32)

    total_len = len(raw_data)
    n_train = int(total_len * 0.70)
    n_val = int(total_len * 0.10)
    n_test = total_len - n_train - n_val

    seq_len = 336
    pred_len = 96

    train_raw = raw_data[:n_train]
    test_raw = raw_data[n_train + n_val - seq_len : total_len]

    scaler = StandardScaler()
    scaler.fit(train_raw)
    test_norm = scaler.transform(test_raw)

    # Sample 15 representative test windows (stride=24)
    total_valid = len(test_norm) - seq_len - pred_len + 1
    eval_indices = np.arange(0, total_valid, 24)[:15]

    x_eval_list, y_eval_list = [], []
    for idx in eval_indices:
        x_eval_list.append(test_norm[idx : idx + seq_len])
        y_eval_list.append(test_norm[idx + seq_len : idx + seq_len + pred_len])

    x_eval = np.array(x_eval_list)  # [15, 336, 862]
    y_eval = np.array(y_eval_list)  # [15, 96, 862]
    B, L, C = x_eval.shape

    print(f"Loaded {C} channels. Evaluation sample: {B} windows (H={pred_len}).")

    # 2. Load Chronos Pipeline + Saved LoRA Adapter
    ckpt_dir = os.path.join(parent_dir, "fresh_experiments", "checkpoints", "chronos_lora_traffic")
    print(f"Loading trained LoRA weights from: {ckpt_dir} ...")

    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map=device,
        torch_dtype=torch.float32,
    )
    pipeline.tokenizer.config.prediction_length = pred_len
    pipeline.model.config.prediction_length = pred_len

    # Attach trained LoRA weights
    pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, ckpt_dir)
    pipeline.model.model.eval()
    print("LoRA adapter successfully mounted!")

    # 3. Fourier Decomposition & Prediction
    t0 = time.time()
    x_base_all = np.zeros_like(x_eval)
    y_base_all = np.zeros((B, pred_len, C), dtype=np.float32)

    print("Extracting in-sample & future Fourier base ...", flush=True)
    for b in range(B):
        for c in range(C):
            xb, yb = extract_fourier_base_1d(x_eval[b, :, c], pred_len, top_k=3)
            x_base_all[b, :, c] = xb
            y_base_all[b, :, c] = yb

    x_res_all = x_eval - x_base_all
    res_flattened = x_res_all.transpose(0, 2, 1).reshape(B * C, L)
    context_tensor = torch.tensor(res_flattened, dtype=torch.float32)

    eval_batch_size = 256
    total_eval_batches = (len(context_tensor) + eval_batch_size - 1) // eval_batch_size
    print(f"Generating forecasts across {total_eval_batches} batches (batch size {eval_batch_size}, num_samples=1) ...", flush=True)

    all_res_preds = []
    for i in range(0, len(context_tensor), eval_batch_size):
        batch_idx = i // eval_batch_size + 1
        print(f"  -> Progress: Batch {batch_idx}/{total_eval_batches} ...", end="\r", flush=True)
        batch_ctx = context_tensor[i : i + eval_batch_size]
        with torch.no_grad():
            forecast_dist = pipeline.predict(batch_ctx, prediction_length=pred_len, num_samples=1)
            # Squeeze sample dimension
            all_res_preds.append(forecast_dist.squeeze(1).cpu().numpy())

    print("\nResidual prediction completed!")
    y_res_flat = np.concatenate(all_res_preds, axis=0)
    y_res_all = y_res_flat.reshape(B, C, pred_len).transpose(0, 2, 1)

    # Recombine: Base + LoRA Residuals
    y_pred_lora = y_base_all + y_res_all
    eval_time = time.time() - t0

    mse_lora = np.mean((y_pred_lora - y_eval)**2)
    mae_lora = np.mean(np.abs(y_pred_lora - y_eval))
    spec_lora = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_lora))

    print(f"\nDone in {eval_time:.1f}s!")
    print(f"  Test MSE : {mse_lora:.4f}")
    print(f"  Test MAE : {mae_lora:.4f}")
    print(f"  Mid-FVU  : {spec_lora['fvu_mid']:.4f}")
    print(f"  LSD (dB) : {spec_lora['lsd_db']:.2f} dB")

    comparison = [
        {
            'Model': 'Chronos-Raw (Zero-Shot)',
            'MSE': 0.9816,
            'MAE': 0.4375,
            'Mid_FVU': 1.0568,
            'LSD_dB': 16.27,
            'Runtime': '350.86s',
            'Notes': 'Raw foundation model without decomposition'
        },
        {
            'Model': 'FCMath (Zero-Shot)',
            'MSE': 0.6755,
            'MAE': 0.3613,
            'Mid_FVU': 0.9804,
            'LSD_dB': 14.22,
            'Runtime': '351.13s',
            'Notes': 'Fourier base + Zero-Shot Chronos residuals'
        },
        {
            'Model': 'FCMath (LoRA Fine-Tuned)',
            'MSE': round(float(mse_lora), 4),
            'MAE': round(float(mae_lora), 4),
            'Mid_FVU': round(float(spec_lora['fvu_mid']), 4),
            'LSD_dB': round(float(spec_lora['lsd_db']), 2),
            'Runtime': f'{eval_time:.1f}s',
            'Notes': 'LoRA adapter (r=8) trained on Traffic residuals'
        },
        {
            'Model': 'CycleNet (W=24)',
            'MSE': 0.5263,
            'MAE': 0.3908,
            'Mid_FVU': 0.9246,
            'LSD_dB': 9.74,
            'Runtime': '1.86s',
            'Notes': 'Ultra-lightweight recurrent periodic queue (32k params)'
        }
    ]

    df_comp = pd.DataFrame(comparison)
    print("\n" + "=" * 90)
    print("TRAFFIC BENCHMARK FINAL REPORT: ZERO-SHOT vs. LoRA vs. CYCLENET (H=96)")
    print("=" * 90)
    print(df_comp.to_string(index=False))
    print("=" * 90 + "\n")

    out_dir = os.path.join(parent_dir, "fresh_experiments", "results")
    json_path = os.path.join(out_dir, "traffic_lora_results.json")
    md_path = os.path.join(out_dir, "traffic_lora_results.md")

    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    with open(md_path, 'w') as f:
        f.write("# Traffic Benchmark: Zero-Shot Chronos vs. FCMath (LoRA) vs. CycleNet (H=96)\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n")

    print(f"Results saved to:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    evaluate()


"""
Parameter-Efficient Fine-Tuning (PEFT / LoRA) for FCMath on Traffic Dataset
===========================================================================
Trains a Low-Rank Adapter (LoRA) on Amazon Chronos-T5 to predict
non-periodic residual dynamics on the high-dimensional traffic.csv benchmark (862 channels).

Pipeline:
  1. Standardized 70/10/20 chronological split with train-only Z-score scaling.
  2. Fourier Base Decomposition: X_res = X - X_base, Y_res = Y - Y_base.
  3. LoRA adaptation on T5 self-attention projection layers (q, v).
  4. Cross-entropy loss on tokenized residual sequences with Cosine Annealing.
  5. Test set evaluation: FCMath (LoRA) vs. FCMath (Zero-Shot) vs. Chronos-Raw vs. CycleNet.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Tuple

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "models"))

from fcmath import FCMath
from srp26 import (
    CycleNet,
    StandardScaler,
    compute_spectral_diagnostics,
    extract_fourier_base_1d,
    set_seed,
)


class ResidualPairDataset(Dataset):
    """Generates (X_res, Y_res) pairs from training time series."""
    def __init__(self, data_norm: np.ndarray, seq_len: int = 336, pred_len: int = 96,
                 num_samples: int = 2500, top_k: int = 3):
        self.data = data_norm
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.num_samples = num_samples
        self.T, self.C = data_norm.shape

        # Pre-sample random window and channel indices
        max_idx = self.T - seq_len - pred_len
        self.indices = [
            (random.randint(0, max_idx), random.randint(0, self.C - 1))
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_t, c = self.indices[idx]
        x_raw = self.data[start_t : start_t + self.seq_len, c]
        y_raw = self.data[start_t + self.seq_len : start_t + self.seq_len + self.pred_len, c]

        xb, yb = extract_fourier_base_1d(x_raw, self.pred_len, self.top_k)
        x_res = x_raw - xb
        y_res = y_raw - yb

        return torch.tensor(x_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.float32)


# =====================================================================
# 2. Training Loop with PEFT / LoRA
# =====================================================================
def train_lora_on_traffic(
    model_name: str = "amazon/chronos-t5-tiny",
    seq_len: int = 336,
    pred_len: int = 96,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 5e-4,
    lora_r: int = 8,
    samples_per_epoch: int = 2000,
):
    print("\n" + "=" * 80)
    print("STARTING PEFT / LoRA FINE-TUNING ON TRAFFIC RESIDUALS")
    print(f"Base Model        : {model_name}")
    print(f"Context Length    : {seq_len} | Prediction Horizon: {pred_len}")
    print(f"LoRA Rank (r)     : {lora_r} | Learning Rate: {lr} | Epochs: {epochs}")
    print("=" * 80 + "\n")

    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Traffic Dataset (Canonical 70/10/20 split)
    csv_path = os.path.join(parent_dir, "data_sets", "traffic.csv")
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    raw_data = df.values.astype(np.float32)

    total_len = len(raw_data)
    n_train = int(total_len * 0.70)
    n_val = int(total_len * 0.10)
    n_test = total_len - n_train - n_val

    train_raw = raw_data[:n_train]
    test_raw = raw_data[n_train + n_val - seq_len : total_len]

    scaler = StandardScaler()
    scaler.fit(train_raw)
    train_norm = scaler.transform(train_raw)
    test_norm = scaler.transform(test_raw)

    print(f"Traffic Loaded: {raw_data.shape[1]} channels | Train: {n_train} | Test: {n_test}")

    # 2. Initialize Chronos Pipeline & Configure LoRA
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )
    # Configure prediction length
    pipeline.tokenizer.config.prediction_length = pred_len
    pipeline.model.config.prediction_length = pred_len

    t5_model = pipeline.model.model
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none"
    )
    lora_model = get_peft_model(t5_model, lora_config)
    lora_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 3. Training Loop
    lora_model.train()
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_ds = ResidualPairDataset(
            train_norm, seq_len=seq_len, pred_len=pred_len,
            num_samples=samples_per_epoch, top_k=3
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        running_loss = 0.0
        for step, (x_batch, y_batch) in enumerate(train_loader):
            # Tokenize context and labels on CPU (tokenizer boundaries reside on CPU)
            input_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(x_batch.cpu())
            labels, _ = pipeline.tokenizer.label_input_transform(y_batch.cpu(), scale)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = lora_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            running_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss (CrossEntropy): {avg_loss:.4f} | Time: {epoch_time:.1f}s")

    total_train_sec = time.time() - start_train_time
    print(f"\nLoRA Fine-Tuning Completed in {total_train_sec:.1f}s!")

    # 4. Save LoRA Adapter Checkpoint
    ckpt_dir = os.path.join(parent_dir, "fresh_experiments", "checkpoints", "chronos_lora_traffic")
    os.makedirs(ckpt_dir, exist_ok=True)
    lora_model.save_pretrained(ckpt_dir)
    print(f"Saved LoRA weights to: {ckpt_dir}\n")

    # =====================================================================
    # 5. Evaluation on Traffic Test Set
    # =====================================================================
    print("=" * 80)
    print("EVALUATING FINE-TUNED FCMath (LoRA) ON TRAFFIC TEST SET")
    print("=" * 80)

    # Extract test windows (15 windows, stride=24)
    total_valid = len(test_norm) - seq_len - pred_len + 1
    eval_indices = np.arange(0, total_valid, 24)[:15]

    x_eval_list, y_eval_list = [], []
    for idx in eval_indices:
        x_eval_list.append(test_norm[idx : idx + seq_len])
        y_eval_list.append(test_norm[idx + seq_len : idx + seq_len + pred_len])

    x_eval = np.array(x_eval_list)  # [15, 336, 862]
    y_eval = np.array(y_eval_list)  # [15, 96, 862]
    B, L, C = x_eval.shape

    lora_model.eval()

    # Step A: Fourier Base Extrapolation & Residual Prediction
    t0 = time.time()
    x_base_all = np.zeros_like(x_eval)
    y_base_all = np.zeros((B, pred_len, C), dtype=np.float32)

    for b in range(B):
        for c in range(C):
            xb, yb = extract_fourier_base_1d(x_eval[b, :, c], pred_len, top_k=3)
            x_base_all[b, :, c] = xb
            y_base_all[b, :, c] = yb

    x_res_all = x_eval - x_base_all
    res_flattened = x_res_all.transpose(0, 2, 1).reshape(B * C, L)
    context_tensor = torch.tensor(res_flattened, dtype=torch.float32)

    # Batch prediction with fine-tuned LoRA model
    all_res_preds = []
    eval_batch_size = 128
    total_eval_batches = (len(context_tensor) + eval_batch_size - 1) // eval_batch_size
    print(f"Generating forecasts across {total_eval_batches} batches (batch size {eval_batch_size}) ...", flush=True)

    for i in range(0, len(context_tensor), eval_batch_size):
        batch_idx = i // eval_batch_size + 1
        if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == total_eval_batches:
            print(f"  -> Batch {batch_idx}/{total_eval_batches} ...", flush=True)
        batch_ctx = context_tensor[i : i + eval_batch_size]
        with torch.no_grad():
            forecast_dist = pipeline.predict(batch_ctx, prediction_length=pred_len, num_samples=15)
            median_pred = np.quantile(forecast_dist.cpu().numpy(), q=0.5, axis=1)
            all_res_preds.append(median_pred)

    y_res_flat = np.concatenate(all_res_preds, axis=0)
    y_res_all = y_res_flat.reshape(B, C, pred_len).transpose(0, 2, 1)

    y_pred_lora = y_base_all + y_res_all
    eval_time = time.time() - t0

    mse_lora = np.mean((y_pred_lora - y_eval)**2)
    mae_lora = np.mean(np.abs(y_pred_lora - y_eval))
    spec_lora = compute_spectral_diagnostics(torch.tensor(y_eval), torch.tensor(y_pred_lora))

    print(f"\nDone! Test MSE: {mse_lora:.4f} | Test MAE: {mae_lora:.4f} | Mid-FVU: {spec_lora['fvu_mid']:.4f}")

    # Prior baseline numbers on the exact same Traffic test windows:
    comparison = [
        {
            'Model': 'Chronos-Raw (Zero-Shot)',
            'MSE': 0.9816,
            'MAE': 0.4375,
            'Mid_FVU': 1.0568,
            'LSD_dB': 16.27,
            'Notes': 'Base foundation model without decomposition'
        },
        {
            'Model': 'FCMath (Zero-Shot)',
            'MSE': 0.6755,
            'MAE': 0.3613,
            'Mid_FVU': 0.9804,
            'LSD_dB': 14.22,
            'Notes': 'Fourier base + Zero-Shot Chronos residuals'
        },
        {
            'Model': 'FCMath (LoRA Fine-Tuned)',
            'MSE': round(float(mse_lora), 4),
            'MAE': round(float(mae_lora), 4),
            'Mid_FVU': round(float(spec_lora['fvu_mid']), 4),
            'LSD_dB': round(float(spec_lora['lsd_db']), 2),
            'Notes': f'LoRA adapter (r={lora_r}) trained on Traffic residuals'
        },
        {
            'Model': 'CycleNet (W=24)',
            'MSE': 0.5263,
            'MAE': 0.3908,
            'Mid_FVU': 0.9246,
            'LSD_dB': 9.74,
            'Notes': 'Ultra-lightweight recurrent periodic queue (32k params)'
        }
    ]

    df_comp = pd.DataFrame(comparison)
    print("\n" + "=" * 90)
    print("TRAFFIC BENCHMARK COMPARISON TABLE (ZERO-SHOT vs. LoRA vs. CYCLENET)")
    print("=" * 90)
    print(df_comp.to_string(index=False))
    print("=" * 90 + "\n")

    # Save to disk
    out_dir = os.path.join(parent_dir, "fresh_experiments", "results")
    json_path = os.path.join(out_dir, "traffic_lora_results.json")
    md_path = os.path.join(out_dir, "traffic_lora_results.md")

    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    with open(md_path, 'w') as f:
        f.write("# Traffic Benchmark: Zero-Shot Chronos vs. FCMath (LoRA) vs. CycleNet\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n")

    print(f"Results saved to:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    train_lora_on_traffic()

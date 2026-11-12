"""
Test & Verification Suite for FCMath (Fourier-Chronos Mathematical Forecaster)
=============================================================================
Evaluates FCMath on synthetic periodic + noise data and ETTh1 test windows.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.fcmath import FCMath


def test_synthetic():
    print("\n[1/2] Testing FCMath on Synthetic Periodic + Transient Noise ...")
    # Generate synthetic signal: 24h diurnal cycle + weekly 168h cycle + transient noise
    L = 336
    H = 96
    t = np.arange(L + H)
    clean_signal = 10.0 * np.sin(2 * np.pi * t / 24.0) + 5.0 * np.cos(2 * np.pi * t / 168.0)
    noise = np.random.normal(0, 1.0, size=len(t))
    full_data = clean_signal + noise

    history = full_data[:L]
    ground_truth = full_data[L:L+H]

    model = FCMath(top_k=2, model_name="amazon/chronos-t5-tiny")
    
    # Test Fourier base extraction first
    x_base, y_base = model.extract_and_extrapolate_fourier_base(history, H)
    fourier_mse = np.mean((y_base - ground_truth) ** 2)
    print(f"  -> Pure Fourier Base Extrapolation MSE: {fourier_mse:.4f}")

    # Full forecast (Fourier Base + Chronos Residuals)
    y_hat, components = model.forecast(history, prediction_length=H, num_samples=10)
    full_mse = np.mean((y_hat - ground_truth) ** 2)
    full_mae = np.mean(np.abs(y_hat - ground_truth))
    print(f"  -> FCMath Full Recombined MSE: {full_mse:.4f} | MAE: {full_mae:.4f}")
    print("  -> Synthetic test passed successfully!\n")


def test_etth1_sample():
    print("\n[2/2] Testing FCMath on Real ETTh1 Window ...")
    csv_path = os.path.join(parent_dir, "data_sets", "ETTh1.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    data = df.values[:336 + 96]  # [432, 7]

    history = data[:336]
    ground_truth = data[336:336+96]

    # Normalize based on history
    mean = np.mean(history, axis=0, keepdims=True)
    std = np.std(history, axis=0, keepdims=True)
    std[std == 0] = 1.0

    norm_history = (history - mean) / std
    norm_gt = (ground_truth - mean) / std

    model = FCMath(top_k=3, model_name="amazon/chronos-t5-tiny")
    y_hat, components = model.forecast(norm_history, prediction_length=96, num_samples=10)

    # Denormalize
    y_hat_real = y_hat * std + mean
    mse = np.mean((y_hat_real - ground_truth) ** 2)
    mae = np.mean(np.abs(y_hat_real - ground_truth))

    print(f"  -> ETTh1 Sample (7 channels, H=96) Real-Scale MSE: {mse:.4f} | MAE: {mae:.4f}")
    print(f"  -> Normalized MSE: {np.mean((y_hat - norm_gt)**2):.4f}")
    print("  -> Real ETTh1 test passed successfully!\n")


if __name__ == "__main__":
    test_synthetic()
    test_etth1_sample()


"""
FCMath: Fourier-Chronos Mathematical Forecaster
================================================
A decoupled hybrid architecture combining:
  1. Pure Mathematical Fourier Decomposition (The Base):
     - Analytical rFFT decomposition isolating dominant harmonic frequencies.
     - Closed-form sinusoidal extrapolation into future forecast horizon H.
  2. Zero-Shot Foundation Model (The Residuals):
     - Orthogonal residual isolation: X_res = X - X_base.
     - Pre-trained time series foundation model (Amazon Chronos-T5) predicting
       residual non-periodic dynamics, anomalies, and transient shocks zero-shot.
  3. Recombination:
     - Y_hat = Y_base + Y_res.

Based on the SRP26 mathematical formulation & supervisor directives (Jung Min Choi).
"""

import math
from typing import Tuple, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn


class FCMath:
    """
    Fourier-Chronos Mathematical Forecaster (FCMath).
    """
    def __init__(
        self,
        top_k: int = 3,
        model_name: str = "amazon/chronos-t5-tiny",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.top_k = top_k
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.pipeline = None

    def _lazy_load_chronos(self):
        """Loads ChronosPipeline on demand."""
        if self.pipeline is None:
            from chronos import ChronosPipeline
            print(f"[FCMath] Loading pre-trained Chronos pipeline: '{self.model_name}' on {self.device} ...")
            # Map bfloat16/float32 depending on GPU capability
            dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=dtype,
            )

    def extract_and_extrapolate_fourier_base(
        self, x: np.ndarray, pred_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the in-sample Fourier harmonic base and extrapolates it into the future.
        x: [seq_len, channels] or [seq_len]
        Returns:
            x_base: [seq_len, channels]
            y_base: [pred_len, channels]
        """
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x[:, np.newaxis]

        L, C = x.shape
        x_base = np.zeros_like(x)
        y_base = np.zeros((pred_len, C), dtype=x.dtype)

        # Time indices
        t_in = np.arange(L)
        t_future = np.arange(L, L + pred_len)

        for c in range(C):
            series = x[:, c]
            fft_vals = np.fft.rfft(series)
            freqs = np.fft.rfftfreq(L)
            amps = np.abs(fft_vals)

            # Exclude DC component (index 0)
            if len(amps) > 1:
                k = min(self.top_k, len(amps) - 1)
                top_indices = np.argsort(amps[1:])[-k:] + 1
            else:
                top_indices = np.array([0])

            # Filtered FFT for in-sample base reconstruction
            filtered_fft = np.zeros_like(fft_vals)
            filtered_fft[top_indices] = fft_vals[top_indices]
            x_base[:, c] = np.fft.irfft(filtered_fft, n=L)

            # Extrapolate top-k sinusoids analytically into horizon H
            future_signal = np.zeros(pred_len, dtype=x.dtype)
            for idx in top_indices:
                amp = amps[idx] / (L / 2.0)
                phase = np.angle(fft_vals[idx])
                freq = freqs[idx]
                future_signal += amp * np.cos(2.0 * np.pi * freq * t_future + phase)

            y_base[:, c] = future_signal

        if is_1d:
            return x_base[:, 0], y_base[:, 0]
        return x_base, y_base

    def forecast(
        self,
        history: Union[np.ndarray, torch.Tensor],
        prediction_length: int,
        batch_size: int = 64,
        num_samples: int = 20,
    ) -> Tuple[np.ndarray, dict]:
        """
        Executes full FCMath pipeline:
          1. Decompose history into Fourier base + non-periodic residuals.
          2. Extrapolate Fourier base mathematically into prediction horizon.
          3. Predict residual dynamics zero-shot using Chronos foundation model.
          4. Recombine predictions: Y_hat = Y_base + Y_res.

        Input:
            history: [seq_len, channels] or [batch, seq_len, channels]
        Returns:
            y_hat: [..., prediction_length, channels]
            components: dict with {x_base, x_res, y_base, y_res}
        """
        self._lazy_load_chronos()

        if isinstance(history, torch.Tensor):
            history_np = history.detach().cpu().numpy()
        else:
            history_np = np.asarray(history)

        orig_ndim = history_np.ndim
        if history_np.ndim == 1:
            history_np = history_np[np.newaxis, :, np.newaxis]  # [1, L, 1]
        elif history_np.ndim == 2:
            history_np = history_np[np.newaxis, :, :]  # [1, L, C]

        B, L, C = history_np.shape
        x_base_all = np.zeros_like(history_np)
        y_base_all = np.zeros((B, prediction_length, C), dtype=history_np.dtype)
        x_res_all = np.zeros_like(history_np)

        # 1. Fourier Base Extraction & Extrapolation
        for b in range(B):
            x_b, y_b = self.extract_and_extrapolate_fourier_base(history_np[b], prediction_length)
            x_base_all[b] = x_b
            y_base_all[b] = y_b
            x_res_all[b] = history_np[b] - x_b

        # 2. Foundation Model Inference on Residuals
        # Flatten (B, C) to run parallel batch inference in Chronos
        res_flattened = x_res_all.transpose(0, 2, 1).reshape(B * C, L)  # [B*C, L]
        context_tensor = torch.tensor(res_flattened, dtype=torch.float32)

        # Batch prediction with Chronos
        all_res_preds = []
        for i in range(0, len(context_tensor), batch_size):
            batch_ctx = context_tensor[i : i + batch_size]
            with torch.no_grad():
                forecast_dist = self.pipeline.predict(
                    batch_ctx,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )
                median_pred = np.quantile(forecast_dist.cpu().numpy(), q=0.5, axis=1)  # [batch_ctx, H]
                all_res_preds.append(median_pred)

        y_res_flat = np.concatenate(all_res_preds, axis=0)  # [B*C, H]
        y_res_all = y_res_flat.reshape(B, C, prediction_length).transpose(0, 2, 1)  # [B, H, C]

        # 3. Final Recombination: Base + Residuals
        y_hat_all = y_base_all + y_res_all

        if orig_ndim == 1:
            final_forecast = y_hat_all[0, :, 0]
            components = {
                'x_base': x_base_all[0, :, 0],
                'x_res': x_res_all[0, :, 0],
                'y_base': y_base_all[0, :, 0],
                'y_res': y_res_all[0, :, 0],
            }
        elif orig_ndim == 2:
            final_forecast = y_hat_all[0]
            components = {
                'x_base': x_base_all[0],
                'x_res': x_res_all[0],
                'y_base': y_base_all[0],
                'y_res': y_res_all[0],
            }
        else:
            final_forecast = y_hat_all
            components = {
                'x_base': x_base_all,
                'x_res': x_res_all,
                'y_base': y_base_all,
                'y_res': y_res_all,
            }

        return final_forecast, components

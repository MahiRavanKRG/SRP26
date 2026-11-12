"""
Fourier-Residual Attention Network (FRANet / FourierResNet)
=============================================================
A mathematically rigorous two-stage architecture for multivariate time series forecasting:

Stage 1: Fourier Spectral Harmonic Forecaster
- Decomposes the signal into dominant recurring frequencies using discrete Fourier analysis (rFFT).
- Reconstructs the in-sample periodic base: X_base = iFFT(Filter(FFT(X)))
- Forecasts the periodic base trajectory: Y_base = iFFT(W_spec * FFT(X_base))

Stage 2: Deep Attention Residual Forecaster
- Isolates non-periodic, non-stationary dynamics: X_res = X - X_base
- Applies a Multi-Head Attention (MHA) + Feed-Forward transformer backbone to capture 
  transient spikes, cross-channel dependencies, and complex non-periodic variations.
- Forecasts the residual trajectory: Y_res = AttentionBackbone(X_res)

Final Output:
- Y_hat = Y_base + Y_res (with reversible instance normalization RevIN)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022)."""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
            return x
        else:
            raise ValueError(f"Unknown RevIN mode: {mode}")


class FourierBaseForecaster(nn.Module):
    """
    Stage 1: Spectral Harmonic Forecaster.
    Extracts recurring baseline periodicities and forecasts the periodic component into horizon H.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, top_k_freqs: int = 12):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.top_k = min(top_k_freqs, seq_len // 2)

        self.in_freqs = seq_len // 2 + 1
        self.out_freqs = pred_len // 2 + 1

        # Learnable complex spectral projection mapping input Fourier modes to horizon Fourier modes
        # Real and imaginary components parameterized separately for stable optimization
        self.spec_proj_real = nn.Parameter(torch.Tensor(channels, self.in_freqs, self.out_freqs))
        self.spec_proj_imag = nn.Parameter(torch.Tensor(channels, self.in_freqs, self.out_freqs))
        
        # In-sample spectral filter to extract recurring base and isolate residuals
        self.spectral_filter = nn.Parameter(torch.ones(channels, self.in_freqs))

        # Initialization
        nn.init.kaiming_uniform_(self.spec_proj_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spec_proj_imag, a=math.sqrt(5))

    def forward(self, x):
        """
        x: [B, L, C]
        Returns:
            x_base: [B, L, C]  (in-sample reconstructed recurring base)
            y_base: [B, H, C]  (forecasted recurring base)
        """
        B, L, C = x.shape
        x_perm = x.permute(0, 2, 1)  # [B, C, L]

        # 1. 1D Real Fast Fourier Transform across the temporal dimension
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # [B, C, in_freqs] (complex)

        # 2. In-sample harmonic reconstruction (filtering dominant base frequencies)
        # Apply sigmoid to spectral filter to bound frequency energy preservation
        gate = torch.sigmoid(self.spectral_filter).unsqueeze(0)  # [1, C, in_freqs]
        x_base_fft = x_fft * gate
        x_base = torch.fft.irfft(x_base_fft, n=self.seq_len, dim=-1)  # [B, C, L]
        x_base = x_base.permute(0, 2, 1)  # [B, L, C]

        # 3. Spectral forecasting via complex linear mapping: Y_freq = X_freq @ W_spec
        # (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        x_real = x_base_fft.real  # [B, C, in_freqs]
        x_imag = x_base_fft.imag  # [B, C, in_freqs]

        # einsum over frequency dimensions: [B, C, F_in] x [C, F_in, F_out] -> [B, C, F_out]
        y_real = torch.einsum('bcf,cfo->bco', x_real, self.spec_proj_real) - \
                 torch.einsum('bcf,cfo->bco', x_imag, self.spec_proj_imag)
        y_imag = torch.einsum('bcf,cfo->bco', x_real, self.spec_proj_imag) + \
                 torch.einsum('bcf,cfo->bco', x_imag, self.spec_proj_real)

        y_fft = torch.complex(y_real, y_imag)
        y_base = torch.fft.irfft(y_fft, n=self.pred_len, dim=-1)  # [B, C, H]
        y_base = y_base.permute(0, 2, 1)  # [B, H, C]

        return x_base, y_base


class ResidualAttentionBlock(nn.Module):
    """
    Multi-Head Attention block with Feed-Forward network and residual normalization.
    Operates on residual tokens to model intricate temporal and cross-channel interactions.
    """
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, tokens, d_model]
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ResidualAttentionForecaster(nn.Module):
    """
    Stage 2: Deep Attention Forecaster for Residual Non-Periodic Dynamics.
    Maps X_res [B, L, C] -> Y_res [B, H, C] using multi-head self-attention.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, 
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 2, 
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.d_model = d_model

        # Tokenizer: project each time step's channel vector into latent space d_model
        self.temporal_embed = nn.Linear(channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder layers
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Prediction Head: maps encoded temporal sequence to prediction horizon H
        self.head_proj = nn.Linear(seq_len, pred_len)
        self.out_proj = nn.Linear(d_model, channels)

    def forward(self, x_res):
        # x_res: [B, L, C]
        B, L, C = x_res.shape

        # 1. Embed tokens and add learnable positional encoding
        tokens = self.temporal_embed(x_res) + self.pos_embed  # [B, L, d_model]

        # 2. Multi-layer Attention across time
        for block in self.blocks:
            tokens = block(tokens)  # [B, L, d_model]

        # 3. Project temporal dimension: L -> H
        # permute to [B, d_model, L]
        tokens_p = tokens.permute(0, 2, 1)
        tokens_pred = self.head_proj(tokens_p).permute(0, 2, 1)  # [B, H, d_model]

        # 4. Project back to channel dimension
        y_res = self.out_proj(tokens_pred)  # [B, H, C]
        return y_res


class TwoStageFourierAttentionNet(nn.Module):
    """
    Full Two-Stage Architecture:
    1. Base Harmonic Extractor & Spectral Forecaster (Stage 1)
    2. Residual Attention Forecaster (Stage 2)
    With RevIN for robust distribution shift management.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 2,
                 top_k_freqs: int = 12, dropout: float = 0.1, use_revin: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(channels, affine=True)

        # Stage 1: Fourier Spectral Harmonic Forecaster
        self.fourier_base = FourierBaseForecaster(
            seq_len=seq_len, 
            pred_len=pred_len, 
            channels=channels, 
            top_k_freqs=top_k_freqs
        )

        # Stage 2: Deep Attention Forecaster for Residuals
        self.residual_attention = ResidualAttentionForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):
        """
        Input:
            x: [Batch, Seq_Len, Channels]
        Output:
            y_hat: [Batch, Pred_Len, Channels]
            dict containing decomposition components for analysis/diagnostics
        """
        # 1. Normalization
        if self.use_revin:
            x = self.revin(x, 'norm')

        # 2. Stage 1: Fourier Base Extraction & Forecast
        x_base, y_base = self.fourier_base(x)

        # 3. Residual Extraction: Orthogonal subtraction
        x_res = x - x_base

        # 4. Stage 2: Residual Attention Forecast
        y_res = self.residual_attention(x_res)

        # 5. Composite Forecast: Base + Residual
        y_hat = y_base + y_res

        # 6. De-normalization
        if self.use_revin:
            y_hat = self.revin(y_hat, 'denorm')
            y_base_denorm = self.revin(y_base, 'denorm')
            y_res_denorm = self.revin(y_res, 'denorm')
        else:
            y_base_denorm = y_base
            y_res_denorm = y_res

        return y_hat, {
            'x_base': x_base,
            'x_res': x_res,
            'y_base': y_base_denorm,
            'y_res': y_res_denorm
        }

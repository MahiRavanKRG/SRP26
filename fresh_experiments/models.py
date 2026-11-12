"""
Fresh Model Implementations for Benchmark Comparison
=====================================================
1. DLinear (Zeng et al., AAAI 2023 baseline with RevIN)
2. CycleNet (Lin et al., NeurIPS 2024 paper architecture: RecurrentCycle queue + Linear residual backbone)
3. TwoStageFourierAttentionNet (Our Proposed Two-Stage Architecture: Fourier Base Forecaster + Residual Attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 0. Shared Normalization: Reversible Instance Normalization (RevIN)
# =====================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., ICLR 2022).
    Defaults to affine=False (pure parameter-free instance norm, as adopted by
    iTransformer and TQNet) to eliminate affine parameter drift on non-stationary series.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str):
        # x: [Batch, Time, Channels]
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


# =====================================================================
# 1. Benchmark 1: DLinear (AAAI 2023)
# =====================================================================
class MovingAverageBlock(nn.Module):
    """Series decomposition block via 1D moving average."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        # Padding front and back to preserve temporal length
        pad_front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        pad_back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([pad_front, x, pad_back], dim=1)
        # avg pool along time: [B, L, C] -> permute to [B, C, L]
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear baseline (Zeng et al., AAAI 2023).
    By default, published DLinear does NOT use RevIN (use_revin=False).
    Can optionally be configured with use_revin=True for DLinear+RevIN comparisons.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, kernel_size: int = 25, use_revin: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.use_revin = use_revin
        if self.use_revin:
            self.revin = RevIN(channels, affine=False)
        self.decom = MovingAverageBlock(kernel_size=kernel_size)

        # Channel-independent mapping L -> H
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, cyc=None):
        # x: [B, L, C]
        if self.use_revin:
            x = self.revin(x, 'norm')
        seasonal, trend = self.decom(x)

        # Map seasonal and trend separately: permute to [B, C, L]
        s_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        t_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        y = s_out + t_out
        if self.use_revin:
            y = self.revin(y, 'denorm')
        return y


# =====================================================================
# 2. Benchmark 2: CycleNet & CycleNet/MLP (NeurIPS 2024 Paper Architecture)
# =====================================================================
class RecurrentCycleQueue(nn.Module):
    """
    Learnable periodic cycle state table theta in R^{W x C}.
    Dynamically gathered via cycle_index mod W.
    """
    def __init__(self, period_w: int, channels: int):
        super().__init__()
        self.period_w = period_w
        self.channels = channels
        # Initialize learnable cycle parameters to zeros (standard CycleNet init)
        self.theta = nn.Parameter(torch.zeros(period_w, channels), requires_grad=True)

    def forward(self, cycle_index: torch.Tensor, length: int) -> torch.Tensor:
        # cycle_index: [Batch]
        device = self.theta.device
        idx = cycle_index.to(device)
        # gather_indices: [Batch, length]
        gather_indices = (idx.unsqueeze(1) + torch.arange(length, device=device).unsqueeze(0)) % self.period_w
        # Returns [Batch, length, Channels]
        return self.theta[gather_indices]


class CycleNet(nn.Module):
    """
    CycleNet as proposed by Lin et al. (NeurIPS 2024):
    1. RevIN normalization.
    2. Subtract in-sample cyclic state theta.
    3. Channel-independent linear backbone on de-cycled residual.
    4. Add future cyclic state theta into forecast horizon.
    5. RevIN denormalization.
    Supports W-ablation: period_w=0 disables RCF queue (Linear backbone alone).
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, period_w: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_w = period_w

        self.revin = RevIN(channels, affine=False)
        if period_w > 0:
            self.cycle_queue = RecurrentCycleQueue(period_w=period_w, channels=channels)
        else:
            self.cycle_queue = None
        self.backbone = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None):
        # 1. Normalization
        x = self.revin(x, 'norm')

        # 2. In-sample cyclic state subtraction (if RCF active)
        if self.period_w > 0 and cycle_index is not None:
            cycle_in = self.cycle_queue(cycle_index, self.seq_len)
            x_res = x - cycle_in
        else:
            x_res = x

        # 3. Residual forecasting (channel-independent Linear)
        y_res = self.backbone(x_res.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Out-of-sample cyclic addition
        if self.period_w > 0 and cycle_index is not None:
            cycle_out = self.cycle_queue(cycle_index + self.seq_len, self.pred_len)
            y = y_res + cycle_out
        else:
            y = y_res

        # 5. Denormalization
        y = self.revin(y, 'denorm')
        return y


class CycleNetMLP(nn.Module):
    """
    CycleNet with MLP backbone (Lin et al., NeurIPS 2024):
    Backbone: Linear(seq_len, d_model) -> GELU -> Linear(d_model, pred_len).
    Supports W-ablation and RCF plug-and-play ablation:
    - period_w=0: MLP backbone alone (no cyclic queue)
    - period_w=24: CycleNet/MLP with RCF
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, period_w: int = 24, d_model: int = 512):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_w = period_w

        self.revin = RevIN(channels, affine=False)
        if period_w > 0:
            self.cycle_queue = RecurrentCycleQueue(period_w=period_w, channels=channels)
        else:
            self.cycle_queue = None

        self.backbone = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None):
        x = self.revin(x, 'norm')

        if self.period_w > 0 and cycle_index is not None:
            cycle_in = self.cycle_queue(cycle_index, self.seq_len)
            x_res = x - cycle_in
        else:
            x_res = x

        y_res = self.backbone(x_res.permute(0, 2, 1)).permute(0, 2, 1)

        if self.period_w > 0 and cycle_index is not None:
            cycle_out = self.cycle_queue(cycle_index + self.seq_len, self.pred_len)
            y = y_res + cycle_out
        else:
            y = y_res

        y = self.revin(y, 'denorm')
        return y


# =====================================================================
# 3. Proposed Architecture: TwoStageFourierAttentionNet
# =====================================================================
class FourierBaseForecaster(nn.Module):
    """
    Stage 1: Spectral Harmonic Forecaster.
    Extracts recurring baseline periodicities using discrete Fourier analysis (rFFT)
    and forecasts the base periodic components in the frequency domain.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.in_freqs = seq_len // 2 + 1
        self.out_freqs = pred_len // 2 + 1

        # Complex projection weights for frequency forecasting: Y_freq = X_freq @ W_spec
        self.spec_proj_real = nn.Parameter(torch.Tensor(channels, self.in_freqs, self.out_freqs))
        self.spec_proj_imag = nn.Parameter(torch.Tensor(channels, self.in_freqs, self.out_freqs))
        
        # In-sample spectral filter to extract recurring baseline frequencies
        self.spectral_filter = nn.Parameter(torch.ones(channels, self.in_freqs))

        # Initialization
        nn.init.kaiming_uniform_(self.spec_proj_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spec_proj_imag, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        orig_dtype = x.dtype
        device_type = x.device.type

        # cuFFT requires powers of 2 in half-precision (FP16); running spectral operations
        # in full precision (FP32) preserves numerical stability and supports arbitrary horizons.
        with torch.autocast(device_type=device_type, enabled=False):
            x_f32 = x.float().permute(0, 2, 1)  # [B, C, L]

            # 1. Temporal Real FFT
            x_fft = torch.fft.rfft(x_f32, dim=-1)  # [B, C, in_freqs] (complex64)

            # 2. In-sample harmonic reconstruction (bounded spectral filter)
            gate = torch.sigmoid(self.spectral_filter.float()).unsqueeze(0)  # [1, C, in_freqs]
            x_base_fft = x_fft * gate
            x_base = torch.fft.irfft(x_base_fft, n=self.seq_len, dim=-1).permute(0, 2, 1)  # [B, L, C]

            # 3. Frequency-domain forecasting via complex matrix multiplication
            x_real = x_base_fft.real
            x_imag = x_base_fft.imag

            # (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
            y_real = torch.einsum('bcf,cfo->bco', x_real, self.spec_proj_real.float()) - \
                     torch.einsum('bcf,cfo->bco', x_imag, self.spec_proj_imag.float())
            y_imag = torch.einsum('bcf,cfo->bco', x_real, self.spec_proj_imag.float()) + \
                     torch.einsum('bcf,cfo->bco', x_imag, self.spec_proj_real.float())

            y_fft = torch.complex(y_real, y_imag)
            y_base = torch.fft.irfft(y_fft, n=self.pred_len, dim=-1).permute(0, 2, 1)  # [B, H, C]

        return x_base.to(orig_dtype), y_base.to(orig_dtype)



class ResidualAttentionBlock(nn.Module):
    """
    Multi-Head Self-Attention block with Pre-LayerNorm (Pre-LN) for superior gradient stability.
    Pre-LN is the academic standard in modern Transformers (PatchTST, LLaMA),
    preventing early gradient explosion or collapse under mixed precision.
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Pre-LayerNorm
        x_norm = self.norm1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # Pre-LayerNorm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x



class ResidualAttentionForecaster(nn.Module):
    """
    Stage 2: Deep Attention Forecaster for Residual Non-Periodic Dynamics.
    Maps X_res [B, L, C] -> Y_res [B, H, C] using multi-head self-attention.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 d_model: int = 64, n_heads: int = 4, num_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        # Project time step channel vector into d_model embedding space
        self.input_proj = nn.Linear(channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Temporal projection L -> H and output channel projection
        self.temporal_head = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(d_model, channels)

    def forward(self, x_res: torch.Tensor):
        # 1. Tokenize: [B, L, C] -> [B, L, d_model]
        tokens = self.input_proj(x_res) + self.pos_emb

        # 2. Self-Attention layers
        for blk in self.blocks:
            tokens = blk(tokens)

        # 3. Forecast horizon projection: [B, d_model, L] -> [B, d_model, H]
        tokens_h = self.temporal_head(tokens.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Project back to channel space: [B, H, C]
        y_res = self.output_proj(tokens_h)
        return y_res


class TwoStageFourierAttentionNet(nn.Module):
    """
    Complete Two-Stage Architecture:
    Stage 1: Fourier Spectral Harmonic Base Forecaster
    Stage 2: Deep Attention Forecaster on Residuals (X - X_base)
    Wrapped in RevIN for distribution shift robustness.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 d_model: int = 64, n_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.revin = RevIN(channels, affine=True)
        self.fourier_base = FourierBaseForecaster(seq_len=seq_len, pred_len=pred_len, channels=channels)
        self.residual_attention = ResidualAttentionForecaster(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            d_model=d_model, n_heads=n_heads, num_layers=num_layers, dropout=dropout
        )

    def forward(self, x: torch.Tensor, cyc=None):
        # 1. RevIN normalization
        x = self.revin(x, 'norm')

        # 2. Stage 1: Fourier Harmonic Base
        x_base, y_base = self.fourier_base(x)

        # 3. Residual isolation (X - X_base)
        x_res = x - x_base

        # 4. Stage 2: Deep Attention on residuals
        y_res = self.residual_attention(x_res)

        # 5. Combine predictions
        y = y_base + y_res

        # 6. RevIN denormalization
        y = self.revin(y, 'denorm')
        return y


# =====================================================================
# 4. SOTA Patch-Based Residual Attention (PatchTST Style)
# =====================================================================
class PatchResidualAttentionForecaster(nn.Module):
    """
    Patch-based Attention Forecaster for Residuals (Nie et al., PatchTST, ICLR 2023).
    Extracts overlapping subseries patches along time, embeds each patch as a token,
    and applies Transformer Multi-Head Self-Attention under Channel Independence.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 patch_len: int = 16, stride: int = 8, d_model: int = 64,
                 n_heads: int = 4, num_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Calculate number of patches: with padding of length (patch_len - stride) at the end
        self.padding_len = patch_len - stride
        total_len = seq_len + self.padding_len
        self.num_patches = (total_len - patch_len) // stride + 1

        # Patch Tokenizer: maps raw patch points (patch_len) to latent embedding (d_model)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Prediction Head: Flatten [num_patches * d_model] -> pred_len
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x_res: torch.Tensor):
        # x_res: [Batch, Seq_Len, Channels]
        B, L, C = x_res.shape

        # Channel Independence: fold batch and channels together -> [B * C, L]
        x_ci = x_res.permute(0, 2, 1).reshape(B * C, L)

        # End padding to preserve edge information
        if self.padding_len > 0:
            last_val = x_ci[:, -1:].repeat(1, self.padding_len)
            x_ci = torch.cat([x_ci, last_val], dim=-1)  # [B * C, L + padding]

        # Unfold into overlapping patches: [B * C, num_patches, patch_len]
        patches = x_ci.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Patch embedding + Positional Encoding
        tokens = self.dropout(self.patch_embed(patches) + self.pos_emb)  # [B * C, num_patches, d_model]

        # Transformer Attention over patches
        for blk in self.blocks:
            tokens = blk(tokens)

        # Flatten patch representations and project to horizon H
        tokens_flat = tokens.reshape(B * C, -1)  # [B * C, num_patches * d_model]
        y_ci = self.head(tokens_flat)            # [B * C, pred_len]

        # Reshape back to [B, pred_len, C]
        y_res = y_ci.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return y_res


class TwoStageFourierPatchAttentionNet(nn.Module):
    """
    SOTA Two-Stage Architecture with Patch-Based Residual Attention:
    Stage 1: Fourier Spectral Harmonic Base Forecaster
    Stage 2: Patch-Based Transformer Multi-Head Self-Attention on Residuals (X - X_base)
    Wrapped in RevIN for distribution shift robustness.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int,
                 patch_len: int = 16, stride: int = 8, d_model: int = 64,
                 n_heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.revin = RevIN(channels, affine=True)
        self.fourier_base = FourierBaseForecaster(seq_len=seq_len, pred_len=pred_len, channels=channels)
        self.residual_patch_attention = PatchResidualAttentionForecaster(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            patch_len=patch_len, stride=stride, d_model=d_model,
            n_heads=n_heads, num_layers=num_layers, dropout=dropout
        )

    def forward(self, x: torch.Tensor, cyc=None):
        # 1. RevIN normalization
        x = self.revin(x, 'norm')

        # 2. Stage 1: Fourier Harmonic Base
        x_base, y_base = self.fourier_base(x)

        # 3. Residual isolation (X - X_base)
        x_res = x - x_base

        # 4. Stage 2: Patch Attention on residuals
        y_res = self.residual_patch_attention(x_res)

        # 5. Combine predictions
        y = y_base + y_res

        # 6. RevIN denormalization
        y = self.revin(y, 'denorm')
        return y


# =====================================================================
# 5. The "Frankenstein" Hybrid: Fourier-TQNet (FTQ-Net)
# =====================================================================
class FourierCrossChannelTQAttention(nn.Module):
    """
    Temporal Query Cross-Channel Attention for Residuals (Frankenstein TQNet x Fourier).
    - Queries Q come from the smooth, in-sample Fourier harmonic base (X_base).
    - Keys K and Values V come from the non-periodic residual (X_res).
    - Computes a C x C inter-variable correlation matrix: Softmax(Q K^T / sqrt(d)) * V.
    - Followed by a 2-layer temporal projection MLP mapping length L -> H.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, n_heads: int = 4, d_mlp: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_heads = n_heads
        self.d_k = seq_len // n_heads

        # Linear projections for Q, K, V across temporal dimension
        self.w_q = nn.Linear(seq_len, seq_len)
        self.w_k = nn.Linear(seq_len, seq_len)
        self.w_v = nn.Linear(seq_len, seq_len)
        self.w_o = nn.Linear(seq_len, seq_len)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(seq_len)

        # Temporal MLP: maps L -> d_mlp -> pred_len H
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, pred_len)
        )

    def forward(self, x_base: torch.Tensor, x_res: torch.Tensor):
        # x_base: [Batch, Seq_Len, Channels] -> permute to [Batch, Channels, Seq_Len]
        # x_res:  [Batch, Seq_Len, Channels] -> permute to [Batch, Channels, Seq_Len]
        B, L, C = x_res.shape

        q_in = x_base.permute(0, 2, 1)  # [B, C, L]
        k_in = x_res.permute(0, 2, 1)   # [B, C, L]
        v_in = x_res.permute(0, 2, 1)   # [B, C, L]

        # Project
        Q = self.w_q(q_in)  # [B, C, L]
        K = self.w_k(k_in)  # [B, C, L]
        V = self.w_v(v_in)  # [B, C, L]

        # Reshape for multi-head attention: [B, C, n_heads, d_k] -> [B, n_heads, C, d_k]
        Q = Q.reshape(B, C, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(B, C, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.reshape(B, C, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Cross-Channel Attention Score: [B, n_heads, C, C]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values: [B, n_heads, C, d_k]
        out = torch.matmul(attn, V)

        # Recombine heads: [B, C, n_heads * d_k] = [B, C, L]
        out = out.permute(0, 2, 1, 3).reshape(B, C, L)
        out = self.w_o(out)

        # Residual shortcut with x_res
        out = self.norm(out + k_in)  # [B, C, L]

        # Temporal MLP forecasting: [B, C, L] -> [B, C, H]
        y_res = self.mlp(out)

        # Permute back to [B, H, C]
        return y_res.permute(0, 2, 1)


class TwoStageFourierCrossAttentionNet(nn.Module):
    """
    Decoupled Fourier Cross-Attention Network:
    Stage 1: Fourier Spectral Harmonic Base Forecaster (Extracts X_base and forecasts Y_base)
    Stage 2: Cross-Channel Attention on Residuals:
             Query Q = X_base (clean global harmonic anchor)
             Keys K, Values V = X_res (transient non-periodic residuals)
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, n_heads: int = 4, d_mlp: int = 128, dropout: float = 0.1):
        super().__init__()
        self.revin = RevIN(channels, affine=True)
        self.fourier_base = FourierBaseForecaster(seq_len=seq_len, pred_len=pred_len, channels=channels)
        self.cross_attention = FourierCrossChannelTQAttention(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            n_heads=n_heads, d_mlp=d_mlp, dropout=dropout
        )

    def forward(self, x: torch.Tensor, cyc=None):
        # 1. Normalization
        x = self.revin(x, 'norm')

        # 2. Stage 1: Fourier Harmonic Base
        x_base, y_base = self.fourier_base(x)

        # 3. Residual isolation
        x_res = x - x_base

        # 4. Stage 2: Cross-Channel Attention
        y_res = self.cross_attention(x_base, x_res)

        # 5. Composite Forecast
        y = y_base + y_res

        # 6. Denormalization
        y = self.revin(y, 'denorm')
        return y


# Backward compatibility alias
TwoStageFourierTQNet = TwoStageFourierCrossAttentionNet


# =====================================================================
# 6. Genuine TQNet Baseline (Lin et al., ICML 2025)
# =====================================================================
class GenuineTQNet(nn.Module):
    """
    Genuine TQNet Baseline (Lin et al., ICML 2025):
    'Temporal Query Network for Efficient Multivariate Time Series Forecasting'
    - Parameter-free Instance Normalization (IN) without learnable affine parameters.
    - Learnable periodic temporal query table theta_tq in R^{C x W}, initialized to zeros.
    - Query slice extracted along (cycle_index + L + h) mod W for prediction horizon H.
    - Keys K and Values V projected from normalized input time series X.
    - Cross-attention matches future periodic queries against historical input dynamics.
    - Channel-independent temporal projection with parameter-free IN denormalization.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int, period_w: int = 24,
                 d_model: int = 64, n_heads: int = 4, d_mlp: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.period_w = period_w
        self.d_model = d_model
        self.n_heads = n_heads

        # 1. Parameter-free Instance Normalization (affine=False)
        self.revin = RevIN(channels, affine=False)

        # 2. Learnable periodic temporal query table theta_tq in R^{C x W}, initialized to zeros
        self.theta_tq = nn.Parameter(torch.zeros(channels, period_w), requires_grad=True)
        nn.init.zeros_(self.theta_tq)

        # 3. Projections for cross-attention
        self.proj_q = nn.Linear(1, d_model)
        self.proj_k = nn.Linear(1, d_model)
        self.proj_v = nn.Linear(1, d_model)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, 1)
        )

        # Residual shortcut from input history
        self.temporal_shortcut = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor = None):
        # x: [Batch, Seq_Len, Channels]
        B, L, C = x.shape
        device = x.device

        # 1. Parameter-free IN normalization
        x_norm = self.revin(x, 'norm')

        # 2. Extract periodic temporal queries from theta_tq
        if cycle_index is None:
            cycle_index = torch.zeros(B, dtype=torch.long, device=device)
        else:
            cycle_index = cycle_index.to(device)

        h_offsets = torch.arange(self.pred_len, device=device).unsqueeze(0)  # [1, H]
        gather_indices = (cycle_index.unsqueeze(1) + self.seq_len + h_offsets) % self.period_w  # [B, H]

        # Slicing theta_tq: [C, W] gathered for each batch sample -> [B, C, H]
        q_periodic = self.theta_tq[:, gather_indices].permute(1, 0, 2)  # [B, C, H]

        # 3. Channel-independent tokenization: fold B and C -> [B*C, H, 1] and [B*C, L, 1]
        q_tokens = self.proj_q(q_periodic.reshape(B * C, self.pred_len, 1))  # [B*C, H, d_model]
        k_tokens = self.proj_k(x_norm.permute(0, 2, 1).reshape(B * C, L, 1))  # [B*C, L, d_model]
        v_tokens = self.proj_v(x_norm.permute(0, 2, 1).reshape(B * C, L, 1))  # [B*C, L, d_model]

        # 4. Cross-Attention
        attn_out, _ = self.mha(q_tokens, k_tokens, v_tokens)
        attn_out = self.norm(attn_out + q_tokens)

        # 5. Output projection + shortcut
        out = self.head(attn_out).squeeze(-1).reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, H, C]
        shortcut = self.temporal_shortcut(x_norm.permute(0, 2, 1)).permute(0, 2, 1)          # [B, H, C]
        y = out + shortcut

        # 6. Parameter-free IN denormalization
        y = self.revin(y, 'denorm')
        return y




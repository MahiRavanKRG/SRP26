"""
CycleNet-Pro — single-file version
===================================
Enhanced CycleNet: FFT-based period detection (instead of a fixed period),
ACF-based true-period vs. noise filtering, RevIN instance normalization,
an automatic multi-cycle vs. single-cycle decision, and channel-dependence
applied only on the residual path (auto-decided from a correlation
diagnostic). Runs on CPU, NVIDIA CUDA, or AMD ROCm without any code changes.

>>> HOW TO USE <<<
Just edit the CONFIG block right below the imports (mainly DATA_PATH and
DATASET_NAME) and run:

    python cyclenet_pro.py

Everything else (FFT period detection, multi/single-cycle decision,
channel-dependence decision, training, evaluation, logging) runs
automatically. Results (MSE/MAE per horizon + average, plus all the
detection/decision details) are written to:

    <OUT_DIR>/<DATASET_NAME>_results.txt
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import find_peaks


# ======================================================================
# 1. CONFIG  --  <<< CHANGE ONLY THIS BLOCK FOR A NEW DATASET/RUN >>>
# ======================================================================
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.environ.get("SRP26_DATA_PATH", os.path.join(_REPO_ROOT, "data_sets", "traffic.csv"))
DATASET_NAME = os.environ.get("SRP26_DATASET_NAME", "traffic")
# used to pick the split convention:
#   name starts with "etth" or "ettm" -> fixed 12/4/4-month calendar split
#   anything else                      -> 70/10/20 ratio split
#   (Electricity, Solar-Energy, Traffic, Weather, or any new dataset)

SEQ_LEN = 336
PRED_LENS = [720]
BATCH_SIZE = 32

EPOCHS = 30
PATIENCE = 5
LR = 1e-3
MODEL_TYPE = "cnn"          # "linear" | "mlp" | "cnn" | "sor"
D_MODEL = 512                  # only used if MODEL_TYPE == "mlp"
CNN_HIDDEN = 32                # only used if MODEL_TYPE == "cnn"
CNN_KERNEL_SIZE = 5            # only used if MODEL_TYPE == "cnn"
CNN_LAYERS = 2                 # only used if MODEL_TYPE == "cnn"

MAX_PERIOD_SEARCH = None        # None -> auto = min(SEQ_LEN, train_len // 2)
N_CANDIDATE_PERIODS = 5

FORCE_MULTI_CYCLE = "auto"      # "auto" | "true" | "false"
FORCE_CHANNEL_DEPENDENCE = "auto"  # "auto" | "true" | "false"

OUT_DIR = "./results"
NUM_WORKERS = 0
SEEDS = [2024]  #, 2025]
# ======================================================================


# ======================================================================
# 2. PERIOD DETECTION: FFT candidates + ACF true-period/noise filter
#    + auto multi-cycle vs. single-cycle decision
# ======================================================================
def find_top_periods_fft(series, n_periods=5, min_period=2, max_period=None):
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if max_period is None:
        max_period = n // 2
    max_period = max(min_period + 1, min(max_period, n // 2))

    series = series - np.mean(series)
    fft_vals = rfft(series)
    freqs = rfftfreq(n)
    power = np.abs(fft_vals) ** 2
    if len(power) > 0:
        power[0] = 0.0  # exclude DC

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid):
        return []

    masked_power = np.where(valid, power, 0.0)
    peaks, _ = find_peaks(masked_power, height=0, distance=1)
    if len(peaks) == 0:
        return []

    peak_powers = power[peaks]
    order = np.argsort(peak_powers)[::-1]
    top_peaks = peaks[order[:n_periods]]

    ranked_periods = []
    for p in top_peaks:
        if freqs[p] <= 0:
            continue
        period = int(round(1.0 / freqs[p]))
        if min_period <= period <= max_period:
            ranked_periods.append(period)

    # Merge near-duplicate periods from spectral leakage (e.g. 23/24/26
    # around a true period-24 signal are the same underlying cycle).
    periods = []
    for period in ranked_periods:
        if period in periods:
            continue
        is_dup = any(abs(period - kept) <= max(2, 0.12 * kept) for kept in periods)
        if not is_dup:
            periods.append(period)
    return periods


def acf_at_lag(series, lag):
    series = np.asarray(series, dtype=np.float64)
    series = series - np.mean(series)
    n = len(series)
    if lag <= 0 or lag >= n:
        return 0.0
    num = np.sum(series[:n - lag] * series[lag:])
    den = np.sum(series ** 2)
    if den == 0:
        return 0.0
    return float(num / den)


def filter_true_periods_acf(series, candidate_periods, conf_multiplier=1.96):
    """Keep only candidate periods whose ACF exceeds the white-noise 95%
    confidence bound 1.96/sqrt(N); everything else is treated as noise."""
    n = len(series)
    bound = conf_multiplier / np.sqrt(max(n, 1))
    true_periods, noise_periods, acf_scores = [], [], {}
    for p in candidate_periods:
        score = acf_at_lag(series, p)
        acf_scores[p] = round(score, 4)
        if abs(score) > bound:
            true_periods.append(p)
        else:
            noise_periods.append(p)
    true_periods.sort(key=lambda p: -abs(acf_scores[p]))
    return true_periods, noise_periods, acf_scores


def _reconstruct_from_periods(series_centered, n, periods, n_harmonics=20):
    fft_vals = rfft(series_centered)
    freqs = rfftfreq(n)
    mask = np.zeros_like(fft_vals, dtype=bool)
    for P in periods:
        if P <= 0:
            continue
        f0 = 1.0 / P
        for k in range(1, n_harmonics + 1):
            idx = np.argmin(np.abs(freqs - k * f0))
            if idx < len(mask):
                mask[idx] = True
    filtered = fft_vals.copy()
    filtered[~mask] = 0
    return irfft(filtered, n=n)


def decide_multi_or_single_cycle(series, true_periods, n_harmonics=20,
                                  gain_threshold_pct=2.0, max_periods=3):
    """Progressively add confirmed periods (strongest first); recommend
    multi-cycle only if an extra period buys >= gain_threshold_pct more
    variance reduction."""
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    series_c = series - np.mean(series)
    total_var = np.var(series_c)

    if total_var == 0 or len(true_periods) == 0:
        return {'use_multi_cycle': False, 'periods_to_use': true_periods[:1], 'details': {}}

    periods_ranked = true_periods[:max_periods]
    details = {}
    reductions = []
    for k in range(1, len(periods_ranked) + 1):
        used = periods_ranked[:k]
        recon = _reconstruct_from_periods(series_c, n, used, n_harmonics=n_harmonics)
        residual_var = np.var(series_c - recon)
        reduction = float((total_var - residual_var) / total_var * 100)
        details[f'top_{k}'] = {'periods': used, 'reduction_pct': round(reduction, 3)}
        reductions.append(reduction)

    use_multi = False
    best_k = 1
    for k in range(1, len(reductions)):
        gain = reductions[k] - reductions[k - 1]
        if gain >= gain_threshold_pct:
            use_multi = True
            best_k = k + 1
    periods_to_use = periods_ranked[:best_k]

    return {'use_multi_cycle': use_multi, 'periods_to_use': periods_to_use, 'details': details}


# ======================================================================
# 3. CHANNEL-DEPENDENCE DIAGNOSTIC (decides residual-path mixing only)
# ======================================================================
def compute_correlation_metrics(data, max_lag=50, max_pairs=300, max_samples_for_lag=3000):
    data = np.asarray(data, dtype=np.float64)
    n_samples, n_channels = data.shape
    if n_channels < 2 or n_samples < 4:
        return {'avg_abs_pearson': 0.0, 'avg_max_abs_corr': 0.0,
                'frac_max_corr_gt_05': 0.0, 'num_channels': n_channels}

    pearson_corr = np.corrcoef(data.T)
    iu = np.triu_indices(n_channels, k=1)
    pearson_vals = pearson_corr[iu]
    pearson_vals = pearson_vals[~np.isnan(pearson_vals)]
    avg_abs_pearson = float(np.mean(np.abs(pearson_vals))) if len(pearson_vals) else 0.0

    pairs = list(zip(*iu))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    if n_samples > max_samples_for_lag:
        start = (n_samples - max_samples_for_lag) // 2
        lag_data = data[start:start + max_samples_for_lag]
    else:
        lag_data = data
    n_lag_samples = lag_data.shape[0]
    eff_max_lag = min(max_lag, max(n_lag_samples // 4, 1))
    lag_step = max(1, eff_max_lag // 10)

    max_corr_vals = []
    for i, j in pairs:
        x = lag_data[:, i] - np.mean(lag_data[:, i])
        y = lag_data[:, j] - np.mean(lag_data[:, j])
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        best = 0.0
        for lag in range(-eff_max_lag, eff_max_lag + 1, lag_step):
            if lag >= 0:
                xs, ys = x[:n_lag_samples - lag], y[lag:]
            else:
                xs, ys = x[-lag:], y[:n_lag_samples + lag]
            if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
                continue
            c = np.corrcoef(xs, ys)[0, 1]
            if not np.isnan(c) and abs(c) > abs(best):
                best = c
        max_corr_vals.append(abs(best))

    avg_max_abs_corr = float(np.mean(max_corr_vals)) if max_corr_vals else 0.0
    frac_gt_05 = float(np.mean(np.array(max_corr_vals) > 0.5)) if max_corr_vals else 0.0

    return {
        'avg_abs_pearson': round(avg_abs_pearson, 4),
        'avg_max_abs_corr': round(avg_max_abs_corr, 4),
        'frac_max_corr_gt_05': round(frac_gt_05, 4),
        'num_channels': n_channels,
    }


def decide_channel_dependence(metrics):
    p = metrics['avg_abs_pearson']
    m = metrics['avg_max_abs_corr']
    f = metrics['frac_max_corr_gt_05']

    if metrics.get('num_channels', 2) < 2:
        return False, "Single-channel dataset - channel dependence is not applicable."
    if p < 0.3 and m < 0.3:
        return False, ("Low channel dependence (avg|Pearson|={:.3f}, avg max|xcorr|={:.3f}) - "
                        "residual path kept channel-independent.").format(p, m)
    elif p < 0.5 and m < 0.5 and f < 0.2:
        return False, ("Moderate channel dependence (avg|Pearson|={:.3f}, avg max|xcorr|={:.3f}, "
                        "frac>0.5={:.3f}) - kept channel-independent by default (borderline case, "
                        "set FORCE_CHANNEL_DEPENDENCE='true' to override).").format(p, m, f)
    else:
        return True, ("High channel dependence (avg|Pearson|={:.3f}, avg max|xcorr|={:.3f}, "
                       "frac>0.5={:.3f}) - enabling a channel-mixing layer on the residual path.").format(p, m, f)


# ======================================================================
# 4. DATA LOADING: plug-and-play CSV/TXT + splitting + scaling + windowing
# ======================================================================
def _first_row_is_all_numeric(path):
    """Peek at the first non-blank line; if every field parses as a float,
    the file has no header row (common for LTSF .txt dataset mirrors)."""
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                break
        else:
            return False
    for sep in (',', '\t', ';', ' '):
        fields = [t for t in line.split(sep) if t != '']
        if len(fields) > 1:
            break
    else:
        fields = [line]
    try:
        [float(t) for t in fields]
        return True
    except ValueError:
        return False


def load_raw_dataframe(path):
    if path.lower().endswith('.txt'):
        headerless = _first_row_is_all_numeric(path)
        df = pd.read_csv(path, sep=None, engine='python', header=None if headerless else 0)
    else:
        df = pd.read_csv(path, sep=None, engine='python')

    drop_cols = [c for c in df.columns if str(c).strip().lower() in
                 ('date', 'timestamp', 'time', 'datetime', 'index', 'unnamed: 0')]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path} after dropping date/index columns.")
    return df.values.astype(np.float32)


def get_splits(n, dataset_name):
    name = (dataset_name or '').lower()
    if name.startswith('etth'):
        train_end = min(12 * 30 * 24, n)
        val_end = min(train_end + 4 * 30 * 24, n)
        test_end = min(val_end + 4 * 30 * 24, n)
        return (0, train_end), (train_end, val_end), (val_end, test_end)
    elif name.startswith('ettm'):
        train_end = min(12 * 30 * 24 * 4, n)
        val_end = min(train_end + 4 * 30 * 24 * 4, n)
        test_end = min(val_end + 4 * 30 * 24 * 4, n)
        return (0, train_end), (train_end, val_end), (val_end, test_end)
    else:
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        return (0, train_end), (train_end, val_end), (val_end, n)


class Scaler:
    """Per-channel standardization fit on the training split only."""
    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class WindowDataset(Dataset):
    """Sliding-window (seq_len -> pred_len) dataset. start_offset is the
    absolute row index (in the full series) of data[0], so the cycle phase
    stays consistent across train/val/test splits."""
    def __init__(self, data, seq_len, pred_len, start_offset):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start_offset = start_offset
        self.n = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return max(self.n, 0)

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        x = self.data[s_begin:s_end]
        y = self.data[s_end:r_end]
        cycle_index = self.start_offset + s_begin
        return (torch.from_numpy(np.ascontiguousarray(x)).float(),
                torch.from_numpy(np.ascontiguousarray(y)).float(),
                torch.tensor(cycle_index, dtype=torch.long))


# ======================================================================
# 5. MODEL: RevIN + RecurrentCycle(s) + CI backbone + optional residual mix
# ======================================================================
class RevIN(nn.Module):
    """Reversible Instance Normalization (per-sample, per-channel)."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.stdev = None

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
            x = x * self.stdev + self.mean
            return x
        else:
            raise ValueError(f"Unknown RevIN mode: {mode}")


class RecurrentCycle(nn.Module):
    """Learnable per-phase cyclic component (table) for one detected period,
    indexed by absolute position mod cycle_len."""
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        device = self.data.device
        idx = index.to(device)
        gather_index = (idx.unsqueeze(1) + torch.arange(length, device=device).unsqueeze(0)) % self.cycle_len
        return self.data[gather_index]


class ChannelMixer(nn.Module):
    """Optional cross-channel mixing applied on the residual/output path only."""
    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or max(channels // 2, 4)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        return x + self.net(x)  # residual connection so it can learn to be a no-op

class SORMixer(nn.Module):
    """Spectral + Overlap Residual Mixer (minimal version)."""
    def __init__(self, seq_len, pred_len, channels,
                 patch_len=16, stride=8, d_model=64, n_bands=2,
                 dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model
        self.n_bands = n_bands

        # Pre-computed orthonormal DCT-II matrix (works without torch.fft.dct)
        self.register_buffer('dct_basis', self._dct_matrix(seq_len))

        # --- DCT branch ---
        self.band_gate = nn.Parameter(torch.ones(n_bands))
        self.proj_dct = nn.Sequential(
            nn.Linear(seq_len, 64),
            nn.GELU(),
            nn.Linear(64, pred_len),
        )

        # --- Overlap MLP branch ---
        self.embed = nn.Linear(patch_len, d_model)
        self.local_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.candidate = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.dropout = nn.Dropout(dropout)

        # Softmax pooling over patches, then project to horizon
        self.patch_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),
        )
        self.proj_local = nn.Linear(d_model, pred_len)

        # --- Fusion ---
        self.fuse = nn.Parameter(torch.ones(2))
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _dct_matrix(L):
        """Orthonormal DCT-II matrix, shape (L, L).
        Y = C @ x  gives the same result as torch.fft.dct(x, type=2, norm='ortho')."""
        n = torch.arange(L, dtype=torch.float32)
        k = torch.arange(L, dtype=torch.float32).unsqueeze(1)   # (L, 1)
        C = torch.cos(torch.pi / L * (n + 0.5) * k)            # (L, L)
        C[0] = C[0] * (1.0 / torch.sqrt(torch.tensor(2.0)))
        C = C * torch.sqrt(torch.tensor(2.0 / L))
        return C

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape

        # DCT branch (manual orthonormal DCT-II via matmul)
        D = self.dct_basis                                     # (L, L)
        F = torch.einsum('lk,blc->bkc', D, x)                  # (B, L, C)

        # Split the DCT axis into n_bands chunks; pad if L not divisible.
        bands = list(torch.chunk(F, self.n_bands, dim=1))
        while len(bands) < self.n_bands:
            bands.append(torch.zeros_like(bands[0]))
        bands = bands[:self.n_bands]

        F = torch.cat([b * self.band_gate[i] for i, b in enumerate(bands)], dim=1)
        x_dct = torch.einsum('kl,bkc->blc', D, F)              # (B, L, C)
        y_dct = self.proj_dct(x_dct.permute(0, 2, 1)).permute(0, 2, 1)  # (B, H, C)

        # Overlap MLP branch
        x_p = x.permute(0, 2, 1)                               # (B, C, L)
        patches = x_p.unfold(-1, self.patch_len, self.stride)  # (B, C, N, P)
        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)
        e = self.dropout(self.embed(patches))                   # (B*C, N, D)

        e_pad = torch.cat([e[:, :1], e, e[:, -1:]], dim=1)     # (B*C, N+2, D)
        z = torch.cat([e_pad[:, :-2], e_pad[:, 1:-1], e_pad[:, 2:]], dim=-1)
        e_local = e + self.local_mlp(z)
        g = self.gate(e_local)
        u = self.candidate(e_local)
        e_local = g * u + (1 - g) * e_local
        e_local = self.dropout(e_local)

        # Softmax-pool patches: (B*C, N, D) -> (B*C, D)
        w = self.patch_pool(e_local)                 # (B*C, N, 1)
        pooled = (e_local * w).sum(dim=1)            # (B*C, D)
        pooled = self.dropout(pooled)
        y_local = self.proj_local(pooled).reshape(B, C, -1).permute(0, 2, 1)

        # Fusion
        y = self.residual_scale * (self.fuse[0] * y_dct + self.fuse[1] * y_local)
        return y

class LinearBackbone(nn.Module):
    """Original CycleNet backbone: one big weight matrix, no parameter
    sharing across time steps. Params ~= seq_len * pred_len."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # (B, C, seq_len) -> (B, C, pred_len)
        return self.linear(x)


class MLPBackbone(nn.Module):
    """Two-layer MLP. Still no weight sharing across time steps."""
    def __init__(self, seq_len, pred_len, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_model), nn.GELU(), nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        return self.net(x)


class CNNBackbone(nn.Module):
    """Fully-convolutional backbone, parameter-shared across both time
    positions and channels. seq_len -> pred_len length change done with
    parameter-free linear interpolation."""
    def __init__(self, seq_len, pred_len, hidden=32, kernel_size=5, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        pad = kernel_size // 2
        layers = []
        in_ch = 1
        for _ in range(num_layers):
            layers += [nn.Conv1d(in_ch, hidden, kernel_size, padding=pad), nn.GELU()]
            in_ch = hidden
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x):  # (B, C, seq_len) -> (B, C, pred_len)
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)
        h = self.encoder(x)
        h = F.interpolate(h, size=self.pred_len, mode='linear', align_corners=False)
        out = self.head(h)
        return out.reshape(B, C, self.pred_len)


def build_backbone(model_type, seq_len, pred_len, enc_in, d_model=512,
                   cnn_hidden=32, cnn_kernel_size=5, cnn_layers=2):
    if model_type == 'linear':
        return LinearBackbone(seq_len, pred_len)
    elif model_type == 'mlp':
        return MLPBackbone(seq_len, pred_len, d_model=d_model)
    elif model_type == 'cnn':
        return CNNBackbone(seq_len, pred_len, hidden=cnn_hidden,
                           kernel_size=cnn_kernel_size, num_layers=cnn_layers)
    elif model_type == 'sor':
        return SORMixer(seq_len, pred_len, enc_in)
    else:
        raise ValueError(f"Unknown model_type: {model_type} (expected "
                         f"'linear', 'mlp', 'cnn', or 'sor')")

class CycleNetPro(nn.Module):
    """
    - RevIN on the input window, reversed on the output.
    - One or more RecurrentCycle tables (single-cycle == original CycleNet;
      multi-cycle sums contributions of several FFT+ACF confirmed periods).
    - Channel-independent linear/MLP backbone maps the de-cycled residual
      seq_len -> pred_len (weights shared across channels).
    - Optional ChannelMixer on the residual/output path only, used only
      when the channel-dependence diagnostic says channels are correlated.
    """
    def __init__(self, seq_len, pred_len, enc_in, periods,
                use_channel_dependence=False, model_type='linear',
                d_model=512, cnn_hidden=32, cnn_kernel_size=5, cnn_layers=2,
                use_revin=True):
        super().__init__()
        assert len(periods) >= 1, "At least one period must be provided."
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.periods = list(periods)
        self.use_revin = use_revin
        self.use_channel_dependence = use_channel_dependence
        self.model_type = model_type

        if use_revin:
            self.revin = RevIN(enc_in, affine=True)

        self.cycle_queues = nn.ModuleList([RecurrentCycle(p, enc_in) for p in self.periods])


        self.backbone = build_backbone(
            model_type, seq_len, pred_len, enc_in,
            d_model=d_model,
            cnn_hidden=cnn_hidden,
            cnn_kernel_size=cnn_kernel_size,
            cnn_layers=cnn_layers,
        )
            
        if use_channel_dependence:
            self.channel_mixer = ChannelMixer(enc_in)

    def forward(self, x, cycle_index):
        if self.use_revin:
            x = self.revin(x, 'norm')

        x_res = x
        for q in self.cycle_queues:
            x_res = x_res - q(cycle_index, self.seq_len)

        if self.model_type == 'sor':
            y_res = self.backbone(x_res)                       # (B, L, C) -> (B, H, C)
        else:
            y_res = self.backbone(x_res.permute(0, 2, 1)).permute(0, 2, 1)  

        y = y_res
        for q in self.cycle_queues:
            y = y + q(cycle_index + self.seq_len, self.pred_len)

        if self.use_channel_dependence:
            y = self.channel_mixer(y)

        if self.use_revin:
            y = self.revin(y, 'denorm')

        return y


# ======================================================================
# 6. TRAINING ENGINE: robust device selection, early stopping, loops
# ======================================================================
def get_device():
    """NVIDIA CUDA and AMD ROCm builds of PyTorch both report through the
    same torch.cuda.* API, so this transparently covers both; else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, loss, model):
        if self.best_loss is None or loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    criterion = nn.MSELoss()
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, cyc in loader:
            x, y, cyc = x.to(device), y.to(device), cyc.to(device)
            out = model(x, cyc)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


def evaluate_metrics(model, loader, device):
    model.eval()
    sq_err_sum, abs_err_sum, n_elems = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, cyc in loader:
            x, y, cyc = x.to(device), y.to(device), cyc.to(device)
            out = model(x, cyc)
            sq_err_sum += torch.sum((out - y) ** 2).item()
            abs_err_sum += torch.sum(torch.abs(out - y)).item()
            n_elems += y.numel()
    mse = sq_err_sum / max(n_elems, 1)
    mae = abs_err_sum / max(n_elems, 1)
    return mse, mae


# ======================================================================
# 7. MAIN
# ======================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()

    # ---------------- Load + split + scale (once, outside seed loop) ----------------
    raw = load_raw_dataframe(DATA_PATH)
    n_total, n_channels = raw.shape
    (tr0, tr1), (v0, v1), (te0, te1) = get_splits(n_total, DATASET_NAME)

    train_raw = raw[tr0:tr1]
    scaler = Scaler().fit(train_raw)
    scaled = scaler.transform(raw)

    # ---------------- FFT period detection + ACF noise filtering ----------------
    mean_series = train_raw.mean(axis=1)
    max_period = MAX_PERIOD_SEARCH or min(SEQ_LEN, max((tr1 - tr0) // 2, 4))
    candidates = find_top_periods_fft(mean_series, n_periods=N_CANDIDATE_PERIODS,
                                       min_period=2, max_period=max_period)
    true_periods, noise_periods, acf_scores = filter_true_periods_acf(mean_series, candidates)
    if not true_periods:
        true_periods = candidates[:1] if candidates else [24]

    cycle_decision = decide_multi_or_single_cycle(mean_series, true_periods)

    if FORCE_MULTI_CYCLE == 'true':
        use_multi = True
        k = min(3, len(true_periods)) if len(true_periods) > 1 else 1
        periods_to_use = true_periods[:max(k, 1)]
    elif FORCE_MULTI_CYCLE == 'false':
        use_multi = False
        periods_to_use = true_periods[:1]
    else:
        use_multi = cycle_decision['use_multi_cycle']
        periods_to_use = cycle_decision['periods_to_use']
    if not periods_to_use:
        periods_to_use = [24]

    # ---------------- Channel-dependence decision ----------------
    corr_metrics = compute_correlation_metrics(train_raw)
    auto_dep, dep_reason = decide_channel_dependence(corr_metrics)
    if FORCE_CHANNEL_DEPENDENCE == 'true':
        use_channel_dep = True
    elif FORCE_CHANNEL_DEPENDENCE == 'false':
        use_channel_dep = False
    else:
        use_channel_dep = auto_dep

    # ---------------- Log header ----------------
    log_lines = []
    log_lines.append(f"Dataset            : {DATASET_NAME}")
    log_lines.append(f"Data path          : {DATA_PATH}")
    log_lines.append(f"Shape (T x C)      : {raw.shape}")
    log_lines.append(f"Device             : {device}")
    log_lines.append(f"Splits (rows)      : train=[{tr0}:{tr1}) val=[{v0}:{v1}) test=[{te0}:{te1})")
    log_lines.append("")
    log_lines.append("--- FFT period detection ---")
    log_lines.append(f"FFT candidate periods           : {candidates}")
    log_lines.append(f"ACF-confirmed true periods       : {true_periods}")
    log_lines.append(f"ACF-rejected periods (noise)     : {noise_periods}")
    log_lines.append(f"ACF scores                       : {acf_scores}")
    log_lines.append("")
    log_lines.append("--- Multi-cycle vs single-cycle decision ---")
    log_lines.append(f"Variance-reduction detail        : {cycle_decision['details']}")
    log_lines.append(f"Decision                         : {'MULTI-CYCLE' if use_multi else 'SINGLE-CYCLE'} "
                      f"(mode={FORCE_MULTI_CYCLE})")
    log_lines.append(f"Periods used by model             : {periods_to_use}")
    log_lines.append("")
    log_lines.append("--- Channel-dependence decision (residual path only) ---")
    log_lines.append(f"Correlation metrics               : {corr_metrics}")
    log_lines.append(f"Decision                          : "
                      f"{'CHANNEL-DEPENDENT residual' if use_channel_dep else 'CHANNEL-INDEPENDENT residual'} "
                      f"(mode={FORCE_CHANNEL_DEPENDENCE})")
    log_lines.append(f"Reason                             : {dep_reason}")
    log_lines.append("")
    log_lines.append(f"Config : seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}, "
                      f"model_type={MODEL_TYPE}, epochs(max)={EPOCHS}, "
                      f"patience={PATIENCE}, lr={LR}")
    log_lines.append(f"Seeds  : {SEEDS}")
    log_lines.append("")

    print("\n".join(log_lines))

    # ---------------- Multi-seed evaluation ----------------
    seed_avg_mse = {}
    seed_avg_mae = {}

    for seed in SEEDS:
        set_seed(seed)
        seed_header = f"\n{'#' * 90}\n### SEED = {seed}\n{'#' * 90}"
        print(seed_header)
        log_lines.append(seed_header)
        log_lines.append(f"{'pred_len':>10} | {'MSE':>10} | {'MAE':>10}")
        log_lines.append("-" * 90)

        this_seed_mse = []
        this_seed_mae = []

        for pred_len in PRED_LENS:
            train_ds = WindowDataset(scaled[tr0:tr1], SEQ_LEN, pred_len, start_offset=tr0)
            val_start = max(v0 - SEQ_LEN, 0)
            val_ds = WindowDataset(scaled[val_start:v1], SEQ_LEN, pred_len, start_offset=val_start)
            test_start = max(te0 - SEQ_LEN, 0)
            test_ds = WindowDataset(scaled[test_start:te1], SEQ_LEN, pred_len, start_offset=test_start)

            if len(train_ds) == 0 or len(test_ds) == 0:
                line = f"{pred_len:>10} | SKIPPED (not enough rows)"
                print(line)
                log_lines.append(line)
                continue

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS) if len(val_ds) > 0 else None
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS)

            model = CycleNetPro(
                seq_len=SEQ_LEN, pred_len=pred_len, enc_in=n_channels,
                periods=periods_to_use, use_channel_dependence=use_channel_dep,
                model_type=MODEL_TYPE, d_model=D_MODEL,
                cnn_hidden=CNN_HIDDEN, cnn_kernel_size=CNN_KERNEL_SIZE,
                cnn_layers=CNN_LAYERS,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            stopper = EarlyStopping(patience=PATIENCE)

            for epoch in range(EPOCHS):
                train_loss = run_epoch(model, train_loader, device, optimizer)
                eval_loader = val_loader if val_loader is not None else train_loader
                val_loss = run_epoch(model, eval_loader, device, optimizer=None)
                stopper.step(val_loss, model)
                print(f"  [seed={seed} | pred_len={pred_len}] epoch {epoch+1}: "
                      f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
                if stopper.should_stop:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            stopper.restore(model)
            mse, mae = evaluate_metrics(model, test_loader, device)
            this_seed_mse.append(mse)
            this_seed_mae.append(mae)

            line = f"{pred_len:>10} | {mse:>10.4f} | {mae:>10.4f}"
            print(line)
            log_lines.append(line)

            ckpt_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_seed{seed}_pred{pred_len}.pt")
            torch.save(model.state_dict(), ckpt_path)

        if this_seed_mse:
            seed_avg_mse[seed] = float(np.mean(this_seed_mse))
            seed_avg_mae[seed] = float(np.mean(this_seed_mae))
            log_lines.append(f"{'SEED AVG':>10} | "
                              f"{seed_avg_mse[seed]:>10.4f} | {seed_avg_mae[seed]:>10.4f}")

    # ---------------- Final: average across seeds ----------------
    final_header = f"\n{'=' * 90}\nFINAL: AVERAGE ACROSS SEEDS {SEEDS}\n{'=' * 90}"
    print(final_header)
    log_lines.append(final_header)
    for seed in SEEDS:
        if seed in seed_avg_mse:
            log_lines.append(f"{seed:>10} | {seed_avg_mse[seed]:>10.4f} | {seed_avg_mae[seed]:>10.4f}")

    if seed_avg_mse:
        final_mse = float(np.mean(list(seed_avg_mse.values())))
        final_mae = float(np.mean(list(seed_avg_mae.values())))
        final_line = f"{'FINAL AVG':>10} | {final_mse:>10.4f} | {final_mae:>10.4f}"
        print(final_line)
        log_lines.append(final_line)

    out_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_results.txt")
    with open(out_path, 'w') as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nSaved results to: {out_path}")


if __name__ == '__main__':
    main()

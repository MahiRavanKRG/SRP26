"""
CycleNet-Pro — single-file version
===================================
Enhanced CycleNet: FFT-based period detection (instead of a fixed period),
ACF-based true-period vs. noise filtering, RevIN instance normalization,
an automatic multi-cycle vs. single-cycle decision, and a swappable
channel-independent backbone (linear / mlp / cnn - change ONE variable to
switch). Runs on CPU, NVIDIA CUDA, or AMD ROCm without any code changes.

>>> HOW TO USE <<<
Just edit the CONFIG block right below the imports (mainly DATA_PATH and
DATASET_NAME) and run:

    python cyclenet_pro.py

To switch backbones, change one line:

    MODEL_TYPE = "linear"   # or "mlp", or "cnn"

Everything else (FFT period detection, multi/single-cycle decision,
training, evaluation, logging) runs automatically. Results (MSE/MAE per
horizon + average, plus all the detection/decision details) are written to:

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
DATA_PATH = os.environ.get("SRP26_DATA_PATH", os.path.join(_REPO_ROOT, "data_sets", "solar_AL.txt"))
DATASET_NAME = os.environ.get("SRP26_DATASET_NAME", "solar_AL")
# used to pick the split convention:
#   name starts with "etth" or "ettm" -> fixed 12/4/4-month calendar split
#   anything else                      -> 70/10/20 ratio split
#   (Electricity, Solar-Energy, Traffic, Weather, or any new dataset)

SEQ_LEN = 336
PRED_LENS = [96, 192, 336, 720]
BATCH_SIZE = 32

EPOCHS = 30
PATIENCE = 5
LR = 1e-3

# ---- Backbone switch: change ONLY this one variable ----
MODEL_TYPE = "cnn"          # "linear" | "mlp" | "cnn"
D_MODEL = 512                  # only used if MODEL_TYPE == "mlp"
CNN_HIDDEN = 32                # only used if MODEL_TYPE == "cnn"
CNN_KERNEL_SIZE = 5            # only used if MODEL_TYPE == "cnn"
CNN_LAYERS = 2                 # only used if MODEL_TYPE == "cnn"

MAX_PERIOD_SEARCH = None        # None -> auto = min(SEQ_LEN, train_len // 2)
N_CANDIDATE_PERIODS = 5

FORCE_MULTI_CYCLE = "auto"      # "auto" | "true" | "false"

OUT_DIR = "./results"
NUM_WORKERS = 0

# 2024 is one of the exact seeds the official CycleNet paper/repo
# (ACAT-SCUT/CycleNet) uses when reporting results.
SEED = 2024
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
# 3. DATA LOADING: plug-and-play CSV/TXT + splitting + scaling + windowing
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
# 5. MODEL: RevIN + RecurrentCycle(s) + swappable CI backbone
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


# ---------------- Swappable backbones (all channel-independent) ----------------
# All three share the same interface: forward(x) where x has shape
# (B, C, seq_len) and the output has shape (B, C, pred_len). Because of
# this shared interface, CycleNetPro's forward() never needs to change
# when you switch MODEL_TYPE - only which backbone class gets built.

class LinearBackbone(nn.Module):
    """Original CycleNet backbone: one big weight matrix, no parameter
    sharing across time steps. Params ~= seq_len * pred_len."""
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # (B, C, seq_len)
        return self.linear(x)  # nn.Linear applies over the last dim regardless
                                # of leading dims -> (B, C, pred_len)


class MLPBackbone(nn.Module):
    """Two-layer MLP. Still no weight sharing across time steps; typically
    has MORE parameters than the linear backbone, not fewer."""
    def __init__(self, seq_len, pred_len, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_model), nn.GELU(), nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        return self.net(x)


class CNNBackbone(nn.Module):
    """
    Fully-convolutional backbone built for parameter sharing + a much
    smaller weight count than the linear/mlp backbones:

      - Every Conv1d kernel is reused at every time position (that's the
        parameter-sharing part) AND across every channel of the series,
        since (B, C, seq_len) is reshaped to (B*C, 1, seq_len) before the
        convolutions run - the same small kernel processes every channel.
      - Kernel size (not seq_len) sets the parameter count, so this stays
        small even for seq_len=336 or pred_len=720:
          e.g. hidden=32, kernel=5, 2 layers -> a few thousand parameters
          total, vs. seq_len*pred_len (~240,000+) for the linear backbone.
      - seq_len -> pred_len length change is done with a parameter-free
        linear interpolation (F.interpolate), not a dense matrix.
    """
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
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)  # shared 1x1 projection back to 1 channel

    def forward(self, x):  # x: (B, C, seq_len)
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)                 # same conv weights applied to every channel
        h = self.encoder(x)                          # (B*C, hidden, L)  (length preserved, 'same' padding)
        h = F.interpolate(h, size=self.pred_len, mode='linear', align_corners=False)  # (B*C, hidden, pred_len)
        out = self.head(h)                            # (B*C, 1, pred_len)
        return out.reshape(B, C, self.pred_len)


def build_backbone(model_type, seq_len, pred_len, d_model=512,
                    cnn_hidden=32, cnn_kernel_size=5, cnn_layers=2):
    if model_type == 'linear':
        return LinearBackbone(seq_len, pred_len)
    elif model_type == 'mlp':
        return MLPBackbone(seq_len, pred_len, d_model=d_model)
    elif model_type == 'cnn':
        return CNNBackbone(seq_len, pred_len, hidden=cnn_hidden,
                            kernel_size=cnn_kernel_size, num_layers=cnn_layers)
    else:
        raise ValueError(f"Unknown model_type: {model_type} (expected 'linear', 'mlp', or 'cnn')")


class CycleNetPro(nn.Module):
    """
    - RevIN on the input window, reversed on the output.
    - One or more RecurrentCycle tables (single-cycle == original CycleNet;
      multi-cycle sums contributions of several FFT+ACF confirmed periods).
    - A channel-independent backbone maps the de-cycled residual
      seq_len -> pred_len; swap MODEL_TYPE ('linear' | 'mlp' | 'cnn') in
      the CONFIG block to change it - forward() below never changes.
    """
    def __init__(self, seq_len, pred_len, enc_in, periods,
                 model_type='linear', d_model=512,
                 cnn_hidden=32, cnn_kernel_size=5, cnn_layers=2,
                 use_revin=True):
        super().__init__()
        assert len(periods) >= 1, "At least one period must be provided."
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.periods = list(periods)
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(enc_in, affine=True)

        self.cycle_queues = nn.ModuleList([RecurrentCycle(p, enc_in) for p in self.periods])

        self.backbone = build_backbone(model_type, seq_len, pred_len, d_model=d_model,
                                        cnn_hidden=cnn_hidden, cnn_kernel_size=cnn_kernel_size,
                                        cnn_layers=cnn_layers)

    def forward(self, x, cycle_index):
        if self.use_revin:
            x = self.revin(x, 'norm')

        x_res = x
        for q in self.cycle_queues:
            x_res = x_res - q(cycle_index, self.seq_len)

        y_res = self.backbone(x_res.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, C)

        y = y_res
        for q in self.cycle_queues:
            y = y + q(cycle_index + self.seq_len, self.pred_len)

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
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()

    # ---------------- Load + split + scale ----------------
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
    backbone_desc = f"model_type={MODEL_TYPE}"
    if MODEL_TYPE == 'mlp':
        backbone_desc += f", d_model={D_MODEL}"
    elif MODEL_TYPE == 'cnn':
        backbone_desc += (f", cnn_hidden={CNN_HIDDEN}, cnn_kernel_size={CNN_KERNEL_SIZE}, "
                           f"cnn_layers={CNN_LAYERS}")
    log_lines.append(f"Config : seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}, {backbone_desc}, "
                      f"epochs(max)={EPOCHS}, patience={PATIENCE}, lr={LR}, seed={SEED}")
    log_lines.append("")
    log_lines.append("=" * 90)
    log_lines.append(f"{'pred_len':>10} | {'MSE':>10} | {'MAE':>10} | {'#params':>10}")
    log_lines.append("=" * 90)

    print("\n".join(log_lines))

    # ---------------- Train / eval per horizon ----------------
    all_mse, all_mae = [], []

    for pred_len in PRED_LENS:
        train_ds = WindowDataset(scaled[tr0:tr1], SEQ_LEN, pred_len, start_offset=tr0)
        val_start = max(v0 - SEQ_LEN, 0)
        val_ds = WindowDataset(scaled[val_start:v1], SEQ_LEN, pred_len, start_offset=val_start)
        test_start = max(te0 - SEQ_LEN, 0)
        test_ds = WindowDataset(scaled[test_start:te1], SEQ_LEN, pred_len, start_offset=test_start)

        if len(train_ds) == 0 or len(test_ds) == 0:
            line = f"{pred_len:>10} | SKIPPED (not enough rows for seq_len={SEQ_LEN} + pred_len={pred_len})"
            print(line)
            log_lines.append(line)
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   drop_last=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS) if len(val_ds) > 0 else None
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS)

        model = CycleNetPro(seq_len=SEQ_LEN, pred_len=pred_len, enc_in=n_channels,
                             periods=periods_to_use, model_type=MODEL_TYPE, d_model=D_MODEL,
                             cnn_hidden=CNN_HIDDEN, cnn_kernel_size=CNN_KERNEL_SIZE,
                             cnn_layers=CNN_LAYERS).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        stopper = EarlyStopping(patience=PATIENCE)

        for epoch in range(EPOCHS):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            eval_loader = val_loader if val_loader is not None else train_loader
            val_loss = run_epoch(model, eval_loader, device, optimizer=None)
            stopper.step(val_loss, model)
            print(f"  [{DATASET_NAME} | pred_len={pred_len}] epoch {epoch+1}: "
                  f"train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
            if stopper.should_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        stopper.restore(model)
        mse, mae = evaluate_metrics(model, test_loader, device)
        all_mse.append(mse)
        all_mae.append(mae)

        line = f"{pred_len:>10} | {mse:>10.4f} | {mae:>10.4f} | {n_params:>10,}"
        print(line)
        log_lines.append(line)

        ckpt_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_pred{pred_len}.pt")
        torch.save(model.state_dict(), ckpt_path)

    if all_mse:
        avg_line1 = "-" * 90
        avg_line2 = f"{'AVERAGE':>10} | {np.mean(all_mse):>10.4f} | {np.mean(all_mae):>10.4f}"
        print(avg_line1)
        print(avg_line2)
        log_lines.append(avg_line1)
        log_lines.append(avg_line2)

    out_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_results.txt")
    with open(out_path, 'w') as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nSaved full log (periods, multi/single-cycle decision, backbone params, "
          f"MSE/MAE) to: {out_path}")


if __name__ == '__main__':
    main()

"""Standardized dataset loading, canonical splits, and non-leaking scalers."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from srp26.utils import resolve_data_path


class StandardTrainScaler:
    """Zero-mean unit-variance scaler fitted strictly on the training split."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "StandardTrainScaler":
        """Compute per-channel mean and standard deviation from training data."""
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0.0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization using training statistics."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert Z-score normalization back to the original feature scale."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform().")
        return (data * self.std) + self.mean


StandardScaler = StandardTrainScaler


class TimeSeriesSlidingWindowDataset(Dataset):
    """Sliding window dataset for multivariate time series forecasting."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 336,
        pred_len: int = 96,
        cycle_len: int = 24,
        start_idx_in_full: int = 0,
        return_cycle: bool = False,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cycle_len = cycle_len
        self.start_idx_in_full = start_idx_in_full
        self.return_cycle = return_cycle

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[s_end:r_end]

        x_t = torch.tensor(seq_x, dtype=torch.float32)
        y_t = torch.tensor(seq_y, dtype=torch.float32)
        if self.return_cycle:
            cycle_idx = torch.tensor(self.start_idx_in_full + s_begin, dtype=torch.long)
            return x_t, y_t, cycle_idx
        return x_t, y_t


class MTSDataset(Dataset):
    """Minimal sliding-window dataset returning ``(seq_x, seq_y)`` float tensors."""

    def __init__(self, data: np.ndarray, seq_len: int = 96, pred_len: int = 24) -> None:
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_end = index + self.seq_len
        seq_x = self.data[index:s_end]
        seq_y = self.data[s_end : s_end + self.pred_len]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


def create_dataloader(
    data: np.ndarray,
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 24,
    shuffle: bool = True,
) -> DataLoader:
    """Build a PyTorch DataLoader over an in-memory ``[Total_Len, Channels]`` array."""
    dataset = MTSDataset(data, seq_len=seq_len, pred_len=pred_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def load_raw_dataframe(csv_path: str | Path) -> np.ndarray:
    """Load CSV or headerless TXT multivariate time series as float32 ``[T, C]``."""
    path_str = str(csv_path)
    if path_str.endswith(".txt"):
        df = pd.read_csv(path_str, header=None)
    else:
        df = pd.read_csv(path_str)
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            df = df.drop(columns=date_cols)
    return df.values.astype(np.float32)


def get_canonical_splits(dataset_name: str, n_samples: int) -> Tuple[int, int, int, str]:
    """
    Return ``(train_end, val_end, test_end, split_label)`` following SRP26 conventions:
      - ``ETTh*``: canonical 12/4/4 months (8640 / 2880 / 2880 hourly steps)
      - ``ETTm*``: canonical 12/4/4 months (34560 / 11520 / 11520 15-min steps)
      - All other benchmarks: continuous 70% / 10% / 20% chronological split
    """
    ds_name = os.path.basename(dataset_name).lower()
    if ds_name.startswith("etth"):
        train_end = min(12 * 30 * 24, n_samples)
        val_end = min(train_end + 4 * 30 * 24, n_samples)
        test_end = min(val_end + 4 * 30 * 24, n_samples)
        return train_end, val_end, test_end, "Canonical 12/4/4 Months (ETTh)"
    if ds_name.startswith("ettm"):
        train_end = min(12 * 30 * 24 * 4, n_samples)
        val_end = min(train_end + 4 * 30 * 24 * 4, n_samples)
        test_end = min(val_end + 4 * 30 * 24 * 4, n_samples)
        return train_end, val_end, test_end, "Canonical 12/4/4 Months (ETTm)"
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.80)
    return train_end, val_end, n_samples, "Standard 70/10/20 Continuous Split"


def build_dataset_dataloaders(
    csv_path: str | Path,
    seq_len: int = 336,
    pred_len: int = 96,
    batch_size: int = 32,
    return_cycle: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Standardized data pipeline enforcing chronological splits, train-only Z-score
    scaling, and lookback prefix overlap on validation/test slices.
    """
    raw_data = load_raw_dataframe(csv_path)
    n_samples, n_channels = raw_data.shape
    train_end, val_end, test_end, split_type = get_canonical_splits(str(csv_path), n_samples)

    scaler = StandardTrainScaler().fit(raw_data[:train_end])
    scaled_data = scaler.transform(raw_data)

    val_start = max(0, train_end - seq_len)
    test_start = max(0, val_end - seq_len)

    train_ds = TimeSeriesSlidingWindowDataset(
        scaled_data[:train_end], seq_len, pred_len, start_idx_in_full=0, return_cycle=return_cycle
    )
    val_ds = TimeSeriesSlidingWindowDataset(
        scaled_data[val_start:val_end], seq_len, pred_len, start_idx_in_full=val_start, return_cycle=return_cycle
    )
    test_ds = TimeSeriesSlidingWindowDataset(
        scaled_data[test_start:test_end], seq_len, pred_len, start_idx_in_full=test_start, return_cycle=return_cycle
    )

    if verbose:
        print(f"Dataset: {csv_path} ({n_samples} steps, {n_channels} channels) | {split_type}")
        print(f"Windows -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "scaler": scaler,
        "channels": n_channels,
        "seq_len": seq_len,
        "pred_len": pred_len,
    }


build_etth1_dataloaders = build_dataset_dataloaders


def load_eval_windows(
    dataset_name: str,
    seq_len: int = 336,
    pred_len: int = 96,
    num_eval_windows: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load evenly spaced normalized test windows ``(x_eval, y_eval, n_channels)``
    using canonical splits and train-only Z-score normalization.
    """
    path = resolve_data_path(dataset_name)
    raw_data = load_raw_dataframe(path)
    total_len, n_channels = raw_data.shape
    train_end, val_end, test_end, _ = get_canonical_splits(dataset_name, total_len)

    scaler = StandardTrainScaler().fit(raw_data[:train_end])
    test_start = max(0, val_end - seq_len)
    test_scaled = scaler.transform(raw_data[test_start:test_end])

    max_start = len(test_scaled) - seq_len - pred_len
    if max_start < 0:
        raise ValueError(f"Test slice too short ({len(test_scaled)}) for seq_len={seq_len}, pred_len={pred_len}.")

    stride = max(1, max_start // max(1, num_eval_windows))
    starts = [i * stride for i in range(num_eval_windows) if i * stride <= max_start]
    if len(starts) < num_eval_windows:
        starts.append(max_start)
    starts = sorted(set(starts))[:num_eval_windows]

    x_list = [test_scaled[s : s + seq_len] for s in starts]
    y_list = [test_scaled[s + seq_len : s + seq_len + pred_len] for s in starts]
    return (
        torch.tensor(np.array(x_list), dtype=torch.float32),
        torch.tensor(np.array(y_list), dtype=torch.float32),
        n_channels,
    )

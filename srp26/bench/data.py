"""Dataset registry and the single v2 data pipeline (protocol §3).

Reuses the canonical split/scaler logic in ``srp26.data`` and adds: checksum
verification, seeded shuffling, the absolute cycle index for CycleNet-style
models, and per-dataset sampling metadata used by the physical-band metrics.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from srp26.data import StandardTrainScaler, TimeSeriesSlidingWindowDataset, get_canonical_splits, load_raw_dataframe
from srp26.bench.env import DATA_DIR, verify_dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    channels: int
    dt_hours: float  # sampling interval
    cycle_len: int  # CycleNet paper cycle length (Lin et al., NeurIPS 2024)

    @property
    def daily_period(self) -> int:
        return int(round(24.0 / self.dt_hours))


DATASETS: Dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETTh1", "ETTh1.csv", 7, 1.0, 24),
    "ETTh2": DatasetSpec("ETTh2", "ETTh2.csv", 7, 1.0, 24),
    "ETTm1": DatasetSpec("ETTm1", "ETTm1.csv", 7, 0.25, 96),
    "ETTm2": DatasetSpec("ETTm2", "ETTm2.csv", 7, 0.25, 96),
    "weather": DatasetSpec("weather", "weather.csv", 21, 1.0 / 6.0, 144),
    "electricity": DatasetSpec("electricity", "electricity.csv", 321, 1.0, 168),
    "traffic": DatasetSpec("traffic", "traffic.csv", 862, 1.0, 168),
    "solar": DatasetSpec("solar", "solar_AL.txt", 137, 1.0 / 6.0, 144),
}


@dataclass
class Bundle:
    spec: DatasetSpec
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_raw: np.ndarray  # unscaled train split (period discovery)
    train_scaled: np.ndarray  # scaled train split (channel-dependence diagnostic)
    split: Dict[str, int]
    sha256: str


def build_bundle(
    dataset: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    generator: torch.Generator,
    eval_batch_size: int = 256,
) -> Bundle:
    spec = DATASETS[dataset]
    digest = verify_dataset(spec.filename)
    raw = load_raw_dataframe(DATA_DIR / spec.filename)
    if raw.shape[1] != spec.channels:
        raise RuntimeError(f"{dataset}: expected {spec.channels} channels, found {raw.shape[1]}.")

    n = raw.shape[0]
    train_end, val_end, test_end, _ = get_canonical_splits(spec.filename, n)
    scaler = StandardTrainScaler().fit(raw[:train_end])
    scaled = scaler.transform(raw).astype(np.float32)

    val_start = max(0, train_end - seq_len)
    test_start = max(0, val_end - seq_len)
    mk = TimeSeriesSlidingWindowDataset
    train_ds = mk(scaled[:train_end], seq_len, pred_len, start_idx_in_full=0, return_cycle=True)
    val_ds = mk(scaled[val_start:val_end], seq_len, pred_len, start_idx_in_full=val_start, return_cycle=True)
    test_ds = mk(scaled[test_start:test_end], seq_len, pred_len, start_idx_in_full=test_start, return_cycle=True)

    return Bundle(
        spec=spec,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator),
        val_loader=DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False),
        test_loader=DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False),
        train_raw=raw[:train_end],
        train_scaled=scaled[:train_end],
        split={"train_end": train_end, "val_end": val_end, "test_end": test_end,
               "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)},
        sha256=digest,
    )

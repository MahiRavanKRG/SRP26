"""Empirical transform-ablation driver (real datasets, Tier-2 scheme).

Runs the grid  {dataset} x {horizon} x {transform} x {seed}  with a single
shared minimal backbone and full DSOF spectral diagnostics, writing per-run
JSON, a master CSV, and per-window MSE vectors (for paired significance).

Usage (smoke test):
    uv run python experiment_transform_ablation.py \
        --datasets ETTh1.csv --horizons 96 \
        --transforms identity rfft --seeds 2024 --epochs 3 --patience 2 \
        --save_dir results/transform_ablation_smoke
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from srp26.data import build_dataset_dataloaders
from srp26.transform_ablation import TransformForecaster, train_transform_model
from srp26.transforms import ALL_TRANSFORMS
from srp26.utils import get_device, resolve_data_path, set_seed

DEFAULT_DATASETS = [
    "ETTh1.csv",
    "ETTh2.csv",
    "ETTm1.csv",
    "ETTm2.csv",
    "weather.csv",
    "electricity.csv",
    "traffic.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SRP26 transform-ablation benchmark")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--transforms", nargs="+", default=ALL_TRANSFORMS)
    p.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    p.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--head", type=str, default="linear", choices=["linear", "mlp"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="results/transform_ablation")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _append_csv(summary_csv: str, row: dict) -> None:
    df_row = pd.DataFrame([row])
    if os.path.exists(summary_csv):
        existing = pd.read_csv(summary_csv)
        mask = (
            (existing["transform"] == row["transform"])
            & (existing["dataset"] == row["dataset"])
            & (existing["pred_len"] == row["pred_len"])
            & (existing["seed"] == row["seed"])
        )
        pd.concat([existing[~mask], df_row], ignore_index=True).to_csv(summary_csv, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)


def main() -> None:
    args = parse_args()
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)
    raw_dir = os.path.join(args.save_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    summary_csv = os.path.join(args.save_dir, "transform_ablation_summary.csv")

    total = len(args.datasets) * len(args.horizons) * len(args.transforms) * len(args.seeds)
    idx = 0
    print(f"Transform ablation: {total} runs on {device} | head={args.head} | epochs={args.epochs}")

    for ds in args.datasets:
        ds_label = os.path.basename(ds).split(".")[0]
        path = resolve_data_path(ds)
        for h in args.horizons:
            loaders = build_dataset_dataloaders(
                path,
                seq_len=args.seq_len,
                pred_len=h,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            channels = loaders["channels"]
            for t in args.transforms:
                for seed in args.seeds:
                    idx += 1
                    set_seed(seed)
                    model = TransformForecaster(
                        t, args.seq_len, h, channels, head=args.head, hidden=args.hidden
                    )
                    record, per_window = train_transform_model(
                        model,
                        loaders["train_loader"],
                        loaders["val_loader"],
                        loaders["test_loader"],
                        device,
                        max_epochs=args.epochs,
                        patience=args.patience,
                        lr=args.lr,
                    )
                    row = {
                        "transform": t,
                        "dataset": ds_label,
                        "pred_len": h,
                        "seed": seed,
                        **record,
                    }
                    print(
                        f"[{idx}/{total}] {ds_label} H={h} {t} seed={seed} "
                        f"-> MSE={record['mse']:.5f} (params={record['params']})"
                    )

                    np.save(os.path.join(raw_dir, f"{ds_label}_H{h}_{t}_s{seed}_wmse.npy"), per_window)
                    with open(os.path.join(args.save_dir, f"{ds_label}_H{h}_{t}_s{seed}.json"), "w") as f:
                        json.dump(row, f, indent=2)
                    _append_csv(summary_csv, row)

    print(f"Done. Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()

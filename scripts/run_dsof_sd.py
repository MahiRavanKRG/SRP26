"""
DSOF-SD Standardized Benchmark Runner
=====================================
Executes Phases 0-3 of DSOF-Guided Spectral Distillation strictly following
``docs/experiment_standardization_protocol.md`` (L=336, canonical splits,
train-only Z-score normalization, multi-seed support).

Examples:
    # Fast end-to-end validation with analytical Fourier teacher baseline:
    uv run python scripts/run_dsof_sd.py --dataset ETTh1.csv --teacher fourier --epochs 10

    # Distill from cached TimesFM 2.5 predictions (or extract & cache once):
    uv run python scripts/run_dsof_sd.py --dataset ETTh1.csv --teacher timesfm --teacher_cache results/cache_timesfm_ETTh1_H96.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dsof_sd import (
    BANDS,
    DSOFSDConfig,
    DSOFSDTrainer,
    FourierBaseTeacher,
    TeacherTargetCache,
    TimesFMTeacher,
    build_dsof_student,
    discover_dataset_periods,
)
from srp26.data import (
    StandardTrainScaler,
    TimeSeriesSlidingWindowDataset,
    get_canonical_splits,
    load_raw_dataframe,
)
from srp26.utils import ensure_hf_cache, get_device, resolve_data_path, set_seed


def build_raw_dataloaders(
    dataset_name: str,
    seq_len: int = 336,
    pred_len: int = 96,
    batch_size: int = 32,
    extract_batch_size: int = 64,
):
    """Build train/val/test DataLoaders following canonical SRP26 split rules."""
    csv_path = resolve_data_path(dataset_name, repo_root=REPO_ROOT)
    raw = load_raw_dataframe(csv_path)
    n_samples, n_channels = raw.shape
    train_end, val_end, test_end, split_label = get_canonical_splits(dataset_name, n_samples)

    scaler = StandardTrainScaler().fit(raw[:train_end])
    scaled = scaler.transform(raw)

    val_start = max(0, train_end - seq_len)
    test_start = max(0, val_end - seq_len)

    train_slice = scaled[:train_end]
    train_ds = TimeSeriesSlidingWindowDataset(train_slice, seq_len, pred_len, start_idx_in_full=0, return_cycle=True)
    val_ds = TimeSeriesSlidingWindowDataset(scaled[val_start:val_end], seq_len, pred_len, start_idx_in_full=val_start, return_cycle=True)
    test_ds = TimeSeriesSlidingWindowDataset(scaled[test_start:test_end], seq_len, pred_len, start_idx_in_full=test_start, return_cycle=True)

    return {
        "train_loader_ordered": DataLoader(train_ds, batch_size=extract_batch_size, shuffle=False),
        "val_loader": DataLoader(val_ds, batch_size=extract_batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=extract_batch_size, shuffle=False),
        "train_scaled": train_slice,
        "channels": n_channels,
        "split_label": split_label,
    }


def _print_report(dataset: str, pred_len: int, student_type: str, seed: int, report: Dict) -> None:
    print("\n" + "=" * 88)
    print(f"DSOF-SD PHASE 3 REPORT | {dataset} | H={pred_len} | student={student_type} | seed={seed}")
    print(f"Params: {report['params']:,} | Runtime: {report['runtime_sec']}s | Gates: {report['gates']}")
    print("-" * 88)
    print(f"{'Model':<20} | {'MSE':>8} | {'MAE':>8} | {'LSD':>8} | {'FVU Low':>8} | {'FVU Mid':>8} | {'FVU High':>8}")
    print("-" * 88)
    for label, key in (("Teacher", "teacher"), ("Warm Student", "warm_student"), ("DSOF-SD Student", "distilled_student")):
        m = report[key]
        b = m["bands"]
        print(
            f"{label:<20} | {m['mse']:>8.4f} | {m['mae']:>8.4f} | {m['lsd']:>8.4f} | "
            f"{b['low']['fvu']:>8.4f} | {b['mid']['fvu']:>8.4f} | {b['high']['fvu']:>8.4f}"
        )
    print("-" * 88)
    print("Per-band LSD (unified rFFT log-amplitude):")
    for band in BANDS:
        t_l = report["teacher"]["bands"][band]["lsd"]
        w_l = report["warm_student"]["bands"][band]["lsd"]
        d_l = report["distilled_student"]["bands"][band]["lsd"]
        print(f"  [{band:<4}] gate={report['gates'][band]:.2f} | teacher={t_l:.4f} | warm={w_l:.4f} | distilled={d_l:.4f}")
    print(f"Hypothesis verdict: {report['hypothesis']}")
    print("=" * 88 + "\n")


def _aggregate_seed_reports(reports: list[Dict]) -> Dict:
    """Compute exact mean and sample/pop std across multi-seed reports."""
    import numpy as np

    def _agg(values: list[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    summary: Dict[str, object] = {
        "dataset": reports[0]["dataset"],
        "pred_len": reports[0]["pred_len"],
        "seq_len": reports[0]["seq_len"],
        "student_type": reports[0]["student_type"],
        "teacher_name": reports[0]["teacher_name"],
        "gate_mode": reports[0]["gate_mode"],
        "seeds": [r["seed"] for r in reports],
        "periods": reports[0]["periods"],
        "params": reports[0]["params"],
        "gates": {b: _agg([r["gates"][b] for r in reports]) for b in BANDS},
        "models": {},
        "hypothesis_passes": sum(1 for r in reports if r["hypothesis"]["passed"]),
        "total_seeds": len(reports),
    }
    for key in ("teacher", "warm_student", "distilled_student"):
        model_stats: Dict[str, object] = {
            metric: _agg([r[key][metric] for r in reports])
            for metric in ("mse", "mae", "lsd")
        }
        model_stats["bands"] = {
            band: {
                metric: _agg([r[key]["bands"][band][metric] for r in reports])
                for metric in ("fvu", "lsd", "mse")
            }
            for band in BANDS
        }
        summary["models"][key] = model_stats
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSOF-Guided Spectral Distillation (DSOF-SD).")
    parser.add_argument("--dataset", type=str, default="ETTh1.csv")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--extract_batch_size", type=int, default=64)
    parser.add_argument("--per_core_batch_size", type=int, default=256)
    parser.add_argument("--student_type", type=str, default="sor", choices=["sor", "linear", "mlp", "cnn", "cyclenet_lite"])
    parser.add_argument("--teacher", type=str, default="fourier", choices=["fourier", "timesfm"])
    parser.add_argument("--teacher_cache", type=str, default="")
    parser.add_argument("--gate_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lambda_max", type=float, default=0.2)
    parser.add_argument("--lambda_phase", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_hf_cache()
    device = get_device()
    seeds = args.seeds if args.seeds else [args.seed]

    raw_info = build_raw_dataloaders(
        args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        extract_batch_size=args.extract_batch_size,
    )

    # Phase 0: Load or build offline teacher cache ONCE across all seeds
    if args.teacher_cache and os.path.exists(args.teacher_cache):
        print(f"Loading cached teacher targets from {args.teacher_cache} ...")
        train_ds, val_ds, test_ds, _ = TeacherTargetCache.load(args.teacher_cache)
    else:
        print(f"Generating Phase 0 offline teacher targets via '{args.teacher}' ...")
        if args.teacher == "timesfm":
            teacher_fn = TimesFMTeacher(
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                per_core_batch_size=args.per_core_batch_size,
            )
        else:
            teacher_fn = FourierBaseTeacher(pred_len=args.pred_len, top_k=3)

        train_ds = TeacherTargetCache.build_from_loader(raw_info["train_loader_ordered"], teacher_fn, device=device, progress_label="train")
        val_ds = TeacherTargetCache.build_from_loader(raw_info["val_loader"], teacher_fn, device=device, progress_label="val")
        test_ds = TeacherTargetCache.build_from_loader(raw_info["test_loader"], teacher_fn, device=device, progress_label="test")

        if args.teacher_cache:
            TeacherTargetCache.save(
                args.teacher_cache,
                train_ds,
                val_ds,
                test_ds,
                metadata={"dataset": args.dataset, "seq_len": args.seq_len, "pred_len": args.pred_len, "teacher": args.teacher},
            )
            print(f"Saved teacher cache to {args.teacher_cache}")

    periods, use_cd = discover_dataset_periods(raw_info["train_scaled"], seq_len=args.seq_len)
    print(f"Discovered student periods: {periods} | channel_dependence={use_cd}")

    reports: list[Dict] = []
    stem = Path(args.dataset).stem
    for seed in seeds:
        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        student = build_dsof_student(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            channels=raw_info["channels"],
            student_type=args.student_type,
            periods=periods,
            use_channel_dependence=use_cd,
        )

        cfg = DSOFSDConfig(
            gate_mode=args.gate_mode,
            warm_epochs=args.warm_epochs,
            distill_epochs=args.epochs,
            lr=args.lr,
            lambda_max=args.lambda_max,
            lambda_phase=args.lambda_phase,
        )
        trainer = DSOFSDTrainer(student=student, config=cfg, device=device)
        report = trainer.fit_and_evaluate(train_loader, val_loader, test_loader)
        report["dataset"] = args.dataset
        report["seq_len"] = args.seq_len
        report["pred_len"] = args.pred_len
        report["student_type"] = args.student_type
        report["teacher_name"] = args.teacher
        report["gate_mode"] = args.gate_mode
        report["seed"] = seed
        report["periods"] = periods
        report["channel_dependence"] = use_cd
        reports.append(report)

        _print_report(args.dataset, args.pred_len, args.student_type, seed, report)

        if len(seeds) == 1 and args.output:
            out_path = Path(args.output)
        else:
            out_path = REPO_ROOT / "results" / "dsof_sd" / f"dsof_sd_{stem}_H{args.pred_len}_{args.student_type}_{args.teacher}_{args.gate_mode}_s{seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved DSOF-SD results to {out_path}")

    if len(reports) > 1:
        summary = _aggregate_seed_reports(reports)
        summary_path = Path(args.output) if args.output else (
            REPO_ROOT / "results" / "dsof_sd" / f"dsof_sd_{stem}_H{args.pred_len}_{args.student_type}_{args.teacher}_{args.gate_mode}_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved {len(reports)}-seed summary to {summary_path}")


if __name__ == "__main__":
    main()


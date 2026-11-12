"""
PyTorch Lightning Runner with Weights & Biases Logging
======================================================
Refactors the model training into the academic gold standard:
- PyTorch Lightning Trainer with 16-mixed precision (AMP).
- Weights & Biases (WandB) logger integration.
- Full dataset iteration without mocking or subsetting.
- Automatic checkpointing, validation, early stopping, and testing.
"""

import os
import sys
import time
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from models import (
    DLinear,
    CycleNet,
    CycleNetMLP,
    GenuineTQNet,
    TwoStageFourierPatchAttentionNet,
    TwoStageFourierCrossAttentionNet
)
from lightning_module import TimeSeriesDataModule, TimeSeriesLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning MTSF Training Suite")
    parser.add_argument("--model", type=str, default="patch_attention",
                        choices=["dlinear", "dlinear_revin", "cyclenet", "cyclenet_mlp", "tqnet", "patch_attention", "fourier_cross_attn", "ftq_net"],
                        help="Model to train")
    parser.add_argument("--dataset", type=str, default="ETTh1.csv",
                        help="Dataset filename in data_sets/ or root")
    parser.add_argument("--seq_len", type=int, default=336, help="Lookback window length")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecast horizon length")
    parser.add_argument("--period_w", type=int, default=24, help="Periodic queue length W (0 to disable RCF)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=40, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision: 16-mixed or 32")
    parser.add_argument("--wandb_project", type=str, default="SRP26-LTSF-Sandbox", help="W&B Project name")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser.parse_args()


def run_single_experiment(model: str = "patch_attention", dataset: str = "ETTh1.csv",
                           seq_len: int = 336, pred_len: int = 96, period_w: int = 24,
                           batch_size: int = 32, lr: float = 3e-4, max_epochs: int = 40,
                           patience: int = 7, precision: str = "16-mixed",
                           wandb_project: str = "SRP26-LTSF-Sandbox", seed: int = 2024):
    L.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    # 1. Dataset Resolution
    data_path = os.path.join(current_dir, '..', 'data_sets', dataset)
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', dataset)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # 2. Lightning DataModule (Full dataset iteration, no subsetting, no mocking)
    # Return cycle index for all models to enable phase tracking
    datamodule = TimeSeriesDataModule(
        csv_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        return_cycle=True
    )
    datamodule.setup()
    channels = datamodule.channels

    # 3. Model Instantiation
    dataset_name = os.path.basename(dataset).split('.')[0]
    if model == "dlinear":
        raw_model = DLinear(seq_len=seq_len, pred_len=pred_len, channels=channels, use_revin=False)
        model_title = "DLinear"
    elif model == "dlinear_revin":
        raw_model = DLinear(seq_len=seq_len, pred_len=pred_len, channels=channels, use_revin=True)
        model_title = "DLinear+RevIN"
    elif model == "cyclenet":
        raw_model = CycleNet(seq_len=seq_len, pred_len=pred_len, channels=channels, period_w=period_w)
        model_title = f"CycleNet-W{period_w}" if period_w != 24 else "CycleNet"
    elif model == "cyclenet_mlp":
        raw_model = CycleNetMLP(seq_len=seq_len, pred_len=pred_len, channels=channels, period_w=period_w)
        model_title = f"CycleNet/MLP-W{period_w}" if period_w != 24 else "CycleNet/MLP"
    elif model == "tqnet":
        raw_model = GenuineTQNet(seq_len=seq_len, pred_len=pred_len, channels=channels, period_w=period_w)
        model_title = "TQNet"
    elif model == "patch_attention":
        raw_model = TwoStageFourierPatchAttentionNet(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            patch_len=16, stride=8, d_model=64, n_heads=4, num_layers=2, dropout=0.2
        )
        model_title = "Fourier-PatchAttention"
    elif model in ["fourier_cross_attn", "ftq_net"]:
        raw_model = TwoStageFourierCrossAttentionNet(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            n_heads=4, d_mlp=128, dropout=0.1
        )
        model_title = "Fourier-CrossAttention"
    else:
        raise ValueError(f"Unknown model: {model}")

    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

    lightning_module = TimeSeriesLightningModule(
        model=raw_model,
        model_name=model_title,
        lr=lr,
        weight_decay=1e-4,
        max_epochs=max_epochs
    )

    # 4. Weights & Biases Logger Setup
    wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    run_name = f"{model_title}_{dataset_name}_L{seq_len}_H{pred_len}_16mixed"
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        save_dir=os.path.join(current_dir, "wandb_logs")
    )

    # 5. Callbacks
    ckpt_dir = os.path.join(current_dir, "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 6. PyTorch Lightning Trainer with 16-Mixed Precision
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=wandb_logger,
        gradient_clip_val=5.0,
        enable_progress_bar=True,
        deterministic=True
    )

    print("\n" + "="*80)
    print(f"Starting PyTorch Lightning Training: {run_name}")
    print(f"Parameters: {total_params:,} | Precision: {precision} | Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"W&B Project: {wandb_project} (Mode: {wandb_mode})")
    print("="*80 + "\n")

    # 7. Full Training
    start_time = time.time()
    trainer.fit(lightning_module, datamodule=datamodule)
    train_time_sec = round(time.time() - start_time, 2)

    # 8. Test Evaluation on Best Restored Checkpoint
    best_path = checkpoint_callback.best_model_path
    print(f"\nTraining Complete ({train_time_sec}s)! Restoring Best Checkpoint: {best_path}")
    test_results = trainer.test(lightning_module, datamodule=datamodule, ckpt_path=best_path)

    t_res = test_results[0]
    res_dict = {
        'model': model_title,
        'dataset': dataset_name,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'period_w': period_w,
        'precision': precision,
        'params': total_params,
        'train_time_sec': train_time_sec,
        'best_checkpoint': best_path,
        'test_mse': round(float(t_res.get('test_total_mse', t_res.get('test_mse', 0.0))), 4),
        'test_mae': round(float(t_res.get('test_mae', 0.0)), 4),
        'mse_low': round(float(t_res.get('test_mse_low', 0.0)), 5),
        'mse_mid': round(float(t_res.get('test_mse_mid', 0.0)), 5),
        'mse_high': round(float(t_res.get('test_mse_high', 0.0)), 5),
        'fvu_low': round(float(t_res.get('test_fvu_low', 0.0)), 4),
        'fvu_mid': round(float(t_res.get('test_fvu_mid', 0.0)), 4),
        'fvu_high': round(float(t_res.get('test_fvu_high', 0.0)), 4),
        'energy_ratio_mid': round(float(t_res.get('test_energy_ratio_mid', 0.0)), 4),
        'log_spectral_distance': round(float(t_res.get('test_log_spectral_distance', 0.0)), 4),
        'failure_rate_pct': round(float(t_res.get('test_failure_rate_pct', 0.0)), 2),
        'parseval_additive_error': round(float(t_res.get('test_parseval_additive_error', 0.0)), 7)
    }

    # Close wandb run cleanly
    if hasattr(wandb_logger, 'experiment') and hasattr(wandb_logger.experiment, 'finish'):
        wandb_logger.experiment.finish()

    return res_dict


def main():
    args = parse_args()
    res = run_single_experiment(
        model=args.model,
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        period_w=args.period_w,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        precision=args.precision,
        wandb_project=args.wandb_project,
        seed=args.seed
    )
    print("\n" + "#"*80)
    print("Experiment Concluded:")
    print(res)
    print("#"*80)


if __name__ == "__main__":
    main()

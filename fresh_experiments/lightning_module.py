"""
PyTorch Lightning Refactored Module & DataModule for Time Series Forecasting
=============================================================================
Academic Gold Standard Implementation:
- LightningModule managing forward, loss step, metrics, optimizer, and scheduler.
- LightningDataModule ensuring full dataset iteration without mocking or subsetting.
- Mixed precision (16-mixed) native support.
- Full Parseval spectral decomposition evaluation on test epoch end.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocessing import build_dataset_dataloaders
from spectral_diagnostics import evaluate_all_spectral_diagnostics


class TimeSeriesDataModule(L.LightningDataModule):
    """
    Lightning DataModule providing full dataset iteration without subsetting or mocking.
    - Fits Scaler strictly on the training partition.
    - Preserves historical lookback prefix on Val and Test splits.
    - Supports dataset or csv_path argument, and flexible (x, y) or (x, y, cyc) batches.
    """
    def __init__(self, csv_path: str = None, dataset: str = None, seq_len: int = 336, pred_len: int = 96, batch_size: int = 32, return_cycle: bool = False):
        super().__init__()
        target_path = dataset if dataset is not None else csv_path
        if target_path is None:
            target_path = "ETTh1.csv"

        if not os.path.exists(target_path):
            alt_1 = os.path.join(current_dir, '..', 'data_sets', target_path)
            alt_2 = os.path.join(current_dir, '..', target_path)
            if os.path.exists(alt_1):
                target_path = alt_1
            elif os.path.exists(alt_2):
                target_path = alt_2

        self.csv_path = target_path
        self.dataset = os.path.basename(target_path)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.return_cycle = return_cycle
        self.data_bundle = None

    def setup(self, stage: str = None):
        if self.data_bundle is None:
            self.data_bundle = build_dataset_dataloaders(
                csv_path=self.csv_path,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                batch_size=self.batch_size,
                return_cycle=self.return_cycle
            )

    @property
    def channels(self) -> int:
        if self.data_bundle is None:
            self.setup()
        return self.data_bundle['channels']

    def train_dataloader(self):
        return self.data_bundle['train_loader']

    def val_dataloader(self):
        return self.data_bundle['val_loader']

    def test_dataloader(self):
        return self.data_bundle['test_loader']


class TimeSeriesLightningModule(L.LightningModule):
    """
    PyTorch Lightning Module encapsulating model forward, training, validation,
    testing, checkpointing, and optimizer management.
    """
    def __init__(self, model: nn.Module, model_name: str = "ForecastingModel",
                 lr: float = 3e-4, weight_decay: float = 1e-4, max_epochs: int = 40):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # Temporary storage for test epoch end metrics
        self.test_preds = []
        self.test_trues = []

    def forward(self, x: torch.Tensor, cyc: torch.Tensor = None):
        return self.model(x, cyc)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, cyc = batch
        else:
            x, y = batch
            cyc = None

        y_hat = self(x, cyc)
        loss = self.mse_criterion(y_hat, y)
        mae = self.mae_criterion(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, cyc = batch
        else:
            x, y = batch
            cyc = None

        y_hat = self(x, cyc)
        loss = self.mse_criterion(y_hat, y)
        mae = self.mae_criterion(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, cyc = batch
        else:
            x, y = batch
            cyc = None

        y_hat = self(x, cyc)
        loss = self.mse_criterion(y_hat, y)
        mae = self.mae_criterion(y_hat, y)


        self.log("test_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        self.test_preds.append(y_hat.detach().cpu())
        self.test_trues.append(y.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            all_preds = torch.cat(self.test_preds, dim=0)
            all_trues = torch.cat(self.test_trues, dim=0)

            diag = evaluate_all_spectral_diagnostics(all_trues, all_preds)
            self.test_diagnostics = diag

            for k, v in diag.items():
                self.log(f"test_{k}", v, on_epoch=True)

            print(f"\n==================================================================")
            print(f"LIGHTNING TEST RESULTS: {self.model_name}")
            print(f"Test MSE: {diag['total_mse']:.4f} | Test MAE: {float(self.mae_criterion(all_preds, all_trues).item()):.4f}")
            print(f"Parseval Band MSE: Low={diag['mse_low']:.4f}, Mid={diag['mse_mid']:.4f}, High={diag['mse_high']:.4f} (Add Error={diag['parseval_additive_error']:.6f})")
            print(f"Normalized Subband FVU: Low={diag['fvu_low']:.4f}, Mid={diag['fvu_mid']:.4f}, High={diag['fvu_high']:.4f}")
            print(f"Energy Ratio: Low={diag['energy_ratio_low']:.4f}, Mid={diag['energy_ratio_mid']:.4f}, High={diag['energy_ratio_high']:.4f}")
            print(f"Log-Spectral Distance (LSD): {diag['log_spectral_distance']:.4f} dB | Failure Rate: {diag['failure_rate_pct']:.1f}%")
            print(f"==================================================================\n")

            self.test_preds.clear()
            self.test_trues.clear()

    def configure_optimizers(self):
        # 1. Decoupled weight decay AdamW (standard in modern LTSF architectures like PatchTST and iTransformer)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 2. Linear Warmup (first 3 epochs) followed by smooth Cosine Annealing decay
        warmup_epochs = min(3, max(1, self.max_epochs // 8))
        main_epochs = max(1, self.max_epochs - warmup_epochs)

        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=main_epochs,
            eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


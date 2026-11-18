"""
DSOF-SD: DSOF-Guided Spectral Distillation
=========================================
Builds a lightweight hybrid time+frequency student (CycleNet-Pro / SORMixer or
CycleNet) distilled from an offline frozen teacher (e.g., TimesFM 2.5) through
Parseval-exact band trust gates ``g_b = 1[FVU_b^teacher < FVU_b^student]``.

Both the spectral distillation loss and DSOF evaluation share the exact same
differentiable single-window rFFT operator (``srp26.spectral.dsof_band_metrics_torch``)
so the training objective is literally the squared band-wise Log-Spectral Distance.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.cyclenet.cyclenet_pro_Full import (
    CycleNetPro,
    compute_correlation_metrics,
    decide_channel_dependence,
    decide_multi_or_single_cycle,
    filter_true_periods_acf,
    find_top_periods_fft,
)
from srp26.fourier import extract_and_extrapolate_fourier_base_torch
from srp26.models_lite import CycleNet
from srp26.spectral import band_masks_rfft_torch, dsof_band_metrics_torch

BANDS: Tuple[str, str, str] = ("low", "mid", "high")


@dataclass
class DSOFSDConfig:
    """Configuration for Phase 1-3 DSOF-Guided Spectral Distillation."""

    low_cutoff: float = 0.1
    high_cutoff: float = 0.4
    fs: float = 1.0
    gate_mode: str = "hard"  # "hard" | "soft" | "lsd" | "consensus" | "benefit"
    soft_alpha: float = 5.0
    benefit_probe_epochs: int = 3  # short distillation per band for G4
    warm_epochs: int = 5
    distill_epochs: int = 25
    patience: int = 7
    lr: float = 3e-4
    weight_decay: float = 1e-3
    lambda_max: float = 0.2
    lambda_phase: float = 0.0
    lambda_anchor: float = 0.0  # ground-truth spectral anchor weight
    recalibrate_teacher: bool = False  # amplitude recalibration
    eps: float = 1e-6


class TeacherTargetDataset(Dataset):
    """Dataset holding cached ``(x, y_true, y_teacher, cycle_index)`` windows."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_teacher: torch.Tensor,
        cycle_index: Optional[torch.Tensor] = None,
    ) -> None:
        if x.shape[0] != y.shape[0] or x.shape[0] != y_teacher.shape[0]:
            raise ValueError("Mismatched sample count across x, y, and y_teacher.")
        self.x = x.float()
        self.y = y.float()
        self.y_teacher = y_teacher.float()
        if cycle_index is None:
            self.cycle_index = torch.arange(x.shape[0], dtype=torch.long)
        else:
            self.cycle_index = cycle_index.long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.y_teacher[idx], self.cycle_index[idx]


class TeacherTargetCache:
    """Offline storage and generation helper for Phase 0 frozen teacher targets."""

    @staticmethod
    def build_from_loader(
        loader: DataLoader,
        teacher_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
        progress_label: str = "",
    ) -> TeacherTargetDataset:
        """Run frozen ``teacher_fn(x)`` once across ``loader`` and cache outputs on CPU."""
        xs, ys, yts, cidxs = [], [], [], []
        offset = 0
        total_batches = len(loader) if hasattr(loader, "__len__") else 0
        t_start = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                if len(batch) == 3:
                    x_b, y_b, c_b = batch
                else:
                    x_b, y_b = batch
                    c_b = torch.arange(offset, offset + x_b.shape[0], dtype=torch.long)
                offset += x_b.shape[0]
                y_t = teacher_fn(x_b.to(device)).detach().cpu()
                xs.append(x_b.cpu())
                ys.append(y_b.cpu())
                yts.append(y_t)
                cidxs.append(c_b.cpu())
                if progress_label and total_batches > 0 and (
                    batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 5) == 0
                ):
                    elapsed = time.time() - t_start
                    rate = offset / max(elapsed, 1e-6)
                    print(
                        f"  [{progress_label}] batch {batch_idx}/{total_batches} "
                        f"({offset} windows, {elapsed:.1f}s, {rate:.1f} win/s)",
                        flush=True,
                    )
        return TeacherTargetDataset(
            torch.cat(xs, dim=0),
            torch.cat(ys, dim=0),
            torch.cat(yts, dim=0),
            torch.cat(cidxs, dim=0),
        )

    @staticmethod
    def save(path: str | Path, train_ds: TeacherTargetDataset, val_ds: TeacherTargetDataset, test_ds: TeacherTargetDataset, metadata: Optional[Dict] = None) -> None:
        """Persist cached splits atomically via ``torch.save``."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train": {"x": train_ds.x, "y": train_ds.y, "y_teacher": train_ds.y_teacher, "cycle_index": train_ds.cycle_index},
            "val": {"x": val_ds.x, "y": val_ds.y, "y_teacher": val_ds.y_teacher, "cycle_index": val_ds.cycle_index},
            "test": {"x": test_ds.x, "y": test_ds.y, "y_teacher": test_ds.y_teacher, "cycle_index": test_ds.cycle_index},
            "metadata": metadata or {},
        }
        torch.save(payload, target_path)

    @staticmethod
    def load(path: str | Path) -> Tuple[TeacherTargetDataset, TeacherTargetDataset, TeacherTargetDataset, Dict]:
        """Load cached splits previously saved by ``TeacherTargetCache.save``."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        splits = tuple(
            TeacherTargetDataset(
                payload[name]["x"],
                payload[name]["y"],
                payload[name]["y_teacher"],
                payload[name]["cycle_index"],
            )
            for name in ("train", "val", "test")
        )
        return splits[0], splits[1], splits[2], payload.get("metadata", {})


class TimesFMTeacher:
    """Lazy wrapper around Google TimesFM 2.5 (200M) for Phase 0 target extraction."""

    def __init__(
        self,
        seq_len: int = 336,
        pred_len: int = 96,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        per_core_batch_size: int = 256,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_id = model_id
        self.per_core_batch_size = per_core_batch_size
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

        self._model = TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
        self._model.compile(
            ForecastConfig(
                max_context=self.seq_len,
                max_horizon=self.pred_len,
                per_core_batch_size=self.per_core_batch_size,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        x_np = x.detach().cpu().numpy()
        bsz, seq_len, channels = x_np.shape
        flat = [x_np[b, :, c] for b in range(bsz) for c in range(channels)]
        point_raw, _ = self._model.forecast(horizon=self.pred_len, inputs=flat)
        arr = np.asarray(point_raw, dtype=np.float32).reshape(bsz, channels, self.pred_len).transpose(0, 2, 1)
        return torch.from_numpy(arr)


class FourierBaseTeacher:
    """Zero-parameter analytical top-K rFFT teacher (useful fast offline reference/baseline)."""

    def __init__(self, pred_len: int, top_k: int = 3) -> None:
        self.pred_len = pred_len
        self.top_k = top_k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, y_base, _ = extract_and_extrapolate_fourier_base_torch(x, pred_len=self.pred_len, top_k=self.top_k)
        return y_base


class DSOFSpectralLoss(nn.Module):
    """
    Phase 1-2 band-gated spectral distillation objective.

    Supports multiple gate calibration modes:
    - ``"hard"``: ``g_b = 1[FVU_b^T < FVU_b^S]`` (original FVU gate)
    - ``"soft"``: sigmoid-smoothed FVU gate
    - ``"lsd"``: G1 spectral-fidelity gate — ``g_b = 1[LSD_b^T < LSD_b^S]``
    - ``"consensus"``: G5 strict gate — ``g_b = 1[(FVU_T < FVU_S) AND (LSD_T < LSD_S)]``
    - ``"benefit"``: G4 empirical per-band probe — keeps band only if short
      distillation *actually* improves both band LSD and band MSE vs. warm

    When ``config.recalibrate_teacher`` is True, teacher FFT amplitudes are
    rescaled per band so ``E_T,b = E_true,b`` before computing the spectral
    loss, removing MMSE amplitude-shrinkage bias.

    When ``config.lambda_anchor > 0``, a ground-truth spectral fidelity term
    ``lambda_anchor * LSD(y_student, y_true)`` is added to prevent the student
    from drifting into amplitude collapse.
    """

    def __init__(self, config: DSOFSDConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("gate_vector", torch.zeros(len(BANDS), dtype=torch.float32))
        self.calibrated = False
        self.calibration_summary: Dict[str, Dict[str, float]] = {}

    @property
    def gates(self) -> Dict[str, float]:
        return {band: float(self.gate_vector[i].item()) for i, band in enumerate(BANDS)}

    @torch.no_grad()
    def calibrate_band_gates(
        self,
        y_true: torch.Tensor,
        y_teacher: torch.Tensor,
        y_student_warm: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute and freeze per-band trust gates ``g_b`` from calibration FVU/LSD."""
        t_metrics = dsof_band_metrics_torch(
            y_true, y_teacher,
            fs=self.config.fs,
            low_cutoff=self.config.low_cutoff,
            high_cutoff=self.config.high_cutoff,
            eps=self.config.eps,
        )
        s_metrics = dsof_band_metrics_torch(
            y_true, y_student_warm,
            fs=self.config.fs,
            low_cutoff=self.config.low_cutoff,
            high_cutoff=self.config.high_cutoff,
            eps=self.config.eps,
        )

        fvu_t_summary: Dict[str, float] = {}
        fvu_s_summary: Dict[str, float] = {}
        lsd_t_summary: Dict[str, float] = {}
        lsd_s_summary: Dict[str, float] = {}
        gates_out: Dict[str, float] = {}

        for idx, band in enumerate(BANDS):
            fvu_t = float(t_metrics["fvu"][band].item())
            fvu_s = float(s_metrics["fvu"][band].item())
            lsd_t = float(t_metrics["lsd"][band].item())
            lsd_s = float(s_metrics["lsd"][band].item())
            fvu_t_summary[band] = fvu_t
            fvu_s_summary[band] = fvu_s
            lsd_t_summary[band] = lsd_t
            lsd_s_summary[band] = lsd_s

            if self.config.gate_mode == "hard":
                g_val = 1.0 if fvu_t < fvu_s else 0.0
            elif self.config.gate_mode == "soft":
                g_val = float(torch.sigmoid(torch.tensor(self.config.soft_alpha * (fvu_s - fvu_t))).item())
            elif self.config.gate_mode == "lsd":
                # G1: spectral-fidelity gate — keep only if teacher's spectrum is closer to truth
                g_val = 1.0 if lsd_t < lsd_s else 0.0
            elif self.config.gate_mode == "consensus":
                # G5: require BOTH FVU and LSD advantage
                g_val = 1.0 if (fvu_t < fvu_s and lsd_t < lsd_s) else 0.0
            elif self.config.gate_mode == "benefit":
                # G4: placeholder — actual gates set by DSOFSDTrainer._calibrate_benefit_gates
                g_val = 0.0
            else:
                raise ValueError(f"Unknown gate_mode: {self.config.gate_mode!r}.")

            self.gate_vector[idx] = g_val
            gates_out[band] = g_val

        self.calibrated = True
        self.calibration_summary = {
            "fvu_teacher": fvu_t_summary,
            "fvu_warm_student": fvu_s_summary,
            "lsd_teacher": lsd_t_summary,
            "lsd_warm_student": lsd_s_summary,
            "gates": gates_out,
        }
        return gates_out

    def _recalibrate_teacher_fft(
        self,
        true_fft: torch.Tensor,
        teacher_fft: torch.Tensor,
        masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Rescale teacher FFT amplitudes per band to match true band energy.

        ``A_T^recal(k) = A_T(k) * sqrt(E_true,b / E_T,b)`` for ``k in band b``.
        This preserves the teacher's spectral *shape* within each band while
        removing the MMSE amplitude-shrinkage bias.
        """
        recal_fft = teacher_fft.clone()
        eps = self.config.eps
        for band, mask in masks.items():
            if not torch.any(mask):
                continue
            # Band energy: mean |FFT|^2 over masked frequencies
            e_true_b = torch.mean(true_fft[..., mask].abs() ** 2).clamp(min=eps)
            e_teacher_b = torch.mean(teacher_fft[..., mask].abs() ** 2).clamp(min=eps)
            scale = torch.sqrt(e_true_b / e_teacher_b)
            recal_fft[..., mask] = teacher_fft[..., mask] * scale
        return recal_fft

    def forward(
        self,
        y_student: torch.Tensor,
        y_true: torch.Tensor,
        y_teacher: torch.Tensor,
        lam_spec: float,
        lam_phase: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute ``L_time + lam_spec * L_spec [+ lam_anchor * L_anchor] [+ lam_phase * L_phase]``."""
        loss_time = F.mse_loss(y_student, y_true)

        # Compute FFTs for student and teacher
        y_s = (y_student if y_student.ndim == 3 else y_student.unsqueeze(-1)).float()
        y_t_target = (y_teacher if y_teacher.ndim == 3 else y_teacher.unsqueeze(-1)).float()
        y_truth = (y_true if y_true.ndim == 3 else y_true.unsqueeze(-1)).float()
        horizon = y_s.shape[1]

        masks, _ = band_masks_rfft_torch(
            horizon,
            fs=self.config.fs,
            low_cutoff=self.config.low_cutoff,
            high_cutoff=self.config.high_cutoff,
            device=y_s.device,
        )

        s_perm = y_s.permute(0, 2, 1).contiguous()
        t_perm = y_t_target.permute(0, 2, 1).contiguous()
        truth_perm = y_truth.permute(0, 2, 1).contiguous()

        student_fft = torch.fft.rfft(s_perm, dim=-1)
        teacher_fft = torch.fft.rfft(t_perm, dim=-1)
        true_fft = torch.fft.rfft(truth_perm, dim=-1)

        # Amplitude recalibration: adjust teacher amplitudes to match true band energy
        if self.config.recalibrate_teacher:
            teacher_fft = self._recalibrate_teacher_fft(true_fft, teacher_fft, masks)

        eps = self.config.eps
        log_amp_student = torch.log(student_fft.abs() + eps)
        log_amp_teacher = torch.log(teacher_fft.abs() + eps)
        sq_log_diff = (log_amp_student - log_amp_teacher) ** 2

        loss_spec = y_student.new_zeros(())
        for idx, band in enumerate(BANDS):
            mask = masks[band]
            if torch.any(mask):
                loss_spec = loss_spec + self.gate_vector[idx] * torch.mean(sq_log_diff[..., mask])

        # Ground-truth spectral anchor: LSD(y_student, y_true)
        loss_anchor = y_student.new_zeros(())
        if self.config.lambda_anchor > 0.0:
            log_amp_true = torch.log(true_fft.abs() + eps)
            sq_log_truth_diff = (log_amp_student - log_amp_true) ** 2
            loss_anchor = torch.mean(sq_log_truth_diff)

        loss_phase = y_student.new_zeros(())
        if lam_phase > 0.0:
            phase_t = torch.angle(teacher_fft)
            phase_s = torch.angle(student_fft)
            phase_diff = 1.0 - torch.cos(phase_s - phase_t)
            for idx, band in enumerate(BANDS):
                mask = masks[band]
                if torch.any(mask):
                    loss_phase = loss_phase + self.gate_vector[idx] * torch.mean(phase_diff[..., mask])

        total = loss_time + lam_spec * loss_spec + self.config.lambda_anchor * loss_anchor + lam_phase * loss_phase
        info = {
            "loss_total": float(total.item()),
            "loss_time": float(loss_time.item()),
            "loss_spec": float(loss_spec.item()),
            "loss_anchor": float(loss_anchor.item()),
            "loss_phase": float(loss_phase.item()),
        }
        return total, info


def discover_dataset_periods(train_data: np.ndarray, seq_len: int, max_periods: int = 3) -> Tuple[List[int], bool]:
    """Automatically extract ACF-verified periodicities and channel dependence from training data."""
    n_samples, _ = train_data.shape
    max_period = min(seq_len, max(8, n_samples // 2))
    avg_series = np.mean(train_data, axis=1) if train_data.ndim == 2 else train_data

    candidates = find_top_periods_fft(avg_series, n_periods=5, min_period=4, max_period=max_period)
    confirmed, _, _ = filter_true_periods_acf(avg_series, candidates)
    if confirmed:
        decision = decide_multi_or_single_cycle(avg_series, confirmed, max_periods=max_periods)
        periods = decision["periods_to_use"] or confirmed[:1]
    elif candidates:
        periods = candidates[:1]
    else:
        periods = [min(24, seq_len)]

    if train_data.ndim == 2:
        corr_metrics = compute_correlation_metrics(train_data)
        use_cd, _ = decide_channel_dependence(corr_metrics)
    else:
        use_cd = False
    return periods, bool(use_cd)


def build_dsof_student(
    seq_len: int,
    pred_len: int,
    channels: int,
    student_type: str = "sor",
    periods: Optional[List[int]] = None,
    use_channel_dependence: bool = False,
) -> nn.Module:
    """
    Instantiate the lightweight student backbone.

    Supports ``CycleNetPro`` backbones (``'sor'`` SORMixer, ``'linear'``,
    ``'mlp'``, ``'cnn'``) or ``'cyclenet_lite'`` (``srp26.CycleNet``).
    """
    stype = student_type.lower()
    if stype == "cyclenet_lite":
        p = periods[0] if periods else min(24, seq_len)
        return CycleNet(seq_len=seq_len, pred_len=pred_len, channels=channels, cycle_len=p)

    return CycleNetPro(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        periods=periods or [min(24, seq_len)],
        use_channel_dependence=use_channel_dependence,
        model_type=stype,
    )


def forward_student(model: nn.Module, x: torch.Tensor, cycle_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Uniform forward helper across ``CycleNetPro(x, cycle_index)`` and ``CycleNet(x)``."""
    if isinstance(model, CycleNetPro):
        if cycle_index is None:
            cycle_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return model(x, cycle_index)
    return model(x)


@torch.no_grad()
def evaluate_predictions_dsof(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    config: DSOFSDConfig,
) -> Dict[str, object]:
    """Compute unified Phase 3 metrics (global MSE/MAE/LSD + per-band MSE/FVU/LSD)."""
    metrics = dsof_band_metrics_torch(
        y_true,
        y_pred,
        fs=config.fs,
        low_cutoff=config.low_cutoff,
        high_cutoff=config.high_cutoff,
        eps=config.eps,
    )
    mae = float(F.l1_loss(y_pred.float(), y_true.float()).item())
    bands_out = {
        band: {
            "mse": round(float(metrics["mse"][band].item()), 6),
            "fvu": round(float(metrics["fvu"][band].item()), 6),
            "lsd": round(float(metrics["lsd"][band].item()), 6),
        }
        for band in BANDS
    }
    return {
        "mse": round(float(metrics["total_mse"].item()), 6),
        "mae": round(mae, 6),
        "lsd": round(float(metrics["global_lsd"].item()), 6),
        "bands": bands_out,
    }


def check_hypothesis(report: Dict[str, object]) -> Dict[str, object]:
    """
    Evaluate the falsifiable DSOF-SD claim from Phase 3 results.

    Holds iff on every active gated band (``g_b > 0``) the distilled student lowers
    or matches band LSD relative to the warm-start student without regressing MSE.
    """
    gates: Dict[str, float] = report["gates"]  # type: ignore[assignment]
    warm = report["warm_student"]  # type: ignore[assignment]
    distilled = report["distilled_student"]  # type: ignore[assignment]

    active_bands = [b for b, g in gates.items() if g > 0.0]
    if not active_bands:
        return {
            "status": "no_transfer",
            "passed": False,
            "reason": "All calibration gates g_b == 0 (teacher showed no per-band advantage under the active gate criterion).",
        }

    band_improved = {
        b: float(distilled["bands"][b]["lsd"]) < float(warm["bands"][b]["lsd"])  # type: ignore[index]
        for b in active_bands
    }
    mse_non_regressing = float(distilled["mse"]) <= float(warm["mse"])  # type: ignore[index]
    passed = all(band_improved.values()) and mse_non_regressing
    return {
        "status": "supported" if passed else "refuted",
        "passed": passed,
        "active_bands": active_bands,
        "band_lsd_improved": band_improved,
        "mse_non_regressing": mse_non_regressing,
    }


class DSOFSDTrainer:
    """
    Orchestrates DSOF-SD Phases 1-3:
      1. Warm-start student on ``L_time`` and freeze DSOF band gates ``g_b``.
      2. Anneal ``lambda_spec: 0 -> lambda_max`` with validation MSE early stopping.
      3. Evaluate ``{teacher, warm_student, distilled_student}`` closed-loop metrics.
    """

    def __init__(
        self,
        student: nn.Module,
        config: Optional[DSOFSDConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or DSOFSDConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.loss_fn = DSOFSpectralLoss(self.config).to(self.device)
        self.warm_state: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _anneal_lambda(epoch: int, total_epochs: int, lam_max: float) -> float:
        if total_epochs <= 1:
            return lam_max
        return lam_max * float(epoch) / float(total_epochs - 1)

    @torch.no_grad()
    def _collect_outputs(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.student.eval()
        ys, yts, yss = [], [], []
        for x_b, y_b, yt_b, c_b in loader:
            pred = forward_student(self.student, x_b.to(self.device), c_b.to(self.device))
            ys.append(y_b.cpu())
            yts.append(yt_b.cpu())
            yss.append(pred.cpu())
        y_all = torch.cat(ys, dim=0)
        yt_all = torch.cat(yts, dim=0)
        ys_all = torch.cat(yss, dim=0)
        return y_all, yt_all, ys_all, torch.empty(0)

    def _run_val_mse(self, val_loader: DataLoader) -> float:
        self.student.eval()
        total_sq, total_n = 0.0, 0
        with torch.no_grad():
            for x_v, y_v, _, c_v in val_loader:
                pred = forward_student(self.student, x_v.to(self.device), c_v.to(self.device))
                total_sq += float(F.mse_loss(pred, y_v.to(self.device), reduction="sum").item())
                total_n += int(y_v.numel())
        return total_sq / max(1, total_n)

    def _calibrate_benefit_gates(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warm_val_metrics: Dict[str, object],
    ) -> Dict[str, float]:
        """
        G4 benefit gate: for each band, run a short distillation with only that
        band active, then keep it iff *both* band LSD and band MSE improve on
        the validation set relative to the warm-start student.
        """
        probe_epochs = self.config.benefit_probe_epochs
        gates: Dict[str, float] = {}
        probe_details: Dict[str, Dict[str, float]] = {}
        warm_bands = warm_val_metrics["bands"]  # type: ignore[index]

        for idx, band in enumerate(BANDS):
            # Reset student to warm state
            assert self.warm_state is not None
            self.student.load_state_dict({k: v.to(self.device) for k, v in self.warm_state.items()})

            # Set a single-band gate vector
            probe_gate = torch.zeros(len(BANDS), dtype=torch.float32, device=self.device)
            probe_gate[idx] = 1.0
            self.loss_fn.gate_vector.copy_(probe_gate)

            # Short distillation with only this band
            probe_opt = torch.optim.AdamW(
                self.student.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
            )
            for _ in range(probe_epochs):
                self.student.train()
                for x_b, y_b, yt_b, c_b in train_loader:
                    probe_opt.zero_grad()
                    pred = forward_student(self.student, x_b.to(self.device), c_b.to(self.device))
                    loss, _ = self.loss_fn(
                        pred,
                        y_b.to(self.device),
                        yt_b.to(self.device),
                        lam_spec=self.config.lambda_max,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                    probe_opt.step()

            # Evaluate on validation
            y_val, yt_val, ys_probe_val, _ = self._collect_outputs(val_loader)
            probe_val_metrics = evaluate_predictions_dsof(y_val, ys_probe_val, self.config)

            warm_lsd_b = float(warm_bands[band]["lsd"])  # type: ignore[index]
            warm_mse_b = float(warm_bands[band]["mse"])  # type: ignore[index]
            probe_lsd_b = float(probe_val_metrics["bands"][band]["lsd"])  # type: ignore[index]
            probe_mse_b = float(probe_val_metrics["bands"][band]["mse"])  # type: ignore[index]

            g_val = 1.0 if (probe_lsd_b < warm_lsd_b and probe_mse_b < warm_mse_b) else 0.0
            gates[band] = g_val
            probe_details[band] = {
                "warm_lsd": warm_lsd_b,
                "probe_lsd": probe_lsd_b,
                "warm_mse": warm_mse_b,
                "probe_mse": probe_mse_b,
                "gate": g_val,
            }
            print(
                f"  [G4 probe {band}] warm_lsd={warm_lsd_b:.4f} -> probe_lsd={probe_lsd_b:.4f}, "
                f"warm_mse={warm_mse_b:.6f} -> probe_mse={probe_mse_b:.6f} => gate={g_val:.0f}",
                flush=True,
            )

        # Restore warm state and set final gates
        self.student.load_state_dict({k: v.to(self.device) for k, v in self.warm_state.items()})
        for idx, band in enumerate(BANDS):
            self.loss_fn.gate_vector[idx] = gates[band]
        self.loss_fn.calibrated = True
        self.loss_fn.calibration_summary["gates"] = gates
        self.loss_fn.calibration_summary["gate_mode"] = "benefit"
        self.loss_fn.calibration_summary["benefit_probes"] = probe_details
        return gates

    def fit_and_evaluate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, object]:
        """Execute full Phase 1-3 training and return structured metrics."""
        t0 = time.time()
        param_count = sum(p.numel() for p in self.student.parameters() if p.requires_grad)

        # ---- Phase 1a: Warm-start on L_time only ----
        warm_opt = torch.optim.AdamW(self.student.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        for _ in range(self.config.warm_epochs):
            self.student.train()
            for x_b, y_b, _, c_b in train_loader:
                warm_opt.zero_grad()
                pred = forward_student(self.student, x_b.to(self.device), c_b.to(self.device))
                loss = F.mse_loss(pred, y_b.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                warm_opt.step()

        self.warm_state = copy.deepcopy({k: v.cpu() for k, v in self.student.state_dict().items()})

        # ---- Phase 1b: Calibrate and freeze band gates on calibration split (val) ----
        y_cal, yt_cal, ys_warm_cal, _ = self._collect_outputs(val_loader)
        gates = self.loss_fn.calibrate_band_gates(y_cal, yt_cal, ys_warm_cal)

        # For benefit gate (G4), override with empirical per-band probes
        if self.config.gate_mode == "benefit":
            warm_val_metrics = evaluate_predictions_dsof(y_cal, ys_warm_cal, self.config)
            gates = self._calibrate_benefit_gates(train_loader, val_loader, warm_val_metrics)

        # Measure warm-start baseline on test before distillation
        # (restore warm state first, in case benefit gate changed it)
        self.student.load_state_dict({k: v.to(self.device) for k, v in self.warm_state.items()})
        y_test, yt_test, ys_warm_test, _ = self._collect_outputs(test_loader)
        warm_test_metrics = evaluate_predictions_dsof(y_test, ys_warm_test, self.config)
        teacher_test_metrics = evaluate_predictions_dsof(y_test, yt_test, self.config)

        # ---- Phase 2: Annealed DSOF-guided spectral distillation ----
        distill_opt = torch.optim.AdamW(self.student.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            distill_opt, T_max=max(1, self.config.distill_epochs), eta_min=1e-5
        )
        best_val_mse = float("inf")
        best_weights: Optional[Dict[str, torch.Tensor]] = copy.deepcopy(self.warm_state)
        patience_left = self.config.patience
        history: List[Dict[str, float]] = []

        # If every gate is closed and there is no ground-truth anchor, Phase 2
        # would degenerate into extra L_time training (an unfair "gain" vs the
        # warm-start baseline). Skip it so distilled == warm.
        all_gates_zero = all(float(g) == 0.0 for g in gates.values())
        phase2_degenerate = all_gates_zero and self.config.lambda_anchor <= 0.0
        if phase2_degenerate:
            print(
                "  [Phase 2 skipped] all gates == 0 and lambda_anchor == 0 -> distilled student == warm student.",
                flush=True,
            )
        distill_epochs_effective = 0 if phase2_degenerate else self.config.distill_epochs

        for epoch in range(distill_epochs_effective):
            lam_spec = self._anneal_lambda(epoch, self.config.distill_epochs, self.config.lambda_max)
            self.student.train()
            epoch_stats = {"epoch": epoch + 1, "lam_spec": lam_spec, "loss_total": 0.0, "loss_time": 0.0, "loss_spec": 0.0, "loss_anchor": 0.0}
            n_batches = 0

            for x_b, y_b, yt_b, c_b in train_loader:
                distill_opt.zero_grad()
                pred = forward_student(self.student, x_b.to(self.device), c_b.to(self.device))
                loss, info = self.loss_fn(
                    pred,
                    y_b.to(self.device),
                    yt_b.to(self.device),
                    lam_spec=lam_spec,
                    lam_phase=self.config.lambda_phase,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                distill_opt.step()

                epoch_stats["loss_total"] += info["loss_total"]
                epoch_stats["loss_time"] += info["loss_time"]
                epoch_stats["loss_spec"] += info["loss_spec"]
                epoch_stats["loss_anchor"] += info["loss_anchor"]
                n_batches += 1

            scheduler.step()
            for key in ("loss_total", "loss_time", "loss_spec", "loss_anchor"):
                epoch_stats[key] /= max(1, n_batches)

            val_mse = self._run_val_mse(val_loader)
            epoch_stats["val_mse"] = val_mse
            history.append(epoch_stats)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_weights = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_weights is not None:
            self.student.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

        # ---- Phase 3: Closed-loop test evaluation ----
        y_test_d, _, ys_distill_test, _ = self._collect_outputs(test_loader)
        distilled_test_metrics = evaluate_predictions_dsof(y_test_d, ys_distill_test, self.config)

        report: Dict[str, object] = {
            "params": param_count,
            "runtime_sec": round(time.time() - t0, 2),
            "config": asdict(self.config),
            "gates": gates,
            "phase2_degenerate": phase2_degenerate,
            "calibration": self.loss_fn.calibration_summary,
            "teacher": teacher_test_metrics,
            "warm_student": warm_test_metrics,
            "distilled_student": distilled_test_metrics,
            "history": history,
        }
        report["hypothesis"] = check_hypothesis(report)
        return report

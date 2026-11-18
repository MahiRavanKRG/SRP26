# SRP26 Controlled Re-Run Protocol (v2)

> **Status:** DRAFT for approval · 2026-11-18
> **Goal:** Re-run every experiment in this project under one controlled, reproducible, field-standard protocol, so that every number in the paper can be regenerated on another machine from one command.
> **Supersedes for reporting:** all numbers in `results/` and `fresh_experiments/results/` produced before v2. Old results are **archived, never deleted**.

---

## 1. Why: contamination found in the existing results

The old experiments are not wrong because of one bug. They are **not comparable with each other**, because each track used its own data loader, trainer, and evaluation. Found during the 2026-11-18 audit:

| # | Contamination | Where | Effect |
|---|---|---|---|
| C1 | **Different training recipes per track.** CycleNet: Adam, constant LR, no AMP. Transform ablation: AdamW + cosine + AMP + grad-clip. | `experiment_cyclenet.py` vs `srp26/transform_ablation.py` | CycleNet vs baseline gaps mix *model* effect with *optimizer* effect. |
| C2 | **Foundation models scored on only 15 test windows** instead of the full test set. | `fresh_experiments/benchmark_fourier_foundation_suite.py`, `run_expanded_foundation_suite.py` | TimesFM ETTh1 reads 0.4601 there vs 0.3950 on the full test set. Not comparable to anything else. |
| C3 | **Single seed** for the main CycleNet benchmark (protocol requires 3). | `results/cyclenet/` | Cannot separate real gains from seed noise. |
| C4 | **Not deterministic.** No `cudnn.deterministic`, no `use_deterministic_algorithms`, DataLoader shuffle not seeded through a `torch.Generator`, AMP on in some tracks. | all runners | Same seed may not give the same number twice. |
| C5 | **Environment drift.** Local is Python 3.13 + `uv.lock`; cluster is Python 3.12.3 with packages installed ad hoc via `uv pip`. | Bigmama `~/SRP26/.venv` | Results depend on the machine they ran on. |
| C6 | **No provenance in result files.** No git commit, no config hash, no package versions, no dataset checksum. Code was uncommitted when runs were made. | all `results/*.json` | Cannot tell which code produced a number. |
| C7 | **Incomplete grid.** Transform ablation is missing electricity H336/H720 and traffic H192/H336/H720; CycleNet main table lacks Electricity/Traffic/Solar in the seeded runner. | `results/transform_ablation/`, `results/cyclenet/` | Conclusions extrapolate beyond the data. |
| C8 | **Overstated significance.** Wilcoxon test over overlapping stride-1 test windows treated as independent. | `scripts/aggregate_transform_ablation.py` | p < 1e-6 is not trustworthy. |
| C9 | **Baselines mixed from different sources** (published numbers with L=96 and our runs with L=336 in one table), and an unsourced DLinear 0.6050. | `paper/main.tex` tables, abstract | Unfair comparison. |
| C10 | **"Automated period discovery" has no control.** The original CycleNet with the known period (24/96/144/168) was never run in the same harness. | — | The core claim (automation helps or is harmless) is untested. |

---

## 2. Principles (the "sterile lab" rules)

1. **One bench.** One data pipeline (`srp26/data.py`), one trainer, one evaluator, one result schema for *every* model. Models differ only in `model.forward`.
2. **Everything is a config.** Each run is defined by a YAML config (dataset, H, model, hyperparameters, seed). The config hash is the run ID. No hidden defaults in scripts.
3. **Seeded and deterministic where cheap (best effort, not a gate).** Fixed seeds for `random`, `numpy`, `torch`, CUDA, and the DataLoader generator; `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; AMP **off** (FP32). This is a debugging aid (a changed number means changed code, not chance), not a scientific requirement. Field standard is that randomness is *measured* by multi-seed mean ± std; reproduction on another machine means landing **within the reported seed spread**, not bit-identical numbers (different GPUs order floating-point operations differently).
4. **Locked environment.** Same Python (3.13) and the same `uv.lock` on laptop and cluster (`uv sync --frozen`). Package versions, GPU name, CUDA/driver, and git commit are written into every result file. Runs from a dirty git tree are refused.
5. **Locked data.** SHA-256 checksum of every dataset file recorded in `data_sets/CHECKSUMS.sha256`; the runner refuses a file whose checksum differs.
6. **Test set is touched once.** Hyperparameters and early stopping use validation only. Test metrics are computed after training finishes and never fed back.
7. **Immutable results.** v2 results go to `results/v2/<run_id>.json` (+ per-window losses as `.npy`). Never overwritten. Old results move to `results/archive_pre_v2/`.
8. **Report everything planned.** Every seed and every cell in the pre-registered grid below is reported, including losses. No best-seed picking.

---

## 3. Field-standard evaluation (time-series forecasting good practice)

| Item | v2 rule | Source of convention |
|---|---|---|
| Lookback | L = 336 for all trained models | Protocol §4.1; PatchTST/DLinear long-input setting |
| Horizons | H ∈ {96, 192, 336, 720} | Informer/Autoformer benchmark convention |
| Splits | ETT 12/4/4 months; others 70/10/20, chronological | Autoformer/PatchTST convention |
| Scaling | StandardScaler fit on train only; metrics on the scaled data | Standard LTSF reporting |
| Test windows | **All** stride-1 windows; `drop_last=False` on test | TFB (Qiu et al., VLDB 2024) flagged `drop_last` as a known benchmark bug |
| Metrics | MSE, MAE (primary); DSOF band FVU/LSD (diagnostic) | — |
| Naive baselines | Repeat-last and seasonal-naive always reported | Forecasting practice (Hyndman & Athanasopoulos, FPP3): a model must beat naive to be useful |
| Seeds | 3 fixed seeds `[2024, 2025, 2026]` (minimum; 5 if budget allows) | Protocol §4.6 |
| Hyperparameters | Same budget for every trained model: LR ∈ {1e-4, 5e-4, 1e-3} chosen on **validation** MSE (seed 2024), then frozen for all seeds | Fair-comparison practice (TFB, BasicTS) |
| Significance | **Primary (all cells):** seed-level mean ± std, plus win counts across seeds. **Secondary (only for the few comparisons claimed in the paper):** Diebold–Mariano test with autocorrelation-robust (HAC) variance on per-window losses, Holm correction | Mean ± std is the LTSF norm; Diebold & Mariano (1995) is standard in classical forecasting and an added strength, not a requirement |
| Published baselines | Shown in a **separate** table, clearly labelled with source and their own L; never mixed into the controlled table | Protocol §4.7 |

---

## 4. Pre-registered experiment grid

Datasets (8): ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Solar. Horizons (4). Seeds (3). → **96 runs per trained model.**

### Tier 1: Core paper results (trained, same harness)

| Group | Models | Runs |
|---|---|---|
| Naive references (no training) | Repeat-last, Seasonal-naive | 64 (seed-free) |
| Linear baselines | Linear (RevIN), DLinear (RevIN) | 192 |
| Period-discovery control | CycleNet-original (known period, Lin et al. 2024) | 96 |
| **Proposed** | CycleNet-Pro: Linear, MLP, CNN, SOR (automatic periods, channel-dependence auto) | 384 |
| Channel-dependence ablation | CycleNet-Pro ×4 with channel mixing forced off | 384 |
| SOTA baseline | PatchTST/42 (official hyperparameters, same harness) | 96 |

### Tier 2: Frequency-question experiments

| Group | Models | Runs |
|---|---|---|
| Transform ablation | identity (= Linear above), dlinear, rfft, dct, stft, dwt_haar, dwt_db4 | 480 new (identity shared) |
| Foundation models, zero-shot, **full test set** | TimesFM 2.5, Chronos-Bolt, Chronos-T5, TTM-R1, each ± Fourier decoupling (FCMath) | 256 evaluations (Chronos sampling seeded) |
| DSOF-SD distillation | linear/SOR/CNN students × gates {hard, soft, benefit, consensus}, TimesFM teacher | ETTh1/ETTh2 × 4 H × 3 seeds (scope extendable) |

### Tier 3: Exploratory tracks (decision needed: rerun or archive as "exploratory, not reported")

Early freq/wavelet-enhanced DLinear (subsumed by Tier 2 transform ablation), Two-Stage Fourier-Attention, Fourier-PatchAttention, Fourier-TQNet, Chronos LoRA fine-tune.

HDMixer: cited from AAAI-24 in the published-baselines table only (not re-implemented) unless the supervisor requires it.

**Estimated compute:** Tier 1 + 2 ≈ 2,000 short runs. Most lightweight runs take seconds to minutes; Traffic/Electricity/Solar and PatchTST dominate. Fits within a few days on `gpu-211` (4× RTX PRO 6000).

---

## 5. Phases

| Phase | What | Exit criterion |
|---|---|---|
| **P0 Clean bench** | Commit current work; move old results to `results/archive_pre_v2/`; write dataset checksums; rebuild cluster env with Python 3.13 + `uv sync --frozen` | Same `uv.lock` hash on both machines; clean git tree |
| **P1 Unified harness** | `srp26/bench/` with config loader, single trainer, evaluator, result schema with provenance; port every model; `configs/` for the grid | Tests pass: no-leakage (scaler/test boundary), split sizes, Parseval identity. Sanity check (non-blocking): same seed twice on one machine ⇒ same MSE |
| **P2 Calibration** | ETTh1 H96, all Tier-1 models × 3 seeds, locally and on Bigmama | Laptop vs cluster means agree **within the seed spread** (differences documented) |
| **P3 LR selection** | Validation-only LR grid, seed 2024 | Frozen `configs/selected_hparams.yaml` committed |
| **P4 Full grid** | Submit Tier 1, then Tier 2 on Bigmama as Slurm array jobs | Every pre-registered run present (completeness check script) |
| **P5 Analysis** | Tables (mean ± std), Diebold–Mariano + Holm, DSOF diagnostics, old-vs-v2 reconciliation note | Every paper number traceable to `results/v2/` |
| **P6 Paper** | Update `paper/main.tex` only from v2 | Compiles with 0 errors |

---

## 6. Reproducing on another machine (target end state)

```bash
git clone <repo> && cd SRP26 && git checkout <v2 tag>
uv sync --frozen --all-extras
uv run python data_sets/download_data.py && uv run python -m srp26.bench.verify_data
uv run python -m srp26.bench.run --config configs/v2/ETTh1/H96/cyclenet_pro_linear.yaml --seed 2024
uv run python -m srp26.bench.report      # regenerates every table
```

---

## 7. Open decisions (defaults in bold)

1. Seeds: **3** (protocol) or 5.
2. Tier 3 exploratory tracks: **archive as exploratory, do not rerun**, or rerun.
3. PatchTST: **rerun in harness**; HDMixer: **cite only**.
4. Cluster login: keep password in `.env` (works today) or switch to an SSH key (recommended; set up by the user).

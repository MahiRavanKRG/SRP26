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
| C11 | **Diagnostic metrics misdefined or misread** (audit 2026-11-18, details in §3a): fixed 0.1/0.4 cycles/step bands put the daily and weekly cycles in the *low* band, not "mid"; FVU ≥ 1 in unpredictable bands read as failure although an MSE-optimal forecast scores exactly 1.0; LSD rewards wrong forecasts (optimal 56.1 dB vs optimal+fake noise 4.7 dB) and has 4 different implementations; rolling failure rate averages 69%, is identical across models and correlates 0.88 with MSE; coherence is always 1.0 with one segment (even vs pure noise). | `srp26/spectral.py`, `cluster/`, `fresh_experiments/` | Diagnostic conclusions in the paper are not supported as stated. |

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
| Metrics | MSE, MAE (primary); skill vs seasonal naive; diagnostics per §3a | — |
| Naive baselines | Repeat-last and seasonal-naive always reported | Forecasting practice (Hyndman & Athanasopoulos, FPP3): a model must beat naive to be useful |
| Seeds | 3 fixed seeds `[2024, 2025, 2026]` (minimum; 5 if budget allows) | Protocol §4.6 |
| Hyperparameters | Same budget for every trained model: LR ∈ {1e-4, 5e-4, 1e-3, 5e-3} chosen on **validation** MSE (seed 2024), then frozen for all seeds | Fair-comparison practice (TFB, BasicTS) |
| Significance | **Primary (all cells):** seed-level mean ± std, plus win counts across seeds. **Secondary (only for the few comparisons claimed in the paper):** Diebold–Mariano test with autocorrelation-robust (HAC) variance on per-window losses, Holm correction | Mean ± std is the LTSF norm; Diebold & Mariano (1995) is standard in classical forecasting and an added strength, not a requirement |
| Published baselines | Shown in a **separate** table, clearly labelled with source and their own L; never mixed into the controlled table | Protocol §4.7 |

---

## 3a. Metric set (v2)

One implementation per metric, in `srp26/bench/metrics.py`, used by every model and every track. All metrics are computed on the full test set in the train-scaled space.

### Primary (accuracy)

| Metric | Definition | Why |
|---|---|---|
| MSE, MAE | Mean over windows, horizon steps, channels | LTSF standard; comparable to published tables |
| Skill vs seasonal naive | `1 − MSE_model / MSE_snaive` (and the MAE analogue, MASE-style). Seasonal period = the dataset's known daily period | Forecasting-practice standard (Hyndman & Athanasopoulos, FPP3); puts all datasets on one scale; > 0 means the model beats naive |

### Diagnostic (the frequency lens: *which time scale* a model gets wrong)

| Metric | Definition | Interpretation rule |
|---|---|---|
| **Band MSE (Parseval)** | Exact split of MSE over the H-length forecast window into 3 bands defined by **period in real time**, per dataset (below). `mse_slow + mse_daily + mse_fast = MSE` (asserted to 1e-8) | Report each band's MSE and its share of total MSE |
| **Band FVU** | `mse_band / true_power_band` (slow band excludes the per-window mean) | **1.0 = predicts nothing in that band**, which is *correct* for unpredictable content. Only FVU clearly > 1 (adds wrong content) or < 1 (captures real structure) is informative. Never call FVU ≈ 1 a failure by itself |
| **Band energy ratio** | `pred_power_band / true_power_band` | < 1 = smoothing, which is expected for MSE-trained models. Descriptive, not a score |
| **Error vs lead time** | MSE at each horizon step 1…H (mean over windows/channels) | How fast accuracy degrades with distance; saved as a vector |
| **Error over test time** | Per-window MSE across the test period (saved `.npy`), plotted against calendar time, alongside the seasonal-naive curve | Localizes *when* a model breaks (regime shifts); replaces the rolling failure rate |

**Physical bands** (period *T* in hours; converted to steps per dataset; DC/mean goes in "slow"):

| Band | Period | Contains |
|---|---|---|
| slow | T > 36 h | trend, level, multi-day and weekly |
| daily | 8 h ≤ T ≤ 36 h | 24 h cycle and its 12 h harmonic |
| fast | T < 8 h | intra-day fluctuation / noise |

Sampling: ETTh*, Electricity, Traffic 1 h; ETTm* 15 min; Weather, Solar 10 min. The forecast window resolves a period only if T ≤ H·Δt. If the window is shorter than ~1 day (Weather/Solar at H=96 = 16 h), the daily band is not resolvable and is reported as merged into "slow" (marked in tables), never silently.

### Secondary (reported in an appendix only, clearly labelled)

| Metric | Rule |
|---|---|
| LSD | One implementation only (rFFT of the forecast window, dB, per band). Labelled **"spectral realism"**, never "fidelity" or "accuracy", and always shown next to MSE, because it ignores phase and rewards adding noise |

### Retired (not computed in v2)

| Metric | Reason |
|---|---|
| Rolling failure rate / mean peak rolling MSE | Self-referential threshold; ~69% for every model; corr 0.88 with MSE |
| Spectral coherence (`spectral_coherence_analysis`) | One Welch segment ⇒ identically 1.0 |
| Quarter-split bands (`compute_spectral_diagnostics`), Welch-LSD variants | Duplicate, inconsistent definitions |

### Metric tests (part of P1)

Parseval sum equals MSE; perfect forecast ⇒ all band MSE = 0; MSE-optimal smooth forecast of sinusoid+noise ⇒ FVU_fast ≈ 1; known 24 h sinusoid lands in the "daily" band for every dataset's Δt; seasonal naive scores skill = 0 against itself.

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
| **P1 Unified harness** | `srp26/bench/` with config loader, single trainer, evaluator, result schema with provenance; port every model; `configs/` for the grid | Tests pass: no-leakage (scaler/test boundary), split sizes, metric tests (§3a). Sanity check (non-blocking): same seed twice on one machine ⇒ same MSE |
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

---

## 8. Implementation record (P0/P1, 2026-11-18)

| Item | Decision |
|---|---|
| Harness | `srp26/bench/` (`env`, `data`, `metrics`, `models`, `train`, `run`, `grid`); gate tests `tests/test_bench.py` (16 tests) |
| Environment | Python 3.13 + `uv.lock`. PyTorch pinned to **2.8.0+cu128** (the previous 2.6.0+cu124 cannot run on Bigmama's Blackwell RTX PRO 6000, sm_120). Local env lives at `D:\uni\.envs\srp26-v2` (`UV_PROJECT_ENVIRONMENT`), outside the repo, because the sandbox leaves a low-integrity ACL on the repo root that cripples new executables created inside it |
| Datasets | SHA-256 in `data_sets/CHECKSUMS.sha256`; cluster copies verified identical |
| Trainer (all trained models) | Adam, cosine LR over `max_epochs`, FP32, MSE loss, batch 32, 30 epochs, patience 5, best-val checkpoint restore |
| LR selection | Grid {1e-4, 5e-4, 1e-3, 5e-3} (5e-3 added: CycleNet/DLinear reference scripts use LRs above 1e-3), seed 2024, validation MSE only, test never evaluated. Selected once on Bigmama, then **the same `configs/v2/selected_lr.json` is used by both result sets** so they differ only in hardware |
| CycleNet-Pro | Imported unchanged from `models/cyclenet/cyclenet_pro_Full.py`: period discovery on the mean of raw training channels (FFT top-5 → ACF filter → multi-cycle gain test); channel dependence from the correlation diagnostic. `experiment_cyclenet.py` used the last channel instead; v2 follows the Full (four-backbone) version. Detected periods are stored in every result |
| Period control | `cyclenet_fixed` = the same CycleNet-Pro code with the CycleNet-paper cycle length (24/96/144/168), linear backbone, no channel mixing |
| Linear / DLinear / transforms | `TransformForecaster` (parameter-free RevIN + transform + shared linear head); `linear` = identity, `dlinear` = moving-average decomposition |
| PatchTST | Re-implementation of PatchTST/42 (patch 16, stride 8, 3 layers, BatchNorm); ETTh1/2: d_model 16, heads 4, d_ff 128, dropout 0.3; others: 128/16/256/0.2. LR 1e-4, 100 epochs, patience 10, batch 128 (32 on Electricity/Traffic/Solar); not LR-tuned. No residual attention (documented deviation) |
| Seasonal naive | Repeat the last day of the lookback; also evaluated inside every run so `skill_mse` is per run |
| Bigmama | Every job pinned to `--gres=gpu:6000:1` (RTX PRO 6000) so the whole set comes from one GPU type; code shipped as `git archive` of a clean commit (`COMMIT` file recorded in every result) |
| Local set | All final runs except PatchTST on Electricity/Traffic/Solar (6 GB laptop GPU) |
| Order | Tier 1 + transform ablation first (this grid). Tier 2 foundation models and DSOF-SD follow in a second grid once Tier 1 is complete |
| Band split (amended) | MSE = **level** (error in the window mean, DC bin) + slow + daily + fast. The old DSOF low band mixed level error into the numerator but excluded it from the denominator (slow FVU 2.1 in smoke test); level is now its own component with no FVU |
| Smoke test | All 16 models run end-to-end on ETTh1 H96 (laptop). CycleNet-Pro auto-selects period [24] and no channel mixing on ETTh1, so it is numerically identical to `cyclenet_fixed` there |

---

## 9. Stage 2 implementation record (Tier 2: foundation models + DSOF-SD, 2026-11-20)

| Item | Decision |
|---|---|
| Code | `srp26/bench/foundation.py` (zero-shot), `srp26/bench/dsof.py` (distillation), additive helpers in `grid.py` (`stage2-jobs`, `stage2-check`), `report.py` (`--foundation-compare`), `run.py` (`--jobs` dispatches on `phase`: `foundation`, `dsof_teacher`, `dsof`). Tests: `tests/test_bench_stage2.py` |
| Data / metrics | Both tracks use `build_bundle` (train-only scaler, canonical splits, checksum) and `StreamingEvaluator` with the seasonal naive of the daily period: the same metric set as Tier 1, in scaled space. Per-window MSE **and MAE** are saved (`.wmse.npy`, `.wmae.npy`); `StreamingEvaluator` gained `window_abs_errors()` and Tier-1 `final` runs now also save `.wmae.npy` (additive; earlier runs only have `.wmse.npy`) |
| Foundation: evaluation windows | **Deviation from §4 "full test set":** non-overlapping test windows, stride = H, starting at test window 0 (dataset indices 0, H, 2H, ...). This covers the whole test period exactly once (only a tail shorter than H at the end is not a window start). Full stride-1 evaluation is infeasible for Chronos-T5 (20 autoregressive samples) on Traffic. The indices are stored in every result (`eval_windows.indices`) |
| Foundation vs trained comparison | `python -m srp26.bench.report --foundation-compare <set> [--trained-set <set>]` re-scores every trained `final` run on the same stride-H windows from its stride-1 per-window losses (MSE from `.wmse.npy`, MAE from `.wmae.npy` when present) -> `results/v2/<set>/foundation_vs_trained_<trained>.csv` |
| TimesFM 2.5 | `TimesFMTeacher` of `models/dsof_sd.py` reused unchanged (normalize_inputs, continuous quantile head, flip invariance, infer_is_positive=False, fix_quantile_crossing); `max_context=336` is compiled as 352 (patch 32; the 16 extra steps are zero-padded and masked), `max_horizon=H` compiled as a multiple of 128 (H=720 -> 768, verified to run). Point = point output (median quantile). FP32 |
| Chronos-Bolt | `amazon/chronos-bolt-base`, bf16; point = 0.5 quantile, index looked up from `pipeline.quantiles` (= 4 for [0.1..0.9]); H > 64 uses the pipeline's own quantile roll-out (`limit_prediction_length=False`) |
| Chronos-T5 | `amazon/chronos-t5-tiny`, bf16, **20 samples**, point = median over samples (`torch.quantile(0.5)`, mean of the two middle samples), torch seeded with the run seed immediately before inference; H > 64 via the pipeline roll-out. The old suite's `num_samples=1` (one random draw) is not replicated |
| TTM-R1 | H = 96 only (other H refused with an error); context 512 by left-padding the 336 lookback with its first value (as the old suite), recorded in `model_notes`. **Unavailable in the locked env:** `tsfm_public` (granite-tsfm) is not installed and adding it would change `uv.lock`, so TTM jobs fail with a clear `RuntimeError` until the dependency is approved. The TTM weights are also not in the local HF cache |
| Fourier base (reference) | Zero-parameter: lookback mean + `y_base` of `extract_and_extrapolate_fourier_base_torch(top_k=3)`. The old suite scored `y_base` alone, whose mean is 0 (DC bin excluded) |
| Fourier decoupling (`--fourier`) | Model forecasts `x_res = x - x_base` (top-3 non-DC harmonics removed, mean kept), prediction = forecast + `y_base`; result name gets suffix `_fourier` |
| Foundation grid | 8 datasets x 4 H x {TimesFM, Chronos-Bolt, Chronos-T5} x {plain, Fourier} + TTM (H=96) x {plain, Fourier} + Fourier base = **240** evaluations, seed 2024 (`configs/v2/jobs_stage2_foundation.jsonl`; the §4 estimate of 256 assumed TTM at every H) |
| DSOF-SD confound | `models/dsof_sd.py` compared the distilled student (5 warm + up to 25 distillation epochs) with the warm student (5 epochs). v2 adds gate **`control`**: the identical `DSOFSDTrainer`, schedule, optimizer, initialisation and data order with `lambda_max = 0`. Distillation is judged against the control; warm-student metrics are still stored |
| DSOF-SD Phase-2 skip | New `DSOFSDConfig.skip_degenerate_phase2` (default True = original behaviour); v2 sets **False** for every gate, so the control trains the full schedule and an all-zero-gate run is identical to the control (test-verified) instead of collapsing to the warm student. `all_gates_zero` is stored per run |
| DSOF-SD RNG alignment | The benefit gate's per-band probes (3 x 3 epochs) are wrapped so the DataLoader generator and torch RNG states are restored afterwards; Phase 2 then sees the same batch order as the control |
| DSOF-SD setup | Student = `build_model("cyclenet_pro_<linear/sor/cnn>")` (v2 period discovery and channel-dependence decision), inner `CycleNetPro` passed to the trainer. Method defaults kept: AdamW lr 3e-4, wd 1e-3, warm 5, distill 25, patience 7, lambda_max 0.2, grad-clip 1, batch 32 (shuffled with the seeded generator, `drop_last=True`). Control gates are calibrated with the hard rule for the record only (they multiply a zero weight). Seeds 2024/2025/2026 |
| DSOF-SD teacher cache | `python -m srp26.bench.dsof cache-teacher` runs TimesFM over all train/val/test windows in dataset order -> `cache/teacher/<teacher>_<D>_H<H>.pt` (git-ignored) with dataset SHA-256, L, H, window counts and each window's last observation; every run verifies count, offset and window alignment before training. `configs/v2/jobs_stage2_teacher.jsonl` (8 jobs) must run before the DSOF jobs (a missing cache is otherwise built by the run itself) |
| DSOF-SD evaluation | Final student, warm student and teacher are each scored on the **full stride-1 test set** with `StreamingEvaluator` (`metrics`, `metrics_warm`, `metrics_teacher`); the trainer's own report (gates, calibration, history, hypothesis, old fixed-cutoff DSOF band metrics) is kept under `trainer_report` |
| DSOF-SD grid | ETTh1, ETTh2 x 4 H x {linear-hard, linear-control, sor-hard, sor-soft, sor-benefit, sor-consensus, sor-control, cnn-benefit, cnn-control} x 3 seeds = **216** runs (`configs/v2/jobs_stage2_dsof.jsonl`) |
| Result paths | `results/v2/<set>/foundation/<D>/H<H>/<model>[_fourier]/s<seed>.json`, `results/v2/<set>/dsof/<D>/H<H>/<student>_<gate>/s<seed>.json` (a non-TimesFM teacher appends `_teacher-<name>`), atomic write, skip if present, provenance as Tier 1 |
| Smoke-only flags | `--max-windows N` (foundation), `--epochs-override W,D` (DSOF), `--hf-id-override` (Chronos-Bolt with `--max-windows` only); all recorded in the result JSON, never used in the grids |
| Smoke test (laptop, ETTh1, set `stage2_smoke`) | Foundation H96, first 3 stride-H windows: TimesFM, TimesFM+Fourier, Chronos-T5, Chronos-T5+Fourier, Fourier base run; TimesFM H720 runs. Chronos-Bolt-**base** weights are not in the local HF cache (config only), so its code path was smoke-tested with `chronos-bolt-tiny` via `--hf-id-override` (H96, H96+Fourier, H192). DSOF sor-control and sor-hard (1 warm + 1 distill epoch) with the Fourier-base teacher; both share the identical warm student. A TimesFM teacher slice reproduces the foundation TimesFM MSE on the same windows (alignment check) |

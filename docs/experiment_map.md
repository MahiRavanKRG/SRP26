# SRP26 Experiment Map — Internal Work Tracker

> **Purpose:** Track *our own* work (models, experiments, results, status) — separate from the external literature map ([`literature_mapping.md`](file:///d:/uni/SRP26/docs/literature_mapping.md)).
> **Last updated:** 2026-11-12 · **Ground truth:** `models/`, `results/`, `fresh_experiments/`, `paper/main.tex`.

---

## 0. At a glance

- **Research question (supervisor mandate):** *"When does frequency-domain modeling actually help vs. fail in lightweight MTS forecasting?"*
- **Proposed model:** **CycleNet-Pro** (automated FFT+ACF period discovery + 4 modular backbones: Linear / MLP / 1D-CNN / SORMixer).
- **Diagnostic:** **DSOF** (Parseval-exact subband decomposition, FVU, Log-Spectral Distance, rolling failure windows).
- **Current direction:** **DSOF-guided teacher–student distillation** (tiny hybrid time+frequency student distilled from a foundation teacher).
- **Protocol:** L=336, H∈{96,192,336,720}, RevIN, train-only scaling, 3 seeds `[2024,2025,2026]` (⚠️ currently only seed 2024 is actually run).

---

## 1. Models we've built (internal inventory)

| Model | File | What it is | Status |
|---|---|---|---|
| DLinear | `models/baselines.py` | Moving-avg trend/seasonal baseline | ✅ baseline |
| FreqEnhancedDLinear | `models/baselines.py` | DLinear + FFT spectral branch | 🔶 early, marginal |
| WaveletEnhancedDLinear | `models/baselines.py` | DLinear + DWT branch | 🔶 early, marginal |
| FourierBaseForecaster | `models/fourier_attention_net.py` | Top-K FFT harmonic extrapolation (0 params) | ✅ component |
| TwoStageFourierAttentionNet | `models/fourier_attention_net.py` | Fourier base + residual attention | ⚠️ overfits (epoch-1) |
| FourierPatchAttention | `fresh_experiments/models.py` | Fourier base + PatchTST residual | 🔶 tested |
| FourierTQNet | `fresh_experiments/models.py` | Fourier base + TQ cross-channel | 🔶 tested |
| FCMath | `models/fcmath.py` | Fourier base + foundation-model residual (+LoRA) | 🔶 tested |
| **CycleNet-Pro (Linear/MLP/CNN/SOR)** | `models/cyclenet/` | Auto period discovery + 4 backbones | ✅ **PRIMARY** |
| SORMixer / SCPM | `models/cyclenet/cyclenet_pro_SOR.py` | DCT-II FiLM spectral patch mixer | ✅ **PRIMARY** |
| Lag-Llama | `models/lag_llama_repo/` | Vendored foundation model | 📦 asset |

**Diagnostics (DSOF):** `metrics.py` (FFT + Butterworth band decomposition, LSD, coherence) · `diagnostics.py` (rolling failure windows) · `fresh_experiments/spectral_diagnostics.py` (unified evaluator).

---

## 2. Experiment tracks (chronological)

### Track 1 — "Does a frequency branch help DLinear?" (early)
- **Files:** `results/spectral_benchmark_summary.md`, `results/{dlinear,freq,wavelet}_ETTh1_*.json`
- **Setup:** ETTh1, L=336, H∈{96,192,336,720}. DLinear vs freq vs wavelet.
- **Result:** freq/wavelet give only **~1–4% MSE** gain and *worse* spectral fidelity (LSD: DLinear **10.2–11.2 dB** vs freq 11.6–13.7, wavelet 11.8–13.1).
- **Takeaway:** naive FFT/DWT augmentation of a linear model ≈ null. (Motivated the move to explicit periodicity + adaptive methods.)

### Track 2 — Two-stage Fourier hybrids (ETTh1/ETTh2)
- **Files:** `fresh_experiments/results/{summary,four_model,five_model,etth2_summary}.md`, `lightning_suite_summary.md`
- **Setup:** DLinear vs CycleNet vs FourierAttention/Patch/TQNet, ETTh1 & ETTh2, H=96.
- **Result:** CycleNet ≈ DLinear (ETTh1: 0.3664 vs 0.3679); Fourier hybrids overfit (converge epoch 1) and are *worse* than the Fourier-free baselines.
- **Takeaway:** slapping a Fourier base on attention doesn't pay; explicit recurrent cycles do as well with far fewer params.

### Track 3 — CycleNet-Pro main benchmark (the paper)
- **Files:** `results/cyclenet/cyclenet_benchmark_summary.md`, `paper/main.tex`, `results/cyclenet/cyclenet_*_H*.json`
- **Setup:** 8 datasets (ETTh1/2, ETTm1/2, Electricity, Traffic, Weather, Solar), 4 horizons, seed 2024.
- **Headline numbers (MSE, CycleNet-Linear):**
  - ETTh1: 0.3786 / 0.4064 / 0.4295 / 0.4564
  - ETTh2: 0.2945 / 0.3571 / 0.376 / 0.4231
  - ETTm2: 0.1658 / 0.2291 / 0.288 / 0.3901
  - Weather: 0.1579 / 0.2 / 0.256 / 0.3273
- **Diagnostic findings:** mid-band FVU lowest on ETTh1 (~0.67); FVU≈1.0 on Weather/ETTm2 (boundary); high-band FVU≥1.0 everywhere.

### Track 4 — Legacy cluster backbones (Sept 2026)
- **Files:** `results/legacy_cyclenet/**`, `results/cyclenet/cyclenet_benchmark_summary.md` (partial)
- **Setup:** CycleNet backbones (Linear/MLP/CNN/SOR) on Electricity, Solar, Traffic, Weather (cluster runs).
- **Result:** MLP wins Electricity (0.1627*) & Solar (−15.9% vs Linear); CNN 97.7% param reduction; SOR best Solar H=96. ⚠️ *discrepancy flagged in audit (0.1627 vs 0.1695 in table).*

### Track 5 — Foundation models + Fourier decoupling (the "inductive-bias rule")
- **Files:** `fresh_experiments/results/{expanded_foundation_suite,fourier_decoupling_foundation_benchmark,timesfm_vs_chronos,chronos_vs_fcmath_all_datasets}_H96.md`
- **Setup:** TimesFM 2.5 (200M), Chronos-Bolt (200M), IBM Granite TTM-R1 (805K), Chronos-T5 (8M), CycleNet (32K), each ± Fourier decoupling (FCMath).
- **Result (the key finding):**
  - Fourier base **helps** rollout/sequential models: Chronos-T5 1.9474→1.2292 (**−36.9%**), CycleNet 0.6476→0.5217 (**−19.4%**).
  - Fourier base **hurts** direct-patch models: TimesFM +22.4%, Chronos-Bolt +16.4%.
  - **Fails universally on Weather** (+22% to +57%) — Gibbs ringing.
- **Efficiency ranking:** TTM-R1 (805K, 0.02–0.23 s) is the efficiency winner; TimesFM is the accuracy winner; CycleNet is the spectral-fidelity winner (LSD ~9.3–10.1 dB vs 12–22 dB for foundation models).

### Track 6 — Traffic: CycleNet vs Fourier-PatchAttention
- **File:** `fresh_experiments/results/traffic_comparison_results.md`
- **Result:** CycleNet (53K params) **0.4092** vs Fourier-PatchAttention (14.7M params) 0.4143 — CycleNet wins by 1.25% with **278× fewer params** and 6.2× faster training.

### Track 7 — LoRA fine-tuning Chronos
- **File:** `fresh_experiments/results/traffic_lora_results.md`
- **Result:** FCMath zero-shot **0.6755** beats FCMath LoRA-finetuned 0.9405 — **LoRA fine-tuning hurt.** CycleNet still best (0.5263).

---

## 3. Key results at a glance (H=96)

| Model | Params | ETTh1 | Weather | Traffic | LSD (spectral) | Verdict |
|---|---|---|---|---|---|---|
| TimesFM 2.5 | 200M | **0.4601** | **0.1895** | **0.3511** | 12–18 dB | accuracy winner |
| Chronos-Bolt | 200M | 0.4687 | 0.2114 | 0.3772 | 12.7–20 dB | strong non-AR |
| IBM TTM-R1 | **805K** | 0.4945 | 0.1916 | 0.6433 | 21.8–22 dB | efficiency winner |
| Chronos-T5 | 8M | 1.6665 | 0.4627 | 1.9474 | 11–22 dB | worst (AR drift) |
| CycleNet | 32K | 0.6159 | 0.2353 | 0.6437 | **9.3–10.1 dB** | spectral-fidelity winner |

---

## 4. Integrity flags (must fix before any paper claim)

1. ⚠️ **DLinear H=720 = 0.6050** (paper's "30.1%" headline) — no source file; must cite published DLinear or produce a log.
2. ⚠️ **DLinear mid-FVU** claimed `[0.808, 0.862]`, actual `spectral_benchmark_summary.md` shows `[0.841, 0.895]` — the baseline is understated in the paper.
3. ⚠️ **Single seed** (2024) despite the 3-seed protocol `[2024,2025,2026]`.
4. ⚠️ **Electricity MLP** 0.1627 (abstract) vs 0.1695 (table).
5. ⚠️ **LSD has two implementations** (`nperseg=256` in `metrics.py` vs `64` in `spectral_diagnostics.py`) — unreliable at short horizons.

---

## 5. Current status & next steps

| Item | Status |
|---|---|
| CycleNet-Pro benchmark (8 datasets) | ✅ done (seed 2024) |
| DSOF Parseval diagnostics | ✅ implemented, used everywhere |
| Multi-seed (3 seeds) | 🚧 pending |
| Fix integrity flags (§4) | 🚧 pending |
| **DSOF-guided teacher–student distillation** | 🧭 in design — teacher = TTM-R1 (efficiency) or TimesFM (accuracy); student = tiny hybrid time+frequency; band-selective spectral loss; evaluate with DSOF. |
| Open decision | Distill **values**, **spectra**, or **both**? Is a foundation teacher's blurry spectrum worth copying at all? |

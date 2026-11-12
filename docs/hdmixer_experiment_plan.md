# HDMixer Benchmark Replication & Experimental Master Plan

> **Objective:** Execute an exhaustive, scientifically rigorous benchmark campaign mirroring the canonical AAAI-24 HDMixer evaluation matrix across 8 standard datasets and 4 prediction horizons, comparing baseline models against our frequency-aware architecture under identical protocols.

---

## 1. Benchmark Matrix: Target Leaderboard (HDMixer Protocol)

All experiments use lookback window length **$L = 336$** and evaluate prediction horizons **$H \in \{96, 192, 336, 720\}$**.

### 1.1 Complete Benchmark Tasks (32 Dataset $\times$ Horizon Configurations)

| Dataset | Variables ($C$) | Frequency | Horizon $H=96$ | Horizon $H=192$ | Horizon $H=336$ | Horizon $H=720$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | 7 | 1-Hour | Task 1 | Task 2 | Task 3 | Task 4 |
| **ETTh2** | 7 | 1-Hour | Task 5 | Task 6 | Task 7 | Task 8 |
| **ETTm1** | 7 | 15-Min | Task 9 | Task 10 | Task 11 | Task 12 |
| **ETTm2** | 7 | 15-Min | Task 13 | Task 14 | Task 15 | Task 16 |
| **Weather** | 21 | 10-Min | Task 17 | Task 18 | Task 19 | Task 20 |
| **Traffic** | 862 | 1-Hour | Task 21 | Task 22 | Task 23 | Task 24 |
| **Electricity** | 321 | 1-Hour | Task 25 | Task 26 | Task 27 | Task 28 |
| **Solar-Energy** | 137 | 10-Min | Task 29 | Task 30 | Task 31 | Task 32 |

---

## 2. Models Under Evaluation

To ensure a credible, peer-reviewable paper, four primary model classes are compared:

1. **HDMixer (Huang et al., AAAI-24):**
   - *Role:* Top SOTA reference baseline.
   - *Values:* Direct replication from published AAAI-24 paper Table 1.
2. **PatchTST (Nie et al., ICLR-23):**
   - *Role:* Standard-bearing patch Transformer baseline.
   - *Values:* Published AAAI-24 / ICLR-23 reported figures under identical $L=336$ setting.
3. **DLinear (Zeng et al., AAAI-23):**
   - *Role:* Canonical time-domain decomposition linear baseline.
   - *Execution:* Trained and evaluated directly across all 32 tasks on our standardized codebase.
4. **Our Architecture (Automated CycleNet / Dual-Domain Spectral Hybrid):**
   - *Role:* Proposed model combining time-domain decomposition and frequency/wavelet projection.
   - *Execution:* Evaluated with 3 random seeds (`2024`, `2025`, `2026`) across all 32 tasks.

---

## 3. Reference SOTA Leaderboard (Target Figures from AAAI-24 Table 1)

For experimental tracking, here are the target baseline metrics from HDMixer (AAAI-24):

| Dataset | Metric | $H=96$ (HD / Patch / DLin) | $H=192$ (HD / Patch / DLin) | $H=336$ (HD / Patch / DLin) | $H=720$ (HD / Patch / DLin) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ETTh1** | MSE | 0.373 / 0.375 / 0.375 | 0.412 / 0.414 / 0.405 | 0.392 / 0.431 / 0.439 | 0.448 / 0.449 / 0.472 |
| | MAE | 0.398 / 0.399 / 0.399 | 0.420 / 0.421 / 0.416 | 0.417 / 0.436 / 0.443 | 0.463 / 0.466 / 0.490 |
| **ETTh2** | MSE | 0.262 / 0.274 / 0.289 | 0.317 / 0.339 / 0.383 | 0.308 / 0.331 / 0.448 | 0.390 / 0.379 / 0.605 |
| | MAE | 0.329 / 0.336 / 0.353 | 0.367 / 0.379 / 0.418 | 0.367 / 0.380 / 0.465 | 0.421 / 0.422 / 0.551 |
| **ETTm1** | MSE | 0.291 / 0.290 / 0.299 | 0.332 / 0.332 / 0.335 | 0.363 / 0.366 / 0.369 | 0.424 / 0.420 / 0.425 |
| | MAE | 0.341 / 0.342 / 0.343 | 0.364 / 0.369 / 0.365 | 0.385 / 0.392 / 0.386 | 0.417 / 0.424 / 0.421 |
| **ETTm2** | MSE | 0.162 / 0.165 / 0.167 | 0.213 / 0.220 / 0.224 | 0.275 / 0.278 / 0.281 | 0.355 / 0.367 / 0.397 |
| | MAE | 0.254 / 0.255 / 0.260 | 0.289 / 0.292 / 0.303 | 0.331 / 0.329 / 0.342 | 0.380 / 0.385 / 0.421 |
| **Weather** | MSE | 0.145 / 0.152 / 0.176 | 0.194 / 0.197 / 0.220 | 0.237 / 0.249 / 0.265 | 0.317 / 0.320 / 0.323 |
| | MAE | 0.199 / 0.199 / 0.237 | 0.241 / 0.243 / 0.282 | 0.281 / 0.283 / 0.319 | 0.333 / 0.335 / 0.362 |
| **Traffic** | MSE | 0.366 / 0.367 / 0.410 | 0.385 / 0.385 / 0.423 | 0.403 / 0.398 / 0.436 | 0.441 / 0.434 / 0.466 |
| | MAE | 0.249 / 0.251 / 0.282 | 0.258 / 0.259 / 0.287 | 0.276 / 0.265 / 0.296 | 0.295 / 0.287 / 0.315 |
| **Electricity** | MSE | 0.129 / 0.130 / 0.140 | 0.145 / 0.148 / 0.153 | 0.170 / 0.167 / 0.169 | 0.193 / 0.202 / 0.203 |
| | MAE | 0.219 / 0.222 / 0.237 | 0.235 / 0.240 / 0.249 | 0.259 / 0.261 / 0.267 | 0.289 / 0.291 / 0.301 |

---

## 4. Multi-Phase Execution Schedule

To manage compute resources efficiently between local GPU (RTX 4050, 6GB VRAM) and University SLURM cluster nodes, experiments are organized into 4 logical phases:

### Phase 1: Small Benchmark Series (ETTh1, ETTh2, ETTm1, ETTm2)
- **Datasets:** 4 datasets $\times$ 4 horizons = 16 tasks.
- **Hardware:** Local GPU / Cluster.
- **Batch Size:** 32.
- **Est. Runtime per Model:** ~1.5 hours total on RTX 4050 (15–20 epochs with patience 5).
- **Goal:** Verify pipeline accuracy, calibrate learning rates, and validate Parseval decomposition on canonical benchmarks.

### Phase 2: Moderate-Scale Benchmarks (Weather, Solar)
- **Datasets:** Weather (21 channels), Solar (137 channels) $\times$ 4 horizons = 8 tasks.
- **Hardware:** Local GPU / Cluster.
- **Batch Size:** 32.
- **Est. Runtime per Model:** ~1.5 hours.
- **Goal:** Test behavior on non-stationary meteorological and solar data where diurnal patterns are heavily pronounced.

### Phase 3: High-Dimensional Industrial Benchmarks (Traffic, Electricity)
- **Datasets:** Electricity (321 channels), Traffic (862 channels) $\times$ 4 horizons = 8 tasks.
- **Hardware:** University SLURM Cluster (Partition: `STUD` / GPU node).
- **Batch Size:** 16 (to fit within GPU memory without out-of-memory errors).
- **Est. Runtime per Model:** ~4–6 hours on cluster GPU.
- **Goal:** Benchmark channel scalability and high-dimensional parameter efficiency.

### Phase 4: Frequency Diagnostics & Ablation Suite
- **Runs:**
  1. Spectral ablation: Time-only (DLinear) vs Frequency-only (rFFT) vs Wavelet-only (DWT) vs Gated Hybrid.
  2. Gating trajectory analysis: Track $\sigma(\alpha)$ convergence over training epochs across all horizons.
  3. Frequency subband decomposition: Calculate exact Parseval MSE/FVU across low, mid, and high bands.
  4. Rolling window error diagnostics: Flag anomalous prediction windows and map them to localized spectral shifts.

---

## 5. Statistical Rigor & Run Specifications

1. **Seeds:** Every task is executed across 3 seeds: `seed = 2024`, `seed = 2025`, `seed = 2026`.
2. **Result Reporting:**
   $$\text{Metric} = \text{Mean} \pm \text{Std}$$
3. **Artifact Storage:**
   - Detailed JSON per run: `results/{model}_{dataset}_H{h}_seed{seed}.json`
   - Master CSV table: `results/hdmixer_full_benchmark.csv`
   - Automated LaTeX Table generator for insertion directly into `paper/main.tex`.

---

## 6. Execution Command Templates

### Local Execution (single task):
```bash
uv run python experiment.py --dataset ETTh1.csv --model dlinear --seq_len 336 --pred_len 96 --epochs 25 --patience 5 --batch_size 32 --lr 1e-3 --seed 2024
```

### Automated Batch Runner:
```bash
uv run python run_full_benchmarks.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Solar --seeds 2024 2025 2026 --epochs 25 --patience 5
```

### SLURM Submission (Cluster):
```bash
sbatch slurm_scripts/run_electricity_720.sh
```

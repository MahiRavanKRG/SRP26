# Spectral Decomposition & Lightweight Multivariate Time Series Forecasting (SRP26)

Research codebase and companion paper investigating **"When does frequency-domain modeling actually help vs. fail in lightweight MTS forecasting?"**  
*University of Hildesheim (ISMLL) — Student Research Project (SRP26)*

---

## Core Research Focus

1. **Automated Period Discovery (CycleNet-Pro):** Parameter-efficient recurrent cycle modeling ([Lin et al., NeurIPS 2024](papers/)) extended with automated FFT + Autocorrelation (ACF) period extraction, multi-cycle vs. single-cycle switching, RevIN normalization, and channel-dependence gating (`models/cyclenet/`).
2. **Parseval-Exact Spectral Diagnostics:** Exact additive error decomposition across physical frequency subbands ($MSE_{\text{total}} = MSE_{\text{low}} + MSE_{\text{mid}} + MSE_{\text{high}}$), Fraction of Variance Unexplained ($FVU$), Log-Spectral Distance ($LSD$), and sliding-window regime-shift localization (`srp26/spectral.py`).
3. **Time-Domain vs. Spectral vs. Foundation Baselines:** Standardized comparison of CycleNet / CycleNet-Pro against DLinear (Zeng et al., AAAI 2023), PatchTST / HDMixer, TimesFM 2.5, Chronos-T5 / Bolt, TinyTimeMixer (TTM), and analytical Fourier decomposition (`FCMath`).

---

## Repository Layout

```text
├── srp26/                     # Shared core package (data splits, Fourier math, Parseval diagnostics, RevIN/CycleNet/DLinear)
├── models/                    # Research models: CycleNet / CycleNet-Pro, DLinear baselines, FCMath, Fourier-Patch/MLP
├── fresh_experiments/         # Standardized L=336 benchmark scripts & saved model checkpoints (Git LFS)
│   └── checkpoints/           # Trained PyTorch Lightning (.ckpt) and LoRA (.safetensors) checkpoints
├── bigmama_deployment/        # Portable Slurm job generator, SSH/SFTP sync & cluster runner (.env configured)
├── cluster/                   # Cluster experiment runners and cluster configuration docs
├── scripts/                   # Sweep & multi-seed orchestration scripts
├── results/                   # Empirical benchmark CSV/JSON outputs and failure diagnostics
├── paper/                     # LaTeX manuscript (main.tex, figures/, bibliography)
├── papers/                    # Reference literature PDFs (CycleNet, DLinear, PatchTST, TimesFM, Chronos, etc.)
├── docs/                      # Binding experimental protocol & standardization specification
├── octpresentation/           # Interim project presentation slides and plots
├── data_sets/                 # Dataset download helper (`download_data.py`); raw CSV/TXT datasets are gitignored
├── tests/                     # Pytest verification suite (`uv run pytest`)
├── .env.example               # Template for cluster SSH & cache configuration (copy to .env locally)
└── pyproject.toml             # uv / Python 3.13+ dependency specification
```

---

## Experimental Standardization Protocol

All benchmark scripts follow [`docs/experiment_standardization_protocol.md`](docs/experiment_standardization_protocol.md):

- **Lookback window:** $L = 336$ across all datasets.
- **Horizons:** $H \in \{96, 192, 336, 720\}$.
- **Chronological splits (no shuffling, no lookahead leakage):**
  - `ETTh1`, `ETTh2`: 12 / 4 / 4 months (`8640 / 2880 / 2880`).
  - `ETTm1`, `ETTm2`: 12 / 4 / 4 months (`34560 / 11520 / 11520`).
  - `Traffic`, `Electricity`, `Weather`, `Solar`: `70% / 10% / 20%`.
- **Prefix overlap:** Validation and test windows include the preceding $L$ steps.
- **Scaling:** `StandardTrainScaler` is fitted **strictly** on the training slice; `RevIN` handles instance-level distribution shifts.

---

## Quick Start

### 1. Clone with Git LFS (for Checkpoints)

Saved checkpoints under `fresh_experiments/checkpoints/` are stored via **Git LFS** (`*.ckpt`, `*.safetensors`):

```bash
git lfs install
git clone <repository-url>
cd SRP26
git lfs pull
```

### 2. Install Dependencies (`uv`)

Requires Python `>= 3.13` and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
```

*(Optionally configure Hugging Face cache and cluster variables cleanly via `.env`:)*

```bash
cp .env.example .env
```

### 3. Download Benchmark Datasets

Raw `.csv` / `.txt` datasets are ignored in `.gitignore` to keep the repository lightweight. Fetch the standard benchmarks locally into `data_sets/`:

```bash
uv run python data_sets/download_data.py
```

### 4. Run Verification Tests

```bash
uv run pytest -v
```

### 5. Run Benchmarks & Experiments

```bash
# Multi-horizon CycleNet-Pro evaluation
uv run python experiment_cyclenet.py --dataset ETTh1 --horizon 96

# Spectral Fourier + Foundation model suite (Chronos / TimesFM / TTM / CycleNet)
uv run python fresh_experiments/benchmark_fourier_foundation_suite.py --datasets ETTh1.csv weather.csv traffic.csv
```

### 6. Build the Paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Cluster Execution (`bigmama_deployment`)

1. Copy `.env.example` to `.env` and fill in `BIGMAMA_HOST`, `BIGMAMA_USER`, and authentication details (`.env` is gitignored).
2. Generate Slurm scripts and submit:

```bash
uv run python bigmama_deployment/generate_bigmama_slurm_jobs.py
uv run python bigmama_deployment/sync_to_bigmama.py
```

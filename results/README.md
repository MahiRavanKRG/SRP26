# SRP26 Central Experiment Registry & Results Master Index

> **Project**: Ultra-lightweight MTS Forecasting Sandbox & Spectral Decomposition (SRP26)  
> **Institution**: ISMLL, University of Hildesheim | **Supervisor**: Jung Min Choi  
> **Researcher**: Aaditya Chaturvedi  
> **Last Updated**: November 18, 2026  
> **Standard Protocol**: Lookback $L=336$, Horizons $H \in \{96, 192, 336, 720\}$, Train-only `StandardScaler`, parameter-free RevIN, 3 Random Seeds `[2024, 2025, 2026]`.

---

## 1. Master Experiment Tracks

| Track ID | Description | Datasets & Horizons | Models / Methods Evaluated | Primary Result Document | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Track 1** | **Early Spectral Baselines** | `ETTh1` ($H \in \{24, 96, 192, 336, 720\}$) | DLinear, FreqEnhancedDLinear, WaveletEnhancedDLinear | [`spectral_benchmark_summary.md`](file:///d:/uni/SRP26/results/spectral_benchmark_summary.md) | ✅ Archived |
| **Track 2** | **CycleNet-Pro Main Benchmark** | 8 Datasets, 4 Horizons | CycleNet-Pro (Linear, MLP, CNN, SORMixer) | [`cyclenet_benchmark_summary.md`](file:///d:/uni/SRP26/results/cyclenet/cyclenet_benchmark_summary.md) | ✅ Completed |
| **Track 3** | **DSOF-SD Dual-Domain Distillation** | `ETTh1`, `ETTh2` ($H=96$, 3 seeds) | Foundation Teacher (TimesFM 2.5) → Distilled Student (SOR, CNN, Linear) | [`dsof_sd_summary.md`](file:///d:/uni/SRP26/results/dsof_sd_summary.md) | ✅ Completed |
| **Track 4** | **Bigmama Cluster Foundation Suite** | `ETTh1` ($H=96$, seed 2024) | TimesFM 2.5, Chronos-Bolt, Granite-TTM, CycleNet, F-CycleNet | [`all_ETTh1_H96_seed2024.json`](file:///d:/uni/SRP26/bigmama/results/all_ETTh1_H96_seed2024.json) | ✅ Verified |
| **Track 5** | **Transform-Ablation Benchmark Suite** | 7 Datasets, 4 Horizons, 3 Seeds (**483 Runs**) | `identity`, `rfft`, `dct`, `dwt_haar`, `dwt_db4`, `stft`, `dlinear` | [`transform_ablation_summary.csv`](file:///d:/uni/SRP26/results/transform_ablation/transform_ablation_summary.csv)<br>[`transform_ablation_report.md`](file:///d:/uni/SRP26/results/transform_ablation/transform_ablation_report.md) | 🏆 **AUTHORITATIVE** |

---

## 2. Directory Structure & File Organization

```
results/
├── README.md                              <- Central Master Record (this file)
├── spectral_benchmark_summary.md          <- Track 1 early baseline summary
├── spectral_benchmark_summary.csv         <- Track 1 raw baseline table
├── cyclenet/                              <- Track 2 CycleNet-Pro benchmark files
│   └── cyclenet_benchmark_summary.md
├── dsof_sd/                               <- Track 3 DSOF dual-domain distillation suite
│   ├── dsof_sd_summary.md                 <- Track 3 consolidated report
│   └── dsof_sd_*_summary.json             <- Per-condition summary JSONs (tracked in Git)
├── transform_ablation/                    <- Track 5 Comprehensive Transform-Ablation Suite
│   ├── transform_ablation_summary.csv     <- MASTER 483-run CSV with all metrics & bands
│   ├── transform_ablation_report.md       <- Full statistical report (mean ± std, Wilcoxon)
│   ├── *.json                             <- Per-run granular metric files (local/cluster)
│   └── raw/*.npy                          <- High-resolution test window arrays (local/cluster)
└── transform_ablation_smoke/              <- Local debug & smoke test verification files
```

---

## 3. Deep Dive: Track 5 Transform-Ablation Suite (483 Runs)

The most rigorous, complete benchmark suite in the repository, evaluating the impact of mathematical signal representation before linear projection:

- **Datasets (7)**: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `weather`, `electricity`, `traffic`
- **Prediction Horizons (4)**: $H \in \{96, 192, 336, 720\}$
- **Lookback Window**: Standardized to $L=336$
- **Transforms Tested**:
  1. `identity`: Standard raw time-domain linear projection ($x \in \mathbb{R}^L \to \hat{y} \in \mathbb{R}^H$).
  2. `rfft`: Real Fast Fourier Transform (orthogonal complex frequency decomposition).
  3. `dct`: Type-II Discrete Cosine Transform (energy compaction without boundary discontinuities).
  4. `dwt_haar`: Discrete Wavelet Transform using Haar wavelet (step/piecewise localization).
  5. `dwt_db4`: Discrete Wavelet Transform using Daubechies-4 wavelet (smooth multi-scale wavelets).
  6. `stft`: Short-Time Fourier Transform (joint time-frequency spectrogram).
  7. `dlinear`: Zeng et al. moving-average decomposition baseline.
- **Seeds**: Full 3-seed protocol `[2024, 2025, 2026]`.
- **Metrics Computed per Run**:
  - Time-domain: MSE, MAE.
  - Spectral: Subband FVU (Fraction of Variance Unexplained) for Low $[0, 0.1)$, Mid $[0.1, 0.4)$, High $[0.4, 0.5]$ bands via Parseval's theorem.
  - Energy Ratio (ER) per band.
  - Log-Spectral Distance (LSD in dB).
  - Rolling Window Failure Rate (windows where error exceeds within-window $\mu + 2\sigma$).

### Master Findings Summary:
1. **Marginal Global Effect**: Transform representations provide statistically significant (Wilcoxon $p < 0.05$) but modest improvements (~1% to 2% MSE) over direct raw identity projection.
2. **Transform Specialization**:
   - `rfft` leads on short-horizon stationary thermal signals (`ETTh1`, `ETTh2` at $H=96$).
   - `dct` wins on long horizons (`ETTh1` at $H=720$) and highly non-stationary fast regimes (`ETTm2` at $H=96$).
   - `dwt_haar` wins on multi-scale periodic load signals (`electricity`).
   - `identity` remains optimal on complex cross-channel spatial flows (`traffic`).
3. **High-Band Spectral Roll-off**: Across all models, predictions severely attenuate high frequencies ($ER_{\text{high}} \ll ER_{\text{high, true}}$), acting as natural low-pass smoothing filters.

---

## 4. Git & Data Hygiene Policy

To prevent Git repository bloat while preserving 100% empirical reproducibility:
1. **Version Controlled in Git**:
   - Master summary tables: `transform_ablation_summary.csv`
   - Comprehensive markdown reports: `transform_ablation_report.md`, `dsof_sd_summary.md`, `README.md`
   - Experiment scripts: `experiment_transform_ablation.py`, `scripts/aggregate_transform_ablation.py`, `scripts/generate_diagnostic_dossier.py`
2. **Excluded from Git (via `.gitignore`)**:
   - Binary NumPy prediction arrays (`results/**/raw/*.npy`, ~14 MB, 483 files).
   - Granular per-run individual JSON dumps (`results/transform_ablation/*.json`, 483 files).
   - All data in these individual JSONs is permanently retained on disk and fully consolidated inside `transform_ablation_summary.csv`.

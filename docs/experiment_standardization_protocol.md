# Multivariate Time Series (MTS) Forecasting: Scientific Standardization Protocol & Requirements

> **Author:** Academic & Experimental Research Standards Guide  
> **Target Venues:** AAAI, ICML, NeurIPS, ICLR, KDD  
> **Reference Benchmarks:** Autoformer (NeurIPS'21), DLinear (AAAI'23), PatchTST (ICLR'23), HDMixer (AAAI'24)

---

## 1. Executive Summary & Purpose

In long-term multivariate time series forecasting (LTSF), empirical comparisons are easily compromised by subtle differences in data splitting, normalization boundaries, lookback offsets, and training stop conditions. A model claiming state-of-the-art (SOTA) performance will be immediately invalidated by reviewers if its experimental protocol deviates from the canonical literature standard.

This document establishes the **exact, reproducible scientific requirements** for all experiments in this project, aligning our evaluation protocol with the standard established by Wu et al. (Autoformer), Nie et al. (PatchTST), and Huang et al. (HDMixer).

---

## 2. Benchmark Datasets & Canonical Splitting Protocol

Leading MTS benchmarks use **fixed chronological splits** rather than random sampling or simple array slicing, preventing temporal leakage.

### 2.1 Summary of Standard Datasets

| Dataset | Time Step | Channels ($C$) | Total Length | Domain | Physical Dynamics |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **ETTh1** | 1 Hour | 7 | 17,420 | Electricity | Strong 24h & 168h periodicity, high trend shifts |
| **ETTh2** | 1 Hour | 7 | 17,420 | Electricity | Similar to ETTh1, distinct seasonal regime changes |
| **ETTm1** | 15 Min | 7 | 69,680 | Electricity | High-frequency sampling (4x ETTh1), multi-scale harmonics |
| **ETTm2** | 15 Min | 7 | 69,680 | Electricity | High-frequency sampling, distinct diurnal curves |
| **Weather** | 10 Min | 21 | 52,696 | Climatology | 21 meteorological indicators, complex inter-channel dynamics |
| **Traffic** | 1 Hour | 862 | 17,544 | Transportation | High-dimensional, stark recurring commute patterns |
| **Electricity** | 1 Hour | 321 | 26,304 | Power Grid | High-dimensional consumption curves, prominent periodicity |
| **Solar-Energy** | 10 Min | 137 | 52,560 | Renewable Energy| Zero-inflated (night cycles), heavy diurnal solar radiation |

---

### 2.2 Exact Chronological Splitting Rules

In LTSF, datasets are divided into **Train**, **Validation**, and **Test** sets. 

> [!IMPORTANT]
> **Lookback Window Overlap Requirement:**  
> When constructing test samples, the history window $[t - L : t]$ must be allowed to draw from the end of the validation set so that the **very first test prediction** begins exactly at the first test timestamp $t_\text{test\_start}$. Slicing the test array in isolation (`data[val_split:]`) incorrectly drops the lookback prefix and changes sample indices.

#### A. ETT Benchmarks (ETTh1, ETTh2, ETTm1, ETTm2)
Standardized by Informer/Autoformer as **12 months train : 4 months validation : 4 months test**:

- **Hourly Datasets (ETTh1, ETTh2):**
  - **Train:** Indices `[0 : 12 * 30 * 24]` = `[0 : 8,640]`
  - **Val:** Indices `[8,640 - L : 12 * 30 * 24 + 4 * 30 * 24]` = `[8,640 - L : 11,520]`
  - **Test:** Indices `[11,520 - L : 12 * 30 * 24 + 8 * 30 * 24]` = `[11,520 - L : 14,400]`

- **15-Minute Datasets (ETTm1, ETTm2):**
  - **Train:** Indices `[0 : 12 * 30 * 24 * 4]` = `[0 : 34,560]`
  - **Val:** Indices `[34,560 - L : 34,560 + 11,520]` = `[34,560 - L : 46,080]`
  - **Test:** Indices `[46,080 - L : 46,080 + 11,520]` = `[46,080 - L : 57,600]`

#### B. Non-ETT Benchmarks (Weather, Traffic, Electricity, Solar)
Divided into **70% Train : 10% Validation : 20% Test** by continuous timestamps:
- Let $N$ be total sequence length:
  - $N_\text{train} = \lfloor 0.7 \times N \rfloor$
  - $N_\text{val} = \lfloor 0.1 \times N \rfloor$
  - $N_\text{test} = N - N_\text{train} - N_\text{val}$
- **Train Range:** `[0 : N_train]`
- **Val Range:** `[N_train - L : N_train + N_val]`
- **Test Range:** `[N_train + N_val - L : N]`

---

## 3. Preprocessing, Scaling, & Distribution Normalization

### 3.1 Global Standardization (`StandardScaler`)
- Compute the empirical channel-wise mean $\mu_c$ and standard deviation $\sigma_c$ **strictly on the Train split only**:
  $$\mu_c = \frac{1}{N_\text{train}} \sum_{t=1}^{N_\text{train}} X_{t,c}, \quad \sigma_c = \sqrt{\frac{1}{N_\text{train}} \sum_{t=1}^{N_\text{train}} (X_{t,c} - \mu_c)^2 + \epsilon}$$
- Apply this transform to Train, Val, and Test. **Never compute scaler statistics on the full dataset.**

### 3.2 Reversible Instance Normalization (RevIN)
Most recent SOTA architectures (PatchTST, HDMixer, CycleNet, DLinear with RevIN) eliminate window-level distribution shift via RevIN (Kim et al., ICLR'22):
1. **Instance Normalization (Input):**
   $$\mu_{x} = \frac{1}{L} \sum_{t=1}^L \mathbf{x}_t, \quad \sigma_{x}^2 = \frac{1}{L} \sum_{t=1}^L (\mathbf{x}_t - \mu_x)^2 + \epsilon$$
   $$\tilde{\mathbf{x}} = \frac{\mathbf{x} - \mu_x}{\sqrt{\sigma_x^2}}$$
2. **Backbone Processing:**
   $$\tilde{\mathbf{y}} = \text{Model}(\tilde{\mathbf{x}})$$
3. **Denormalization (Output):**
   $$\hat{\mathbf{y}} = \tilde{\mathbf{y}} \odot \sqrt{\sigma_x^2} + \mu_x$$
- For models using RevIN, the model predicts the standardized residual, restoring scale and trend at inference.

---

## 4. Input Window ($L$) & Prediction Horizon ($H$) Settings

To ensure 1:1 comparability with the canonical HDMixer, PatchTST, and DLinear leaderboards:

- **Lookback Window ($L$):** Fixed at **$L = 336$** for all 8 standard datasets.  
  *(Historical note: Older papers used $L=96$; modern benchmarks since DLinear/PatchTST/HDMixer fix $L=336$ because attention and linear models reach optimal saturation at 336).*
- **Prediction Horizons ($H$):** Evaluated at 4 standard horizons:
  $$H \in \{96, 192, 336, 720\}$$
  - Short horizon: $H = 96$ (4 days hourly / 1 day 15-min)
  - Medium horizon: $H = 192$ (8 days hourly / 2 days 15-min)
  - Long horizon: $H = 336$ (14 days hourly / 3.5 days 15-min)
  - Ultra-long horizon: $H = 720$ (30 days hourly / 7.5 days 15-min)

---

## 5. Training, Optimization, & Early Stopping Protocol

In scientific reporting, under-training or differing termination criteria introduce uncontrolled variance. All experiments must adhere to the following protocol:

| Parameter | Canonical Standard | Notes |
| :--- | :--- | :--- |
| **Optimizer** | `Adam` or `AdamW` | Standard learning rate $\eta \in [10^{-4}, 10^{-3}]$, weight decay $10^{-4}$ |
| **Batch Size** | 32 (or 16 for high-channel datasets like Traffic) | Consistent across all comparative models |
| **Loss Function** | Mean Squared Error ($\ell_2$ loss) | Loss computed on normalized target values |
| **Max Epochs** | **20 to 30 epochs** | 5 epochs is insufficient for convergence |
| **Validation Frequency** | Once per epoch | Compute validation loss across the entire Val set |
| **Early Stopping** | **Patience = 3 to 5 epochs** | If validation loss fails to decrease for $P$ epochs, stop |
| **Model Checkpointing** | Save best validation checkpoint | Evaluate on Test set using the weights from the epoch with lowest Val loss |
| **Random Seeds** | 3 independent runs: **`[2024, 2025, 2026]`** | Report `Mean ± Std` for both MSE and MAE |

---

## 6. Standard Evaluation Metrics

### 6.1 Canonical Time-Domain Metrics
Evaluated over all $N_\text{test}$ prediction windows across all $C$ channels:

1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N_\text{test} \cdot H \cdot C} \sum_{i=1}^{N_\text{test}} \sum_{t=1}^H \sum_{c=1}^C (Y_{i,t,c} - \hat{Y}_{i,t,c})^2$$
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N_\text{test} \cdot H \cdot C} \sum_{i=1}^{N_\text{test}} \sum_{t=1}^H \sum_{c=1}^C |Y_{i,t,c} - \hat{Y}_{i,t,c}|$$

*Note: In the Autoformer/PatchTST/HDMixer standard, test MSE and MAE are reported on the normalized values ($z$-score standardized).*

---

### 6.2 Advanced Frequency-Domain & Diagnostic Metrics (Project Specific)

To address the supervisor's specific scientific questions regarding spectral integrity and over-smoothing, every experiment must compute:

1. **Exact Parseval Subband MSE:**
   $$\text{MSE}_\text{total} \equiv \text{MSE}_\text{low} + \text{MSE}_\text{mid} + \text{MSE}_\text{high}$$
2. **Fraction of Variance Unexplained ($FVU_\text{band}$):**
   $$FVU_\text{band} = \frac{\sum_{k \in \mathcal{B}} w_k |E(k)|^2}{\text{Var}_\text{true}(\mathcal{B})}$$
   - $FVU < 1.0$: Model outperforms a trivial mean prediction in that band.
   - $FVU \ge 1.0$: Model fails to track physical signal variance (predicting noise).
3. **Band Energy Ratio ($\mathcal{E}_\text{ratio, band}$):**
   $$\mathcal{E}_\text{ratio, band} = \frac{P_\text{pred}(\mathcal{B})}{P_\text{true}(\mathcal{B})}$$
   Quantifies spectral damping/over-smoothing ($\mathcal{E} \ll 1.0$) vs energy inflation ($\mathcal{E} > 1.0$).
4. **Log-Spectral Distance (LSD in dB):**
   $$\text{LSD} = \sqrt{\frac{1}{K} \sum_{k=0}^{K-1} \left( 10 \log_{10} P_\text{true}(k) - 10 \log_{10} P_\text{pred}(k) \right)^2}$$
5. **Sliding-Window Failure Diagnostics:**
   Track rolling MSE across windows of length $W = 12$ steps. Flag windows where local error exceeds $\mu_\text{err} + 3\sigma_\text{err}$ to pinpoint regime shift collapses.

---

## 7. Standardized Implementation Architecture

To ensure these rules are implemented without error, the project's data loading must be unified into a production-grade module:
- Slicing with historical lookback offset: `[border1 - seq_len : border2]`
- Automatic split selection based on dataset name
- Scaler fitting on Train partition only
- Seed initialization (`torch.manual_seed`, `np.random.seed`)

A verified reference implementation of this dataloader will replace the naive slicing in `dataloader.py`.

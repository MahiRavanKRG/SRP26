# TQNet — Factual Specification Report (for reimplementation audit)

**Source:** `D:\uni\SRP26\.pdf_txt\tqnet.txt` (full plain-text extraction, 1549 lines; read end-to-end).
**Paper:** "Temporal Query Network for Efficient Multivariate Time Series Forecasting", Shengsheng Lin, Haojun Chen, Haijie Wu, Chunyun Qiu, Weiwei Lin (SCUT / Pengcheng Lab). ICML 2025, PMLR 267. arXiv:2505.12917v2 [cs.LG] 11 Sep 2025. Code: https://github.com/ACAT-SCUT/TQNet
**Rule applied below:** every claim is quoted/paraphrased from the text with its section; anything the paper never states is marked **"not stated"** (do not fill in from convention).

---

## 0. Notation (as defined in the paper, §3)

> "The goal of multivariate time series forecasting (MTSF) is to predict future sequences Y_t ∈ R^{C×H} based on observed historical sequences X_t ∈ R^{C×L} at time step t, where C represents the number of variables (or channels), H is the forecasting horizon, and L is the length of the look-back window."

- `X_t ∈ R^{C×L}` = input (channels × look-back), **not** transposed anywhere: rows/channels indexing is used throughout.
- `Ȳ_t ∈ R^{C×H}` = model output.
- `W` = "prior cycle length of the dataset" (§A.1), a **hyperparameter fixed per dataset**, also the length of the learnable TQ vector.
- Notation collision to watch: `W` is used both for the cycle length **and** for the attention projection matrices (`W^Q, W^K, W^V, W^O`). The paper never gives the shapes of `W^Q/W^K/W^V/W^O`.

---

## 1. Model architecture: the exact TQ definition

### 1.1 The Temporal Query (TQ) technique (§3.2)

> "The Temporal Query (TQ) technique uses periodically shifted learnable vectors as queries in the attention mechanism to capture global inter-variable correlations (see Figure 3). Mathematically, given the periodic length W of the dataset, we initialize θ_TQ ∈ R^{C×W}, which serves as a set of learnable parameters. These parameters are initialized to zeros and dynamically updated during training to adaptively capture the underlying correlations among variables."

Extraction of the per-sample query (Eq. 1):

```
Q = θ_TQ^{t,L} ∈ R^{C×L}                                             (1)
```

> "Here, θ_TQ^{t,L} is a segment of the learnable parameters θ_TQ ∈ R^{C×W}, extracted periodically for each input sample. Specifically, for a given time step t, the starting index is computed as t mod W, and a segment of length L is cyclically selected from θ_TQ (as shown in Figure 3)."

Periodicity identity (Eq. 2):

```
θ_TQ^{t,L} = θ_TQ^{(t+i·W),L},  i ∈ N                               (2)
```

> "This implies that for samples spaced W time steps apart, the extracted TQ vectors will remain identical."
> "The periodic shifting mechanism ensures that θ_TQ^{t,L} cycles periodically over time … This mechanism also facilitates efficient parameter reuse, enabling the modeling of more robust sequence correlations by mitigating (i.e., averaging) the impact of localized noisy perturbations."

Exact index arithmetic (§A.1, Eq. 9):

```
index_TQ = (t mod W + Range(L)) mod W                               (9)
θ_TQ^{t,L} = θ_TQ[:, index_TQ]      # shape C × L
```

> "where Range(L) generates the array [0, 1, …, L−1]."

**Queries = learnable, periodically shifted, zero-initialized vectors.** They are *not* projections of the input. Keys and values *are* projections of the input.

### 1.2 TQ-MHA block (§3.2, eqs. 3–4)

> "Conventional self-attention mechanisms, such as those in iTransformer, derive the queries, keys, and values (Q, K, V) directly from the input sample sequences. … To address this, we propose a TQ-enhanced MHA mechanism, where the queries are generated from the TQ vectors to capture global correlations:"

```
Head_h = Softmax( (Q_h K_h^T) / sqrt(L) ) V_h                        (3)

where  Q_h = θ_TQ^{t,L} W_h^Q ,  K_h = X_t W_h^K ,  V_h = X_t W_h^V

MHA(Q, K, V) = Concat(head_1, …, head_H) W^O                          (4)
```

> "where W^O is the output projection matrix of MHA module. By incorporating θ_TQ^{t,L} as query vectors, the TQ-enhanced MHA mechanism fuses global priors with local input features… Specifically, the periodic design of θ_TQ^{t,L} captures stable global patterns (as shown in Figure 1(b)), while keys and values from X_t retain sample-specific information (as shown in Figure 1(c))."

Residual (from Algorithm 1, line 8): `h_attn ∈ R^{C×L} ← Multi-Head Attention(Q = θ_TQ^{t,L}, K = X_t, V = X_t) + X_t`

> "To stabilize training and enhance robustness, a residual connection is applied by adding X_t to the output of the MHA, yielding h_attn ∈ R^{C×L}." (§A.1)

Note: the residual is the **raw input X_t**, not the query and not a projection of the query.

### 1.3 MLP (temporal module) and output head (§3.2, §A.1)

> "Following the TQ-enhanced attention mechanism, a lightweight MLP is employed to model temporal dependencies within the sequence."
> "The MLP in TQNet consists of two fully connected layers interspersed with GeLU activations (Hendrycks & Gimpel, 2016), defined as:"

```
h_mlp = Linear(GeLU(Linear(h_attn)))                                  (5)
Ȳ_t   = Linear(Dropout(h_mlp))                                        (6)
```

§A.1 pipeline, verbatim:

> "In the temporal dependency modeling stage, the dimensionality of h_attn is first transformed from R^{C×L} to R^{C×d} via a linear layer, resulting in h′_attn. Then, a two-layer multi-layer perceptron (MLP) with GeLU activation is used to extract non-linear temporal dependencies. Similar to the attention mechanism, a residual connection is added to stabilize training. This process outputs h_mlp ∈ R^{C×d}."
> "In the output stage, a linear layer with Dropout maps the learned hidden representations h_mlp to the desired forecasting horizon H, resulting in the final output Ȳ_t ∈ R^{C×H}. If instance normalization was applied during the input stage, de-normalization is performed here to restore the predictions to their original scale."

So the temporal path is: `Linear` (L → d, per channel) → `Linear` (d → d, GeLU) → `Linear` (d → d) + residual `h′_attn` → `Linear` (d → H), i.e. effectively a shared per-channel MLP mapping the L time steps to H.

- Value of `d` (MLP hidden width): **not stated**.
- Whether the `Linear(L→d)` is weight-shared across channels: **not stated explicitly** (implied by the tensor shapes `R^{C×L} → R^{C×d}`; a single `Linear` applied on the last axis).

### 1.4 Tensor shape / role summary (as written in the paper)

| Tensor | Shape | Role |
|---|---|---|
| `X_t` | `C × L` | raw input window; source of K, V; also the residual |
| `θ_TQ` | `C × W` | learnable parameter, **zero-initialized**, fixed `W` per dataset |
| `θ_TQ^{t,L}` | `C × L` | cyclically extracted segment, starting at `t mod W`; source of Q |
| `Q_h = θ_TQ^{t,L} W_h^Q` | `C × ·` | per-head query (projection shape not given) |
| `K_h = X_t W_h^K`, `V_h = X_t W_h^V` | `C × ·` | per-head key/value from raw input |
| `Q_h K_h^T` | `C × C` | **inter-variable (channel) attention map** |
| `h_attn` | `C × L` | `MHA(...) + X_t` |
| `h′_attn` | `C × d` | `Linear(h_attn)` |
| `h_mlp` | `C × d` | `MLP(h′_attn) + h′_attn` |
| `Ȳ_t` | `C × H` | `Linear(Dropout(h_mlp))`, then de-normalized |

### 1.5 Algorithm 1 (verbatim, §A.1)

```
Require: Historical look-back input X_t ∈ R^{C×L}, cycle index t mod W ∈ N_0
Ensure:  Forecasting output Ȳ_t ∈ R^{C×H}
1: Initialize learnable parameters θ_TQ ∈ R^{C×W} ← 0
2: if Instance Normalization is applied then
3:     µ, σ ← Mean(X_t), STD(X_t)
4:     X_t ← (X_t − µ)/sqrt(σ² + ε)
5: end if
6: index_TQ ∈ N_0^L ← (t mod W + Range(L)) mod W
7: θ_TQ^{t,L} ∈ R^{C×L} ← θ_TQ[:, index_TQ]
8: h_attn ∈ R^{C×L} ← Multi-Head Attention(Q = θ_TQ^{t,L}, K = X_t, V = X_t) + X_t
9: h′_attn ∈ R^{C×d} ← Linear(h_attn)
10: h_mlp ∈ R^{C×d} ← Multi-Layer Perceptron(h′_attn) + h′_attn
11: Ȳ_t ∈ R^{C×H} ← Linear(Dropout(h_mlp))
12: if Instance Normalization is applied then
13:     Ȳ_t ← Ȳ_t × sqrt(σ² + ε) + µ
14: end if
```

---

## 2. Attention direction: channel (inter-variable), not temporal

**The attention is cross-channel / inter-variable (C × C), not over time steps.** Evidence:

1. `Q_h ∈ R^{C×·}`, `K_h ∈ R^{C×·}`, so `Q_h K_h^T ∈ R^{C×C}` — one row per channel (§3.2, Eq. 3, and the `W^O` concat in Eq. 4).
2. Scale factor `1/sqrt(L)`: the attention is over the `C` dimension while the per-"token" feature dimension is the look-back length `L` — the iTransformer-style inversion (tokens = variates).
3. §3.2: "The TQ technique uses periodically shifted learnable vectors as queries … to capture **global inter-variable correlations**"; "the queries are generated from the TQ vectors to capture global correlations".
4. Figure 1(b) caption: "Correlations learned by the proposed TQ technique" is a **C × C** correlation matrix on the Traffic dataset (axes labelled 0…30+ channels), compared against "Global Correlation" (a) and "Sample Correlation" (c).
5. Ablation text: removing attention means "lack the ability to directly model **inter-channel correlations**" (§4.3).
6. §A.3: `Corr(·) ∈ {A ∈ [−1,1]^{C×C}, A = A^T, diag(A) = 1}` — "denotes a normalized correlation measure between the vectors", i.e. the quantity TQ is shown to approximate is the C × C inter-variable correlation.

**How queries are constructed:** learnable parameters only — `θ_TQ ∈ R^{C×W}` initialized to zero (Algorithm 1 line 1), sliced cyclically by time index (`t mod W`), then linearly projected per head (`W_h^Q`). They are *not* derived from the input sequence.

**Only one attention layer is used.**

> "TQNet consists of only a single-layer multi-head attention (MHA) mechanism and a shallow multi-layer perceptron (MLP)." (§1)
> "TQNet … employs only a single-layer attention mechanism and a lightweight multi-layer perceptron (MLP)." (Abstract)

---

## 3. Full pipeline

Order (Figure 2, Algorithm 1):

```
X_t (C×L)
  → [optional] Instance Normalization  (per-sample mean/var removal)
  → TQ-MHA:  Q = TQ slice, K = V = X_t ; + residual X_t   → h_attn (C×L)
  → Linear (L → d)                                        → h'_attn (C×d)
  → MLP = Linear(GeLU(Linear(·))) + residual               → h_mlp (C×d)
  → Linear + Dropout (d → H)                               → Ȳ_t (C×H)
  → [if IN applied] Instance De-Normalization
```

### 3.1 Normalization

> "**Instance Normalization (IN)** Distributional shifts are common in real-world data … To address this challenge, recent research has explored various approaches, such as RevIN (Kim et al., 2021) and FAN (Ye et al., 2024). In this paper, we adopt a simple yet effective IN method that used in iTransformer and CycleNet, which involves removing the mean and variance of the input data before and after the model's operation:" (§3.2)

```
X_t = (X_t − µ) / sqrt(σ² + ε)                                        (7)
Ȳ_t = Ȳ_t × sqrt(σ² + ε) + µ                                          (8)
```

> "where µ and σ² denote the mean and variance of the input data, and ε is a small constant added for numerical stability."

- It is called IN and is cited as "the IN method used in iTransformer and CycleNet" — **this is not RevIN** (RevIN is explicitly cited only as related work). **No learnable affine parameters (γ, β) are mentioned anywhere** → affine: **not stated / apparently absent**.
- The normalization **axis is not stated**: Algorithm 1 gives `µ, σ ← Mean(X_t), STD(X_t)` with no axis specified (per-channel over L vs. scalar over the whole C×L sample is not clarified).
- Figure 2 caption: "an **optional** IN module to mitigate distributional drift"; Algorithm 1 wraps IN in `if Instance Normalization is applied`.

### 3.2 Patching / embedding

**No patching, no embedding, no tokenization is used.** Every occurrence of "patch" in the text refers to PatchTST as a baseline or to HDMixer's title in the references. The model consumes the raw `C × L` tensor directly. → Patching: **absent / not used**.

### 3.3 Series decomposition

**None.** TQNet has no moving-average / trend–seasonal decomposition, no FFT block, no frequency module. The only "periodicity" mechanism is the cyclically shifted learnable query `θ_TQ`. → Series decomposition: **not used**.

### 3.4 Forecasting head

A single linear map `Linear(Dropout(h_mlp)): R^{C×d} → R^{C×H}` (Eq. 6), applied per channel (channel-independent head over the temporal representation). Direct multi-step (whole-horizon) output; **no autoregression, no channel-independent ensembling across separate models** (a single model handles all C channels jointly through the attention).

### 3.5 Ablation variants of the architecture (§4.3)

- "When the TQ technique is removed, TQNet reduces to a standard self-attention module combined with an MLP module" (i.e. Q = K = V = X_t).
- "When the attention mechanism is removed, TQNet becomes a variant of an MLP augmented with **channel identifiers** (Shao et al., 2022), which explicitly distinguish different channels but lack the ability to directly model inter-channel correlations."
- "When both the TQ module and the attention mechanism are removed, TQNet further simplifies into a pure MLP model."
- §B.3 stacking test: three stacked TQ-MHA + MLP layers.

---

## 4. Training hyperparameters (§4.1, §A.2)

| Item | Value (verbatim) |
|---|---|
| Framework | "implemented using PyTorch (Paszke et al., 2019)" (§4.1); hardware: "a single NVIDIA GeForce RTX 4090 GPU with 24 GB memory" (§A.2) |
| Optimizer | "Adam optimizer (Kingma & Ba, 2014)" (§A.2; §4.1 says "models optimized using L2 loss") |
| Loss | "L2 loss" / "optimized with the L2 loss function" (§4.1, §A.2) |
| Learning rate | "3 × 10⁻³ for most datasets, except for smaller datasets (i.e., the ETT series), where it is reduced to 1 × 10⁻³" (§A.2) |
| LR schedule | **not stated** (no scheduler, warmup, decay, or cosine mentioned anywhere) |
| Weight decay | **not stated** |
| Epochs | "TQNet is trained for 30 epochs" (§A.2) |
| Early stopping | "early stopping based on a patience of 5 on the validation set" (§A.2) |
| Batch size | "varies based on the dataset's scale to maximize GPU utilization while avoiding out-of-memory errors. For instance, a batch size of 16 is used for the Traffic dataset, while 64 is used for the Weather dataset." (§A.2) — per-dataset values for the other 10 datasets: **not stated** |
| Attention heads | "the number of attention heads in the multi-head attention mechanism is fixed at 4" (§A.2) |
| Dropout | "the dropout rate is set to 0.5 by default. Additionally, the dropout rate in the output layer is set to 0.5 for smaller datasets (e.g., the ETT series and Weather) and 0 for larger datasets." (§A.2) |
| Seed | "all experiments are conducted with a fixed random seed of 2024" (§A.2, for the main tables) |
| Parameter init | `θ_TQ ← 0` (Algorithm 1 line 1; §3.2 "These parameters are initialized to zeros") |

---

## 5. Experimental protocol (§4.1, §A.2, Tables)

- **Look-back L:** "Following the setting of iTransformer, TQNet defaults to a look-back length of 96." Table 2/5/6 captions: "The look-back length L is uniformly fixed at 96." Exception only for the ETS baseline: "ETS requires the look-back window L to satisfy L ≥ 2W … we set L = 336 and L = 720 for ETS" (§B.4). §B.5 / Figure 8 sweep look-back from ~96 to 720 on Electricity for TQNet/iTransformer/PatchTST/DLinear (H = 96).
- **Prediction horizons H:** "{12, 24, 48, 96} for the PEMS series, and {96, 192, 336, 720} for the remaining datasets" (§4.1).
- **Splits (§A.2):** "the training-validation-test splits are consistent with prior works, such as iTransformer and TimesNet. Specifically, the data splits follow a **6:2:2 ratio for the ETT and PEMS series datasets** and a **7:1:2 ratio for the remaining datasets**." (Note: this is what the paper states; it differs from the 12/4/4-month ETT convention in some other papers.)
- **Lookback prefix overlap** for val/test slices: **not stated**.
- **Number of random seeds:** main results (Tables 2, 5, 6) are single-run at fixed seed 2024. Robustness study (Table 9, §B.6) uses seeds **2024, 2025, 2026** and learning rates **0.001, 0.002, 0.003**, reporting **Mean and Std** for each. §B.6: "we conducted multiple runs of the model under different random seeds and learning rates. The results indicate that TQNet consistently maintains low standard deviations across all settings."
- **Mean±std reporting:** only in Table 9 ("Mean represents the average value, and Std denotes the standard deviation"). Example stds: ETTh1 H=720 MSE std 0.012 (largest seed-std in the table); Electricity H=96/192/336 MSE std 0.000; Weather H=720 MSE std 0.001, MAE std 0.000; PEMS08 H=96 MSE std 0.006 (LR std 0.020); Solar H=96 MSE std 0.009 (seed) / 0.005 (LR). Main tables 2/5 report one number per cell with no ±.
- **Scaler fitting (StandardScaler on train only):** **not stated**. The only normalization described is the in-model Instance Normalization (§3.2 Eqs. 7–8). No mention of data pre-scaling, quantile/standard scaling, or where the statistics come from.
- **Model selection:** best checkpoint on the validation set via early stopping with patience 5; **which metric is monitored (train loss vs val MSE/MAE/MSE+MAE): not stated**. Whether results are reported from the best-val checkpoint or the last epoch: **not stated**.

---

## 6. Metrics, datasets, baselines

### 6.1 Metrics
- "evaluated based on both Mean Squared Error (MSE) and Mean Absolute Error (MAE)" (§4.1). No other metric (no RMSE, MAPE, sMAPE, MASE, CRPS) is reported.
- Reported per horizon and "averaged across all four prediction horizons" for the summary tables.

### 6.2 Datasets — Table 1 (§4.1), 12 datasets

| Dataset | Channels | Timesteps | Interval | HParam. W | Domain |
|---|---|---|---|---|---|
| ETTh1 | 7 | 14,400 | 1 hour | 24 | Electricity |
| ETTh2 | 7 | 14,400 | 1 hour | 24 | Electricity |
| ETTm1 | 7 | 57,600 | 15 mins | 96 | Electricity |
| ETTm2 | 7 | 57,600 | 15 mins | 96 | Electricity |
| Electricity | 321 | 26,304 | 1 hour | 168 | Electricity |
| Solar | 137 | 52,560 | 10 mins | 144 | Energy |
| Traffic | 862 | 17,544 | 1 hour | 168 | Transportation |
| Weather | 21 | 52,696 | 10 mins | 144 | Weather |
| PEMS03 | 358 | 26,208 | 5 mins | 288 | Transportation |
| PEMS04 | 307 | 16,992 | 5 mins | 288 | Transportation |
| PEMS07 | 883 | 28,224 | 5 mins | 288 | Transportation |
| PEMS08 | 170 | 17,856 | 5 mins | 288 | Transportation |

Table 1 caption: "The hyperparameter W of TQ technique is configured to align with the stable cycle length of each dataset, following the guidelines in CycleNet." §A.3: W is "determined based on domain-specific knowledge of periodic patterns and sampling intervals, as suggested by CycleNet, or alternatively through computational techniques such as the autocorrelation function (ACF)" and the values "are aligned with the cycle length settings used in CycleNet, as summarized in Table 1." (Code pointer: "acf plot.ipynb in https://github.com/ACAT-SCUT/TQNet".)

### 6.3 Baselines (§4.1) — 9 deep baselines in the main table
TimeXer (2024c), CycleNet (2024b), iTransformer (2024c), MSGNet (2024), TimesNet (2023), PatchTST (2023), Crossformer (2023), DLinear (2023), SCINet (2022a).
Additional baselines used only in the multivariate-to-univariate table (Table 6): **NSformer (Non-stationary Transformer, 2022b)** and **Autoformer (2021)**; plus the statistical **ETS (Exponential Smoothing, Hyndman & Athanasopoulos 2018)** in Table 8.
Reproduction note (Table 5 caption): "the reproduced baseline results are sourced from TimeXer, iTransformer, and CycleNet."

### 6.4 Headline results
- "TQNet consistently achieves Top 2 performance in 22 out of 24 forecasting error metrics" (§4.2); Table 2 `Count` row: TQNet 22, TimeXer 11, CycleNet 9, iTransformer 4, SCINet 2, MSGNet/TimesNet/PatchTST/Crossformer/DLinear 0.
- **TQNet per-horizon MSE/MAE (Table 5), L = 96:**

| Dataset | H=96 | H=192 | H=336 | H=720 | Avg |
|---|---|---|---|---|---|
| ETTh1 | 0.371 / 0.393 | 0.428 / 0.426 | 0.476 / 0.446 | 0.487 / 0.470 | 0.441 / 0.434 |
| ETTh2 | 0.295 / 0.343 | 0.367 / 0.393 | 0.417 / 0.427 | 0.433 / 0.446 | 0.378 / 0.402 |
| ETTm1 | 0.311 / 0.353 | 0.356 / 0.378 | 0.390 / 0.401 | 0.452 / 0.440 | 0.377 / 0.393 |
| ETTm2 | 0.173 / 0.256 | 0.238 / 0.298 | 0.301 / 0.340 | 0.397 / 0.396 | 0.277 / 0.323 |
| Electricity | 0.134 / 0.229 | 0.154 / 0.247 | 0.169 / 0.264 | 0.201 / 0.294 | 0.164 / 0.259 |
| Solar-Energy | 0.173 / 0.233 | 0.199 / 0.257 | 0.211 / 0.263 | 0.209 / 0.270 | 0.198 / 0.256 |
| Traffic | 0.413 / 0.261 | 0.432 / 0.271 | 0.450 / 0.277 | 0.486 / 0.295 | 0.445 / 0.276 |
| Weather | 0.157 / 0.200 | 0.206 / 0.245 | 0.262 / 0.287 | 0.344 / 0.342 | 0.242 / 0.269 |
| PEMS03 | 0.060 / 0.161 | 0.077 / 0.182 | 0.104 / 0.215 | 0.148 / 0.253 | 0.097 / 0.203 |
| PEMS04 | 0.067 / 0.166 | 0.077 / 0.181 | 0.097 / 0.206 | 0.123 / 0.233 | 0.091 / 0.197 |
| PEMS07 | 0.051 / 0.143 | 0.063 / 0.159 | 0.081 / 0.179 | 0.103 / 0.203 | 0.075 / 0.171 |
| PEMS08 | 0.071 / 0.170 | 0.096 / 0.196 | 0.149 / 0.244 | 0.253 / 0.309 | 0.142 / 0.229 |

- **Baseline averages (Table 2, MSE/MAE):** TimeXer 0.437/0.437 (ETTh1) … complete row-vector form: ETTh1 0.437/0.437; ETTh2 0.368/0.396; ETTm1 0.382/0.397; ETTm2 0.274/0.322; Electricity 0.171/0.270; Solar 0.237/0.302; Traffic 0.466/0.287; Weather 0.241/0.271; PEMS03 0.112/0.214; PEMS04 0.105/0.209; PEMS07 0.085/0.182; PEMS08 0.175/0.250. CycleNet 0.457/0.441, 0.388/0.409, 0.379/0.396, 0.266/0.314, 0.168/0.259, 0.210/0.261, 0.472/0.301, 0.243/0.271, 0.118/0.226, 0.119/0.232, 0.113/0.214, 0.150/0.246. iTransformer 0.454/0.448, 0.383/0.407, 0.407/0.410, 0.288/0.332, 0.178/0.270, 0.233/0.262, 0.428/0.282, 0.258/0.278, 0.113/0.222, 0.111/0.221, 0.101/0.204, 0.150/0.226. DLinear 0.456/0.452, 0.559/0.515, 0.403/0.407, 0.350/0.401, 0.212/0.300, 0.330/0.401, 0.625/0.383, 0.265/0.317, 0.278/0.375, 0.295/0.388, 0.329/0.396, 0.379/0.416.
- **Ablation of Q/K source (Table 3, avg over 4 horizons, MSE/MAE):**
  - Electricity: (Q=Raw,K=Raw) 0.175/0.267; **(Q=TQ,K=Raw) 0.164/0.259**; (Q=TQ,K=TQ) 0.179/0.269
  - Solar: 0.208/0.257; **0.198/0.256**; 0.213/0.268
  - Traffic: 0.426/0.279; **0.445/0.276**; 0.429/0.281  (note: here Q=Raw,K=Raw has lower MSE but higher MAE than the TQ setting)
  - PEMS03: 0.114/0.222; **0.097/0.203**; 0.111/0.221
  - PEMS04: 0.112/0.222; **0.091/0.197**; 0.113/0.222
  - PEMS07: 0.094/0.195; **0.075/0.171**; 0.092/0.195
  - PEMS08: 0.170/0.252; **0.142/0.229**; 0.174/0.257
- **Component ablation (Table 4 left, per horizon MSE/MAE):** Electricity Avg — Original 0.164/0.259, MHA→None 0.169/0.262, TQ→None 0.175/0.267, Pure MLP 0.190/0.276. PEMS03 Avg — 0.097/0.203, 0.101/0.206, 0.113/0.221, 0.135/0.241. PEMS04 Avg — 0.091/0.197, 0.102/0.210, 0.113/0.223, 0.151/0.258.
- **Integration study (Table 4 right, Avg):** iTransformer Electricity 0.175/0.267 → +TQ 0.163/0.260, PEMS03 0.159/0.268 → 0.104/0.202, PEMS04 0.120/0.230 → 0.091/0.195. PatchTST Electricity 0.191/0.281 → +TQ&MHA 0.171/0.270, PEMS03 0.133/0.240 → 0.106/0.211, PEMS04 0.150/0.257 → 0.096/0.205. DLinear Electricity 0.210/0.295 → +TQ&MHA 0.182/0.279, PEMS03 0.265/0.359 → 0.282/0.273, PEMS04 0.261/0.352 → 0.129/0.235.
- **W ablation (§4.3, Figure 6, Electricity, avg over H):** "TQNet achieves its best performance when W matches the weekly periodicity of the dataset (W = 168¹)" (footnote 1: "The Electricity dataset is sampled hourly, so the weekly length corresponds to 24 × 7 = 168"); "When W matches only the daily periodicity (W = 24), TQNet delivers suboptimal performance"; "when W is set to the length of two weeks (W = 2 × 168 = 336), TQNet still achieves competitive performance"; "when W does not align with any true periodic length (e.g., W = 23 or W = 167), the TQ technique introduces semantic inconsistencies, which lead to performance degradation"; "a special case arises when W = 1 … it functions as a weakened form of channel identifier". Figure 6 x-axis ticks: `W/o. TQ, 1, 23, 24, 167, 168, 336`; plotted values are graphical (text layer fragments near the curve: 0.169/0.267, 0.172, 0.167/0.261, 0.164/0.259, 0.167 — the mapping of fragments to ticks is not reliably recoverable from the text extraction). No numeric table is given for this ablation.
- **Multivariate-to-univariate (Table 6, L = 96, avg over H, MSE/MAE):** TQNet ETTm2 0.118/0.254, Electricity 0.323/0.404, Traffic 0.135/0.210 (vs TimeXer 0.120/0.258, 0.327/0.408, 0.156/0.235; iTransformer 0.127/0.267, 0.365/0.442, 0.161/0.246). Note: Electricity H=720 TQNet 0.427/0.477 loses to TimeXer 0.365/0.439.
- **Stacking (Table 7, avg over H):** Original vs 3-layer — ETTh1 0.441/0.434 vs 0.443/0.447; ETTh2 0.378/0.402 vs 0.382/0.405; ETTm1 0.377/0.393 vs 0.391/0.404; ETTm2 0.277/0.323 vs 0.280/0.324; Electricity 0.164/0.259 vs 0.167/0.262; Solar 0.198/0.256 vs 0.203/0.265; Traffic 0.445/0.276 vs 0.451/0.285; Weather 0.242/0.269 vs 0.242/0.271; PEMS03 0.097/0.203 vs 0.095/0.197; PEMS04 0.091/0.197 vs 0.084/0.189; PEMS07 0.075/0.171 vs 0.072/0.167; PEMS08 0.142/0.229 vs 0.147/0.225.
- **vs ETS (Table 8, univariate target, avg over H, MSE/MAE):** TQNet ETTh1 0.074/0.213 (ETS 0.206/0.337); ETTh2 0.178/0.337 (ETS 231.3/8.006); ETTm1 0.048/0.164 (0.169/0.268); ETTm2 0.121/0.263 (12.19/1.788); Electricity 0.299/0.388 (0.778/0.599); Solar 0.260/0.299 (18.06/1.160); Traffic 0.122/0.201 (92.69/0.490); Weather 0.002/0.029 (0.007/0.044); PEMS03 0.151/0.245 (0.618/0.357); PEMS04 0.063/0.186 (0.345/0.352); PEMS07 0.094/0.233 (0.253/0.359); PEMS08 0.178/0.309 (0.426/0.458).
- **Seed / LR study (Table 9):** columns Seed 2024 / 2025 / 2026 / Mean / Std and LR 0.001 / 0.002 / 0.003 / Mean / Std, for ETTh1, ETTh2, ETTm1, ETTm2, Electricity, PEMS03, PEMS04, PEMS07, PEMS08 (all 4 horizons each), Solar and Traffic (all 4 horizons), and Weather (all 4 horizons). Note the seed-2024 columns reproduce the main-table numbers (e.g. Electricity 96: 0.134/0.229; ETTh1 96: 0.371/0.393), which confirms the main tables are single-seed(2024) results.

---

## 7. Complexity / efficiency claims

- **Attention complexity:** "its attention mechanism has a quadratic complexity, which may raise concerns for certain real-world applications" (§4.3, Efficiency Analysis). (Quadratic in `C` — consistent with a C × C attention map.)
- **Practical overhead:** "due to the overall lightweight design of TQNet and advancements in modern GPU parallelization technologies, the model can operate with **near-linear computational overhead across varying numbers of channels**. As illustrated in Figure 7 Right, TQNet maintains minimal computational overhead even with different channel counts, closely approaching the efficiency of linear-based methods (i.e., DLinear). This indicates that in most practical scenarios (e.g., when the number of channels C < 1000), TQNet can operate with remarkably low computational costs" (§4.3).
- **Figure 7 Left** (Electricity, H = 720): compares "prediction accuracy, parameter size, and training time"; marker sizes are labelled `100k, 300k, 1M, 3M, 10M` (parameter counts), with models DLinear, PatchTST, TQNet, iTransformer, TimeXer, CycleNet. **Text conclusion:** "TQNet delivers state-of-the-art forecasting accuracy with significantly smaller parameter sizes and shorter training times compared to other models."
- **Exact parameter count, FLOPs/MACs, GPU memory per model, and wall-clock numbers: not stated** in the text (only plotted in Figure 7 without numeric labels). The only quantitative model-size statement anywhere in the text is the qualitative "significantly smaller parameter sizes"; the only numeric dataset-scale statement is "high-dimensional datasets with nearly 1,000 variables" (§6, and Traffic = 862 channels).
- Parameter count is analytically `|θ_TQ| = C × W` plus the MHA projections and the MLP/head weights (never enumerated).

---

## 8. Stated limitations (§5, verbatim bullets)

1. "Similar to CycleNet, TQNet heavily relies on the inherent periodicity of the data to determine the hyperparameter W. This dependency may limit its generalization to datasets without clear periodic patterns."
2. "When the underlying multivariate correlations in the data are weak or insignificant, enforcing strong multivariate modeling may introduce unnecessary complexity and even negatively impact performance."
3. "As shown in Figure 8 in Appendix B, the benefits of multivariate modeling diminish as the look-back window becomes sufficiently long. On the one hand, a longer look-back provides richer temporal information, which can partially substitute for multivariate cues. On the other hand, longer input windows introduce more noise, increasing the risk of overfitting in correlation modeling." … "We believe that addressing these challenges will further enhance the robustness and generalizability of TQNet, and we leave this for future exploration."

Additional acknowledged weakness (§A.3): "since TQNet considers only a single periodic length W per dataset, it may encounter challenges when dealing with multi-period scenarios (e.g., hourly, daily, weekly, or monthly patterns)"; "effectively modeling multi-periodicity remains an open research question". Suggested workarounds: pick the longest (overlapping) periodicity, or "ensemble multiple TQNet models, each configured with a different W".

---

## 9. Audit checklist — what the paper does NOT specify (do not assume these)

1. **Projection shapes:** dimensions of `W^Q, W^K, W^V, W^O`, the per-head dimension, and how `H = 4` heads reconcile with the `1/sqrt(L)` scaling in Eqs. 3 and 10. With 4 heads, standard practice would scale by `sqrt(d_head)` with `d_head = L/4`, but the paper writes `sqrt(L)` in both places. **Not stated.**
2. **`W^Q` applied to a C×L query slice:** Eq. 3 defines `Q_h = θ_TQ^{t,L} W_h^Q`, but Algorithm 1 line 8 writes the MHA call with `Q = θ_TQ^{t,L}` (no projection shown). **Inconsistent as written.**
3. **MLP hidden width `d`** — never given a value.
4. **Whether `Linear(L→d)` and the MLP are weight-shared across channels** — only implied by shapes.
5. **Attention/MLP dropout distinction:** "the dropout rate is set to 0.5 by default" — which dropout (attention weights? residual? MLP?) is not itemized.
6. **Normalization axis for IN** (per-channel-over-L vs. whole-sample scalar) and absence of affine parameters (no γ/β mentioned).
7. **Meaning of `t`:** the paper says "cycle index `t mod W`" / "for a given time step t" but never states whether `t` is the first or the last index of the look-back window, nor whether it indexes the input window or the forecast start. **Not stated.**
8. **Weight decay, LR schedule/warmup, gradient clipping, mixed precision** — not stated.
9. **Batch sizes for all datasets except Traffic (16) and Weather (64)** — not stated.
10. **StandardScaler / preprocessing pipeline, and whether scaler statistics are fit on train only** — not stated (only in-model IN is described).
11. **Lookback-prefix overlap for val/test windows** — not stated.
12. **Which metric/quantity the early-stopping patience monitors, and whether reported numbers come from the best-val epoch or the final epoch** — not stated.
13. **Number of repeats for the main tables** (single seed 2024) and whether mean±std was used anywhere except Table 9 — Table 9 only.
14. **FLOPs, memory, runtime, parameter counts** — only plotted (Figure 7), never stated numerically.
15. **Training-time epoch count interpretation:** Figure 7 axis reads "Training Time (seconds/epochs)" — the unit is ambiguous.
16. Tables 2/5 note that baseline numbers are "reproduced … sourced from TimeXer, iTransformer, and CycleNet" — the provenance for each individual baseline row is not stated.

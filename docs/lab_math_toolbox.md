# Mathematical Toolbox — Forecasting Techniques Laboratory

> **Document:** SRP26-FTM-002 · Rev A · 2026-11-16
> **Companion to:** [`docs/lab_manual_forecasting_techniques.md`](file:///d:/uni/SRP26/docs/lab_manual_forecasting_techniques.md)
> **Grounding rule:** every formula is sourced to papers in `papers/` (or the project's own `metrics.py`/`diagnostics.py`). "Used by" lists are from the collection, not the general literature.
> **Convention:** bold lowercase = vectors (x), bold uppercase = matrices (W), `·` = dot/matrix product, `⊙` = element-wise, `*` = complex conjugate, `j = √−1`.

---

## 0. The single governing equation

Every operation below exists to serve **one model** (Ch. 3 of the manual):

```
x(t) = T(t)  +  Σₖ Aₖ(t)·e^(j(ωₖt + φₖ(t)))  +  ε(t)
         trend          dominant periodic             residual
```

The math in the papers is the toolbox for **decomposing** x into those terms, **modeling** each term, and **measuring** where the model errs (Parseval = energy conservation).

---

## 1. Transforms (the basis changes)

### 1.1 Fourier family

**DFT (Discrete Fourier Transform)** — the workhorse transform. *Used by:* FEDformer, FreDF, FreEformer, FilterTS, FIRE, FBM, TIFO, RouteTS, FreDN, OLMA, FreqCycle, project DSOF.

```
X_k = Σ_{n=0}^{N−1} x_n · e^(−j·2πkn/N)          (forward)
x_n = (1/N) Σ_{k=0}^{N−1} X_k · e^(+j·2πkn/N)    (inverse)
```

**Hermitian symmetry (real input).** For real x, `X_{N−k} = X_k*`, so only `⌈(N+1)/2⌉` bins carry information — this is why FreEformer and FBM keep half the spectrum, and why TimeES cuts estimation cost via symmetry.

**Amplitude / phase decomposition.**
```
|X_k| = √(Re(X_k)² + Im(X_k)²)      magnitude (energy per frequency)
φ_k   = atan2(Im(X_k), Re(X_k))     phase (timing)
```
*Used by:* FIRE (models amplitude and phase as independent channels), APRNet (amplitude-phase reconstruction), CoST-style SSL views.

**rFFT (real FFT).** DFT restricted to the non-redundant bins of a real signal. *Used by:* project `metrics.py`/`diagnostics.py`.

**STFT (Short-Time Fourier Transform).** Windowed DFT to get *time-localized* spectra.
```
X(m, ω) = Σ_n x[n]·w[n−m]·e^(−jωn)
```
*Used by:* FIA-Net (window aggregation), FreqCycle SFPL (sliding-window segments).

**NUDFT (Non-Uniform DFT).** Fourier for *irregular* timestamps `t_n`.
```
X(f) = Σ_n x(t_n)·e^(−j2πf t_n)
```
*Used by:* TFMixer (irregular multivariate forecasting).

**DCT (Discrete Cosine Transform, DCT-II).** Real orthonormal transform that *decorrelates* autocorrelated inputs.
```
X_k = Σ_{n=0}^{N−1} x_n · cos( π k (n + ½) / N )
```
*Used by:* AutoCorr-KAN (removes autocorrelation-induced spectral bias), project SORMixer (DCT-II spectral FiLM).

**Per-frequency transfer function** — FreDF's reframing: forecasting = learn one complex matrix per frequency.
```
Ẑ^m_out = Ẑ^m_in · H^m ,   H^m ∈ C^{D×D}
ŷ        = Σ_m IFFT(Ẑ^m_out) · w_m          (learned fusion weights w_m)
```

**Fourier-mode subset selection** — FEDformer: multiply a *random* subset of M modes by a learned complex kernel.
```
Y_{m,d_o} = Σ_{d_i} Q_{m,d_i} · R_{d_i,d_o,m} ,   R ∈ C^{D×D×M}
```
(Random subsetting of size `s = O(k²/ε²)` justified by a coherence/restricted-isometry argument; deliberately includes high frequencies.)

**Cosine/sine basis expansion** — FBM reinterprets DFT as explicit basis functions:
```
x_n = Σ_k [ H_R,k · cos(2πkn/N) + H_I,k · sin(2πkn/N) ]
feature map = H_R ⊗ C + H_I ⊗ S        (time×frequency)
```
(Fixes "inconsistent starting cycles" and "inconsistent series length".)

### 1.2 Wavelet family

**DWT (Discrete Wavelet Transform), one level.** Low-pass `φ` (approximation) and high-pass `ψ` (detail):
```
cA[n] = Σ_k x[k]·φ(2n − k)      (approximation / trend)
cD[n] = Σ_k x[k]·ψ(2n − k)      (detail / fluctuation)
```
**IDWT** reconstructs `x = Σ cA·φ + Σ cD·ψ`. *Used by:* WaveForM, WPMixer, SWIFT, DPWMixer, DB2-TransF, WaveTuner, HADL, McWC.

**Haar wavelet** — the lossless, orthogonal special case (filters `[1/√2, 1/√2]`, `[1/√2, −1/√2]`):
```
cA[n] = (x[2n] + x[2n+1]) / √2
cD[n] = (x[2n] − x[2n+1]) / √2
```
Energy-preserving (`ΣcA² + ΣcD² = Σx²` → **Parseval**), no aliasing, no ringing. *Used by:* SWIFT (near-lossless downsampling), DPWMixer (lossless Haar *pyramid replaces average pooling*), WaveForM (Haar, 3 levels).

**Learnable Daubechies (DB2) filters** — DB2-TransF replaces attention with a learnable wavelet layer:
```
A[n] = Σ_k α_k ⊙ x[2n + k] ,   D[n] = Σ_k β_k ⊙ x[2n + k]
α, β learnable (initialized near canonical DB2)
```

**SWT (Stationary Wavelet Transform) / LSWT.** Same filters but *no downsampling* → shift-invariant, equal-length output at every level; LSWT makes the filters learnable. *Used by:* SimpleTM (SWT tokenization), NPMixer and HPMixer (learnable SWT).

**Wavelet packet / full binary tree** — WaveTuner decomposes both low **and** high bands at every level (2^m subbands), fixing the recursive low-frequency-only bias.

**Wavelet derivative transform (WDT)** — WaveTS applies the wavelet transform to the *derivative*:
```
operate on  Δx_t = x_t − x_{t−1}      (magnify rate-of-change / regime shifts)
```

### 1.3 Other operators

**Koopman operator** — Sonnet: a linear operator `K` advances observables of a nonlinear system; forced unitary for stability:
```
g_{t+1} = K g_t ,   K = U · diag(e^(jp_k)) · U†      (U from QR, phases p_k learnable)
```

**Evolutionary spectra (Priestley)** — TimeES: a *time-varying* spectral density `f(t, ω)` instead of a static PSD; forecasts the *evolving* spectrum, split into deterministic evolution + stochastic randomness. Complexity reduced `O(NM) → O(NK)` by Hermitian symmetry + energy sparsity.

**Graph spectral (graph Fourier)** — xCPD / SpikF-GO: use the graph Laplacian's eigenvectors as the basis:
```
L = I − D^(−1/2) A D^(−1/2) ,   L = U Λ Uᵀ
spectral filter g_θ(L) = U · g_θ(Λ) · Uᵀ
```

**Hyper-complex (quaternion-like) algebra** — FIA-Net: treat STFT windows as hyper-complex vectors and combine them with HC-MLP (≈3× fewer parameters).

---

## 2. Decomposition operations

**Moving-average trend/season split** — DLinear (baseline):
```
T = AvgPool(x)            (kernel k)
S = x − T
```
*Also:* Autoformer/FEDformer use a learned mixture-of-experts seasonal-trend decomposition (`MOEDecomp`).

**Recurrent cycle subtraction (RCF)** — CycleNet:
```
Q ∈ R^{W×D}  (learnable recurrent cycle, one per channel)
c = cyclically replicate/roll(Q, t mod W)     (length L and H)
r = x − c                                     (residual)
ŷ = Linear/MLP(r) + c_{t+1:t+H}
```

**Frequency disentanglement (anti spectral leakage)** — FreDN: a learnable per-frequency mask splits trend vs season *in the frequency domain*:
```
X̃_trend  = X̃ ⊙ σ(M)
X̃_season = X̃ ⊙ (1 − σ(M))
M init: w(k) = −log(1 + |k|)     (spectral decay of smooth trends)
ŷ = TimeMLP( IFFT(X̃_trend) )  +  IFFT( ReImBlock(X̃_season) )
```

**Amplitude / phase independent modeling** — FIRE / APRNet:
```
Â = W_amp · A + b_amp ,   φ̂ = W_φ · φ + b_φ     (two separate linear maps)
```

**Band split (low / mid / high)** — FilterTS, ReFocus, project DSOF: partition the spectrum by frequency (or by amplitude threshold `τ_i = quantile(|X_i|, α)` in FilterTS) and model each band with its own bias.

**Differencing / derivative** — stationarization:
```
Δx_t = x_t − x_{t−1}
```
*Used by:* WDAN (first-order differencing on the trend to predict normalization), WaveTS (derivative input).

---

## 3. Periodicity & autocorrelation

**ACF (Autocorrelation Function).**
```
R(τ) = Σ_t (x_t − μ)(x_{t+τ} − μ) / Σ_t (x_t − μ)²
```

**Automated period discovery** — project CycleNet-Pro: candidate periods = peaks of `|FFT(x)|`; confirm with ACF; variance-reduction metric chooses single- vs multi-cycle.

**Learnable frequency basis** — FreqLens:
```
f_i = f_min + (f_max − f_min)·σ(θ_i)         (sigmoid-mapped, f_max = 0.5 Nyquist)
ψ_i(t) = cos(2π f_i t + φ_i)                 (learnable phase φ_i)
H_i(t) = c_i · ψ̄_i(t)                        (per-frequency component; Σ_i H_i = forecast)
```

**Input autocorrelation matrix (spectral-bias theory)** — AutoCorr-KAN:
```
R = E[ x xᵀ ]         (lagged-input autocorrelation matrix)
```
The MSE-loss Hessian's conditioning is governed by R's extreme eigenvalues; strong autocorrelation ⇒ ill-conditioned R ⇒ widened low-vs-high-frequency learning-rate gap. DCT ≈ diagonalizes R for Markov processes → the fix.

---

## 4. Normalization / stationarization

**Instance normalization / RevIN.**
```
x̂ = (x − μ) / σ        (per-instance)
ŷ = σ · f(x̂) + μ       (reverse after forecasting)
```
*Used by:* FreEformer, FilterTS, WPMixer, DPWMixer, FreDN, Dualformer, and the project protocol.

**Z-score (StandardScaler).** Fit on train split only; transform all. *Used by:* project protocol (binding).

**Wavelet-disentangled adaptive normalization** — WDAN: DWT-split trend vs fluctuation, then *tailor* normalization per component (differencing for the non-stationary trend).

**Adaptive / style-based normalization (AdaIN)** — CANet: Style Blending Gate + AdaIN to avoid over-stationarization.

---

## 5. Modeling operations (per-component biases)

**Linear projection.** `Y = XW + b`. *Used by:* CycleNet, SWIFT, DLinear, RouteTS (global branch).

**Complex linear / transfer.** `Z_out = Z_in ⊙ H` (complex multiplication). *Used by:* FreDF, FEDformer, RouteTS.

**ReIm shared-parameter block (no complex matmul).**
```
Y = MLP(Re(X)) + j · MLP(Im(X))      (>50% param/compute cut)
```
*Used by:* FreDN.

**Scaled dot-product attention.**
```
Attn(Q,K,V) = softmax(QKᵀ / √d) V
```

**Differential attention** — WDformer: `softmax(Q₁K₁ᵀ/√d) − softmax(Q₂K₂ᵀ/√d)` (subtract two softmax maps to de-noise).

**Enhanced (rank-repair) attention** — FreEformer: `softmax(QKᵀ/√D) + Softplus(B)`, row-wise L1-normalized (`Softplus` keeps B entries positive → provably raises rank).

**Rotary-route attention (RoRA)** — WaveRoRA: RoPE-injected relative position + a few routing tokens aggregating K/V (linear complexity).

**Inverted/variate attention (over channels)** — FreEformer, WDformer, FAiT; FAiT adds a *high-pass complement branch* to counter self-attention's low-pass smoothing.

**Spectral-coherence attention (MVCA)** — Sonnet:
```
P_qk = Q_f ⊙ K_f*                 (cross-spectral density)
C_qk = |P_qk|² / (P_qq · P_kk + ε)   (normalized coherence)
Attn = softmax(scale(C_qk)) · V
```

**Mixture-of-Experts (MoE) routing.**
```
y = Σ_i g_i(x) · E_i(x) ,   g = softmax(TopK(router_logits))
```
*Used by:* FreqMoE, Ada-MoGE (adaptive expert count + Gaussian band-pass), WaveMoE, MoFE-Time, WaveTuner (per-band router).

**KAN (learnable activation).**
```
f(x) = Σ_q Φ_q( Σ_p φ_{p,q}(x_p) )     (activations on edges)
```
WaveTuner: per-band **Chebyshev-polynomial KAN of order `b+i`** (higher order → higher frequency). TFKAN/KFS/AR-KAN: KAN in time+frequency.

**SSM / Mamba (selective state space).**
```
h_t = Ā h_{t−1} + B̄ x_t ,   y_t = C h_t      (linear-time recurrence)
```
AdaMamba: generate input-dependent frequency bases and generalize the forgetting gate to a *time-frequency* gate.

**Diffusion (probabilistic).**
```
forward:  q(x_t | x_0) = N( √ᾱ_t x_0 , (1 − ᾱ_t) I )
reverse:  x_{t−1} = 1/√α_t ( x_t − (1−α_t)/√(1−ᾱ_t) ε_θ(x_t, t) ) + σ_t z
```
*Used by:* FALDA (Fourier-decomposes and diffuses the noise term), StaTS (learned Spectral Trajectory Scheduler + Frequency-Guided Denoiser).

**Complex Gaussian VAE (seasonal)** — DecoVAE: seasonal latent ~ complex normal capturing amplitude/phase; trend latent with a differential (Hodrick-Prescott-like) smoothness regularizer.

**Gaussian band-pass filters** — Ada-MoGE: decompose the spectrum into smooth Gaussian bands before routing to experts.

**Graph spectral routing** — xCPD: group patches into low/mid/high bands by spectral-energy response, route each to a channel-adaptive expert.

---

## 6. Loss & supervision

**Point losses.**
```
MSE     = (1/N) Σ ‖y − ŷ‖²
MAE/L1  = (1/N) Σ |y − ŷ|
SmoothL1/Huber — WPMixer, WaveTuner, FIRE (robust, sparse-data)
```

**Frequency-domain supervision** — OLMA (model-agnostic, loss-only):
```
L^(c) = α Σ_t ‖ F(ŷ_{t,:}) − F(y_{t,:}) ‖₁        (DFT over channels)
L^(t) = β Σ_i ‖ F(ŷ_{:,i}) − F(y_{:,i}) ‖₁
      + γ Σ_i ‖ W(ŷ_{:,i}) − W(y_{:,i}) ‖₁        (DFT + DWT over time)
```
Theoretic basis: a unitary transform (DFT) lowers the *marginal entropy* of correlated Gaussians → lowers the error floor. (Caveat: channel-DFT can *raise* entropy on some datasets, e.g. ECL.)

**Band-balanced gradient** — BEAT: per-band weights ∝ relative subband error (a dynamical gradient balancer).

**Probabilistic losses.** NLL (negative log-likelihood), CRPS (proper scoring rule). *Used by:* DecoVAE, TimeES, FALDA.

**Output alignment** — AliO: penalize same-timestamp disagreement across lagged inputs in time **and** frequency, quantified by TAM (Time Alignment Metric).

### 6.1 Do these losses make sense? — a verdict

Three tests for any forecasting loss: (1) **proper scoring rule?** (2) **consistent with the reported metric?** (3) **identifiable** — does it change the function class, or is it a no-op under the transform?

| Loss | Form | Verdict |
|---|---|---|
| MSE | ½‖y−ŷ‖² | ✔ Proper (Gaussian MLE), metric-consistent. Weakness: optimizes the *mean* → over-smoothing (the "frequency bias" complaint). |
| MAE / L1 | ‖y−ŷ‖₁ | ✔ Proper (Laplace MLE), robust to outliers. |
| SmoothL1 / Huber | piecewise L1/L2 | ✔ Robust compromise; sensible for sparse/noisy data. |
| NLL | −log p(y\|·) | ✔ Proper scoring rule. |
| CRPS | ∫(F(x)−1{y≤x})² dx | ✔✔ **Strictly proper** — rewards calibration + sharpness. The gold standard. |
| ELBO (VAE) | E_q[log p] − KL | ✔ Standard variational bound. |
| OLMA frequency | L1 over channel-DFT + time DFT/DWT | ⚠ Coherent, partly mis-justified (below). |
| FreDN frequency MAE | L1 over FFT coefficients | ⚠ A band-weighted L1; acts as a regularizer. |
| FIRE | weighted Huber + FFT + phase reg | ⚠ Complex; the phase regularizer is the defensible part. |
| AliO alignment | TAM + consistency | ⚠ "Nice property" loss; not accuracy-guaranteed. |
| MWMixer cross-scale | consistency across scales | ⚠ Plausible regularizer. |
| BEAT reweighting | w_b ∝ relative band error | ⚠ Gradient surgery; can destabilize. |
| FreqLens log-barrier | −Σ log(f_{i+1}−f_i) | ✔ Sensible diversity regularizer (prevents frequency collapse). |

**The orthogonality trap (the key technical point).** For an **orthonormal** transform (DFT with 1/√N, or an orthogonal DWT), Parseval gives
```
‖F(x) − F(ŷ)‖₂  =  ‖x − ŷ‖₂
```
so an **L2 spectral loss is exactly the same loss as an L2 time loss** — adding it cannot help. A frequency loss is only *mathematically distinct* if at least one holds:
1. it uses **L1** in frequency (OLMA does — this is the real lever);
2. it is **band-selective** (low/mid/high weighted differently);
3. it uses a **non-orthonormal** transform (unnormalized DFT, or a learned non-orthogonal filter bank);
4. it adds **phase regularization** (phase is nonlinear in the signal).

Most papers do not state which case they are in — so a bare "we add a frequency loss" is underspecified, and for L2 it is **vacuous**. OLMA is the exception: it uses L1 *and* does DFT across the **channel** dimension (a genuinely different axis), and it is honest that channel-DFT *raises* entropy on ECL.

**The entropy argument, honestly.** For a joint Gaussian, the *sum of marginal entropies* satisfies `Σ H(Xᵢ) ≥ H(X)` with equality iff the covariance is diagonal (Hadamard). A decorrelating unitary (PCA/KLT/DCT) therefore **does** reduce the marginal-entropy sum — that part is sound. But the leap "lower marginal entropy ⇒ lower MSE floor" is heuristic: **entropy is not a lower bound on MSE** in general (variance, not entropy, sets the L2 floor). OLMA's claim is a decorrelation heuristic dressed as a bound — defensible as motivation, not as a theorem.

**The metric–loss mismatch (the deeper critique).** The field trains with MSE/MAE and evaluates with MSE/MAE — *both* blind to band-level error. A model can lower MSE while injecting high-band noise (`FVU_high > 1`). A frequency loss "fixes the spectrum" without evidence it fixes the *metric*. The principled response is DSOF: measure per-band FVU and add a band loss **only where a band is mis-fit**.

**What a principled loss looks like (SRP26).**
```
L = L_time(y, ŷ)                                  # the reported metric: MSE or MAE
  + λ · Σ_band w_band · max(0, FVU_band − target)²   # band-selective, Parseval-consistent
probabilistic variant:  L = CRPS(y, ŷ)               # strictly proper
```
Keep the transform orthonormal and state the normalization explicitly, so time- and frequency-domain terms are comparable.

---

## 7. Diagnostic math (project DSOF — the "instrument panel")

**Parseval's theorem (the conservation law).**
```
Σ_n |x_n|² = (1/N) Σ_k |X_k|²        (energy is conserved across the transform)
⇒ E_total = E_low + E_mid + E_high     (subband error decomposes exactly)
```

**PSD (Power Spectral Density).**
```
S_k = |X_k|² / N
```

**FVU (Fraction of Variance Unexplained).**
```
FVU_band = MSE_band / Var_band          (1.0 = as bad as predicting the mean; >1.0 = injecting noise)
```

**Spectral coherence.**
```
C_xy(f) = |P_xy(f)|² / ( P_xx(f) · P_yy(f) )        ∈ [0,1]
```

**Sliding-window failure localization.** Rolling-window MSE aligned against the *local* PSD → localize exactly *when* (a regime shift, an anomaly) the model's spectral behavior broke.

*Source:* project `metrics.py`, `diagnostics.py`; see [`docs/paper_review.md`](file:///d:/uni/SRP26/docs/paper_review.md) for the band-cutoff caveat (diurnal `1/24 ≈ 0.042` falls in the **low** band, so cutoffs must be re-justified per dataset).

---

**END OF TOOLBOX — Rev A.** New math from newly-ingested papers is added here with provenance; the five-step recipe (manual Ch. 3) is the index into this file.

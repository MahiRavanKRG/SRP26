# Time-Series Data Transformation Reference (mathematical & algorithmic)

| Field | Value |
|---|---|
| **Document No.** | SRP26-FTM-002 |
| **Title** | Time-Series Data Transformation Reference — exhaustive over the paper collection + external appendix |
| **Revision** | Rev A |
| **Date** | 2026-11-16 |
| **Scope** | Every data transform applied by the ~85 collected papers, with its **mathematical form** and **algorithmic (computational) form**, plus a set of transforms **not** in the collection (appendix). |
| **Companion to** | [`docs/lab_manual_forecasting_techniques.md`](file:///d:/uni/SRP26/docs/lab_manual_forecasting_techniques.md) (Axis 1) |

---

## Part 0 — The generative axes (the exhaustive frame)

> A transform is a change of basis: `x(t) = Σₖ cₖ·φₖ(t)`. Instances are infinite; the **axes** are finite. Any transform is fully located by these five coordinates, so this frame is closed under "future papers."

| Axis | Values |
|---|---|
| **A1 Linearity** | linear (matrix/basis) · nonlinear (learned map, flow) |
| **A2 Basis** | fixed (sinusoid, cosine, wavelet, chirp, Laplacian eigenvector) · learned (learnable filters, learned frequencies) |
| **A3 Locality** | global (whole window) · windowed (fixed) · multi-scale (wavelet, log-frequency) · local (patching) |
| **A4 Adaptivity** | static · input-conditional (router, gate, stationarity weight, band selection) |
| **A5 Domain** | time · frequency · time-frequency · graph-spectral · phase-space (Koopman) · 2D/3D layout · latent |

**The physics:** A3 is the Gabor/Heisenberg trade-off — time localization and frequency localization cannot both be perfect. Fourier = (perfect freq, zero time); STFT = fixed compromise; wavelet = log-scale compromise. This one constraint explains the field's Fourier→wavelet→learned migration.

---

## Part 1 — Transform families: math + algorithm

### T1. Identity / time domain (no transform)
- **Math:** `φₖ(t) = δ(t−k)`; `x(t)` as-is. Decomposition is only *layout* (patching): `x → [x_{1:P}, x_{P+1:2P}, …]`.
- **Algorithm:** (1) take raw lookback `x ∈ R^{C×L}`; (2) optionally segment into non/overlapping patches `P`; (3) feed a backbone (linear/MLP/CNN/attention).
- **Papers:** DLinear, PatchTST, HDMixer (LEP), CycleNet (time cycle), TQNet.
- **Notes:** perfect time locality, zero frequency view. Wins on transients/regime shifts.

### T2. DFT / FFT / rFFT (global Fourier)
- **Math:** forward `X[k] = Σ_{n=0}^{N−1} x[n]·e^{−j2πkn/N}`; inverse `x[n] = (1/N)Σ_k X[k]·e^{+j2πkn/N}`. For real input, Hermitian symmetry `X[N−k] = conj(X[k])` → keep `N/2+1` bins (rFFT).
- **Algorithm:** (1) (opt.) RevIN; (2) FFT along the time axis (Cooley–Tukey, O(N log N)); (3) select modes (all / top-K by amplitude / random subset); (4) operate (complex MLP, per-frequency transfer `H`, amplitude/phase split, or reweighting); (5) zero-pad + iFFT back to time.
- **Papers:** FEDformer (random/top-K modes), FreDF (per-frequency transfer), FreEformer (real/imag tokens), FilterTS (filtering), FIRE (amp/phase), FBM (basis map), TIFO (per-frequency weights), RouteTS (amplitude routing), OLMA (DFT loss).
- **Notes:** perfect frequency, zero time; leaks on non-stationary data.

### T3. DCT (Discrete Cosine Transform, DCT-II)
- **Math:** `X[k] = Σ_{n=0}^{N−1} x[n]·cos( π/N ·(n+½)·k )`, orthonormal with a scaling on `k=0`. Approximate eigenbasis of an AR(1) covariance → **decorrelates** autocorrelated inputs.
- **Algorithm:** (1) compute DCT-II (via FFT of an even-symmetric extension, O(N log N)); (2) optionally standardize each coefficient by training-set stats; (3) feed a backbone (e.g., KAN).
- **Papers:** AutoCorr-KAN (DCT-KAN), HADL (DCT for denoising), project SORMixer (DCT-II FiLM conditioning).
- **Notes:** real-valued, phase-free; the fix for autocorrelation-induced spectral bias (F5 in the manual).

### T4. STFT (Short-Time Fourier Transform)
- **Math:** `X[m,k] = Σ_n x[n]·w[n − mR]·e^{−j2πkn/N}`, window `w`, hop `R`. Spectrogram `S[m,k] = |X[m,k]|²`.
- **Algorithm:** (1) slide window `w` (Hann) with hop `R`; (2) FFT each frame; (3) stack frames as a time-frequency matrix; (4) aggregate windows (e.g., hyper-complex/MLP).
- **Papers:** FIA-Net (STFT + hyper-complex MLP); FreqCycle SFPL (sliding-window segment FFT) is a close relative.
- **Notes:** fixed window = fixed resolution (one tile on the Gabor plane). Good for local periodic bursts; bad when scales span many octaves.

### T5. DWT (Discrete Wavelet Transform — Mallat cascade)
- **Math:** with low-pass `h`, high-pass `g` (quadrature mirror): approximation `cA_{j+1}[n] = Σ_k h[k−2n]·cA_j[k]`, detail `cD_{j+1}[n] = Σ_k g[k−2n]·cA_j[k]`, then **downsample by 2**. Haar: `h=[1/√2, 1/√2]`, `g=[1/√2, −1/√2]`.
- **Algorithm:** (1) convolve level-`j` approximation with `h` and `g`; (2) downsample by 2 → next `cA`, plus `cD`; (3) repeat `m` levels; (4) model each band; (5) **IDWT**: upsample (insert zeros), convolve with synthesis filters, sum.
- **Papers:** WaveForM (Haar L=3 + graph), WPMixer (per-band mixing), SWIFT (Haar L=1), DPWMixer (Haar pyramid), AWEMixer, WaveRoRA, WDformer, McWC, AWGformer.
- **Notes:** multi-scale; high-freq → good time, low-freq → good freq. The recursive low-pass bias (only `cA` cascaded) is the flaw WaveTuner fixes.

### T6. SWT (Stationary / undecimated wavelet — "à trous")
- **Math:** same filters `h`,`g` but **no downsampling**; instead insert `2^j−1` zeros between filter taps at level `j`. Output at every level keeps length `N` → translation-invariant.
- **Algorithm:** (1) upsample filters (insert zeros) each level; (2) convolve without decimation; (3) model each equal-length band.
- **Papers:** SimpleTM (SWT tokenization), NPMixer & HPMixer (LSWT = learnable SWT).
- **Notes:** no time-misalignment (vs DWT's critical sampling); preferred for forecasting; costs O(N·levels) instead of O(N).

### T7. CWT (Continuous Wavelet Transform)
- **Math:** `W(a,b) = (1/√a) ∫ x(t)·ψ*((t−b)/a) dt`, mother wavelet `ψ`, scale `a`, position `b`.
- **Algorithm:** (1) choose a mother (Morlet/Mexican-hat); (2) for a geometric grid of scales `a`, convolve with the scaled+flipped wavelet; (3) obtain a scale×time map.
- **Papers:** (implicit in the "fully learnable deep wavelet transform" paper; most collection papers use DWT/SWT, not CWT).
- **Notes:** continuous, highly redundant; the theoretical parent of the discrete versions.

### T8. Wavelet packet decomposition (full binary tree)
- **Math:** at **every** node apply **both** `h` and `g` (not just the approximation), producing `2^m` subbands (aa, ad, da, dd … for `m=2`).
- **Algorithm:** (1) recursively split each band with `h`/`g` + downsample; (2) get a full binary tree of subbands; (3) weight/select/model each subband.
- **Papers:** WaveTuner (WPT + adaptive router + per-band KAN).
- **Notes:** fixes the DWT low-frequency bias (T5) — high-frequency detail bands are kept, not discarded.

### T9. Learnable wavelet filter bank
- **Math:** make `h`,`g` (or the DB2 coefficients `α_k`,`β_k`) trainable tensors: `A[n] = Σ_k α_k⊙x[2n+k]`, `D[n] = Σ_k β_k⊙x[2n+k]`, initialized near a canonical wavelet and optimized by backprop.
- **Algorithm:** (1) init filters near Haar/DB2; (2) run the cascade/windowed transform; (3) let gradients update the filters; (4) stack with a backbone.
- **Papers:** DB2-TransF (learnable Daubechies replaces attention), NPMixer/HPMixer (LSWT), "fully learnable deep wavelet transform".
- **Notes:** A2=learned basis — the current frontier; adapts the basis to the data instead of fixing it.

### T10. Wavelet derivative transform (WDT)
- **Math:** apply the wavelet transform to the **first derivative** of the series, `W{Δx}` where `Δx[t]=x[t]−x[t−1]`, magnifying rate-of-change and abrupt regime shifts.
- **Algorithm:** (1) compute `Δx`; (2) DWT-decompose `Δx` into multi-scale coefficients; (3) model/refine coefficients; (4) inverse WDT reconstruct.
- **Papers:** WaveTS / Multi-Order Wavelet Derivative Transform.
- **Notes:** targets change-points; complements raw-signal wavelets (which are change-point insensitive).

### T11. Koopman operator (phase-space / spectral dynamics)
- **Math:** find linear `K` s.t. observables evolve `g(x_{t+1}) = K·g(x_t)`; make `K` unitary `K = U diag(e^{ip}) U†` (QR + eigen-phases `p`), then iterate stably.
- **Algorithm:** (1) map input to observables `g`; (2) learn/QR-decompose a matrix to a unitary; (3) encode frequencies as eigen-phases `diag(e^{ip})`; (4) evolve in the eigenspace (often in the frequency domain); (5) map back.
- **Papers:** Sonnet (Koopman + spectral-coherence attention).
- **Notes:** linearizes nonlinear dynamics; the "operator" view of a transform (A5=phase-space).

### T12. Graph Fourier transform (GFT) — spectral *across variables*
- **Math:** graph Laplacian `L = D − A`; eigen-decomposition `L = U Λ Uᵀ`; graph spectrum `x̂ = Uᵀ x`. "Frequency" = Laplacian eigenvalue (smoothness), **across variables/channels**, not time.
- **Algorithm:** (1) build `A` (correlation / learned); (2) form `L`, eigendecompose; (3) project channel features onto `U`; (4) band-split by eigenvalue (low/mid/high smoothness); (5) route/filter per band.
- **Papers:** xCPD (graph-spectral channel-patch routing), SpikF-GO (spiking Fourier graph), WaveForM/MSGNet (learned graph, though these are GNN rather than GFT per se).
- **Notes:** A5=graph-spectral; orthogonal to time-domain transforms — answers "which channels mix, and how smoothly".

### T13. NUDFT (Non-Uniform Discrete Fourier Transform)
- **Math:** `X(f) = Σ_n x(t_n)·e^{−j2πf·t_n}` with irregular timestamps `t_n` (no equal spacing assumption).
- **Algorithm:** (1) take `(t_n, x_n)` pairs; (2) evaluate the sum at a chosen frequency grid `f` (matrix-vector product); (3) model in the frequency domain; (4) inverse-NUDFT for seasonal extrapolation.
- **Papers:** TFMixer (NUDFT for irregular MTS).
- **Notes:** the fix for F7 (irregular sampling) — the one collection transform that drops the equidistant assumption.

### T14. Evolutionary spectra (time-varying spectrum)
- **Math:** Priestley's model `x(t) = ∫ A_t(ω)·e^{jωt} dZ(ω)` — a spectrum `A_t(ω)` that itself evolves in time; decompose into *spectral evolution* (deterministic) + *spectral randomness* (stochastic).
- **Algorithm:** (1) estimate a parameterized time-varying spectrum; (2) separate deterministic evolution from aleatoric randomness; (3) exploit Hermitian symmetry + energy sparsity to cut complexity `O(NM)→O(NK)`; (4) forecast the evolving spectrum with a (linear) backbone.
- **Papers:** TimeES.
- **Notes:** the principled answer to non-stationarity for *probabilistic* forecasting — a spectrum, not a single FFT.

### T15. Period reshape (1D → 2D / 3D layout)
- **Math:** reshape lookback `x[1:L]` into `P×F` where `P` = dominant period, `F = L/P` (stack periods as rows); or a 3D bucket `(variable-group × phase-aligned time × offset)`.
- **Algorithm:** (1) find dominant period `P` (FFT peak / ACF); (2) reshape 1D→2D (or 3D); (3) process with 2D conv / vision backbone; (4) flatten back.
- **Papers:** Times2D (period reshape + derivative heatmaps), MSGNet (FFT period → 2D graph), TriTS (period-aware reshape → Visual Mamba), PHAT (3D periodic bucket).
- **Notes:** converts "temporal pattern" into "spatial/texture pattern"; A5=2D/3D layout.

### T16. Derivative / differencing (time-domain)
- **Math:** `Δx[t] = x[t] − x[t−1]`; `Δ²x[t] = Δx[t] − Δx[t−1]`.
- **Algorithm:** (1) compute first/second difference; (2) map to heatmaps or feed as a feature; (3) forecast.
- **Papers:** Times2D (FSDH = 1st/2nd derivative heatmaps), WaveTS (wavelet-of-derivative), DTAF (frequency differencing, spectral analog).
- **Notes:** exposes slope/curvature = turning points; complementary to amplitude modeling.

### T17. Normalization (distribution transform)
- **Math (RevIN):** `x̂ = (x − μ)/σ` (per instance, then affine); forecast; **reverse** `x = x̂·σ + μ`. (WDAN splits DWT trend/fluctuation and normalizes each differently.)
- **Algorithm:** (1) compute per-instance mean/std; (2) normalize; (3) model; (4) de-normalize. WDAN: (1) DWT split; (2) per-band normalization; (3) model; (4) invert.
- **Papers:** RevIN (⚠ referenced), WDAN, CANet, DTAF.
- **Notes:** not a "basis" but a distribution transform; removes non-stationary shift before any spectral transform.

### T18. Tokenization / quantization (discrete representation)
- **Math:** map real values to a discrete vocab `x → z ∈ {1..V}` (binning, digit strings, or **wavelet coefficients** thresholded+quantized).
- **Algorithm:** (1) scale; (2) decompose (wavelet) or bin; (3) threshold + quantize to tokens; (4) feed a token model (LLM/TSFM); (5) decode.
- **Papers:** WaveToken (wavelet tokenizer), TempoWave (multi-wavelet digit embeddings).
- **Notes:** the bridge to foundation models; A5=latent/discrete.

### T19. Hyper-complex (quaternion / Clifford) representation
- **Math:** represent windows as hyper-complex numbers `q = a + bi + cj + dk`; combine via the Hamilton product (or Clifford geometric product) so multiple STFT windows aggregate with shared algebra.
- **Algorithm:** (1) STFT → window set; (2) pack as quaternions; (3) hyper-complex MLP (Hamilton-product layers); (4) forecast.
- **Papers:** FIA-Net (hyper-complex MLP), SimpleTM (geometric product attention).
- **Notes:** cuts parameters ~3× vs naive window aggregation; A1=nonlinear-ish (multiplicative algebra).

### T20. Learnable frequency bases (FreqLens-style)
- **Math:** parameterize frequencies `f_i = f_min + (f_max−f_min)·σ(θ_i)`; basis `ψ_i(t) = cos(2π f_i t + φ_i)` with learnable phase; project `c_i = ⟨h, ψ̄_i⟩`; reconstruct `Σ_i c_i·ψ̄_i`.
- **Algorithm:** (1) init N frequencies via sigmoid; (2) build cosine bases; (3) project hidden features; (4) sparse top-K select; (5) per-frequency MLP → attribution; (6) sum + residual path.
- **Papers:** FreqLens (also the "learnable frequencies" spirit of FreDN/WaveTuner).
- **Notes:** the endpoint of A2=learned: even the *frequencies* are parameters; enables axiomatic (Shapley) attribution.

---

## Appendix — External transforms (known, but NOT in this collection)

> For completeness: these are standard transforms in signal processing / time series that **no paper in the collection uses**. They are listed so the reference is closed against the wider literature, not just this subfield.

### X1. Hilbert transform (analytic signal / instantaneous phase)
- **Math:** analytic signal `x_a(t) = x(t) + j·H{x}(t)`, where `H` is the Hilbert transform `H{x}(t) = (1/π)∫ x(τ)/(t−τ) dτ`; in frequency domain `H(ω) = −j·sign(ω)`. Instantaneous amplitude `|x_a|`, instantaneous phase `arg(x_a)`, instantaneous frequency `d(arg x_a)/dt`.
- **Algorithm:** (1) FFT; (2) zero negative frequencies, double positive (zero DC/Nyquist); (3) iFFT → analytic signal; (4) read amplitude/phase.
- **Why absent:** the collection decomposes *periodicity*, not *instantaneous* phase. Relevant for cycle-amplitude/phase tracking (e.g., amplitude-modulated cycles).

### X2. EMD / VMD / SSA (adaptive, data-driven decomposition — no fixed basis)
- **EMD:** repeatedly sift: subtract the mean of the upper/lower envelope from the signal → intrinsic mode functions (IMFs).
- **VMD:** variational: minimize the sum of mode bandwidths subject to exact reconstruction (`min Σ ‖∂t[(δ(t)+j/πt)∗u_k]e^{−jω_k t}‖²`).
- **SSA:** (1) embed into a trajectory (Hankel) matrix; (2) SVD; (3) group eigenvectors; (4) diagonal-average to reconstruct components.
- **Why absent:** the collection favors *fixed/learnable basis* methods (Fourier/wavelet); these classic nonlinear decompositions are the "statistical" alternative.

### X3. S-transform (STFT with frequency-dependent window)
- **Math:** `S(τ,f) = ∫ x(t)·(|f|/√(2π))·e^{−(τ−t)²f²/2}·e^{−j2πft} dt` — Gaussian window whose width shrinks as frequency rises.
- **Algorithm:** (1) for each frequency, scale the Gaussian window accordingly; (2) windowed FFT; (3) time-frequency plane with adaptive resolution.
- **Why absent:** wavelet packets give similar adaptive resolution and are more popular here.

### X4. Fractional Fourier transform (FrFT) — rotation in the time-frequency plane
- **Math:** `F^α` is a rotation by angle `α = aπ/2`; `F^0`=identity, `F^1`=Fourier. Kernel involves a chirp: `K_α(t,u) ∝ e^{j(t²cotα − 2tu·cscα + u²cotα)/2}`.
- **Algorithm:** (1) choose order `a`; (2) compute via chirp-multiply → FFT → chirp-multiply; (3) the result interpolates between time and frequency.
- **Why absent:** the collection treats time/frequency as a *split*, not a *continuum* — FrFT is the continuous bridge.

### X5. Chirplet transform (basis of chirps)
- **Math:** 4-parameter atom `g(t) = e^{−α t²}·e^{j(β t² + ω t + φ)}` (Gaussian envelope + linear/quadratic frequency modulation).
- **Algorithm:** match/superpose chirplet atoms to represent frequency-sweeping components.
- **Why absent:** collection signals are assumed (quasi-)stationary-periodic; chirps model *drifting* frequency (e.g., Doppler), which only TimeES/Sonnet's quadratic FM (wavelet atoms) touch.

### X6. PCA / SVD / whitening (learned orthogonal linear bases)
- **Math:** `X = U S Vᵀ`; project onto top-`k` columns of `V`; whiten `X̂ = X·V·S⁻¹`.
- **Algorithm:** (1) SVD/eigendecompose; (2) keep top components; (3) project/whiten.
- **Why absent:** the collection's "learned basis" papers (DB2-TransF, FreqLens) learn *filter* bases, not data-covariance bases; PCA/whitening is the statistical counterpart.

### X7. Normalizing flows / invertible neural maps (learned nonlinear transforms)
- **Math:** invertible `z = f(x)`, `x = f⁻¹(z)`; `log p(x) = log p(z) + log|det(∂f/∂x)|`.
- **Algorithm:** (1) stack invertible layers (coupling/affine); (2) maximize likelihood; (3) sample by inverting.
- **Why absent:** the collection's learned transforms stay *linear-in-basis* (wavelet filters); full invertible maps are the generative extreme.

### Summary of the closed set
The collection's transforms = {T1…T20}; the wider known set adds {X1…X7}. A future paper's transform is either a member of these, or a new point on the Part 0 axes (which is how it should be recorded — by coordinates, not by a new name).

---

## Part 2 — Paper-by-paper transform index (exhaustive over the collection)

> Every paper's transform, mapped to its family `T#` (Part 1) and its five axis coordinates. Legend: **B**asis = fixed/learned/adaptive · **Loc**ality = global/windowed/multi-scale/local · **Lin** = linear/nonlinear · **Dom**ain = time/frequency/time-frequency/graph-spectral/phase-space/2D/latent. ⚠ = baseline/referenced, PDF not in `papers/`.

### 2.1 Lane A — Fourier / spectral

| Paper (stamp) | Transform (→ T#) | B | Loc | Lin | Dom |
|---|---|---|---|---|---|
| FEDformer (2201) | DFT + random/top-k mode subset + complex kernel (T2) | fixed | global | linear | frequency |
| FreDF (2407) | DFT + per-frequency complex transfer + fusion (T2) | fixed | global | linear | frequency |
| FreEformer (2501) | DFT + real/imag variate tokens (T2) | fixed | global | linear | frequency |
| FilterTS (2505) | DFT + static/dynamic cross-variable filtering (T2) | adaptive | global | linear | frequency |
| FIRE (2510) | DFT + independent amplitude/phase channels (T2) | fixed | global | linear | frequency |
| TIFO (2602) | DFT + per-frequency stationarity weights (T2) | adaptive | global | linear | frequency |
| RouteTS (2608) | DFT + amplitude routing (top-k→freq, residual→time) (T2) | adaptive | global | linear | frequency |
| FBM (2507) | DFT → cosine/sine basis map (time×freq) (T2) | fixed | global | linear | time-frequency |
| OLMA (2505) | DFT/DWT **loss** over channel+time (T2) | fixed | global | linear | frequency |
| Dualformer (2601) | DFT + layer-wise frequency-band allocation (T2) | adaptive | global | linear | frequency |
| TimeES (2609) | Evolutionary spectra (time-varying) (T14) | learned | global | linear | frequency |
| FreqLens (2602) | Learnable frequency bases + attribution (T20) | learned | global | linear | frequency |
| Fre-CW (2508) | DFT loss (adversarial attack) (T2) | fixed | global | linear | frequency |
| FIA-Net (2502) | STFT + hyper-complex window aggregation (T4+T19) | fixed | windowed | linear | time-frequency |
| APRNet (2508) | FFT (temporal+channel) + amp/phase + KAN (T2) | learned | global | nonlinear | frequency |
| NFT (2404) | 2-D DFT (vars×time) + TCN-learned coeffs (T2) | learned | global | nonlinear | frequency |
| ReFocus (2502) | FFT + mid-frequency conv emphasis (T2) | adaptive | global | nonlinear | frequency |
| Times2D (2504) | FFT-period 1D→2D reshape + derivative heatmaps (T15+T16) | adaptive | multi-scale | linear | 2D |
| AEFIN (2505) | DFT + top-K + IDFT → stable/unstable split (T2) | adaptive | global | linear | frequency |
| DDTime (2511) | DFT loss (dataset distillation) (T2) | fixed | global | linear | frequency |
| SpecReTF (2606) | STFT frames + amplitude-PDF + JSD + phase (T4) | fixed | windowed | linear | time-frequency |
| AliO (2610) | FFT alignment loss (phase+amplitude) (T2) | fixed | global | linear | frequency |
| CoSPOT (2609) | DFT + DWT → spectral prompts (T2+T5) | learned | multi-scale | linear | time-frequency |
| FreSia (2610) | FFT top-K + learnable frequency pool (T2) | learned | global | nonlinear | frequency |
| FM-LLM (2608) | Fourier-Analysis-Network (cos/sin) expansion + DFT loss (T2) | learned | windowed | nonlinear | frequency |
| SpecTF (2602) | rFFT + complex cross-attention fusion (T2) | learned | global | nonlinear | frequency |

### 2.2 Lane B — Wavelet / multi-resolution

| Paper (stamp) | Transform (→ T#) | B | Loc | Lin | Dom |
|---|---|---|---|---|---|
| WaveForM (AAAI23) | Haar DWT L3 + learned graph per band (T5) | fixed | multi-scale | linear | time-frequency |
| WPMixer (2412) | multi-level DWT + per-band mixer (T5) | fixed | multi-scale | linear | time-frequency |
| SWIFT (2501) | Haar DWT L1 (lossless downsampling) (T5) | fixed | multi-scale | linear | time-frequency |
| DPWMixer (2512) | Haar pyramid (lossless, replaces pooling) (T5) | fixed | multi-scale | linear | time-frequency |
| DB2-TransF (2512) | learnable Daubechies bank (replaces attention) (T9) | learned | multi-scale | linear | time-frequency |
| WaveTuner (2511) | wavelet packet + adaptive router + per-band KAN (T8) | learned | multi-scale | nonlinear | time-frequency |
| FreDN (2511) | FFT + learnable trend/season disentangler (T2) | learned | global | linear | frequency |
| Sonnet (2505) | Koopman + learnable wavelet atoms + coherence (T11) | learned | multi-scale | nonlinear | phase-space |
| AWEMixer (2511) | DWT db4 L3 + RFFT frequency router (T5+T2) | fixed | multi-scale | linear | time-frequency |
| WaveMoE (2604) | DWT bior2.2 L2 + MoE (T5) | fixed | multi-scale | linear | time-frequency |
| NPMixer (2605) | SWT (fixed dilated filters¹) (T6) | fixed¹ | multi-scale | linear | time-frequency |
| MWMixer (2609) | MODWT (undecimated) + inverse (T6) | fixed | multi-scale | linear | time-frequency |
| WaveRoRA (2410) | DWT Sym3 L4 + route attention (T5) | fixed | multi-scale | linear | time-frequency |
| Wave-MaskMix (2408) | DWT + coefficient mask/mix augmentation (T5) | fixed | multi-scale | linear | time-frequency |
| WDformer (2509) | DWT L-level + differential attention (T5) | fixed | multi-scale | linear | time-frequency |
| WaveTS / WDT (2505) | wavelet-of-derivative, multi-order (T10) | fixed | multi-scale | linear | time-frequency |
| Wavelet-MoE (WaveTS-B/M) (2508) | learnable orthogonal DWT + MoE (T9) | learned | multi-scale | linear | time-frequency |
| HADL (2502) | Haar DWT (drop detail) + DCT-II (T5+T3) | fixed | multi-scale→global | linear | frequency |
| TempoWave (2606) | multi-wavelet digit embedding (T5) | fixed | multi-scale | linear | latent |
| WaveToken (2412) | DWT + bin quantization → tokens (T5+T18) | fixed | multi-scale | linear+nonlin | time-frequency |
| AWGformer (2601) | learnable wavelet + adaptive level (T9) | adaptive | multi-scale | linear | time-frequency |
| SimpleTM (2505) | learnable SWT + geometric-product attention (T6+T19) | learned | multi-scale | linear | time-frequency |
| WDAN (2506) | DWT + per-band adaptive normalization (T5+T17) | fixed | multi-scale | linear | time-frequency |
| Fully-learnable DWT / DeSpaWN (2022) | learnable FDWT cascade + hard-threshold (T9) | learned | multi-scale | linear+nonlin | time-frequency |

> ¹ NPMixer's abstract claims "learnable" SWT; the extracted main text shows fixed dilated filters. Recorded as fixed; flag for verification.

### 2.3 Lane C — Periodicity / cycles

| Paper (stamp) | Transform (→ T#) | B | Loc | Lin | Dom |
|---|---|---|---|---|---|
| CycleNet (2409) | learnable recurrent-cycle subtraction (time; T1) | learned | local | linear | time |
| FreqCycle (2603) | cycle + segmented STFT-like frequency (T5/T4) | learned | multi-scale | linear | time-frequency |
| TQNet (2505) | periodically-shifted query (time; T1) | learned | local | linear | time |
| LightGTS (2506) | FFT cycle discovery + 1D→P×N period patching (T15+T17) | adaptive | global | linear | time |
| HPMixer (2602) | learnable cycle + LSWT residual + hierarchical patch (T6) | learned | multi-scale | linear | time-frequency |
| PENGUIN (2508) | RevIN + periodic-nested relative attention bias (T1) | fixed | windowed | linear | time |
| MSGNet (2404) | FFT scale discovery + 3D reshape + Mixhop graph conv (T15) | adaptive | multi-scale | linear | 2D |
| PHAT (2602) | 3D periodic-bucket reshape + pos/neg attention (T15) | fixed | multi-scale | linear | 2D/3D |
| MLF (2505) | multi-period inputs + redundancy filter + adaptive patch (T1) | learned | local | linear | time |
| DTAF (2511) | MoE non-stationary filter + per-patch rFFT + freq differencing (T16+T2) | adaptive | windowed | nonlinear | time-frequency |
| CANet (2504) | RevIN + style-blend AdaIN + Fourier threshold (T17+T2) | learned | multi-scale | nonlinear | time |
| TimeEmb (2510) | rFFT − static bank + learnable complex filter (T2) | learned | global | linear | frequency |
| BEAT (2501) | DWT subbands + gradient balancer (T5, training-side) | fixed | multi-scale | linear | time-frequency |

### 2.4 Lane D — Next-gen backbones / foundation

| Paper (stamp) | Transform (→ T#) | B | Loc | Lin | Dom |
|---|---|---|---|---|---|
| AutoCorr-KAN (2604) | DCT-II decorrelation (T3) | fixed | global | linear | frequency |
| MoFE-Time (2507) | learned harmonic basis synthesis (no FFT) (T20) | learned | global | nonlinear | time-frequency |
| TFKAN (2506) | rFFT + complex KAN on Re/Im + IFFT (T2) | fixed | global | linear | time-frequency |
| KFS (2508) | FFT + top-K energy selection + irFFT (T2) | adaptive | global | linear | frequency |
| AR-KAN (2509) | AR filter bank (Yule-Walker on ACF; time) (T1) | learned | local | linear | time |
| AdaMamba (2604) | adaptive trig modulation (cos/sin, input-dep ω) (T20) | adaptive | global | nonlinear | time-frequency |
| FRWKV+ (2605) | rFFT → real/imag RWKV → irFFT (T2) | fixed | global | linear | frequency |
| FreqMoE (2501) | FFT → learnable band masks → MoE gating → IFFT (T2) | learned | global | nonlinear | frequency |
| Ada-MoGE (2512) | FFT + Gaussian band-pass (DoG) + adaptive experts (T2) | adaptive | global | nonlinear | frequency |
| FALDA (2505) | DFT top/bottom-k → 3-way split + diffusion (T2) | fixed | global | linear | frequency |
| StaTS (2603) | rFFT + multi-band complex reweight + diffusion schedule (T2) | adaptive | global | nonlinear | frequency |
| DecoVAE (2608) | MA trend + complex FFT seasonal VAE (T2) | fixed | global | linear+nonlin | frequency |
| SpikF-GO (2606) | spiking FFT over hypervariate graph (T12) | learned | global | nonlinear | graph-spectral |
| xCPD (2603) | graph Laplacian eigen → graph Fourier → band routing (T12) | learned | global | nonlinear | graph-spectral |
| TriTS (2604) | EMA + DWT + ACF 1D→2D + Mamba (T5+T15) | fixed | multi-scale | linear | time-frequency+2D |
| FAiT (2606) | inverted-attn low/high-pass + DFT modulation (T2) | adaptive | global | nonlinear | frequency |
| McWC (2606) | multi-cycle + wavelet + FFT loss (T5+T2) | learned | multi-scale | nonlinear | time-frequency |
| TFMixer (2602) | learnable NUDFT (irregular) + inverse (T13) | learned | global | nonlinear | time-frequency |

### 2.5 Lane E — Non-stationarity / baselines

| Paper (stamp) | Transform (→ T#) | B | Loc | Lin | Dom |
|---|---|---|---|---|---|
| DLinear ⚠ (2205) | moving-average trend/seasonal (time; T1) | fixed | local | linear | time |
| PatchTST ⚠ (2211) | patching + CI (time; T1) | fixed | local | linear | time |
| RevIN ⚠ (2109) | instance normalization + reverse (T17) | fixed | local | linear | time |
| HDMixer (2402) | length-extendable patching (time; T1) | fixed | local | linear | time |
| APRNet (2508) | (already in 2.1) | | | | |

---

### 2.6 Reverse index — transform family → papers

| Family | Papers using it |
|---|---|
| **T1 time/identity** | DLinear, PatchTST, HDMixer, CycleNet, TQNet, PENGUIN, AR-KAN, MLF |
| **T2 FFT/DFT/rFFT** | FEDformer, FreDF, FreEformer, FilterTS, FIRE, TIFO, RouteTS, FBM, OLMA, Dualformer, Fre-CW, APRNet, NFT, ReFocus, AEFIN, DDTime, AliO, FreSia, FM-LLM, SpecTF, FreDN, AWEMixer, TFKAN, KFS, FRWKV+, FreqMoE, Ada-MoGE, FALDA, StaTS, DecoVAE, FAiT, McWC, CANet, TimeEmb |
| **T3 DCT** | AutoCorr-KAN, HADL |
| **T4 STFT** | FIA-Net, SpecReTF, FreqCycle(SFPL) |
| **T5 DWT** | WaveForM, WPMixer, SWIFT, DPWMixer, WaveTuner(→T8), WaveRoRA, Wave-MaskMix, WDformer, WaveTS(→T10), TempoWave, WaveToken, AWGformer, CoSPOT, McWC, TriTS, BEAT, HADL |
| **T6 SWT/MODWT** | SimpleTM, NPMixer, MWMixer, HPMixer |
| **T8 wavelet packet** | WaveTuner |
| **T9 learnable wavelet** | DB2-TransF, Wavelet-MoE, AWGformer, Fully-learnable DWT |
| **T10 wavelet derivative** | WaveTS/WDT |
| **T11 Koopman** | Sonnet |
| **T12 graph Fourier** | xCPD, SpikF-GO, (MSGNet uses learned graph + FFT period) |
| **T13 NUDFT** | TFMixer |
| **T14 evolutionary spectra** | TimeES |
| **T15 period reshape (1D→2D/3D)** | Times2D, MSGNet, TriTS, PHAT, LightGTS |
| **T16 derivative/differencing** | Times2D, WaveTS, DTAF |
| **T17 normalization** | RevIN, WDAN, CANet, DTAF, StaTS |
| **T18 tokenization/quantization** | WaveToken, TempoWave |
| **T19 hyper-complex** | FIA-Net, SimpleTM |
| **T20 learnable frequency bases** | FreqLens, MoFE-Time, AdaMamba |

> The transform-card files backing this table live in `_local_archive/lab_extraction/transforms_*.md` (one card per paper with full math + algorithm); PHAT and MLF were extracted separately to `phat.txt` / `mlf.txt` in the same folder.

---

**END OF REFERENCE — Rev A.** Coverage: 20 transform families (T1–T20) + 7 external transforms (X1–X7) + **81 collected papers** indexed by their transform and axis coordinates — every transform paper read from its PDF (3 surveys + the slide deck + the project paper are reference material, not transform papers). Any new paper is recorded by its transform coordinates (Part 0), not by a new name.





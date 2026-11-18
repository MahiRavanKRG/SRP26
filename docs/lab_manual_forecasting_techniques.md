# Forecasting Techniques Research Laboratory — Operations Manual

| Field | Value |
|---|---|
| **Document No.** | SRP26-FTM-001 |
| **Title** | Forecasting Techniques Research Laboratory — Operations Manual |
| **Revision** | Rev B (Axis 1 Transform Reference expansion) |
| **Date** | 2026-11-16 |
| **Applicability** | All researchers (human) and AI agents working on the SRP26 time-series-forecasting program |
| **Prepared under** | University of Hildesheim · Supervisor: Jung Min Choi · Researcher: Aaditya Chaturvedi |
| **Governing sources** | [`AGENTS.md`](file:///d:/uni/SRP26/AGENTS.md) · [`docs/ai_development_methodology.md`](file:///d:/uni/SRP26/docs/ai_development_methodology.md) · [`docs/experiment_standardization_protocol.md`](file:///d:/uni/SRP26/docs/experiment_standardization_protocol.md) · [`docs/literature_mapping.md`](file:///d:/uni/SRP26/docs/literature_mapping.md) · `papers/` (collection) |

### Record of Revisions

| Rev | Date | Change | Authority |
|---|---|---|---|
| A | 2026-11-16 | Initial issue: full manual | Aaditya Chaturvedi |
| B | 2026-11-16 | Expand Axis 1 into a Transform Reference: localization-spectrum taxonomy, transform↔forecasting schemes (3 tiers), and empirical comparison protocol | Aaditya Chaturvedi |

### Distribution

- **Humans:** project team, supervisor (for review).
- **AI agents:** paste `Chapter 12 (AI Priming Directive)` verbatim at session start; load this whole file when the task concerns technique research, paper classification, or related work.

---

## Table of Contents

1. [Part I — Mission & Governance](#part-i--mission--governance)
2. [Part II — Theory: the unifying law](#part-ii--theory-the-unifying-law)
3. [Part III — Technique taxonomy & registry](#part-iii--technique-taxonomy--registry)
4. [Part IV — Service history (lineage)](#part-iv--service-history-lineage)
5. [Part V — Procedures](#part-v--procedures)
6. [Part VI — Knowledge transfer](#part-vi--knowledge-transfer)
7. [Appendices A–C](#appendices)

---

## Abbreviations & Glossary

> **Rule:** this glossary is normative. Use these definitions; do not re-derive them per session. Every term carries its provenance. Terms marked ⚠ are referenced but the source PDF is not in `papers/` (they are canonical ancestors cited by the in-collection papers).

### Signal & transform terms

- **Fourier Transform / FFT (Fast Fourier Transform) / DFT (Discrete Fourier Transform).** Maps a time-domain sequence to complex coefficients of sine/cosine basis functions, giving a *global* (all-time) frequency spectrum. Source: FEDformer, FreDF, FilterTS, FreEformer (in `papers/`).
- **rFFT (real FFT).** DFT variant for real inputs returning only non-redundant bins (Nyquist symmetry). Source: project `metrics.py`/`diagnostics.py`, DSOF framework.
- **STFT (Short-Time Fourier Transform).** Applies DFT inside sliding windows to get time-localized frequency content. Source: FIA-Net (`papers/[2025] Efficient Time Series Forecasting via Hyper-Complex Models…`).
- **DCT (Discrete Cosine Transform).** Real-valued orthonormal transform that decorrelates autocorrelated inputs; used to fix spectral bias in KANs. Source: AutoCorr-KAN (`papers/[2026] Autocorrelation Reintroduces Spectral Bias in KANs…`), project SORMixer (DCT-II FiLM).
- **Wavelet Transform / DWT (Discrete Wavelet Transform) / CWT (Continuous).** Decomposes a signal into approximation (low-frequency) and detail (high-frequency) coefficients at multiple scales, giving *localized* time-frequency resolution. Source: WaveForM, WPMixer, SWIFT, DPWMixer, WaveTuner (in `papers/`).
- **IDWT (Inverse DWT).** Reconstructs the time series from wavelet coefficients. Source: DPWMixer, WaveTS.
- **Stationary Wavelet Transform (SWT) / LSWT (Learnable SWT).** Shift-invariant DWT that keeps all time positions (no downsampling); LSWT makes the filter coefficients learnable. Source: SimpleTM, NPMixer, HPMixer.
- **Wavelet level / subband.** One layer of the recursive DWT decomposition; subbands = the approximation + detail bands at each scale. Source: WaveTuner, FilterTS.
- **Daubechies wavelets (db-N).** A specific orthogonal wavelet family; DB2-TransF learns their coefficients. Source: DB2-TransF.
- **Spectral leakage.** Energy from one frequency contaminating neighbouring bins, caused by finite windows / non-stationarity; a central obstacle to frequency methods. Source: FreDN ("spectral entanglement"), FBM.
- **PSD (Power Spectral Density).** Energy distribution across frequency; the diagnostic "where is the signal's energy." Source: project DSOF, `diagnostics.py`; FreqLens.
- **ACF (Autocorrelation Function).** Correlation of a series with lagged copies of itself; used to confirm candidate periods from FFT. Source: project CycleNet-Pro period discovery; AutoCorr-KAN.
- **Koopman operator.** Linear operator that advances observables of a (nonlinear) dynamical system; basis for spectral dynamics modeling. Source: Sonnet.
- **Evolutionary spectra.** Time-varying (non-stationary) generalization of a power spectrum; models how frequency content evolves. Source: TimeES.

### Modeling & training terms

- **Trend / seasonality / residual decomposition.** Additive split `x = T + S + R`; trend = slow smooth component, seasonality = periodic component, residual = leftover. Source: DLinear (referenced ⚠), CycleNet, FIRE, DecoVAE.
- **RevIN (Reversible Instance Normalization).** Per-instance normalization applied before the model and reversed after, to remove distribution shift. Source: referenced in AGENTS.md / protocol; used by WDAN, DTAF, CANet.
- **Channel independence (CI) vs dependence (CD).** CI = each variable modeled alone; CD = variables mixed. Adaptive CI/CD trades off noise vs cross-channel signal. Source: xCPD, FilterTS, project channel-mixing criterion.
- **Patch / patching.** Splitting the series into sub-series tokens before modeling. Source: HDMixer (LEP), HPMixer, NPMixer.
- **Mixture-of-Experts (MoE).** A router sends inputs to specialized expert networks; used to specialize per frequency band or per channel. Source: FreqMoE, Ada-MoGE, WaveMoE, MoFE-Time, WaveTuner.
- **KAN (Kolmogorov–Arnold Network).** Network with learnable activation functions on edges (vs fixed activations on nodes). Source: TFKAN, KFS, AR-KAN, AutoCorr-KAN, WaveTuner.
- **SSM / Mamba.** State-space sequence model with linear-time recurrence and a data-dependent "forgetting" gate. Source: AdaMamba, FRWKV+.
- **Diffusion (probabilistic forecasting).** Iteratively denoise from noise toward a sample, to output a distribution, not a point. Source: FALDA, StaTS, DecoVAE.
- **Foundation model / TSFM** (TSFM = this lab's shorthand). Large model pretrained on many datasets, then zero-shot/fine-tuned. Source: "Foundation Models for Time Series: A Survey"; WaveToken, LightGTS, WaveMoE.
- **Tokenizer / tokenization.** Mapping numeric series to a discrete vocabulary (often for LLMs). Source: WaveToken, TempoWave.
- **Distillation.** Training a small "student" model to mimic a large "teacher." Source: project current direction (DSOF-guided teacher–student); DDTime (dataset distillation).
- **Fine-tuning / zero-shot / few-shot.** Degrees of adaptation of a pretrained model to a target dataset. Source: foundation-model survey, CoSPOT, FM-LLM.
- **Lookback window (L) / horizon (H).** `L` = number of past steps input; `H` = number of future steps predicted. Standardized `L=336`, `H∈{96,192,336,720}` per [`experiment_standardization_protocol.md`](file:///d:/uni/SRP26/docs/experiment_standardization_protocol.md).
- **MSE / MAE.** Mean Squared Error / Mean Absolute Error — the primary point-forecast metrics.
- **CRPS (Continuous Ranked Probability Score).** Proper scoring rule for probabilistic forecasts. Source: DecoVAE, TimeES.

### Diagnostic terms (project-specific, normative)

- **Parseval's theorem / Parseval-exact decomposition.** Total signal energy is conserved across frequency bands, so subband error sums to total error exactly. Basis of the DSOF diagnostic. Source: [`docs/paper_review.md`](file:///d:/uni/SRP26/docs/paper_review.md) §4; `metrics.py`.
- **FVU (Fraction of Variance Unexplained).** Subband error normalized by subband variance; `FVU = 1` means "as bad as predicting the mean"; `> 1` means the model injects noise. Source: project DSOF.
- **Spectral coherence.** Phase alignment between predicted and true spectra per band. Source: project `diagnostics.py` (see failure-mode note in Ch 11).
- **Subband boundaries (project default).** Low / Mid / High bands as defined in [`docs/ai_development_methodology.md`](file:///d:/uni/SRP26/docs/ai_development_methodology.md) §2; dataset-specific cutoffs must be re-justified (see Ch 11 WARNING).

---

# Part I — Mission & Governance

## Chapter 1 — Purpose & Scope

### 1.1 Mission

This laboratory exists to **research forecasting techniques**, not merely to build models. A **technique** is a transferable, nameable mechanism with defined mathematics, provenance, and boundary conditions. A **model** is one particular assembly of techniques. Models are built; techniques are researched.

> **Normative distinction (binding).**
> - **Technique** — e.g., "lossless Haar-wavelet pyramid replaces average pooling" (DPWMixer); "route dominant frequencies to the frequency domain and the residual to the time domain" (RouteTS); "learn a per-frequency complex transfer function" (FreDF). Techniques are reusable across models.
> - **Model** — e.g., CycleNet, DLinear, HDMixer, FEDformer. Models are vehicles that instantiate one or more techniques and are benchmarked as wholes.

Every contribution a researcher or AI produces must be expressed first as a **technique delta** ("what transferable mechanism is new here?"), and only second as a model ("what architecture was benchmarked?").

### 1.2 Scope

- **Domain:** multivariate time-series forecasting (MTSF), with emphasis on lightweight models and frequency/wavelet/periodicity (spectral) techniques.
- **Evidence base:** the ~85 papers in `papers/` and the 3 survey "books" (see Appendix C).
- **Out of scope (treated only where they advance a core technique):** pure LLM wrappers, spiking networks, adversarial attacks, financial-only pipelines, multimodal text fusion — these are indexed in Appendix A but not core.

### 1.3 How to read this manual

1. **New researcher / AI:** read Part II (the law) first, then the Glossary, then Chapter 8 (paper classification) before touching any paper.
2. **Classifying a paper:** run Chapter 8's decision tree → assign a Tier → record in Appendix A.
3. **Developing a technique:** follow Chapter 9's checklist; do not build a model until the technique delta is stated.
4. **Running experiments:** Chapter 10 (protocol) is mandatory; Chapter 11 (diagnostics) is mandatory before any "frequency helps" claim.

---

## Chapter 2 — Authority & Ground Truth

### 2.1 Ground-truth hierarchy (binding, inherited from AGENTS.md)

1. **Code + results** in `models/`, `experiment_cyclenet.py`, `results/` are ground truth.
2. **This manual's taxonomy** is the reference vocabulary; do not silently rename techniques.
3. **Manuscript/prose** is derived reflection and must yield to (1) and (2).

### 2.2 Zero-hallucination rule (extended to this manual)

- Every technique, taxonomy entry, and tier verdict in this manual must cite a source: a file in `papers/` (path) or a project document (path).
- No number, no mechanism, and no "boundary condition" may be asserted from memory of the general literature. If the source is not in the collection, mark it ⚠ and name the paper that cites it.
- When a paper's abstract overstates a marginal contribution, the manual records the verdict **"overstated"** rather than repeating the claim.

### 2.3 Proposed-model invariant (binding)

The project's proposed model is **Automated CycleNet / CycleNet-Pro** (FFT+ACF period discovery, multi-cycle queues, optional channel mixing). **DLinear, PatchTST, HDMixer are baselines** — never the proposed model. See [`AGENTS.md`](file:///d:/uni/SRP26/AGENTS.md) §2.

---

# Part II — Theory: the unifying law

## Chapter 3 — The Unifying Law

> The ~85 collected papers are not ~85 independent inventions. They are ~85 variations of **one model** — the same move physics makes when it explains a wave by decomposing it and applying a conservation law.

### 3.1 The law (single equation)

```
x(t) = T(t)  +  Σₖ Aₖ(t)·e^(i·(ωₖ·t + φₖ(t)))  +  ε(t)
         │              │                          │
       trend      dominant periodic/             residual:
                  quasi-periodic                transients,
                  components                    spikes, noise
```

- **T(t)** — slow, smooth trend (near-zero frequency).
- **Σₖ Aₖ·e^(i(ωₖt+φₖ))** — a finite set of dominant periodic components, each with amplitude `Aₖ`, frequency `ωₖ`, phase `φₖ`. The field's core claim is that real series concentrate most energy in a *few* such components.
- **ε(t)** — everything else: localized high-frequency transients, spikes, and noise.

**Forecasting = extrapolating each component under a persistence assumption** — the future of `T`, of each `(Aₖ,ωₖ,φₖ)`, and of `ε` resembles its past. The entire field is about (a) how to *decompose* into these components, (b) how to *model* each with the right inductive bias, and (c) how to make the decomposition *adapt* so the persistence assumption survives non-stationarity.

### 3.2 The conservation law

**Parseval's theorem** is the field's conservation law: total energy is preserved across any orthogonal decomposition, so

```
E_total = E_low + E_mid + E_high        (exactly)
```

and therefore **subband error decomposes exactly**. This is why the project's DSOF diagnostic (Ch 11) can state *where* a model errs rather than only *how much*. It is the direct analog of energy conservation in physics: it tells you no model can "create" accuracy — it can only move error between bands.

### 3.3 The five-step recipe (every paper is a variant of this)

1. **Transform** the signal to expose structure (Fourier / Wavelet / derivative / 2D period-reshape / Koopman / learned filters).
2. **Decompose** into components (trend–season–residual; low/mid/high bands; multi-scale pyramid; cyclic-vs-residual).
3. **Model each component** with a matched inductive bias (linear for global periodic; MLP/KAN for nonlinear residual; attention for cross-channel; Mamba for long-range state; MoE for specialized experts).
4. **Adapt** the decomposition to be input-dependent and non-stationarity-aware (gates, routers, learned filters, RevIN, frequency differencing, stationarity weighting).
5. **Reconstruct + forecast**, supervised in the time domain, the frequency domain, or both.

**The shared belief:** *the right decomposition + the right per-component bias + adaptive routing beats a monolithic black box.* Every "new model" in the collection is a claim that a specific new choice at one or more of steps 1–5 improves the whole.

## Chapter 4 — Operating Assumptions & Failure Modes

> This chapter is the *boundary-conditions* catalog. It is the "fault modes" section of the manual: when and why the unifying law breaks. It is also the project's open research question.

### 4.1 The persistence assumption

Steps 1–5 succeed **only if the spectral structure of the past persists into the future**. Two distinct violations, each with a named symptom in the collection:

1. **Stationarity violation / regime shift** — the dominant frequencies `ωₖ` or their amplitudes `Aₖ` change across time. Symptom: global-FFT methods leak energy and smear transients (FEDformer-lineage → wavelet migration). Sources: FreDN ("spectral entanglement"), TIFO, DTAF, CANet, WDAN.
2. **Spectral leakage** — finite windows and non-stationarity smear energy into wrong bins, so "decompose then model" models the wrong components. Sources: FreDN, FBM ("inconsistent starting cycles").

### 4.2 Named failure modes (each maps to a source paper)

| # | Failure mode | What happens | Symptom metric | Source paper(s) |
|---|---|---|---|---|
| F1 | **Low-pass / over-smoothing** | Transformer/attention and pooling attenuate high-frequency detail | High-band FVU > 1.0 (worse than predicting the mean) | Dualformer, FAiT, DPWMixer; project DSOF finding |
| F2 | **Mid-frequency gap** | Energy concentrated at low-freq, mid-band nearly empty → models ignore mid-band, hurting long horizons | mid-band energy ratio ≈ 0, mid-FVU high | ReFocus, FreqCycle |
| F3 | **Recursive low-frequency bias** | Standard DWT keeps decomposing only the low band, underusing informative high-freq details | high-band info discarded | WaveTuner |
| F4 | **Fixed band/normalization mismatch** | Hardcoded cutoffs or scalers don't match the dataset's real periods | diurnal cycle (1/24) misassigned to wrong band | project DSOF (see Ch 11 WARNING); WDAN, CANet |
| F5 | **Spectral bias from autocorrelation** | Lagged (autocorrelated) inputs make even KANs prefer low frequencies | low-freq overfitting; DCT fixes it | AutoCorr-KAN |
| F6 | **Cross-domain heterogeneity** | Variables synchronized in time differ in frequency; single shared spectrum fails | per-channel spectral mismatch | AdaMamba, FilterTS, xCPD |
| F7 | **Irregular sampling** | Non-uniform timestamps violate the equidistant assumption of FFT/DWT | need NUDFT | TFMixer |
| F8 | **Over-stationarization** | Aggressive normalization destroys the signal it was meant to stabilize | accuracy loss after RevIN-style norm | CANet, WDAN |

### 4.3 The project's open question (binding research question)

> **"When does frequency-domain modeling actually help vs fail in lightweight MTS forecasting?"** (supervisor mandate, AGENTS.md §3).

The failure-mode catalog above is the concrete instance of this question. A technique is considered **researched** (not merely built) only when its help-region and fail-region in this catalog are stated (Ch 9 checklist).

---
### 4.4 Open problems (from the survey books)

> Grounded in the SSL survey (Zhang et al., TPAMI'24), the Foundation-Models survey (arXiv:2504.04011), and the frequency-domain pitch deck. Each open slot maps to a Ch 4 failure mode or a Ch 5 axis.

| Open problem | Source | Maps to |
|---|---|---|
| Inductive bias for time-series SSL — how to inject periodicity/trend/frequency priors | SSL survey §7.2 | Axis 4 (Adaptation) |
| SSL on irregular/sparse data **without** interpolation | SSL survey §7.3 | F7 |
| Multivariate support (most foundation models are univariate-only) | FM survey §5 | Axis 5 (Channel) |
| Probabilistic / uncertainty output (most are point-only) | FM survey §5 | Axis 7 (Output) |
| Interpretability / explainability & trust | FM survey §5 | TECH-12, TECH-34 |
| Scale vs computational-efficiency | FM survey §5 | Axis 3 |
| Error accumulation in autoregressive (long-horizon) generation | FM survey §5 | F1 |
| **Where frequency decomposition helps vs struggles** | Freq-domain pitch ("Topic") | **= SRP26 research question** |

---

# Part III — Technique taxonomy & registry

## Chapter 5 — The seven design axes (the "parts catalog")

> Any paper in the collection can be located by which axes it touches and what option it picks on each. This is the machine-and-human-parseable skeleton. An axis entry lists the known options and one exemplar paper each (all exemplars are in `papers/`).

### Axis 1 — Transform (how the signal is exposed)

> **The unifying idea.** A transform is a *choice of basis*: it rewrites the same signal `x(t) = Σₖ cₖ · φₖ(t)` in different coordinates. The time domain (identity) uses spikes `φₖ = δ(t−k)` — you see *when*, not *what frequency*. Global Fourier uses sinusoids `e^{iωₖt}` — you see *what frequencies*, spread over the whole window. Wavelets use localized wavelets at a scale + position; STFT uses sinusoids inside a fixed window. These are **not** three unrelated tricks — they are points on a **time↔frequency localization spectrum** governed by the Gabor/Heisenberg trade-off: *you cannot have perfect time and perfect frequency resolution at once* (FIA-Net, FEDformer, WaveForM — all in `papers/`).

> - **Global Fourier** = perfect frequency, zero time → right for *stationary* periodicity, wrong for regime shifts (spectral leakage; F1/F5).
> - **STFT** = one fixed compromise → wrong when the data has both very fast and very slow structure.
> - **Wavelet** = log-scale resolution (fine time at high frequency, fine frequency at low frequency) → the best general compromise, which is exactly why the field migrated Fourier → wavelet (PS-1).
> - **Learnable / adaptive bases** (DB2-TransF, NPMixer, WaveTuner) = don't fix the basis, *learn* it — the current frontier (PS-3).

**The localization spectrum** (the table to keep — maps every option to the Gabor trade-off):

| Transform | Time loc. | Freq loc. | Use when | Exemplar |
|---|---|---|---|---|
| Identity / time domain | perfect | none | transients, regime shifts, local patterns | DLinear ⚠, PatchTST ⚠, CycleNet, TQNet |
| Global FFT/DFT | none | perfect | strong *stationary* periodicity | FEDformer, FreDF, FilterTS, FIRE, FBM, TIFO, RouteTS |
| DCT (real cosine) | none | perfect (real) | decorrelation / denoising (fixes F5) | AutoCorr-KAN, HADL, SORMixer (project) |
| STFT | windowed | windowed | local periodic bursts | FIA-Net |
| DWT / SWT | high-freq: good | low-freq: good | multi-scale, non-stationary | WaveForM, WPMixer, SWIFT, DPWMixer, NPMixer, HPMixer |
| Learnable wavelet bank | learned | learned | fixed basis is suboptimal | DB2-TransF, NPMixer, WaveTuner |
| Wavelet derivative | good | good | change-points, abrupt regimes | WaveTS/WDT |
| Exotic bases | varies | varies | exogenous dynamics / inter-series / irregular sampling / non-stationary | Sonnet (Koopman), xCPD (graph-Fourier), TFMixer (NUDFT), TimeES (evolutionary) |
| 2D/3D period reshape | period-aligned | period-aligned | strong single/multi period | Times2D, MSGNet ⚠, TriTS, PHAT |

> **Two clarifications (normative).**
> 1. **Per-frequency transfer function** (FreDF) and **top-k / random mode subsetting** (FEDformer) are *modeling choices made inside* the global-FFT option, not distinct bases — they are sub-options of the "Global FFT/DFT" row above.
> 2. **Normalization** (RevIN ⚠, WDAN, CANet) and **differencing** (DTAF, WaveTS) are *distribution* transforms, not *basis* transforms — they are indexed under Axis 4 (Adaptation), not here. Do not double-count them as "basis" options.

**Transform ↔ forecasting scheme (the three tiers).** This is the trap that decides whether an empirical transform comparison is valid. A global transform maps `L` lookback coefficients ↔ `L` time steps, so *inverting it reconstructs the past, not the future*. Every pipeline must therefore choose how the model goes from `L` input coefficients to `H` future steps — and that choice is itself a confound unless fixed. Three schemes, measuring three different things:

| Tier | Scheme | What it measures | Project example |
|---|---|---|---|
| **1. Analytic-only** | closed-form extrapolation of the basis (top-k harmonics, trend extension); no learning | the basis's own *signal prior* — zero-parameter forecast ability | `srp26/fourier.py` (`extract_and_extrapolate_fourier_base_torch`) |
| **2. Fixed dumb head** | transform → shared linear/MLP head (`coeff_L → coeff_H`) → inverse | the transform's value *to a learned model* — **the headline comparison** | `models/baselines.py` `SpectralProjectionBranch`, `WaveletProjectionBranch` |
| **3. Non-causal upper bound** | model sees the whole `L+H` window in the basis, evaluates only the future | the basis's *representation ceiling* — separates "can't represent" from "can't extrapolate" | to be added |

Run all three. Tier 2 is the ranking you report; Tier 1 is the free-lunch control; Tier 3 stops you blaming the transform when the head is the bottleneck.

**Empirical comparison protocol** (isolate the transform as the single independent variable; per Ch 9 checklist + Ch 10):

1. **Fixed carrier.** One shared minimal backbone — the project `DLinear` linear head (a pure linear map) or a shared 3–4-layer MLP, identical width/depth across *every* transform. A linear head maximizes the ablation's signal-to-noise: the transform is then the *only* source of spectral power. (DLinear stays a baseline per AGENTS.md §2 — it is used here as the controlled carrier, not the proposed model.)
2. **Vary only the transform**, per-channel / channel-independent, with `L=336`, `H∈{96,192,336,720}`, canonical splits, RevIN on, 3 seeds `[2024,2025,2026]` (Ch 10).
3. **Energy normalization (mandatory).** FFT/DCT are (near-)orthonormal; STFT-with-window and some wavelets are not — they rescale signal energy and silently corrupt MSE. Normalize each transform to preserve Parseval energy (Ch 3.2), or the ranking measures "which transform rescales" rather than "which basis helps."
4. **Parameter/compute accounting.** `rFFT` yields `L/2+1 = 169` coeffs, DCT yields `L = 336`, DWT ~`L` across subbands → different head widths. Report `#params` (and ideally FLOPs) per transform; optionally project every basis to a common width to match capacity.
5. **Metrics.** Headline MSE/MAE **plus** DSOF subband FVU / energy ratio / LSD / rolling failure rate (Ch 11, `srp26/spectral.py`) — these turn "which is better" into "where/why it fails," which *is* the research question (Ch 4.3).
6. **Statistical rigor.** Across 3 seeds the deltas are small and noisy. Report mean ± std **and** a paired test (Wilcoxon signed-rank / paired bootstrap) per `(dataset, horizon)`, transform-vs-identity. Do not average across `H` — report per-horizon (long-horizon breakdown is the known ETTh2 failure).
7. **Synthetic tier for causality (recommended).** Benchmark datasets confound non-stationarity + multi-scale + noise, so they can *rank* transforms but not *explain* them. Add controlled signals — single/multi sinusoid, chirp, trend+seasonality, regime shift / changepoint, white noise, multi-period — each an isolated boundary condition; matching their rankings to the real-data rankings yields a *mechanism*, not just a score.

> **Full transform reference** (math + algorithm for every transform, exhaustive over the collection + an external appendix): [`docs/transforms_reference.md`](file:///d:/uni/SRP26/docs/transforms_reference.md) (SRP26-FTM-002).

### Axis 2 — Decomposition (how components are split)

| Option | What it does | Exemplar |
|---|---|---|
| Trend–season–residual | Additive split, model parts separately | DLinear ⚠, FIRE, DecoVAE, HPMixer |
| Band split (low/mid/high) | Split spectrum into bands, model each | FilterTS, ReFocus, FIRE, project DSOF |
| Multi-scale pyramid | Coarse-to-fine scale ladder | SWIFT, DPWMixer, MWMixer, WPMixer |
| Cycle vs residual | Learn a recurrent cycle, subtract it, model residual | CycleNet |
| Frequency disentanglement | Separate trend/period/noise *in frequency* (anti-leakage) | FreDN |
| Static vs dynamic filter | Global (dataset) + per-sample (dynamic) components | FilterTS |
| Period-heterogeneity bucket | 3D tensor by variable-group × phase × offset | PHAT |

### Axis 3 — Per-component model (the inductive bias per component)

| Option | Bias it enforces | Exemplar |
|---|---|---|
| Linear / single-layer | Global periodic structure is near-linear | CycleNet, SWIFT, DLinear ⚠, RouteTS |
| MLP / Mixer | Nonlinear residual, cheap | HDMixer, FreqMoE, DPWMixer |
| Attention (Transformer) | Long-range / cross-variable | FEDformer, FreEformer, Dualformer |
| Inverted attention | Attention over *channels* | FAiT, WDformer |
| KAN | Learnable activations, spectral plasticity | TFKAN, KFS, WaveTuner |
| Mamba / SSM | Linear-time long-range state | AdaMamba, FRWKV+ |
| MoE (per-band / per-channel) | Specialized experts routed by band/channel | FreqMoE, Ada-MoGE, WaveMoE |
| Diffusion / VAE | Probabilistic output per component | FALDA, StaTS, DecoVAE |

### Axis 4 — Adaptation (how the decomposition reacts to non-stationarity)

| Option | What it does | Exemplar |
|---|---|---|
| Learnable filter / gate | Filter weights depend on the input | FilterTS, AWEMixer, FreqLens |
| Router (MoE / frequency router) | Route input to the right band/expert | AWEMixer, Ada-MoGE, xCPD, RouteTS |
| RevIN / adaptive normalization | Remove per-instance distribution shift | WDAN, CANet, DTAF |
| Frequency differencing | Highlight components whose spectrum shifts | DTAF |
| Stationarity weighting | Weight *stationary* frequencies more | TIFO |
| Layer-wise band allocation | High-freq in shallow layers, low-freq deep | Dualformer |
| Periodic-position gating | Use periodic position cues as signed corrections | FRWKV+ |

### Axis 5 — Channel strategy (how variables interact)

| Option | What it does | Exemplar |
|---|---|---|
| Channel-independent (CI) | One variable at a time | PatchTST ⚠, project default |
| Channel-dependent (CD) | Mix all variables | FreEformer, WDformer |
| Adaptive CI/CD | Decide per patch/variable | xCPD, FilterTS, project criterion (Pearson+lag-xcorr) |
| Cross-variable as filters | Use other variables as frequency filters | FilterTS |
| Spectral coherence across variables | Couple variables by their spectral coherence | Sonnet |

### Axis 6 — Loss / supervision

| Option | What it does | Exemplar |
|---|---|---|
| Time-domain MSE/MAE only | Default | most |
| Frequency-domain loss | Supervise spectrum too (entropy reduction) | OLMA, Fre-CW |
| Band-balanced gradient | Upweight worse bands | BEAT |
| Output alignment | Penalize same-timestamp disagreement across lags | AliO |
| Cross-scale consistency | Align predictions across scales | MWMixer |

### Axis 7 — Output

| Option | What it does | Exemplar |
|---|---|---|
| Point forecast | Single trajectory | most |
| Probabilistic | Distribution (quantiles/CRPS) | FALDA, StaTS, DecoVAE, TimeES |
| Interpretable attribution | Per-frequency explanation of the forecast | FreqLens, FIRE, NFT |

---

## Chapter 6 — The technique registry

> The canonical technique set, each with a stable **TECH-ID**. This is the lab's currency: models are judged by *which techniques they carry* and *what each technique contributes* (not by the model's name). Provenance = the paper(s) that introduced/defined it in this collection.

### TECH-01 — Fourier-mode subset attention
- **Mechanism:** apply attention over a *selected* subset of Fourier modes (random or top-k by amplitude) instead of raw time tokens; reconstruct by inverse FFT.
- **Axes:** Transform, PerComponentModel. **Provenance:** FEDformer. **Lineage:** ancestor of the whole Fourier line (→ FreDF → FreEformer → FilterTS).
- **Boundary:** strong for global periodic structure; weak on non-stationary/transient data (F1, F2). **Tier:** Canonical.

### TECH-02 — Per-frequency transfer function (dynamic fusion)
- **Mechanism:** reformulate forecasting as learning, per frequency, a transfer function mapping past to future spectrum; predict each Fourier component separately and *dynamically fuse* them. Rejects the assumption that high-freq = noise.
- **Axes:** Transform, Decompose, Adaptation. **Provenance:** FreDF ("Not All Frequencies Are Created Equal"). **Lineage:** FEDformer → FreDF.
- **Boundary:** the role of each frequency depends on the dataset; fixed "discard high-freq" rules fail. Provides a generalization bound. **Tier:** Law (conceptual reframing).

### TECH-03 — Real/imag independent spectrum attention
- **Mechanism:** DFT → Transformer over the complex spectrum with real and imaginary parts processed independently; adds a learnable matrix to the attention map + row-wise L1 norm to fix low-rank, low-diversity attention.
- **Axes:** Transform, PerComponentModel. **Provenance:** FreEformer. **Lineage:** FreDF → FreEformer.
- **Boundary:** the attention fix is generic and reported to also improve non-frequency Transformers. **Tier:** Engineering.

### TECH-04 — Static + dynamic cross-variable frequency filtering
- **Mechanism:** a *static* global filter (learned over the whole training set, captures stable frequencies) plus a *dynamic* cross-variable filter (other variables act as filters to reinforce shared frequency components); convolutions become frequency-domain multiplications.
- **Axes:** Transform, Decompose, Adaptation, ChannelStrategy. **Provenance:** FilterTS. **Lineage:** FreEformer → FilterTS.
- **Boundary:** excels at shared cross-variable periodicity; the static/dynamic split is the key abstraction. **Tier:** Law (the static-vs-dynamic frequency distinction is reusable).

### TECH-05 — Amplitude/phase independent decomposition
- **Mechanism:** model amplitude and phase of frequency basis components *independently*, with adaptive per-basis weights and a targeted loss; yields interpretable, robust forecasts.
- **Axes:** Decompose, Adaptation, Output. **Provenance:** FIRE. **Lineage:** FreDF/FEDformer.
- **Boundary:** interpretability over raw accuracy; amplitude/phase separation avoids entanglement. **Tier:** High-value Engineering.

### TECH-06 — Fourier basis mapping (fixing cycle/sequence-length inconsistency)
- **Mechanism:** reinterpret FFT real/imag parts as cosine/sine basis coefficients; fixes "inconsistent starting cycles" and "inconsistent series length"; plug-and-play as a first projection layer (linear/MLP/Transformer variants FBM-L/NL/NP/S).
- **Axes:** Transform, Adaptation. **Provenance:** FBM. **Lineage:** FEDformer.
- **Boundary:** explicit frequency + preserved temporal detail; also adds centralization/patching/rolling-window techniques. **Tier:** Law (basis-correctness insight).

### TECH-07 — STFT window aggregation (hyper-complex MLP)
- **Mechanism:** STFT partitions the sequence, then a complex-valued MLP aggregates adjacent windows; hyper-complex algebra combines all windows with ~3× fewer params.
- **Axes:** Transform, PerComponentModel. **Provenance:** FIA-Net. **Lineage:** STFT line.
- **Boundary:** long-range dependencies at lower parameter cost. **Tier:** Engineering.

### TECH-08 — Stationarity-aware frequency weighting
- **Mechanism:** learn *which frequency components are stationary across the dataset* and weight them up while suppressing non-stationary ones; the Fourier transform induces an eigen-decomposition in frequency space.
- **Axes:** Transform, Adaptation. **Provenance:** TIFO. **Lineage:** FEDformer/FreDF.
- **Boundary:** directly attacks non-stationary distribution shift; plug-and-play. **Tier:** Law (answers "which frequencies are trustworthy").

### TECH-09 — Frequency-time routing
- **Mechanism:** partition the spectrum by amplitude; route *dominant* frequencies to a complex-valued linear predictor in the frequency domain, and revert *residual* spectral energy to the time domain for a lightweight MLP.
- **Axes:** Decompose, PerComponentModel, Adaptation. **Provenance:** RouteTS. **Lineage:** FreDF/TIFO.
- **Boundary:** principled answer to global-periodicity-vs-local-transience. **Tier:** Law (domain-routing principle).

### TECH-10 — Layer-wise frequency allocation
- **Mechanism:** dual time+frequency branches; a hierarchical sampling module assigns *different frequency bands to different layers* (high-freq preserved in shallow layers, low-freq trends in deep layers); periodicity-aware weighting balances branches by harmonic energy ratio.
- **Axes:** Transform, Adaptation. **Provenance:** Dualformer. **Lineage:** FEDformer/Transformer low-pass critique.
- **Boundary:** counteracts the Transformer low-pass effect (F1); strong on weakly-periodic data. **Tier:** Law (layer-wise band principle).

### TECH-11 — Evolutionary-spectra forecasting
- **Mechanism:** derive a parameterizable evolutionary-spectra formulation; recast non-stationary random-process modeling as learning an *evolving* spectrum modulated by random variables; complexity reduced by Hermitian symmetry + energy sparsity.
- **Axes:** Transform, Output. **Provenance:** TimeES (NeurIPS'26). **Lineage:** probabilistic/diffusion line.
- **Boundary:** unifies deterministic + probabilistic under one theory. **Tier:** Law (theory-grounded).

### TECH-12 — Learnable frequency discovery + axiomatic attribution
- **Mechanism:** parameterize frequency bases via sigmoid mapping and *learn* them from data (with diversity regularization), then attribute predictions to frequencies with Shapley-equivalent axioms (Completeness/Faithfulness/Null/Symmetry).
- **Axes:** Transform, Adaptation, Output. **Provenance:** FreqLens. **Lineage:** FIRE/FreDF interpretability.
- **Boundary:** recovers physically meaningful periods (24 h, 12 h, weekly) with provable attribution. **Tier:** Law (axiomatic attribution).

### TECH-13 — Lossless wavelet downsampling / lossless Haar pyramid
- **Mechanism:** use the (orthogonal, lossless) Haar wavelet instead of average pooling for downsampling — trends and fluctuations are disentangled *without information loss*, so high-freq transients survive.
- **Axes:** Transform, Decompose. **Provenance:** SWIFT (downsampling), DPWMixer (pyramid replaces pooling). **Lineage:** WaveForM → WPMixer → SWIFT/DPWMixer.
- **Boundary:** non-stationary and edge-deployment friendly; lossless pyramid is the key correction to aliasing. **Tier:** High-value Engineering (the lossless-pooling principle).

### TECH-14 — Learnable wavelet filter bank (incl. learnable stationary WT)
- **Mechanism:** make the wavelet decomposition filters *learnable* (learnable Daubechies coefficients; learnable stationary wavelet transform) so the basis adapts to the data instead of a fixed family.
- **Axes:** Transform, Adaptation. **Provenance:** DB2-TransF (learnable Daubechies), NPMixer (LSWT), HPMixer (LSWT), WaveTuner (subband tuning). **Lineage:** WPMixer.
- **Boundary:** replaces attention with a wavelet layer (DB2-TransF); shift-invariant (SWT). **Tier:** High-value Engineering (the "learn the basis" principle).

### TECH-15 — Full-spectrum subband tuning
- **Mechanism:** fix the DWT's *recursive low-frequency bias* by tuning *all* subbands: an adaptive router assigns subband weights; multi-branch KANs of distinct order model each subband.
- **Axes:** Transform, Decompose, Adaptation. **Provenance:** WaveTuner. **Lineage:** WPMixer/WaveForM.
- **Boundary:** directly targets F3 (recursive low-freq bias). **Tier:** Law (corrects a systemic wavelet flaw).

### TECH-16 — Wavelet derivative transform
- **Mechanism:** apply the wavelet transform to the *derivative* of the series, magnifying rate-of-change cues and regime shifts that raw coefficients miss.
- **Axes:** Transform. **Provenance:** WaveTS (Multi-Order WDT). **Lineage:** WPMixer.
- **Boundary:** change-point/abrupt-regime sensitivity. **Tier:** High-value Engineering.

### TECH-17 — Wavelet tokenization (foundation models / LLMs)
- **Mechanism:** scale + decompose the series, threshold + quantize wavelet coefficients into a small vocabulary, and pretrain to forecast coefficients; or embed scalar observations as multi-wavelet digit-wise tokens.
- **Axes:** Transform, Output. **Provenance:** WaveToken (TSFM tokenizer), TempoWave (LLM number interface). **Lineage:** wavelet line × foundation-model line.
- **Boundary:** small vocab (1024), zero-shot + in-domain. **Tier:** High-value Engineering.

### TECH-18 — Wavelet data augmentation
- **Mechanism:** mask/mix wavelet coefficients (WaveMask/WaveMix) to augment training while preserving temporal coherence.
- **Axes:** Transform, Loss (training-side). **Provenance:** Wave-MaskMix. **Tier:** Engineering.

### TECH-19 — Wavelet disentangled adaptive normalization
- **Mechanism:** DWT split into low-freq trend and high-freq fluctuation, then *tailor normalization to each* (differencing for the non-stationary trend to predict normalization params).
- **Axes:** Adaptation, Decompose. **Provenance:** WDAN. **Lineage:** RevIN ⚠ → WDAN.
- **Boundary:** non-stationarity (F4/F8). **Tier:** High-value Engineering.

### TECH-20 — Recurrent cycle subtraction
- **Mechanism:** learn a global *recurrent cycle* matrix (period W), subtract it from the input, and feed the unpredictable *residual* to a single linear/MLP layer.
- **Axes:** Decompose, PerComponentModel. **Provenance:** CycleNet (NeurIPS'24). **Lineage:** the project's direct parent.
- **Boundary:** strong on stationary seasonal cycles (Traffic, Electricity, ETTh1); weak when periods shift or cycles are absent (Weather, ETTm2). Hardcoded W and single-period are its known flaws. **Tier:** Canonical (the project's anchor).

### TECH-21 — Automated period discovery (FFT + ACF)
- **Mechanism:** FFT proposes candidate frequencies, ACF confirms them against noise; variance-reduction selects single- vs multi-cycle; optional channel-mixing enabled only when cross-variable correlation is high.
- **Axes:** Decompose, Adaptation, ChannelStrategy. **Provenance:** project CycleNet-Pro (`models/cyclenet/`), inspired by FreqLens learnable discovery + FreqCycle. **Lineage:** CycleNet.
- **Boundary:** removes hardcoded periods; the ACF confirmation filters FFT noise. **Tier:** project contribution (technique under research).

### TECH-22 — Period-heterogeneity phase alignment
- **Mechanism:** arrange multivariate input as a 3D "periodic bucket" (variable-group × phase-aligned time × offset), restrict interactions within buckets, mask cross-bucket; positive-negative attention models periodic alignment vs deviation.
- **Axes:** Decompose, ChannelStrategy. **Provenance:** PHAT. **Lineage:** CycleNet/TimesNet ⚠.
- **Boundary:** datasets where variables have *distinct, changing* periods. **Tier:** Law (period heterogeneity).

### TECH-23 — Periodic-nested group attention
- **Mechanism:** periodic-aware relative attention bias + grouped multi-query attention to handle *multiple coexisting* periodicities (daily + weekly) inside self-attention.
- **Axes:** PerComponentModel, Decompose. **Provenance:** PENGUIN. **Lineage:** Autoformer ⚠ / CycleNet.
- **Tier:** High-value Engineering.

### TECH-24 — Temporal query (periodically shifted learnable queries)
- **Mechanism:** use periodically shifted learnable vectors as attention *queries* (global inter-variable patterns) with keys/values from raw input (local sample correlations); single attention layer + MLP.
- **Axes:** PerComponentModel, ChannelStrategy. **Provenance:** TQNet (ICML'25). **Lineage:** channel-correlation line.
- **Boundary:** high-dimensional datasets at near-linear cost. **Tier:** High-value Engineering.

### TECH-25 — Multi-scale time-frequency (low-freq cycle + mid/high enhancement)
- **Mechanism:** FECF extracts low-freq shared periodic patterns in time; SFPL uses learnable filters + adaptive weighting to boost mid/high-freq energy; hierarchical MFreqCycle decouples nested multi-periodicity.
- **Axes:** Decompose, Adaptation. **Provenance:** FreqCycle. **Lineage:** CycleNet + ReFocus.
- **Boundary:** targets F2 (mid/high-freq neglect); pairs with ReFocus. **Tier:** High-value Engineering.

### TECH-26 — Frequency-band MoE
- **Mechanism:** decompose into frequency bands, route each band to a specialized expert, gate by band characteristics; or adapt expert count to the input's frequency distribution (Gaussian band-pass).
- **Axes:** PerComponentModel, Decompose, Adaptation. **Provenance:** FreqMoE, Ada-MoGE, Wavelet-MoE, MoFE-Time, WaveMoE. **Lineage:** FreDF/FilterTS × MoE.
- **Boundary:** efficiency (FreqMoE < 50k params; Ada-MoGE 0.2M); avoids discarding bands as noise. **Tier:** High-value Engineering (the band-MoE principle is Law-worthy).

### TECH-27 — Frequency-gated state space (Mamba)
- **Mechanism:** generate input-dependent frequency bases inside the Mamba update; generalize the temporal forgetting gate into a unified time-frequency forgetting gate.
- **Axes:** PerComponentModel, Adaptation. **Provenance:** AdaMamba. **Lineage:** Mamba ⚠ × frequency.
- **Boundary:** cross-domain heterogeneity (F6). **Tier:** High-value Engineering.

### TECH-28 — Time-frequency KAN / DCT spectral-bias fix
- **Mechanism:** dual time+frequency KAN branches; or DCT preprocessing to decorrelate autocorrelated inputs (theoretical result: autocorrelation reintroduces spectral bias in KANs; DCT removes it).
- **Axes:** Transform, PerComponentModel. **Provenance:** TFKAN, KFS, AR-KAN, AutoCorr-KAN. **Lineage:** KAN ⚠ × frequency.
- **Boundary:** F5 (spectral bias from autocorrelation) is the counterintuitive theory result. **Tier:** Law (AutoCorr-KAN theory) + Engineering (TFKAN/KFS/AR-KAN).

### TECH-29 — Fourier + conditional diffusion / spectral trajectory
- **Mechanism:** Fourier decomposition isolates the *noise* term which a conditional diffusion models; or a *learned* noise schedule with spectral regularization (Spectral Trajectory Scheduler) + frequency-guided denoiser.
- **Axes:** Output, Decompose, Adaptation. **Provenance:** FALDA, StaTS. **Lineage:** diffusion ⚠ × Fourier.
- **Boundary:** probabilistic forecasts; reduces epistemic uncertainty; fewer sampling steps. **Tier:** High-value Engineering.

### TECH-30 — Koopman spectral operator + coherence attention
- **Mechanism:** learnable wavelet transform + Koopman spectral analysis; Multivariable Coherence Attention models variable dependencies via spectral coherence.
- **Axes:** Transform, ChannelStrategy. **Provenance:** Sonnet. **Lineage:** Koopman ⚠ × wavelet.
- **Boundary:** exogenous-variable integration; coherence fixes naive attention. **Tier:** Law (coherence attention principle).

### TECH-31 — Frequency-domain loss / entropy reduction
- **Mechanism:** supervise in the frequency domain (DFT+DWT) along time and channel dims; a unitary transform (DFT) lowers label entropy → lower error floor; addresses frequency bias in training.
- **Axes:** Loss. **Provenance:** OLMA, Fre-CW (adversarial), BEAT (band-balanced gradient).
- **Boundary:** training-side, model-agnostic, pluggable; caveat — channel-DFT raises entropy on ECL (frequency loss is not uniformly beneficial). **Tier:** Law (entropy-lower-bound + frequency-bias insight).

### TECH-32 — Output alignment (time + frequency)
- **Mechanism:** reduce prediction disagreement for the same timestamp across lagged inputs, in both time and frequency; quantified by a Time Alignment Metric (TAM).
- **Axes:** Loss, Output. **Provenance:** AliO (NeurIPS'25). **Tier:** High-value Engineering.

### TECH-33 — Spectral prompts for frozen LLMs
- **Mechanism:** keep an LLM frozen; inject frequency-basis prompts (compositional spectral prompts / frequency-guided prompts / spectral token aligner) so the LLM reads spectral structure; asymmetric MoE decoder separates periodic vs residual experts.
- **Axes:** Transform, PerComponentModel, Adaptation. **Provenance:** CoSPOT, FreSia, FM-LLM, SpecTF. **Lineage:** WaveToken × LLM ⚠.
- **Boundary:** few-shot/zero-shot + distribution shift; small updated parameter count. **Tier:** High-value Engineering.

### TECH-34 — Parseval subband diagnostics (DSOF)
- **Mechanism:** split the forecast error into Low/Mid/High bands using Parseval-exact decomposition; report subband FVU, PSD, spectral coherence, and sliding-window failure localization.
- **Axes:** Output (diagnostic), Loss (guiding). **Provenance:** project DSOF (`metrics.py`, `diagnostics.py`). **Lineage:** FreqLens (attribution) + ReFocus/FreqCycle (band importance).
- **Boundary:** exposes F1 (high-band FVU>1), F2 (mid-band gap), F4 (band-cutoff mismatch). **Tier:** project contribution — the diagnostic layer the rest of the field mostly lacks.

### TECH-35 — Learnable frequency disentanglement (anti spectral leakage)
- **Mechanism:** a learnable Frequency Disentangler separates trend vs periodic vs noise components *directly in the frequency domain*, countering spectral leakage and non-stationarity ("spectral entanglement"); a ReIm block shares real/imag parameters to halve complex-valued cost.
- **Axes:** Decompose, Adaptation. **Provenance:** FreDN. **Lineage:** FreDF / FilterTS.
- **Boundary:** non-stationary series where spectral leakage entangles components; ReIm sharing cuts params/compute ≥ 50%. **Tier:** Law (names "spectral entanglement" as a phenomenon).

### Tier legend (normative)

| Tier | Meaning | Verdict rule |
|---|---|---|
| **Canonical** | The foundational ancestor(s) a lineage descends from | Introduced a now-dominant pattern the field still cites |
| **Law** | A genuine abstraction, theorem, or reframing that re-explains the field | Has a named principle beyond "SOTA numbers"; reusable as a *technique* independent of the model |
| **High-value Engineering** | A well-executed recombination that matters empirically | New axis option or a correction to a known flaw, but no new governing principle |
| **Incremental** | Marginal delta over a clear template | Same axes, same options, new acronym + standard benchmarks; often "overstated" |

---
# Part IV — Service history (lineage)

## Chapter 7 — The evolution tree & paradigm shifts

> This is the "service history" of the technique family: who descends from whom, and *why* the field changed direction. Derived from [`docs/literature_mapping.md`](file:///d:/uni/SRP26/docs/literature_mapping.md). Read it once to understand the "physics" of the field's motion; do not memorize model names.

### 7.1 Lineage lanes

```
Frequency-domain / spectral forecasting
├── A. Global Fourier        FEDformer → FreDF → FreEformer → FilterTS → FIRE → TIFO → RouteTS
│                             ├─ FIA-Net(STFT) · FBM · OLMA · APRNet · Fre-CW · TimeES · FreqLens
├── B. Wavelet/multi-res     WaveForM → WPMixer → SWIFT → WaveTuner → AWEMixer → WaveMoE → NPMixer → MWMixer
│                             ├─ WaveRoRA · Wave-MaskMix · WDformer · DPWMixer · DB2-TransF · WaveTS · HADL · TempoWave
├── C. Periodicity/cycles    CycleNet → TQNet → LightGTS → PHAT → FreqCycle → HPMixer → PENGUIN
├── D. Foundation/LLM × freq WaveToken → MoFE-Time → FM-LLM → FreSia · CoSPOT · TempoWave · SpecTF
├── E. Next-gen backbones    KAN(TFKAN→KFS→AR-KAN→AutoCorr-KAN) · Mamba(AdaMamba·FRWKV+) · MoE(FreqMoE→WaveMoE) · Diff( FALDA→StaTS→TimeES→DecoVAE) · Spiking(SpikF-GO)
└── F. Non-stationarity/diag RevIN ⚠ → DLinear ⚠ → WDAN → DTAF → CANet → TimeEmb → TIFO · FreqLens · DSOF(ours)
```

### 7.2 The four paradigm shifts (service bulletins)

| # | Shift | From → To | Why (the observed fault) | Key papers |
|---|---|---|---|---|
| PS-1 | **Representation** | Global Fourier (2022–24) → localized wavelets (2025–26) | FFT leaks on non-stationary data (F1/F5) | FEDformer → WaveForM → WaveTuner |
| PS-2 | **Parameter** | Heavy attention → lightweight mixers/linears → MoE/state-space | quadratic attention too costly for edge/LTSF | DLinear ⚠, CycleNet → FreqMoE, AdaMamba |
| PS-3 | **Target** | Static preprocessing → learnable in-model spectral operators | fixed transforms can't adapt to the data | FreDN, FIRE, TIFO, DB2-TransF, NPMixer |
| PS-4 | **Integration** | Frequency as a standalone model → frequency as a module inside LLM/KAN/Mamba/diffusion | cross-pollination; frequency becomes a plug-in | WaveToken, FM-LLM, AdaMamba, StaTS |

### 7.3 Where the project sits (normative)

- **Parents:** CycleNet (periodicity) + FEDformer/FreDF (frequency) + the 2025–26 wavelet/adaptive line.
- **Proposed model:** CycleNet-Pro (TECH-21) with four backbones incl. SORMixer (DCT-II spectral FiLM).
- **Diagnostic:** DSOF (TECH-34) — Parseval-exact subband FVU, PSD, failure localization.
- **Current direction:** DSOF-guided teacher–student distillation (small hybrid time+frequency student from a foundation teacher), with band-selective spectral loss.

---

# Part V — Procedures

## Chapter 8 — Paper ingest & classification procedure (SOP)

> Purpose: reduce any new paper to (a) its technique delta and (b) a tier, in ~30 minutes. Run this **before** trusting a paper's claims.

### 8.1 Decision tree

```
START
├─ Q1. Is the "new thing" a TRANSFERABLE mechanism (technique) or only a MODEL assembly?
│   ├─ only a model assembly (new acronym, same axes/options)  → Tier: Incremental
│   └─ a nameable mechanism  → go Q2
├─ Q2. Does it introduce a NEW abstraction, theorem, or reframing (a "law"), or correct a known systemic flaw?
│   ├─ YES → Tier: Law   (record the named principle)
│   └─ NO  → go Q3
├─ Q3. Does it introduce a NEW option on a design axis, OR fix a named failure mode (Ch 4) with evidence?
│   ├─ YES → Tier: High-value Engineering
│   └─ NO  → Tier: Incremental
└─ Q4. Is the headline claim backed by the method + evaluation as written?
    ├─ NO → mark "OVERSTATED" regardless of tier
    └─ YES → record headline claim verbatim
```

### 8.2 Ingest record (mandatory fields)

For every paper added to the collection, complete this block and append a row to Appendix A:

```
- Stamp (arXiv YYMM / year) · Venue
- One new thing (1 sentence, the technique delta)
- Axes touched (from Ch 5)
- Technique(s) carried (TECH-ID(s) from Ch 6)
- Tier (Canonical / Law / Engineering / Incremental)
- Boundary condition claimed (helps/fails) or "not stated"
- Overstated? (yes/no + reason)
```

### 8.3 Anti-drift rules

1. The **technique delta** is the unit of record, not the model name. Two papers with different names but the same technique delta are the *same contribution* — the later one is Incremental.
2. "SOTA on N benchmarks" alone is **not** a contribution. It does not change the tier.
3. A paper that only swaps the backbone (e.g., "wavelet + X" where X ∈ {mixer, Mamba, KAN, MoE}) is Engineering at best, Incremental by default.

---

## Chapter 9 — Technique development procedure (SOP)

> Purpose: research a *technique*, not build a model. Execute in order; do not skip the boundary-conditions step (it is what separates research from tinkering).

### 9.1 Checklist

- [ ] **1. State the hypothesis** in the law's terms (Ch 3): which step (transform/decompose/model/adapt/reconstruct) is wrong, and what replacement is proposed.
- [ ] **2. Name the technique** with a stable TECH-ID candidate (kebab-case) and a one-line mechanism.
- [ ] **3. Identify the failure mode it targets** (F1–F8, Ch 4) or state "new failure mode" with evidence.
- [ ] **4. Define the boundary conditions *before* training:** on which datasets/regimes it must help (stationary periodic) vs fail (non-stationary/regime-shift). Use the Ch 4 catalog.
- [ ] **5. Select baselines that carry the *closest* technique** (same axes, previous option) so the delta is isolated — not just any strong model.
- [ ] **6. Follow the evaluation protocol (Ch 10) exactly** — L, H, splits, RevIN, seeds.
- [ ] **7. Run the DSOF diagnostic (Ch 11)** and report *where* error moved (bands), not just *how much* (total MSE).
- [ ] **8. Report the boundary result honestly:** "helps on X, fails on Y" — including the fail cases. A technique with no fail-region is uncharacterized, not superior.

### 9.2 Definition of "done"

A technique is **researched** only when: named → mechanism stated → failure mode targeted → boundary conditions measured (helps-region + fail-region) → band-level evidence (DSOF) → honest claim. A model without these is **built**, not researched.

---

## Chapter 10 — Evaluation protocol (binding)

> Full specification in [`docs/experiment_standardization_protocol.md`](file:///d:/uni/SRP26/docs/experiment_standardization_protocol.md). Summary (normative):

| Parameter | Value |
|---|---|
| Lookback `L` | **336** |
| Horizons `H` | **{96, 192, 336, 720}** |
| ETT splits | 12 / 4 / 4 months (Autoformer/PatchTST convention) |
| Non-ETT splits | 70% train / 10% val / 20% test |
| History overlap | val/test slices must include preceding `L` steps |
| Scaler | `StandardScaler` fitted **on train only** |
| Normalization | RevIN for non-stationary distribution shift |
| Seeds | **[2024, 2025, 2026]** — report mean ± std |
| Env | `uv run python <script>.py` |

**Zero-tolerance:** every reported number for our models must trace to a file in `results/`. Baseline numbers must cite the exact source table (HDMixer AAAI-24, PatchTST ICLR-23, DLinear AAAI-23).

---

## Chapter 11 — Diagnostic & fault-isolation procedure (DSOF)

> Purpose: answer "did frequency modeling help, and *where*?" using Parseval-exact subband decomposition. This is the lab's instrument panel; no "frequency helps" claim is admissible without it.

### 11.1 The four readings

| Reading | What it reports | Instrument |
|---|---|---|
| **Subband FVU** | Fraction of variance unexplained per Low/Mid/High band | `metrics.py` |
| **PSD** | Energy distribution of prediction vs truth | `diagnostics.py` |
| **Spectral coherence** | Phase alignment per band | `diagnostics.py` |
| **Failure localization** | Sliding-window error vs local spectrum | `diagnostics.py` |

### 11.2 Fault-isolation table (symptom → diagnosis → source)

| Symptom | Diagnosis | Fault (Ch 4) |
|---|---|---|
| High-band FVU > 1.0 | Model injects noise in high band (worse than mean) | F1 (over-smoothing + compensation) |
| Mid-band FVU high + mid energy ≈ 0 | Mid-frequency gap neglected | F2 |
| FVU ≈ 1.0 on Weather/ETTm2 | Non-stationary regime; persistence assumption broken | F1/F4 |
| Coherence stuck at 1.0 | Degenerate estimator (single segment) — metric bug, not physics | metric defect |
| Diurnal cycle (1/24) classified as "low" | Band cutoffs mismatched to dataset | F4 |

### 11.3 WARNING (normative)

Band cutoffs are **not** universal. With `low=0.1, high=0.4` normalized: `f=1/24≈0.042` and `f=1/168≈0.006` fall in the **low** band — so "mid-band captures diurnal/weekly cycles" is **false** for ETTh1. Re-justify cutoffs per dataset before writing any band narrative. Source: [`docs/paper_review.md`](file:///d:/uni/SRP26/docs/paper_review.md) §2.3.

---

# Part VI — Knowledge transfer

## Chapter 12 — AI Priming Directive (paste at session start)

```markdown
You are assisting the SRP26 Forecasting Techniques Research Laboratory
(University of Hildesheim, Supervisor: Jung Min Choi).
Read before acting: AGENTS.md, docs/ai_development_methodology.md,
docs/lab_manual_forecasting_techniques.md, and the latest log in .ai_history/.

BINDING RULES:
1. The unit of knowledge is the TECHNIQUE (transferable mechanism), not the model.
   Every contribution = a technique delta, not "a new model name".
2. Proposed model is Automated CycleNet / CycleNet-Pro. DLinear, PatchTST,
   HDMixer are baselines — never the proposed model.
3. Ground truth = code in models/ + results in results/. Never hallucinate numbers.
   Every technique/claim must cite a file in papers/ or a project doc.
4. Classify any paper with Chapter 8 (decision tree) and record it in Appendix A.
5. Run experiments only per Chapter 10 (L=336, H in {96,192,336,720}, canonical
   splits, RevIN, seeds [2024,2025,2026], uv run).
6. No "frequency helps" claim without DSOF diagnostics (Chapter 11).
7. Use `uv run python <script>.py`; LaTeX edits must compile with 0 errors.
```

## Chapter 13 — The human briefing (teaching the mental model)

**The 60-second version.** Every forecasting paper in this collection is the same move: **decompose the signal into trend + a few periodic components + residual, then model each part with the right tool, then adapt that split so it survives change.** The "physics" is Parseval's theorem — energy (and therefore error) is conserved across bands, so you can measure *where* a model fails, not just *how much*. The open question is the boundary: **when does the decomposition's persistence assumption break** (non-stationarity, regime shift, spectral leakage).

**Teaching script (to a colleague or a fresh AI):**
1. Give them Chapter 3 (the law + five-step recipe). Check understanding by asking: "put FEDformer on the five steps."
2. Give them Chapter 5 (the axes). Check: "place any new paper by its axis options."
3. Give them Chapter 8 (the decision tree). Check: "is 'wavelet + Mamba' a Law or Incremental? Why?"
4. Give them Chapter 11 (diagnostics). Check: "what does FVU>1 in the high band mean?"

**One-line thesis to memorize:**
> *"The right decomposition + the right per-component bias + adaptive routing beats a black box — but only where the spectrum persists; the research is finding that boundary."*

---
# Appendices

## Appendix A — Full paper index (by lane)

> Tier legend: **L**=Law · **C**=Canonical · **E**=High-value Engineering · **I**=Incremental. ⚠ = canonical ancestor referenced by the collection but whose PDF is not in `papers/`. Stamp = arXiv YYMM.

### Lane A — Global Fourier / spectral

| Stamp | Paper | Technique delta (one line) | Tier |
|---|---|---|---|
| 2201 | FEDformer | Fourier-mode subset attention (random/top-k modes) | C |
| 2405 | NFT (Neural Fourier Transform) | Multi-dim FFT + temporal CNN, interpretable | E |
| 2407 | FreDF ("Not All Frequencies…") | Per-frequency transfer function + dynamic fusion | L |
| 2501 | FreEformer | Real/imag independent spectrum attention + low-rank attention fix | E |
| 2502 | FIA-Net | STFT window aggregation via hyper-complex MLP | E |
| 2502 | ReFocus | Re-energize neglected mid-frequency band + shared key-frequency | E |
| 2504 | Times2D | 1D→2D period reshape + derivative heatmaps | E |
| 2505 | FilterTS | Static + dynamic cross-variable frequency filtering | L |
| 2505 | AEFIN | Fourier + cross-attention for non-stationary | I |
| 2505 | OLMA | Frequency-domain supervision + entropy lower bound | L |
| 2505 | Sonnet | Koopman operator + multivariable spectral-coherence attention | L |
| 2507 | FBM | Fourier-as-basis mapping; fixes cycle/seq-length inconsistency | L |
| 2508 | APRNet | Amplitude/phase reconstruction + KAN local correlation | E |
| 2508 | Fre-CW | Frequency-domain adversarial attack (niche) | I |
| 2510 | FIRE | Amplitude/phase independent interpretable decomposition | E |
| 2510 | TimeEmb | Static–dynamic disentangle + frequency filtering | E |
| 2511 | FreDN | Learnable frequency disentangler (anti spectral leakage) | L |
| 2511 | DDTime | Spectral-alignment dataset distillation (niche) | E |
| 2601 | Dualformer | Layer-wise frequency band allocation | L |
| 2602 | FreqLens | Learnable frequency discovery + axiomatic (Shapley) attribution | L |
| 2602 | TIFO | Stationarity-aware frequency weighting | L |
| 2602 | TFMixer | NUDFT for irregular time series (niche) | E |
| 2606 | SpecReTF | Spectral retrieval-augmented forecasting (niche) | E |
| 2608 | RouteTS | Route dominant freqs→freq domain, residual→time | L |
| 2609 | TimeES | Evolutionary-spectra theory (deterministic + probabilistic) | L |
| 2610 | AliO | Time+frequency output alignment | E |

### Lane B — Wavelet / multi-resolution

| Stamp | Paper | Technique delta | Tier |
|---|---|---|---|
| 2301 | WaveForM | Graph-enhanced multi-scale DWT | C |
| 2412 | WPMixer | Wavelet patch mixing (multi-resolution) | C |
| 2412 | WaveToken | Wavelet tokenizer for foundation models | E |
| 2408 | Wave-MaskMix | Wavelet coefficient masking/mixing augmentation | E |
| 2410 | WaveRoRA | Wavelet + linear-complexity rotary-route attention | E |
| 2501 | SWIFT | Lossless wavelet downsampling, ultra-light | E |
| 2502 | HADL | DWT+DCT denoising + low-rank linear head | E |
| 2505 | WaveTS/WDT | Multi-order wavelet *derivative* transform | E |
| 2506 | WDAN | Wavelet disentangled adaptive normalization | E |
| 2508 | Wavelet-MoE | Wavelet + MoE channel clustering | E |
| 2509 | WDformer | Wavelet + differential attention | E |
| 2511 | AWEMixer | FFT router + adaptive wavelet subband weighting | E |
| 2511 | WaveTuner | Full-spectrum subband tuning (fixes recursive low-freq bias) | L |
| 2512 | DPWMixer | Lossless Haar pyramid replaces pooling | E |
| 2512 | DB2-TransF | Learnable Daubechies replaces attention | E |
| 2512 | Ada-MoGE | Adaptive Gaussian band-pass MoE | E |
| 2601 | AWGformer | Adaptive wavelet-guided transformer | I |
| 2604 | WaveMoE | Wavelet-MoE foundation model | E |
| 2605 | NPMixer | Learnable stationary wavelet patch mixer (Choi) | E |
| 2606 | TempoWave | Multi-wavelet number embeddings for LLMs | E |
| 2606 | McWC | Cyclicity + wavelet + channel correlation | E |
| 2609 | MWMixer | Multi-scale frequency compensation | E |
| — | SimpleTM | SWT tokenization + geometric product attention | E |
| — | Fully-learnable DWT (monitoring) | Learnable DWT for unsupervised monitoring (niche) | E |

### Lane C — Periodicity / cycles

| Stamp | Paper | Technique delta | Tier |
|---|---|---|---|
| 2405 | CycleNet | Recurrent cycle subtraction + linear residual | C |
| 2505 | TQNet | Periodically-shifted temporal queries (channel correlation) | E |
| 2506 | LightGTS | Periodical tokenization, lightweight foundation | E |
| 2508 | PENGUIN | Periodic-nested group attention | E |
| 2511 | MLF | Multi-period learning for finance (niche) | I |
| 2602 | PHAT | Period-heterogeneity phase-aligned bucket + pos/neg attention | L |
| 2602 | HPMixer | Periodicity + learnable stationary wavelet | E |
| 2603 | FreqCycle | Low-freq cycle + mid/high-freq filter enhancement | E |
| — | MSGNet ("Learning multi-scale inter-series correlations") | FFT multi-scale graph convolution | E |

### Lane D/E — Foundation/LLM × frequency & next-gen backbones

| Stamp | Paper | Technique delta | Tier |
|---|---|---|---|
| 2501 | FreqMoE | Frequency-band mixture-of-experts | E |
| 2505 | FALDA | Fourier decomposition + conditional diffusion | E |
| 2506 | TFKAN | Dual time+frequency KAN branches | E |
| 2507 | MoFE-Time | Time+frequency MoE, pretrain–finetune | E |
| 2508 | KFS | KAN + Parseval-based dominant-frequency selection | E |
| 2509 | AR-KAN | Autoregressive-weight-enhanced KAN | E |
| 2511 | StaTS | Spectral-trajectory diffusion schedule (2026-03) | E |
| 2602 | SpecTF | Text↔spectral fusion (multimodal) | E |
| 2603 | xCPD | Graph-spectral channel-patch routing | E |
| 2604 | AdaMamba | Adaptive frequency-gated Mamba | E |
| 2604 | AutoCorr-KAN | DCT removes autocorrelation-induced spectral bias | L |
| 2604 | TriTS | Time+frequency+2D-vision disentanglement | E |
| 2605 | FRWKV+ | Frequency-space linear + periodic-position gating | E |
| 2606 | FAiT | Inverted attention as high-pass complement | E |
| 2606 | SpikF-GO | Spiking Fourier graph operators (niche) | E |
| 2608 | FM-LLM | Frequency-MoE for frozen LLMs | E |
| 2608 | DecoVAE | Complex-Gaussian trend-seasonal VAE | E |
| 2609 | CoSPOT | Compositional spectral prompts for LLMs | E |
| 2610 | FreSia | Frequency-semantic alignment for LLMs | E |

### Lane F — Non-stationarity & baselines (referenced)

| Stamp | Paper | Technique delta | Tier |
|---|---|---|---|
| 2109 | RevIN ⚠ | Reversible instance normalization | C |
| 2205 | DLinear ⚠ | Moving-average trend/seasonal baseline | C (baseline) |
| 2211 | PatchTST ⚠ | Patching + channel independence | C (baseline) |
| 2402 | HDMixer | Length-extendable patcher + hierarchical mixing | C (baseline) |
| 2501 | BEAT | Band-balanced gradient tuning | E |
| 2504 | CANet | Chrono-adaptive non-stationary normalization | E |
| 2511 | DTAF | Temporal stabilization + frequency differencing | E |

> **Duplicate note:** HDMixer is stored as 3 duplicate PDFs (`00533-AAAI24.HuangQ.pdf`, `[2024] HDMixer AAAI24.HuangQ.pdf`, `[2024] HDMixer Hierarchical Dependency….pdf`) — one paper, indexed once.

### Flagged (overstated or soft) claims

| Paper | Flag (source: verified against full text) |
|---|---|
| WaveForM | "300% gain" measured vs only 2 baselines on a 6-variable dataset — overstated |
| WPMixer | "significantly outperforms SOTA" but trails TimeMixer on Electricity — overstated |
| SWIFT | accuracy "SOTA" is soft; efficiency (25% params of a linear layer) is the solid claim |
| DB2-TransF | abstract "comparable or superior" vs conclusion "outperformed SOTA" — internally inconsistent |

### Survey "books" (not tiered)

| File | Role |
|---|---|
| Self-supervised learning for time series analysis (survey) | SSL taxonomy + prospects |
| Foundation Models for Time Series (survey) | TSFM taxonomy |
| Timeseries Forecasting in the Frequency Domain | **Slide deck** (SRP pitch, J.M. Choi) — defines time/freq domain + FFT/DWT/SWT roles; not a full survey |
| srpconf2025 | The project's own paper |

---

## Appendix B — Model → technique cross-reference

> Look up a model name to find which techniques (TECH-IDs, Ch 6) it carries.

| Model | Carries TECH-ID(s) |
|---|---|
| FEDformer | 01 |
| FreDF | 02 |
| FreEformer | 03 |
| FilterTS | 04 |
| FIRE | 05 |
| FBM | 06 |
| FIA-Net | 07 |
| TIFO | 08 |
| RouteTS | 09 |
| Dualformer | 10 |
| TimeES | 11 |
| FreqLens | 12 |
| SWIFT / DPWMixer | 13 |
| DB2-TransF / NPMixer / HPMixer / WaveTuner | 14, 15 |
| WaveTS | 16 |
| WaveToken / TempoWave | 17 |
| Wave-MaskMix | 18 |
| WDAN | 19 |
| CycleNet | 20 |
| CycleNet-Pro (ours) | 21 |
| PHAT | 22 |
| PENGUIN | 23 |
| TQNet | 24 |
| FreqCycle | 25 |
| FreqMoE / Ada-MoGE / WaveMoE / MoFE-Time | 26 |
| FreDN | 35 |
| AdaMamba | 27 |
| TFKAN / KFS / AR-KAN / AutoCorr-KAN | 28 |
| FALDA / StaTS / DecoVAE | 29 |
| Sonnet | 30 |
| OLMA / BEAT / Fre-CW | 31 |
| AliO | 32 |
| CoSPOT / FreSia / FM-LLM / SpecTF | 33 |
| DSOF (ours) | 34 |

---

## Appendix C — Primary-source register (exact paths)

> The full curated abstract set is [`papers/bibliography.md`](file:///d:/uni/SRP26/papers/bibliography.md). All PDFs live in `papers/`. Below are exact paths for the **Canonical + Law** papers (the ones to open first); use `glob papers/*` to enumerate the rest.

- FEDformer — `papers/FEDformer Frequency Enhanced Decomposed Transformer.pdf`
- FreDF — `papers/[2024] Not All Frequencies Are Created EqualTowards a Dynamic Fusion of Frequencies in Time-Series Forecast.pdf`
- FilterTS — `papers/[2025] FilterTS Comprehensive Frequency Filtering for Multivariate Time Series Forecasting.pdf`
- FBM — `papers/[2025] Fourier Basis Mapping A Time-Frequency Learning Framework for Time Series Forecasting.pdf`
- FreDN — `papers/[2025] FreDN Spectral Disentanglement for Time Series Forecasting via Learnable Frequency Decomposition.pdf`
- OLMA — `papers/[2025] OLMA One Loss for More Accurate Time Series Forecasting.pdf`
- Dualformer — `papers/[2026] Dualformer Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting.pdf`
- TIFO — `papers/[2026] TIFO Time-Invariant Frequency Operator for Stationarity-Aware Representation Learning.pdf`
- RouteTS — `papers/[2026] RouteTS Frequency-Time Routing for Time Series Forecasting.pdf`
- TimeES — `papers/[2026] TimeES Probabilistic and Deterministic Time Series Forecasting via Evolutionary Spectra.pdf`
- FreqLens — `papers/[2026] FreqLens Interpretable Frequency Attribution for Time Series Forecasting.pdf`
- AutoCorr-KAN — `papers/[2026] Autocorrelation Reintroduces Spectral Bias in KANs for Time Series Forecasting.pdf`
- PHAT — `papers/[2026] PHAT Modeling Period Heterogeneity for Multivariate Time Series Forecasting.pdf`
- WaveTuner — `papers/[2025] WaveTuner Comprehensive Wavelet Subband Tuning for Time Series Forecasting.pdf`
- Sonnet — `papers/[2025] Sonnet Spectral Operator Neural Network for Multivariable Time Series Forecasting.pdf`
- CycleNet — `papers/CycleNet Enhancing Time Series Forecasting.pdf`
- WaveForM — `papers/WaveForM Graph enhanced wavelet learning for long.pdf`
- WPMixer — `papers/WPMixer Efficient Multi-Resolution Mixing.pdf`
- HDMixer — `papers/[2024] HDMixer Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting.pdf`

---

**END OF MANUAL — Rev A.** This document supersedes ad-hoc literature summaries. Any change to a TECH-ID, tier, or the Law formula requires a Revision entry in the Record of Revisions.




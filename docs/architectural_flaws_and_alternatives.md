# Architectural Diagnosis & Inductive Bias Analysis (SRP26)
**Date:** 31 October 2026  
**Author:** Aaditya Chaturvedi  
**Project:** Ultra-lightweight MTS Forecasting Sandbox & Spectral Decomposition (SRP26)  
**Supervisor:** Jung Min Choi  

---

## 1. Executive Summary

This document captures the formal scientific diagnosis of why the initial two-stage decoupled architecture (rigid Fourier base + heavy multi-head residual attention) underperformed lightweight baselines (CycleNet and DLinear) by 10–20%, accompanied by concrete architectural alternatives to test.

---

## 2. Diagnosis 1: The Base Model Flaw (Spectral Leakage & Gibbs Phenomenon)

### The Mechanism of Failure:
When using a rigid complex-linear Fourier projection ($\text{rFFT} \to W_{\text{spec}} \to \text{irFFT}$) for the Stage 1 base forecast:
1. **Periodic Boundary Assumption:** Discrete Fourier Transform mathematically treats the finite lookback window $L$ as an infinitely repeating periodic signal.
2. **Spectral Leakage & Gibbs Phenomenon:** In real-world time series, the beginning ($t=0$) and end ($t=L-1$) of the lookback window almost never match. Any non-repeating trend, localized level shift, or boundary discontinuity creates severe high-frequency ripples (Gibbs phenomenon) and spectral leakage.
3. **Artifact Propagation into Residuals:** Because the residual is computed via direct subtraction:
   $$X_{\text{res}} = X - X_{\text{base}}$$
   all artificial boundary oscillations and harmonic ringing from the Fourier stage are dumped directly into the residual sequence $X_{\text{res}}$.
4. **Impact on the Forecaster:** The secondary model is now forced to predict fictitious high-frequency spikes caused by mathematical transform artifacts rather than true physical dynamics.

### Alternatives to Test:
- **CycleNet Recurrent Queue ($C_t$):** Replaces rigid global Fourier basis functions with a local periodic state table updated via exponential moving average or residual gradient, avoiding boundary ringing entirely.
- **DLinear Moving-Average Decomposition:** Uses simple time-domain moving-average pooling ($\text{AvgPool1d}$) with front/back edge-padding, yielding smooth, non-oscillating trend and seasonal components.
- **DCT-II (Discrete Cosine Transform) or Butterworth Low-Pass Filter:** DCT-II enforces even boundary conditions ($x_{-n} = x_{n-1}$), drastically dampening Gibbs boundary ripples compared to DFT.

---

## 3. Diagnosis 2: The Secondary Model Flaw (Why Attention Fails on Residuals)

### The Mechanism of Failure:
When deploying high-capacity Multi-Head Self-Attention (Transformer / PatchTST) on residual sequences:
1. **Mismatched Inductive Bias:**
   - Self-attention's inductive bias is designed to discover global pairwise semantic affinities across long temporal horizons ($Q K^T / \sqrt{d_k}$).
   - However, once the dominant harmonic cycle and low-frequency trend are removed ($X_{\text{res}}$), the remaining signal consists largely of **uncorrelated high-frequency noise, transient shocks, and sensor perturbations**.
2. **Spurious Token Correlations & Overfitting:**
   - In pure residual noise, there are no persistent long-range temporal affinities.
   - The dot-product attention matrix finds spurious coincidental correlations between unrelated noise spikes, assigning high attention weights to random fluctuations.
   - Result: The training loss drops rapidly ($<0.1$), but the validation loss surges after only 1–2 epochs (severe early stopping breakdown).
3. **Parameter & Compute Inefficiency:**
   - Scaling up to Transformer backbones (e.g., 14.7M parameters on `Traffic`) represents a $278\times$ parameter explosion over CycleNet (53k parameters) without providing any representational benefit on unstructured noise.

### Alternatives to Test:
- **Compact 2-Layer MLP with Heavy Regularization:**
  - LayerNorm + Linear + GELU + Dropout ($0.3–0.4$) + Weight Decay ($10^{-3}$).
  - Dense parameter sharing across time prevents memorization of localized noise spikes.
- **1D Depthwise-Separable Convolution (Temporal ConvNet):**
  - Imposes a strict local receptive field bias. It captures localized transient shocks and recovery dynamics without attempting to build spurious global correlations across long time horizons.
  - Linear computational complexity $\mathcal{O}(L \cdot C)$.
- **SORMixer / Inter-Patch Mixer with Channel Gating:**
  - Tokenizes sub-series patches and mixes strictly between immediate neighbor tokens ($i-1, i, i+1$) using dynamic sigmoid gating $g_i$.
  - Adds lightweight cross-channel linear projection to resolve CycleNet's channel blindness without quadratic attention overhead.


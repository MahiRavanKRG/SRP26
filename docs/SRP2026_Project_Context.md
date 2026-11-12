# Comprehensive Context Document: SRP 2026 - Time Series Forecasting in the Frequency Domain

## 1. Project Overview & Core Problem Statement
**Institution:** University of Hildesheim (MSc Data Analytics, Student Research Project 2026)[cite: 20]
**Supervisor:** Jung Min Choi[cite: 20]
**Team:** Syed Natiq Hussain Shah, Abdelhadi Hadj Sahraoui, Philip Gentner, Mohamad Taleb, Aaditya Chaturvedi[cite: 20, 21].

**Problem Statement:** 
Standard forecasting models analyze data purely in the time domain, which makes it difficult to separate genuine repeating seasonal patterns from random, irregular noise[cite: 21]. While recent advanced models introduce frequency-domain transforms to isolate these recurring events, they often apply them as a blanket solution without understanding their limits[cite: 21]. This project aims to systematically evaluate the boundary conditions of frequency-domain representations in multivariate time series forecasting[cite: 21]. The goal is to determine exactly when frequency decomposition succeeds, when it degrades performance (e.g., localized anomalies, regime shifts), and how it can be integrated into lightweight models without creating excessive parameter bloat[cite: 14, 21].

## 2. Theoretical Background & Evaluated Architectures

### A. CycleNet (The Chosen Baseline)
*   **Original Architecture:** Focuses on long-term forecasting by separating stable periodic patterns from changing residuals[cite: 27]. It learns a global "recurrent cycle" matrix, subtracts this cycle from the raw data, and feeds the leftover unpredictable "residual" noise into a highly lightweight backbone (a single Linear layer or shallow MLP)[cite: 27].
*   **Original Flaws:** Cycle length ($W$) was manually hardcoded, it only modeled a single primary period, and it treated all variables as strictly channel-independent, missing cross-channel correlations[cite: 2].

### B. MSGNet (Multi-Scale Graph Network)
*   **Mechanism:** Uses Fast Fourier Transform (FFT) to extract global frequency components and identify dominant periods[cite: 6]. It reshapes 1D sequences into 2D tensors based on these periods and applies adaptive MixHop graph convolutions to learn inter-series correlations at different time scales[cite: 6, 9].
*   **Limitation:** Assumes stationary periodicities and struggles if the frequency characteristics abruptly shift[cite: 11, 21].

### C. SimpleTM
*   **Mechanism:** A lightweight baseline using the Stationary Wavelet Transform (SWT) for tokenization[cite: 7, 10]. SWT avoids downsampling, preserving the exact time location of frequency features[cite: 7]. It uses a "Geometric Product Attention" mechanism inspired by Clifford Algebra to capture both dot-product similarity and cross-channel complementarity via wedge products[cite: 10].

### D. HDMixer
*   **Mechanism:** Rejects rigid time windows[cite: 15]. Uses a Length-Extendable Patcher (LEP) with dynamic boundary shifting and "rubber band" resampling to capture continuous events[cite: 15]. Analyzes data through a Hierarchical Dependency Explorer (HDE) focusing on short-term local patterns, long-term temporal trends, and cross-variable relationships[cite: 15].
*   **Limitation:** Deemed too computationally expensive for the student GPU cluster constraints[cite: 28].

## 3. Phase 1 Experiments: First Interim Presentation (May 2026)
Before the pipeline was automated, the team ran preliminary baseline tests for the first interim presentation[cite: 21]. All initial experiments used a sequence length of 336[cite: 21].

**CycleNet Preliminary Results:**
*   **ETTh1:** $H=96$ (MSE: 0.378, MAE: 0.397) | $H=192$ (MSE: 0.423, MAE: 0.422) | $H=336$ (MSE: 0.460, MAE: 0.442)[cite: 21]
*   **ETTh2:** $H=96$ (MSE: 0.291, MAE: 0.344) | $H=192$ (MSE: 0.365, MAE: 0.394) | $H=336$ (MSE: 0.402, MAE: 0.426)[cite: 21]
*   **Weather:** $H=96$ (MSE: 0.161, MAE: 0.210) | $H=192$ (MSE: 0.208, MAE: 0.251) | $H=336$ (MSE: 0.261, MAE: 0.291)[cite: 21]

**HDMixer Preliminary Results:**
*   **ETTh1:** $H=96$ (MSE: 0.387, MAE: 0.409) | $H=192$ (MSE: 0.423, MAE: 0.427) | $H=336$ (MSE: 0.432, MAE: 0.447)[cite: 17, 21]
*   **ETTh2:** $H=96$ (MSE: 0.279, MAE: 0.345) | $H=192$ (MSE: 0.351, MAE: 0.388) | $H=336$ (MSE: 0.369, MAE: 0.405)[cite: 19, 21]

## 4. Phase 2 & 3: The Automated CycleNet Pipeline (Aug-Sept 2026)
Following supervisor feedback to modify CycleNet and avoid manual parameter tuning, the team successfully automated the architecture's limitations[cite: 28].

1.  **Automated Period Detection (FFT + ACF):** Replaces hardcoded cycles. The model runs FFT to find candidate frequencies, then mathematically confirms them using Autocorrelation Function (ACF) scores to filter out noise[cite: 30, 32].
    *   *Example (Traffic):* FFT candidates [24, 12, 8, 168]. ACF confirmed [168, 24, 12, 8][cite: 30].
    *   *Example (Weather):* FFT candidates [144, 72, 120]. ACF confirmed all three[cite: 39].
2.  **Dynamic Multi-Cycle Learning:** Uses a variance-reduction metric[cite: 30]. If multiple periods significantly reduce variance together, the model dynamically shifts from single-cycle to multi-cycle modeling[cite: 30, 32].
    *   *Traffic:* Selected MULTI-CYCLE using periods [168, 24] (reduction: 63.85%)[cite: 30].
    *   *Electricity & Solar:* Selected SINGLE-CYCLE using periods [24] and [144] respectively[cite: 32, 34].
3.  **Adaptive Channel-Dependence (Residual Path):** The pipeline checks Pearson and cross-correlation across variables[cite: 32, 34]. If variables are highly correlated, it enables a channel-mixing layer on the residual path, solving CycleNet's channel-blindness[cite: 30, 32].
    *   *Solar Dataset:* Average Pearson = 0.914, activating the channel-independent residual constraint (enabling channel mixing)[cite: 32].

## 5. Final Cluster Execution Data (Sept 2026)
The team executed final benchmarks across Linear, MLP, and CNN backbones. All configurations used `seq_len=336`, `batch_size=32`, `patience=5`, and `lr=0.001`[cite: 30, 31, 32, 33, 34, 35, 38, 39]. To ensure statistical stability, the team executed 3-seed averaging (seeds: 2024, 2025, 2026) for Electricity and Solar datasets[cite: 36, 37, 40, 41].

| Dataset | Backbone | Multi-Seed Avg? | MSE (Avg across H) | MAE (Avg across H) | Specific Horizon Breakdowns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Electricity** | MLP | Yes (3 seeds) | **0.1627** | **0.2563** | H=96: MSE 0.1322, H=720: MSE 0.2039[cite: 37] |
| **Electricity** | Linear | Yes (3 seeds) | 0.1692 | 0.2625 | H=96: MSE 0.1406, H=720: MSE 0.2096[cite: 40] |
| **Electricity** | CNN | No (Seed 2024) | 0.2398 | 0.3255 | 13,723 params. H=336 dropped to 0.1963[cite: 38] |
| **Solar** | MLP | Yes (3 seeds) | **0.2125** | **0.2636** | H=96: MSE 0.1868, H=720: MSE 0.2225[cite: 36] |
| **Solar** | Linear | Yes (3 seeds) | 0.2526 | 0.3003 | H=96: MSE 0.2209, H=720: MSE 0.2718[cite: 41] |
| **Traffic** | Linear | No | **0.4214** | **0.2901** | H=96: MSE 0.4023, H=720: MSE 0.4453[cite: 31] |
| **Traffic** | MLP | No | 0.4309 | 0.2873 | H=96: MSE 0.3976, H=720: MSE 0.4824[cite: 30] |
| **Weather** | Linear | No (Seed 2024) | 0.2434 | 0.2795 | Params scaled from 35k (H=96) to 245k (H=720)[cite: 39] |

## 6. Critical Missing Requirements (The Final Deliverables)
While the training loops and basic metrics (MSE/MAE) are complete, the supervisor explicitly required advanced mathematical evaluations that have not yet been executed[cite: 14]. 

**Pending Tasks:**
1.  **Band-wise Error Decomposition:** Write a script to split the final prediction errors into low-frequency (long-term trends) and high-frequency (sudden spikes/noise) components[cite: 14]. This proves whether the model is predicting actual noise or just coasting on easy trends[cite: 14].
2.  **Spectral Coherence & Power Spectral Density (PSD):** Measure how well the frequency waves of the model's predictions align with the frequency waves of the ground truth data[cite: 14]. Does the model over-smooth the signal[cite: 14]?
3.  **Structured Failure Mode Analysis:** Build a sliding-window error analysis tool. Locate specific time windows where the model's predictions crash (e.g., during sudden anomalies or regime shifts) and explain what the frequency data was doing at that exact moment to cause the failure[cite: 14, 21].
4.  **Final Academic Report:** Synthesize the automated pipeline architecture, the multi-seed results, and the advanced frequency evaluations into a cohesive LaTeX paper[cite: 14, 21].
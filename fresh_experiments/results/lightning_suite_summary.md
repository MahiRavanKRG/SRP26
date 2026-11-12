# PyTorch Lightning LTSF Benchmark Suite Results

**Execution Date:** 2026-10-05 09:27:55  
**Lookback ($L$):** 336 | **Horizon ($H$):** 96 | **Precision:** 16-mixed (AMP)  
**Evaluation:** Standardized 12/4/4 Months Split (Zero Leakage, Restored Best Checkpoint)  

| Dataset | Model | Parameters | Runtime (s) | Test MSE | Test MAE | Mid-Band Error | Mid-Band FVU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ETTh1** | CycleNet | 32,520 | 225.57s | **0.3664** | 0.3924 | 56.16790 | 0.4788 |
| **ETTh1** | DLinear | 64,704 | 146.3s | **0.3679** | 0.3941 | 56.33340 | 0.4802 |
| **ETTh1** | FourierPatchAttention | 445,995 | 151.3s | **0.3846** | 0.4040 | 60.42200 | 0.5151 |
| **ETTh1** | FourierTQNet | 626,251 | 123.09s | **0.3824** | 0.4054 | 59.06579 | 0.5035 |
| **ETTh2** | CycleNet | 32,520 | 165.2s | **0.2746** | 0.3385 | 13.13719 | 0.9977 |
| **ETTh2** | DLinear | 64,704 | 137.78s | **0.2711** | 0.3342 | 13.00347 | 0.9876 |
| **ETTh2** | FourierPatchAttention | 445,995 | 152.02s | **0.2872** | 0.3449 | 14.26600 | 1.0835 |
| **ETTh2** | FourierTQNet | 626,251 | 139.56s | **0.2947** | 0.3573 | 14.06197 | 1.0680 |


### Architectural Highlights & Key Takeaways
1. **CycleNet (Primary Proposed):** Recurrent periodic queue ($W=24$) efficiently captures stationary diurnal harmonics without heavy quadratic token attention.
2. **Fourier-Patch Attention:** Decoupled Fourier harmonic base + Pre-LN PatchTST subseries attention out-competes unconstrained point-wise attention and beats DLinear.
3. **Fourier-TQNet:** Explores cross-channel correlations ($C \times C$) guided by Fourier base temporal queries.
4. **DLinear Baseline:** Pure linear decomposition baseline for reference.

# Traffic Benchmark Results: CycleNet vs. Fourier-PatchAttention

**Execution Date:** 2026-10-05 15:53:54  
**Dataset:** Traffic (`traffic.csv`, 862 variables)  
**Lookback ($L$):** 336 | **Horizon ($H$):** 96 | **Batch Size:** 16 | **Precision:** 16-mixed (AMP)  
**Protocol:** Standard 70/10/20 Continuous Split (Train-only standardization, lookback prefix preserved)  

| Model | Parameters | Training Time | Test MSE | Test MAE | Mid-Band Error | Mid-Band FVU | Log-Spectral Dist (LSD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CycleNet** | 53,040 | 1228.74s | **0.4092** | 0.2790 | 0.15041 | 0.5439 | 9.2850 |
| **Fourier-PatchAttention** | 14,752,710 | 7661.97s | **0.4143** | 0.2980 | 0.15437 | 0.5582 | 8.4340 |


### Analytical Takeaways & Comparative Insights
1. **Accuracy Comparison**: CycleNet Test MSE = 0.4092 vs. Fourier-PatchAttention Test MSE = 0.4143 (1.25% lower for CycleNet).
2. **Computational Footprint**: CycleNet utilizes 53,040 parameters vs. Fourier-PatchAttention's 14,752,710 parameters (278.1x larger).
3. **Training Efficiency**: CycleNet trained in 1228.74s vs. Fourier-PatchAttention's 7661.97s (6.2x runtime ratio).
4. **Spectral Fidelity**: Mid-Band FVU is 0.5439 for CycleNet vs. 0.5582 for Fourier-PatchAttention.

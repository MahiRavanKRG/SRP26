# Empirical Benchmark: 5-Model Comparative Study on ETTh1

- **Lookback Window ($L$):** 336 steps | **Horizon ($H$):** 96 steps
- **Device:** NVIDIA GeForce RTX 4050 Laptop GPU

| Model                                                                     |   Parameters | Best Epoch   |   Time (s) |   Test MSE |   Test MAE |   Mid-Band MSE |   Mid-Band FVU |
|:--------------------------------------------------------------------------|-------------:|:-------------|-----------:|-----------:|-----------:|---------------:|---------------:|
| DLinear (AAAI 2023)                                                       |       64,718 | 17/24        |       45.2 |     0.3772 |     0.4018 |        57.14   |         0.4871 |
| CycleNet (NeurIPS 2024, W=24)                                             |       32,534 | 5/12         |       21.5 |     0.3678 |     0.3971 |        55.8536 |         0.4761 |
| TwoStageFourierAttentionNet (Fourier Base + Residual Attention)           |      238,898 | 1/8          |       56   |     0.4109 |     0.4273 |        62.3595 |         0.5316 |
| TwoStageFourierPatchAttentionNet (Fourier Base + PatchTST Residuals)      |      445,995 | 1/8          |       54.6 |     0.3822 |     0.4052 |        58.285  |         0.4969 |
| TwoStageFourierTQNet (Fourier Base + TQ Cross-Channel Residual Attention) |      626,251 | 1/8          |       40   |     0.3946 |     0.4127 |        59.8905 |         0.5106 |

## Model Taxonomy & Insights:
1. **DLinear**: Simple moving average baseline with RevIN.
2. **CycleNet**: NeurIPS 2024 periodic state queue indexed by t mod W.
3. **Point-wise Attention**: Generic MHA on raw points (overfits severely).
4. **Patch-based Attention**: PatchTST subseries patch tokenization (curbs overfitting).
5. **FTQ-Net (Frankenstein Hybrid)**: Fourier harmonic base extraction + TQ cross-channel residual attention.

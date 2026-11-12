# Empirical Comparison: DLinear vs. CycleNet vs. Point-wise MHA vs. Patch-based Attention

- **Dataset:** ETTh1 (7 channels) | **Lookback $L$:** 336 | **Horizon $H$:** 96
- **Device:** NVIDIA GeForce RTX 4050 Laptop GPU

| Model                                                                |   Parameters | Best Epoch   |   Time (s) |   Test MSE |   Test MAE |   Mid-Band MSE |   Mid-Band FVU |
|:---------------------------------------------------------------------|-------------:|:-------------|-----------:|-----------:|-----------:|---------------:|---------------:|
| DLinear (AAAI 2023)                                                  |       64,718 | 17/24        |       45.2 |     0.3772 |     0.4018 |        57.14   |         0.4871 |
| CycleNet (NeurIPS 2024, W=24)                                        |       32,534 | 5/12         |       21.5 |     0.3678 |     0.3971 |        55.8536 |         0.4761 |
| TwoStageFourierAttentionNet (Fourier Base + Residual Attention)      |      238,898 | 1/8          |       56   |     0.4109 |     0.4273 |        62.3595 |         0.5316 |
| TwoStageFourierPatchAttentionNet (Fourier Base + PatchTST Residuals) |      445,995 | 1/8          |       54.6 |     0.3822 |     0.4052 |        58.285  |         0.4969 |

## Key Findings:
- **DLinear Test MSE:** 0.3772
- **CycleNet Test MSE:** 0.3678
- **Point-wise Attention Test MSE:** 0.4109
- **Patch-based Attention Test MSE:** 0.3822 (Best Epoch: 1)

# Empirical Comparison on ETTh2 (L=336, H=96)

- **Dataset:** ETTh2 (Non-stationary regime shift benchmark)
- **Device:** NVIDIA GeForce RTX 4050 Laptop GPU

| Model                                                                |   Parameters | Best Epoch   |   Time (s) |   Test MSE |   Test MAE |   Mid-Band MSE |   Mid-Band FVU |
|:---------------------------------------------------------------------|-------------:|:-------------|-----------:|-----------:|-----------:|---------------:|---------------:|
| DLinear (AAAI 2023)                                                  |       64,718 | 3/10         |       23.3 |     0.2776 |     0.341  |        13.1822 |         1.0012 |
| CycleNet (NeurIPS 2024, W=24)                                        |       32,534 | 7/14         |       30.2 |     0.2748 |     0.3381 |        12.981  |         0.9859 |
| TwoStageFourierPatchAttentionNet (Fourier Base + PatchTST Residuals) |      445,995 | 1/8          |       47.9 |     0.2829 |     0.3472 |        13.5231 |         1.0271 |
| TwoStageFourierTQNet (Fourier Base + TQ Cross-Channel Attention)     |      626,251 | 6/13         |       57.4 |     0.3219 |     0.3724 |        13.4883 |         1.0244 |


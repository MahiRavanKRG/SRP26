# Fresh 3-Model Empirical Benchmark on ETTh1

- **Dataset:** ETTh1 (Hourly Electricity Transformer Temperature, 7 channels)
- **Lookback Window ($L$):** 336 steps (14 days)
- **Forecasting Horizon ($H$):** 96 steps (4 days)
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU
- **Random Seed:** 2024

## 1. Comparative Results Table

| Model                                                           |   Parameters | Best Epoch   |   Time (s) |   Test MSE |   Test MAE |   Mid-Band MSE |   Mid-Band FVU |
|:----------------------------------------------------------------|-------------:|:-------------|-----------:|-----------:|-----------:|---------------:|---------------:|
| DLinear (AAAI 2023)                                             |       64,718 | 17/24        |       45.2 |     0.3772 |     0.4018 |        57.14   |         0.4871 |
| CycleNet (NeurIPS 2024, W=24)                                   |       32,534 | 5/12         |       21.5 |     0.3678 |     0.3971 |        55.8536 |         0.4761 |
| TwoStageFourierAttentionNet (Fourier Base + Residual Attention) |      238,898 | 1/8          |       56   |     0.4109 |     0.4273 |        62.3595 |         0.5316 |

## 2. Key Observations

1. **DLinear**: Test MSE = **0.3772**, MAE = **0.4018** (converged at epoch 17)
2. **CycleNet**: Test MSE = **0.3678**, MAE = **0.3971** (converged at epoch 5)
3. **TwoStageFourierAttentionNet**: Test MSE = **0.4109**, MAE = **0.4273** (converged at epoch 1)

All models stopped automatically when validation loss ceased to improve, ensuring no overfitted checkpoints were evaluated.

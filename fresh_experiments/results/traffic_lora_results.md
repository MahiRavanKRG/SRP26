# Traffic Benchmark: Zero-Shot Chronos vs. FCMath (LoRA) vs. CycleNet (H=96)

| Model                    |    MSE |    MAE |   Mid_FVU |   LSD_dB | Runtime   | Notes                                                   |
|:-------------------------|-------:|-------:|----------:|---------:|:----------|:--------------------------------------------------------|
| Chronos-Raw (Zero-Shot)  | 0.9816 | 0.4375 |    1.0568 |    16.27 | 350.86s   | Raw foundation model without decomposition              |
| FCMath (Zero-Shot)       | 0.6755 | 0.3613 |    0.9804 |    14.22 | 351.13s   | Fourier base + Zero-Shot Chronos residuals              |
| FCMath (LoRA Fine-Tuned) | 0.9405 | 0.4458 |    1.0705 |    13.17 | 122.1s    | LoRA adapter (r=8) trained on Traffic residuals         |
| CycleNet (W=24)          | 0.5263 | 0.3908 |    0.9246 |     9.74 | 1.86s     | Ultra-lightweight recurrent periodic queue (32k params) |

# Chronos Base vs. FCMath Benchmark: ETTh1.csv (H=96)

| Model                         | Type                         |    MSE |    MAE |   Mid_FVU |   LSD_dB |   Inference_Sec |
|:------------------------------|:-----------------------------|-------:|-------:|----------:|---------:|----------------:|
| Chronos-Raw (chronos-t5-tiny) | Pure Foundation Model        | 0.4662 | 0.4136 |    1.0831 |    16.27 |           18.7  |
| FCMath (Standard)             | Fourier Base + Chronos       | 0.4141 | 0.4225 |    0.9167 |    15.63 |           10.8  |
| FCMath-Tuned (Trend+Fourier)  | Trend-Fourier Base + Chronos | 0.3965 | 0.4259 |    1.012  |    16.47 |           11.17 |
| CycleNet (W=24)               | Ultra-Lightweight Queue      | 0.2769 | 0.3479 |    0.6316 |    12.84 |           28.72 |

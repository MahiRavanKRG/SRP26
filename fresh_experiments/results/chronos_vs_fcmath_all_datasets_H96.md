# Master Benchmark: Chronos Base vs. FCMath vs. CycleNet across Datasets (H=96)

| Dataset         |   Channels | Model             |    MSE |    MAE |   Mid_FVU |   LSD_dB |   Time_s |
|:----------------|-----------:|:------------------|-------:|-------:|----------:|---------:|---------:|
| ETTh1.csv       |          7 | Chronos-Raw       | 0.5208 | 0.4412 |    1.0795 |    17.67 |    20.57 |
| ETTh1.csv       |          7 | FCMath (Standard) | 0.4083 | 0.4239 |    0.9248 |    16.13 |     6.91 |
| ETTh1.csv       |          7 | FCMath-Tuned      | 0.3925 | 0.4303 |    1.0076 |    16.4  |     7.41 |
| ETTh1.csv       |          7 | CycleNet (W=24)   | 0.4705 | 0.4881 |    1.4686 |     9.94 |     0.35 |
| ETTh2.csv       |          7 | Chronos-Raw       | 0.102  | 0.2009 |    1.0709 |    23.78 |     6.49 |
| ETTh2.csv       |          7 | FCMath (Standard) | 0.1514 | 0.2643 |    1.0523 |    21.84 |     7.37 |
| ETTh2.csv       |          7 | FCMath-Tuned      | 0.1455 | 0.2615 |    1.029  |    23.39 |     6.51 |
| ETTh2.csv       |          7 | CycleNet (W=24)   | 0.0903 | 0.2116 |    1.5931 |    21.01 |     0.2  |
| weather.csv     |         21 | Chronos-Raw       | 0.2108 | 0.2788 |    0.9753 |    21.63 |    39.94 |
| weather.csv     |         21 | FCMath (Standard) | 0.4131 | 0.4278 |    1.0771 |    20.18 |     8.66 |
| weather.csv     |         21 | FCMath-Tuned      | 0.3105 | 0.3676 |    1.0211 |    21.21 |     8.64 |
| weather.csv     |         21 | CycleNet (W=24)   | 0.2357 | 0.3271 |    1.6036 |    22.8  |     0.34 |
| electricity.csv |        321 | Chronos-Raw       | 0.2752 | 0.3297 |    0.8703 |    12.64 |   130.68 |
| electricity.csv |        321 | FCMath (Standard) | 0.1885 | 0.2904 |    0.8217 |    13.59 |   130    |
| electricity.csv |        321 | FCMath-Tuned      | 0.2019 | 0.3047 |    0.8071 |    14.73 |   130.86 |
| electricity.csv |        321 | CycleNet (W=24)   | 0.1808 | 0.3002 |    0.9762 |     9.57 |     0.73 |
| traffic.csv     |        862 | Chronos-Raw       | 0.9816 | 0.4375 |    1.0568 |    16.27 |   350.86 |
| traffic.csv     |        862 | FCMath (Standard) | 0.6755 | 0.3613 |    0.9804 |    14.22 |   351.13 |
| traffic.csv     |        862 | FCMath-Tuned      | 0.7299 | 0.4034 |    0.9733 |    15.17 |   354.82 |
| traffic.csv     |        862 | CycleNet (W=24)   | 0.5263 | 0.3908 |    0.9246 |     9.74 |     1.86 |
| ETTm1.csv       |          7 | Chronos-Raw       | 0.6435 | 0.5228 |    1.1737 |    13.76 |     2.98 |
| ETTm1.csv       |          7 | FCMath (Standard) | 1.7572 | 0.835  |    1.4625 |    14.26 |     2.92 |
| ETTm1.csv       |          7 | FCMath-Tuned      | 1.6464 | 0.8219 |    1.451  |    15.07 |     2.98 |
| ETTm1.csv       |          7 | CycleNet (W=24)   | 0.3506 | 0.4188 |    1.8776 |     8.66 |     0.24 |

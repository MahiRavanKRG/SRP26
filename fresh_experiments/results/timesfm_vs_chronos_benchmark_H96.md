# Google TimesFM 2.5 (200M) vs. Amazon Chronos vs. CycleNet (H=96)

| Dataset     | Model                      | Params   |    MSE |    MAE |   Mid_FVU |   LSD_dB |   Runtime_Sec |
|:------------|:---------------------------|:---------|-------:|-------:|----------:|---------:|--------------:|
| ETTh1.csv   | TimesFM-Raw (Zero-Shot)    | 200M     | 0.4601 | 0.4151 |    0.6283 |    17.23 |          1.33 |
| ETTh1.csv   | FTimesFM (Fourier+TimesFM) | 200M     | 0.5318 | 0.4847 |    0.6664 |    16.86 |          0.43 |
| ETTh1.csv   | Chronos-Raw (Zero-Shot)    | 8M       | 1.1036 | 0.5775 |    2.7592 |    11.32 |          2.71 |
| ETTh1.csv   | FCMath (Fourier+Chronos)   | 8M       | 0.7456 | 0.5264 |    1.3435 |    15.36 |          2.29 |
| ETTh1.csv   | CycleNet (W=24)            | 32K      | 0.6159 | 0.5344 |    1.4015 |    10.05 |          0.38 |
| weather.csv | TimesFM-Raw (Zero-Shot)    | 200M     | 0.1895 | 0.1914 |    0.9794 |    18.73 |          1.78 |
| weather.csv | FTimesFM (Fourier+TimesFM) | 200M     | 0.3266 | 0.3437 |    1.0134 |    18.18 |          1.38 |
| weather.csv | Chronos-Raw (Zero-Shot)    | 8M       | 0.5993 | 0.3186 |    8.4595 |    22.78 |          4.85 |
| weather.csv | FCMath (Fourier+Chronos)   | 8M       | 0.6509 | 0.3849 |    3.4215 |    24.45 |          4.02 |
| weather.csv | CycleNet (W=24)            | 32K      | 0.2353 | 0.2762 |    1.6518 |    20.21 |          0.35 |
| traffic.csv | TimesFM-Raw (Zero-Shot)    | 200M     | 0.3511 | 0.226  |    0.4772 |    12.04 |         45.78 |
| traffic.csv | FTimesFM (Fourier+TimesFM) | 200M     | 0.4931 | 0.3523 |    0.5376 |    13.4  |         46.64 |
| traffic.csv | Chronos-Raw (Zero-Shot)    | 8M       | 1.9127 | 0.5641 |    1.833  |    12.69 |        108.53 |
| traffic.csv | FCMath (Fourier+Chronos)   | 8M       | 1.2292 | 0.5328 |    1.2393 |    12.99 |        107.78 |
| traffic.csv | CycleNet (W=24)            | 32K      | 0.6437 | 0.4608 |    0.9661 |     9.93 |          1.6  |

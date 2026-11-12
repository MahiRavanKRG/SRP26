| Dataset     | Model                     | Paradigm                 | Params   |    MSE |    MAE |   Mid_FVU |   LSD_dB |   Runtime_Sec |
|:------------|:--------------------------|:-------------------------|:---------|-------:|-------:|----------:|---------:|--------------:|
| ETTh1.csv   | Fourier Base Only (Top-3) | Pure Math Closed-Form    | 0        | 0.9647 | 0.7545 |    0.8924 |    35.15 |          0.01 |
| ETTh1.csv   | Google TimesFM 2.5        | Raw (Zero-Shot)          | 200M     | 0.4601 | 0.4151 |    0.6283 |    17.23 |          0.73 |
| ETTh1.csv   | Google TimesFM 2.5        | FTimesFM (Fourier+Res)   | 200M     | 0.5091 | 0.4659 |    0.6639 |    18.4  |          0.4  |
| ETTh1.csv   | Chronos-Bolt-Base         | Raw (Zero-Shot)          | 200M     | 0.4687 | 0.4175 |    0.625  |    18.01 |          1    |
| ETTh1.csv   | Chronos-Bolt-Base         | F-Bolt (Fourier+Res)     | 200M     | 0.5169 | 0.476  |    0.6828 |    17.91 |          0.56 |
| ETTh1.csv   | IBM Granite TTM-R1        | Raw (Zero-Shot)          | 805K     | 0.4945 | 0.4575 |    0.7235 |    21.83 |          0.02 |
| ETTh1.csv   | IBM Granite TTM-R1        | F-TTM (Fourier+Res)      | 805K     | 0.5155 | 0.485  |    0.7756 |    20.01 |          0.01 |
| ETTh1.csv   | Amazon Chronos-T5         | Raw (Zero-Shot)          | 8M       | 0.8396 | 0.5672 |    1.1214 |    11.9  |          3.42 |
| ETTh1.csv   | Amazon Chronos-T5         | FCMath (Fourier+Res)     | 8M       | 0.682  | 0.5277 |    0.9303 |    14.58 |          3.39 |
| ETTh1.csv   | CycleNet (W=24)           | Raw (Recurrent)          | 32K      | 0.6215 | 0.5333 |    1.4589 |    10.08 |          0.34 |
| ETTh1.csv   | CycleNet (W=24)           | F-CycleNet (Fourier+Res) | 32K      | 0.576  | 0.516  |    1.1114 |     9.31 |          0.87 |
| weather.csv | Fourier Base Only (Top-3) | Pure Math Closed-Form    | 0        | 0.6754 | 0.613  |    1.0077 |    20.16 |          0.01 |
| weather.csv | Google TimesFM 2.5        | Raw (Zero-Shot)          | 200M     | 0.1895 | 0.1914 |    0.9794 |    18.73 |          1.38 |
| weather.csv | Google TimesFM 2.5        | FTimesFM (Fourier+Res)   | 200M     | 0.2971 | 0.3268 |    0.9982 |    18.54 |          1.18 |
| weather.csv | Chronos-Bolt-Base         | Raw (Zero-Shot)          | 200M     | 0.2114 | 0.2167 |    0.9875 |    20.33 |          1.64 |
| weather.csv | Chronos-Bolt-Base         | F-Bolt (Fourier+Res)     | 200M     | 0.2978 | 0.3212 |    0.9996 |    18.65 |          1.58 |
| weather.csv | IBM Granite TTM-R1        | Raw (Zero-Shot)          | 805K     | 0.1916 | 0.2117 |    0.9766 |    22.22 |          0.01 |
| weather.csv | IBM Granite TTM-R1        | F-TTM (Fourier+Res)      | 805K     | 0.2852 | 0.3186 |    1.0056 |    20.21 |          0.01 |
| weather.csv | Amazon Chronos-T5         | Raw (Zero-Shot)          | 8M       | 0.2943 | 0.2927 |    1.0922 |    18.95 |          8.29 |
| weather.csv | Amazon Chronos-T5         | FCMath (Fourier+Res)     | 8M       | 0.3597 | 0.3459 |    1.6491 |    19.08 |          8.41 |
| weather.csv | CycleNet (W=24)           | Raw (Recurrent)          | 32K      | 0.2318 | 0.2758 |    1.623  |    20.21 |          0.17 |
| weather.csv | CycleNet (W=24)           | F-CycleNet (Fourier+Res) | 32K      | 0.3219 | 0.3515 |    1.4947 |    20.07 |          0.28 |
| traffic.csv | Fourier Base Only (Top-3) | Pure Math Closed-Form    | 0        | 0.7392 | 0.4784 |    0.8388 |    59.14 |          0.01 |
| traffic.csv | Google TimesFM 2.5        | Raw (Zero-Shot)          | 200M     | 0.3511 | 0.226  |    0.4772 |    12.04 |         40.99 |
| traffic.csv | Google TimesFM 2.5        | FTimesFM (Fourier+Res)   | 200M     | 0.4299 | 0.2914 |    0.5327 |    13.44 |         42.55 |
| traffic.csv | Chronos-Bolt-Base         | Raw (Zero-Shot)          | 200M     | 0.3772 | 0.2334 |    0.4917 |    12.65 |         73.96 |
| traffic.csv | Chronos-Bolt-Base         | F-Bolt (Fourier+Res)     | 200M     | 0.4392 | 0.2921 |    0.5247 |    12.86 |         74.76 |
| traffic.csv | IBM Granite TTM-R1        | Raw (Zero-Shot)          | 805K     | 0.6433 | 0.4339 |    0.6756 |    21.75 |          0.23 |
| traffic.csv | IBM Granite TTM-R1        | F-TTM (Fourier+Res)      | 805K     | 0.6036 | 0.4232 |    0.7013 |    20.45 |          0.24 |
| traffic.csv | Amazon Chronos-T5         | Raw (Zero-Shot)          | 8M       | 1.9474 | 0.5692 |    1.9198 |    12.73 |        373.78 |
| traffic.csv | Amazon Chronos-T5         | FCMath (Fourier+Res)     | 8M       | 1.2292 | 0.5328 |    1.2393 |    12.99 |        107.78 |
| traffic.csv | CycleNet (W=24)           | Raw (Recurrent)          | 32K      | 0.6476 | 0.4604 |    0.9913 |     9.8  |          2.12 |
| traffic.csv | CycleNet (W=24)           | F-CycleNet (Fourier+Res) | 32K      | 0.5217 | 0.384  |    0.7476 |     9.51 |          2.39 |
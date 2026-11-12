# Expanded Time-Series Foundation Model Suite Benchmark (H=96)

| Dataset     | Model                    | Architecture                   | Params   |    MSE |    MAE |   Mid_FVU |   LSD_dB |   Runtime_Sec |
|:------------|:-------------------------|:-------------------------------|:---------|-------:|-------:|----------:|---------:|--------------:|
| ETTh1.csv   | Google TimesFM 2.5       | Patched Transformer (Non-AR)   | 200M     | 0.4601 | 0.4151 |    0.6283 |    17.23 |          0.79 |
| ETTh1.csv   | Amazon Chronos-Bolt-Base | Patch Encoder-Decoder (Non-AR) | 200M     | 0.4687 | 0.4175 |    0.625  |    18.01 |          1.09 |
| ETTh1.csv   | IBM Granite TTM-R1       | MLP-Mixer Foundation Model     | 805K     | 0.4945 | 0.4575 |    0.7235 |    21.83 |          0.03 |
| ETTh1.csv   | Amazon Chronos-T5        | Discrete Tokenized T5 (AR)     | 8M       | 1.6665 | 0.6155 |    3.4616 |    12.56 |          3.86 |
| ETTh1.csv   | CycleNet (W=24)          | Recurrent Periodic Queue       | 32K      | 0.6159 | 0.5344 |    1.4015 |    10.05 |          0.34 |
| weather.csv | Google TimesFM 2.5       | Patched Transformer (Non-AR)   | 200M     | 0.1895 | 0.1914 |    0.9794 |    18.73 |          1.43 |
| weather.csv | Amazon Chronos-Bolt-Base | Patch Encoder-Decoder (Non-AR) | 200M     | 0.2114 | 0.2167 |    0.9875 |    20.33 |          1.69 |
| weather.csv | IBM Granite TTM-R1       | MLP-Mixer Foundation Model     | 805K     | 0.1916 | 0.2117 |    0.9766 |    22.22 |          0.02 |
| weather.csv | Amazon Chronos-T5        | Discrete Tokenized T5 (AR)     | 8M       | 0.4627 | 0.3196 |    6.4814 |    22.09 |          9.21 |
| weather.csv | CycleNet (W=24)          | Recurrent Periodic Queue       | 32K      | 0.2353 | 0.2762 |    1.6518 |    20.21 |          0.22 |
| traffic.csv | Google TimesFM 2.5       | Patched Transformer (Non-AR)   | 200M     | 0.3511 | 0.226  |    0.4772 |    12.04 |         40.08 |
| traffic.csv | Amazon Chronos-Bolt-Base | Patch Encoder-Decoder (Non-AR) | 200M     | 0.3772 | 0.2334 |    0.4917 |    12.65 |         68.05 |
| traffic.csv | IBM Granite TTM-R1       | MLP-Mixer Foundation Model     | 805K     | 0.6433 | 0.4339 |    0.6756 |    21.75 |          0.23 |
| traffic.csv | Amazon Chronos-T5        | Discrete Tokenized T5 (AR)     | 8M       | 1.9474 | 0.5692 |    1.9198 |    12.73 |        373.78 |
| traffic.csv | CycleNet (W=24)          | Recurrent Periodic Queue       | 32K      | 0.6437 | 0.4608 |    0.9661 |     9.93 |          1.92 |

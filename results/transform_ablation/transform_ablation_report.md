# Transform-Ablation Summary

- Runs: 482 | Seeds: [2024, 2025, 2026] | Transforms: ['dct', 'dlinear', 'dwt_db4', 'dwt_haar', 'identity', 'rfft', 'stft']


## Dataset: `ETTh1`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.371776 |  0.000545 |
| dct         |   0.372412 |  0.000797 |
| dwt_haar    |   0.375319 |  0.001372 |
| dwt_db4     |   0.375379 |  0.001995 |
| identity    |   0.376926 |  0.002563 |
| dlinear     |   0.379916 |  0.002912 |
| stft        |   0.391586 |  0.001733 |

**Best:** `rfft` — mean MSE 0.371776 ± 0.000545


### H=192 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.405507 |  0.000399 |
| dct         |   0.406209 |  0.000426 |
| dwt_db4     |   0.408851 |  0.001786 |
| identity    |   0.409005 |  0.001574 |
| dwt_haar    |   0.409185 |  0.002462 |
| dlinear     |   0.414465 |  0.007105 |
| stft        |   0.420407 |  0.002937 |

**Best:** `rfft` — mean MSE 0.405507 ± 0.000399


### H=336 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.431043 |  0.001186 |
| dct         |   0.432457 |  0.000132 |
| dwt_db4     |   0.433324 |  0.001316 |
| dwt_haar    |   0.434306 |  0.002057 |
| identity    |   0.436004 |  0.001474 |
| dlinear     |   0.437663 |  0.002476 |
| stft        |   0.440844 |  0.001058 |

**Best:** `rfft` — mean MSE 0.431043 ± 0.001186


### H=720 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.443189 |  0.000812 |
| dct         |   0.443957 |  0.000964 |
| identity    |   0.447377 |  0.002344 |
| dwt_db4     |   0.447492 |  0.000368 |
| dlinear     |   0.447773 |  0.004823 |
| dwt_haar    |   0.448611 |  0.000705 |
| stft        |   0.448801 |  0.000481 |

**Best:** `rfft` — mean MSE 0.443189 ± 0.000812


## Dataset: `ETTh2`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.2732   |  0.000145 |
| dct         |   0.273848 |  0.001014 |
| dwt_haar    |   0.279334 |  0.001665 |
| identity    |   0.279627 |  0.002023 |
| dwt_db4     |   0.280079 |  0.002437 |
| dlinear     |   0.282031 |  0.004001 |
| stft        |   0.287459 |  0.002506 |

**Best:** `rfft` — mean MSE 0.273200 ± 0.000145


### H=192 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.337626 |  0.000312 |
| dct         |   0.337746 |  0.001932 |
| dwt_db4     |   0.340925 |  0.001526 |
| dwt_haar    |   0.341074 |  0.001472 |
| identity    |   0.345492 |  0.004712 |
| stft        |   0.34864  |  0.002543 |
| dlinear     |   0.350551 |  0.00893  |

**Best:** `rfft` — mean MSE 0.337626 ± 0.000312


### H=336 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.360905 |  0.000893 |
| dct         |   0.361414 |  0.00114  |
| dwt_db4     |   0.363848 |  0.001606 |
| dwt_haar    |   0.364326 |  0.001461 |
| stft        |   0.367028 |  0.000977 |
| identity    |   0.369585 |  0.004595 |
| dlinear     |   0.370402 |  0.001004 |

**Best:** `rfft` — mean MSE 0.360905 ± 0.000893


### H=720 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| dct         |   0.390938 |  0.000292 |
| rfft        |   0.391226 |  0.000412 |
| dwt_haar    |   0.393225 |  0.001535 |
| dwt_db4     |   0.394014 |  0.000519 |
| stft        |   0.394523 |  0.000626 |
| identity    |   0.395778 |  0.003281 |
| dlinear     |   0.395862 |  0.001149 |

**Best:** `dct` — mean MSE 0.390938 ± 0.000292


## Dataset: `ETTm1`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.303096 |  0.001192 |
| dct         |   0.303466 |  0.000401 |
| dlinear     |   0.306666 |  0.003193 |
| dwt_haar    |   0.307899 |  0.004041 |
| dwt_db4     |   0.308205 |  0.004124 |
| identity    |   0.310942 |  0.002741 |
| stft        |   0.31953  |  0.002381 |

**Best:** `rfft` — mean MSE 0.303096 ± 0.001192


### H=192 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.339348 |  0.000267 |
| dct         |   0.34004  |  0.000499 |
| dwt_haar    |   0.343197 |  0.000801 |
| dwt_db4     |   0.343537 |  0.00067  |
| identity    |   0.344385 |  0.000528 |
| dlinear     |   0.347987 |  0.004136 |
| stft        |   0.349042 |  0.003012 |

**Best:** `rfft` — mean MSE 0.339348 ± 0.000267


### H=336 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.373101 |  0.000504 |
| dct         |   0.374481 |  0.002362 |
| dwt_haar    |   0.375477 |  0.001606 |
| dwt_db4     |   0.377788 |  0.002374 |
| identity    |   0.382016 |  0.002706 |
| dlinear     |   0.38339  |  0.002072 |
| stft        |   0.384464 |  0.004343 |

**Best:** `rfft` — mean MSE 0.373101 ± 0.000504


### H=720 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.429896 |  0.000661 |
| dct         |   0.430756 |  0.000695 |
| dwt_db4     |   0.433475 |  0.0017   |
| dwt_haar    |   0.433523 |  0.001659 |
| stft        |   0.43618  |  0.001933 |
| identity    |   0.44096  |  0.001227 |
| dlinear     |   0.441054 |  0.002903 |

**Best:** `rfft` — mean MSE 0.429896 ± 0.000661


## Dataset: `ETTm2`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| dct         |   0.165272 |  0.001178 |
| rfft        |   0.16546  |  0.00081  |
| dwt_haar    |   0.165674 |  0.000412 |
| dwt_db4     |   0.165932 |  0.000545 |
| identity    |   0.16718  |  0.003058 |
| dlinear     |   0.167804 |  0.001478 |
| stft        |   0.17335  |  0.002733 |

**Best:** `dct` — mean MSE 0.165272 ± 0.001178


### H=192 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.219629 |  0.000475 |
| dct         |   0.220817 |  0.000499 |
| dwt_haar    |   0.220987 |  0.001219 |
| dwt_db4     |   0.221471 |  0.000894 |
| identity    |   0.222603 |  0.000235 |
| dlinear     |   0.22345  |  0.004114 |
| stft        |   0.225219 |  0.000957 |

**Best:** `rfft` — mean MSE 0.219629 ± 0.000475


### H=336 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.273818 |  0.000349 |
| dct         |   0.274508 |  0.000642 |
| dwt_haar    |   0.27546  |  0.000804 |
| dwt_db4     |   0.275601 |  0.00096  |
| identity    |   0.275989 |  0.001428 |
| stft        |   0.279702 |  0.001219 |
| dlinear     |   0.28181  |  0.002608 |

**Best:** `rfft` — mean MSE 0.273818 ± 0.000349


### H=720 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.368164 |  0.000424 |
| dct         |   0.368501 |  0.000474 |
| dwt_db4     |   0.36953  |  0.002119 |
| dwt_haar    |   0.369625 |  0.001751 |
| identity    |   0.370351 |  0.003209 |
| stft        |   0.371538 |  0.000582 |
| dlinear     |   0.372803 |  0.000912 |

**Best:** `rfft` — mean MSE 0.368164 ± 0.000424


## Dataset: `electricity`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| dwt_haar    |   0.14051  |  0.000244 |
| dlinear     |   0.140654 |  0.000504 |
| rfft        |   0.140683 |  0.000147 |
| dct         |   0.140713 |  0.000308 |
| dwt_db4     |   0.141019 |  0.00043  |
| identity    |   0.141023 |  0.000529 |
| stft        |   0.145975 |  1.5e-05  |

**Best:** `dwt_haar` — mean MSE 0.140510 ± 0.000244


### H=192 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| dlinear     |   0.154697 |  0.000391 |
| rfft        |   0.154873 |  0.000148 |
| dct         |   0.154904 |  0.000427 |
| dwt_haar    |   0.155189 |  0.000603 |
| identity    |   0.155315 |  0.000749 |
| dwt_db4     |   0.155769 |  3.9e-05  |
| stft        |   0.159525 |  0.000636 |

**Best:** `dlinear` — mean MSE 0.154697 ± 0.000391


## Dataset: `traffic`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| identity    |   0.409765 |  6.6e-05  |
| dlinear     |   0.40982  |  9.9e-05  |
| dct         |   0.410429 |  0.001225 |
| rfft        |   0.410653 |  0.000292 |
| dwt_haar    |   0.410825 |  0.001794 |
| dwt_db4     |   0.412968 |  0.002072 |
| stft        |   0.419395 |  5.9e-05  |

**Best:** `identity` — mean MSE 0.409765 ± 0.000066


## Dataset: `weather`


### H=96 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.173943 |  9.4e-05  |
| identity    |   0.174457 |  0.001903 |
| dwt_haar    |   0.174739 |  0.002815 |
| dwt_db4     |   0.175262 |  0.003126 |
| dct         |   0.175743 |  0.000694 |
| dlinear     |   0.176361 |  0.002647 |
| stft        |   0.181975 |  0.000134 |

**Best:** `rfft` — mean MSE 0.173943 ± 0.000094


### H=192 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.217645 |  0.00087  |
| dct         |   0.217704 |  0.000124 |
| dwt_haar    |   0.21805  |  0.001954 |
| identity    |   0.218706 |  0.001957 |
| dwt_db4     |   0.218984 |  0.002124 |
| dlinear     |   0.220609 |  0.00246  |
| stft        |   0.223091 |  0.000723 |

**Best:** `rfft` — mean MSE 0.217645 ± 0.000870


### H=336 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.265532 |  0.001065 |
| identity    |   0.265717 |  0.001225 |
| dwt_db4     |   0.266175 |  0.00105  |
| dct         |   0.266213 |  0.000282 |
| dwt_haar    |   0.266451 |  0.001109 |
| dlinear     |   0.267035 |  0.000693 |
| stft        |   0.270282 |  0.001394 |

**Best:** `rfft` — mean MSE 0.265532 ± 0.001065


### H=720 (ranked by mean MSE)

| transform   |   mean_mse |   std_mse |
|:------------|-----------:|----------:|
| rfft        |   0.333324 |  4.9e-05  |
| dct         |   0.333572 |  0.001304 |
| dwt_haar    |   0.333722 |  0.000186 |
| dwt_db4     |   0.333882 |  0.000314 |
| identity    |   0.334301 |  0.000503 |
| dlinear     |   0.334961 |  0.000482 |
| stft        |   0.335305 |  0.0001   |

**Best:** `rfft` — mean MSE 0.333324 ± 0.000049


## Paired Wilcoxon signed-rank vs identity (per-window MSE)

p < 0.05 = transform differs from the time-domain (identity) baseline.

| dataset     |   pred_len | transform   |   n_windows |   p_value | significant   |
|:------------|-----------:|:------------|------------:|----------:|:--------------|
| ETTh1       |         96 | dct         |        8355 |  0        | True          |
| ETTh1       |         96 | dlinear     |        8355 |  0        | True          |
| ETTh1       |         96 | dwt_db4     |        8355 |  0        | True          |
| ETTh1       |         96 | dwt_haar    |        8355 |  0        | True          |
| ETTh1       |         96 | rfft        |        8355 |  0        | True          |
| ETTh1       |         96 | stft        |        8355 |  0        | True          |
| ETTh1       |        192 | dct         |        8067 |  0        | True          |
| ETTh1       |        192 | dlinear     |        8067 |  0        | True          |
| ETTh1       |        192 | dwt_db4     |        8067 |  0.05854  | False         |
| ETTh1       |        192 | dwt_haar    |        8067 |  0.851048 | False         |
| ETTh1       |        192 | rfft        |        8067 |  0        | True          |
| ETTh1       |        192 | stft        |        8067 |  0        | True          |
| ETTh1       |        336 | dct         |        7635 |  0        | True          |
| ETTh1       |        336 | dlinear     |        7635 |  0        | True          |
| ETTh1       |        336 | dwt_db4     |        7635 |  0        | True          |
| ETTh1       |        336 | dwt_haar    |        7635 |  0        | True          |
| ETTh1       |        336 | rfft        |        7635 |  0        | True          |
| ETTh1       |        336 | stft        |        7635 |  0        | True          |
| ETTh1       |        720 | dct         |        6483 |  0        | True          |
| ETTh1       |        720 | dlinear     |        6483 |  0.004368 | True          |
| ETTh1       |        720 | dwt_db4     |        6483 |  0.3767   | False         |
| ETTh1       |        720 | dwt_haar    |        6483 |  0.025194 | True          |
| ETTh1       |        720 | rfft        |        6483 |  0        | True          |
| ETTh1       |        720 | stft        |        6483 |  0.000298 | True          |
| ETTh2       |         96 | dct         |        8355 |  0        | True          |
| ETTh2       |         96 | dlinear     |        8355 |  0        | True          |
| ETTh2       |         96 | dwt_db4     |        8355 |  0.593417 | False         |
| ETTh2       |         96 | dwt_haar    |        8355 |  0.420207 | False         |
| ETTh2       |         96 | rfft        |        8355 |  0        | True          |
| ETTh2       |         96 | stft        |        8355 |  0        | True          |
| ETTh2       |        192 | dct         |        8067 |  0        | True          |
| ETTh2       |        192 | dlinear     |        8067 |  0        | True          |
| ETTh2       |        192 | dwt_db4     |        8067 |  0        | True          |
| ETTh2       |        192 | dwt_haar    |        8067 |  0        | True          |
| ETTh2       |        192 | rfft        |        8067 |  0        | True          |
| ETTh2       |        192 | stft        |        8067 |  0.143653 | False         |
| ETTh2       |        336 | dct         |        7635 |  0        | True          |
| ETTh2       |        336 | dlinear     |        7635 |  0.000887 | True          |
| ETTh2       |        336 | dwt_db4     |        7635 |  0        | True          |
| ETTh2       |        336 | dwt_haar    |        7635 |  0        | True          |
| ETTh2       |        336 | rfft        |        7635 |  0        | True          |
| ETTh2       |        336 | stft        |        7635 |  0        | True          |
| ETTh2       |        720 | dct         |        6483 |  0        | True          |
| ETTh2       |        720 | dlinear     |        6483 |  0.970554 | False         |
| ETTh2       |        720 | dwt_db4     |        6483 |  3e-06    | True          |
| ETTh2       |        720 | dwt_haar    |        6483 |  0        | True          |
| ETTh2       |        720 | rfft        |        6483 |  0        | True          |
| ETTh2       |        720 | stft        |        6483 |  0.149201 | False         |
| ETTm1       |         96 | dct         |       34275 |  0        | True          |
| ETTm1       |         96 | dlinear     |       34275 |  0        | True          |
| ETTm1       |         96 | dwt_db4     |       34275 |  0        | True          |
| ETTm1       |         96 | dwt_haar    |       34275 |  0        | True          |
| ETTm1       |         96 | rfft        |       34275 |  0        | True          |
| ETTm1       |         96 | stft        |       34275 |  0        | True          |
| ETTm1       |        192 | dct         |       33987 |  0        | True          |
| ETTm1       |        192 | dlinear     |       33987 |  0        | True          |
| ETTm1       |        192 | dwt_db4     |       33987 |  0        | True          |
| ETTm1       |        192 | dwt_haar    |       33987 |  0        | True          |
| ETTm1       |        192 | rfft        |       33987 |  0        | True          |
| ETTm1       |        192 | stft        |       33987 |  0        | True          |
| ETTm1       |        336 | dct         |       33555 |  0        | True          |
| ETTm1       |        336 | dlinear     |       33555 |  0        | True          |
| ETTm1       |        336 | dwt_db4     |       33555 |  0        | True          |
| ETTm1       |        336 | dwt_haar    |       33555 |  0        | True          |
| ETTm1       |        336 | rfft        |       33555 |  0        | True          |
| ETTm1       |        336 | stft        |       33555 |  0.00015  | True          |
| ETTm1       |        720 | dct         |       32403 |  0        | True          |
| ETTm1       |        720 | dlinear     |       32403 |  0.087043 | False         |
| ETTm1       |        720 | dwt_db4     |       32403 |  0        | True          |
| ETTm1       |        720 | dwt_haar    |       32403 |  0        | True          |
| ETTm1       |        720 | rfft        |       32403 |  0        | True          |
| ETTm1       |        720 | stft        |       32403 |  0        | True          |
| ETTm2       |         96 | dct         |       34275 |  0        | True          |
| ETTm2       |         96 | dlinear     |       34275 |  0        | True          |
| ETTm2       |         96 | dwt_db4     |       34275 |  0        | True          |
| ETTm2       |         96 | dwt_haar    |       34275 |  0        | True          |
| ETTm2       |         96 | rfft        |       34275 |  0        | True          |
| ETTm2       |         96 | stft        |       34275 |  0        | True          |
| ETTm2       |        192 | dct         |       33987 |  0        | True          |
| ETTm2       |        192 | dlinear     |       33987 |  0        | True          |
| ETTm2       |        192 | dwt_db4     |       33987 |  0        | True          |
| ETTm2       |        192 | dwt_haar    |       33987 |  0        | True          |
| ETTm2       |        192 | rfft        |       33987 |  0        | True          |
| ETTm2       |        192 | stft        |       33987 |  0.009189 | True          |
| ETTm2       |        336 | dct         |       33555 |  0        | True          |
| ETTm2       |        336 | dlinear     |       33555 |  0        | True          |
| ETTm2       |        336 | dwt_db4     |       33555 |  0        | True          |
| ETTm2       |        336 | dwt_haar    |       33555 |  0        | True          |
| ETTm2       |        336 | rfft        |       33555 |  0        | True          |
| ETTm2       |        336 | stft        |       33555 |  0        | True          |
| ETTm2       |        720 | dct         |       32403 |  0        | True          |
| ETTm2       |        720 | dlinear     |       32403 |  0        | True          |
| ETTm2       |        720 | dwt_db4     |       32403 |  0.001357 | True          |
| ETTm2       |        720 | dwt_haar    |       32403 |  1e-06    | True          |
| ETTm2       |        720 | rfft        |       32403 |  0        | True          |
| ETTm2       |        720 | stft        |       32403 |  0.029002 | True          |
| electricity |         96 | dct         |       15498 |  0        | True          |
| electricity |         96 | dlinear     |       15498 |  0        | True          |
| electricity |         96 | dwt_db4     |       15498 |  0.001822 | True          |
| electricity |         96 | dwt_haar    |       15498 |  0        | True          |
| electricity |         96 | rfft        |       15498 |  0        | True          |
| electricity |         96 | stft        |       15498 |  0        | True          |
| electricity |        192 | dct         |       15209 |  0        | True          |
| electricity |        192 | dlinear     |       15210 |  0        | True          |
| electricity |        192 | dwt_db4     |       15210 |  0        | True          |
| electricity |        192 | dwt_haar    |       15210 |  0        | True          |
| electricity |        192 | rfft        |       15210 |  0        | True          |
| electricity |        192 | stft        |       15210 |  0        | True          |
| traffic     |         96 | dct         |       10242 |  0        | True          |
| traffic     |         96 | dlinear     |       10242 |  0.010208 | True          |
| traffic     |         96 | dwt_db4     |       10242 |  0        | True          |
| traffic     |         96 | dwt_haar    |       10240 |  0        | True          |
| traffic     |         96 | rfft        |       10242 |  0        | True          |
| traffic     |         96 | stft        |       10242 |  0        | True          |
| weather     |         96 | dct         |       31335 |  0        | True          |
| weather     |         96 | dlinear     |       31335 |  0        | True          |
| weather     |         96 | dwt_db4     |       31335 |  0        | True          |
| weather     |         96 | dwt_haar    |       31335 |  0        | True          |
| weather     |         96 | rfft        |       31335 |  0        | True          |
| weather     |         96 | stft        |       31335 |  0        | True          |
| weather     |        192 | dct         |       31046 |  0        | True          |
| weather     |        192 | dlinear     |       31047 |  0        | True          |
| weather     |        192 | dwt_db4     |       31047 |  0.103169 | False         |
| weather     |        192 | dwt_haar    |       31047 |  0        | True          |
| weather     |        192 | rfft        |       31047 |  0        | True          |
| weather     |        192 | stft        |       31047 |  0        | True          |
| weather     |        336 | dct         |       30615 |  0        | True          |
| weather     |        336 | dlinear     |       30615 |  0        | True          |
| weather     |        336 | dwt_db4     |       30615 |  0        | True          |
| weather     |        336 | dwt_haar    |       30615 |  0        | True          |
| weather     |        336 | rfft        |       30615 |  0.145033 | False         |
| weather     |        336 | stft        |       30615 |  0        | True          |
| weather     |        720 | dct         |       29463 |  0        | True          |
| weather     |        720 | dlinear     |       29463 |  0        | True          |
| weather     |        720 | dwt_db4     |       29463 |  0        | True          |
| weather     |        720 | dwt_haar    |       29463 |  0        | True          |
| weather     |        720 | rfft        |       29463 |  0        | True          |
| weather     |        720 | stft        |       29463 |  0        | True          |

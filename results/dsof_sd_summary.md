# DSOF-SD Empirical Benchmark Summary (TimesFM 2.5 Teacher)

Protocol: `L=336`, `H=96`, canonical ETT 12/4/4 split, train-only `StandardScaler`, 3 seeds `[2024, 2025, 2026]`.
Offline Phase 0 teacher caches were generated via `TimesFM 2.5 (200M)` (`per_core_batch_size=256`) and stored locally outside Git.

| Dataset | Student | Gate | Teacher MSE | Warm MSE (mean±std) | DSOF-SD MSE (mean±std) | Warm MAE | DSOF-SD MAE | Warm FVU Mid | DSOF-SD FVU Mid | Warm FVU High | DSOF-SD FVU High | Active Gates (mean) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| ETTh1.csv | linear (32,534p) | hard | 0.3950 | 0.3788±0.0000 | 0.3761±0.0023 | 0.4001 | 0.3974 | 0.7244 | 0.6954 | 1.1210 | 1.0851 | L=0.00, M=1.00, H=1.00 |
| ETTh1.csv | sor (76,668p) | hard | 0.3950 | 0.3931±0.0014 | 0.3949±0.0032 | 0.4163 | 0.4194 | 0.7563 | 0.7568 | 1.0639 | 1.2342 | L=0.33, M=1.00, H=1.00 |
| ETTh1.csv | sor (76,668p) | soft | 0.3950 | 0.3931±0.0014 | 0.3997±0.0079 | 0.4163 | 0.4227 | 0.7563 | 0.7569 | 1.0639 | 1.1783 | L=0.49, M=0.59, H=0.68 |
| ETTh2.csv | linear (32,601p) | hard | 0.3137 | 0.2893±0.0040 | 0.2879±0.0057 | 0.3513 | 0.3485 | 1.0406 | 0.9930 | 1.0585 | 1.0518 | L=1.00, M=1.00, H=1.00 |
| ETTh2.csv | sor (76,735p) | hard | 0.3137 | 0.2986±0.0041 | 0.2942±0.0036 | 0.3563 | 0.3525 | 0.9935 | 0.9455 | 0.9921 | 0.9517 | L=1.00, M=1.00, H=1.00 |
| ETTh2.csv | sor (76,735p) | soft | 0.3137 | 0.2986±0.0041 | 0.2955±0.0063 | 0.3563 | 0.3538 | 0.9935 | 0.9542 | 0.9921 | 0.9431 | L=0.55, M=0.63, H=0.58 |

## Key Empirical Takeaways
1. **ETTh2 (non-stationary / drifted regime)**: DSOF-SD beats both the warm student and zero-shot TimesFM 2.5 teacher (0.3137) on mean MSE and MAE across all tested configurations (`sor hard`: 0.2986 -> 0.2942), cutting Mid FVU (0.9935 -> 0.9455) and High FVU (0.9921 -> 0.9517).
2. **ETTh1 (stationary periodic regime)**: Warm `CycleNet-Pro (sor)` already matches/beats TimesFM 2.5 on MSE (0.3931±0.0014 vs 0.3950) and total LSD (1.5709 vs 1.9543). On `linear` students, DSOF-SD still lowers Mid FVU (0.7244 -> 0.6954), High FVU (1.1210 -> 1.0851), and mean MSE (0.3788 -> 0.3761).
3. **FVU vs. LSD Spectral Asymmetry**: Lower teacher FVU in Mid/High bands reflects MMSE amplitude shrinkage of unpredictable high-frequency noise (reducing FVU while increasing raw-target LSD), acting as a spectral variance regularizer rather than direct high-frequency detail synthesis.

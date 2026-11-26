# SRP26 v2 benchmark — final summary (2026-11-21 ~14:10 WEST, automated check)

All phases complete. Raw result files are committed under `results/v2/` (bigmama/ and local/).

| Phase | Count | Status |
|---|---|---|
| LR selection (Bigmama) | 2328 result files (2278 base + edge extensions); 512/512 cells selected (`configs/v2/selected_lr.json`, ab10334) | complete |
| Final, Bigmama | 1696/1696 (every job in `configs/v2/jobs_final_bigmama.jsonl` has a result) | complete |
| Final, laptop (ETT only, scope change 2026-11-20) | 848/848 ETT (866 files total incl. 18 pre-scope-change non-ETT) | complete |
| Foundation (incl. 16 TTM) | 240/240 | complete |
| DSOF-SD | 216/216 | complete |

Last arrays: 394451 (big datasets; tasks 2/3 show FAILED = the two Traffic CUDA OOMs, both rerun OK as 399575 / 399890) and sweep 4 = 400029 (4/4 COMPLETED).

## Reports (generated with `uv run --no-sync python -m srp26.bench.report ...`)
- `results/v2/bigmama/summary.md` — MSE/MAE mean ± std over seeds 2024-2026, all datasets x horizons x 19 models.
- `results/v2/local/summary.md` — laptop set.
- `results/v2/compare_local_vs_bigmama.csv` — 312 ETT cells compared; **311/312 agree within seed spread**, median |rel diff| 0.000%. Only outlier: ETTm2 H96 `cyclenet_pro_mlp_ci` (local 0.1752 vs bigmama 0.1705, +2.8%, pooled std 0.0012).
- `results/v2/bigmama/foundation_vs_trained_bigmama.csv` — 1936 rows, foundation models vs trained models.

## Things worth a look (observations only, nothing changed)
- Several `*_ci` variants give bit-identical numbers to their non-CI twins on ETTh1 (all cyclenet_pro_* pairs), and `cyclenet_fixed` equals `cyclenet_pro_linear_ci` on most cells. May be by design (same config resolves to the same model), but worth confirming before writing the paper tables.
- Traffic H720 `cyclenet_pro_linear`: 0.6986 ± 0.2013 — one unstable seed.
- PatchTST is best on Solar and Weather; `cyclenet_pro_mlp_ci` / `cyclenet_fixed` are competitive or best on Electricity/Traffic; the TF (rfft/dct) linear baselines are strongest on ETTh2.

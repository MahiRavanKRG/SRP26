# Project Invariants & AI Operational Rules

This workspace enforces strict alignment between the empirical code, the supervisor mandate, and the research manuscript.

## 1. Ground Truth Architecture
- **Proposed Model:** Automated CycleNet (CycleNet-Pro) with automated FFT+ACF period discovery and recurrent periodic state (`models/cyclenet/`).
- **Baseline Models:** DLinear (Zeng et al., 2023), PatchTST, HDMixer. DLinear is strictly a baseline, NEVER the paper's proposed model.
- **Diagnostic Method:** Parseval Subband Error Decomposition ($FVU_{\text{low}}, FVU_{\text{mid}}, FVU_{\text{high}}$) and Power Spectral Density analysis (`metrics.py`).

## 2. Experimental Protocols
- Lookback window: $L = 336$. Horizons: $H \in \{96, 192, 336, 720\}$.
- ETT dataset splits: 12/4/4 months. Non-ETT splits: 70/10/20.
- Prefix preservation: Always include lookback $L$ history in test/val partitions.
- Preprocessing: `StandardScaler` fitted strictly on train data. RevIN enabled.
- Evaluation seeds: `[2024, 2025, 2026]`.
- All runs must execute through `uv run ...`.

## 3. Data Integrity & Reporting
- No metric hallucination: paper tables must directly trace to CSV/log files in `results/`.
- All manuscript edits (`paper/main.tex`) must compile cleanly with `pdflatex` (0 errors).
- Daily progress and major decisions must be logged in `.ai_history/session_log_YYYYMMDD.md`.

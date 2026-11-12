# AGENTS.md — AI Development Governance & Ground-Truth Rules
> **Project:** Ultra-lightweight MTS Forecasting Sandbox & Spectral Decomposition (SRP26)  
> **Institution:** University of Hildesheim | Supervisor: Jung Min Choi  
> **Researcher:** Aaditya Chaturvedi  
> **Scope:** Mandatory guidelines and ground truths for all AI agents, pair programmers, and assistants working on this repository.

---

## 1. Prime Directive: Ground-Truth Reality & Anti-Drift

1. **Codebase & Logs are Ground Truth**:
   - The code in `models/`, `experiment_cyclenet.py`, and empirical metric files in `results/` represent the actual state of the research.
   - AI assistants must **never hallucinate architectures, replace actual models with simpler toys, or alter the research narrative** across sessions.
2. **Verification Gate Before Modifying Paper or Code**:
   - Before editing `paper/main.tex` or running experiments, check active implementations and recent session logs in [`.ai_history/`](file:///d:/uni/SRP26/.ai_history/).
   - Never assume past prompts or verbal summaries are accurate without cross-referencing files in the workspace.

---

## 2. Model Taxonomy: CycleNet vs. DLinear vs. DSOF

The project previously suffered from AI confusion where the paper shifted focus to DLinear while the code and supervisor mandate was CycleNet. The following taxonomy is **strictly binding**:

| Entity | Role in Research | Description & Location |
| :--- | :--- | :--- |
| **CycleNet / CycleNet-Pro** | **Primary Proposed Model** | Automated recurrent cycle model with FFT+ACF period discovery, multi-cycle queues, and channel-mixing (`models/cyclenet/`, `experiment_cyclenet.py`). |
| **DLinear** | **Baseline 1 (Time-domain)** | Naive moving-average decomposition baseline (Zeng et al., AAAI 2023). DLinear is **strictly a baseline**, NEVER the proposed model. |
| **PatchTST / HDMixer** | **Baselines 2 & 3 (SOTA)** | High-capacity Transformer and Mixer baselines for competitive comparison. |
| **DSOF / Parseval Diagnostics** | **Diagnostic Framework** | Dual-Domain Spectral Operator & Parseval subband decomposition framework used to diagnose *all* models (`metrics.py`, `diagnostics.py`). |

### Invariant Rules for Model Reference:
- In `paper/main.tex`: The proposed method is **Automated CycleNet** (or CycleNet with spectral/periodic analysis). DLinear is evaluated to expose where moving averages fail (e.g., long-horizon breakdown on `ETTh2`).
- Diagrams in `paper/figures/`: Must depict the actual **CycleNet** architecture (periodic queue, residual connection, channel projection, and frequency evaluation), NOT plain DLinear.

---

## 3. Core Research Question (Supervisor Mandate)

The paper and experiments must directly serve the research question formulated with supervisor **Jung Min Choi**:
> **"When does frequency-domain modeling actually help vs. fail in lightweight MTS forecasting?"**

Key objectives required for this study:
1. **Automated Period Discovery**: No hardcoded periodicities; automated extraction via FFT + Autocorrelation (ACF).
2. **Boundary Conditions**: Identify exactly where recurrent periodic modeling excels (stationary signals: Traffic, Electricity, ETTh1) and where it reaches boundary limits (non-stationary drift / regime shifts: Weather, ETTm2).
3. **Rigorous Spectral Diagnostics**: Measure Fraction of Variance Unexplained ($FVU$) across physical subbands (Low, Mid, High) using Parseval's theorem, Power Spectral Density (PSD), and sliding-window failure localization.

---

## 4. Scientific Standardization Protocol (Binding Invariants)

All experiments and reported numbers must adhere strictly to [`docs/experiment_standardization_protocol.md`](file:///d:/uni/SRP26/docs/experiment_standardization_protocol.md):

1. **Lookback Window**: Standardized to $L = 336$ for all benchmark datasets.
2. **Prediction Horizons**: $H \in \{96, 192, 336, 720\}$.
3. **Chronological Splits**:
   - **ETT Datasets (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`)**: Canonical 12/4/4 months split (Autoformer/PatchTST convention).
   - **Other Datasets (`Weather`, `Traffic`, `Electricity`, `Solar`)**: 70% Train / 10% Validation / 20% Test.
4. **Lookback Prefix Overlap**:
   - Validation and test slices MUST include the preceding $L$ historical steps to avoid test boundary loss and truncation artifacts.
5. **Preprocessing & Normalization**:
   - `StandardScaler` must be fitted **strictly on the training split**. Never leak validation or test statistics into scalers.
   - Utilize Reversible Instance Normalization (RevIN) to handle non-stationary distribution shifts.
6. **Multi-Seed Stability**:
   - Report mean and standard deviation over 3 fixed random seeds: `[2024, 2025, 2026]`.
7. **Zero-Tolerance Hallucination on Metrics**:
   - Every reported number in paper tables for our models must originate from a verified CSV/JSON file in `results/`.
   - Never fabricate, smooth, or interpolate baseline numbers. Always cite the exact source table (e.g., HDMixer AAAI-24, PatchTST ICLR-23).

---

## 5. Environment & Execution Standards

- **Virtual Environment & Dependencies**:
  - Always use `uv` for dependency management: `uv run python <script>.py`.
  - Do not use raw system python or install packages without `uv`.
- **LaTeX Compilation**:
  - Whenever modifying `paper/main.tex` or any files in `paper/`, verify compilation with `pdflatex` or `latexmk`. The document must compile with **0 errors**.
- **Session History Logging**:
  - Maintain session history logs in `.ai_history/session_log_YYYYMMDD.md`.
  - Log completed steps, major decisions, benchmark results, and open items at the conclusion of every work session.

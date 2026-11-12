# Expert Panel Audit: SRP26 Mid-Term Presentation
**Reviewer Role:** Expert on expert review panel (Thesis Defence Simulation)
**Sources audited:** `presentation.tex`, `Table.md`, `cyclenet_pro_revised.py`, `tune_sor.py`, `paper/main.tex`, `architecture.tikz`, `WhatsApp Chat with STUDENT RESEARCH PROJECT.txt`

---

> [!CAUTION]
> This audit is adversarial. Every finding below is a question a panel member **will** ask. Items marked 🔴 are blocking/damaging. Items marked 🟡 are exploitable weaknesses. Items marked 🟢 are genuine strengths to lean on.

---

## PART I — CRITICAL FACTUAL ERRORS & UNVERIFIABLE CLAIMS (🔴 Blocking)

### 1. The DLinear MSE = 0.6050 on ETTh2 H=720 — Where does this number come from?

- **Claim (Slide 13, paper abstract):** "DLinear MSE = **0.6050**" on ETTh2 at H=720; CycleNet-Pro achieves 0.4231 (30.1% reduction).
- **Problem:** `Table.md` does **not contain a DLinear column**. The table only has: Linear (ours), MLP (ours), CNN (ours), SOR (ours), CycleNet/L, CycleNet/M, PatchTST/42, HDMixer. There is no "DLinear" row anywhere in the empirical data.
- **Where does 0.6050 come from?** It is cited in the paper abstract and the presentation as if it were measured. If it is taken from the original Zeng et al. (AAAI 2023) paper, it must be explicitly cited as **published baseline figure** with $L=336$ or $L=96$ clearly stated (the original DLinear paper used $L=336$). If it was measured by the team, the result log must exist in `results/`.
- **Panel question:** *"Please show me the log file containing DLinear MSE = 0.6050 on ETTh2 at H=720."*
- **Action required:** Either add a verified DLinear row to Table.md from published sources (with citation), or remove the specific number and replace with a qualitative reference to published literature.

---

### 2. FVU Numbers Are Unverifiable — No Code Path Produces Them in This Run

- **Claims (Slides 13, 14, 15):**
  - ETTh1: `FVU_mid ∈ [0.674, 0.720]` for CycleNet vs `[0.808, 0.862]` for DLinear (31.8% reduction).
  - Weather H=720: `FVU_mid = 1.0044`
  - ETTm2 H=336: `FVU_mid = 1.0517`
  - ETTm2 H=336: `FVU_high = 1.2428`
  - ETTh1 H=720: `FVU_high = 1.0727`
  - DLinear: `FVU_high = 1.188 - 1.223`
- **Problem:** The `cyclenet_pro_revised.py` code does **not implement any Parseval/FVU diagnostic**. The `evaluate_metrics()` function computes only raw MSE and MAE. There is no `metrics.py` or `diagnostics.py` referenced in the code or visible in the workspace `results/` folder. These numbers are core to the paper's research contribution but have no verifiable provenance.
- **Panel question:** *"Can you show us the script and output log that produced FVU_mid = 1.0517 for ETTm2?"*
- **Action required:** This is the most critical hole. The diagnostic framework code must be shown, or these claims must be marked as "theoretical projections pending implementation" in the slides.

---

### 3. Multi-Seed Reporting: Protocol Says 3 Seeds, Code Runs 1

- **Claim (Slide 11, `AGENTS.md`, `main.tex`):** "Multi-seed stability: Report mean ± std over 3 fixed seeds [2024, 2025, 2026]."
- **Reality in code (`cyclenet_pro_revised.py`, line 78):** `SEEDS = [2024]  #, 2025]`
- The seed list is explicitly commented out. The HPO tuner (`tune_sor.py`) uses `TUNE_SEED = 42`, a fourth seed not in the protocol.
- **Panel question:** *"Your protocol mandates three seeds. Your code runs one. The numbers in Table 1 — are they averages over 3 seeds or single-seed runs?"*
- **Action required:** Add a footnote to the presentation table clearly stating "all results are single-seed (seed=2024) unless otherwise noted; multi-seed averaging is a planned next step." Remove the multi-seed claim from the protocol slide or downgrade it to a *target* protocol.

---

### 4. Electricity MLP Claim: MSE 0.1627 (abstract) vs. 0.1695 (Table.md)

- **Claim (paper abstract):** "MLP non-linearities achieve substantial accuracy gains... Electricity MSE=**0.1627**"
- **Reality in `Table.md` (Electricity MLP avg):** `0.1695 / 0.2684`
- The best Electricity result in the table for MLP is **AVG = 0.1695**, not 0.1627. The 0.1627 number does not appear in Table.md.
- A discrepancy of ~4% between the abstract claim and the table is small but verifiable instantly by a panel member with a spreadsheet.
- **Action required:** Verify which number is correct. The abstract must match the table exactly.

---

### 5. 1D-CNN Parameter Claim: 97.7% vs. "5,559 vs 241k on ETT" (Slide 8 & 16)

- **Slide 8:** "Slashes parameters by **97.7%** (5.5k vs. 241k on ETT)"
- **Slide 16 table:** ETTh1 CNN = **5,559 params** vs. Linear = **241,920 params**. That is a reduction of 97.7% — consistent.
- **BUT** the slide 16 table shows MLP params = **433,664** on Solar, while Linear is also shown as **241,920** — both use the same ETTh1 architecture (same C, L, H). On Traffic (C=862), the parameter count changes dramatically. The slides present a single canonical number without specifying the dataset it applies to.
- **Panel question:** *"Your 97.7% parameter reduction — on which dataset? With 862 channels in Traffic, the CNN parameter count is the same kernel-shared architecture, so the absolute number changes, but the percentage is dataset-agnostic since CNN doesn't scale with C. Please clarify."*
- This is actually defensible but needs a clear footnote.

---

## PART II — ARCHITECTURAL INCONSISTENCIES (🟡 Exploitable Weaknesses)

### 6. The Architecture Diagram Still Shows a Missing Link: De-Cycling → Backbone

- **Architecture TikZ (line 56):** The flow arrow from RevIN to the backbone is labeled `X_res`, implying that RevIN output goes directly to the backbone **after** de-cycling.
- **But the flow in code** (`CycleNetPro.forward()`, lines 849-851) is: `x_res = x` (RevIN-normalized input), then `x_res = x_res - q(...)` for each cycle queue. The de-cycling happens **after RevIN**, not in a separate branch.
- **The diagram shows Period Discovery as a separate parallel branch to RevIN**, but in the code the period discovery happens **once before training**, not at inference time on each batch. The diagram implies it's a per-inference step.
- **Panel question:** *"Your diagram shows Period Discovery happening at inference time as part of the forward pass. But your code runs FFT+ACF once on training data, then freezes the discovered periods as integer constants in the model constructor. Which is correct?"*

---

### 7. SCPM FiLM Conditioning: The Slide Description is Slightly Off

- **Slide 8 says:** `Z_cond = Z ⊙ (1 + γ) + β`
- **Actual code** (`SORMixer.forward()`, line 706): `h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)`
- The formula matches. ✅
- **BUT:** The spectral conditioner reads from `xc` (the **de-cycled residual input** to the SCPM backbone), not from the original time series. The slide implies it conditions on the full signal spectrum. This is a nuance worth pre-empting.
- **Panel question:** *"Your FiLM conditioner reads the DCT of the residual after de-cycling. Why the residual and not the original signal? Doesn't the residual, by construction, have reduced spectral energy at the cycle frequencies, making the conditioning less informative?"*
- This is a genuine architectural design choice that should be defended, not glossed over.

---

### 8. Inter-Patch Mixer: "Neighbor-Aware" Claim vs. Implementation

- **Slide 8:** "Neighbor-aware gated MLP across local patch order"
- **Code (`_InterPatchMixer`, lines 545-553):** The left/right neighbor concatenation and local MLP followed by sigmoid gate is exactly this. ✅
- **Potential weakness:** The gate produces `g * candidate + (1-g) * h`. This is a convex interpolation, not a standard residual. The presentation describes it as a "gated MLP" — technically accurate, but a panel member might ask why this specific parameterization over a standard residual connection or a standard GLU.

---

### 9. "Adaptive Channel Mixer" in Architecture — No Description in Slides

- **Architecture diagram (TikZ line 44):** Shows "Adaptive Channel Mixer (MLP Gate)" as a dedicated step **after** re-composition.
- **Presentation text:** Slide 5 mentions "Residual Channel Mixing: Selectively mixes inter-channel dependencies only when correlated" but this is never shown in the backbone slides or the formula-level explanation.
- **Code:** The `ChannelMixer` is a simple 2-layer MLP with residual, enabled only when `use_channel_dependence=True`.
- **Panel question:** *"Your architecture diagram shows an 'Adaptive Channel Mixer' after re-composition. But you never describe its implementation or show the criterion used to decide whether it's activated. What is the threshold for enabling it?"*
- The answer is in the code (Pearson correlation > 0.5 + fraction threshold), but this is not in any slide.

---

## PART III — BENCHMARK INTEGRITY CONCERNS (🟡 Exploitable Weaknesses)

### 10. The Lookback Window Inconsistency is Acknowledged but Burying the Lede

- **Table.md footnote (line 1):** "Lookback lengths differ: CycleNet rows use L=96; PatchTST/42, HDMixer, and our four backbones use L=336."
- **Slide 11 claim:** "Standardized Window: Lookback L=336"
- The presentation's benchmarking protocol slide says L=336 for **all** experiments but then the table footnote clarifies that the published CycleNet/L and CycleNet/M baselines use L=96.
- **This is not a small issue.** L=336 vs L=96 is not apples-to-apples. Our models have access to 3.5× more historical data. Comparing our Linear MSE of 0.4050 to published CycleNet/L MSE of 0.3780 on ETTh1 H=192 — and showing the published model **wins** — is honest. But the table title says "ours" wins ETTh1 H=192 for Linear. Check this: our Linear (0.4050) vs CycleNet/L (0.4260) and CycleNet/M (0.4360) — wait, CycleNet/L is **0.3780** at H=96, not H=192. At H=192: CycleNet/L=0.4260, ours=0.4050. **Our Linear wins H=192.** But HDMixer wins every ETTh1 row avg.
- **Panel question:** *"You compare models at different lookback lengths. Can you name a single row in this table where your model beats HDMixer or PatchTST while also using L=96?"*

---

### 11. Solar Dataset: Our Best Models Win, But PatchTST and HDMixer Have No Results

- **Table.md (Solar rows):** PatchTST and HDMixer columns are `–` (dashes).
- **Slide 12:** Shows SOR achieving best Solar H=96 MSE (0.1866), followed by "–" for baselines.
- **Panel question:** *"You claim SOR achieves best-in-class performance on Solar at H=96. But you're comparing against no published baseline — PatchTST and HDMixer don't report Solar results. This means you're essentially best among your own ablations. How do you interpret this finding?"*
- The honest answer (Solar-Energy is a photovoltaic dataset; HDMixer and PatchTST don't benchmark it) is fine, but needs to be stated explicitly.

---

### 12. Traffic CNN H=720: MSE = 0.6355 — Is This a Win or a Loss?

- **Slide 12 table:** Traffic CNN H=720 = **0.6355** — this is the **worst result** in the entire Traffic section (PatchTST achieves 0.4383, HDMixer 0.441, our Linear 0.4739). The slide lists it without comment other than "Traffic CNN 720 Resolved."
- **Slide 12 footnote says:** "Traffic CNN 720 Resolved: Newly ingested run yields 0.6355, completing the 4-horizon suite."
- **Panel question:** *"Your CNN backbone achieves MSE=0.6355 on Traffic at H=720, which is 45% worse than PatchTST and 34% worse than your own Linear backbone. Why is this presented without explanation? Does this represent a fundamental failure of the CNN on high-channel datasets?"*
- This needs a narrative: CNN is designed for edge devices where parameters matter more than accuracy; on high-channel datasets (Traffic has 862 channels), the CNN's kernel-shared architecture **intentionally** sacrifices accuracy for parameter efficiency.

---

## PART IV — PRESENTATIONAL & NARRATIVE GAPS (🟡 Weaknesses)

### 13. Slide 19 (Thank You) Only Shows One Author's Email

- The title slide now has 4 authors but the closing slide (line 613-616) only shows:
  - `Aaditya Chaturvedi` with `chaturvedia@uni-hildesheim.de`
- This contradicts the multi-author presentation. The closing slide should either list all emails or show a group contact.

---

### 14. "Empirical Boundary Discovery: Proven where periodic modeling succeeds (ETTh1, ETTh2)"

- **Slide 18 achievement bullet (line 586):** "Empirical Boundary Discovery: Proven where periodic modeling succeeds (**ETTh1, ETTh2**)..."
- **But Slide 13 shows the DLinear failure on ETTh2 at H=720**, implying ETTh2 is a case where CycleNet **rescues** moving-average baselines — not simply a "periodic modeling success" case.
- **Slide 14 says:** "Non-stationary series... Weather, ETTm2" are the failures.
- **The story is internally inconsistent:** ETTh2 is used simultaneously as (a) an example of our model succeeding over DLinear and (b) a dataset with "moderate non-stationarity" in slide 14. The narrative needs to be cleaner: ETTh2 is a **partially non-stationary** dataset where DLinear collapses but CycleNet-Pro still works, distinguishing it from Weather/ETTm2 where even our model approaches FVU=1.0.

---

### 15. Presentation Describes "10-Dimensional Random Search" — Tune_sor.py is Actually 10 Parameters

- **Slide 10:** "10-dimensional random search tuner"
- **`tune_sor.py` SEARCH_SPACE** has: seq_len, patch_len, stride_ratio, sor_d_model, mixer_layers, n_bands, spectral_hidden, sor_dropout, lr, weight_decay = **10 dimensions**. ✅ This is correct.
- BUT the search is described as "random search" while the presentation later suggests Optuna/200-trial sweeps in the session history. The slide says "random search" — is it Optuna (Bayesian/TPE) or pure random? These are different things. If it's Optuna TPE, calling it "random search" undersells the methodology. If it's pure random, the 200-trial budget is wasteful compared to Bayesian optimization.

---

## PART V — GENUINE STRENGTHS TO DEFEND (🟢)

| Strength | Why it's defensible |
|---|---|
| Automated period discovery (FFT+ACF) | The code implementation is clean, correct, and the 12% spectral leakage merging threshold is principled (lines 127-131) |
| FiLM zero-initialization | Zero-initializing γ,β ensures training starts from identity — this is a well-known trick from FiLM/BERT and provably stable. Defended in the code (line 582-583) |
| Honest boundary reporting | Claiming FVU≈1.0 on Weather/ETTm2 is intellectually honest and pre-empts the obvious attack of "your model doesn't always win" |
| Channel independence decision | The Pearson + lag-xcorr dual criterion is more principled than hardcoded CI choices in PatchTST |
| CNN parameter efficiency | 97.7% reduction is real and measurable from the code architecture |
| No data leakage in scaler | `Scaler.fit()` called on `train_raw` only, then transform applied globally — verified in code (line 960-961) |

---

## PRIORITY ACTION LIST

| Priority | Fix |
|---|---|
| 🔴 URGENT | Either produce or cite DLinear MSE=0.6050 from a verified source |
| 🔴 URGENT | Either produce FVU diagnostic logs or clearly label FVU numbers as "computed via separate diagnostic script" |
| 🔴 HIGH | Correct Electricity MLP abstract claim (0.1627 → 0.1695 or fix the table) |
| 🔴 HIGH | Add footnote: all Table 1 results are single-seed (seed=2024); multi-seed averaging is in progress |
| 🟡 MEDIUM | Add footnote clarifying Traffic CNN H=720 weakness is intentional edge-device design trade-off |
| 🟡 MEDIUM | Clarify ETTh2 narrative: partially non-stationary (DLinear collapses, CycleNet holds) vs. Weather (both degrade) |
| 🟡 MEDIUM | Slide 19 closing: add all 4 authors or use group contact |
| 🟡 MEDIUM | Architecture diagram: clarify that Period Discovery runs once at training time, not at inference per batch |
| 🟡 LOW | Clarify tuner is "random search" vs. Optuna TPE if applicable |

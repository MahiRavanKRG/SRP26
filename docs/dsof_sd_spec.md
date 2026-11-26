# DSOF-SD: DSOF-Guided Spectral Distillation

## 1. Falsifiable Hypothesis

> A tiny hybrid time + frequency student (`CycleNet-Pro` / `SORMixer`, ~50–200K parameters) can inherit a frozen foundation teacher's (`TimesFM 2.5`) **useful** low/mid-band periodic structure via **band-gated log-amplitude distillation**, while refusing to inherit its high-frequency blur — with both effects measured identically by the Parseval-exact DSOF diagnostic.

- **Success criterion:** On bands where the calibrated gate $g_b > 0$ (teacher beats warm-start student), distilled student $\text{LSD}_b$ and $\text{FVU}_b$ drop relative to the warm-start student **without** regression in total validation/test MSE.
- **Refutation / negative finding:** If calibration gives $g_b = 0$ across bands (teacher has nothing to give) or distillation regresses total MSE, the framework surfaces that failure transparently per band.

---

## 2. Unified Loss & Diagnostic Formulation

Using one-sided rFFT ($M = \lfloor H/2 \rfloor + 1$ frequency bins) partitioned into normalized frequency bands:
- $B_{\text{low}} = [0, 0.1)$
- $B_{\text{mid}} = [0.1, 0.4)$
- $B_{\text{high}} = [0.4, 0.5]$

Both training (`DSOFSpectralLoss`) and evaluation (`dsof_band_metrics_torch`) share the exact same single-window rFFT log-amplitude operator $L_{\text{amp}}(\hat{y}, y)_b = \frac{1}{|B_b|}\sum_{k \in B_b}\left(\log(A_{\hat{y}}(k) + \varepsilon) - \log(A_y(k) + \varepsilon)\right)^2 = \text{LSD}_b(\hat{y}, y)^2$:

1. **Phase 0 (Offline Teacher Cache):** Cache $\hat{y}_T = \text{Teacher}(x)$ once across training/calibration windows.
2. **Phase 1 (Calibration Gate on Train/Val Only):**
   - Warm-start student $S_\theta$ for $E_{\text{warm}}$ epochs on $\mathcal{L}_{\text{time}} = \text{MSE}(\hat{y}_S, y)$.
   - Compute Parseval band FVU $\text{FVU}_b^T$ and $\text{FVU}_b^S$ on the calibration split (never test!).
   - Freeze band trust gates $g_b$:
     - Hard: $g_b = \mathbb{I}[\text{FVU}_b^T < \text{FVU}_b^S]$
     - Soft: $g_b = \sigma(\alpha \cdot (\text{FVU}_b^S - \text{FVU}_b^T))$
3. **Phase 2 (Annealed Band-Gated Distillation):**
   $$\mathcal{L} = \text{MSE}(\hat{y}_S, y) + \lambda_{\text{spec}}(t) \sum_{b} g_b \, \text{LSD}_b(\hat{y}_S, \hat{y}_T)^2 + \lambda_{\text{phase}} \sum_{b} g_b \, \text{mean}_{k \in B_b}(1 - \cos(\phi_S(k) - \phi_T(k)))$$
   with $\lambda_{\text{spec}}(t)$ annealed linearly $0 \to \lambda_{\max}$ and $\lambda_{\text{phase}} = 0$ by default.
4. **Phase 3 (Closed-Loop Evaluation):** Compare `{teacher, warm_student, distilled_student}` on per-band Parseval MSE, FVU, unified rFFT LSD, and global MSE / MAE / LSD.

---

## 3. Verified Design Grounding Table

| Design Choice | Actual Justification / Scope | Literature / Corpus Basis | Novelty Attribution |
| :--- | :--- | :--- | :--- |
| Amplitude target sparsity | Frequency representations concentrate dominant periodic modes | FEDformer (2022) | Baseline spectral prior |
| Log-amplitude objective = $\text{LSD}^2$ | Unifies loss and per-window rFFT LSD metric exactly | Internal (`srp26/spectral.py`) | Author unification |
| Band-gated transfer $g_b = \mathbb{I}[\text{FVU}_b^T < \text{FVU}_b^S]$ | Empirical teacher blur vs. student spectral fidelity; selective frequency weighting | Empirical LSD findings + TIFO (frequency-selective weighting) | **Core author novelty** |
| Amplitude vs. phase decoupling ($\lambda_{\text{phase}}=0$ default) | Phase wraps discontinuously under small time shifts | FIRE, DecoVAE | Signal-processing + corpus-supported |
| Warm-start + $\lambda_{\text{spec}}$ annealing | Prevents spectral mismatch from destabilizing time-domain values early | Standard distillation practice (Hinton et al., 2015) | Standard engineering |
| Hybrid time + spectral student (`CycleNetPro` / `SORMixer`) | Multi-period recurrent queue + DCT band gating + local patch MLP | `models/cyclenet/cyclenet_pro_Full.py` (`SORMixer`) | Workspace architecture |
| Frozen teacher cache | Single forward pass over train split | Standard offline distillation | Standard engineering |


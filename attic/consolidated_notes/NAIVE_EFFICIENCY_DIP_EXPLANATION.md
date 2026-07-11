# Explaining the naive measurement-efficiency dip — and whether Method 3 overclaims completeness

**Status:** closed (explanatory), 2026-06-18. Quantitative, validated. Authors: analysis with Claude.
**TL;DR:** The high-T / dip-window depression in the naive `measured/(measured+extrapolated)`
efficiency curve is **dominantly a p-value goodness-of-fit (GOF) artifact** that wrongly rejects
*good, well-behaved* traps, **plus** a smaller residual of genuine high-T fading + faintness.
Swapping the per-temperature GOF cut from `p_value>0.05` to `reduced_chi2<X` raises the observed
dip from **0.414 → 0.642**, landing on the **hybrid (0.676)** prediction. Under the definition
"if a trap is good and well-behaved, would we have caught it," **Method 3 is not meaningfully
overclaiming**; the naive dip is simply a poor estimator of good-trap completeness because it is
penalized by both the analysis artifact and genuine fading. The fix is a ~7%-census, FPR-neutral
re-selection — diagnostically important, not operationally mandatory.

---

## The question
`plot_completeness_overlay` (Method 3) predicts high completeness at the high-T / long-τ₁₃₅ arm,
but the naive efficiency curve **dips** there (the "double hump"). Are we **overclaiming
completeness**? And why does the dip happen?

## Two different metrics (the key to avoiding a false paradox)
- **Naive efficiency / the dip = per-(trap, temperature).** For each trap and each temperature T,
  the point is "measured" if its intensity fit at T passed `GoodIntensityFit`, else "extrapolated".
  `eff(τ_e(T)) = measured/total`, binned by the SRH emission time τ_e(T). The p-value GOF cut acts
  **directly** on this.
- **Trap census = per-trap.** Whether a trap is *characterized* (≥4 good temps + a good Arrhenius
  energy fit). Robust to recovering one more of a trap's temperatures.

These respond very differently to the same cut change. Concretely: of 17,658 high-T intensity
points recovered by the cut swap (X=10), **11,036 (62%) sit on traps that were already
characterized** — they lift the per-point dip while adding **zero** traps to the census. So a large
dip effect coexists with a near-zero census effect; no contradiction.

## What drives the dip (chain of evidence)
1. **Split** (`tools/high_t_failure_split.py`): of dip-window high-T failures, ~26% are genuine
   undetectable fading; ~74% are recoverable/analysis-limited.
2. **GOF-cut FPR scan** (`tools/gof_cut_fpr_scan.py`): ~50% of dip-window failures are
   "p-value-binding casualties" — they pass amplitude/δχ²/τ pre-cuts and fail **only** the
   `p_value>0.05` GOF cut (the dof-bias against well-sampled traps; cf. `srh-gof-overrejects`).
   Decoy false positives fit *better* (reduced_chi2≈0.82) than real casualties (≈3.0), so reduced_chi2
   does not separate trap from non-trap — the **multi-temperature Arrhenius + orientation gate** does.
3. **Visual check** (`figures/gof_visual_check.png`): recovered casualties at reduced_chi2≈3–5 are
   clean dipole bumps; decoys are weak smooth trends whose τ(T) runs **anti-Arrhenius**
   (e.g. τ rising 2.48→5.07 s from 197→203 K) — physically impossible for a trap.

## Quantitative result — the dip rises under the swapped cut
`tools/naive_dip_swap_check.py` (baseline reproduces the reference observed dip **0.414 exactly**;
same 3798 characterized traps held fixed; only the per-point `measured` mask swapped):

| Dip (min over τ_e ∈ 3e-3–3e-2 s) | value |
|---|---|
| Observed, `p_value>0.05` (current) | **0.414** |
| Observed, `reduced_chi2<3` | 0.534 |
| Observed, `reduced_chi2<5` | 0.596 |
| Observed, `reduced_chi2<10` | **0.642** |
| Hybrid (Method 3 × empirical GOF survival) | 0.676 |
| Pure Method 3 (unconditional/conditional) | 0.943 |

See `figures/naive_dip_swap_check.png`. Fixing the GOF cut closes the gap from the observed dip to
the **hybrid** almost entirely (0.414 → 0.642 ≈ 0.676), i.e. the hybrid's GOF-survival scaling was
correctly capturing the p-value penalty. A **residual** gap to pure Method 3 (0.943) remains.

## Interpretation — are we overclaiming?
The dip (0.414) decomposes into:
- **~0.23 recoverable** — the `p_value` GOF cut rejecting *good, well-behaved, well-sampled* traps
  (a dof-bias artifact). Validated as real recoveries: gained traps lie on clean Arrhenius lines
  (`figures/char_delta_validate.png`).
- **~0.30 residual** to pure Method 3 — genuine high-T fading + faintness/amplitude effects, which
  the GOF swap does **not** touch.

Under the deliverable's definition — *"if a trap is good and well-behaved, would we have caught it?"*:
- The p-value losses were dropping genuinely good traps → they should **not** count against
  completeness → the naive dip **understates** good-trap completeness by ~0.23.
- The residual is mostly **genuine fading** — traps that are *not* good/well-behaved at that high T
  (they faded) → also should not count against good-trap completeness.
- Therefore **Method 3's high prediction is approximately the idealized good-trap detectability and
  is not a meaningful overclaim for that definition.** The naive measured-efficiency dip is simply
  the wrong estimator to read as completeness — it is depressed by an analysis artifact *and* by
  including genuinely-faded points in the denominator.

**The double hump is explained:** p-value GOF over-rejection of good well-sampled traps (dominant,
recoverable) + genuine high-T fading/faintness (residual, real).

## Cost/benefit of actually applying the fix (not required)
`tools/characterization_delta.py` (baseline replication **exact**: 4767/4767 vs stored GoodEnergyFit,
count 3798): the trap **census** delta is non-monotonic and peaks at a **tight** cut —
**X=3 → +275** characterized traps (+7.2%); X=5 → +209; X=10 → **−27** (net loss). Loose cuts admit
noisy high-reduced_chi2 temps that push a trap's Arrhenius energy-fit reduced_chi2 over 10 and
**break** previously-good traps (502 lost at X=10). The energy fit is a false-positive backstop
(decoys never characterize — FPR flat ~35–39 across all X), **not** a launderer of bad fits.
So if one ever operationalizes this, the per-temperature GOF should be **tight (X≈3)**, independent
of the energy-fit's own `reduced_chi2<10` (`srh_reduced_chi2_max`, a different stage).

## Caveats / not done
- The 0.943 / 0.676 references are from the existing Stage-11 minimal summary
  (`11_naive_efficiency_closure_summary.json`); the 0.414 was reproduced exactly here.
- The swap check holds the **trap set fixed** (per-point effect); the census churn (~7%) is a
  separate, second-order axis (quantified above).
- "Lightweight" = the observed dip recomputed under the swap; a full Stage-11 re-run with the
  swapped cut feeding the conditional/hybrid predictions was **not** done (not needed for the
  explanation).
- High-τ (>1 s) spikes in the figure are sparse-bin noise, irrelevant to the dip.

## Artifacts
- Tools: `tools/high_t_failure_split.py`, `tools/gof_cut_fpr_scan.py`,
  `tools/characterization_delta.py`, `tools/char_delta_validate.py`, `tools/naive_dip_swap_check.py`,
  `tools/gof_visual_check.py`.
- Figures: `figures/naive_dip_swap_check.png` (headline), `figures/gof_cut_fpr_scan.png`,
  `figures/gof_visual_check.png`, `figures/char_delta_validate.png`.
- Caches: `trap_completeness_method3/cache/gof_cut_fpr_scan_summary.json`,
  `trap_completeness_method3/cache/high_t_failure_split_summary.json`.
- Live-pipeline cut source of truth: **`dipole_new.py`** (energy GOF `reduced_chi2<10` + orientation
  gate), not `dipole.py`.

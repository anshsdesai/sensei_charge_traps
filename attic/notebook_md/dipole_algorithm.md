# Dipole finding, fitting, and grading: legacy vs. minimal (the production algorithm)

> **In plain terms.** This file is the *how-to* for turning sensor images into a trap catalog, in
> three stages: **find** the dipoles (dark-pixel-on-bright-pixel pairs), **fit** each one to measure how
> fast its trap releases charge, and **grade** which fits are trustworthy enough to keep. It compares the
> original method ("legacy") with the corrected one ("minimal") we use now. The single most important
> fix was adding a **pedestal** — a constant offset the old fit model had no way to represent, which was
> quietly wrecking the high-temperature measurements. If you read one file about how the measurement
> works, read this. The *story of why* this method was chosen (and what was tried and rejected) is in
> [[signed_refit]]. New words are in the [glossary](glossary.md).

**Thread status:** current. The headline analysis runs `--pipeline minimal --detection calibrated`
(code in `dipole_new.py`; legacy `dipole.py` retained as a comparison / systematic). This file is
the definitive description of *what the algorithm does*. For *how it was chosen* — the multi-month
signed-refit investigation, the adversarial reviews, and the pieces that were tried and rejected —
see [[signed_refit]]. For where the resulting catalog feeds, see [[completeness_efficiency]].

Source consolidated here (now in `attic/`): `DIPOLE_ALGORITHM_LEGACY_VS_MINIMAL.md` (2026-06-22).

---

## How the two pipelines relate

The two versions live side by side and never overwrite each other's outputs. One switch in
`run_charge_traps.py` selects which runs:

- `--pipeline legacy` (default) → `dipole.py` with its original settings.
- `--pipeline minimal` → `dipole_new.py` with **all** new options on at once: sturdier noise
  estimate + relaxed symmetry when finding, physical error bars when building curves, and the
  pedestal-aware signed fit with absolute error bars when fitting.

In the code each new behavior still defaults to the old behavior unless minimal turns it on (so the
legacy path is byte-identical). A second switch matters for the "is the signal real" decision:

- `--detection calibrated` → **what the final analysis uses.** It builds a per-temperature
  detection threshold measured from the data itself (§3), instead of a fixed guessed value. The
  headline result is `--pipeline minimal --detection calibrated`.

`dipole_new.py` is the synthesis recommended in [[signed_refit]]: it keeps the physically required
corrections (signed intensity, constant pedestal, profile fit in τ, empirical null calibration,
orientation consistency, simple SRH) and deliberately drops the overbuilt machinery of the full
signed-refit runbook (regional covariance, fitted overdispersion, robust outlier rejection,
intrinsic scatter, acquisition-family exclusion).

---

## 1. Finding the traps

**What we look for.** A charge trap grabs an electron as charge is clocked past it, holds it
briefly, then drops it one row later — a **dipole**: two stacked pixels, one darker than its
surroundings and the one right below it brighter by about the same amount.

**Shared recipe (both versions):**
1. Estimate the noise level σ (typical pixel-to-pixel wiggle).
2. Flatten the background by subtracting each row's median.
3. Set a threshold at 3σ.
4. Find pairs with a multiply trick: for every pixel multiply its value by the pixel directly
   above. Bright-on-dark gives a negative product; flag where the product is more negative than the
   squared 3σ threshold — a strong up-down swing.
5. Symmetry check: a real trap moves the *same* charge, so the two lobes should be roughly equal.
6. Repeat-appearance filter (`getDipoleList2`): keep a coordinate only if it appears in **more than
   one image** at the same temperature. This is the main false-alarm filter, **unchanged** between
   versions.

**What minimal changed:**

**(a) Robust noise estimate (`robust_sigma=True`).** Legacy uses the standard deviation, which
squares deviations — a handful of hot defects or high-T dark-current blotches dominate it, σ comes
out too big, the 3σ bar too high, and genuine dipoles fall under it (the finder goes quietly blind,
worst at high temperature). Minimal uses the **MAD** (median absolute deviation × 1.4826): medians
instead of averages, no squaring, so outliers cannot inflate it. For clean data the two agree; MAD
only diverges when there are outliers to ignore. Measured effect: legacy/robust σ ratio is
1.05–1.09 across temperature (larger at high T where dark current sprinkles in abnormal pixels).
*(The squared `(3σ)²` cutoff is bookkeeping for the multiply trick, not a second statistical step —
both versions square it; only the σ feeding in changed.)*

**(b) Disableable symmetry check (`symmetry_perc`).** Legacy hard-wires lobe matching at 30%.
Minimal switches it **off**, because at high temperature a readout pedestal (§2) makes a *genuine*
trap's two lobes unequal, so the strict 30% rule would wrongly reject real traps. Safe because the
repeat-appearance filter and later fitting cuts do the real rejection.

**(c) Reproducible file lists (`image_files`).** Legacy re-scans the folder with a wildcard (result
depends on what's on disk). Minimal accepts a frozen, sorted list for audit trails and fair
comparisons.

The finder operating point was chosen by a pre-declared calibration scan (see [[signed_refit]] §
"Finder"): both lobes must be **separately** significant at 3σ (product-only rules can admit one
sub-threshold lobe and are not production-admissible). The conservative separate-lobe 3σ point
recovers ~91% of strong injections at ~10× less horizontal-structured leakage than the looser 2.5σ
alternative.

---

## 2. Fitting the traps

Two layers: **build the curves** (`getDipoleSpectra2`) — intensity vs delay per trap per
temperature with an error bar per point; then **fit per temperature** (`fitTrapIntensity`) — get τ
at each temperature, then fit all τ across temperature for energy E and cross-section σ.

**Error bars on the curve points (this decides how much each point pulls the fit):**
- **Legacy — patch spread (`error_model='patch'`).** Error bar = spread of pixel values in the
  35×35 patch around the trap. That mixes true random fluctuation with genuine pixel-to-pixel
  nonuniformity, so it **overestimates** the real random noise by ~2.5× (patch σ ≈ 190 e⁻ vs true
  temporal pair noise ≈ 32–39 e⁻).
- **Minimal — physical noise (`error_model='physical'`).** Error bar built only from what genuinely
  fluctuates shot-to-shot, as two pieces added in quadrature:
  1. a **baseline read/temporal noise** looked up from a small table indexed by *temperature ×
     quadrant*. That table was measured from **trap-free pixel pairs** — take the same empty pixel in
     two images and subtract; whatever cancels was fixed pattern, what's left (divided by √2) is the
     honest random-noise floor for that temperature and amplifier (≈32–39 e⁻);
  2. each pixel's own **Poisson shot noise**, √(counts), which is larger for brighter pixels, so it is
     computed per-pixel rather than from the table.

  This replaces the legacy patch spread, which mixed that true random fluctuation together with fixed
  pixel-to-pixel nonuniformity and so overestimated the noise ~2.5×. The old patch spread is still
  computed and stored as `patch_sigma` for comparison, but it no longer sets the fit weights.

**Sign.** Legacy takes `absolute=True` (throws away dipole orientation). Minimal keeps the sign
(`absolute=False`), which the grading stage uses to check a trap always pumps in the same direction.

**The model:**
- **Legacy — 2 knobs:** `I(t) = N_pumps·coeff·(e^{−t/τ} − e^{−8t/τ})`, `coeff ≥ 0`.
- **Minimal — 3 knobs with a pedestal (`fit_offset=True`):**
  `I(t) = 3000·A·(e^{−t/τ} − e^{−8t/τ}) + I0`, both A and I0 **signed.** The pedestal is physical: as
  the dark-current background is clocked out through the trap's pixel during readout, the trap defers
  some of it too, adding a roughly constant dipole present in *every* image regardless of delay. The
  2-knob model has nowhere to put that constant, so it leaks into A and τ and biases them — most
  badly at high temperature where the pedestal is largest. This is the mechanism that made the
  per-temperature fits fail above ~165 K in the legacy analysis (χ² rejected bright hot curves);
  adding the offset restores reduced χ² ≈ 1 at all temperatures, and injection shows the recovered τ
  is unbiased given a constant pedestal.

  The fitted `|offset|` tracks dark current and is highly significant (a fixed-charge-per-cycle
  deferral): median `|offset|` grows 96 e⁻ @140 K → 1058 e⁻ @170 K, with offset/error rising 8 → ~95
  over 140–185 K, while the pumped amplitude stays ~T-independent (≈1000–1800 e⁻). *Caveat:* at
  125–130 K (τ ≫ delay window) offset and coeff are partially degenerate — treat cold-tail offsets
  with care.

![The readout pedestal. **A:** one hot trap's signed intensity-vs-delay curve flattens onto a large
constant offset (dotted line); the 3-knob model with a pedestal term lands (reduced χ² ≈ 1), while
the same curve fit *without* a pedestal is forced through a zero baseline and misses badly (reduced
χ² ≈ 1370) — this is why the legacy per-temperature fits failed at high T. **B:** across every
well-behaved trap, the median pedestal and its significance (|offset|/error) both climb steeply
through 155–170 K, the signature of a fixed-charge-per-cycle dark-current deferral.](figures/pedestal.svg)

*Figure regenerated by [`figures/make_figures.py`](figures/make_figures.py) from
`fit_dipole_spectra_minimal_caldet_err_4.h5`.*

**Absolute error bars (`errors_are_absolute=True`).** Legacy passes `sigma=` to `curve_fit`
*without* `absolute_sigma=True`, so SciPy quietly rescales the parameter covariance by reduced χ²
(`pcov_rel = pcov_abs · χ²/dof`). That changes every downstream significance/uncertainty cut — badly
where reduced χ² < 1 (e.g. median ~0.44 at 207 K → stored errors were only √0.44 ≈ 0.66 of the
supplied ones). Since the minimal error bars are already physically calibrated, minimal trusts them
as-is. This was the one unambiguous **bug** found in the audit; it is fixed behind the flag, with the
corrected artifact under a new filename.

---

## 3. Grading the traps (which fits count as "good")

Two gate sets: per-temperature (is *this* τ trustworthy?) and per-trap (does it behave like one real
physical trap across all temperatures?).

### Per-temperature gates (`fitTrapIntensity`)

Both versions require the fit to converge, the curve to describe the data (a GOF check), and τ
measured to better than 50%. They differ in the "signal is real" test:

- **Legacy — "is the peak tall enough?"** Passes if the tallest curve point exceeds 3× the noise
  (checked as 3× the average error bar and 3× the image noise). Reasonable when intensities are
  absolute-valued and there is no pedestal.
- **Minimal — "is the pump strength significantly non-zero?"** With a pedestal in the model, the
  tallest point mostly measures the pedestal, so the old test is meaningless. Minimal requires the
  fitted pump strength ≥ 3× its own uncertainty (`amplitude_significance ≥ 3`), plus a backstop
  `delta_chi2_vs_constant`: the pumped curve must beat a flat line by a margin — **and that margin
  is the calibrated detection table, §3.1.**

### 3.1 The calibrated detection table (`--detection calibrated`)

*How much* better than a flat line is "better enough"?

- **The fixed guess (NOT used for the headline).** Early code used a single fixed margin (Δχ² =
  11.83) for every temperature, intended as a strict 3σ bar. But this fit has a hidden free knob —
  τ — that the fit can slide around even with no real trap (τ is *unidentified* when the pumped
  amplitude is zero, violating the Wilks regularity assumptions, and the τ scan adds a
  look-elsewhere effect). So the fixed 11.83 is **not** a real 3σ cut: ~1.9% (measured: 1.972%) of
  trap-free locations pass it — closer to 2.3σ than the intended 0.27%.
- **The calibrated table.** Measure the false-alarm rate directly: (1) pick thousands of trap-free
  control locations (~2,000/quadrant, random pixel pairs not at/adjacent to any found trap); (2) run
  the **exact same signed pedestal profile fit** on each, so their "beat the flat line" statistic is
  computed identically; (3) per temperature, set the threshold where only a **0.1%** target fraction
  of pure-noise locations would exceed it (with 8,192 references, the 7th-largest value, via an
  add-one finite-sample p-value that folds the τ look-elsewhere effect into the reference
  distribution). Reproducible (fixed seed) and cached.

Resulting per-temperature thresholds (max GLS Δχ² over the 801-point log-τ grid; independent-null
FPR at that threshold):

| T (K) | Δχ² thr | FPR | T (K) | Δχ² thr | FPR | T (K) | Δχ² thr | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 125 | 25.22 | 0.049% | 165 | 19.68 | 0.159% | 193 | 49.79 | 0.049% |
| 130 | 29.91 | 0.037% | 170 | 18.34 | 0.122% | 195 | 24.51 | 0.037% |
| 135 | 27.12 | 0.012% | 175 | 21.56 | 0.110% | 197 | 24.93 | 0.037% |
| 140 | 24.13 | 0.110% | 180 | 38.45 | 0.085% | 200 | 29.87 | 0.073% |
| 145 | 22.16 | 0.085% | 183 | 49.76 | 0.159% | 203 | 25.94 | 0.085% |
| 150 | 23.16 | 0.085% | 185 | 21.25 | 0.098% | 207 | 30.10 | 0.085% |
| 155 | 21.16 | 0.110% | 187 | 32.28 | 0.073% | 210 | 23.55 | 0.073% |
| 160 | 25.60 | 0.061% | 190 | 37.03 | 0.073% | | | |

Aggregate independent-null FPR 0.081%; the legacy fixed 11.83 admits ~20× the calibrated rate. The
thresholds are explicitly empirical — no Wilks/fixed-σ interpretation.

### Per-trap gates (`fit_energy_cross_section`)

After collecting good-temperature τ's, both require ≥ 4 good temperatures (`wellBehavedThreshold`)
then fit E/σ across temperature. Two changes in minimal:

**(a) Orientation-consistency gate (minimal only).** A single real trap pumps in **one** direction,
so all its significant temperatures should share a sign. Minimal classifies:
`single_orientation_positive/negative` (allowed through); `ambiguous_sign_conflict` (one minority-
sign temperature) and `dual_response` (≥2 positive and ≥2 negative) — flagged **inconsistent and
rejected**; `structured_background_overlap` (shares a pixel with the persistent-horizontal
morphology list) — rejected regardless of sign. Legacy has no concept of sign (it threw it away).
Injection efficiency of the correct single-orientation label is 99.6%; sign accuracy on accepted
fits 100%; end-to-end vertical-null single-orientation rate 0.0%.

**(b) Different GOF rule for the depth fit.**
- **Legacy — reduced-χ² < 5.**
- **Minimal — reduced-χ² < 10** (looser), and it removes legacy's intrinsic-scatter modelling and
  automatic outlier rejection.

The reason is subtle and important. An earlier attempt graded this fit with a **p-value cut**
(`p > 0.05`), which **gets stricter the more temperatures you have**: with ~15 precise τ points it
has enough power to reject a trap over a ~1% real-but-unmodelled high-T lean in the Arrhenius line,
so the pass rate fell from ~65% down to ~21% as the number of temperatures grew — the *best-measured
traps were preferentially thrown away* (see memory `srh-gof-overrejects`). A reduced-χ² cut does not
harden with more points, so it keeps well-measured traps while still removing genuine non-trap blends
(whose reduced-χ² is far above threshold). Reduced-χ² = 10 is deliberately loose for this reason.
**Project note (memory `gof-cut-fpr-scan`):** a *tighter* per-temperature cut around X≈3 actually
*grows* the trap census (+275, +7.2%), while X=10 is net-negative (−27) because loose cuts admit
noisy high-χ² temperatures that break a trap's own Arrhenius fit. The energy fit is a
false-positive backstop (decoys never characterize; FPR flat across X), not a launderer of bad fits.
X = 10 is the in-code default, not necessarily the last word.

**Threshold history (the value drifted — be careful which snapshot a number belongs to).** The cut
started as the biased `p_value > 0.05`, was replaced by `reduced_chi2 < 4` (`srh_reduced_chi2_max`,
`dipole_new.py`) in the 2026-06-17 synthesis, and the in-code default was later loosened to `< 10`.
The p-value → reduced-χ² swap alone recovered +1,187 traps (1,741 → 2,928) with **0 newly dropped**
(a pure superset) and the τ@135K shape barely moved (median 5.74 → 6.07 s, fraction > 100 s
0.020 → 0.019) — a yield / review-defensibility fix, **not** a change to the SER projection. The
null test (§4) shows loosening 4 → 10 is **FPR-free** (only 1 control ever reaches the cut), so it is
a pure yield-vs-purity call to which the SER is insensitive. Catalog counts below are quoted with
their threshold.

---

## 4. Catalog sizes, purity, and what still gets dropped

A trap enters the catalog iff `WellBehavedTrap (≥4 good temps) & OrientationConsistent &
GoodEnergyFit & not EnergyFitFailed` (final gate in `audit_hdf5_records.py`).

**Selection funnel:**

| stage | legacy | minimal |
|---|---:|---:|
| dipoles found (finder) | 5,171 | 9,333 (a strict superset — 0 legacy coords missed) |
| pass full selection, `reduced_chi2 < 4` | 2,135 | 2,928 |
| current `minimal_caldet` catalog (`reduced_chi2 < 10`-era) | — | ~3,798 |

The finder finds more (relaxed MAD threshold + no symmetry cut, both legitimately additive). The
minimal selection keeps ~68% more traps than the old p-value GOF while the τ@135K **shape** — including
the SER-relevant long-τ / high-E tail — barely moves.

**Purity — the trap-free null test.** Running 8,000 trap-free control pairs (`DETECTION_SEED`) through
the full minimal selection yields **1** false positive → catalog FPR ≈ **0.01%** (~0.4 expected noise
false positives in ~2,928). Purity is enforced *upstream* by the calibrated per-temperature detection
guard (0.1%/temp), the ≥4-temperature requirement, and sign consistency; the SRH reduced-χ² cut's real
job is rejecting **blends** (real charge, wrong model) — which a trap-free null cannot measure, hence
loosening it is FPR-free.

**Overlap with legacy** (do not confuse with the *full-refit* overlap of 711 — that is a different
comparison, see [[signed_refit]] §6): minimal 2,928 vs legacy 2,135 share **1,061** at exact (row,col);
legacy-only (dropped) 1,074, minimal-only (new) 1,867. Exact-coord matching is a *floor* — ±1-row
matching raises "shared" substantially, since the relaxed finder places candidates within a row or two
of a real trap. Even so the pipelines are a substantial re-selection, carried together as an
algorithmic systematic.

**What minimal still drops** (legacy-passers dropped by minimal; 1,458/1,464 failed exactly *one*
criterion). The dropped classes carry the **same** E and τ@135K distribution as the kept set (median
E ≈ 0.30 eV, τ@135 ≈ 6 s, τ > 100 s in 1–4% of every group), so dropping them does not erase the SER
tail:

| category | N | verdict |
|---|---:|---|
| `dual_response` (≥2 temps of each sign) | 64 | genuine contamination (two traps in one pixel) — correctly cut |
| `orientation_stray` (one stray >3σ opposite) | 28 | mostly real contamination — arguable |
| `srh_gross_redchi2` (reduced χ² ≥ 10) | 208 | non-Arrhenius blend (τ(T) splits into two branches) — correctly cut |
| `srh_moderate_redchi2` (4 ≤ reduced χ² < 10) | 407 | marginal — recoverable via χ² < 10 |
| `lt4_3temp` (exactly 3 good temps) | 215 | recoverable via `wellBehavedThreshold=3` |
| `lt4_clearbad` (≤1 good temp) | 32 | clearly bad |

Representative examples are rendered under `figures/dropped_catalog/` (indexed in `_index.md`), with
`E_tau135_comparison.pdf` giving the visual proof that kept and dropped span the same E/τ space. The
efficiency-corrected energy-fit cost of this selection is ~4% for minimal (τ-error pull RMS 1.01,
honest error bars), vs legacy's ~1.7× optimistic pull (1.69) propped by its loose reduced-χ² < 5 gate
— one reason **minimal is the headline and legacy a reference**. Details in [[completeness_efficiency]] §5.

---

## Quick reference

| Stage | Legacy (`dipole.py`) | Minimal (`dipole_new.py`, `--pipeline minimal`) |
|---|---|---|
| Noise estimate for finding | Standard deviation (inflated by outliers/hot pixels) | MAD, outlier-resistant (`robust_sigma=True`) |
| Lobe-symmetry check | Hard-wired 30% | Off (`symmetry_perc=None`); later cuts reject |
| Image list | Wildcard rescan | Optional frozen sorted list (`image_files`) |
| Error bars on curve points | Patch spread (~2.5× too big) | Physical temporal + shot noise (`error_model='physical'`) |
| Intensity sign | Absolute value (discarded) | Kept (`absolute=False`) |
| Curve model | 2 knobs: strength (≥0) + τ | 3 knobs: signed strength + τ + pedestal (`fit_offset=True`) |
| Error-bar handling | Rescaled by fitter (bug) | Trusted as-is (`errors_are_absolute=True`) |
| "Signal is real" gate | Tallest point > 3× noise | Pump strength > 3σ + Δχ² vs flat above a **calibrated per-T threshold** (`--detection calibrated`) |
| Δχ² threshold source | n/a (fixed 11.83) | Measured from trap-free controls at 0.1% FPR per temperature |
| Orientation gate | None | Reject sign-inconsistent traps |
| Across-temperature GOF | reduced-χ² < 5 | reduced-χ² < 10 (avoids over-rejecting well-measured traps; tighter X≈3 grows census) |
| Min. good temperatures | 4 | 4 (unchanged) |
| Repeat-appearance filter | >1 image at same temperature | unchanged |

**Catalog sizes** are tabulated in §4 above. In short: legacy 5,171 found / 2,135 characterized;
minimal 9,333 found / 2,928 (χ²<4) → ~3,798 (`minimal_caldet`) characterized; 1,061 shared at exact
coord. The full signed-refit machinery — a separate, more elaborate catalog that was *not* adopted —
is accounted separately in [[signed_refit]].

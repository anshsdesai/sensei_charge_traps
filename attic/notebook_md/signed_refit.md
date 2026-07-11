# The signed-refit investigation: how the trap catalog algorithm was chosen

> **In plain terms.** This is the *story* of how the measurement method was chosen — months of
> exploration, compressed. Short version: the old method broke at high temperature; the fix (the
> "pedestal") snowballed into an ambitious, elaborate rewrite; independent adversarial review found the
> elaborate rewrite was **fooling itself** — claiming more precision than the data can actually support
> (its noise model needed an 8–30× fudge factor, and several of its cuts were secretly biased). So we
> kept only the handful of fixes that are genuinely required by the physics and threw the rest away. The
> survivor is the "minimal" method documented in [[dipole_algorithm]]. Read this if you want to know
> *why* the algorithm is what it is and which dead ends were ruled out. This is history — nothing here
> is a live to-do. New words are in the [glossary](glossary.md).

**Thread status:** resolved. The investigation concluded that the legacy analysis should not be
reverted to, but the *full* signed-refit machinery should not be the production catalog either. The
chosen path is the **minimal corrected pipeline** (`dipole_new.py`, `--pipeline minimal
--detection calibrated`), documented as an algorithm in [[dipole_algorithm]]. This file records the
exploration: what was tried, what broke under adversarial review, and why the minimal synthesis is
the answer.

**Why this thread was so large.** The legacy per-temperature dipole fits failed above ~165 K. The
fix (a readout pedestal) required, in turn: signed intensities, a physical error model, an honest
detection threshold, orientation consistency, and a re-examined energy fit. Each change exposed
another, and an ambitious "full refit" (a 10-step runbook with regional covariance, fitted
overdispersion, robust outlier rejection, intrinsic scatter, and acquisition-family exclusion) was
built and then found to over-claim precision the data cannot support. The de-scoping to the minimal
pipeline is the deliverable of the whole thread.

**Timeline / sources** (all now in `attic/`):

| date | file | role |
|---|---|---|
| 2026-06-12 | `SIGNED_REFIT_AUDIT.md` | first change audit — the pedestal discovery |
| 2026-06-13 | `SIGNED_REFIT_PHYSICS_AUDIT.md` | physics audit — 4 problems + the corrected refit sequence |
| 2026-06-13 | `SIGNED_REFIT_RUNBOOK.md` | the Step 1–10 execution checklist (the "full machinery") |
| 2026-06-13 | `SIGNED_REFIT_RUNBOOK_S01_S05_AUDIT.md` | *earlier unpolished draft* of the Steps 1–5 review (redundant) |
| 2026-06-13 | `SIGNED_REFIT_STEP1_5_REVIEW.md` | adversarial review, Steps 1–5 |
| 2026-06-13→14 | `signed_refit_*.md` (13 files) | per-step calibration reports |
| 2026-06-15 | `SIGNED_REFIT_STEP6_10_REVIEW.md` | adversarial review, Steps 6–10 |
| 2026-06-15 | `MINIMAL_SIGNED_REFIT_RECOMMENDATION.md` | **the decision** — the minimal synthesis |

---

## 1. The trigger — high-T fit failures are a readout pedestal, not noise

The legacy per-temperature dipole fits (`intensity = N_pumps·coeff·(e^{−t/τ} − e^{−8t/τ})`, coeff ≥
0, through zero) failed above ~165 K: the χ² test rejected bright hot curves. The cause is a real,
`t_ph`-**independent** dipole floor: a charge deficit at the trap pixel with charge deferred along
the readout direction, growing with temperature, present in every image. Mechanism: the trap acts
as a CTI/deferred-charge defect on the rising high-T dark-current background during readout. The
floor is ~64 e⁻ at 150 K rising to ~700 e⁻ at ≥175 K, and `floor/σ` crosses 1 at 160–165 K —
exactly where the pass-rate collapses.

**Fix.** Add a constant offset (pedestal) to the model and switch the intensity to **signed** (the
pumped dipole and the pedestal dipole can point in different directions). `model + offset` restores
reduced χ² ≈ 0.15–1 at all temperatures; injection with a constant pedestal recovers τ without
visible bias. This established the two corrections that survive into the production pipeline: the
**signed intensity** and the **constant pedestal**.

The first audit also flagged a proposed replacement energy criterion (`robust_energy_fit` + fitted
intrinsic dispersion + robust outlier rejection) and one genuinely open physics question: hot-T τ
leans ~0.1–0.2 dex below the straight SRH line ("curved Arrhenius"), churning ~650 of 2135 legacy
traps and halving the long-lived tail (τ135 > 1 hr: 0.024 → 0.011). Cause: either a `t_ph`-dependent
pedestal (dark current accumulated during the pump train — instrumental, fixable) or genuine σ(T)
(physics, quote as systematic). See memory `high-t-arrhenius-lean`.

---

## 2. The physics audit — four problems and a corrected sequence

The physics audit (`SIGNED_REFIT_PHYSICS_AUDIT.md`) endorsed keeping the sign and the pedestal but
named four things not yet calibrated well enough to publish:

**(i) Supplied errors were not absolute (a code bug — fixed).** Both fit stages passed `sigma=` to
`curve_fit` without `absolute_sigma=True`, so SciPy rescaled the covariance by reduced χ²:
`pcov(rel) = pcov(abs)·χ²/dof`. Those rescaled errors fed amplitude significance, the `σ_τ/τ ≤ 0.5`
cut, SRH weights, and quoted E/σ errors. Consequential where reduced χ² < 1 (median ~0.44 at 207 K
→ stored errors only √0.44 ≈ 0.66 of the supplied). Fixed behind `errors_are_absolute=True`; legacy
default remains `False` for reproducibility.

**(ii) The per-point error model is not yet physical.** For `I = (a − b)/2`,
`Var(I) = [Var(a) + Var(b) − 2 Cov(a,b)]/4`. The implementation used
`σ_I² = σ_base(T,q)² + (S_a + S_b)/4`, which assumes **independent** pixel shot noise. But a pumped
dipole is a charge **transfer**: the two lobes are anti-correlated, and a random transferred charge
X contributes Var(X) — not E[X]/4 — and over 3000 pump cycles Var(X) ≠ mean. Also
`pair_noise_table.npz` was one scalar per (T, quadrant) computed across the whole dwell scan, not a
per-dwell-point uncertainty. The legacy patch σ is also wrong (it measures spatial nonuniformity,
not repeated uncertainty of a fixed pair). Recommendation: build a scan-level empirical covariance
from masked control-pair curves; treat signal-dependent pumping variance as a *separate, derived*
model component.

**(iii) The fixed Δχ² = 11.83 threshold is not χ² with 2 dof.** Under the constant-only null the
pumped amplitude is zero and τ is undefined, violating the Wilks regularity assumptions; the τ scan
adds a look-elsewhere effect. So 11.83 cannot be a calibrated 3σ test — its false-positive rate must
be obtained empirically. (Confirmed later: it admits 1.97% of nulls.)

**(iv) Robust SRH fit + intrinsic scatter accept more decoys.** End-to-end decoy control:

| control | ≥1 good T | ≥4 good T | characterized |
|---|---:|---:|---:|
| 1600 random pairs | 73.94% | 21.63% | 0.81% |
| 1004 horizontal-null pairs | 87.95% | 46.61% | 2.89% |

The strict SRH stage removes most false "well-behaved" sites — catalog purity currently rides on the
final energy-fit rejection. Applying the proposed robust energy fit with intrinsic scatter *raised*
decoy acceptance (0 dex → 7.5%; 0.20 dex → 13.6%). **Do not** wire robust fitting or intrinsic
scatter into selection until a threshold is chosen from injections + null controls.

The audit issued a 9-point corrected refit sequence, which became the runbook below.

---

## 3. The full signed-refit machinery (Steps 1–10) — what was built

The runbook executed the corrected sequence. Each step had an acceptance gate and a per-step report.
The key engineering and its measured numbers:

- **Step 1 — frozen manifest.** One authoritative FITS per (T, dtph); 200 K run IDs 160–184 (older
  duplicates excluded); NPUMPS=3000 confirmed. *Confound frozen in:* 160 K and 170 K are the **only**
  points from the February `dp_scan1` family (300k charge shifts, 1.5× illumination); everything
  else is March–April `temp_scan_run1` (200k shifts). This aliases temperature with acquisition
  family at the cold, high-leverage end of the Arrhenius lever arm.
- **Steps 2–4 — control-pair noise model + closure.** Regional empirical covariance from masked
  vertical control pairs (per-dwell σ ≈ 35 e⁻, the shot-noise floor of a background pixel);
  whitened on 376,832 held-out curves. Whitened width 1.027 (good) **but the nominal χ² tails do NOT
  close**: constant-model `p<0.05` = 7.7% (target 5%), `p<0.01` = 2.3% (target 1%), KS-vs-uniform
  p = 0; worst at warm scans (183 K → 9.2%). Diagnostic trial-pump statistic: 1.94% of *null* curves
  exceed Δχ² = 11.83 (p99.9 = 24.1) — quantitatively demolishing "11.83 = 3σ."
- **Step 5 — profile fitter in τ.** At each fixed τ, solve signed amplitude A and pedestal I0 by
  GLS (both linear at fixed τ); profile over an 801-point log-τ grid; keep boundary/multimodal flags.
- **Step 6 — empirical detection calibration.** Per-temperature Δχ² thresholds at 0.1% FPR on 8,192
  held-out controls, add-one finite-sample p-value, τ look-elsewhere inside the reference. Aggregate
  null FPR 0.081%; ≥4-temperature nulls 2 of 8,192. **This piece is publishable as-is** and is what
  the minimal pipeline adopts. (Threshold table in [[dipole_algorithm]] §3.1.)
- **Step 7 — finder calibration.** Pre-declared scan of finder configs; operating point chosen for
  completeness at ≥4 robust-image-σ injections subject to null gates:

  | config | strong-signal completeness | horizontal-axis FPR | union candidates | gate |
  |---|---:|---:|---:|---|
  | legacy_reference (3σ, product, 30% balance) | 79.99% | 0.000% | 5,171 | FAIL |
  | robust_separate_3σ_p2 | 91.16% | 0.0116% | 5,341 | PASS |
  | robust_separate_2.5σ_p2 | 94.32% | 0.1159% | 8,241 | PASS |

  The full refit **selected the 2.5σ point** (8,241 candidates). *The minimal recommendation later
  preferred the conservative 3σ point* (91% completeness at ~10× less horizontal leakage) — see §5.
- **Step 8 — orientation policy.** Structurally prevents any opposite-sign merge into a single-trap
  fit; adds a persistent-horizontal-morphology class. Injection single-orientation efficiency 99.6%,
  sign accuracy 100%, end-to-end vertical-null rate 0.0%. **Adopted in minimal.**
- **Step 9 — intensity cutflow.** 8,241 candidates → 38,000 accepted candidate-temperature fits →
  **2,703 final single-trap sites.** Final orientation classes: 1,743 `single_positive`, 960
  `single_negative`, 411 `ambiguous_sign_conflict`, 341 `dual_response`, 2 `structured_background`.
- **Step 9/R1 — signal-dependent variance closure.** To make candidate residual widths close, a free
  per-amplitude-bin multiplier φ was fitted onto the independent-Bernoulli pumping variance
  `3000·q(1−q)`. Frozen factors: **φ = 8.31, 16.06, 30.00, 23.04** (non-monotonic; the cap was
  raised from 20 to 100 so φ≈30 is genuinely data-demanded, not pinned).
- **Step 10 — simple SRH fit.** p-channel SRH, no intrinsic scatter, no outlier rejection.
  Population: 2,703 → 2,407 fit successes → **1,287 SRH-consistent** (profile-deviance p ≥ 0.05).
  Primary variant promoted to `no_160_170`. Vs legacy: only 711 sites are good-energy in both;
  common-population shifts +0.0117 eV and **+0.83 in ln σ (≈2.3×)**.

---

## 4. The adversarial reviews — what broke

Two adversarial physicist/statistician passes concluded the engineering (provenance, plumbing,
no-collapse) was sound but the **physics** of the E/σ catalog was not yet closed.

### Steps 1–5 review — the through-line problem

Everything in Steps 2–5 is calibrated and validated on control pairs that are **empty by
construction.** The central physics risk — that a real pumped dipole (a charge transfer sitting on
more charge) has variance the empty controls cannot see — is never measured. The covariance applied
to candidates is a **lower bound** on their true variance → candidate χ² inflated → p-values too
small → amplitude/τ significances overstated → the Δχ²=1 τ intervals **under-cover on real (bright)
traps.** *This is already visible in the closure:* whitened width and tail rate rise with
control-pair brightness (quartile 1→4: width 1.025→1.035, `p<0.05` 7.49%→8.28%) — within the tiny
brightness range of nulls, let alone real candidates. **BLOCKING.**

Also: the Step 5 coverage validation is **circular** (it draws synthetic data from the *same*
covariance the fitter uses, so it can only prove the GLS algebra is right and that Δχ²=1 ≈ 68% when
noise is exactly Gaussian — which items above say it is not). Fix: inject known signal onto **real
held-out control residual curves** and refit. And the "PASS" headline on Step 4 overstates a closure
that does not close at the nominal level.

*Cleared on inspection:* an unintended dwell-dependent `null_template` subtraction was shown benign
for τ (max template Δχ²-vs-constant 0.12; it carries no pump-shaped component and acts like a
common-mode removal the offset absorbs).

### Steps 6–10 review — the variance knob and the selection biases

- **B1 (BLOCKING) — φ = 8–30 is a tuned knob, not a validated model.** The data demand 8–30× the
  independent-Bernoulli pumping variance, non-monotonically in amplitude — the signature of an
  unmodeled, amplitude-correlated variance term absorbed by a free multiplier fitted to the very
  residual-width target it is then judged against. The binomial model fails at face value and is
  rescued by a 30× factor.
- **H1 (HIGH) — cold-τ Malmquist selection bias.** At 130–140 K the dominant Step 9 rejection is
  `interval_not_two_sided` (135 K: 3,145 of 3,379 detected keep no lifetime). A lifetime is assigned
  only where the profile brackets τ on both sides inside the dwell window — selection on the *noisy
  measured* τ near the window edge. Downward τ fluctuations bracket and survive; upward ones hit the
  long-τ limit and are dropped. Surviving cold τ are biased **short**, biasing the Arrhenius slope E
  **low** — on exactly the cold lever arm that sets the energy.
- **H2 (HIGH) — Step 10 SRH GOF uses the Wilks χ² that Step 6 disproved.** The SRH consistency
  p-value is `chi2.sf(Σ profile Δχ², N−2)`, but those profiles are deliberately non-parabolic/
  asymmetric, so their sum is not χ²_{N−2}. The catalog's most consequential number — 47% non-SRH —
  rests on an uncalibrated reference. Needs the same empirical calibration Step 6 applied to
  detection.
- **H3 (HIGH) — `no_160_170` promotion conflates acquisition family with cold leverage.** 160/170 K
  are simultaneously the only `dp_scan1` family and two of the coldest, highest-leverage points. The
  promotion trigger used a mis-estimated significance (hypot of two independently seeded bootstraps
  on *paired* data, ignoring their correlation), and dropping the points removes the only independent
  cold anchor, so the energy shift is degenerate between "family" and "lever arm."
- **M1 — the two are coupled.** The same φ = 8–30 that closes Step 9 widens τ intervals → smaller
  Step 10 deviance → more `srh_consistent`. Both the R1 "PASS" and the headline SRH fraction ride on
  the same un-validated knob, in the same direction.
- **M3 — the +0.83 ln σ (≈2.3×) vs legacy** propagates directly into simulation capture rates and
  needs a cause (gain, signed model, or SRH-constant convention), not just a number.
- **M2/M5 — load-bearing unconfirmed gain (400 ADU/e⁻, sidecars carry only the fallback 200)** and a
  thin structured-purity margin at the 2.5σ finder (the only barrier between structured backgrounds
  and the catalog caught **2** sites).

**Net verdict:** Steps 6–10 are PASS as *engineering* gates, BLOCKED as *physics* validation of the
E/σ catalog that would seed `tau_at_<T>k_hist.npz` and the simulation.

---

## 5. The decision — the minimal corrected pipeline

`MINIMAL_SIGNED_REFIT_RECOMMENDATION.md` resolved the thread:

> Do not revert to the legacy analysis, but do not use the complete signed-refit machinery as the
> primary catalog. Build a minimal corrected analysis: signed intensities, a constant pedestal, a
> profile fit in τ, empirical null calibration, orientation consistency, and the simple SRH law. Keep
> the legacy analysis as a first-class comparison and source of algorithmic systematic.

**Keep in the primary analysis:** frozen manifest & provenance; signed intensity
`I = (image[row] − image[row−1])/2` with a fixed lobe order; constant pedestal
`I(t) = 3000·A·(e^{−t/τ} − e^{−8t/τ}) + I0`; profile fit in τ (A, I0 linear at fixed τ); empirical
per-temperature detection calibration (Step 6); orientation consistency (Step 8, publish
ambiguous/dual as their own classes); simple SRH with no intrinsic scatter and no outlier rejection.

**Simplify:** prefer the conservative separate-lobe **3σ** finder (91% completeness at ~10× less
horizontal leakage) over the 2.5σ point; use a *pooled* noise model (one covariance per (T,
quadrant) or diagonal + common-mode) rather than thousands of region-specific covariances; retain
one-sided profile limits and boundary flags near scan edges (do not treat the two-sided subset as
unbiased); prefer direct/nearby-T lifetimes and require SRH quality mainly for long extrapolations.

**Demote to systematic/diagnostic variants only** (must pass end-to-end signal + null calibration
before ever defining the catalog): full regional covariance; fitted signal-dependent overdispersion
(the φ = 8–30 knob); the 2.5σ finder; the 160/170 K exclusion; robust SRH outlier rejection; nonzero
intrinsic SRH scatter.

The recommendation is implemented as `dipole_new.py` (the "minimal" pipeline) with
`--detection calibrated`. Three catalogs are carried together and their spread propagated to the
τ(135 K) population and the CCD simulation: **legacy**, **minimal corrected (primary)**, and **full
signed-refit (systematic comparison)**. See [[dipole_algorithm]] for the resulting production
algorithm and memory `live-pipeline-cuts-are-dipole-new`.

**Why this is the right altitude** (memory `analysis-figure-of-merit`): the deliverable is the masked
single-electron residual, not per-trap τ precision. σ is 99.7% E-degenerate and absorbed into the
V_p systematic ([[physics]] §3); the regional-covariance / φ-overdispersion machinery buys apparent
precision the data cannot support, while the finder/pedestal/orientation corrections are
well-targeted and physically required.

---

## 6. Full-refit catalog vs legacy (for the record)

| quantity | legacy (`dipole.py`) | full signed refit | minimal (`dipole_new.py`) |
|---|---:|---:|---:|
| candidates / detected | 5,171 | 8,241 (2.5σ) | 9,333 (`minimal_caldet`) |
| ≥4 good-T sites | 2,514 | 6,545 | — |
| characterized / SRH-consistent | 2,135 | 1,287 (`no_160_170`) | ~3,798 |
| in both vs legacy | — | 711 | ~711 |

Low overlap between any two is intrinsic: the pedestal + signed + physical-error changes are a real
re-selection, not a small correction. The full-refit's low characterized count (1,287) partly
reflects its cold-τ Malmquist rejection (H1) and Wilks SRH cut (H2), both of which the minimal
pipeline avoids by using reduced-χ² GOF and retaining one-sided limits.

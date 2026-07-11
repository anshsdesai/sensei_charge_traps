# Adversarial Physics/Statistics Review — Signed Refit Steps 6–10

- Reviewer pass: adversarial (physicist + statistician), 2026-06-14/15.
- Scope: `SIGNED_REFIT_RUNBOOK.md` Steps 6–10 and their code/artifacts.
- Companion: `SIGNED_REFIT_STEP1_5_REVIEW.md` (Steps 1–5).
- Remediation tasks landed in the runbook as **R8–R16** (see "Adversarial
  review remediation (Steps 6–10)").

Severity legend: **BLOCKING** (catalog not scientifically defensible until
resolved) · **HIGH** (materially biases E/σ or purity) · **MEDIUM** (must be
explained or bounded) · **LOW** (tighten/note) · **OK** (verified sound).

## Artifacts inspected

| Step | Code | Report | Key artifact |
|---|---|---|---|
| 6 | `signed_refit_detection_calibration.py` | `signed_refit_detection_calibration.md` | `signed_refit_detection_calibration.npz` |
| 7 | `signed_refit_finder.py`, `signed_refit_finder_calibration.py` | `signed_refit_finder_calibration.md` | `signed_refit_finder_config.json` |
| 8 | `signed_refit_orientation.py`, `..._validation.py` | `signed_refit_orientation_policy.md` | `signed_refit_orientation_policy.json` |
| 9 | `signed_refit_intensity_pipeline.py` | `signed_refit_intensity_cutflow.md` | `signed_refit_intensity_fits_v1.h5` |
| 9 (R1) | `signed_refit_candidate_variance_closure.py`, `signed_refit_variance_model.py` | `signed_refit_candidate_variance_closure_v2.md` | `signed_refit_candidate_variance_closure_v2.npz` |
| 10 | `signed_refit_srh_pipeline.py` | `signed_refit_srh_validation.md` | `signed_refit_srh_fits_v1.h5` |

## What is sound (OK)

- **Step 6 detection calibration.** The statistic is the max GLS Δχ² over the
  frozen 801-point log-τ grid, calibrated empirically per temperature on 8,192
  held-out controls with an add-one finite-sample p-value; the τ look-elsewhere
  effect is inside the reference distribution; the code explicitly refuses a
  Wilks interpretation (`empirical_threshold`, `_empirical_survival`). Legacy
  Δχ²=11.83 is shown to admit ~20× the calibrated rate. This piece is
  publishable as-is.
- **Step 8 orientation logic** makes a silent opposite-sign merge structurally
  impossible and adds a persistent-horizontal-morphology class that catches
  coherent non-pumping structure sign-consistency alone misses (38/375
  horizontal-axis sites retain a coherent sign).
- **Step 9 provenance.** Coordinates are rebuilt from the frozen finder and
  hard-fail on any count or SHA mismatch; no stale cache can pass as the
  artifact. SHA pinning is consistent across 6→10.

## BLOCKING

### B1 — R1 "PASS" is a tuned knob, not a validated variance model

`estimate_overdispersion_bin` bisects a single free multiplier
`pump_overdispersion` (φ) over [1, `MAX_OVERDISPERSION`] until the candidate
residual width equals 1.0. The frozen per-amplitude-bin factors are
**8.31, 16.06, 30.00, 23.04** (`signed_refit_candidate_variance_closure_v2.md`).

- The data therefore demand **8–30×** the independent-Bernoulli pumping variance
  `3000·q(1−q)`, and the demand is **non-monotonic** in amplitude. That is the
  signature of an unmodeled, amplitude-correlated variance term being absorbed
  by φ — not of a correct model that needed a small finite-sample correction.
- The identical `30.0039` in both cross-fit folds for bin 3 is the bisection
  grid node `1 + 99·300/1024`, **not** the cap (`MAX_OVERDISPERSION` was raised
  from the v1 value of 20 to **100** in v2). So φ≈30 is a genuine data-demanded
  value, which is the more damning reading: the binomial model is off by ~1.5
  orders of magnitude there.
- The Steps 1–5 review already flagged this (v1 φ pinned at the 20 cap). v2
  raised the cap and re-binned so φ now lands below it; the gate passes, the
  physics objection does not. Marking R1 **COMPLETE-PASS** overstates the
  result: the binomial model fails at face value and is rescued by a 30× factor
  fitted to the same residual-width target it is then judged against.

Until the φ requirement is either physically explained (e.g. a derived
correlated-trapping / common-mode-per-cycle term that *predicts* 8–30 without
fitting) or the residual variance is shown to be model-correct without a
per-bin free multiplier, R1 should read **BLOCKED**, and any E/σ uncertainty
that consumes these τ errors inherits that status.

## HIGH

### H1 — Cold-temperature τ selection (Malmquist-type) bias into the Arrhenius fit

At 130–140 K the dominant Step 9 rejection is `interval_not_two_sided`
(135 K: 3,145 of 3,379 detected; only 186 keep a lifetime). The lifetime is
assigned only where the profile brackets τ on both sides inside the dwell
window. This is selection on the **noisy measured** τ relative to the window
edge: at a marginal cold temperature, downward τ fluctuations bracket and
survive, upward fluctuations hit the long-τ limit and are dropped. The surviving
cold-temperature τ are therefore biased **short**, biasing the Arrhenius slope
(E) low and the intercept (σ) correspondingly.

The cutflow names this "a scan-window effect" and stores the dropped points but
**never assesses the bias on the survivors that feed Step 10.** This lands
exactly on the cold lever arm that sets the energy. It must be quantified with
injections through the full Step 9→10 chain (the natural home is Step 11, which
is not started) before the energy catalog is trusted.

### H2 — Step 10 SRH goodness-of-fit uses the Wilks χ² that Step 6 disproved

`signed_refit_srh_pipeline.py:537`: `pvalue = chi2.sf(best_value, N−2)`, where
`best_value` is the **sum of per-temperature profile Δχ²** evaluated at the
SRH-predicted τ (`profile_objective` → `interpolate_profile`). Those profiles
are deliberately non-parabolic/asymmetric (the entire purpose of Step 5), so
their sum is not χ²_{N−2}-distributed. The catalog's single most consequential
number — **47% non-SRH (1,120/2,407)** — rests on this uncalibrated reference.
It needs the same empirical calibration Step 6 applied to detection: push
SRH-consistent multi-T τ sets through the identical machinery and measure the
deviance null, then classify against the measured distribution.

### H3 — `no_160_170` promotion conflates acquisition family with cold leverage

160/170 K are simultaneously the only `dp_scan1` family **and** two of the
coldest, highest-leverage points. Two problems:

1. The population trigger fired on a +0.0029 eV median shift vs a 0.0010 eV
   "combined σ", but that σ is `np.hypot` of two *independently seeded*
   bootstraps (`bootstrap_population_median`, `seed+1`/`seed+2`) on **paired**
   data — it does not estimate the variance of the paired difference. The
   reported significance of the trigger is therefore not the right statistic
   (it ignores the strong positive correlation between the two variants).
2. Dropping these two points removes the only independent cold anchor, so the
   energy shift is degenerate between "acquisition family" and "lever-arm
   change." Promoting `no_160_170` relabels the R4 confound rather than
   resolving it, and — together with H1 — discards part of the weakest,
   most selection-biased end of the fit.

## MEDIUM

### M1 — φ inflation and the SRH-consistent count are coupled in the same direction

The signal-variance covariance carries `φ·3000·q(1−q)`, so the same φ=8–30 that
closes Step 9 (B1) directly **widens** τ intervals → smaller Step 10 deviance →
higher SRH p-value → more `srh_consistent`. φ was tuned to a single-temperature
residual-width=1 target with no check that the inflated τ errors are correct for
the multi-temperature Arrhenius χ². Both the R1 "PASS" and the headline
SRH-consistent fraction ride on the same un-validated knob, in the same
direction. This must be broken before Step 11 (e.g. validate the τ-error scale
against the Step 10 deviance null from H2, not just the per-point width).

### M2 — The unconfirmed 400 ADU/e⁻ gain is now load-bearing in Steps 6–10

Decision 3 froze gain = 400 provisionally even though every selected sidecar
carries only the fallback 200. In Steps 6–10 that constant sets the finder's
"≥4 robust image-σ" completeness gate, every electron amplitude, the
`|D_t P_c| ≤ 1` physicality cut (a **hard** gate in Steps 8/9), and the
shot-noise terms. A factor-2 error shifts the operating point and the catalog;
it does not wash out. (Carryover of R7-gain, now structural.)

### M3 — Unexplained +0.83 ln σ (≈2.3×) and +0.012 eV versus legacy

`signed_refit_srh_validation.md` reports common-population shifts of +0.8275 in
ln σ and +0.0117 eV against the legacy catalog and moves on. A 2.3× systematic
in cross-section propagates directly into the simulation capture rates; it needs
a cause (gain, signed model, or SRH-constant convention), not just a number.

### M4 — Degenerate fits retained in the population

Max site leverage is 0.998 and 8.3% (2,127/25,703) of fitted points place the
pump peak outside the dwell window. A leverage-≈1 site's E/σ is set by
essentially one temperature and its SRH p-value is near-meaningless, yet it
counts toward `srh_consistent` and the population median. These are
stored-but-not-cut; for the population energy/σ they should be excluded, not
only flagged.

### M5 — Structured-background purity margin is thin

The selected 2.5σ / 0.50-balance finder was chosen for completeness (94.3% vs
91.2% at 3σ) at ~10× the horizontal-axis stress leakage (0.1159% vs 0.0116%).
The "~7 false sites" projection assumes the candidate field is as clean as
ordinary controls, but the finder enriches structured backgrounds, where the
≥4-temperature pass rate is ~100× higher (5/375 horizontal vs 2/8,192 ordinary).
The only barrier between structured backgrounds and the single-trap catalog is
the horizontal-morphology overlap list, which caught **2** sites. That is a
narrow margin for the central purity claim and should be stress-tested with
additional independent structured nulls.

## LOW

- **L1** — Each per-temperature threshold is the 7th-order statistic of 8,192,
  so realized per-T FPR scatters 0.012–0.159% around the 0.1% target purely from
  estimation noise. Still under budget, but "p≤0.001" is nominal, not realized;
  consider smoothing/pooling the tail or reporting the realized band as the
  operating rate.
- **L2** — `profile_parameter_interval` anchors the nuisance profile with
  `min(optimizer, value-at-best-nuisance)` to suppress branch jumps; this biases
  SRH parameter intervals **narrow** (overconfident) when the nuisance profile
  is multimodal. Confirm with multimodal synthetic recovery, not only
  unimodal.

## Recommended runbook status changes

- Step 6: keep **PASS** (add L1 note).
- Step 7: **PASS with M2/M5 caveats** — operating point depends on an
  unconfirmed gain and a thin structured-purity margin.
- Step 8: keep **PASS**.
- Step 9: downgrade from **PASS** to **PASS (engineering) / BLOCKED (physics)**
  pending B1, H1, M1.
- Step 10: downgrade from **PASS** to **PASS (engineering) / BLOCKED (physics)**
  pending H2, H3, M1, M3, M4.

The acceptance gates that currently read PASS are real engineering gates
(provenance, plumbing, no-collapse). They are not yet physics validation of the
E/σ catalog that will seed `tau_at_<T>k_hist.npz` and the simulation.

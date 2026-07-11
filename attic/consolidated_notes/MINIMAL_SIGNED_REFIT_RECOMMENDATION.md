# Minimal Signed Refit Recommendation

Date: 2026-06-15  
Status: Proposed analysis direction  

## Purpose

This note compares the legacy dipole/trap characterization with the ongoing
signed-refit work and proposes a smaller synthesis. The objective is to retain
corrections that are physically required by the data without claiming more
statistical precision than the acquisition supports.

The recommendation is:

> Do not revert to the legacy analysis, but do not use the complete current
> signed-refit machinery as the primary catalog. Build a minimal corrected
> analysis consisting of signed intensities, a constant pedestal, a profile
> fit in `tau`, empirical null calibration, orientation consistency, and the
> simple SRH law.

The legacy analysis should remain a first-class comparison and source of
algorithmic systematic uncertainty.

## Physics Judgment

### Is the legacy version better?

The legacy version is simpler, transparent, and close to methods already used
in the pocket-pumping literature. It is therefore a valuable baseline.
However, it is not the better physical model for these data because it:

- takes the absolute value of the dipole intensity and discards orientation;
- forces the intensity curve through zero despite a measured high-temperature
  pedestal;
- uses local spatial patch scatter as the uncertainty of a fixed pixel pair;
- uses local covariance estimates whose scale depends on reduced chi-square;
- accepts SRH fits through an ad hoc `reduced chi-square < 5` criterion.

The inflated legacy errors and loose SRH cut can make the catalog appear stable
by hiding real model failures.

### Is the new version worth continuing?

Yes, but only in a reduced form.

The new work identified real physical and statistical problems:

- the signed dipole orientation carries physical information;
- a large, nearly `t_ph`-independent pedestal is present at high temperature;
- local nonlinear fits can fail or settle on different `tau` minima;
- `tau` is unidentified under the no-pumping null, so a fixed Wilks-style
  likelihood-ratio threshold is not justified;
- sign-changing and dual-response sites should not be silently treated as one
  trap.

Those findings justify continuing the signed analysis. They do not, by
themselves, justify every layer of the current covariance and selection system.

### Where the ongoing analysis is overbuilt

The following pieces currently provide more apparent precision than the data
can support:

- thousands of region-specific covariance matrices;
- signal-dependent pumping variances requiring fitted multipliers of roughly
  `8-30`;
- nominal goodness-of-fit probabilities whose tails still do not close;
- severe two-sided-interval selection at the cold scan boundary;
- an uncalibrated profile-deviance p-value for the multi-temperature SRH fit;
- promotion of the `no_160_170` fit despite its confounding of acquisition
  family and temperature leverage;
- robust SRH outlier rejection or fitted intrinsic scatter that can absorb
  instrumental effects, dual traps, or genuine non-SRH behavior.

The current catalog changes are too large to describe as a small correction.
The legacy catalog has 2,135 good-energy traps, the current signed SRH catalog
has 1,287 primary SRH-consistent traps, and only 711 sites pass both. For common
sites, the current fit shifts the median energy by about `+0.012 eV` and the
cross-section by approximately a factor of `2.3`.

## Component Decisions

### Keep in the primary analysis

1. **Frozen image manifest and provenance**

   Use one explicit acquisition for each `(temperature, dtph)` and record the
   manifest identity in every output.

2. **Signed intensity**

   Use

   ```text
   I = (image[row, col] - image[row - 1, col]) / 2
   ```

   with one fixed lobe-order convention throughout the pipeline.

3. **Constant pedestal**

   Fit

   ```text
   I(t) = 3000 A [exp(-t/tau) - exp(-8t/tau)] + I0.
   ```

   The pedestal is directly motivated by the observed deferred-charge
   structure and prevents high-temperature `tau` fits from being biased or
   rejected merely because the zero-offset model is incomplete.

4. **Profile fit in `tau`**

   At each fixed `tau`, solve the signed amplitude `A` and pedestal `I0`
   linearly. Profile over a fixed log-`tau` grid and retain boundary and
   multimodality flags.

5. **Empirical detection calibration**

   Calibrate the maximum profile improvement on real control-pair curves using
   the actual dwell grid. Do not describe a universal fixed
   `delta chi-square` threshold as a nominal sigma significance.

6. **Orientation consistency**

   Require a stable significant sign for a single-trap SRH fit. Publish
   opposite-sign sites as `ambiguous` or `dual_response` rather than deleting
   them or combining their lifetimes.

7. **Simple SRH model**

   Fit the unmodified p-channel SRH relation with no intrinsic scatter and no
   automatic outlier rejection. A poor SRH fit is scientific information, not
   necessarily a point to repair.

### Simplify in the primary analysis

1. **Finder**

   Prefer the conservative separate-lobe `3 sigma` finder with a relaxed
   balance requirement and persistence across multiple dwell values.

   In the current calibration it recovers about `91%` of strong injections,
   compared with about `94%` for the selected `2.5 sigma` finder, while reducing
   horizontal structured leakage by roughly a factor of ten and producing a
   much smaller candidate set.

2. **Noise model**

   Use a pooled model such as one covariance per `(temperature, quadrant)`, or
   a diagonal dwell-dependent variance plus a small common-mode component.
   Validate it on held-out controls.

   The full regional covariance model should be retained as a systematic
   variation. Its complexity is only worthwhile if it materially changes
   physics outputs after common selections.

3. **Lifetime reporting near scan boundaries**

   Retain one-sided profile limits and boundary flags. Do not silently treat
   the subset with two-sided intervals as an unbiased sample.

4. **SRH use**

   Use direct or nearby-temperature lifetime measurements whenever possible.
   Require an acceptable SRH description primarily when extrapolation to
   `135 K` is substantial.

### Keep only as systematic or diagnostic variants

- full region-level covariance;
- fitted signal-dependent overdispersion;
- the `2.5 sigma` finder;
- exclusion of the 160 K and 170 K acquisition family;
- alternative intensity-quality thresholds;
- alternative SRH goodness thresholds;
- robust SRH outlier rejection;
- nonzero intrinsic SRH scatter.

Robust rejection and intrinsic scatter should not define the production
catalog unless they pass end-to-end signal and null calibration.

## Proposed Primary Pipeline

The proposed primary characterization is:

```text
frozen image manifest
    -> conservative signed dipole finder
    -> signed spectra
    -> pooled empirical control covariance
    -> profile fit of A, tau, I0
    -> empirical per-temperature detection test
    -> stable-orientation single-trap classification
    -> simple SRH fit using all acquisition families
    -> measured/extrapolated tau(135 K) catalog
```

Three outputs should be carried together:

1. **Legacy catalog:** exact historical selection and outputs.
2. **Minimal corrected catalog:** proposed primary result.
3. **Full signed-refit catalog:** high-complexity systematic comparison.

The spread among these analyses should be propagated to the `tau(135 K)`
population and the downstream CCD simulation rather than selecting one catalog
solely because it produces the cleanest residuals.

## Implementation Sketch

### Step 1: Freeze the analysis contract

- Reuse `signed_refit_manifest.csv`.
- Freeze the signed lobe-order convention.
- Define versioned filenames for the minimal analysis.
- Record gain, pump count, dwell conversion, and temperature assumptions.

Suggested outputs:

```text
minimal_refit_config.json
minimal_refit_manifest_summary.md
```

Acceptance check:

- every output records the configuration and manifest hashes;
- legacy and full-refit artifacts are never overwritten.

### Step 2: Freeze the conservative finder

- Start from the calibrated separate-lobe `3 sigma` configuration.
- Require opposite-sign vertical lobes, relaxed balance, and persistence at
  least twice; also report the effect of requiring three detections.
- Keep trail and horizontal-response flags as diagnostics rather than broad
  automatic vetoes unless their null rejection is demonstrated.
- Run the existing injection and structured-null calibration at this exact
  operating point.

Acceptance check:

- strong-signal completeness and structured-null leakage are reported;
- candidate count is a consequence, not an optimization target.

### Step 3: Build the minimal control-error model

- Reuse the frozen control coordinates.
- Estimate per-dwell variance for each `(temperature, quadrant)`.
- Add a pooled common-mode covariance only if held-out residual correlations
  require it.
- Avoid amplitude-bin tuning in the primary error model.
- Store the regional covariance model as an alternate fit.

Acceptance check:

- held-out residual width and correlations are reported by temperature;
- analytical chi-square tails are not claimed when they do not close;
- the simpler model is rejected only if it demonstrably biases injected `tau`.

### Step 4: Implement the minimal profile fitter

- Reuse the profile-over-`tau` solution.
- Solve `A` and `I0` by generalized least squares at fixed `tau`.
- Store the complete profile, best-fit sign, pedestal, and one- or two-sided
  lifetime interval.
- Keep boundary-limited and multimodal fits in the artifact with explicit
  classifications.

Acceptance check:

- injections on real held-out residual curves recover `tau` without material
  bias;
- positive and negative orientations behave symmetrically;
- the offset fit is compared with the zero-offset fit as a closure diagnostic.

### Step 5: Calibrate detection empirically

- Reuse the maximum profile improvement statistic.
- Calibrate its null distribution per temperature, or pool nearby
  temperatures only after demonstrating compatibility.
- Choose one false-positive budget before examining the final trap population.
- Include finder selection in the end-to-end null test.

Acceptance check:

- ordinary and structured-null rates satisfy the declared budget;
- no nominal Wilks or fixed-sigma interpretation is used.

### Step 6: Classify physical response

- Require at least four accepted temperatures for a single-trap SRH fit.
- Require one stable orientation across accepted temperatures.
- Publish separate classifications for:

```text
single_orientation
ambiguous_sign_conflict
dual_response
structured_background
insufficient_temperature_coverage
```

Acceptance check:

- no accepted opposite-sign response enters the single-trap SRH sample;
- classification efficiency is tested on signed injections.

### Step 7: Fit the simple SRH relation

- Fit all available accepted temperatures with the existing p-channel SRH
  equation.
- Use no intrinsic scatter and no automatic outlier removal.
- Keep 160 K and 170 K in the primary fit.
- Produce `no_160_170` and other acquisition-family exclusions as systematic
  variants.
- Flag high-leverage and prediction-outside-profile fits.

Acceptance check:

- SRH consistency is calibrated with end-to-end SRH-consistent injections, or
  described as a descriptive goodness statistic rather than an exact p-value;
- population summaries exclude parameter-boundary and nearly single-point
  leverage cases;
- non-SRH traps remain visible as a separate population.

### Step 8: Construct the 135 K population

- Label each `tau(135 K)` as directly measured, locally interpolated, or
  extrapolated.
- Prefer measured or nearby-temperature information when available.
- Require stronger SRH quality for long extrapolations.
- Generate matched legacy, minimal, and full-refit `tau(135 K)` distributions.

Acceptance check:

- report catalog overlap and migration;
- report how much of the long-lifetime tail is directly constrained;
- propagate method variation into the histogram or simulation ensemble.

### Step 9: Regenerate completeness consistently

- Inject signals using the same finder, pedestal model, noise controls, and
  profile fitter used for the minimal catalog.
- Pass injections through the complete multi-temperature and orientation
  selection.
- Explicitly test cold-boundary selection bias.

Acceptance check:

- recovered energy and `tau(135 K)` are unbiased over the quoted sensitivity
  region;
- efficiency corrections are not derived from a different selection than the
  production catalog.

### Step 10: Propagate the algorithmic systematic

Run the downstream simulation with:

```text
legacy population
minimal corrected population
full signed-refit population
minimal finder/noise/acquisition-family variants
```

The primary result should use the minimal corrected population. The envelope or
structured comparison of the variants should be quoted as the trap-analysis
systematic.

## Recommended Near-Term Work

The next implementation target should be a compact driver, separate from both
`run_charge_traps.py` and the full Step 1-10 signed machinery, that reuses:

- the frozen manifest;
- the conservative finder calibration;
- the profile-`tau` fitter;
- empirical detection calibration;
- orientation classification;
- the simple SRH model.

It should deliberately omit:

- regional candidate-specific covariance as the default;
- fitted pumping overdispersion as the default;
- robust SRH outlier rejection;
- intrinsic SRH scatter;
- automatic promotion of an acquisition-family exclusion.

This creates an analysis that is physically better than the legacy version,
substantially simpler than the current full refit, and straightforward to
explain in the manuscript.

## References

- `paper/paper.tex`
- `dipole.py`
- `SIGNED_REFIT_PHYSICS_AUDIT.md`
- `SIGNED_REFIT_STEP1_5_REVIEW.md`
- `SIGNED_REFIT_STEP6_10_REVIEW.md`
- `signed_refit_detection_calibration.md`
- `signed_refit_finder_calibration.md`
- `signed_refit_intensity_cutflow.md`
- `signed_refit_srh_validation.md`
- SENSEI trap analysis: <https://arxiv.org/html/2510.23336v1>
- Oscura pocket-pumping analysis: <https://arxiv.org/html/2406.18502>

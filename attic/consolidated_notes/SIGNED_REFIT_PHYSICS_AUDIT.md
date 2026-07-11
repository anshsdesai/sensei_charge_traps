# Signed Refit Physics Audit

Date: 2026-06-13

This audit reviews the signed dipole finder, per-temperature intensity fits,
uncertainties, and the SRH energy/cross-section fit. It compares the code with
the acquisition sequence, the paper draft, cached diagnostics, synthetic
injections, and the end-to-end decoy control.

## Bottom line

Do not revert wholesale to the legacy analysis. Retaining the dipole sign and
fitting a constant pedestal are both physically and statistically better than
the legacy absolute-value, zero-offset fit.

Do not publish or propagate the current signed catalog yet. Four parts of the
new selection are not calibrated well enough:

1. The fit covariance treated measured error bars as relative weights.
2. The proposed "physical" error formula is not established for pumped,
   anti-correlated charge transfer and discards measured dwell-time dependence.
3. The fixed `delta_chi2 = 11.83` threshold does not have the claimed null
   distribution because `tau` is unidentified when the pumped amplitude is zero.
4. The robust SRH fit and fitted intrinsic scatter accept substantially more
   decoys and can absorb temperature-correlated model failures.

The first issue is an unambiguous code bug and is fixed behind a backward
compatible flag. The other three require calibrated controls before changing
the production catalog.

## Findings by severity

### Critical: supplied errors were not absolute

Both intensity and SRH fits passed `sigma=` to `scipy.optimize.curve_fit` without
`absolute_sigma=True`. SciPy's default leaves the best-fit parameters unchanged
but multiplies their covariance by reduced chi-square:

```text
pcov(relative sigma) = pcov(absolute sigma) * chi2 / dof
```

The code then used those rescaled errors for:

- pumped-amplitude significance;
- the `sigma_tau / tau <= 0.5` cut;
- weights in the SRH fit;
- quoted energy and cross-section errors.

This is especially consequential where reduced chi-square is below one. The
signed validation had median reduced chi-square about 0.44 at 207 K, so the old
stored standard errors were only about `sqrt(0.44) = 0.66` of the errors implied
by the supplied sigma values.

Fix:

- `fitTrapIntensity(..., errors_are_absolute=True)` now passes
  `absolute_sigma=True` in both fit stages.
- The legacy default remains `False` for reproducibility.
- `run_signed_pipeline.py` and `run_decoy_control.py` opt into the corrected
  behavior.
- The corrected output has a new filename:
  `fit_dipole_spectra_signed_abssigma_err_4.h5`.

The existing `fit_dipole_spectra_signed_err_4.h5` is statistically stale.

### Critical: the current per-point error model is not yet physical

For `I = (a-b)/2`, the general variance is

```text
Var(I) = [Var(a) + Var(b) - 2 Cov(a,b)] / 4.
```

The implementation uses

```text
sigma_I^2 = sigma_base(T,q)^2 + [S_a + S_b] / 4
```

after clipping each `S` below zero. This assumes independent pixel shot noise.
A pumped dipole is instead a charge transfer: the two lobes are anti-correlated.
If a random transferred charge `X` adds to one pixel and subtracts from the
other, its contribution to `Var(I)` is `Var(X)`, not `E[X]/4`. For 3000 pump
cycles, `Var(X)` also depends on capture and emission probabilities and is not
generally equal to its mean.

There are two additional problems:

- `pair_noise_table.npz` is one scalar per `(temperature, quadrant)`, calculated
  as variation across the entire dwell scan. It is not a measurement of each
  dwell point's uncertainty.
- The existing completeness audit found non-negligible dwell dependence and
  explicitly selected a `(temperature, quadrant, dtph)` conditional noise
  model. Its range/median statistic reaches 2.90 at 207 K. The signed scalar
  table throws this information away.

The legacy local-patch sigma is also not the right answer: it measures spatial
nonuniformity rather than repeated uncertainty of a fixed pair. The correct
next step is neither current implementation. There are no repeated same-setting
or unpumped acquisitions, so repeat-based separation of baseline and pumping
variance is not available.

Recommendation:

1. Track a large ensemble of masked, noncandidate vertical pairs through every
   image in each actual dwell-time scan.
2. Estimate the full empirical null covariance across dwell points, stratified
   by temperature, quadrant, and detector region. This preserves common-mode
   and row-median-subtraction correlations without requiring repeat images.
3. Use matched nearby control pairs or shrink region-level covariance estimates
   toward a well-conditioned common estimate; a single scalar per scan is not
   sufficient.
4. Treat signal-dependent pumping variance as a separate model uncertainty. It
   can be bounded with the pump-cycle counting model and checked using residual
   closure binned by fitted amplitude, but cannot be measured independently
   from these acquisitions.
5. Validate whitened residuals against dwell time, temperature, quadrant,
   amplitude, and pedestal before using chi-square p-values.

### High: the `delta_chi2` threshold is not chi-square with two degrees of freedom

Under the constant-only null, the pumped amplitude is zero and `tau` has no
defined value. This violates the regularity assumptions behind the usual
likelihood-ratio chi-square result. Searching over `tau` also introduces a
look-elsewhere effect.

Therefore `delta_chi2 >= 11.83` cannot be described as a calibrated 3-sigma,
two-parameter test. It may still be a useful ranking statistic, but its false
positive rate must be obtained empirically.

Recommendation:

- At fixed `tau`, solve amplitude and offset by weighted linear least squares.
- Scan/profile over log(tau), which avoids local-initialization failures.
- Calibrate the maximum profile improvement using null control pairs on each
  actual dwell grid and noise condition.
- Include candidate finding in the calibration because the same images are used
  both to select coordinates and to fit them.

### High: decoy rates show that intensity-fit selection remains permissive

The end-to-end decoy control produced:

| Control | At least 1 good T | At least 4 good T | Characterized |
|---|---:|---:|---:|
| 1600 random pairs | 73.94% | 21.63% | 0.81% |
| 1004 horizontal-null pairs | 87.95% | 46.61% | 2.89% |

The current strict SRH stage removes most false "well-behaved" sites. This means
the catalog purity currently depends heavily on the final energy-fit rejection,
not just on identifying a pumped dipole.

Applying the proposed robust energy fit to the decoys with no intrinsic scatter
accepted 61 of 814 decoys having at least four fitted temperatures (7.5%).
Adding 0.10, 0.15, and 0.20 dex intrinsic scatter increased this to 10.7%,
12.2%, and 13.6%, respectively. The proposed data-derived value near 0.061 dex
therefore also increases false acceptance relative to the current 42 decoys.

Do not wire `robust_energy_fit` or `estimate_intrinsic_dispersion` into the
catalog selection until an end-to-end threshold is chosen from signal
injections and null controls.

### High: temperature uncertainty and acquisition identity are omitted

The SRH slope is very sensitive to temperature:

```text
d ln(tau) / dT = -E/(k_B T^2) - 2/T.
```

Around 180 K, a 0.5 K uncertainty gives roughly 0.025-0.04 dex uncertainty for
0.3-0.5 eV traps. This is comparable to the proposed intrinsic scatter.
Temperature is a correlated systematic shared by all traps at a setpoint, not
independent per-trap noise.

The pipeline currently derives integer temperature from filenames. The inspected
FITS headers do not provide the actual sensor temperature. A 3.5 hour settling
sleep does not replace a calibrated sensor measurement and uncertainty.

The 200 K input also contains 29 images but only 25 unique dwell values. Dwell
values 750, 1200, 2000, and 3000 occur in two acquisition runs. The current
spectra merge them and fit one offset. The completeness workflow separately
chooses the April run, IDs 160-184. The production refit needs an explicit,
versioned image manifest and a duplicate policy.

### High: signed orientation is useful but not enforced across temperature

Keeping signed intensity is correct. The orientation carries physical
information about the trap's sub-pixel phase. The present code fits an
independent signed amplitude at every temperature, then discards its sign when
performing the SRH fit.

A site whose fitted orientation changes with temperature can therefore provide
an apparently smooth set of positive `tau` values. Such sites may be noise,
overlapping traps, or the "dual-response" behavior excluded in related pocket
pumping analyses.

Recommendation:

- Record the fitted orientation and its significance at every temperature.
- Require a stable significant sign for a single-trap SRH fit.
- Flag rather than silently repair sign-changing or dual-response sites.
- Test the sign-stability rule on injected traps and all decoy classes.

### Medium: finder completeness increased at an unquantified purity cost

Disabling lobe symmetry and using the robust MAD threshold increased candidates
from 5171 to 9333. Removing symmetry is understandable in the presence of a
readout pedestal, but the product cut

```text
a * b < -(3 sigma)^2
```

does not require both lobes to exceed `3 sigma`; a very strong lobe can admit a
weak opposite-sign neighbor. Persistence currently means at least two distinct
dwell values, despite comments and related descriptions saying more than two.

Recommendation:

- Treat the finder as permissive candidate generation, not trap identification.
- Add pair charge-conservation and vertical trail diagnostics.
- Decide explicitly between at least two and at least three dwell detections.
- Optimize finder operating points using injected completeness versus
  end-to-end false characterization, stratified by temperature and quadrant.

### Medium: the offset model itself is supported

The signed model

```text
I(t) = 3000 A [exp(-t/tau) - exp(-8t/tau)] + I0
```

matches the clock sequence and the form used in the paper and related pocket
pumping work. The factor of eight is consistent with the sequence; fixed
five-tick overheads are too small to explain the observed 0.1-0.2 dex high-T
shift.

Synthetic injections with a constant pedestal recover `tau` without a visible
bias over the tested amplitudes and dwell grids. At 190 K, accepted short-tau
traps also show tail residuals consistent with a flat pedestal. These checks
support retaining the offset, while not proving that every high-temperature
curve is a single trap plus a perfectly constant pedestal.

### Medium: the SRH algebra is right, but the paper should use hole notation

`log_energy_cross_section` implements

```text
tau = exp(E_t / k_B T) / [sigma * v_th * N_v]
```

with the expected `T^-2` prefactor for p-channel hole emission. The numerical
unit conversion is internally consistent.

The paper draft calls the effective density of states `N_c` and describes a
conduction-band quantity. For this p-channel hole-emission model it should be
`N_v`, and the trap depth should be defined relative to the valence-band edge.
The code comments already use `N_v`.

## Quantitative catalog checks

The existing signed artifact contains:

- 9333 candidates;
- 6545 sites with at least four accepted temperature fits;
- 2263 sites passing the current strict energy criterion.

The legacy artifact contains:

- 5171 candidates;
- 2514 sites with at least four accepted temperature fits;
- 2135 characterized sites.

Only 789 sites are characterized in both catalogs. This low overlap is too large
to summarize as a small correction to the old analysis.

For low-temperature SRH fits extrapolated to recovered points above 160 K, the
median pull was -8.91 and 71.3% of points had absolute pull above five. This is
not compatible with the stated independent error model. Possible causes include
temperature calibration, acquisition-family offsets, dual traps, field-assisted
emission, temperature-dependent cross-section, or underestimated/correlated
errors. It should not be hidden by a fitted per-point intrinsic scatter.

## Recommended refit sequence

The editable execution checklist, acceptance gates, and per-step notes template
are maintained in `SIGNED_REFIT_RUNBOOK.md`.

1. Freeze an input manifest with one identified acquisition per `(T,dtph)`.
   Use the recent 200 K run IDs 160-184 and exclude the older duplicates. Treat
   the filename temperature as stable and exact for this analysis because no
   slow-control logs are available, while documenting that assumption.
2. Keep signed pair intensity and the constant pedestal.
3. Build a scan-level empirical covariance model from masked control-pair curves
   and test whitened-residual closure. Include pumping variance only as a
   separately derived and validated model component.
4. Replace the local nonlinear intensity fit with a log(tau) profile fit:
   amplitude and pedestal are linear at fixed tau.
5. Calibrate the profile detection statistic on candidate-finder null controls.
6. Require stable dipole orientation or classify the site as dual/ambiguous.
7. Propagate asymmetric profile intervals for tau rather than relying only on
   local covariance near scan boundaries.
8. Fit the simple SRH model before introducing outlier rejection or intrinsic
   scatter. Treat the unavailable temperature-calibration uncertainty as an
   explicit limitation rather than estimating it from the trap residuals.
9. Choose catalog thresholds from injected completeness and null purity, then
   regenerate all completeness and simulation inputs from the same version.

## Changes made during this audit

- Added the backward-compatible `errors_are_absolute` option to
  `fitTrapIntensity`.
- Enabled it in the signed and decoy drivers.
- Applied it to both intensity and energy/cross-section covariance.
- Added provenance fields to each stored fit.
- Versioned the corrected signed fit artifact name.
- Added `test_fit_absolute_sigma.py`.
- Corrected the paper's p-channel SRH notation from conduction-band `N_c` to
  valence-band `N_v` and defined the hole trap depth relative to `E_v`.

No finder threshold, error formula, robust SRH rule, paper count, or production
catalog was changed by this audit.

## Confirmed analysis assumptions

1. No slow-control temperature logs are available. Treat each filename
   temperature as stable and exact, and carry this as an unquantified systematic
   limitation rather than fitting a temperature nuisance parameter.
2. The recent 200 K acquisition, run IDs 160-184, is authoritative.
3. No repeated same-setting or unpumped scans are available. Noise calibration
   must use control-pair ensembles from the existing dwell scans.

One catalog-policy question remains: whether sign-changing/dual-response traps
should form a separate physical class or be excluded from the single-trap SRH
catalog.

## References

- SENSEI charge-trap analysis: https://arxiv.org/abs/2510.23336
- Oscura pocket-pumping analysis: https://arxiv.org/abs/2406.18502
- SciPy `curve_fit` covariance convention:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

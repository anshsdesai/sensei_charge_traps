# Signed Dipole Refit Runbook

Status: Draft  
Created: 2026-06-13  
Owner:  
Analysis version/tag: `signed-refit-input-v1`  

This is the working checklist for producing a defensible signed dipole catalog.
Each step has an acceptance gate and a notes section to complete before moving
forward.

## Fixed assumptions

- Treat filename temperatures as stable and exact for this analysis.
- Use the most recent 200 K acquisition, run IDs 160-184.
- No repeated same-setting, unpumped, or dedicated dark-control scans exist.
- Keep signed dipole intensities.
- Keep the constant intensity pedestal.
- Treat `fit_dipole_spectra_signed_err_4.h5` as stale.
- Do not use `pair_noise_table.npz` as the final production error model.
- Do not enable robust SRH outlier rejection or intrinsic scatter by default.

---

## Step 1: Freeze the input image manifest

### Objective

Define exactly one authoritative FITS image for every intended `(temperature,
dtph)` point so all later products use the same acquisition set.

### Tasks

- [x] Index all candidate FITS images and parse temperature, `dtph`, quadrant
      applicability, run ID, date, pump count, clock settings, and image shape.
- [x] Select run IDs 160-184 at 200 K.
- [x] Exclude the older duplicate 200 K images.
- [x] Confirm that every other temperature has no unresolved duplicate `dtph`.
- [x] Check that `NPUMPS=3000` and relevant voltage/clock settings are consistent.
- [x] Save a machine-readable manifest and a human-readable summary.
- [x] Add manifest identity or hash to every downstream artifact.

### Outputs

- `signed_refit_manifest.csv`
- `signed_refit_manifest_summary.md`

### Acceptance criteria

- [x] Every selected `(temperature, dtph)` is unique.
- [x] Every exclusion has a recorded reason.
- [x] All selected images have compatible pumping and image settings.
- [x] A fresh pipeline run can consume the manifest without using `glob` as the
      scientific data-selection rule.

### Completion notes

Status: Complete - PASS  
Date completed: 2026-06-13  
Files produced: `signed_refit_manifest.csv`,
`signed_refit_manifest_summary.md`, `signed_refit_manifest.py`,
`test_signed_refit_manifest.py`; manifest-aware changes in
`run_signed_pipeline.py` and `dipole.py`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_manifest.py`; `conda run -n sensei_charge_traps_new python
signed_refit_manifest.py --validate-only`; `conda run -n
sensei_charge_traps_new python -m unittest test_signed_refit_manifest.py
test_fit_absolute_sigma.py`; `conda run -n sensei_charge_traps_new python -m
py_compile signed_refit_manifest.py test_signed_refit_manifest.py
run_signed_pipeline.py dipole.py`.  
Results: Indexed 481 candidate CCD2 FITS files across 23 temperatures. Selected
477 unique `(temperature, dtph)` images. Excluded exactly four superseded 200 K
images at `dtph={750,1200,2000,3000}` from run IDs 21-24. Selected 200 K run IDs
are exactly 160-184. All selected files use `NPUMPS=3000`, `vl=-2.75`, `vh=7.5`,
unit binning, four `580x3600` image HDUs, and identical readout-delay headers.
Manifest regeneration was deterministic. Manifest SHA-256:
`477cca5d74a2dcf953aaeb6e8f614f3f5a807f87f09a992024609acb73c07b67`.
All five unit tests passed.  
Problems or deviations: The 160 K and 170 K `dp_scan1` acquisitions use 300000
charge-generating shifts; `temp_scan_run1` uses 200000. This is internally
consistent within each acquisition family and is retained as an intentional
illumination difference. The intensity fit has an independent amplitude at each
temperature.  
Decision: Acceptance gate passed. Freeze `signed-refit-input-v1` and require its
manifest SHA-256 on signed downstream artifacts.  

---

## Step 2: Define control-pair samples

### Objective

Construct a large ensemble of vertical pixel pairs that samples the null
background while excluding known or plausible charge traps and detector defects.

### Tasks

- [x] Build an exclusion mask from legacy and signed candidate coordinates.
- [x] Dilate the candidate mask sufficiently to remove neighboring lobes and
      deferred-charge trails.
- [x] Apply detector boundary, overscan, hot-pixel, hot-column, bleed, and other
      established quality masks.
- [x] Sample control pairs across each quadrant and detector region.
- [x] Keep pair coordinates fixed across all `dtph` images at a temperature.
- [x] Define spatial regions large enough for statistics but small enough to
      capture detector nonuniformity.
- [x] Reserve independent control subsets for model fitting and validation.
- [x] Save coordinates, masks, region labels, and selection provenance.

### Outputs

- `signed_refit_control_pairs.npz`
- `signed_refit_control_pair_summary.md`

### Acceptance criteria

- [x] Control pairs do not overlap candidate or defect masks.
- [x] Each `(temperature, quadrant, region)` has adequate control statistics.
- [x] Training and validation controls are disjoint.
- [x] Control distributions have been inspected for obvious trap contamination.

### Completion notes

Status: Complete - PASS  
Date completed: 2026-06-13  
Files produced: `signed_refit_control_pairs.npz`,
`signed_refit_control_pair_summary.md`, `signed_refit_controls.py`,
`test_signed_refit_controls.py`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_controls.py`; `conda run -n sensei_charge_traps_new python
signed_refit_controls.py --validate-only`; `conda run -n
sensei_charge_traps_new python -m unittest test_signed_refit_controls.py
test_signed_refit_manifest.py`. Control version:
`signed-refit-controls-v2`.  
Results: Selected 65,536 fixed controls across four quadrants and a 4 x 8
region grid. Every quadrant/region contains 384 training and 128 validation
controls, for 49,152 training and 16,384 validation controls total. Candidate
exclusion uses the union of legacy and initial signed catalogs, a 20-pixel
Chebyshev halo, and a 20-row by 2-column vertical trail exclusion. Persistent
defects and hot columns were derived from a robust static median of one
representative image per temperature. The minimum control distance from a
candidate center is 21 pixels. All seven manifest/control tests passed.
Control-artifact SHA-256:
`a8de148506e2d8f863844d903904c0b1c93f2e8785f14be92ee8a9a015b7fea6`.  
Problems or deviations: No separate bad-pixel map exists for these
pocket-pumping scans. A robust persistent-defect/hot-column mask was therefore
derived from the existing images independently of the sampled control-curve
fluctuations. The first implementation treated a broad quadrant-2 spatial
baseline as hot columns; this was corrected by estimating column outliers within
each detector column region. The superseded v1 controls used an 8-pixel halo.
Step 4 localized their failed warm-scan closure to repeatable pump-like controls
9-20 pixels from cataloged candidates, including Q0 `(438, 2863)` and Q3
`(499, 2951)`. A 20-pixel halo matches the already adopted deferred-charge
scale, leaves at least 2,272 eligible pairs in every region, and was fixed
before generating the new v2 training/validation sample.  
Decision: Acceptance gate passed. Freeze `signed-refit-controls-v2`; Step 3 must
use these coordinates and may not resample controls.  

---

## Step 3: Calibrate the empirical noise model

### Objective

Estimate the null covariance of signed pair-intensity curves across the actual
`dtph` grid for each temperature, quadrant, and detector region.

### Tasks

- [x] Extract signed control intensity `I=(a-b)/2` after the same row correction
      used for candidates.
- [x] Estimate and remove each control pair's fixed spatial offset.
- [x] Measure per-`dtph` variance rather than one scalar per temperature.
- [x] Measure covariance between `dtph` points.
- [x] Compare classical and robust covariance estimators.
- [x] Regularize or shrink covariance matrices to ensure stable inversion.
- [x] Quantify dependence on quadrant, detector region, temperature, `dtph`,
      background charge, and pair brightness.
- [x] Choose a mapping from each candidate to an appropriate covariance model.
- [x] Save covariance matrices and all preprocessing metadata.

### Outputs

- `signed_refit_noise_model.npz` or `.h5`
- `signed_refit_noise_model_report.md`

### Acceptance criteria

- [x] Every production scan has an invertible covariance model.
- [x] Covariance condition numbers and regularization strengths are documented.
- [x] Region-level structure is either modeled or shown to be negligible.
- [x] The model is frozen before examining candidate acceptance changes.

### Completion notes

Status: Complete - PASS for the frozen null covariance; tail and candidate-use
qualifications are recorded in R1, R2, R5, R6, and R7.  
Date completed: 2026-06-13  
Files produced: `signed_refit_noise_model.h5`,
`signed_refit_noise_model_report.md`, `signed_refit_noise_model.py`,
`test_signed_refit_noise_model.py`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_noise_model.py`; `conda run -n sensei_charge_traps_new python
signed_refit_noise_model.py --validate-only`; `conda run -n
sensei_charge_traps_new python -m unittest test_signed_refit_noise_model.py
test_signed_refit_controls.py test_signed_refit_manifest.py`. Noise-model
version: `signed-refit-noise-v2`.  
Results: Extracted all 65,536 fixed control curves from the 477-image frozen
manifest. Built 2,944 covariance matrices for 23 temperatures x 4 quadrants x
32 detector regions. Each covariance uses 384 training controls; all 128
validation controls per cell remain unused for Step 4. The frozen estimator
removes each pair's median offset and the regional null template, winsorizes at
5 robust sigma, applies Oracle Approximating Shrinkage, and enforces a relative
eigenvalue floor of `1e-8`. Condition numbers range from 1.09 to 613 (median
10.7); shrinkage ranges from 0.00593 to 0.911 (median 0.0599). Median regional
per-dwell sigma is 35.1 electrons. The median maximum absolute off-diagonal
correlation is 0.517, demonstrating that a diagonal or scalar error model would
discard material structure. Noise-model SHA-256:
`d07dfec56bc8b5cad98282fe7a1c3c2fd3e5c157af660325338b7cd87535f39a`.
All ten manifest/control/noise-model tests passed.  
Problems or deviations: Some regional covariance matrices contain very strong
correlations (maximum 0.969). The R7 split-coordinate diagnostic found that
these modes reproduce in disjoint controls (median full correlation-matrix
agreement 0.774; 100% same sign for stored `|rho|>=0.8`; 99.7% retain
held-out `|rho|>=0.5`). They are persistent detector-coordinate/row-response
structure, not an unmodeled scalar offset from one image. Robust and classical
covariance differ strongly in a minority of cells (maximum fractional
Frobenius difference 0.978), while their median difference is negligible.
The permanent R5 guard gives null-template pump projection
`Delta chi2 max=0.121`, conditional `|z| max=0.347`.  
Decision: Acceptance gate passed. Freeze `signed-refit-noise-v2`. Candidate
coordinates map to the exact `(temperature, quadrant, 4x8 detector region)`
covariance on that scan's sorted `dtph` grid. Do not use held-out validation
curves until Step 4.  

---

## Step 4: Validate noise-model closure

### Objective

Demonstrate on held-out control pairs that the calibrated covariance produces
valid residual and test-statistic distributions.

### Tasks

- [x] Whiten held-out control curves with their assigned covariance matrices.
- [x] Check whitened residual mean, width, tails, and remaining correlations.
- [x] Stratify closure by temperature, quadrant, region, `dtph`, and brightness.
- [x] Compare empirical constant-model chi-square with its expected distribution.
- [x] Measure false-positive rates for trial pump-curve statistics.
- [x] Investigate the 200, 203, 207, and 210 K behavior separately.
- [x] Define any required covariance inflation or empirical tail correction.
- [x] Repeat validation after corrections without refitting the validation sample.

### Outputs

- `signed_refit_noise_closure.md`
- Diagnostic figures under `figures/signed_refit_noise/`

### Acceptance criteria

- [x] Whitened residual widths are acceptably close to one.
- [x] Residual correlations are acceptably small or explicitly modeled.
- [x] Empirical tail probabilities are stable across major strata.
- [x] Any remaining mismatch is quantified and propagated into threshold
      calibration.

### Completion notes

Status: Complete - PASS for covariance widths, with documented non-closure of
nominal chi-square tails; empirical Step 6 calibration is mandatory.  
Date completed: 2026-06-13  
Files produced: `signed_refit_noise_closure.npz`,
`signed_refit_noise_closure.md`, `signed_refit_noise_closure.py`,
`test_signed_refit_noise_closure.py`,
`diagnose_warm_noise_correlation.py`,
`figures/signed_refit_noise/closure_global_distributions.png`, and
`figures/signed_refit_noise/closure_by_temperature.png`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_noise_closure.py`; `conda run -n sensei_charge_traps_new python
signed_refit_noise_closure.py --validate-only`; `conda run -n
sensei_charge_traps_new python -m unittest test_signed_refit_noise_closure.py
test_signed_refit_noise_model.py test_signed_refit_controls.py
test_signed_refit_manifest.py`. Closure version:
`signed-refit-noise-closure-v2`.  
Results: Evaluated 376,832 held-out curves containing 7,815,168 whitened
coordinates. Global whitened mean is -0.0008 and width is 1.0265. Quadrant
widths range from 1.022 to 1.033; temperature widths range from 1.002 to 1.060.
Individual covariance-cell width p05/median/p95 is 0.986/1.019/1.063. The
global constant-model `p<0.05` rate is 7.72%, the global `|z|>3` rate is
0.472%, and median reduced chi-square is 0.9925. The maximum warm-scan residual
correlations are 0.041 at 200 K, 0.057 at 203 K, 0.054 at 207 K, and 0.028 at
210 K, all below the predefined 0.10 limit. A preliminary 256-point
profile-over-`tau` null scan exceeds `delta_chi2=11.83` for 1.94% of held-out
controls; its empirical 95th/99th/99.9th percentiles are
9.205/13.839/24.136. Closure-artifact SHA-256:
`2eccb226572d7f8d680b70d21c58343c0393a7d18db10f8b91aac00f11a61615`.
All 12 manifest/control/noise-model/closure tests passed.  
Problems or deviations: Nominal analytical tails do not close:
`p<0.05=7.72%` and `p<0.01=2.28%`, with 183 K at width 1.060 and
`p<0.05=9.22%`. The superseded v1 closure failed at 200, 203, 207, and
210 K with correlations 0.227, 0.223, 0.287, and 0.188. Diagnostic comparison
of frozen OAS, robust sample, classical sample, and quadrant-pooled covariances
showed that covariance choice did not remove the mode. A few enormous,
repeatable pump-like held-out curves dominated it; trimming the largest 1% of
curve norms reduced every warm correlation below 0.04. The upstream candidate
halo was therefore corrected from 8 to the independently meaningful 20-pixel
deferred-charge scale, and Steps 2-4 were regenerated. The final v2 validation
sample was not used to fit the covariance or alter the Step 4 thresholds. The
later R6 remediation split each 128-control validation cell 64/64, calibrating
temperature covariance scales of 1.0000-1.0760 on one half and obtaining
evaluation widths 0.984-1.050 on the other. Analytical tails remain nonuniform,
so Step 6 must use the measured scan-stratified empirical trial distribution
rather than treating `delta_chi2=11.83` as universal 3 sigma.  
Decision: Freeze `signed-refit-noise-closure-v2` as the raw closure artifact.
Use the R6 temperature scale for candidate parameter covariance and the Step 6
empirical null for detection; do not claim nominal chi-square tail closure.  

---

## Step 5: Implement the profile-`tau` intensity fitter

### Objective

Fit the signed physical model robustly and derive valid, potentially asymmetric
uncertainties on `tau`.

### Model

```text
I(t) = 3000 A [exp(-t/tau) - exp(-8t/tau)] + I0
```

At fixed `tau`, solve the signed amplitude `A` and pedestal `I0` by generalized
linear least squares using the calibrated covariance.

### Tasks

- [x] Scan a sufficiently dense grid in `log(tau)`.
- [x] Solve `A` and `I0` analytically at each grid point.
- [x] Refine the profile minimum numerically if useful.
- [x] Support scan-dependent covariance matrices.
- [x] Store the complete likelihood or chi-square profile.
- [x] Derive asymmetric profile intervals for `tau`.
- [x] Flag boundary-limited and multimodal profiles.
- [x] Preserve fitted amplitude sign and significance.
- [x] Add synthetic unit tests covering positive/negative amplitudes, pedestal,
      scan boundaries, covariance, and low signal.
- [x] Compare results with the current nonlinear `curve_fit` implementation.

### Outputs

- Profile fitter in `dipole.py` or a focused fitting module
- Unit tests
- `signed_refit_profile_fitter_validation.md`

### Acceptance criteria

- [x] Synthetic fitted `tau` is unbiased over the relevant grid.
- [x] Profile intervals have approximately correct empirical coverage.
- [x] Results are insensitive to initial guesses.
- [x] Boundary and multimodal cases are identified rather than assigned misleading
      Gaussian errors.

### Completion notes

Status: Complete - PASS for profile algebra and model-conditional validation;
empirical candidate-error use is qualified by R1/R3.  
Date completed: 2026-06-13  
Files produced: `signed_refit_profile_fitter.py`,
`test_signed_refit_profile_fitter.py`,
`signed_refit_profile_fitter_validation.py`,
`signed_refit_profile_fitter_validation.npz`, and
`signed_refit_profile_fitter_validation.md`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_profile_fitter_validation.py`; `conda run -n
sensei_charge_traps_new python
signed_refit_profile_fitter_validation.py --validate-only`; `conda run -n
sensei_charge_traps_new python -m unittest
test_signed_refit_profile_fitter.py test_signed_refit_noise_closure.py
test_signed_refit_noise_model.py test_signed_refit_controls.py
test_signed_refit_manifest.py`; and `conda run -n sensei_charge_traps_new
python -m py_compile signed_refit_profile_fitter.py
signed_refit_profile_fitter_validation.py
test_signed_refit_profile_fitter.py`. Profile-fitter version:
`signed-refit-profile-tau-v1`; validation version:
`signed-refit-profile-validation-v1`.  
Results: Implemented a global 801-point log-`tau` profile over
`0.1*min(dtph)` to `10*max(dtph)`, followed by bounded continuous refinement.
At each `tau`, signed amplitude and pedestal are solved exactly by generalized
least squares with the frozen scan/region covariance and null template. The
result retains the complete `chi2`, amplitude, pedestal, and conditional
amplitude-error profiles; signed amplitude significance; constant-model
improvement; asymmetric `Delta chi2=1` limits; and explicit boundary and
competing-minimum flags. Candidate coordinates can be mapped directly to the
exact v2 `(temperature, quadrant, region)` calibration with its noise-model
SHA-256.  
The original validation used 300 realizations in each of six predefined scenarios (1,800
curves total) spanning `tau=0.0003-0.3 s`, both signs, nonzero pedestals, warm
and cold temperatures, both dwell grids, and real frozen regional covariance
matrices. Aggregate 68% profile coverage is 67.39%; individual scenario
coverage is 64.0-70.0%; maximum absolute median bias is 0.0068 dex; sign
recovery and two-sided interval rates are 100%. This 67.39% result is explicitly
model-conditional because the draws use the same Gaussian covariance as the
fitter. The R3 replacement injected 1,152 signals onto untouched real held-out
residual curves, including binomial pumping and excess pair-shot draws.
Characterization-eligible fits covered at 68.60%; fitted-amplitude quartile
closure widths were 0.988-0.998. Weak `|A|=0.03` curves passed detection only
0.78% of the time and are classified as non-characterizable rather than given
a lifetime for Step 10. The maximum fitted-`tau`
difference between 401- and 1201-point grids after refinement is
`5.74e-7 dex`. An out-of-window long-`tau` injection is upper-bound limited,
and a deterministic low-signal realization is flagged with four competitive
modes rather than a Gaussian error. The best-start full-covariance nonlinear
fit agrees with the profile minimum to a median `4.19e-7 dex`, but the nonlinear
fit's p95 spread across starting guesses is 5.475 dex. The current diagonal
nonlinear result differs from the covariance profile by up to 0.081 dex at p95.
All 19 upstream/profile tests passed. Validation-artifact SHA-256:
`df7678d1cf4e0ce3352a32d66e1653a50c69f256144004417c02da21a6b165fe`.  
Problems or deviations: The conditional amplitude error is evaluated at the
best `tau`; the complete profile and `delta_chi2_vs_constant` are retained for
the empirical detection calibration rather than treating that conditional
error as a final catalog significance. The `tau` search bounds are intentionally
finite. Fits whose profile reaches either bound or lacks a `Delta chi2=1`
crossing are marked boundary-limited and must not be converted to symmetric
errors. No candidate threshold or false-positive interpretation is assigned in
this step.  
Decision: Acceptance gate passed. Freeze
`signed-refit-profile-tau-v1` and
`signed-refit-profile-validation-v1` for the fixed-covariance algebra. Candidate
uncertainties must use `signed-refit-profile-tau-signal-variance-v1`, the R6
scale, and the R1 acceptance gate.  

---

## Step 6: Calibrate dipole-detection significance

### Objective

Replace the assumed `delta_chi2=11.83` interpretation with an empirical
false-positive calibration that includes the search over `tau`.

### Tasks

- [x] Define the profile improvement over the constant model.
- [x] Run the full profile fit on held-out control pairs.
- [x] Measure the null statistic distribution on every scan grid.
- [x] Quantify the look-elsewhere effect from scanning `tau`.
- [x] Test whether thresholds must vary by temperature, quadrant, or region.
- [x] Include structured controls such as horizontal pairs and near-defect sites.
- [x] Select candidate thresholds from an explicit target false-positive rate.
- [x] Record both the statistic and calibrated empirical p-value for each fit.

### Outputs

- `signed_refit_detection_calibration.npz`
- `signed_refit_detection_calibration.md`

### Acceptance criteria

- [x] The selected threshold has a measured null false-positive rate.
- [x] The rate is stable across important detector and acquisition strata.
- [x] The threshold definition does not rely on an invalid Wilks-theorem claim.
- [x] The target catalog-level false-positive budget is stated.

### Completion notes

Status: Complete - PASS; STOP at acceptance gate  
Date completed: 2026-06-13  
Files produced: `signed_refit_detection_calibration.npz`,
`signed_refit_detection_calibration.md`,
`signed_refit_detection_calibration.py`, and
`test_signed_refit_detection_calibration.py`. The reusable Step 5 fitter now
also provides vectorized full-grid profile statistics through
`ProfileTauFitter.batch_profile_statistic`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_detection_calibration.py`; `conda run -n
sensei_charge_traps_new python
signed_refit_detection_calibration.py --validate-only`; `conda run -n
sensei_charge_traps_new python -m unittest
test_signed_refit_detection_calibration.py
test_signed_refit_profile_fitter.py test_signed_refit_noise_closure.py
test_signed_refit_noise_model.py test_signed_refit_controls.py
test_signed_refit_manifest.py`; and `conda run -n sensei_charge_traps_new
python -m py_compile signed_refit_detection_calibration.py
signed_refit_profile_fitter.py
test_signed_refit_detection_calibration.py`. Calibration version:
`signed-refit-detection-calibration-v1`.  
Results: Defined the detection statistic as the maximum generalized-least-
squares `delta chi2` improvement over the constant model across the complete
frozen 801-point log-`tau` grid. The Step 4 validation sites were divided
without examining their curves into 8,192 calibration and 8,192 evaluation
sites, preserving 64 controls from each `(quadrant, region)` in each half. For
each temperature, the threshold is the lowest observed calibration statistic
whose add-one finite-sample empirical p-value is at most 0.001; with 8,192
references this is normally the seventh-largest statistic. The thresholds vary
from 18.336 to 49.786, confirming that one universal `delta chi2` threshold is
not appropriate. The output stores every null statistic and its empirical
p-value, plus temperature references and production lookup functions
`empirical_p_value` and `passes_detection_threshold`.  
The independent ordinary-null evaluation FPR is 0.0812% globally, below the
0.1% target. Temperature FPRs range from 0.0122% to 0.1587%; quadrant rates
range from 0.0637% to 0.1189%; region rates range from 0.0170% to 0.1698%.
All remain within the predefined 0.30%, 0.20%, and 0.40% stability ceilings.
The old universal `delta_chi2=11.83` rule accepts 1.972% of the same independent
ordinary nulls, about 24 times the calibrated global rate. Two of 8,192
ordinary evaluation sites pass at least four temperatures; the one-sided 95%
upper bound projects to 7.17 such sites among 9,333 preliminary candidates.  
Structured stress tests used 375 clean coordinates that repeatedly triggered
the opposite-sign finder horizontally and 15,420 vertical pairs outside the v2
masks but within five pixels of persistent-defect masks. Near-defect FPR is
0.0753%, with one site passing at least four temperatures. Horizontal-trigger
FPR is higher at 0.8464%, with five sites passing at least four temperatures,
but remains below the predefined 1% Step 6 stress ceiling. All 25
manifest/control/noise/profile/detection tests passed. Calibration-artifact
SHA-256:
`74512e4b16778c874e41b59abbab8697fa0161be575cc3d865502e71c8591fbf`.  
Problems or deviations: A temperature threshold cannot be described as a
universal Gaussian-sigma or chi-square cut because `tau` is undefined at zero
amplitude and the profile searches many correlated `tau` values. Only empirical
p-values from the stored reference distribution are valid. The explicit
catalog budget is a per-temperature intensity-fit budget: `p <= 0.001` permits
at most 214.7 expected ordinary-null false temperature fits among 9,333 sites x
23 temperatures. It is not yet an end-to-end false-trap guarantee because the
candidate finder enriches structured backgrounds. In particular, the elevated
horizontal-trigger rate must be included when Step 7 chooses finder symmetry,
lobe, and persistence requirements. The structured coordinates are adversarial
stress samples selected from the same images, not independent random nulls.  
Decision: Acceptance gate passed. Freeze
`signed-refit-detection-calibration-v1`, require temperature-specific empirical
p-values rather than `delta_chi2=11.83`, and stop before Step 7 as requested.  

---

## Step 7: Calibrate the candidate finder

### Objective

Choose finder settings that maximize completeness without allowing the enlarged
candidate list to dominate the catalog's false-positive budget.

### Tasks

- [x] Compare robust and legacy image-noise estimators.
- [x] Scan lobe requirements, including no symmetry cut and relaxed alternatives.
- [x] Test requiring both lobes separately above threshold.
- [x] Compare persistence requirements of at least two versus at least three
      distinct `dtph` detections.
- [x] Add pair charge-conservation and deferred-trail diagnostics.
- [x] Inject synthetic dipoles into real images or control-pair curves.
- [x] Run the complete finder plus profile fitter on null controls.
- [x] Produce completeness-versus-purity curves by temperature and amplitude.
- [x] Freeze one finder operating point before catalog production.

### Outputs

- `signed_refit_finder_calibration.md`
- Finder configuration stored with the pipeline

### Acceptance criteria

- [x] Finder completeness is measured on injections.
- [x] End-to-end false characterization is measured on null controls.
- [x] The selected operating point has a documented scientific tradeoff.
- [x] Finder settings are not chosen by maximizing final trap count.

### Completion notes

Status: Complete - PASS.  
Date completed: 2026-06-13  
Files produced: `signed_refit_finder.py`,
`signed_refit_finder_calibration.py`, `signed_refit_finder_calibration.npz`,
`signed_refit_finder_calibration.md`, `signed_refit_finder_config.json`,
`test_signed_refit_finder.py`,
`test_signed_refit_finder_calibration.py`,
`figures/signed_refit_finder/completeness_purity_tradeoff.png`, and
`figures/signed_refit_finder/selected_completeness_by_temperature.png`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_finder_calibration.py`; `conda run -n
sensei_charge_traps_new python signed_refit_finder_calibration.py
--validate-only`; `conda run -n sensei_charge_traps_new python -m unittest
test_signed_refit_finder.py test_signed_refit_finder_calibration.py`.
Calibration version: `signed-refit-finder-calibration-v2`; finder version:
`signed-refit-finder-v1`.  
Results: Scanned six fixed configurations over all 477 selected images and all
23 temperatures. The legacy histogram sigma is 5.3-8.8% larger than the robust
row-subtracted MAD sigma. Binomial pumping injections used 2,048 clean real
residual coordinates at each of 125, 145, 170, 183, 200, and 210 K, six
amplitudes (`A=0.03-0.80`), and four tau values (`0.3 ms-0.3 s`). The selected
configuration recovers 94.315% of injections whose sampled expected peak is at
least four robust image sigma; temperature-stratified strong-signal
completeness is 88.708-98.535%. The selected complete finder-plus-profile rates
are 0/376,832 ordinary curves, 0/8,625 vertical curves at horizontal-trigger
coordinates, 0/354,660 near-defect curves, and 10/8,625 (0.1159%) true
horizontal-axis negative-control curves. Their one-sided 95% upper bounds are
0.0008%, 0.0347%, 0.0008%, and 0.1966%. The selected union contains 8,241 sites,
compared with 9,329 for the permissive robust product rule and 5,171 for the
legacy reference; count was not used by the selection rule. Artifact SHA-256:
`beb5983a12f7a2fb09b5e0ec3f6d2f453e18cae5e853878c33859c1e33f121fe`.  
Problems or deviations: The v1 pilot double-counted `N_PUMPS` in the injection
probability, clipping signals to full transfer; it was rejected and superseded.
After correcting that unit error, the original `A>=0.10` completeness gate was
physically impossible because `A=0.10` peaks near 195 e- while image-level
thresholds are typically 420-1,050 e-. Version v2 pre-registers a dimensionless
sampled peak-SNR gate and adds `A=0.80` so every representative temperature has
strong injections. Ordinary controls were masked away from the preliminary
candidate union in Step 2, so their zero rate is conditional; near-defect and
horizontal controls carry the adversarial check. The horizontal-axis profile
uses the vertical covariance/threshold as a stress test, not an independently
calibrated horizontal p-value. The 20-row trail-isolation veto was not selected:
it lowers strong-signal completeness to 66.756%.  
Decision: Acceptance gate passed. Freeze robust MAD noise, require both lobes
separately above `2.5 sigma`, require relative lobe-magnitude mismatch
`<=0.50`, and require persistence in at least two distinct dwell images. Do not
apply the trail-isolation diagnostic as a veto. The frozen configuration is
`signed_refit_finder_config.json`. Stop before Step 8.  

---

## Step 8: Define orientation and ambiguous-site policy

### Objective

Use the signed amplitude to distinguish a consistent single trap from noise,
overlapping traps, or dual-response behavior.

### Tasks

- [x] Measure amplitude sign and significance at every accepted temperature.
- [x] Define sign consistency using only significant amplitudes.
- [x] Inspect sign-changing examples visually.
- [x] Compare sign-change rates in injections, candidates, random controls, and
      structured controls.
- [x] Decide whether dual-response sites are excluded or published separately.
- [x] Define handling of temperatures with insignificant amplitude.
- [x] Freeze the policy before the SRH fit.

### Outputs

- `signed_refit_orientation_policy.md`
- Per-site orientation classification in the fit artifact

### Acceptance criteria

- [x] A single-trap SRH catalog cannot silently combine opposite orientations.
- [x] The policy has measured signal efficiency and null rejection.
- [x] Ambiguous and dual-response classifications remain available for auditing.

### Completion notes

Status: Complete - PASS after correcting the end-to-end null conditioning.  
Date completed: 2026-06-13  
Files produced: `signed_refit_orientation.py`,
`signed_refit_orientation_validation.py`,
`signed_refit_orientation_candidates.npz`,
`signed_refit_orientation_validation.npz`,
`signed_refit_orientation_policy.json` (frozen),
`signed_refit_orientation_policy.md`, `test_signed_refit_orientation.py`,
`test_signed_refit_orientation_validation.py`,
`figures/signed_refit_orientation/sign_changing_candidate_examples.png`, and
`figures/signed_refit_orientation/orientation_signal_null_rates.png`.  
Commands/version: `conda run -n sensei_charge_traps_new python -m unittest
test_signed_refit_orientation.py test_signed_refit_orientation_validation.py`;
`conda run -n sensei_charge_traps_new python -m py_compile
signed_refit_orientation.py signed_refit_orientation_validation.py
test_signed_refit_orientation.py test_signed_refit_orientation_validation.py`;
`conda run -n sensei_charge_traps_new python
signed_refit_orientation_validation.py`; `conda run -n
sensei_charge_traps_new python signed_refit_orientation_validation.py
--validate-only`. Policy version:
`signed-refit-orientation-v2`; validation version:
`signed-refit-orientation-validation-v2`.  
Results: Independently rebuilt the exact Step 7 finder union (8,241 sites) and
profiled all sites over 23 temperatures using the frozen Step 6 detection
statistic. The strict rule ignores insignificant temperatures, requires at
least four significant temperatures, and excludes every accepted opposite-sign
conflict. A vertical pair sharing either lobe pixel with the independently
frozen persistent-horizontal morphology is classified
`structured_background_overlap`. The result is 3,313 single-orientation
candidates, 419 one-sign-conflict ambiguous sites, 467 dual responses, 488
insufficient sites, 3,552 with no significant temperature, and 2 structured
overlaps. On 512 real-residual injections, all 512 had at least four detected
true-signal temperatures; correct single-orientation efficiency is 99.609%,
accepted active-fit sign accuracy is 100.000%, and no injection overlaps the
horizontal morphology. The complete vertical finder -> profile -> orientation
chain leaves 0/16,384 ordinary, 0/375 horizontal-trigger, and 0/15,420
near-defect controls as single orientation. Validation artifact SHA-256:
`ca98e27569103373c2d8ba3fe15dba1931788d4777b87f19b93daebf513fcf6e`.  
Problems or deviations: Validation v1 incorrectly applied its 0.1% production
null ceiling to every structured coordinate without requiring entry through
the frozen vertical finder. That made the intentionally horizontal-selected
stress class appear to fail at 38/375, even though those sites are not vertical
catalog candidates. The corrected v2 validation retains that important raw
result: orientation consistency alone rejects 57/95 horizontal-axis sites with
at least four significant temperatures, while 38 remain coherent. Direction
must therefore come from morphology, not sign. In the actual vertical chain,
only 2/375 horizontal-trigger coordinates enter the Step 7 union and neither
has a significant orientation. Exactly two vertical candidates share a pixel
with persistent horizontal morphology; neither is single-orientation eligible
and both remain auditable as structured background. Representative sign-changing
examples were visually inspected; they show coherent temperature bands or
isolated conflicts and no silent sign conversion. Candidate amplitudes here use
the null-covariance detection fit only for policy calibration and are not Step 9
definitive fits.  
Decision: Acceptance gate passed. Freeze
`signed_refit_orientation_policy.json`: only empirically significant
temperatures contribute; at least four are required; any accepted opposite sign
excludes the site from a single-trap SRH fit; two or more accepted fits of each
sign are `dual_response`; and persistent-horizontal pixel overlap is
`structured_background_overlap`. Step 9 must recompute the classifications
from its definitive signal-dependent accepted-temperature mask and may not
restore ambiguous, dual, or structured sites to the single-trap class. Stop
before Step 9.  

---

## Step 9: Regenerate per-temperature intensity fits

### Objective

Produce the definitive per-temperature signed fit artifact using the frozen
manifest, covariance model, profile fitter, detection calibration, finder, and
orientation policy.

### Tasks

- [x] Run the frozen candidate finder.
- [x] Extract signed spectra only from manifest images.
- [x] Assign each spectrum its calibrated covariance model.
- [x] Run the profile fitter.
- [x] Apply the empirically calibrated detection and quality criteria.
- [x] Store profiles, asymmetric intervals, amplitudes, offsets, signs, empirical
      p-values, boundary flags, and provenance.
- [x] Produce cutflows by temperature, quadrant, and rejection reason.
- [x] Compare representative accepted and rejected curves visually.

### Outputs

- Versioned signed coordinate artifact
- Versioned signed spectra artifact
- Versioned per-temperature profile-fit artifact
- `signed_refit_intensity_cutflow.md`

### Acceptance criteria

- [x] No stale cache can be mistaken for the new artifact.
- [x] Every accepted fit can be traced to its image and calibration inputs.
- [x] Cutflows and fit diagnostics show no unexplained temperature collapse.
- [x] Null-control false-positive rates remain consistent with calibration.

### Completion notes

Status: Complete - PASS  
Date completed: 2026-06-14  
Files produced: `signed_refit_candidates_v1.npz`,
`signed_refit_spectra_v1.h5`, `signed_refit_intensity_fits_v1.h5`,
`signed_refit_intensity_cutflow.md`,
`figures/signed_refit_intensity/accepted_intensity_fits.png`,
`figures/signed_refit_intensity/rejected_intensity_fits.png`,
`signed_refit_intensity_pipeline.py`,
`signed_refit_candidate_variance_closure_v2.npz`,
`signed_refit_candidate_variance_closure_v2.md`, and
`test_signed_refit_intensity_pipeline.py`.  
Commands/version: `conda run -n sensei_charge_traps_new python
signed_refit_candidate_variance_closure.py`; `conda run -n
sensei_charge_traps_new python signed_refit_intensity_pipeline.py`; `conda run
-n sensei_charge_traps_new python signed_refit_intensity_pipeline.py
--validate-only`; `conda run -n sensei_charge_traps_new python -m unittest
discover -p "test_signed_refit*.py"`. Pipeline version:
`signed-refit-intensity-v1`; 63 signed-refit tests passed.  
Results: The frozen finder reproduced 8,241 sites
(`2390/2495/1611/1745` by quadrant). The artifact stores 51,070 complete
profile-likelihood curves and accepts 38,000 characterizable
candidate-temperature fits. The final orientation policy leaves 2,703
single-trap sites: 1,743 positive and 960 negative. It preserves 411 ambiguous,
341 dual-response, and two structured-background sites; 254 sites that would
have become single after dropping a conflicting temperature remain excluded by
the Step 8 lock. The independent ordinary-null empirical FPR is 0.0929%
overall, with a maximum 0.1953% at one temperature, consistent with Step 6.
The R1 two-fold cross-fit closes at width 0.993 overall and 0.942-1.022 by
fitted-amplitude quartile, with width spread 0.080.  
Problems or deviations: The previous coordinate "hash" reduced to detector
checkerboard parity and was replaced by an avalanched 64-bit coordinate hash
before the R1 cross-fit. Also, a stored "equivalent" observed null threshold is
not mathematically equivalent between adjacent null order statistics; Step 9
therefore uses the frozen rule directly by computing finite-sample empirical
`p<=0.001`. The 130-140 K characterization efficiency is low because 1,814,
3,145, and 2,627 profiles respectively do not bracket a two-sided lifetime
interval, predominantly at the upper lifetime boundary. These detections remain
stored but do not receive a lifetime for Step 10. No profile-fit exceptions
occurred. The 400 ADU/e- global gain remains provisional per Decision 3.  
Decision: Acceptance gate passed. Freeze the versioned Step 9 artifacts and
their hashes. Step 10 may consume only `single_trap_eligible` sites and accepted
temperature fits from `signed_refit_intensity_fits_v1.h5`, while applying the
pre-registered 160/170 K acquisition-family systematic. Stop before Step 10.  

---

## Step 10: Fit trap energy and cross-section

### Objective

Fit the simple p-channel SRH model to accepted `tau(T)` measurements without
using uncalibrated outlier rejection or intrinsic scatter.

### Model

```text
tau(T) = exp(Etrap / k_B T) / [sigma * v_th(T) * N_v(T)]
```

### Tasks

- [x] Require the frozen minimum number of accepted temperatures.
- [x] Use profile-derived `tau` likelihoods or asymmetric intervals where
      practical.
- [x] Use the p-channel `N_v` convention and define energy from the valence edge.
- [x] Fit the simple SRH model first.
- [x] Record goodness of fit without repeatedly overwriting the criterion.
- [x] Flag extrapolation range and leverage of individual temperature points.
- [x] Examine residuals versus temperature, acquisition family, amplitude,
      pedestal, quadrant, and orientation class.
- [x] Run the frozen fit both with all temperatures and with 160/170 K removed.
      Record `Delta E`, `Delta log(sigma)`, goodness-of-fit, and leverage for
      every site; the no-160/170 K result is a mandatory acquisition-family
      systematic, not an optional diagnostic.
- [x] Before selecting a headline result, pool standardized full-fit residuals
      at 160/170 K across sites. Classify `dp_scan1` as anomalous if a
      site-bootstrap 99% interval excludes zero and the absolute pooled median
      exceeds 0.10 dex, or if removing 160/170 K shifts the population-median
      energy by more than its combined one-sigma uncertainty. If either trigger
      fires, use the no-160/170 K fit as primary and retain the full fit as the
      systematic variant.
- [x] Investigate systematic high-temperature residuals before relaxing the model.
- [x] Keep failed or non-SRH sites as explicit classifications.

### Outputs

- Versioned SRH fit artifact
- `signed_refit_srh_validation.md`

### Acceptance criteria

- [x] Energy and cross-section uncertainties use the calibrated `tau` information.
- [x] Selection criteria are fixed and represented once in code.
- [x] High-temperature deviations are either explained, modeled, or retained as
      a documented failure mode.
- [x] No intrinsic scatter is tuned on the same data and then treated as known in
      a chi-square p-value.
- [x] The pre-registered 160/170 K acquisition-family comparison and promotion
      rule are reported before any family-based exclusion is made.

### Completion notes

Status: PASS  
Date completed: 2026-06-14  
Files produced: `signed_refit_srh_pipeline.py`;
`test_signed_refit_srh_pipeline.py`; `signed_refit_srh_fits_v1.h5`;
`signed_refit_srh_validation.md`; and the three diagnostics under
`figures/signed_refit_srh/`.  
Commands/version: `signed-refit-srh-v1`;
`conda run -n sensei_charge_traps_new python signed_refit_srh_pipeline.py`;
`conda run -n sensei_charge_traps_new python signed_refit_srh_pipeline.py
--validate-only`; and `conda run -n sensei_charge_traps_new python -m unittest
discover -p "test_signed_refit*.py" -v` (67 tests passed).  
Results: 2,703 Step 9 single-trap sites entered. The full fit produced 1,195
SRH-consistent sites. The no-160/170 K variant left 2,407 successful fits,
including 1,287 SRH-consistent sites; 264 sites then had fewer than four
temperatures. The family residual median was +0.0360 dex with a 99% interval
[+0.0298,+0.0477] dex, below the frozen 0.10 dex magnitude trigger. The
population-median energy shifted +0.0029 eV, exceeding its combined 0.0010 eV
one-sigma uncertainty, so the pre-registered population trigger promoted the
no-160/170 K variant. The primary catalog retains 1,120 explicit non-SRH fits,
28 parameter-boundary fits, and four fits outside profile support. No
high-temperature median residual exceeded 0.10 dex. Per-site full/no-family
energy, `ln(sigma)`, p-value, leverage, and their systematic differences are
stored. The artifact SHA-256 is
`4330a5ada27cd5da528426aa2f42a8d297de0ce0b61377ab03fdd11419de6e1d`.  
Problems or deviations: The first production attempt exposed a nuisance-profile
optimizer branch-jump while finding confidence limits. The profiler was anchored
to the fitted branch, constrained limits now terminate at the registered
physical bounds, and synthetic recovery tests were added before regeneration.
For transparency, 2,127/25,703 primary fitted temperature points place the
pump maximum outside the sampled dwell window, and maximum site leverage reaches
0.998; these flags remain stored rather than being used for post-hoc rejection.  
Decision: Acceptance gate passed. Freeze `signed-refit-srh-v1`, use
`no_160_170` as the primary Step 10 result, retain the all-temperature fit as
the acquisition-family systematic, and stop before Step 11. The 951 legacy
well-behaved sites excluded before Step 10 remain a required follow-up
population audit rather than grounds for changing the frozen Step 9 policy
without validation.  

---

## Step 11: Validate purity and completeness end to end

### Objective

Measure catalog false-characterization and trap-recovery probabilities using the
final frozen analysis chain.

### Tasks

- [ ] Re-run random-pair controls.
- [ ] Re-run horizontal and other structured null controls.
- [ ] Inject signed dipole curves spanning amplitude, pedestal, `tau`, temperature,
      quadrant, region, and orientation.
- [ ] Include the empirical covariance and any pumping-variance model.
- [ ] Pass injections through finder, intensity fit, orientation policy, and SRH fit.
- [ ] Measure false characterization at every selection stage.
- [ ] Measure completeness in `(E, sigma)` and `tau(135 K)` space.
- [ ] Check closure against observed per-temperature acceptance.
- [ ] Quantify systematic variants of thresholds and covariance assumptions.

### Outputs

- `signed_refit_end_to_end_validation.md`
- Versioned completeness grids
- Versioned decoy-control results

### Acceptance criteria

- [ ] The final catalog false-positive estimate meets the stated budget.
- [ ] Completeness is reported only where injection closure is adequate.
- [ ] Selection corrections use the same frozen pipeline as the data.
- [ ] Systematic uncertainty is propagated to population results.

### Completion notes

Status: Not started  
Date completed:  
Files produced:  
Commands/version:  
Results:  
Problems or deviations:  
Decision:  

---

## Step 12: Freeze the catalog and regenerate downstream products

### Objective

Publish one internally consistent analysis version and update every dependent
artifact from it.

### Tasks

- [ ] Freeze code commit, manifest, configuration, and random seeds.
- [ ] Save the final trap catalog with classifications and provenance.
- [ ] Regenerate `tau_at_<T>k_hist.npz`.
- [ ] Regenerate completeness and upper-limit products.
- [ ] Regenerate paper figures and quoted counts.
- [ ] Regenerate simulation inputs and affected simulation outputs.
- [ ] Update the paper's methods and limitations.
- [ ] Archive prior signed artifacts as superseded, without deleting them.
- [ ] Run a final independent consistency review.

### Outputs

- Final versioned trap catalog
- Final `tau_at_<T>k_hist.npz` files
- Updated figures, tables, paper text, and simulation inputs
- `signed_refit_release_notes.md`

### Acceptance criteria

- [ ] All published numbers trace to one catalog version.
- [ ] No legacy or stale signed cache is consumed downstream.
- [ ] Paper equations, notation, selection descriptions, and code agree.
- [ ] Known limitations are documented.
- [ ] Release notes include changes relative to both legacy and initial signed
      catalogs.

### Completion notes

Status: Not started  
Date completed:  
Files produced:  
Commands/version:  
Results:  
Problems or deviations:  
Decision:  

---

## Adversarial review remediation (Steps 1–5)

Source: `SIGNED_REFIT_STEP1_5_REVIEW.md` (2026-06-13 adversarial physics review).
These items qualify the existing "PASS" gates; do not treat Steps 1–5 as
scientifically closed until R1 is resolved.

### R1 — BLOCKING: derive a signal-dependent (pumping/shot) variance term

- [x] Build a pumping-variance model: the null control covariance only captures
      the background shot-noise floor (~35 e⁻), not the extra `Var(X)` of a real
      3000-cycle charge transfer, and `Var(X) ≠ E[X]`.
- [x] Close residuals binned by **fitted pump amplitude** (not just static control
      brightness), confirming the closure trend width 1.025→1.035 /
      `p<0.05` 7.49%→8.28% across control-brightness quartiles is removed.
- [x] Add an acceptance gate requiring amplitude-stratified closure before any τ
      uncertainty is consumed by the Step 10 SRH fit.
- Implementation: `signed_refit_variance_model.py` adds
  `C_candidate=s_T C_null + diag(phi*3000*q*(1-q) + V_pair,extra)`, with
  `q=|A|[exp(-t/tau)-exp(-8t/tau)]` and
  `V_pair,extra=max((a+b)_candidate-(a+b)_reference,0)/4`.
  `SignalDependentProfileTauFitter` iterates this covariance with the fitted
  amplitude and lifetime.
- Injection result: **PASS** on 1,152 signals injected onto untouched real
  residual curves. Characterization-eligible coverage is 68.60%; fitted-amplitude
  quartile widths are 0.988-0.998.
- Real-candidate result: **PASS** after applying the frozen Step 8 orientation
  policy plus the physical `|D_t P_c|<=1` requirement, without selecting on fit
  residuals. The original split was discovered to be detector checkerboard
  parity, so v2 uses an avalanched 64-bit coordinate hash and symmetric two-fold
  cross-fitting. Four null-fit-amplitude bins give frozen `phi` values
  `8.3117, 16.0634, 30.0039, 23.0403`. Across 9,526 eligible out-of-fold fits,
  aggregate width is 0.993; fitted-amplitude quartile widths are
  0.942-1.022, width spread is 0.080, and all pre-registered gates pass.
- Status: **REOPENED — BLOCKED (2026-06-15, R8).** The v2 closure is a *tuned*
  pass, not a validated variance model: the residual width is forced to 1.0 by
  bisecting a free per-bin `pump_overdispersion` φ. The frozen factors
  `8.31/16.06/30.00/23.04` are 8–30× the independent-binomial `3000·q(1-q)` and
  non-monotonic in amplitude — a misspecified-variance signature, not a
  finite-sample fix. (`30.0039` is the bisection node `1+99·300/1024`, not the
  cap; `MAX_OVERDISPERSION` was raised 20→100 in v2, so φ≈30 is genuinely
  data-demanded.) See `SIGNED_REFIT_STEP6_10_REVIEW.md` B1 and runbook R8. The
  earlier "COMPLETE - PASS" recorded below describes the engineering closure
  only; it does not certify the variance physics.
- Superseded engineering note: v2 (`signed_refit_candidate_variance_closure_v2.npz`)
  reaches aggregate width 0.993 with quartile widths 0.942-1.022; Step 9 stores
  its SHA-256 and chooses the variance bin from the null-covariance amplitude
  before the signal-dependent refit. Candidate `tau` uncertainties may **not**
  feed Step 10/11 population results until R8 is resolved.

### R2 — Restate Step 3/4 closure verdict

- [x] Change the headline from "PASS" to "PASS with documented non-closure of
      nominal χ² tails": held-out `p<0.05`=7.7%, `p<0.01`=2.3%, worst at warm
      scans (183 K width 1.060, `p<0.05` 9.2%). Empirical Step 6 calibration is
      therefore mandatory, not optional.
- Status: complete in the Step 3/4 completion notes,
  `signed_refit_noise_closure.md`, and its report generator.

### R3 — De-circularize the Step 5 coverage validation

- [x] Replace `multivariate_normal(0, covariance)` injection with injection onto
      **real held-out control residual curves** so coverage is tested against the
      true (heavy-tailed) noise PDF.
- [x] Relabel the 67.4% coverage as "coverage under the assumed
      covariance," and tighten the loose per-scenario coverage band (0.58–0.78).
- Status: complete. `signed_refit_variance_validation.md` reports the disjoint
  real-residual result and explicitly classifies weak/unidentified lifetimes as
  non-characterizable.

### R4 — Pre-register the 160/170 K acquisition-family systematic

- [x] 160 K and 170 K are the only `dp_scan1` points (300000 charge shifts, Feb)
      vs `temp_scan_run1` (200000, Mar–Apr) — temperature is aliased with family
      at the cold end of the Arrhenius lever arm.
- [x] Commit, before unblinding, to stratifying Step 10 SRH residuals by
      acquisition family and dropping 160/170 K as a systematic if anomalous.
- Status: complete. Step 10 now requires all-temperature and no-160/170 K fits
  for every site and contains a pre-registered residual/population-shift trigger
  for promoting the no-160/170 K result.

### R5 — CLEARED: dwell-dependent `null_template` does not bias τ

- [x] Verified by projecting all 2944 region templates onto the pump shape:
      template Δχ²-vs-constant max 0.12 (p95 0.02), conditional |z| max 0.35σ.
      Subtraction acts as benign common-mode removal.
- [x] Add the template→pump-shape projection to
      `signed_refit_noise_model_report.md` as a permanent regression guard.
- Status: complete. `validate_noise_model` now fails if maximum template
  `Delta chi2>1` or conditional `|z|>1`; observed values remain 0.121 and 0.347.

### R6 — Robust-estimator variance bias

- [x] 5σ-winsorization + OAS shrink the variance estimate low (closure width >1
      everywhere). Either justify with a measured inflation factor or fold the
      closure width into an explicit covariance scale-up.
- Status: complete. One half of each held-out cell calibrates a temperature
  scale `s_T=1.0000-1.0760`; the disjoint half closes at width 0.984-1.050.
  Candidate covariance uses `s_T C_null`. Heavy tails are not declared fixed
  and remain assigned to Step 6's empirical detection distribution. The
  observed deficit is calibrated empirically rather than attributed uniquely
  to OAS; OAS preserves the trace of the winsorized covariance.

### R7 — Smaller consistency items

- [x] Explain the up-to-0.969 null off-diagonal correlations physically
      (likely per-image common-mode), or confirm they are not image-specific.
- [x] Pin the lobe-order/sign contract `I=(image[row]−image[row−1])/2` for Step 9
      candidate extraction.
- [x] Confirm `ELECTRONIZE_SCALE=400` ADU/e⁻ is the working gain for this
      runbook (sets every chi-square).
- Correlation result: the large modes reproduce across disjoint coordinates
  (matrix-correlation median 0.774; 365 cells have stored `|rho|>=0.8`, all keep
  the same sign and 99.7% retain held-out `|rho|>=0.5`). They are consistent
  with persistent detector-coordinate/row-response structure, not a one-image
  scalar offset; the exact electronics/spatial mechanism is not identified.
- Sign result: the contract is executable in
  `signed_refit_candidate_variance_closure.py` and recorded in its artifact.
- Gain result: provisionally accepted by the analysis owner on 2026-06-13. All
  477 selected images have CSV/XML sidecars, but
  five CSVs are empty, all populated CSV rows have blank `gain1..gain4`, and all
  XMLs contain only fallback `gain=200`. No selected sidecar provides an
  image-by-image measurement supporting 400. The established MINOS global 400
  is therefore frozen provisionally for the current runbook, recorded in every
  Step 7 artifact, and must be revisited if external calibration provenance
  supplies a different value.

---

## Adversarial review remediation (Steps 6–10)

Source: `SIGNED_REFIT_STEP6_10_REVIEW.md` (2026-06-14/15 adversarial review).
These items qualify the existing "PASS" gates on Steps 6–10. Steps 6–8 stand
modulo R12/R16; Steps 9 and 10 are engineering-complete but **not** physics-closed
until R8, R9, R10, and R11 are resolved. Severity tags match the review.

### R8 — BLOCKING: φ=8–30 overdispersion is a fitted knob, not a validated model

- [ ] The candidate variance closes only because `pump_overdispersion`
      (`estimate_overdispersion_bin`) is bisected per amplitude bin to force the
      residual width to 1.0. Frozen factors are `8.31, 16.06, 30.00, 23.04` —
      8–30× the independent-Bernoulli pumping variance `3000·q(1-q)`, and
      non-monotonic in amplitude. `30.0039` in both folds is the bisection node
      `1+99·300/1024`, not the cap (`MAX_OVERDISPERSION` is 100), so φ≈30 is
      genuinely data-demanded.
- [ ] Either derive a physical correlated-trapping / per-cycle common-mode term
      that *predicts* the 8–30 excess without fitting it, or show the candidate
      residual variance closes without a per-bin free multiplier.
- [ ] Until then revert the R1 status to **BLOCKED**; τ uncertainties carrying φ
      may not be treated as validated for Step 10/11 population results.
- Relates to [[signed-refit-step1-5-review]] R1 and [[recapture-residual-masked-ab]].

### R9 — HIGH: quantify the cold-temperature τ selection bias on E/σ

- [ ] At 130–140 K, `interval_not_two_sided` removes ~90%+ of detections
      (135 K: 3,145/3,379). Lifetimes survive only where the noisy measured τ
      brackets inside the dwell window, biasing surviving cold τ short and the
      Arrhenius slope/intercept with it.
- [ ] Inject SRH-consistent traps spanning τ across the cold window edge through
      the full Step 9→10 chain; measure the induced E and ln σ bias versus the
      truncation fraction. Report it as a mandatory systematic, not a footnote.
- Relates to [[high-t-arrhenius-lean]] (cold-end counterpart of the hot lean).

### R10 — HIGH: empirically calibrate the Step 10 SRH deviance

- [ ] `signed_refit_srh_pipeline.py` uses `chi2.sf(sum_of_profile_delta_chi2,
      N-2)`. The summed profiles are non-parabolic, so this Wilks calibration is
      the exact assumption Step 6 disproved for detection.
- [ ] Build the deviance null empirically (push SRH-consistent multi-T τ sets
      through the same fitter) and classify `non_srh` against the measured
      distribution. Re-report the 47% non-SRH fraction under that calibration.

### R11 — HIGH: de-confound the 160/170 K family promotion

- [ ] 160/170 K are simultaneously the only `dp_scan1` family and the
      coldest/highest-leverage points; dropping them is degenerate between
      "family" and "lever-arm" effects.
- [ ] Replace the independent-bootstrap "combined σ" with a **paired** bootstrap
      of the per-site energy difference (the current `np.hypot` of two
      independently seeded bootstraps does not estimate the difference variance).
- [ ] Add an orthogonal family test that does not remove the cold anchor (e.g.
      a `dp_scan1`-vs-`temp_scan_run1` offset nuisance, or family-matched
      injections) before declaring the systematic resolved.
- Relates to R4 / Decision 2.

### R12 — MEDIUM: break the φ ↔ SRH-consistency coupling

- [ ] The φ that closes Step 9 widens τ intervals → smaller Step 10 deviance →
      more `srh_consistent`. The same un-validated knob drives both the R1 PASS
      and the headline consistency fraction.
- [ ] Validate the τ-error scale against the R10 deviance null (multi-temperature
      coverage), not only the per-point residual width.

### R13 — MEDIUM: resolve the 400 ADU/e⁻ gain before freezing 6–10

- [ ] Gain 400 (vs sidecar fallback 200) now sets the finder completeness gate,
      electron amplitudes, the hard `|D_t P_c| ≤ 1` cut, and shot terms.
- [ ] Obtain external calibration provenance or run Steps 7–10 at 200 as a
      systematic variant and report the catalog sensitivity. (Carryover of R7.)

### R14 — MEDIUM: explain the +2.3× cross-section and +0.012 eV legacy shift

- [ ] Attribute the common-population +0.8275 ln σ and +0.0117 eV shifts to a
      cause (gain, signed model, or SRH-constant convention) and document it;
      these propagate into simulation capture rates.

### R15 — MEDIUM: exclude degenerate fits from population E/σ

- [ ] Leverage reaches 0.998 and 8.3% of points place the pump peak outside the
      window. Define a leverage / window cut for population energy and
      cross-section medians rather than only storing the flags.

### R16 — MEDIUM: widen the structured-background purity test

- [ ] The 2.5σ finder multiplies horizontal leakage ~10× over 3σ, and only 2
      sites are caught by horizontal-morphology overlap. Add independent
      structured nulls (additional non-pumping axes, crowded-defect classes) and
      report the ≥4-temperature purity without assuming an ordinary-clean field.

### Smaller items

- [ ] L1: per-temperature thresholds are single 7th-order statistics (realized
      FPR 0.012–0.159%); consider pooling/smoothing the tail or quoting the
      realized band.
- [ ] L2: confirm `profile_parameter_interval` branch anchoring on **multimodal**
      synthetic profiles (current heuristic biases intervals narrow).

---

## Decision log

Use this section for decisions that affect multiple steps.

### Decision 1

Date: 2026-06-13  
Question: How should candidate signal-dependent variance be represented?  
Options considered: null covariance only; Poisson variance equal to signal;
independent-cycle binomial transfer plus matched excess pair-shot variance.  
Decision: Use the explicit binomial-plus-pair-shot model in
`signed_refit_variance_model.py`, with a separately calibrated null scale.  
Evidence: The signed transfer contributes `X` directly to `(a-b)/2`, so
`Var(X)=Nq(1-q)`, not `E[X]/4`. Real-residual injections close for identifiable
signals.  
Consequences: Detection continues to use the frozen Step 6 null statistic;
candidate parameter fits use the signal-dependent covariance. The Step 9
two-fold real-candidate closure is now PASS and freezes four amplitude-stratified
overdispersion factors.

### Decision 2

Date: 2026-06-13  
Question: How should the 160/170 K `dp_scan1` family alias be handled?  
Options considered: exclude up front; accept without a systematic; test/drop
under a pre-registered Step 10 rule.  
Decision: Run mandatory full and no-160/170 K fits and promote the exclusion
variant only under the frozen residual/population-shift trigger in Step 10.  
Evidence: 160/170 K are the only 300000-shift February scans and sit at a
high-leverage part of the Arrhenius fit.  
Consequences: Step 10 cannot report one SRH result without the family systematic.

### Decision 3

Date: 2026-06-13  
Question: Can the current 400 ADU/e- scale be confirmed from selected-image
calibration sidecars?  
Options considered: use per-image CSV gains; use XML default 200; retain the
established MINOS global 400 pending provenance.  
Decision: Retain 400 as the owner-approved provisional global gain for the
current runbook. Do not block Step 9 solely on missing sidecar provenance, but
keep the absolute-scale caveat explicit and rerun Steps 7-9 if a different
calibration is established.  
Evidence: No selected CSV contains a fitted gain; all XML values are fallback
200 entries.  
Consequences: The current finder, electron amplitudes, and shot terms use 400.
The XML fallback 200 is not substituted silently; a later gain change requires
regeneration of all gain-dependent artifacts.

### Decision 4

Date: 2026-06-13  
Question: Which candidate-finder operating point should feed catalog
production?  
Options considered: legacy histogram/product/symmetry; robust product without
balance; robust separate-lobe rules at 3.0 or 2.5 sigma; persistence of two or
three dwell images; optional 20-row trail isolation.  
Decision: Use robust row-subtracted MAD noise, both lobes separately above
2.5 sigma, relative lobe-magnitude mismatch at most 0.50, and persistence in at
least two distinct dwell images. Keep trail isolation as a diagnostic only.  
Evidence: Strong-signal injection completeness is 94.315% with a 0.1159%
horizontal-axis end-to-end stress rate (95% upper bound 0.1966%). The 3.0-sigma
separate-lobe alternative gives 91.164% completeness and 0.0116% horizontal
stress rate; the selected point remains below the predeclared 1% structured
ceiling while recovering more signal. The permissive product rule reaches
97.220% completeness but fails the physical separate-lobe requirement and has a
4.359% horizontal stress rate.  
Consequences: Step 9 must load `signed_refit_finder_config.json`; finder settings
must not be retuned using final trap count.

### Decision 5

Date: 2026-06-13  
Question: How should orientation conflicts and coherent non-pumping-axis
responses be handled before the SRH fit?  
Options considered: majority-sign threshold; require a fixed sign fraction;
discard isolated conflicting temperatures; exclude every accepted sign
conflict and classify persistent-horizontal overlap separately.  
Decision: Only empirical `p<=0.001` temperature fits contribute a sign. Require
at least four significant temperatures. Any accepted opposite sign excludes a
site from the single-trap class; one minority-sign fit is
`ambiguous_sign_conflict`, and at least two fits of each sign is
`dual_response`. A vertical pair sharing either lobe pixel with the frozen
persistent-horizontal morphology is `structured_background_overlap`.  
Evidence: Real-residual injections give 99.609% correct single-orientation
efficiency and 100% accepted-fit sign accuracy. The complete vertical finder,
profile, orientation, and morphology chain leaves zero single-orientation
survivors in 16,384 ordinary, 375 horizontal-trigger, and 15,420 near-defect
controls. Orientation alone is not a direction test: 38/375 intentionally
horizontal-selected sites retain a coherent raw sign, so the horizontal
morphology classification is required. Only two of 8,241 vertical candidates
share horizontal morphology, and no injection does.  
Consequences: Step 9 must load `signed_refit_orientation_policy.json`, recompute
labels from its definitive accepted-temperature mask, preserve all
classifications, and never restore ambiguous, dual-response, or structured
sites to a single-trap SRH fit.

### Decision 6

Date: 2026-06-14  
Question: How should the real-candidate variance calibration and finite-sample
detection rule be applied in the definitive intensity artifact?  
Options considered: retain one checkerboard calibration/evaluation split and a
single `phi`; tune one global factor on all candidates; use coordinate
cross-fitting with amplitude strata; apply the stored observed statistic
threshold instead of the empirical rank p-value.  
Decision: Use an avalanched coordinate hash and symmetric two-fold cross-fitting
in four bins of the null-covariance fitted amplitude. Freeze the geometric mean
fold factors `8.3117, 16.0634, 30.0039, 23.0403`. Apply detection by the exact
finite-sample empirical `p<=0.001` calculation, not by the tabulated lowest
observed passing statistic.  
Evidence: The old low-bit linear split was exactly row+column+quadrant parity.
The corrected out-of-fold closure has aggregate width 0.993, quartile widths
0.942-1.022, and width spread 0.080. Candidate statistics can lie between
adjacent null order statistics, where the empirical rank p-value and a
"lowest observed passing" threshold are not equivalent.  
Consequences: Step 9 stores the closure hash, null-fit amplitude, chosen
overdispersion factor, and empirical p-value for every attempted fit. A future
detection recalibration must preserve the rank-test definition or regenerate
Steps 8-9.

### Decision 7

Date: 2026-06-14  
Question: Which acquisition-family variant should be primary for the Step 10
energy and cross-section catalog?  
Options considered: use all temperatures; remove 160/170 K only if their pooled
residual exceeds the frozen 0.10 dex threshold; promote the no-160/170 K variant
if either pre-registered family trigger fires.  
Decision: Use the no-160/170 K fit as primary and retain the all-temperature fit
as a systematic variant.  
Evidence: The pooled 160/170 K residual was +0.0360 dex with a site-bootstrap
99% interval [+0.0298,+0.0477] dex, so the residual-magnitude trigger did not
fire. Removing those temperatures shifted the paired population-median energy
from 0.2857 to 0.2886 eV, a +0.0029 eV shift compared with a combined 0.0010 eV
one-sigma uncertainty, so the frozen population trigger did fire.  
Consequences: The primary Step 10 artifact contains 1,287 simple-SRH-consistent
sites and 1,120 explicit non-SRH sites among 2,407 successful fits. The 264
sites that fall below four temperatures after the family exclusion remain
classified as insufficient rather than being restored with a weaker criterion.
Step 11 must consume the primary links in `signed_refit_srh_fits_v1.h5` and keep
the full-fit family systematic available.

# Stage 11: Naive Efficiency Curve Closure Test

## Objective

Test whether the Method 3 detection model reproduces ("closes") the naive pooled
measured/extrapolated efficiency curve from `charge_trap_figures.ipynb`
(`figures/efficiency_tau_curve.pdf`), and attribute its odd features
(low plateau at `tau ~ 1e-4..1e-2 s`, dip near `1e-2 s`, narrow peak at
`~0.3..1 s`, hard zeros outside `~3e-5..30 s`) to specific model mechanisms.

## Background

The naive estimator pools per-(trap, temperature) points over the 2135
`n_good = 4` characterized traps: for each trap and each of the 23 campaign
temperatures, `tau(T)` is computed from the global `E`/`log_sigma` fit and the
point is tagged "measured" if that temperature has `GoodIntensityFit = True`,
else "extrapolated". Efficiency per `tau` bin = measured / total with bins
`np.geomspace(1e-7, 1e8, 75)`.

Key verified facts:

- The `fitTrapIntensity` selection (`useIntensityErr=True`) is identical to the
  Stage 08 cut set: fit converges with `tau` in `[1e-8, 1000]`, `p_value > 0.05`,
  `max(intensities) >= 3 * mean(intensity_err)`, `max(intensities) >= 3 * image_sigma`,
  `fit_tau_err / fit_tau <= 0.5` (`dipole.py:497-554` vs Stage 08 summary `cuts`).
- The Stage 01 CSV selection (`WellBehavedTrap & ~EnergyFitFailed & GoodEnergyFit`,
  `audit_hdf5_records.py:163`) matches the notebook's trap filter, so the observed
  curve is rebuilt exactly from `cache/01_records_ngood4.csv`.

## Inputs

- `cache/01_records_ngood4.csv` (Stage 01): per-trap `E_eV`, `tau_135_seconds`,
  `good_temperatures_K`.
- `cache/08_pdet_grid_tau1000_v1.h5` (Stage 08 extended-tau): `p_det(T, tau, A)` on
  `23 x 79 x 35`, plus per-grid-point `controlling_cut_fraction` and `cut_labels`.
  The extended grid is used so the prediction has no artificial 20 s cutoff.
- `cache/05_amplitude_prior_v1.npz` (Stage 05): default high-confidence depth prior
  and `P_c(T)` temperature factors.
- Reused code: `src/validation_sensitivity.py` (`read_known_traps`,
  `tau_at_temperature`, `load_stage05`, `interp_rows_by_amplitude`).

## Method

1. Rebuild the observed curve from the Stage 01 records with the notebook's exact
   binning (`geomspace(1e-7, 1e8, 75)`).
2. For each (trap, temperature) point compute `tau(T)` and look up
   `p_det(T, tau(T), A)` from the extended-tau grid (zero off-grid, Stage 09 policy),
   marginalizing amplitude over quantile-stratified draws from the Stage 05 default
   depth prior scaled by `P_c(T)`.
3. Predict the curve two ways:
   - unconditional: numerator weight per point = `mean_A p_det`;
   - conditional on characterization (the correct closure target, since every
     observed trap already passed `n_good >= 4`): numerator weight =
     `E_A[p_t * P(N_{-t} >= 3)] / E_A[P(N >= 4)]` via leave-one-out
     Poisson-binomial DP over the other 22 temperatures.
4. Decompose: per-tau-bin composition by temperature (observed and predicted), and
   per-bin controlling-cut attribution interpolated from the Stage 08 cutflow
   (off-grid points attributed to a synthetic `tau_outside_stage08_grid` category).
5. Closure metrics: per-bin pulls and chi-square over bins with `total >= 20`,
   plus feature-specific numbers (plateau mean over `1e-4..1e-2 s`, minimum in the
   dip window `3e-3..3e-2 s`, peak value, half-max crossings).

## Outputs

- `cache/11_naive_efficiency_closure_v1.h5` (curves, decompositions, per-trap diagnostics)
- `cache/11_naive_efficiency_closure_summary.json`
- `cache/figures/11_closure_overlay.png`, `11_closure_pulls.png`,
  `11_temperature_decomposition.png`, `11_cut_attribution.png`

## Required Checks

- Rebuilt observed curve visually matches `figures/efficiency_tau_curve.pdf`
  (peak ~0.98 at 0.3..1 s, plateau ~0.15, zeros outside ~3e-5..30 s).
- Interpolated "pass" controlling-cut fraction per bin is consistent with the
  unconditional predicted efficiency (internal consistency of the attribution).
- Conditional-denominator failures (`E_A[P(char)] < 1e-6` for a real trap) are
  counted and reported; fallback to unconditional weight is flagged.
- Smoke run (reduced amplitude samples) before production; deterministic
  (quantile-stratified amplitude draws; any RNG seeded and recorded).

## Stop Conditions

- If the rebuilt observed curve does not match the notebook PDF, stop and
  reconcile selections before interpreting closure.
- If closure fails grossly (pulls >> 3 across whole regions), write the failure
  pattern into Open Questions before tuning anything.

## Results

- Completed: 2026-06-12
- Commands:
  - `conda run -n sensei_charge_traps python src/observed_cutflow.py`
  - `conda run -n sensei_charge_traps python src/naive_efficiency_closure.py --amplitude-samples 512 --output-tag v1`
- Inputs: `cache/01_records_ngood4.csv`, `cache/08_pdet_grid_tau1000_v1.h5`,
  `cache/05_amplitude_prior_v1.npz`, `../fit_dipole_spectra_err_4.h5`
- Outputs: `cache/11_observed_cutflow_v1.csv` (+ summary JSON),
  `cache/11_naive_efficiency_closure_v1.h5` (+ summary JSON),
  `cache/figures/11_closure_overlay.png`, `11_closure_pulls.png`,
  `11_temperature_decomposition.png`, `11_cut_attribution.png`

### Findings

1. **Baseline Method 3 prediction does not close** (mean |delta eff| = 0.283):
   it predicts ~0.93 efficiency across `tau ~ 1e-4..1e-2 s` where the observed
   plateau sits at ~0.155. Conditioning on characterization changes nothing
   (P(char) ~ 1 for all known traps; 0 conditional-denominator fallbacks).
2. **Observed cutflow identifies the mechanism** (49,105 points, 0 stored-vs-
   recomputed `GoodIntensityFit` mismatches): for bright, in-window curves the
   chi-square `p_value` cut fails at 1-44% below 160 K but 44% -> 93% from
   165 K to 210 K, with median p-value numerically zero at `T >= 175 K` —
   gross misfit of the analytic intensity model, not noise marginality.
   Amplitude cuts stay ~5% at high T. Conversely at `T <= 155 K` the median
   reduced chi-square is 0.12-0.25, i.e. `intensity_err` overestimates the
   true scatter by ~2-3x.
3. **Hybrid closure**: multiplying `p_det` by the empirical per-temperature
   bright-point GOF survival (23 scalars; all tau structure still from the
   Stage 08 model) reproduces the curve: mean |delta eff| drops 0.283 -> 0.053;
   plateau 0.142 vs observed 0.155; zeros, peak position/width, and falloff
   all emerge. Residuals (peak 0.94 vs 0.98, falloff tail underpredicted) are
   consistent with mechanism (2b): real noise below `intensity_err` at low T
   would raise model `p_det` at the window edges.

### Interpretation of the naive curve's odd features

- Hard zeros: `tau(T)` outside the dtph window at every temperature (structural).
- ~0.15 plateau at `1e-4..1e-2 s`: those bins are ~95% high-temperature points
  (`T >= 175 K`), where real curves are bright but fail the GOF cut at 75-93%.
- Dip/bump structure in the plateau: temperature-composition handover folded
  through the GOF failure ramp.
- Narrow ~0.98 peak at `0.3..1 s`: bins dominated by 145-160 K points where
  everything passes.
- The curve is therefore a selection/data-quality artifact, not trap physics.

### Required Checks

- Rebuilt observed curve matches the notebook PDF: PASS (peak 0.977 at 0.387 s,
  49,105 points / 15,233 measured).
- Pass-fraction vs unconditional-prediction consistency: PASS (max abs diff 2e-15).
- Conditional-denominator fallbacks: PASS (0 of 2135 traps).
- Smoke run before production: PASS (64-sample smoke, identical structure).

### Open Questions

- **The high-T GOF breakdown is itself unexplained physics/instrumentation**:
  why do bright dipole intensity curves at `T >= 165 K` deviate grossly from
  `intensity_function`? Candidates: pumping-cycle model corrections when tau
  approaches clock phase times, dark-current systematics in the dipole pixels,
  multi-trap pixels. This matters beyond the naive curve, because the Stage 10
  completeness statement's long-`tau_135` reach (e.g. >= 95% to ~700 s) relies on
  high-temperature detections that the idealized Stage 08 model treats as ~95%
  GOF-efficient when the real rate is ~10-25%.
- Optional sensitivity: rerun Stage 08 with injection noise scaled to the true
  low-T scatter (~0.4x `intensity_err`) to close the remaining peak/falloff residual.

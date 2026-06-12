# 06 Single-Curve Recovery

## Objective

Run a fully inspectable fake-trap injection-recovery walkthrough for one temperature, one amplitude,
one noise value, and several `tau` values.

## Why This Matters

This stage turns Method 3 from equations into an understandable procedure. Before computing grids,
inspect individual synthetic curves, noisy realizations, fits, recovered parameters, and cut
decisions.

## Inputs

- `agents/04_intensity_error_scaling.md`
- `agents/05_amplitude_prior.md`
- `../dipole.py`

## Procedure

1. Choose one representative temperature, preferably one with a well-populated `seconds` grid.
2. Choose one representative `sigma` from the Stage 03 noise distribution.
3. Choose one representative amplitude from the Stage 05 amplitude prior.
4. Select several `tau` values: short, near-peak reachable, long rising-edge, and effectively
   undetectable.
5. Generate true curves from the intensity equation.
6. Add synthetic noise using the Stage 04 model.
7. Fit with the same intensity model and apply the good-trap cuts.
8. Save plots and a table showing true parameters, fitted parameters, and cut outcomes.

## Required Checks

- The true curve peak location and long-`tau` rising-edge behavior match analytic expectations.
- Bright near-band examples pass in most realizations.
- Far outside-band examples fail for understandable reasons.
- The cutflow identifies which cut controls each failure.

## Outputs

- `cache/06_single_curve_recovery.csv`
- `cache/06_single_curve_recovery_summary.json`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if synthetic fitting cannot reproduce the expected behavior for simple hand-picked cases.
- Stop if any cut cannot be implemented in the same way as the paper pipeline.

## Results

Completed 2026-05-20T14:04:14-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/single_curve_recovery.py
```

Input artifacts:

- `agents/04_intensity_error_scaling.md`
- `agents/05_amplitude_prior.md`
- `cache/04_intensity_error_scaling.csv`
- `cache/04_intensity_error_scaling.json`
- `cache/03_noise_map_v1.h5`
- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- `../dipole.py`

Output artifacts:

- `cache/06_single_curve_recovery.csv`
- `cache/06_single_curve_recovery_summary.json`
- `cache/figures/06_single_curve_recovery_examples.png`
- `src/single_curve_recovery.py`

Short numerical summary:

- Used deterministic seed `2026052006`.
- Walkthrough configuration: `T = 160 K`, quadrant `0`, `18` dwell points, `128`
  realizations per tau case, and `5` hand-picked tau cases.
- Representative amplitude was `3168.45 e-`, from the Stage 05 default depth median
  `2511.61 e-` multiplied by the Stage 05 `P_c(160 K) = 1.26152`.
- Stage 03/04 noise setting used exact `(T, quadrant, dtph)` trap-free local sigma samples for
  all synthetic points; fallback fraction was `0.0`. The selected trap-free sigma median was
  `273.224 e-`, and the representative Stage 04 image-sigma threshold was `309.510 e-`.
- CSV contains `640` realizations plus header.
- Pass fractions by tau case:
  - `short_outside_band`: `0.0078125`
  - `near_peak_reachable`: `0.96875`
  - `long_reachable_peak`: `0.953125`
  - `long_rising_edge`: `0.0546875`
  - `effectively_undetectable_long`: `0.0`
- Median recovered `fit_tau / true_tau` was `1.0199` for `near_peak_reachable` and `1.0244`
  for `long_reachable_peak`. The long-rising-edge case mostly failed because `fit_tau_err /
  fit_tau > 0.5`, as expected when only the rising edge is visible.
- Dominant failure cuts were understandable:
  - Short outside-band: mostly `max_intensity_lt_3_mean_intensity_err` (`111/128`).
  - Effectively undetectable long: mostly `max_intensity_lt_3_mean_intensity_err` (`117/128`).
  - Long rising-edge: mostly `tau_relative_error_gt_0p5` (`109/128`).

Required checks:

- The true curve peak location and long-`tau` rising-edge behavior match analytic expectations:
  PASS.
- Bright near-band examples pass in most realizations: PASS (`124/128`).
- Far outside-band examples fail for understandable reasons: PASS.
- The cutflow identifies which cut controls each failure: PASS.
- Paper single-temperature cuts are implemented in the Stage 06 helper: PASS.

Stop conditions encountered:

- None. Synthetic fitting reproduced the expected simple cases, and every paper-pipeline
  single-temperature cut from `fitTrapIntensity` was implemented in the Stage 06 helper.

## Open Questions

- Stage 07 can reuse the small Method 3 fit/cut function in `src/single_curve_recovery.py` for
  single-temperature `p_det` and should aggregate the same `controlling_failure_cut` labels into
  cutflow fractions.
- Stage 06 used the Stage 04 temperature/quadrant median `image_sigma` as the peak-threshold
  field. Stage 07 should decide whether to sample this threshold from the detected-trap
  temperature/quadrant distribution or keep a deterministic representative threshold.

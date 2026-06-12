# 07 Single-Temperature Detection Probability

## Objective

Compute `p_det(tau, A, sigma, T)` for one representative temperature and inspect the cutflow across
the grid.

## Why This Matters

A one-temperature grid exposes problems before the expensive full grid. It also makes clear whether
the transition from detectable to undetectable is controlled by peak height, fit uncertainty,
chi-square, or another cut.

## Inputs

- `agents/06_single_curve_recovery.md`
- `agents/03_trap_free_noise_map.md`
- `agents/04_intensity_error_scaling.md`
- `agents/05_amplitude_prior.md`
- `../dipole.py`

## Procedure

1. Choose one representative temperature and document why.
2. Define a modest `tau` grid and amplitude grid that cover short, peak-reachable, and long-rising
   regimes.
3. Use one fixed representative `sigma`, then repeat with low/median/high `sigma` quantiles.
4. Run injection-recovery with deterministic seeds.
5. Store final recovery fraction and cutflow fractions for every grid point.
6. Compare the dominant peak-cut transition to the analytic error-function approximation.
7. Repeat a small subset with an alternate seed to estimate Monte Carlo noise.

## Required Checks

- `p_det` approaches 1 for bright, peak-reachable traps.
- `p_det` approaches 0 for very faint or unreachable traps.
- Cutflow fractions sum consistently.
- Monte Carlo seed variation is small compared with the intended precision.

## Outputs

- `cache/07_single_temperature_pdet_<TEMP>K_v1.h5`
- `cache/07_single_temperature_pdet_summary.json`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if recovery behavior is non-monotonic in a way that cannot be explained.
- Stop if cutflow accounting is incomplete.
- Stop if runtime makes the full grid infeasible without changing grid or realization counts.

## Results

Completed 2026-05-20T14:34:26-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/single_temperature_pdet.py
```

Input artifacts:

- `agents/06_single_curve_recovery.md`
- `agents/03_trap_free_noise_map.md`
- `agents/04_intensity_error_scaling.md`
- `agents/05_amplitude_prior.md`
- `cache/06_single_curve_recovery_summary.json`
- `cache/03_noise_map_v1.h5`
- `cache/04_intensity_error_scaling.csv`
- `cache/04_intensity_error_scaling.json`
- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- `../dipole.py`

Output artifacts:

- `cache/07_single_temperature_pdet_160K_v1.h5`
- `cache/07_single_temperature_pdet_summary.json`
- `cache/figures/07_single_temperature_pdet_160K_v1.png`
- `src/single_temperature_pdet.py`

Short numerical summary:

- Chose `160 K`, quadrant `0`, matching the Stage 06 diagnostic temperature. This temperature has
  a populated `18`-point dwell grid and already passed the single-curve recovery walkthrough.
- Main grid shape is `10 tau x 7 A x 3 sigma = 210` grid points, with `80` realizations per grid
  point (`16800` fits). Alternate-seed subset used `9` grid points with `80` realizations each
  (`720` fits).
- Tau grid in seconds: `2e-05`, `0.000168314421`, `0.000673257686`, `0.004039546114`,
  `0.015709346001`, `0.058348999432`, `0.336628842874`, `0.561048071457`, `2.0`, `10.0`.
- Amplitude grid in electrons: `316.845`, `633.691`, `1108.959`, `1901.072`, `3168.454`,
  `5069.527`, `7921.135`. The representative Stage 05 amplitude is `3168.454 e-`.
- Fixed local-noise sigma values are Stage 03 trap-free `160 K`/quadrant `0` quantiles:
  p16 `263.477 e-`, median `273.224 e-`, p84 `291.248 e-`. The representative image-sigma peak
  threshold remains the Stage 04 median `image_sigma = 309.510 e-`.
- Overall `p_det` median is `0.20625`, mean `0.42714`, min `0.0`, max `1.0`.
- Bright, peak-reachable traps at the largest amplitude have `p_det` median `0.95`, range
  `0.9125`-`0.975`.
- Low-amplitude unreachable long-tau cases have `p_det` median `0.0` and max `0.0125`.
- Dominant controlling-cut fractions over all realizations: pass `0.42714`,
  `max_intensity_lt_3_mean_intensity_err` `0.40173`, `tau_relative_error_gt_0p5` `0.07077`,
  `max_intensity_lt_3_image_sigma` `0.05042`, `p_value` `0.04994`, `fit_failed` `0.0`.
- The analytic peak error-function approximation was evaluated at each grid point using
  `max(3 * local_sigma, 3 * image_sigma)` as the controlling peak threshold. It identified `58`
  transition points with RMSE `0.279` versus the full injection-recovery `p_det`; this is useful
  for locating the peak-threshold boundary but not a full replacement for fitting/cutflow effects.
- Alternate seed `2026052707` differed from the baseline by median absolute `p_det` delta `0.0`,
  max delta `0.025`, and max delta / expected binomial sigma `0.707`.

Required checks:

- `p_det` approaches 1 for bright, peak-reachable traps: PASS.
- `p_det` approaches 0 for very faint or unreachable traps: PASS.
- Controlling cutflow fractions sum consistently: PASS. HDF5 readback maximum deviation from
  unity was `2.22e-16`.
- Monte Carlo seed variation is small compared with intended precision: PASS.
- No large unexplained amplitude non-monotonicity: PASS.
- Dominant peak-cut transition compared to analytic error-function approximation: PASS.

Stop conditions encountered:

- None. Recovery behavior was monotonic within the Monte Carlo tolerance, cutflow accounting was
  complete, and the runtime was short enough that Stage 08 can scale after refining the grid.

## Open Questions

- Stage 08 should increase grid density near the peak-threshold transition where the erf
  approximation changes rapidly.
- Stage 08 should decide whether the sigma dimension remains a fixed local-noise quantile grid or
  returns to exact dtph-wise sigma draws for each synthetic point.

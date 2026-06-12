# 10 Validation And Sensitivity

## Objective

Validate Method 3 against observed traps and empirical Method 2 bands, then quantify sensitivity to
cuts, amplitude prior, noise assumptions, and `n_good`.

## Why This Matters

The final completeness statement must be quantitative and conditional. This stage turns the model
outputs into claims with clearly stated assumptions and hard limits.

## Inputs

- `agents/09_characterization_probability.md`
- `agents/08_full_pdet_grid.md`
- `agents/05_amplitude_prior.md`
- `agents/03_trap_free_noise_map.md`
- `../trap_completeness_method.md`

## Procedure

1. Overlay observed characterized traps on the `P(characterized | tau_135, E)` map.
2. Compare Method 3 high-detection regions against empirical Method 2 sensitivity bands.
3. Compute coverage versus `tau_135` under the observed `E` distribution.
4. Repeat key outputs for sensitivity variants:
   - `n_good = 3` vs `4`,
   - shifted/fainter amplitude priors,
   - alternate `P_c(T)` assumptions,
   - stricter/looser cuts where feasible,
   - low/median/high noise quantile approximations.
5. Identify the regime that remains genuinely unbounded.
6. Draft the final conditional completeness statement.

## Required Checks

- Demonstrated detections generally fall in regions of high predicted characterization probability.
- Sensitivity variants are summarized numerically, not just visually.
- The final statement distinguishes recoverable fraction from population bound.
- The final statement explicitly conditions on the observed `E` distribution and amplitude prior.

## Outputs

- `cache/10_validation_sensitivity_summary.json`
- `cache/10_completeness_statement.md`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if validation shows known traps predominantly in low-probability regions.
- Stop if sensitivity to amplitude prior dominates and cannot be bounded honestly.
- Stop if the final claim would overstate the population bound beyond the assumptions.

## Results

Completed 2026-05-23T11:39:03-07:00.

Command used:

```bash
wsl -e /home/ansh/miniforge3/envs/sensei_charge_traps/bin/python \
  /mnt/c/Users/Ansh/Projects/sensei_charge_traps/trap_completeness_method3/src/validation_sensitivity.py
```

Input artifacts:

- `cache/09_characterization_probability_v1.h5`
- `cache/09_characterization_probability_summary.json`
- `cache/08_pdet_grid_v1.h5`
- `cache/08_pdet_grid_summary.json`
- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- `cache/01_records_ngood4.csv`
- `cache/01_records_ngood3.csv`
- `../trap_completeness_method.md`

Output artifacts:

- `cache/10_validation_sensitivity_summary.json`
- `cache/10_completeness_statement.md`
- `cache/figures/10_known_trap_probability_hist.png`
- `cache/figures/10_completeness_vs_tau.png`
- `cache/figures/10_method2_vs_method3_hist.png`
- `src/validation_sensitivity.py`

Short numerical summary:

- Known-trap overlay passed. For the `n_good = 4` characterized-trap CSV, `2135` traps were
  evaluated on the Stage 09 map; `99.95%` have `P_4 >= 0.5`, `99.86%` have `P_4 >= 0.8`, and
  median `P_4 = 0.99999996`. Only one known trap has `P_4 < 0.5`.
- The default `n_good = 4` completeness curve, averaged over the observed `n_good = 4` `E`
  distribution, is `1.000` at `tau_135 = 1 s`, `1.000` at `10 s`, `0.995` at `100 s`,
  `0.936` at `10^3 s`, `0.686` at `10^4 s`, and `0.0799` at `10^5 s`.
- Under the same observed-`E` distribution, the default `n_good = 4` curve is `>=95%` complete
  for `tau_135 = 5.62e-4` to `668 s`, and `>=90%` complete for `tau_135 = 3.98e-4` to
  `1.58e3 s`.
- `n_good = 3` expands the `>=90%` interval to `1.41e-4` to `1.88e3 s`.
- Fainter amplitude variants were computed numerically from the Stage 08 grid and Stage 05 variant
  priors. At `tau_135 = 10^3 s`, mean `P_4` is `0.936` for the default prior, `0.841` for
  faint-by-2, and `0.330` for faint-by-4. The faint-by-2 `>=90%` interval is `0.141` to
  `335 s`; faint-by-4 has no `>=90%` interval on the Stage 09 tau grid.
- Excluding `160 K` and `170 K` was also computed numerically. It is negligible for the final
  observed-`E` curve at the reported precision: mean `P_4(10^3 s) = 0.935810` with all
  temperatures and `0.935810` with those two temperatures excluded.
- Method 2 empirical-band comparison: the empirical `eff >= 0.5` band-count rule marks `18.83%`
  of the Stage 09 grid recoverable, while Method 3 has `71.07%` of the grid at `P_4 >= 0.8`.
  `94.41%` of Method 2 recoverable grid points also have Method 3 `P_4 >= 0.8`; only `25.01%` of
  Method 3 `P_4 >= 0.8` points are inside Method 2, consistent with Method 2 being a conservative,
  data-starved proxy for the analytic model.
- The all-temperatures-out-of-Stage-08-tau-band regime covers `16.13%` of the Stage 09 grid.
  Within that regime, maximum `P_4` is exactly `0.0`; it spans `E = 0.04` to `0.4745 eV` and begins
  at `tau_135 = 168 s` at the lowest grid energy, moving to larger `tau_135` with increasing `E`.
- April-only `200 K` provenance is inherited from Stage 08: CCD2 run IDs `160-184`, repeated
  low-`dtph` rows collapsed, and `200 K` image-sigma thresholds recomputed from April-only FITS.

Implementation checks:

- The Stage 10 recomputation of the default `n_good = 4` map uses
  `dipole.log_energy_cross_section` and reproduces the Stage 09 map to max absolute difference
  `1.07e-14` (`p99 = 3.00e-15`).
- Known characterized traps mostly fall in high predicted characterization-probability regions:
  PASS.
- Sensitivity variants are numerical, not just visual: PASS.
- Final statement is explicitly conditional on the observed/high-confidence amplitude prior and
  observed `E` distribution: PASS.
- Final statement distinguishes recoverable completeness from an unconditional population bound:
  PASS.

Stop conditions encountered:

- None. The amplitude sensitivity is important, especially for the faint-by-4 stress test, but it
  is bounded numerically and handled by conditional wording rather than by overstating a population
  bound.

## Open Questions

- Paper language should use the recoverable completeness fraction as the primary statement. A
  population upper bound should only be quoted with the explicit extra assumption that the hidden
  population shares both the observed `E` distribution and the observed/high-confidence amplitude
  prior.

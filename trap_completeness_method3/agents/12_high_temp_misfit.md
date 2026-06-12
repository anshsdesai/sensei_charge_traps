# Stage 12: High-Temperature GOF Failure Mechanism

## Objective

Explain why bright, in-window dipole intensity curves fail the chi-square
`p_value` cut at 44-93% for `T >= 165 K` (Stage 11 finding) before revising the
completeness figures.

## Inputs

- `cache/11_observed_cutflow_v1.csv` (Stage 11)
- `../fit_dipole_spectra_err_4.h5` (real spectra)
- `../proc/*.fits` (raw 190 K long-dtph image for the sign/monopole test)

## Method

`src/high_temp_misfit_diagnostics.py`: for ~300 bright in-window curves at each
of 150/160/165/175/190/200 K, compute normalized-residual structure vs dtph and
vs `t_ph/tau`, and test refit variants (constant offset; free second exponent).
Follow-up inline checks: folded-Gaussian rectification test in the signal-free
tail (`t_ph > 10 tau`), offset-vs-amplitude/tau correlations, and a raw-image
monopole-vs-dipole test at the largest 190 K dtph.

## Results

- Completed: 2026-06-12
- Outputs: `cache/12_high_temp_misfit_summary.json`,
  `cache/figures/12_mean_residual_by_dtph.png`, `12_residual_vs_scaled_time.png`,
  `12_fix_tests.png`, `12_examples_190K.png`

### Findings

1. **A single constant offset fixes the fits at every temperature.** Median
   reduced chi-square: base model 0.17 / 0.36 / 1.3 / 8.5 / 10.8 / 12.5 at
   150/160/165/175/190/200 K; with `model + C` it returns to 0.06-0.21
   everywhere. A free second exponent does not fix it (the floor is not a
   pumping-shape change). Median fitted `C`: 64 / 180 / 268 / 694 / 713 / 664
   electrons.
2. **The floor is real signal, not |.| rectification noise.** In the tail
   (`t_ph > 10 tau`) mean/std = 12-16 at 175-200 K (folded Gaussian predicts
   1.32); floor ~440-716 e- vs ~107 e- rectification prediction.
3. **The floor is a readout dipole, not a hot pixel.** In the raw 190 K
   `dtph = 15.5 s` image the trap pixel is depleted ~1000 e- while
   `(a+b)/2 - background` is only -45..-116 e- (all four quadrants; ~75%
   consistent sign). A dark-current generation site would be a monopole.
   Interpretation: the trap acts as a CTI/deferred-charge defect during
   readout on the large high-T dark-current background; `t_ph`-independent,
   grows with dark current (T) and correlates positively with `tau` at fixed T
   (`rho ~ +0.5..+0.7` at 190/200 K).
4. **GOF ramp onset is quantitative**: floor/sigma crosses ~1 at 160-165 K
   (floor 64->700 e- vs sigma ~200-280 e-), matching the Stage 11 pass-rate
   collapse.
5. **Independent confirmation of the noise mismatch**: tail scatter across
   images is ~31-46 e- vs `intensity_err` ~190 e- — the patch-based error is a
   spatial sigma that overestimates the temporal noise of a fixed pixel pair
   ~2.5x at all temperatures (explains chi2red ~0.15 "too good" fits at low T).

### Implications

- The 2-parameter `intensity_function` lacks a real `t_ph`-independent floor
  term; the chi-square cut therefore rejects bright high-T curves. The high-T
  data are *recoverable*: with `model + C` they fit well, so the per-temperature
  tau information at `T >= 165 K` exists in the data but is discarded by the
  current pipeline.
- Any completeness statement for the *current* pipeline must use the real
  (collapsed) high-T pass rates; alternatively the pipeline fit model can be
  upgraded to 3 parameters to recover those temperatures.

### Open Questions

- Microdynamics of the readout floor (capture from passing packets vs trailing
  release profile) — not needed for the figure fix.
- Whether 3-parameter refits leave tau unbiased at high T (spot-check before
  any pipeline upgrade).

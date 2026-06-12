# 04 Intensity-Error Scaling

## Objective

Determine how synthetic injection noise should be drawn by relating stored `intensity_err` to
`image_sigma`, temperature, dwell time, quadrant, and intensity.

## Why This Matters

The injection-recovery model perturbs synthetic intensity curves. The noise scale should match the
measurement noise in the spectra, not an ad hoc scalar threshold.

## Inputs

- `agents/01_hdf5_records_audit.md`
- `agents/02_fits_noise_parity.md`
- `agents/03_trap_free_noise_map.md`
- `../fit_dipole_spectra_err_4.h5`
- `../utils.py`

## Procedure

1. Extract `intensity_err`, `image_sigma`, `seconds`, `intensities`, and `GoodIntensityFit` from
   per-temperature spectra.
2. Compute ratios such as `intensity_err / image_sigma` and summarize by temperature, delay index,
   quadrant, and intensity.
3. Check whether intensity errors are approximately constant within a spectrum or depend on delay.
4. Compare detected-trap `image_sigma` to trap-free `p_sigma` from Stage 03.
5. Choose the injection noise model for Stage 06 onward and write it explicitly.

## Required Checks

- The chosen synthetic noise model is justified numerically.
- Any temperature or delay dependence in `intensity_err` is either modeled or shown to be negligible.
- The model distinguishes the peak-threshold `image_sigma` cut from per-point `intensity_err`.

## Outputs

- `cache/04_intensity_error_scaling.json`
- `cache/04_intensity_error_scaling.csv`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if `intensity_err` cannot be related to stored quantities well enough for injection.
- Stop if Stage 03 has not produced a usable trap-free noise distribution and the analysis needs it.

## Results

Completed 2026-05-20T13:25:30-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/intensity_error_scaling.py
```

Input artifacts:

- `agents/01_hdf5_records_audit.md`
- `agents/02_fits_noise_parity.md`
- `agents/03_trap_free_noise_map.md`
- `../fit_dipole_spectra_err_4.h5`
- `cache/03_noise_map_v1.h5`
- `../utils.py`

Output artifacts:

- `cache/04_intensity_error_scaling.json`
- `cache/04_intensity_error_scaling.csv`
- `src/intensity_error_scaling.py`

Short numerical summary:

- Scanned `118933` per-temperature spectra and `2487251` finite intensity points.
- `GoodIntensityFit = True` spectra: `20121`; `GoodIntensityFit = False` spectra: `98812`.
- Global `intensity_err` median is `191.425 e-`, with p16/p84 `159.028`/`242.114 e-`
  and p95 `335.272 e-`.
- Global `intensity_err / image_sigma` median is `0.882245`, with p16/p84
  `0.818311`/`0.990370`; this confirms `image_sigma` is not the per-point error scale.
- Detected local `intensity_err` medians divided by Stage 03 trap-free medians over the `92`
  temperature/quadrant groups have median `1.06851`, p16/p84 `1.00404`/`1.07924`,
  and max `1.09240`.
- Detected global `image_sigma` medians divided by Stage 03 trap-free local medians have median
  `1.20079`, p16/p84 `1.15172`/`1.24299`; this is a separate global threshold field.
- Delay dependence is usually small but not uniformly negligible: the median temperature-level
  range of delay-median `intensity_err` values divided by their median is `0.0718`, while the
  largest values are `2.8971` at `207 K`, `1.2251` at `203 K`, `0.5988` at `200 K`, and
  `0.4655` at `210 K`.
- Intensity dependence is weak: Pearson correlation of `log1p(abs(intensity))` with
  `log(intensity_err)` is `0.0698`.

Chosen synthetic noise model:

- For each synthetic intensity point at temperature `T`, quadrant `q`, and dwell `dtph`, draw
  `sigma` from Stage 03 `cache/03_noise_map_v1.h5` samples matching `(T, q, dtph)`, then draw
  per-point noise from `Normal(0, sigma)`.
- If a later stage uses a grid point whose exact `dtph` is absent from Stage 03, fall back to
  matching `(T, q)` and record the fallback fraction in that stage's cutflow.
- Do not use HDF5 `image_sigma` as the curve-perturbation noise scale. It remains a separate
  global temperature/quadrant comparison or peak-threshold field.
- Do not condition noise on measured intensity; the observed intensity dependence is numerically
  small compared with temperature/quadrant/dwell structure.

Required checks:

- The chosen synthetic noise model is justified numerically: PASS.
- Temperature and delay dependence are either modeled or shown negligible: PASS. Temperature,
  quadrant, and exact `dtph` are modeled because high-temperature delay dependence is not
  negligible.
- The model distinguishes the peak-threshold `image_sigma` cut from per-point `intensity_err`:
  PASS.
- Stage 03 trap-free noise distribution is present and usable: PASS.
- All HDF5 points used for scaling have finite `intensity_err`: PASS (`0` nonfinite points).
- Trap-free comparison exists for all `92` temperature/quadrant groups: PASS.
- Minimum trap-free sample count for an exact temperature/quadrant/`dtph` group is `300`: PASS.

Stop conditions encountered:

- None.

## Open Questions

- Later injection-recovery stages should record how often they use the exact `(T, q, dtph)` noise
  distribution versus the `(T, q)` fallback.
- The 203 K and 207 K high-dwell noise tails are real in the stored `intensity_err` summaries and
  should be preserved in Stage 06/07 diagnostics rather than smoothed away.

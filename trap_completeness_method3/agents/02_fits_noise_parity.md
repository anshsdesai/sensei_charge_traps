# 02 FITS Noise Parity

## Objective

Recompute the source-equivalent local patch noise statistic from raw FITS files at known trap
coordinates and identify which stored HDF5 field it matches.

## Why This Matters

This is the hard gate for Method 3. The trap-free noise map is only trustworthy if the same local
noise statistic can be reproduced from the raw images. If parity fails, building `p_sigma(sigma | T)`
would create a black-box selection function.

## Inputs

- `agents/00_data_inventory.md`
- `agents/01_hdf5_records_audit.md`
- `../proc/*.fits`
- `../fit_dipole_spectra_err_4.h5`
- `../dipole.py`
- `../utils.py`

## Procedure

1. Select a small, reproducible sample of characterized traps across quadrants and temperatures.
2. For each sampled trap and temperature, find the matching FITS images that contributed to the
   stored spectrum.
3. Recompute the local background-subtracted patch statistic using the same logic as
   `getDipoleSpectra2` / `histogram_around_point(size=35)`. Note that the implementation's
   interior slice is effectively `34 x 34`.
4. Compare recomputed values to stored `d[T]["intensity_err"]` and `d[T]["image_sigma"]`, then
   document the field semantics.
5. Report absolute and fractional residuals by temperature, quadrant, and image/dtph where possible.
6. Save a parity table and concise diagnostic plots.

## Required Checks

- Median fractional mismatch is small or fully explained by averaging/selection details.
- No strong temperature- or quadrant-dependent mismatch remains unexplained.
- Any mismatch between per-image and per-spectrum aggregation is documented.

## Outputs

- `cache/02_noise_parity_sample.csv`
- `cache/02_noise_parity_summary.json`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if FITS-to-HDF5 matching is ambiguous and cannot be resolved from headers or filenames.
- Stop if recomputed local patch noise disagrees with stored `intensity_err` beyond an understood
  tolerance.
- Do not proceed to Stage 03 until this packet records a pass or an explicit, justified correction.

## Results

Completed 2026-05-20T12:36:33-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/fits_noise_parity.py
```

Input artifacts:

- `../cache/01_records_ngood4.csv`
- `../../fit_dipole_spectra_err_4.h5`
- `../../proc/`
- `../../dipole.py`
- `../../utils.py`

Output artifacts:

- `../cache/02_noise_parity_sample.csv`
- `../cache/02_noise_parity_summary.json`
- `../src/fits_noise_parity.py`

Short numerical summary:

- Sampled `12` characterized traps: `3` tau-quantile-spaced traps per quadrant.
- Sampled `6` temperatures: `125`, `135`, `160`, `185`, `200`, and `210 K`.
- Wrote `1428` parity rows. Per sampled trap this is `18` dwell points at `125`, `135`,
  `160`, `185`, and `210 K`, and `29` dwell points at `200 K`.
- Recomputed `476` unique FITS/quadrant images.
- Full FITS index has `4` duplicate CCD2 `(T, dtph)` matches, all at `200 K` low dwell times.
  In the sampled rows, `96` duplicate-FITS rows were resolved by exact `intensity_err` parity;
  `0` sampled rows remained ambiguous or missing.
- Recomputed local patch sigma matches stored HDF5 `intensity_err` exactly in the sample:
  median, p95, and max fractional residual are all `0.0`.
- Stored HDF5 `image_sigma` is not the local patch statistic. It is a single whole-image
  row-median-subtracted quadrant sigma per temperature/quadrant. In the sample, it matches the
  `dtph=1000000` FITS image sigma exactly for all `24` sampled temperature/quadrant groups
  (`max_best_match_abs_residual = 0.0`).
- Direct local-patch-vs-`image_sigma` fractional residuals are therefore nonzero by field
  semantics, not by failed FITS reproduction: median `0.13494`, p95 `0.20390`, max `1.82160`.

Required checks:

- FITS-to-HDF5 matching is unambiguous for sampled rows after exact parity resolution of duplicate
  200 K low-dwell files: PASS.
- Median fractional mismatch is small or fully explained: PASS. The source-equivalent local patch
  statistic reproduces `intensity_err` exactly; the local-vs-`image_sigma` mismatch is explained
  because `image_sigma` is a whole-image field.
- No strong temperature- or quadrant-dependent local mismatch remains unexplained: PASS. The
  local-vs-`intensity_err` residual is exactly zero in every sampled temperature and quadrant.
- Per-image/per-spectrum aggregation is documented: PASS. `image_sigma` is a single per-spectrum
  whole-image value matching the `dtph=1000000` image in this sample, while `intensity_err` is the
  per-dwell local patch sigma.

Stage 02 gate:

- PASS_WITH_CORRECTION. Stage 03 may proceed because the `getDipoleSpectra2` local patch
  definition was validated against `intensity_err`. Do not use HDF5 `image_sigma` as a local-noise
  field; keep it as the separate whole-image threshold-like field.

Stop conditions encountered:

- None.

## Open Questions

- Stage 03 should build `p_sigma(sigma | T)` from trap-free local patches using the same effective
  source slice as `histogram_around_point(size=35)`: `row-half:row+half` with `half=17`, i.e. a
  `34 x 34` interior patch.
- Stage 04 should treat `image_sigma` as the global detection-threshold field and `intensity_err`
  as the local per-point uncertainty field.
- The `200 K` HDF5 grid has `29` seconds/intensity points per trap and duplicate low-dwell entries.
  Downstream code should use the actual per-temperature HDF5 grids instead of assuming `18` points.

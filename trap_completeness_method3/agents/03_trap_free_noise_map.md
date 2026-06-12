# 03 Trap-Free Noise Map

## Objective

Build an unbiased spatial noise distribution `p_sigma(sigma | T)` from trap-free local patches
using the Stage 02 source-equivalent patch convention across the raw FITS images.

## Why This Matters

Method 3 escapes survivorship bias by measuring the detection threshold from image background
statistics rather than from detected traps. This stage creates that trap-independent noise model.

## Inputs

- `agents/02_fits_noise_parity.md`
- `../proc/*.fits`
- `../dipole_coord_list.npz`
- `../fit_dipole_spectra_err_4.h5`
- `../dipole.py`
- `../utils.py`

## Procedure

1. Use the Stage 02 parity result to define the exact local-noise statistic.
2. Build an exclusion mask around known dipole coordinates and image boundaries.
3. Sample a reproducible grid or random set of trap-free source-equivalent local patches by
   temperature and quadrant.
4. Compute local noise for each patch using the parity-approved statistic.
5. Summarize distributions by temperature and quadrant: count, median, 16/84 percentiles, and tails.
6. Compare trap-free patch `sigma` distributions to detected-trap `image_sigma` as a cross-check.
7. Save a noise-map artifact with metadata.

## Required Checks

- Patch sampling avoids known trap coordinates and image edges.
- Each temperature has enough trap-free samples to estimate the central distribution and tails.
- Detected-trap `image_sigma` is used only as a comparison, not as the injection input.

## Outputs

- `cache/03_noise_map_v1.h5`
- `cache/03_noise_map_summary.csv`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if Stage 02 has not passed.
- Stop if trap-free patch selection is contaminated by known trap sites or masked regions.
- Stop if a temperature has too few usable patches for a stable distribution.

## Results

Completed 2026-05-20T12:53:00-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/build_trap_free_noise_map.py --samples-per-image-quad 300
```

Input artifacts:

- `agents/02_fits_noise_parity.md`
- `../proc/*.fits`
- `../dipole_coord_list.npz`
- `../fit_dipole_spectra_err_4.h5`
- `../dipole.py`
- `../utils.py`

Output artifacts:

- `cache/03_noise_map_v1.h5`
- `cache/03_noise_map_summary.csv`
- `src/build_trap_free_noise_map.py`

Short numerical summary:

- Selected `481` CCD2 `proc*dtph*.fits` files, matching the Stage 02/source-code FITS selection.
- Skipped `481` non-CCD2 dwell FITS and `42` non-`dtph` FITS.
- Processed `1924` FITS/quadrant images.
- Sampled `300` trap-free patch centers per selected FITS/quadrant.
- Wrote `577200` trap-free local-noise samples across `92` temperature/quadrant groups.
- Per temperature/quadrant sample counts range from `5400` to `8700`.
- Trap-free median local sigma across temperature/quadrant groups ranges from `126.326` to
  `340.270` electrons.
- Example total sample counts and median ranges by temperature:
  `125 K`: `21600` samples, medians `175.151`-`217.423`;
  `135 K`: `21600` samples, medians `181.932`-`226.011`;
  `160 K`: `21600` samples, medians `270.292`-`340.270`;
  `170 K`: `21600` samples, medians `255.909`-`323.173`;
  `185 K`: `21600` samples, medians `152.339`-`193.252`;
  `200 K`: `34800` samples, medians `135.461`-`172.668`;
  `210 K`: `21600` samples, medians `126.326`-`159.803`.
- Acquisition-family note: the available `160 K` and `170 K` files in `proc/` are from
  `dp_scan1 / SC300000`, not `temp_scan_run1 / SC200000`. Their higher local-noise medians are
  therefore treated as measured acquisition-context noise, not smoothed away as a temperature
  trend.
- Cross-check against detected-trap local `intensity_err`: trap-free/detected-local median ratio
  ranges from `0.9154` to `1.0047`, with median `0.9359`.
- Cross-check against detected-trap whole-image `image_sigma`: trap-free/detected-global median
  ratio ranges from `0.7730` to `0.8884`, with median `0.8328`. This is comparison-only and is
  consistent with Stage 02's field-semantics result.

Required checks:

- Stage 02 gate passed with correction and this stage used the parity-approved local patch
  statistic: PASS.
- Patch sampling avoids known trap coordinates and image edges: PASS. Candidate centers within
  Chebyshev distance `<= 17` pixels of any known dipole coordinate were rejected; the minimum
  sampled center distance to a known dipole site was `18` pixels.
- Every sampled local-noise patch has the source-equivalent full interior shape: PASS (`34 x 34`,
  matching `histogram_around_point(size=35)`).
- Each temperature has enough trap-free samples to estimate the central distribution and tails:
  PASS. The minimum temperature/quadrant group contains `5400` samples.
- Detected-trap `image_sigma` is used only as a comparison, not as the injection input: PASS.

Stop conditions encountered:

- None.

## Open Questions

- Stage 03 used uniform random spatial sampling over valid area and fixed samples per selected
  FITS/quadrant. This gives equal quadrant weight per image and naturally weights temperatures by
  the available CCD2 dwell images. If later stages want a trap-spatial prior, they should apply it
  explicitly rather than baking it into the trap-free noise map.
- Stage 04 should use `sigma` from `cache/03_noise_map_v1.h5` as the trap-independent local-noise
  input and keep HDF5 `image_sigma` as a global threshold comparison field only.
- Stage 08 should include `160 K` and `170 K` in the primary model with their empirical
  `dp_scan1 / SC300000` noise, and Stage 10 should include an exclusion sensitivity if the
  acquisition-family difference becomes a concern for the final claim.

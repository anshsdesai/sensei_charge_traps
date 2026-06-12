# 08 Full Detection-Probability Grid

## Objective

Build the full marginalized per-temperature detection grid `p_det(tau, A, T)` with cutflow
diagnostics and reproducible metadata.

## Why This Matters

This artifact is the central Method 3 selection function. It replaces empirical band edges from the
characterized sample with physics, measured noise, and pipeline cuts.

## Inputs

- `agents/03_trap_free_noise_map.md`
- `agents/04_intensity_error_scaling.md`
- `agents/05_amplitude_prior.md`
- `agents/07_single_temperature_pdet.md`
- `../dipole.py`

## Procedure

1. Define final `tau` and amplitude grids based on Stage 07.
2. For each measurement temperature, use the actual `seconds` grid and the Stage 04 noise model.
3. Marginalize over the Stage 03 trap-free local-noise samples by drawing exact
   `(T, quadrant, dtph)` sigmas for each synthetic intensity point where available. If exact
   `dtph` samples are unavailable, fall back to `(T, quadrant)` and record the fallback fraction.
   Store the marginalized `p_det(tau, A, T)` as the primary product; optionally store diagnostic
   low/median/high-noise slices, not a full explicit sigma dimension.
4. Run injection-recovery using deterministic seeds.
5. Save `p_det`, uncertainty estimates, cutflow fractions, grids, and metadata.
6. Produce quick-look summaries by temperature: approximate short/long edges, peak-reachable
   window, and dominant failure modes.

## Required Checks

- Every temperature has a detection grid or an explicit reason it is excluded.
- Every artifact includes grids, cuts, seeds, input artifact versions, and code/notebook path.
- Cutflow fractions are stored for every grid point or for a documented compressed representation.
- Warm-temperature long-`tau` behavior is inspected explicitly.
- Exact-`dtph` local-noise draw fractions and fallback fractions are stored by temperature.

## Outputs

- `cache/08_pdet_grid_v1.h5`
- `cache/08_pdet_grid_summary.json`
- Optional plots under `cache/figures/`

## Stop Conditions

- Stop if Stage 07 did not establish sane one-temperature behavior.
- Stop if any temperature lacks a trustworthy `seconds` grid or noise model.
- Stop if runtime or memory forces a grid change; record the proposed change before rerunning.

## Results

Pilot completed 2026-05-22T12:11:37-07:00. This is a pilot only, not the final dense production
grid.

Command used:

```bash
/home/ansh/miniforge3/envs/sensei_charge_traps/bin/python \
  trap_completeness_method3/src/full_pdet_grid_pilot.py \
  --realizations 24 --seed 2026052208
```

Input artifacts:

- `agents/03_trap_free_noise_map.md`
- `agents/04_intensity_error_scaling.md`
- `agents/05_amplitude_prior.md`
- `agents/07_single_temperature_pdet.md`
- `cache/03_noise_map_v1.h5`
- `cache/04_intensity_error_scaling.csv`
- `cache/04_intensity_error_scaling.json`
- `cache/05_amplitude_prior_v1.npz`
- `cache/05_amplitude_prior_summary.json`
- `cache/07_single_temperature_pdet_160K_v1.h5`
- `cache/07_single_temperature_pdet_summary.json`
- `../dipole.py`

Output artifacts:

- `cache/08_pdet_grid_pilot_v1.h5`
- `cache/08_pdet_grid_pilot_summary.json`
- `src/full_pdet_grid_pilot.py`

Model and data-selection notes:

- Primary artifact is marginalized `p_det(T, tau, A)` with no explicit sigma axis.
- Each realization draws one quadrant uniformly, uses that quadrant's Stage 04 `image_sigma`
  threshold, and draws per-point local sigmas from Stage 03 exact `(T, quadrant, dtph)` samples.
- If exact samples are unavailable, the script falls back to `(T, quadrant)` and records fallback
  fractions by temperature.
- For `200 K`, the pilot uses the April 2-4 extended CCD2 sequence only. The current upstream HDF5
  was not regenerated; repeated low-dtph Stage 04 rows are collapsed to one row per dtph, and
  Stage 03 `200 K` noise samples are filtered to CCD2 run IDs `160`-`184`. The remaining caveat is
  that Stage 04 representative `image_sigma` thresholds still come from the current upstream
  HDF5-derived summary.
- The available `160 K` and `170 K` measurements are from `dp_scan1 / SC300000`, unlike the
  neighboring `temp_scan_run1 / SC200000` temperatures. The primary Stage 08 model keeps them with
  their empirical noise because they are part of the upstream characterization data; Stage 10 should
  test excluding them as an acquisition-family sensitivity.

Short numerical summary:

- Grid shape is `23 T x 19 tau x 7 A = 3059` grid points, with `24` realizations per grid point
  (`73416` fits).
- Measurement temperatures covered: `125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180,
  183, 185, 187, 190, 193, 195, 197, 200, 203, 207, 210 K`.
- Delay-grid sizes are `18` points for standard-grid temperatures and `25` points for extended
  grids, including April-only `200 K`; maximum stored delay count is `25`.
- Overall `p_det` median is `0.75`, mean `0.5332`, p16/p84 `0.0`/`0.9583`, min/max `0.0`/`1.0`.
- Bright peak-reachable region has median `p_det = 0.9583`, mean `0.9494`, min `0.8333`.
- Faint region has median `p_det = 0.0`, mean `0.1327`.
- Warm long-`tau` bright region has median `p_det = 0.5417`, mean `0.5078`; behavior was
  explicitly inspected but remains coarse at pilot statistics.
- Dominant controlling-cut fractions over all realizations: pass `0.5332`,
  `max_intensity_lt_3_mean_intensity_err` `0.2791`, `tau_relative_error_gt_0p5` `0.0805`,
  `p_value` `0.0576`, `max_intensity_lt_3_image_sigma` `0.0496`, `fit_failed` `0.0`.
- Exact Stage 03 noise draw fraction is `1.0` for every temperature; fallback fraction is `0.0`
  for every temperature after April-only `200 K` filtering.
- Runtime was `145.65 s` wall time (`504` fits/s). HDF5 size is `0.486 MB`; JSON summary size is
  `0.019 MB`.

Required checks:

- Every temperature has a detection grid: PASS.
- Artifact includes grids, cuts, seed, input artifact paths, and code path: PASS.
- Primary grid has no explicit sigma axis: PASS (`p_det` shape is `23 x 19 x 7`).
- Controlling cutflow fractions are stored for every grid point and sum to one: PASS; maximum
  HDF5 readback deviation was `2.22e-16`.
- Warm-temperature long-`tau` behavior inspected explicitly: PASS.
- Exact/fallback local-noise draw counts and fallback fractions are stored by temperature: PASS.
- April-only `200 K` selection is documented in metadata: PASS.

Stop conditions encountered:

- Stage 07 sane one-temperature behavior was available: PASS.
- All temperatures have a usable seconds grid and noise model under the pilot selection: PASS.
- Runtime and memory were summarized before scaling: PASS.
- Stopped after the pilot as requested; did not scale to the final dense grid.

Production grid completed 2026-05-22. The run was launched after the pilot passed and after the
production grid/range choices were reviewed.

Production command:

```bash
bash trap_completeness_method3/src/run_full_pdet_grid.sh
```

Production output artifacts:

- `cache/08_pdet_grid_v1.h5`
- `cache/08_pdet_grid_summary.json`
- `src/full_pdet_grid.py`
- `src/run_full_pdet_grid.sh`

Production numerical summary:

- Grid shape is `23 T x 55 tau x 35 A = 44275` grid points, with `100` realizations per grid
  point (`4427500` fits).
- Tau range is `2e-05` to `20.0 s`.
- Amplitude range is `200.0` to `15000.0 e-`.
- All 23 measurement temperatures are present.
- April-only `200 K` handling was applied: `200 K` has `25` delay points, Stage 03 local-noise
  draws are filtered to April-only `200 K` source FITS, and `200 K` `image_sigma` thresholds were
  recomputed from April-only FITS inside Stage 08.
- Overall `p_det` median is `0.6`, mean `0.5059`, p16/p84 `0.0`/`0.96`, min/max `0.0`/`1.0`.
- Binomial uncertainty median is `0.0218`, mean `0.0206`, max `0.05`.
- Dominant controlling-cut fractions over all realizations: pass `0.5059`,
  `max_intensity_lt_3_mean_intensity_err` `0.3063`, `tau_relative_error_gt_0p5` `0.0837`,
  `p_value` `0.0576`, `max_intensity_lt_3_image_sigma` `0.0465`, `fit_failed` `6.78e-07`.
- Exact Stage 03 noise draw fraction is `1.0` for every temperature; fallback fraction is `0.0`
  for every temperature.
- Runtime was `8722.7 s` (`2.42 h`) at `507.6 fits/s`. HDF5 size is `1.02 MB`; JSON summary
  size is `0.014 MB`.

Production required checks:

- Every temperature has a detection grid: PASS.
- Primary grid has no explicit sigma axis: PASS (`p_det` shape is `23 x 55 x 35`).
- Controlling cutflow fractions are stored for every grid point and sum to one: PASS; maximum
  HDF5 readback deviation was `2.22e-16`.
- Exact/fallback local-noise draw counts and fallback fractions are stored by temperature: PASS.
- April-only `200 K` grid and image-sigma handling applied: PASS.

## Model Choice

Use a marginalized grid as the primary Stage 08 artifact: `p_det(tau, A, T)` after drawing local
noise from the Stage 03 exact `(T, quadrant, dtph)` trap-free distributions. Preserve diagnostics
for low/median/high-noise slices only if they help Stage 10 sensitivity checks. This matches the
Stage 04 noise model and avoids carrying an artificial fixed-sigma axis into Stage 09.

## Open Questions

- Stage 09 should consume `cache/08_pdet_grid_v1.h5`, not the pilot artifact.
- Stage 09 should preserve the April-only `200 K` provenance note in any downstream metadata.
- Decide whether uniform quadrant marginalization is sufficient for Stage 09 or whether a spatial
  or quadrant prior is needed.
- Include an explicit Stage 10 sensitivity that excludes `160 K` and `170 K` if the
  `dp_scan1 / SC300000` acquisition family should not be treated as comparable to the main
  temperature scan.

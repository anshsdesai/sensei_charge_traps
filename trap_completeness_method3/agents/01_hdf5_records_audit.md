# 01 HDF5 Records Audit

## Objective

Load the characterized-trap HDF5 files and summarize the trap-level and per-temperature quantities
available for Method 3.

## Why This Matters

Method 3 uses the HDF5 files for observed trap parameters, the `t_ph` grids, stored local noise,
stored intensity errors, and fitted amplitudes. Before using those fields, verify their presence,
shape, units, and coverage.

## Inputs

- `README.md`
- `AGENT_PROTOCOL.md`
- `agents/00_data_inventory.md`
- `../fit_dipole_spectra_err_4.h5`
- `../fit_dipole_spectra_err_3.h5`
- `../utils.py`
- `../dipole.py`

## Procedure

1. Load `fit_dipole_spectra_err_4.h5` with `utils.load_spectra_hdf5`.
2. Build a flat characterized-trap table using the filter:
   `WellBehavedTrap and not EnergyFitFailed and GoodEnergyFit`.
3. Repeat for `fit_dipole_spectra_err_3.h5` as a sensitivity comparison.
4. For each characterized trap, record quadrant, coordinate, `E`, `log_sigma`, `tau_135`,
   good-temperature count, and measured temperatures.
5. For per-temperature sub-dicts, summarize availability of `seconds`, `intensities`,
   `intensity_err`, `image_sigma`, `fit_coeff`, `fit_tau`, `GoodIntensityFit`, and `fit_p_value`.
6. Compare `seconds` grids within each temperature and report whether they are consistent.
7. Save compact summaries under `cache/`.

## Required Checks

- Characterized trap count for `n_good = 4` is consistent with the paper expectation near 2121.
- Characterized trap count for `n_good = 3` is consistent with the working-note expectation near
  2517.
- Per-temperature sub-dicts contain the fields needed by Stages 02, 04, and 05.
- `tau_135` can be recomputed from `energy_BestFitEnergy`,
  `energy_BestFitCrossSection`, and `log_energy_cross_section`.

## Outputs

- `cache/01_records_ngood4.csv`
- `cache/01_records_ngood3.csv`
- `cache/01_hdf5_field_summary.json`
- Optional plots saved under `cache/figures/`

## Stop Conditions

- Stop if the HDF5 loader fails.
- Stop if required fields are absent or have inconsistent shapes that cannot be explained.
- Stop if `tau_135` recomputation fails.

## Results

Completed 2026-05-20T12:22:57-07:00.

Command used:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python trap_completeness_method3/src/audit_hdf5_records.py
```

Input artifacts:

- `README.md`
- `AGENT_PROTOCOL.md`
- `agents/00_data_inventory.md`
- `../fit_dipole_spectra_err_4.h5`
- `../fit_dipole_spectra_err_3.h5`
- `../utils.py`
- `../dipole.py`

Output artifacts:

- `../cache/01_records_ngood4.csv`
- `../cache/01_records_ngood3.csv`
- `../cache/01_hdf5_field_summary.json`
- `../src/audit_hdf5_records.py`

Short numerical summary:

- Both HDF5 files contain `5171` dipole groups across `4` quadrants and `23` measurement temperatures.
- Characterized-trap count is `2135` for `n_good = 4` and `2517` for `n_good = 3`.
- Every per-temperature sub-dict contains `seconds`, `intensities`, `intensity_err`, `image_sigma`, and `GoodIntensityFit`.
- The `seconds` grid is identical within each temperature in both HDF5 files: `23 / 23` temperatures have a single shared grid, with `18` dwell-time samples per record.
- `fit_coeff`, `fit_tau`, and `fit_p_value` are present for `5052` to `5166` records per temperature. The remaining `896` temperature records per file are exactly the `IntensityFitFailed = True` cases, so downstream stages must guard access to `fit_*` attrs.
- `tau_135` recomputation from `energy_BestFitEnergy`, `energy_BestFitCrossSection`, and `log_energy_cross_section` succeeded for all characterized traps in both files.

Required checks:

- Characterized trap count for `n_good = 4` is consistent with the paper expectation near `2121`: PASS (`2135`, `+14`, `+0.66%`).
- Characterized trap count for `n_good = 3` is consistent with the working-note expectation near `2517`: PASS (`2517` exactly).
- Per-temperature sub-dicts contain the fields needed by Stages 02, 04, and 05: PASS with caveat that `fit_coeff`, `fit_tau`, and `fit_p_value` are absent only when `IntensityFitFailed = True`.
- `tau_135` can be recomputed from `energy_BestFitEnergy`, `energy_BestFitCrossSection`, and `log_energy_cross_section`: PASS.

Stop conditions encountered:

- None.

## Open Questions

- Downstream stages should explicitly gate any use of `fit_coeff`, `fit_tau`, or `fit_p_value` on `IntensityFitFailed = False` or `GoodIntensityFit = True`.

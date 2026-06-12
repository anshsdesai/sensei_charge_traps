# 00 Data Inventory

## Objective

Inventory the raw FITS files, HDF5 fit products, key cache files, interpreter, and importable
dependencies needed for Method 3.

## Why This Matters

All later stages depend on knowing which temperatures, dwell-time grids, and fit products are
actually available. This stage should catch missing raw data, wrong interpreter selection, and
inconsistent file naming before any modeling starts.

## Inputs

- `../proc/*.fits`
- `../fit_dipole_spectra_err_4.h5`
- `../fit_dipole_spectra_err_3.h5`
- `../dipole_coord_list.npz`
- `../dipole.py`
- `../utils.py`
- `../requirements.yaml`

## Procedure

1. Count FITS files by temperature and `dtph` parsed from filename.
2. Record total FITS count and list any filenames that do not match the expected naming pattern.
3. Verify that primary and comparison HDF5 files exist and record their sizes.
4. Verify that `dipole_coord_list.npz` exists and record its size.
5. Verify the canonical WSL runtime from `README.md`:
   `/home/ansh/miniforge3/bin/conda run -n sensei_charge_traps python`.
   Confirm it imports `numpy`, `scipy`, `h5py`, `astropy`, and `iminuit`.
6. Save a compact machine-readable inventory under `cache/00_data_inventory.json`.

## Required Checks

- Every measurement temperature listed in `trap_completeness_method.md` has at least one FITS file
  or is explicitly reported missing.
- HDF5 files for `n_good = 4` and `n_good = 3` are present.
- The canonical WSL `sensei_charge_traps` environment imports the required scientific packages.

## Outputs

- `cache/00_data_inventory.json`
- Optional summary table: `cache/00_fits_by_temperature_dtph.csv`

## Stop Conditions

- Stop if `proc/` is missing or empty.
- Stop if the canonical WSL `sensei_charge_traps` environment cannot import the required scientific
  packages. Do not spend time discovering or setting up alternate environments.
- Stop if the primary HDF5 file is missing.

## Results

Completed 2026-05-20 11:46:46-07:00.

Command used:

```bash
date -Is && /home/ansh/miniforge3/envs/sensei_charge_traps/bin/python - <<'PY'
# inventory script wrote cache/00_data_inventory.json and cache/00_fits_by_temperature_dtph.csv
PY
```

Input artifacts:

- `../proc/*.fits`
- `../fit_dipole_spectra_err_4.h5`
- `../fit_dipole_spectra_err_3.h5`
- `../dipole_coord_list.npz`
- `../dipole.py`
- `../utils.py`
- `../requirements.yaml`
- `../trap_completeness_method.md`

Output artifacts:

- `../cache/00_data_inventory.json`
- `../cache/00_fits_by_temperature_dtph.csv`

Short numerical summary:

- `1004` FITS files scanned.
- `1004` FITS files parsed successfully.
- `0` unparsed FITS filenames.
- `23 / 23` expected Method 3 temperatures present.
- `1` extra temperature found: `230 K` (`2` FITS files). These are dummy files. Ignore them. 
- `27` distinct `dtph` groups including `MISSING`.
- Primary HDF5 size: `578,885,472` bytes.
- Comparison HDF5 size: `579,665,032` bytes.
- `dipole_coord_list.npz` size: `83,766` bytes.

Required checks:

- `proc/` exists and is non-empty: PASS.
- `fit_dipole_spectra_err_4.h5` present: PASS.
- `fit_dipole_spectra_err_3.h5` present: PASS.
- Canonical WSL runtime imports `numpy`, `scipy`, `h5py`, `astropy`, and `iminuit`: PASS.
- Every Method 3 temperature listed in `trap_completeness_method.md` has at least one FITS file: PASS.

Stop conditions encountered:

- None.


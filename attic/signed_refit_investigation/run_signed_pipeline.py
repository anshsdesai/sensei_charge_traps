"""Regenerate the dipole pipeline with signed intensities, physical errors, and the offset fit.

Produces versioned artifacts alongside the legacy ones (originals untouched):
  - pair_noise_table_manifest_v1.npz
  - dipole_coord_list_signed_manifest_v1.npz
  - dipole_spectra_signed_manifest_v1.h5
  - fit_dipole_spectra_signed_manifest_v1_abssigma_err_4.h5
      3-parameter offset fits with absolute supplied errors

All stages consume signed_refit_manifest.csv and stamp its SHA-256 into their
outputs. Each stage is skipped if its matching output already exists.
"""

import argparse
import re

import h5py
import numpy as np

from utils import (
    get_qdata,
    crop_qdata,
    approximate_electronize,
    save_spectra_hdf5,
    load_spectra_hdf5,
)
from dipole_new import getDipoleList2, getDipoleSpectra2, fitTrapIntensity
from signed_refit_manifest import load_selected_image_files

NOISE_TABLE_FILE = 'pair_noise_table_manifest_v1.npz'
COORDS_FILE = 'dipole_coord_list_signed_manifest_v1.npz'
SPECTRA_FILE = 'dipole_spectra_signed_manifest_v1.h5'
FITS_FILE = 'fit_dipole_spectra_signed_manifest_v1_abssigma_err_4.h5'
GOOD_QUADS = [0, 1, 2, 3]
ELECTRONIZE_EVAL = 400
N_NOISE_PAIRS = 4000
NOISE_SEED = 20260612


def parse_temperatures(image_files):
    temps = set()
    for image in image_files:
        found = re.findall(r'_(\d+)k', image)
        if found:
            temps.add(int(found[0]))
    return sorted(temps)


def build_noise_table(image_files, temperatures):
    """Temporal pair noise sigma_base(T, quad): per-pair std of I=(a-b)/2 across
    the dtph scan for random (trap-free, to leading order) pixel pairs."""
    rng = np.random.default_rng(NOISE_SEED)
    table = {}
    for temp in temperatures:
        files = sorted(f for f in image_files if re.search(fr'_{temp}k_', f))
        for quad in GOOD_QUADS:
            pair_values = []
            rr = cc = None
            for f in files:
                img = approximate_electronize(crop_qdata(get_qdata(f, quad)), ELECTRONIZE_EVAL).astype(float)
                med = np.median(img, axis=1)
                img = (img.T - med).T
                if rr is None:
                    nr, nc = img.shape
                    rr = rng.integers(1, nr, N_NOISE_PAIRS)
                    cc = rng.integers(0, nc, N_NOISE_PAIRS)
                pair_values.append((img[rr, cc] - img[rr - 1, cc]) / 2.0)
            pair_values = np.array(pair_values)  # (nimg, npairs)
            sigma_base = float(np.median(np.std(pair_values, axis=0, ddof=1)))
            table[(temp, quad)] = sigma_base
            print(f"noise table: {temp} K quad {quad}: sigma_base = {sigma_base:.1f} e- ({len(files)} images)")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image_dir', default='proc/')
    parser.add_argument('--manifest', default='signed_refit_manifest.csv')
    parser.add_argument('--well_behaved_threshold', type=int, default=4)
    args = parser.parse_args()

    image_files, manifest_sha256 = load_selected_image_files(args.manifest)
    temperatures = parse_temperatures(image_files)
    print(f"Manifest: {args.manifest} ({manifest_sha256})")
    print(f"Selected images: {len(image_files)}; temperatures: {temperatures}")

    # Stage 0: temporal pair-noise table.
    try:
        npz = np.load(NOISE_TABLE_FILE)
        if str(npz['manifest_sha256']) != manifest_sha256:
            raise RuntimeError(f"{NOISE_TABLE_FILE} was produced from a different manifest")
        noise_table = {
            (int(t), int(q)): float(s)
            for t, q, s in zip(npz['temperature_K'], npz['quadrant'], npz['sigma_base_e'])
        }
        print(f"Loaded {NOISE_TABLE_FILE} ({len(noise_table)} entries)")
    except FileNotFoundError:
        print("Building temporal pair-noise table...")
        noise_table = build_noise_table(image_files, temperatures)
        keys = sorted(noise_table)
        np.savez(
            NOISE_TABLE_FILE,
            temperature_K=np.array([k[0] for k in keys]),
            quadrant=np.array([k[1] for k in keys]),
            sigma_base_e=np.array([noise_table[k] for k in keys]),
            n_pairs=N_NOISE_PAIRS,
            seed=NOISE_SEED,
            manifest_sha256=manifest_sha256,
        )

    # Stage 1: dipole coordinates with the robust finder.
    try:
        coords_npz = np.load(COORDS_FILE)
        if str(coords_npz['manifest_sha256']) != manifest_sha256:
            raise RuntimeError(f"{COORDS_FILE} was produced from a different manifest")
        full_dipole_coord_list = []
        for i in range(len(GOOD_QUADS)):
            if f'quad_idx_{i}' in coords_npz:
                full_dipole_coord_list.append([tuple(c) for c in coords_npz[f'quad_idx_{i}']])
            else:
                full_dipole_coord_list.append([])
        print(f"Loaded {COORDS_FILE}")
    except FileNotFoundError:
        print("Finding dipoles (robust sigma, no symmetry cut)...")
        full_dipole_coord_list = getDipoleList2(
            args.image_dir,
            temperatures,
            GOOD_QUADS,
            robust_sigma=True,
            symmetry_perc=None,
            image_files=image_files,
        )
        save_dict = {f'quad_idx_{i}': np.array(c) for i, c in enumerate(full_dipole_coord_list)}
        save_dict['manifest_sha256'] = np.array(manifest_sha256)
        np.savez(COORDS_FILE, **save_dict)
    print("Dipoles per quadrant:", [len(c) for c in full_dipole_coord_list])

    # Stage 2: signed spectra with physical errors.
    import os
    if os.path.exists(SPECTRA_FILE):
        with h5py.File(SPECTRA_FILE, 'r') as h5:
            if h5.attrs.get('manifest_sha256', '') != manifest_sha256:
                raise RuntimeError(f"{SPECTRA_FILE} was produced from a different manifest")
        print(f"Loading {SPECTRA_FILE}")
        full_dipole_dict = load_spectra_hdf5(SPECTRA_FILE)
    else:
        print("Extracting signed spectra with physical errors...")
        full_dipole_dict = getDipoleSpectra2(
            args.image_dir,
            GOOD_QUADS,
            full_dipole_coord_list,
            absolute=False,
            error_model='physical',
            noise_table=noise_table,
            image_files=image_files,
        )
        save_spectra_hdf5(full_dipole_dict, SPECTRA_FILE)
        with h5py.File(SPECTRA_FILE, 'a') as h5:
            h5.attrs['manifest_sha256'] = manifest_sha256
        print(f"Saved {SPECTRA_FILE}")

    # Stage 3: 3-parameter offset fits.
    if os.path.exists(FITS_FILE):
        with h5py.File(FITS_FILE, 'r') as h5:
            if h5.attrs.get('manifest_sha256', '') != manifest_sha256:
                raise RuntimeError(f"{FITS_FILE} was produced from a different manifest")
        print(f"{FITS_FILE} already exists; nothing to do.")
        return
    print("Fitting trap intensities (offset model, signed amplitudes)...")
    fitTrapIntensity(
        full_dipole_dict,
        useIntensityErr=True,
        wellBehavedThreshold=args.well_behaved_threshold,
        fit_offset=True,
        errors_are_absolute=True,
    )
    save_spectra_hdf5(full_dipole_dict, FITS_FILE)
    with h5py.File(FITS_FILE, 'a') as h5:
        h5.attrs['manifest_sha256'] = manifest_sha256
    print(f"Saved {FITS_FILE}")


if __name__ == '__main__':
    main()

import argparse
import glob
import json
import os
import re
import pickle
import importlib
import warnings
import numpy as np
from collections import Counter, defaultdict

# Silence the (benign) astropy FITS-header warnings emitted on every image read
# ("non-ASCII characters in header", "Unexpected bytes trailing END keyword").
# Without this they print once per file (~thousands of times), burying progress
# and slowing the run to a crawl. Set process-wide before any FITS is opened.
try:
    from astropy.utils.exceptions import AstropyWarning
    warnings.filterwarnings('ignore', category=AstropyWarning)
except Exception:
    pass

# Import dependencies as defined in the notebook
from utils import *

# Temporal pair-noise model parameters (minimal pipeline only).
ELECTRONIZE_EVAL = 400
N_NOISE_PAIRS = 4000
NOISE_SEED = 20260612

# Detection-calibration parameters (minimal pipeline, --detection calibrated).
DETECTION_SEED = 20260615
N_CONTROL_PER_QUAD = 2000
DETECTION_TARGET_FPR = 0.001


def build_noise_table(image_files, temperatures, good_quads):
    """Temporal pair-noise sigma_base(T, quad): per-pair std of I=(a-b)/2 across
    the dtph scan for random (trap-free, to leading order) pixel pairs. This is
    the physical per-point error used by the minimal pipeline, replacing the
    legacy spatial-patch scatter (which overestimates the fixed-pair noise)."""
    rng = np.random.default_rng(NOISE_SEED)
    table = {}
    for temp in temperatures:
        temp = int(temp)
        files = sorted(f for f in image_files if re.search(fr'_{temp}k_', f))
        if not files:
            continue
        print(f"noise table: reading {len(files)} images for {temp} K...", flush=True)
        for quad in good_quads:
            pair_values = []
            rr = cc = None
            for f in files:
                try:
                    img = approximate_electronize(
                        crop_qdata(get_qdata(f, quad)), ELECTRONIZE_EVAL
                    ).astype(float)
                except Exception as exc:
                    print(f"  WARNING: skipping unreadable image {f}: {exc}", flush=True)
                    continue
                med = np.median(img, axis=1)
                img = (img.T - med).T
                if rr is None:
                    nr, nc = img.shape
                    rr = rng.integers(1, nr, N_NOISE_PAIRS)
                    cc = rng.integers(0, nc, N_NOISE_PAIRS)
                pair_values.append((img[rr, cc] - img[rr - 1, cc]) / 2.0)
            pair_values = np.array(pair_values)
            sigma_base = float(np.median(np.std(pair_values, axis=0, ddof=1)))
            table[(temp, int(quad))] = sigma_base
            print(f"noise table: {temp} K quad {quad}: "
                  f"sigma_base = {sigma_base:.1f} e- ({len(files)} images)", flush=True)
    return table


def load_or_build_noise_table(noise_file, image_files, temperatures, good_quads, recompute):
    if os.path.exists(noise_file) and not recompute:
        npz = np.load(noise_file)
        table = {
            (int(t), int(q)): float(s)
            for t, q, s in zip(npz['temperature_K'], npz['quadrant'],
                               npz['sigma_base_e'])
        }
        print(f"Loaded {noise_file} ({len(table)} entries)")
        return table
    print(f"Building temporal pair-noise table -> {noise_file} ...")
    table = build_noise_table(image_files, temperatures, good_quads)
    keys = sorted(table)
    np.savez(
        noise_file,
        temperature_K=np.array([k[0] for k in keys]),
        quadrant=np.array([k[1] for k in keys]),
        sigma_base_e=np.array([table[k] for k in keys]),
        n_pairs=N_NOISE_PAIRS,
        seed=NOISE_SEED,
    )
    return table


def build_detection_table(dp, image_dir, good_quads, candidate_coords, noise_table,
                          grid_width, grid_height, target_fpr, use_intensity_err,
                          well_behaved_threshold):
    """Empirical per-temperature Delta-chi2 detection threshold.

    Samples trap-free control pixel pairs, runs the IDENTICAL signed offset fit
    used for candidates (so the null statistic is computed exactly the same way),
    and takes the (1 - target_fpr) quantile of the control delta_chi2_vs_constant
    per temperature. This replaces the uncalibrated fixed 11.83 cut, which is not
    a true 3-sigma value because tau is unidentified and scanned under the null.
    """
    rng = np.random.default_rng(DETECTION_SEED)
    # Restrict control coordinates to the bounding box of real candidates so the
    # indices are guaranteed valid in the cropped image frame.
    all_rows = [int(c[0]) for q in candidate_coords for c in q]
    all_cols = [int(c[1]) for q in candidate_coords for c in q]
    if all_rows:
        rmin, rmax = max(2, min(all_rows)), min(grid_height - 2, max(all_rows))
        cmin, cmax = max(0, min(all_cols)), min(grid_width - 2, max(all_cols))
    else:
        rmin, rmax, cmin, cmax = 10, grid_height - 15, 10, grid_width - 15

    control_coords = []
    for qi in range(len(good_quads)):
        cand = set(tuple(c) for c in candidate_coords[qi]) if qi < len(candidate_coords) else set()
        coords, attempts = [], 0
        while len(coords) < N_CONTROL_PER_QUAD and attempts < N_CONTROL_PER_QUAD * 50:
            attempts += 1
            r = int(rng.integers(rmin, rmax + 1))
            c = int(rng.integers(cmin, cmax + 1))
            if (r, c) in cand or (r - 1, c) in cand or (r + 1, c) in cand:
                continue
            coords.append((r, c))
        control_coords.append(coords)
    print(f"Detection calibration: {sum(len(c) for c in control_coords)} "
          f"trap-free control pairs (target FPR={target_fpr:g})")

    control_spectra = dp.getDipoleSpectra2(
        image_dir, good_quads, control_coords,
        absolute=False, error_model='physical', noise_table=noise_table,
    )
    # delta_chi2_threshold=-inf: never filter, we only want the stored statistic.
    control_fits = dp.fitTrapIntensity(
        control_spectra, useIntensityErr=use_intensity_err,
        wellBehavedThreshold=well_behaved_threshold,
        fit_offset=True, errors_are_absolute=True,
        delta_chi2_threshold=-np.inf,
    )
    per_temp = defaultdict(list)
    for q in good_quads:
        if q not in control_fits:
            continue
        for dpkey, d in control_fits[q].items():
            if not isinstance(dpkey, tuple):
                continue
            for temp, td in d.items():
                if not isinstance(temp, int):
                    continue
                v = td.get('delta_chi2_vs_constant')
                if v is not None and np.isfinite(v):
                    per_temp[temp].append(float(v))
    table = {}
    for temp in sorted(per_temp):
        vals = np.array(per_temp[temp])
        thr = float(np.quantile(vals, 1.0 - target_fpr))
        table[int(temp)] = thr
        print(f"  {temp} K: {vals.size} controls, threshold = {thr:.2f}")
    return table


def load_or_build_detection_table(det_file, dp, image_dir, good_quads, candidate_coords,
                                  noise_table, grid_width, grid_height, target_fpr,
                                  use_intensity_err, well_behaved_threshold, recompute):
    if os.path.exists(det_file) and not recompute:
        npz = np.load(det_file)
        table = {int(t): float(v) for t, v in zip(npz['temperature_K'], npz['threshold'])}
        print(f"Loaded {det_file} ({len(table)} thresholds)")
        return table
    print(f"Building detection calibration -> {det_file} ...")
    table = build_detection_table(
        dp, image_dir, good_quads, candidate_coords, noise_table,
        grid_width, grid_height, target_fpr, use_intensity_err, well_behaved_threshold,
    )
    temps = sorted(table)
    np.savez(
        det_file,
        temperature_K=np.array(temps),
        threshold=np.array([table[t] for t in temps]),
        target_fpr=target_fpr,
        n_control_per_quad=N_CONTROL_PER_QUAD,
        seed=DETECTION_SEED,
    )
    return table


def run_analysis(
    image_dir='proc/',
    image_dir_search='proc/*.fits',
    good_quads=[0, 1, 2, 3],
    grid_width=3072,
    grid_height=512,
    num_samples=10000,
    num_bins=100,
    confidence_level=0.90,
    use_intensity_err=True,
    well_behaved_threshold=4,
    pipeline='legacy',
    detection='fixed',
    delta_chi2_threshold=11.83,
    detection_target_fpr=DETECTION_TARGET_FPR,
    overwrite='none',
):
    minimal = (pipeline == 'minimal')
    # Select the dipole helper module: legacy baseline vs minimal synthesis
    # (pedestal + temporal pair-noise + absolute sigma + sign consistency +
    # clean SRH p-value). Outputs are suffixed so the two never collide.
    dp = importlib.import_module('dipole_new' if minimal else 'dipole')
    suffix = '_minimal' if minimal else ''
    # Detection mode only affects the minimal fit + downstream; tag those files
    # so 'fixed' and 'calibrated' runs coexist without overwriting each other.
    det_tag = '_caldet' if (minimal and detection == 'calibrated') else ''
    redo_inputs = (overwrite == 'all')
    redo_fit = overwrite in ('all', 'fit')
    print(f"--- Charge Traps Analysis (pipeline={pipeline}, detection="
          f"{detection if minimal else 'n/a'}, overwrite={overwrite}) ---")

    # 1. Parse Temperatures
    temperatures = []
    for image in glob.glob(image_dir_search):
        found = re.findall(r'_\d+k', image)
        if not found:
            continue
        if 'dtph' not in image:
            continue
        temperatures.append(found[0][1:-1])

    temperatures_strs = np.array(temperatures)
    print("Temperature counts:", Counter(temperatures_strs))
    temps = np.sort(np.array([int(t) for t in temperatures_strs]))
    test_temps = np.unique(temps)
    print("Unique temperatures:", test_temps)

    # 2. Load or Compute Dipole Coordinate List
    coords_file = f'dipole_coord_list{suffix}.npz'
    if os.path.exists(coords_file) and not redo_inputs:
        coords_npz = np.load(coords_file)
        full_dipole_coord_list = []
        for i in range(len(good_quads)):
            if f'quad_idx_{i}' in coords_npz:
                full_dipole_coord_list.append([tuple(c) for c in coords_npz[f'quad_idx_{i}']])
            else:
                full_dipole_coord_list.append([])
        print(f"Loaded {coords_file}")
    else:
        print(f"Computing {coords_file} ...")
        if minimal:
            # Robust MAD threshold and no amplitude-symmetry cut (the pedestal
            # makes genuine dipoles asymmetric at high T).
            full_dipole_coord_list = dp.getDipoleList2(
                image_dir, test_temps, good_quads,
                robust_sigma=True, symmetry_perc=None,
            )
        else:
            full_dipole_coord_list = dp.getDipoleList2(image_dir, test_temps, good_quads)
        save_dict = {f'quad_idx_{i}': np.array(coords) for i, coords in enumerate(full_dipole_coord_list)}
        np.savez(coords_file, **save_dict)

    total = sum(len(q_list) for q_list in full_dipole_coord_list)
    print(f"Total dipoles: {total}")

    # 3. Monte Carlo Distance Histograms
    mc_file = f'mc_dist{suffix}.npz'
    if os.path.exists(mc_file) and not redo_inputs:
        mcdict = np.load(mc_file)
        montecarlo_hists = mcdict['montecarlo_hists']
        mc_bin_centers = mcdict['mc_bin_centers']
        mc_mean = mcdict['mc_mean']
        ci_lower = mcdict['ci_lower']
        ci_upper = mcdict['ci_upper']
        print(f"Loaded {mc_file}")
    else:
        print(f"Computing {mc_file} ...")
        mc_bin_centers, mc_mean, ci_lower, ci_upper, montecarlo_hists = dp.monte_carlo_distance_histograms(
            n_points=total // 4,
            grid_width=grid_width,
            grid_height=grid_height,
            num_samples=num_samples,
            num_bins=num_bins,
            confidence_level=confidence_level,
            spline_smoothing=0,
            return_spline=False
        )
        np.savez(mc_file,
                 mc_bin_centers=mc_bin_centers,
                 mc_mean=mc_mean,
                 ci_lower=ci_lower,
                 ci_upper=ci_upper,
                 montecarlo_hists=montecarlo_hists)

    # 3b. Temporal pair-noise table (minimal pipeline only)
    noise_table = None
    if minimal:
        # Use the SAME image selection as getDipoleSpectra2 (dtph science frames,
        # CCD 2) rather than a raw *.fits glob, which would also pick up non-dtph
        # / other-CCD files whose data section astropy cannot read.
        noise_image_files = sorted(glob.glob(image_dir + 'proc*_*dtph**_2_*'))
        noise_table = load_or_build_noise_table(
            f'pair_noise_table{suffix}.npz',
            noise_image_files,
            test_temps,
            good_quads,
            redo_inputs,
        )

    # 3c. Empirical detection threshold table (minimal + calibrated only)
    delta_chi2_table = None
    if minimal and detection == 'calibrated':
        delta_chi2_table = load_or_build_detection_table(
            f'detection_calibration{suffix}.npz',
            dp, image_dir, good_quads, full_dipole_coord_list, noise_table,
            grid_width, grid_height, detection_target_fpr,
            use_intensity_err, well_behaved_threshold,
            redo_fit,
        )

    # 4. Dipole Spectra
    spectra_file = f'dipole_spectra{suffix}.h5'
    if os.path.exists(spectra_file) and not redo_inputs:
        dipole_spectra = load_spectra_hdf5(spectra_file)
        print(f"Loaded {spectra_file}")
    else:
        print(f"Computing {spectra_file} ...")
        if minimal:
            dipole_spectra = dp.getDipoleSpectra2(
                image_dir, good_quads, full_dipole_coord_list,
                absolute=False, error_model='physical', noise_table=noise_table,
            )
        else:
            dipole_spectra = dp.getDipoleSpectra2(image_dir, good_quads, full_dipole_coord_list)
        save_spectra_hdf5(dipole_spectra, spectra_file)

    # 5. Fit Trap Intensity
    threshold_str = f'_{well_behaved_threshold}'
    threshold_tag = threshold_str if well_behaved_threshold != 4 else ''
    intensity_str = '_err' if use_intensity_err else ''
    spectra_filename = f'fit_dipole_spectra{suffix}{det_tag}{intensity_str}{threshold_str}.h5'

    if os.path.exists(spectra_filename) and not redo_fit:
        fit_dipole_spectra = load_spectra_hdf5(spectra_filename)
        print(f"Loaded {spectra_filename}")
    else:
        print(f"Computing {spectra_filename} ...")
        if minimal:
            fit_dipole_spectra = dp.fitTrapIntensity(
                dipole_spectra,
                useIntensityErr=use_intensity_err,
                wellBehavedThreshold=well_behaved_threshold,
                fit_offset=True,
                errors_are_absolute=True,
                delta_chi2_threshold=delta_chi2_threshold,
                delta_chi2_table=delta_chi2_table,
            )
        else:
            fit_dipole_spectra = dp.fitTrapIntensity(
                dipole_spectra,
                useIntensityErr=use_intensity_err,
                wellBehavedThreshold=well_behaved_threshold,
            )
        save_spectra_hdf5(fit_dipole_spectra, spectra_filename)

    # 6. Extract Tau at 135K for Simulation Sampling
    print("Calculating tau_e at 135K distribution...")
    tau_at_135s = []

    for q in good_quads:
        if q not in fit_dipole_spectra:
            continue
        for dpkey in list(fit_dipole_spectra[q]):
            if type(dpkey) != tuple:
                continue
            testdp = fit_dipole_spectra[q][dpkey]
            if testdp.get('WellBehavedTrap', False) and not testdp.get('EnergyFitFailed', True):
                if testdp.get('GoodEnergyFit', False):
                    cs = testdp.get('energy_BestFitCrossSection')
                    e = testdp.get('energy_BestFitEnergy')
                    if cs is not None and e is not None:
                        logtau_at_135 = dp.log_energy_cross_section(135, e, np.log(cs))
                        tau_at_135s.append(np.exp(logtau_at_135))

    tau_at_135s = np.array(tau_at_135s)
    tau_at_135s = tau_at_135s[np.isfinite(tau_at_135s)]
    if len(tau_at_135s) == 0:
        raise RuntimeError("No well-behaved traps with good energy fits found - tau histogram would be empty.")
    # Histogram floor stays at 1e-7 s (fast traps below this are negligible for
    # the SER and are intentionally dropped). The ceiling MUST cover the
    # long-lived tail: the high-energy (~0.5 eV) population reaches tau(135K) of
    # 1e8-1e10 s, and the legacy fixed 1e8 s upper edge silently clipped exactly
    # this SER-relevant, mask-immune population out of the sampled distribution.
    tau_lo = 1e-7
    n_below = int(np.sum(tau_at_135s < tau_lo))
    n_above_legacy = int(np.sum(tau_at_135s > 1e8))
    tau_hi = max(1e8, float(tau_at_135s.max()) * 1.05)
    bins = np.geomspace(tau_lo, tau_hi, 100)
    hist, bin_edges = np.histogram(tau_at_135s, bins=bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    print(f"tau(135K): {tau_at_135s.size} traps, "
          f"range [{tau_at_135s.min():.2e}, {tau_at_135s.max():.2e}] s; "
          f"{n_above_legacy} extend past the old 1e8 s cap (now retained); "
          f"{n_below} fast traps below {tau_lo:g} s dropped.")

    tau_hist_file = f'tau_at_135k_hist{suffix}{det_tag}{threshold_tag}.npz'
    np.savez(tau_hist_file, tau_at_135s=tau_at_135s, hist=hist, bin_edges=bin_edges, bin_centers=bin_centers)
    print(f"Saved {tau_hist_file} for simulation sampling.")

    # 7. Per-trap (tau135, sigma) pairs for the simulation's SRH capture model.
    # Refits from the stored per-temperature taus, so it is correct even when
    # the fit HDF5 cache predates a constants change in log_energy_cross_section.
    from make_trap_pairs import make_pairs
    make_pairs(fitfile=spectra_filename, out=f'trap_tau135_sigma_pairs{suffix}{det_tag}{threshold_tag}.npz')

    print("--- Analysis Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run charge traps analysis.")

    parser.add_argument('--image_dir', type=str, default='proc/', help="Directory containing images")
    parser.add_argument('--image_dir_search', type=str, default='proc/*.fits', help="Search string for globbing images")
    parser.add_argument('--pipeline', choices=['legacy', 'minimal'], default='legacy',
                        help="legacy: through-zero fit, spatial-patch errors, symmetry-cut finder. "
                             "minimal: pedestal + temporal pair-noise + absolute sigma + sign "
                             "consistency + clean SRH p-value (uses dipole_new.py). Outputs are "
                             "suffixed with _minimal so the two never overwrite each other.")
    parser.add_argument('--detection', choices=['fixed', 'calibrated'], default='fixed',
                        help="Minimal pipeline only. fixed: use --delta_chi2_threshold (default "
                             "11.83, an UNCALIBRATED guard). calibrated: build the empirical "
                             "per-temperature Delta-chi2 threshold from trap-free control pairs "
                             "at --detection_target_fpr and use it automatically. Calibrated "
                             "outputs carry a _caldet tag so they coexist with the fixed run.")
    parser.add_argument('--delta_chi2_threshold', type=float, default=11.83,
                        help="Fixed-detection Delta-chi2 guard (minimal pipeline). NOT a true "
                             "3-sigma value (tau is unidentified+scanned under the null); vary to "
                             "test sensitivity.")
    parser.add_argument('--detection_target_fpr', type=float, default=DETECTION_TARGET_FPR,
                        help="Target per-temperature false-positive rate for --detection calibrated.")
    parser.add_argument('--overwrite', choices=['none', 'fit', 'all'], default='none',
                        help="none: reuse all cached stages. fit: recompute the fit, detection "
                             "table, tau(135K), and trap pairs only (reuse coords/MC/noise/spectra) "
                             "- use this for threshold/calibration scans. all: recompute every stage.")
    parser.add_argument('--grid_width', type=int, default=3072, help="Grid width for Monte Carlo")
    parser.add_argument('--grid_height', type=int, default=512, help="Grid height for Monte Carlo")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of Monte Carlo samples")
    parser.add_argument('--num_bins', type=int, default=100, help="Number of bins for histograms")
    parser.add_argument('--confidence_level', type=float, default=0.90, help="Confidence level for CI bounds")
    parser.add_argument('--no_intensity_err', action='store_false', dest='use_intensity_err', help="Disable useIntensityErr flag")
    parser.add_argument('--well_behaved_threshold', type=int, default=4, help="Threshold for well behaved signals")

    args = parser.parse_args()

    run_analysis(
        image_dir=args.image_dir,
        image_dir_search=args.image_dir_search,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        num_samples=args.num_samples,
        num_bins=args.num_bins,
        confidence_level=args.confidence_level,
        use_intensity_err=args.use_intensity_err,
        well_behaved_threshold=args.well_behaved_threshold,
        pipeline=args.pipeline,
        detection=args.detection,
        delta_chi2_threshold=args.delta_chi2_threshold,
        detection_target_fpr=args.detection_target_fpr,
        overwrite=args.overwrite,
    )

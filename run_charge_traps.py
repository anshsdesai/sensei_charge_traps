import argparse
import glob
import json
import re
import pickle
import numpy as np
from collections import Counter

# Import dependencies as defined in the notebook
from utils import *
from dipole import *

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
    well_behaved_threshold=4
):
    print("--- Starting Charge Traps Analysis ---")
    
    # 1. Parse Temperatures
    temperatures = []
    for image in glob.glob(image_dir_search):
        # Using a raw string r'' to fix the \d escape sequence warning
        found = re.findall(r'_\d+k', image)
        if not found:
            continue
        temp = found[0][1:-1]
        
        if 'dtph' not in image:
            continue
            
        temperatures.append(temp)
        
    temperatures_strs = np.array(temperatures)
    print("Temperature counts:", Counter(temperatures_strs))
    
    temps = np.sort(np.array([int(t) for t in temperatures_strs]))
    test_temps = np.unique(temps)
    print("Unique temperatures:", test_temps)

    # 2. Load or Compute Dipole Coordinate List
    try:
        coords_npz = np.load('dipole_coord_list.npz')
        full_dipole_coord_list = []
        for i in range(len(good_quads)):
            if f'quad_idx_{i}' in coords_npz:
                full_dipole_coord_list.append([tuple(c) for c in coords_npz[f'quad_idx_{i}']])
            else:
                full_dipole_coord_list.append([])
        print("Loaded dipole_coord_list.npz")
    except FileNotFoundError:
        print("dipole_coord_list.npz not found. Computing...")
        full_dipole_coord_list = getDipoleList2(image_dir, test_temps, good_quads)
        save_dict = {f'quad_idx_{i}': np.array(coords) for i, coords in enumerate(full_dipole_coord_list)}
        np.savez('dipole_coord_list.npz', **save_dict)

    # Calculate total dipoles across valid quadrants
    total = sum(len(full_dipole_coord_list[q]) for q in good_quads)
    total = sum(len(q_list) for q_list in full_dipole_coord_list)
    print(f"Total dipoles: {total}")

    # 3. Monte Carlo Distance Histograms
    try:
        mcdict = np.load('mc_dist.npz', allow_pickle=True)
        mcdict = np.load('mc_dist.npz')
        montecarlo_hists = mcdict['montecarlo_hists']
        mc_bin_centers = mcdict['mc_bin_centers']
        mc_mean = mcdict['mc_bin_cemc_meannters']
        mc_mean = mcdict['mc_mean']
        ci_lower = mcdict['ci_lower']
        ci_upper = mcdict['ci_upper']
        print("Loaded mc_dist.npz")
    except FileNotFoundError:
        print("mc_dist.npz not found. Computing histograms...")
        mc_bin_centers, mc_mean, ci_lower, ci_upper, montecarlo_hists = monte_carlo_distance_histograms(
            n_points=total // 4,
            grid_width=grid_width,
            grid_height=grid_height,
            num_samples=num_samples,
            num_bins=num_bins,
            confidence_level=confidence_level,
            spline_smoothing=0,
            return_spline=False
        )
        
        np.savez('mc_dist.npz',
                 mc_bin_centers=mc_bin_centers,
                 mc_bin_cemc_meannters=mc_mean,
                 mc_mean=mc_mean,
                 ci_lower=ci_lower,
                 ci_upper=ci_upper,
                 montecarlo_hists=montecarlo_hists)


    # 4. Dipole Spectra
    try:
        dipole_spectra = load_spectra_hdf5('dipole_spectra.h5')
        print("Loaded dipole_spectra.h5")
    except FileNotFoundError:
        print("dipole_spectra.h5 not found. Computing...")
        dipole_spectra = getDipoleSpectra2(image_dir, good_quads, full_dipole_coord_list)
        save_spectra_hdf5(dipole_spectra, 'dipole_spectra.h5')

    # 5. Fit Trap Intensity
    threshold_str = f'_{well_behaved_threshold}'
    intensity_str = '_err' if use_intensity_err else ''
    spectra_filename = f'fit_dipole_spectra{intensity_str}{threshold_str}.h5'
    
    try:
        fit_dipole_spectra = load_spectra_hdf5(spectra_filename)

        print(f"Loaded {spectra_filename}")
    except FileNotFoundError:
        print(f"{spectra_filename} not found. Computing...")
        fit_dipole_spectra = fitTrapIntensity(
            dipole_spectra, 
            useIntensityErr=use_intensity_err, 
            wellBehavedThreshold=well_behaved_threshold
        )
        save_spectra_hdf5(fit_dipole_spectra, spectra_filename)
            
    # 6. Extract Tau at 135K for Simulation Sampling
    print("Calculating tau_e at 135K distribution...")
    tau_at_135s = []
    
    for q in good_quads:
        if q not in fit_dipole_spectra:
            continue
        dpkeys = list(fit_dipole_spectra[q])
        for dp in dpkeys:
            if type(dp) != tuple:
                continue
            testdp = fit_dipole_spectra[q][dp]
            
            if testdp.get('WellBehavedTrap', False) and not testdp.get('EnergyFitFailed', True):
                if testdp.get('GoodEnergyFit', False):
                    # fitTrapIntensity stores these directly on the trap dict
                    cs = testdp.get('energy_BestFitCrossSection')
                    e = testdp.get('energy_BestFitEnergy')

                    if cs is not None and e is not None:
                        logtau_at_135 = log_energy_cross_section(135, e, np.log(cs))
                        tau_at_135 = np.exp(logtau_at_135)
                        tau_at_135s.append(tau_at_135)

    tau_at_135s = np.array(tau_at_135s)
    if len(tau_at_135s) == 0:
        raise RuntimeError("No well-behaved traps with good energy fits found - tau histogram would be empty.")
    bins = np.geomspace(1e-7, 1e8, 100)
    hist, bin_edges = np.histogram(tau_at_135s, bins=bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    np.savez('tau_at_135k_hist.npz', tau_at_135s=tau_at_135s, hist=hist, bin_edges=bin_edges, bin_centers=bin_centers)
    print("Saved tau_at_135k_hist.npz for simulation sampling.")

    # 7. Per-trap (tau135, sigma) pairs for the simulation's SRH capture model.
    # Refits from the stored per-temperature taus, so it is correct even when
    # the fit HDF5 cache predates a constants change in log_energy_cross_section.
    from make_trap_pairs import make_pairs
    make_pairs(fitfile=spectra_filename, out='trap_tau135_sigma_pairs.npz')

    print("--- Analysis Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run charge traps analysis.")
    
    parser.add_argument('--image_dir', type=str, default='proc/', help="Directory containing images")
    parser.add_argument('--image_dir_search', type=str, default='proc/*.fits', help="Search string for globbing images")
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
        well_behaved_threshold=args.well_behaved_threshold
    )
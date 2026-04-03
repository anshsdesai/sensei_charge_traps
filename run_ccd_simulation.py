import sys
import os
import numpy as np
from astropy.io import fits
# from utils import *
from ccd_simulation import *
from tqdm.autonotebook import tqdm
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm.autonotebook import tqdm
import itertools
from ccd_simulation import run_single_trial
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parallel CCD charge trap simulations.")
    parser.add_argument('--num_runs', type=int, default=500, help="Number of simulation trials to run.")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of CPU workers (defaults to max cores - 1).")
    parser.add_argument('--runconditions', type=str, default='minos', choices=['minos', 'snolab'], help="Run conditions configuration to use.")
    parser.add_argument('--binning', type=float, default=1.0, help="Scale factor to divide pixel readout times (simulates binning).")
    
    args = parser.parse_args()

    snolab_dir = './snolab_image/'
    file = 'proc_corr_proc_skp_sensei_2023-02-14_135K_run7_commissioning_NROW520_NBINROW1_NCOL3200_NBINCOL1_EXPOSURE72000_CLEAR10800_5_83.fits'
    
    try:
        with fits.open(snolab_dir + file) as hdul:
            q = hdul[0]
            header = q.header
            nrow=header['NROW']
            ncol=header['NCOL']
            exposure=header['EXPOSURE']
            nsamp=header['NSAMP']
            delayH=header['HIERARCH DELAY_H_OVERLAP']
            delayRG=header['HIERARCH DELAY_RG_WIDTH']
            delayIped=header['HIERARCH DELAY_INTEG_PED']
            delaySW=header['HIERARCH DELAY_SWHIGH']
            delayIsig=header['HIERARCH DELAY_INTEG_SIG']
            delayOG=header['HIERARCH DELAY_OG_LOW']
            delayDG=header['HIERARCH DELAY_DG_LOW']
    except FileNotFoundError:
        print(f"Error: Could not find reference FITS file {snolab_dir + file}")
        sys.exit(1)

    tpix = (pixel_time(nsamp, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG) / 15e6) / args.binning
    tpix_vertical = (pixel_time_vertical(nsamp, ncol, delayH, delayIped, delayIsig, delaySW, delayRG, delayOG, delayDG) / 15e6) / args.binning

    # Load tau distribution for simulation sampling
    try:
        tau_data = np.load('tau_at_135k_hist.npz')
        tau_weights = tau_data['hist']
        tau_values = tau_data['bin_centers']
        print("Loaded tau_at_135k_hist.npz successfully.")
    except FileNotFoundError:
        print("Error: tau_at_135k_hist.npz not found. Please run run_charge_traps.py first to generate this file.")
        sys.exit(1)

    num_runs = args.num_runs
    num_workers = args.num_workers if args.num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Starting parallel execution with {num_workers} CPU cores for {num_runs} runs...")
    print(f"Conditions: {args.runconditions}, tpix: {tpix:.3e} s, tpix_vertical: {tpix_vertical:.3e} s")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # executor.map can take multiple iterables. It stops when the shortest one (range) runs out.
        # itertools.repeat() efficiently yields the exact same memory reference over and over.
        results = list(tqdm(
            executor.map(
                run_single_trial, 
                range(num_runs),                     # r (changes 0 to num_runs-1)
                itertools.repeat(tpix),              # tpix (constant)
                itertools.repeat(tpix_vertical),     # tpix_vertical (constant)
                itertools.repeat(tau_weights),       # tau_weights (constant)
                itertools.repeat(tau_values),        # tau_values (constant)
                itertools.repeat(args.runconditions) # run conditions (string constant)
            ), 
            total=num_runs, 
            desc="Running Trials"
        ))
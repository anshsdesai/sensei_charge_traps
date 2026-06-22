import sys
import os
import numpy as np
from astropy.io import fits
# from utils import *
from ccd_simulation import *
from tqdm.autonotebook import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm.autonotebook import tqdm
import itertools
from ccd_simulation import run_single_trial
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parallel CCD charge trap simulations.")
    parser.add_argument('--num_runs', type=int, default=200, help="Number of simulation trials to run.")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of CPU workers (defaults to half the cores; each worker holds ~2.3 GB, so cores-1 can exhaust RAM).")
    parser.add_argument('--runconditions', type=str, default='minos', choices=['minos', 'snolab'], help="Run conditions configuration to use.")
    parser.add_argument('--binning', type=float, default=1.0, help="Scale factor to divide pixel readout times (simulates binning).")
    parser.add_argument('--out', type=str, default='./', help="Output directory.")
    parser.add_argument('--tauhistfile', type=str, default='tau_at_135k_hist.npz', help="the histogram file to used to sample tau values.")
    parser.add_argument('--pairsfile', type=str, default='trap_tau135_sigma_pairs.npz',
                        help="Per-trap (tau135, sigma) pairs file (from make_trap_pairs.py) used to assign capture cross-sections.")
    parser.add_argument('--packet-volume-um3', type=float, default=3.0,
                        help="Effective volume (um^3) explored by a single carrier in a pixel well; sets the capture rate kc = sigma*v_th/V. "
                             "Default 3 um^3: collecting-phase area (~12x5-10 um^2) x thermal vertical spread in the buried channel (~0.02-0.07 um). "
                             "Vary by ~a decade each way as a systematic.")
    parser.add_argument(
        '--phase-capture-ticks',
        type=float,
        default=300.0,
        help="Effective V1/V3 phase-overlap capture window in 15 MHz sequencer ticks. "
             "Default 300 ticks = one V1/V3 hold in temp_scan_run1_imgseq.xml.",
    )
    parser.add_argument(
        '--trap-density-scale',
        type=float,
        default=1.0,
        help="Multiplier applied to the baseline characterized-trap population per "
             "full CCD (the upper-limit run scales this up).",
    )
    parser.add_argument(
        '--exp-indep-charge-mode',
        choices=['pre_readout', 'post_readout'],
        default='pre_readout',
        help="Place exposure-independent single-electron charge before active-area "
             "trap transport (default; charge can be trapped) or after it "
             "(readout-generated model; not trappable).",
    )
    parser.add_argument(
        '--clear-mode',
        choices=['instantaneous', 'sequencer', 'three_hour', 'binned_0h'],
        default='sequencer',
        help="Clear free surface charge instantaneously (legacy), transport it "
             "through the temp_scan_run1_clearseq.xml clock sequence ('sequencer'), "
             "run that sequence continuously for 3 hours ('three_hour'), or use no "
             "clear at all and instead take a binned 0 h image after every real "
             "exposure ('binned_0h') to reset the array.",
    )
    parser.add_argument(
        '--binning-0h-factor',
        type=float,
        default=32.0,
        help="Row-binning factor for the 0 h images in clear-mode 'binned_0h'. "
             "Binning shortens the readout, so the per-row trap dwell for those "
             "images is tpix_vertical / binning_0h_factor (default 32).",
    )
    parser.add_argument(
        '--exposure-order',
        choices=['shuffled', 'ordered'],
        default='shuffled',
        help="Order of the per-trial exposure sequence: 'shuffled' (default) "
             "permutes all images to decouple each exposure from its predecessor; "
             "'ordered' uses the old fixed 0->4->6->10->20 cycle (in binned_0h "
             "mode, the fixed 4->6->10->20 real-exposure cycle).",
    )
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
    # fname = 'tau_at_135k_hist.npz' if not args.upperlimit  else ''
    fname = args.tauhistfile
    try:
        tau_data = np.load(fname)
        tau_weights = tau_data['hist']
        tau_edges = tau_data['bin_edges']
        print(f"Loaded {fname} successfully.")
    except FileNotFoundError:
        print(f"Error: {fname} not found. Please run run_charge_traps.py first to generate this file.")
        sys.exit(1)

    # Load measured (tau135, sigma) pairs for the SRH capture/recapture model
    try:
        pair_data = np.load(args.pairsfile)
        pair_tau135 = pair_data['tau135']
        pair_sigma = pair_data['sigma']
        print(f"Loaded {args.pairsfile} ({len(pair_tau135)} trap pairs).")
    except FileNotFoundError:
        print(f"Error: {args.pairsfile} not found. Please run make_trap_pairs.py first to generate this file.")
        sys.exit(1)

    # Baseline trap population = the number of *characterized* traps, i.e. the
    # integral of the tau histogram (each entry is one characterized trap). This
    # replaces the raw detected-dipole count: detection is a poor false-positive
    # filter (random/horizontal-null decoys reach "well-behaved" at 20-50%),
    # whereas characterization rejects ~97-99% of decoys, so the characterized
    # count is the FP-clean population for which the sampled (tau, sigma) are
    # actually validated. Detection/characterization incompleteness (real traps
    # that failed the fit) is bracketed by the upper-limit population variant,
    # not the baseline.
    n_baseline_traps = int(round(float(np.sum(tau_weights))))
    print(f"Baseline trap count: {n_baseline_traps} characterized traps "
          f"(tau-histogram integral of {fname}).")

    num_runs = args.num_runs
    # Each worker holds its own CCD instance (~2.3 GB private memory), so the
    # default uses half the cores rather than cores-1 to stay within RAM.
    num_workers = args.num_workers if args.num_workers is not None else max(1, multiprocessing.cpu_count() // 2)
    
    print(f"Starting parallel execution with {num_workers} CPU cores for {num_runs} runs...")
    print(
        f"Conditions: {args.runconditions}, trap density scale: {args.trap_density_scale:g}, "
        f"packet volume: {args.packet_volume_um3:g} um^3, "
        f"transport: {TRAP_TRANSPORT_MODEL}, "
        f"phase capture: {args.phase_capture_ticks:g} ticks ({args.phase_capture_ticks / 15e6:.3e} s), "
        f"exposure-independent charge: {args.exp_indep_charge_mode}, "
        f"clear: {args.clear_mode}, "
        + (f"readout binning: {args.binning:g}x, " if args.binning != 1.0 else "")
        + (f"0h binning: {args.binning_0h_factor:g}x, " if args.clear_mode == 'binned_0h' else "")
        + f"exposure order: {args.exposure_order}, "
        + f"tpix: {tpix:.3e} s, tpix_vertical: {tpix_vertical:.3e} s"
    )

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
                itertools.repeat(tau_edges),         # tau_edges (constant)
                itertools.repeat(pair_tau135),       # measured tau135 pairs (constant)
                itertools.repeat(pair_sigma),        # measured sigma pairs (constant)
                itertools.repeat(args.runconditions), # run conditions (string constant)
                itertools.repeat(args.out),          # output directory (constant)
                itertools.repeat(args.trap_density_scale), # trap density multiplier (constant)
                itertools.repeat(args.packet_volume_um3),  # single-carrier packet volume (constant)
                itertools.repeat(args.phase_capture_ticks),    # V1/V3 phase-overlap capture window
                itertools.repeat(args.exp_indep_charge_mode), # pre/post active-area readout
                itertools.repeat(args.clear_mode),         # instantaneous/sequencer/three_hour/binned_0h
                itertools.repeat(args.binning_0h_factor),  # 0h row-binning factor (binned_0h mode)
                itertools.repeat(args.exposure_order),     # shuffled/ordered exposure sequence
                itertools.repeat(n_baseline_traps),        # baseline characterized-trap count
                itertools.repeat(fname),                   # tau histogram file (provenance)
                itertools.repeat(args.pairsfile),          # (tau, sigma) pairs file (provenance)
                itertools.repeat(args.binning),            # global readout binning factor (provenance)
            ),
            total=num_runs,
            desc="Running Trials"
        ))

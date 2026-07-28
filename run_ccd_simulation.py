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
from trap_population import load_population
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parallel CCD charge trap simulations.")
    parser.add_argument('--num_runs', type=int, default=200, help="Number of simulation trials to run.")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of CPU workers (defaults to half the cores; each worker holds ~2.3 GB, so cores-1 can exhaust RAM).")
    parser.add_argument('--run-offset', type=int, default=0,
                        help="Starting trial index (default 0). Trials run for r in "
                             "[run_offset, run_offset+num_runs); each writes "
                             "ccd_traps_run{r}.h5 and seeds the PRNG from r, so disjoint "
                             "offsets produce disjoint, non-colliding trials. Used to split "
                             "one scenario's trials across many short HTCondor jobs.")
    parser.add_argument('--runconditions', type=str, default='minos', choices=['minos', 'snolab'], help="Run conditions configuration to use.")
    parser.add_argument('--binning', type=float, default=1.0, help="Scale factor to divide pixel readout times (simulates binning).")
    parser.add_argument('--out', type=str, default='./', help="Output directory.")
    parser.add_argument(
        '--population-model', choices=['esigma', 'legacy_tau'], default='esigma',
        help="Trap population generator. 'esigma' is the physical default; legacy_tau is retained for A/B validation.",
    )
    parser.add_argument('--population-file', type=str, default='trap_population_esigma.npz')
    parser.add_argument('--temperature-K', type=float, default=135.0)
    parser.add_argument('--base-seed', type=int, default=20260728)
    parser.add_argument('--tauhistfile', type=str, default='tau_at_135k_hist.npz', help="the histogram file to used to sample tau values.")
    parser.add_argument('--pairsfile', type=str, default='trap_tau135_sigma_pairs.npz',
                        help="Per-trap (tau135, sigma) pairs file (from make_trap_pairs.py) used to assign capture cross-sections.")
    parser.add_argument('--packet-volume-um3', type=float, default=3.0,
                        help="Effective volume (um^3) explored by a single carrier in a pixel well; sets the capture rate kc = sigma*v_th/V. "
                             "Default 3 um^3: collecting-phase area (~12x5-10 um^2) x thermal vertical spread in the buried channel (~0.02-0.07 um). "
                             "Vary by ~a decade each way as a systematic.")
    parser.add_argument(
        '--well-shift-overlap-factor',
        type=float,
        default=2.0,
        help="Multiplier on the per-phase vertical dwell that sets the effective "
             "V1/V3 capture window per row-shift. By charge conservation + cycle "
             "symmetry each gate accumulates 2 of the 6 phase-steps of "
             "charge-presence, so the justified value is 2.0 (1.0 = sole-well "
             "lower bound). Readout window = factor x 600 ticks (exposeseq), "
             "clear window = factor x 300 ticks (clearseq). See notebook/physics.qmd 2.4.",
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
    parser.add_argument(
        '--v3-phase-fraction',
        type=float,
        default=0.5,
        help="Fraction of traps assigned to the V3 clock phase (Bernoulli per "
             "trap). A V3 trap's readout/clear emission faces a same-step "
             "recapture roll (packet crosses the trap on row exit); a V1 "
             "trap's emission always escapes (packet crossed on entry). "
             "Default 0.5; 1.0 reproduces the pre-2026-07 all-V3 kernel for "
             "A/B comparison.",
    )
    parser.add_argument(
        '--zero-exp-dep-rate',
        action='store_true',
        help="Zero the injected single-electron dark current (exp_dep_rate=0) so "
             "trap emission is the only exposure-dependent single-e source. "
             "High-energy cosmic events and exposure-independent spurious charge "
             "are unaffected (trap-only hypothesis test).",
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

    if args.population_model == 'esigma':
        try:
            population = load_population(args.population_file)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading population {args.population_file}: {exc}")
            sys.exit(1)
        population_energy_edges = population['energy_edges_eV']
        population_log10_sigma_edges = population['log10_sigma_edges']
        population_counts = population['counts_2d']
        population_sha256 = population['sha256']
        n_baseline_traps = int(round(population['population_count_full_ccd']))
        tau_weights = np.array([1.0])
        tau_edges = np.array([1.0, 2.0])
        pair_tau135 = np.array([1.0])
        pair_sigma = np.array([1e-15])
        fname = ''
        pair_fname = ''
        active_population_file = args.population_file
        print(
            f"Loaded {args.population_file}: {population_counts.sum():g} "
            f"full-CCD traps, SHA256 {population_sha256[:12]}..."
        )
    else:
        fname = args.tauhistfile
        try:
            tau_data = np.load(fname)
            tau_weights = tau_data['hist']
            tau_edges = tau_data['bin_edges']
        except FileNotFoundError:
            print(f"Error: {fname} not found.")
            sys.exit(1)
        try:
            pair_data = np.load(args.pairsfile)
            pair_tau135 = pair_data['tau135']
            pair_sigma = pair_data['sigma']
        except FileNotFoundError:
            print(f"Error: {args.pairsfile} not found.")
            sys.exit(1)
        population_energy_edges = np.array([])
        population_log10_sigma_edges = np.array([])
        population_counts = np.empty((0, 0))
        population_sha256 = ''
        pair_fname = args.pairsfile
        active_population_file = ''
        n_baseline_traps = int(round(float(np.sum(tau_weights))))
        print(f"Loaded legacy tau/sigma population with {n_baseline_traps} traps.")
    num_runs = args.num_runs
    # Each worker holds its own CCD instance (~2.3 GB private memory), so the
    # default uses half the cores rather than cores-1 to stay within RAM.
    num_workers = args.num_workers if args.num_workers is not None else max(1, multiprocessing.cpu_count() // 2)
    
    print(f"Starting parallel execution with {num_workers} CPU cores for {num_runs} runs...")
    print(
        f"Conditions: {args.runconditions}, trap density scale: {args.trap_density_scale:g}, "
        f"packet volume: {args.packet_volume_um3:g} um^3, "
        f"transport: {TRAP_TRANSPORT_MODEL}, "
        f"well-shift overlap factor: {args.well_shift_overlap_factor:g} "
        f"(readout {args.well_shift_overlap_factor * 600:g} ticks / "
        f"{args.well_shift_overlap_factor * 600 / 15e6:.3e} s, "
        f"clear {args.well_shift_overlap_factor * 300:g} ticks), "
        f"V3 phase fraction: {args.v3_phase_fraction:g}, "
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
                range(args.run_offset, args.run_offset + num_runs),  # r (run_offset .. run_offset+num_runs-1)
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
                itertools.repeat(args.well_shift_overlap_factor),  # V1/V3 well-shift capture-window multiplier
                itertools.repeat(args.exp_indep_charge_mode), # pre/post active-area readout
                itertools.repeat(args.clear_mode),         # instantaneous/sequencer/three_hour/binned_0h
                itertools.repeat(args.binning_0h_factor),  # 0h row-binning factor (binned_0h mode)
                itertools.repeat(args.exposure_order),     # shuffled/ordered exposure sequence
                itertools.repeat(n_baseline_traps),        # baseline characterized-trap count
                itertools.repeat(fname),                   # tau histogram file (provenance)
                itertools.repeat(pair_fname),              # legacy pairs file (provenance)
                itertools.repeat(args.binning),            # global readout binning factor (provenance)
                itertools.repeat(args.zero_exp_dep_rate),  # zero single-e dark current (trap-only test)
                itertools.repeat(args.v3_phase_fraction),  # V1/V3 clock-phase split (1.0 = old all-V3 kernel)
                itertools.repeat(args.population_model),
                itertools.repeat(population_energy_edges),
                itertools.repeat(population_log10_sigma_edges),
                itertools.repeat(population_counts),
                itertools.repeat(active_population_file),
                itertools.repeat(population_sha256),
                itertools.repeat(args.temperature_K),
                itertools.repeat(args.base_seed),
            ),
            total=num_runs,
            desc="Running Trials"
        ))

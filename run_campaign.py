"""Run the full CCD trap-simulation campaign: MINOS + SNOLAB conditions,
baseline + 90% CL upper-limit trap populations, and the packet-volume
systematic band V_p = {1, 3, 10} um^3.

Scenarios run sequentially (each one already uses all cores internally) in
priority order, so the campaign can be interrupted at any point and the most
important results exist first. Completed scenarios (output dir already holds
num_runs HDF5 files) are skipped, so the script is resumable.

For the upper-limit population the trap density is scaled by
hist.sum() / 5171 (the efficiency-corrected population estimate divided by
the detected-trap count), since the upper-limit histogram encodes the total
population per CCD, not just the tau shape.

Usage:
    python run_campaign.py                 # full campaign, 500 trials each
    python run_campaign.py --num_runs 200  # faster pass
    python run_campaign.py --only minos_baseline   # filter by label substring
    python run_campaign.py --list          # show the scenario table and exit
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

BASELINE_HIST = 'tau_at_135k_hist.npz'
UPPER_HIST = 'tau_at_135k_hist_upper_limit.npz'
N_DETECTED_TRAPS = 5171.0
VP_BASELINE = 3.0
VP_BAND = (1.0, 3.0, 10.0)


def build_scenarios():
    """Priority-ordered scenario list: the four headline science scenarios at
    the central V_p first, then the V_p systematic band (which the single-trial
    scan showed to be a ~25% effect on the masked excess)."""
    scenarios = []
    # 1. All four science scenarios at the central V_p
    for hist in (BASELINE_HIST, UPPER_HIST):
        for cond in ('minos', 'snolab'):
            scenarios.append((cond, hist, VP_BASELINE))
    # 2. V_p systematic band, baseline population then upper limit
    for hist in (BASELINE_HIST, UPPER_HIST):
        for vp in VP_BAND:
            if vp == VP_BASELINE:
                continue
            for cond in ('minos', 'snolab'):
                scenarios.append((cond, hist, vp))
    return scenarios


def label_for(cond, histfile, vp):
    pop = 'upper' if 'upper' in histfile else 'baseline'
    return f"{cond}_{pop}_vp{vp:g}".replace('.', 'p')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_runs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--out_base', type=str, default='campaign')
    parser.add_argument('--only', type=str, default=None,
                        help="Only run scenarios whose label contains this substring.")
    parser.add_argument('--upper-density-scale', type=float, default=None,
                        help="Override the trap-density scale for upper-limit runs "
                             "(default: hist.sum()/5171 from the upper-limit file).")
    parser.add_argument('--list', action='store_true', help="Print scenarios and exit.")
    args = parser.parse_args()

    if args.upper_density_scale is None:
        upper_scale = float(np.load(UPPER_HIST)['hist'].sum()) / N_DETECTED_TRAPS
    else:
        upper_scale = args.upper_density_scale

    scenarios = build_scenarios()
    todo = []
    for cond, histfile, vp in scenarios:
        label = label_for(cond, histfile, vp)
        if args.only and args.only not in label:
            continue
        outdir = os.path.join(args.out_base, label) + os.sep
        n_done = len(glob.glob(os.path.join(outdir, 'ccd_traps_run*.h5')))
        scale = upper_scale if 'upper' in histfile else 1.0
        todo.append((label, cond, histfile, vp, scale, outdir, n_done))

    print(f"{'label':<28} {'hist':<36} {'V_p':>5} {'density x':>9} {'done':>6}")
    for label, cond, histfile, vp, scale, outdir, n_done in todo:
        status = 'SKIP (complete)' if n_done >= args.num_runs else f'{n_done}/{args.num_runs}'
        print(f"{label:<28} {histfile:<36} {vp:>5g} {scale:>9.2f} {status:>6}")
    if args.list:
        return

    for label, cond, histfile, vp, scale, outdir, n_done in todo:
        if n_done >= args.num_runs:
            print(f"\n=== {label}: already complete, skipping ===")
            continue
        cmd = [
            sys.executable, 'run_ccd_simulation.py',
            '--num_runs', str(args.num_runs),
            '--runconditions', cond,
            '--tauhistfile', histfile,
            '--packet-volume-um3', str(vp),
            '--trap-density-scale', str(scale),
            '--out', outdir,
        ]
        if args.num_workers is not None:
            cmd += ['--num_workers', str(args.num_workers)]
        print(f"\n=== {label} ===\n{' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!!! {label} failed (exit {result.returncode}); stopping campaign.")
            sys.exit(result.returncode)

    print("\nCampaign complete.")


if __name__ == '__main__':
    main()

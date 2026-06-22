"""Run the full CCD trap-simulation campaign: MINOS + SNOLAB conditions,
baseline + 90% CL upper-limit trap populations. By default only the central
packet volume V_p = 3 um^3 is run; pass --vp-scan to also sweep the systematic
band V_p = {1, 3, 10} um^3.

Scenarios run sequentially (each one already uses all cores internally) in
priority order, so the campaign can be interrupted at any point and the most
important results exist first. Completed scenarios (output dir already holds
num_runs HDF5 files) are skipped, so the script is resumable.

The baseline trap population is the characterized-trap count (the tau-histogram
integral). For the upper-limit population the trap density is scaled by
hist_upper.sum() / hist_baseline.sum() (the efficiency-corrected population
estimate divided by the baseline characterized count), since the upper-limit
histogram encodes the total population per CCD, not just the tau shape.

Usage:
    python run_campaign.py                 # central V_p only, 200 trials each
    python run_campaign.py --vp-scan       # also sweep the V_p systematic band
    python run_campaign.py --only minos_baseline   # filter by label substring
    python run_campaign.py --list          # show the scenario table and exit
"""
import argparse
import glob
import os
import subprocess
import sys

import h5py
import numpy as np

FLAVOR_HIST_SUFFIX = {
    'legacy': '',
    'minimal_caldet': '_minimal_caldet',
}
# Coord-list suffix per flavor. Only `_minimal` changes the raw dipole finding;
# the `_caldet` detection tag does not, so it is dropped here (see
# run_charge_traps.py:242, coords_file = f'dipole_coord_list{suffix}.npz').
FLAVOR_COORD_SUFFIX = {
    'legacy': '',
    'minimal_caldet': '_minimal',
}
VP_BASELINE = 3.0
# V_p values swept, central (baseline) first then the systematic band.
VP_ORDER = (VP_BASELINE, 1.0, 10.0)
# Readout binning factor for unbinned data (the default everywhere). Binned
# variants are opt-in via --binning-factors and only emitted at the central V_p
# sequencer clear (see binnings_for).
BINNING_BASELINE = 1.0
# Both clear strategies are run for every scenario: the standard ~3.26 s
# sequencer clear and the 3-hour continuous clear ('three_hour').
CLEAR_MODES = ('sequencer', 'three_hour', 'binned_0h')
CLEAR_LABELS = {
    'instantaneous': 'clearinstant',
    'sequencer': 'clearseq',
    'three_hour': 'clear3h',
    'binned_0h': 'bin0h',
}
EXPECTED_TRAP_TRANSPORT_MODEL = 'phase_limited_v1v3'


def build_scenarios(upper=True, vp_values=VP_ORDER):
    """Priority-ordered scenario list with V_p (packet volume) as the outermost
    axis: the central V_p (all four headline science scenarios) first, then the
    systematic band if requested. Within each V_p, baseline population before
    upper limit and MINOS before SNOLAB. The clear mode is swept innermost (in
    the caller), so every clear runs for a given V_p before the next V_p starts.

    `vp_values` defaults to the full systematic band; pass `(VP_BASELINE,)` to
    run only the central packet volume (the headline configuration)."""
    scenarios = []
    # Central V_p first, then the systematic band low->high.
    hists = ('baseline', 'upper') if upper else ('baseline',)
    for vp in vp_values:
        for hist in hists:
            for cond in ('minos', 'snolab'):
                scenarios.append((cond, hist, vp))
    return scenarios


EXPOSURE_ORDER_LABELS = {'shuffled': 'shuf', 'ordered': 'ord'}


def label_for(
    cond,
    histfile,
    vp,
    exp_indep_charge_mode='pre_readout',
    clear_mode='sequencer',
    exposure_order='shuffled',
    flavor='legacy',
    binning=BINNING_BASELINE,
):
    pop = 'upper' if 'upper' in histfile else 'baseline'
    mode = 'expind_post' if exp_indep_charge_mode == 'post_readout' else 'expind_pre'
    clear = CLEAR_LABELS[clear_mode]
    order = EXPOSURE_ORDER_LABELS[exposure_order]
    flavor_tag = '' if flavor == 'legacy' else f'_{flavor}'
    # Unbinned runs keep their original label (no suffix) so previously completed
    # output dirs are still recognized as complete.
    bin_tag = '' if binning == BINNING_BASELINE else f'_bin{binning:g}'
    return f"{cond}_{pop}_vp{vp:g}_{mode}_{clear}_{order}{bin_tag}{flavor_tag}".replace('.', 'p')


def exposure_orders_for(vp, policy):
    """Which exposure orders to run for a given V_p under the campaign policy.
    'headline' adds the 'ordered' (old fixed-cycle) variant only at the central
    V_p headline scenarios; 'all' runs both everywhere; 'none' is shuffled-only."""
    if policy == 'all':
        return ['shuffled', 'ordered']
    if policy == 'headline' and vp == VP_BASELINE:
        return ['shuffled', 'ordered']
    return ['shuffled']


def binnings_for(clear_mode, vp, exposure_order, factors):
    """Readout-binning factors to run for a given clear mode / V_p / exposure
    order. Always runs unbinned (factor 1.0); binned variants from
    --binning-factors are added only for the standard sequencer clear at the
    central V_p in the science-default 'shuffled' order -- the configuration the
    real binned (32x1 superpixel) data was taken in, and the one where the trap
    effect has the statistics to resolve a binning shift. (The legacy 'ordered'
    cycle is a diagnostic, so it is left unbinned to avoid doubling the expensive
    binned runs.) Binning only divides the per-row readout dwell
    (tpix/tpix_vertical); under the phase-limited V1/V3 transport model this
    shortens the emission window while leaving the fixed V1/V3 capture window
    (phase_capture_ticks) unchanged."""
    out = [BINNING_BASELINE]
    if clear_mode == 'sequencer' and vp == VP_BASELINE and exposure_order == 'shuffled':
        out += [f for f in factors if f != BINNING_BASELINE]
    return out


def _decode_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def count_compatible_runs(outdir, phase_capture_ticks):
    """Count existing HDF5 outputs compatible with the active transport model."""
    compatible = 0
    incompatible = []
    for filename in glob.glob(os.path.join(outdir, 'ccd_traps_run*.h5')):
        try:
            with h5py.File(filename, 'r') as f:
                model = _decode_h5_attr(f.attrs.get('trap_transport_model', ''))
                ticks = f.attrs.get('phase_capture_ticks', np.nan)
        except OSError as exc:
            incompatible.append((filename, f'unreadable HDF5: {exc}'))
            continue
        try:
            ticks = float(ticks)
        except (TypeError, ValueError):
            ticks = np.nan
        if (
            model == EXPECTED_TRAP_TRANSPORT_MODEL
            and np.isclose(ticks, phase_capture_ticks, rtol=0.0, atol=1.0e-12)
        ):
            compatible += 1
        else:
            incompatible.append((filename, f'model={model!r}, phase_capture_ticks={ticks!r}'))
    return compatible, incompatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--flavor',
        choices=['legacy', 'minimal_caldet'],
        default='legacy',
        help="Which trap-population histograms to use: 'legacy' (default) uses "
             "tau_at_135k_hist[_upper_limit].npz; 'minimal_caldet' appends "
             "_minimal_caldet to both names and tags output directories accordingly.",
    )
    parser.add_argument('--num_runs', type=int, default=200,
                        help="Trials per scenario. 200 gives sub-1%% SEM on the "
                             "masked excess (run-to-run CV ~8%%); raise for tighter "
                             "headline numbers.")
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--run-offset', type=int, default=None,
                        help="Forwarded to run_ccd_simulation as --run-offset, to run a "
                             "trial sub-range [offset, offset+num_runs). When set (even to 0) "
                             "the campaign-level 'already complete' skip is disabled and "
                             "per-file idempotency in run_ccd_simulation handles resume, so "
                             "each HTCondor job can own a disjoint chunk of a scenario's trials.")
    parser.add_argument('--skip_upper', action='store_true',help='If set will skip the upper limit scenario.')
    parser.add_argument('--out_base', type=str, default='campaign')
    parser.add_argument('--only', type=str, default=None,
                        help="Only run scenarios whose label contains this substring.")
    parser.add_argument('--upper-density-scale', type=float, default=None,
                        help="Override the trap-density scale for upper-limit runs "
                             "(default: hist_upper.sum()/hist_baseline.sum(), the "
                             "upper-limit population over the baseline characterized "
                             "count).")
    parser.add_argument(
        '--exp-indep-charge-mode',
        choices=['pre_readout', 'post_readout'],
        default='pre_readout',
        help="Exposure-independent charge model passed to each simulation. "
             "The mode is included in output-directory labels.",
    )
    parser.add_argument(
        '--clear-modes',
        nargs='+',
        choices=['instantaneous', 'sequencer', 'three_hour', 'binned_0h'],
        default=list(CLEAR_MODES),
        help="Clear models to run for every scenario (each adds a separate set of "
             "output dirs, with the clear version in the label). Default runs both "
             "the standard sequencer clear and the 3-hour continuous clear.",
    )
    parser.add_argument(
        '--exposure-order-policy',
        choices=['headline', 'all', 'none'],
        default='headline',
        help="When to also run the 'ordered' (old fixed-cycle) exposure variant "
             "alongside the default 'shuffled' order: 'headline' (default) adds it "
             "only at the central V_p headline scenarios; 'all' runs both for every "
             "scenario; 'none' runs shuffled only.",
    )
    parser.add_argument(
        '--phase-capture-ticks',
        type=float,
        default=300.0,
        help="Effective V1/V3 phase-overlap capture window in 15 MHz sequencer ticks.",
    )
    parser.add_argument(
        '--binning-factors',
        nargs='+',
        type=float,
        default=[BINNING_BASELINE],
        help="Readout-binning factors to simulate (each scales the per-row dwell, "
             "i.e. faster readout). The default [1.0] runs unbinned only, leaving "
             "the campaign unchanged. Pass e.g. '--binning-factors 32' to add a "
             "32x1-binned variant; binned variants are emitted only for the "
             "sequencer clear at the central V_p, and get a '_bin<f>' label suffix. "
             "Under phase_limited_v1v3 binning shortens the emission window but "
             "leaves the fixed V1/V3 capture window unchanged.",
    )
    parser.add_argument(
        '--vp-scan',
        action='store_true',
        help="Also sweep the packet-volume (V_p) systematic band "
             f"{VP_ORDER} instead of running only the central V_p={VP_BASELINE:g} "
             "um^3. Off by default: the band multiplies the runtime ~3x and, "
             "because the capture rate saturates, varying V_p by a decade barely "
             "moves the results.",
    )
    parser.add_argument('--list', action='store_true', help="Print scenarios and exit.")
    args = parser.parse_args()
    upper = not args.skip_upper
    vp_values = VP_ORDER if args.vp_scan else (VP_BASELINE,)
    # Chunked (HTCondor) mode: --run-offset given (even 0). Each job owns a
    # disjoint trial range; the campaign-level completeness skip is disabled and
    # run_ccd_simulation's per-file skip provides idempotent resume.
    chunked = args.run_offset is not None
    run_offset = args.run_offset if chunked else 0

    hist_suffix = FLAVOR_HIST_SUFFIX[args.flavor]
    baseline_hist = f'tau_at_135k_hist{hist_suffix}.npz'

    # Baseline trap population = the characterized-trap count for this flavor,
    # i.e. the integral of the baseline tau histogram (matches what
    # run_ccd_simulation now seeds from). It is the divisor that converts the
    # upper-limit histogram sum into a density scale, so the upper-limit run
    # places exactly hist_upper.sum() traps. (Characterized, not detected:
    # characterization rejects ~97-99% of decoys; detection does not.)
    n_baseline_traps = int(round(float(np.load(baseline_hist)['hist'].sum())))
    print(f"Baseline trap count ({args.flavor}): {n_baseline_traps} characterized "
          f"traps from {baseline_hist}.")

    if upper:
        upper_hist = f'tau_at_135k_hist{hist_suffix}_upper_limit.npz'
        if args.upper_density_scale is None:
            upper_scale = float(np.load(upper_hist)['hist'].sum()) / n_baseline_traps
        else:
            upper_scale = args.upper_density_scale
    # The (tau135, sigma) pairs are refit from the same per-trap selection as the
    # histogram, so they must track the flavor too; otherwise the sim mixes (e.g.)
    # the minimal tau histogram with the legacy cross-section pairs.
    pairs_file = f'trap_tau135_sigma_pairs{hist_suffix}.npz'

    

    scenarios = build_scenarios(upper=upper, vp_values=vp_values)
    todo = []
    for cond, histfile, vp in scenarios:
        histfile = upper_hist if 'upper' in histfile else baseline_hist
        for clear_mode in args.clear_modes:
            for exposure_order in exposure_orders_for(vp, args.exposure_order_policy):
                for binning in binnings_for(clear_mode, vp, exposure_order, args.binning_factors):
                    label = label_for(
                        cond,
                        histfile,
                        vp,
                        args.exp_indep_charge_mode,
                        clear_mode,
                        exposure_order,
                        args.flavor,
                        binning,
                    )
                    if args.only and args.only not in label:
                        continue
                    outdir = os.path.join(args.out_base, label) + os.sep
                    n_done, incompatible = count_compatible_runs(
                        outdir,
                        args.phase_capture_ticks,
                    )
                    scale = upper_scale if 'upper' in histfile else 1.0
                    todo.append(
                        (label, cond, histfile, vp, scale, outdir, n_done,
                         incompatible, clear_mode, exposure_order, binning)
                    )

    print(f"{'label':<48} {'hist':<36} {'V_p':>5} {'clear':>10} {'order':>9} {'bin':>5} {'density x':>9} {'done':>6} {'bad':>5}")
    for label, cond, histfile, vp, scale, outdir, n_done, incompatible, clear_mode, exposure_order, binning in todo:
        status = 'SKIP (complete)' if n_done >= args.num_runs else f'{n_done}/{args.num_runs}'
        print(f"{label:<48} {histfile:<36} {vp:>5g} {clear_mode:>10} {exposure_order:>9} {binning:>5g} {scale:>9.2f} {status:>6} {len(incompatible):>5}")
    if args.list:
        return

    for label, cond, histfile, vp, scale, outdir, n_done, incompatible, clear_mode, exposure_order, binning in todo:
        if incompatible:
            print(
                f"\n!!! {label}: {len(incompatible)} existing HDF5 files have "
                f"incompatible trap transport metadata for model "
                f"{EXPECTED_TRAP_TRANSPORT_MODEL!r} and phase_capture_ticks="
                f"{args.phase_capture_ticks:g}. Use a new --out_base or "
                "delete/regenerate the incompatible files."
            )
            for filename, reason in incompatible[:5]:
                print(f"    {filename}: {reason}")
            if len(incompatible) > 5:
                print(f"    ... and {len(incompatible) - 5} more")
            sys.exit(1)
        if not chunked and n_done >= args.num_runs:
            print(f"\n=== {label}: already complete, skipping ===")
            continue
        cmd = [
            sys.executable, 'run_ccd_simulation.py',
            '--num_runs', str(args.num_runs),
            '--run-offset', str(run_offset),
            '--runconditions', cond,
            '--tauhistfile', histfile,
            '--pairsfile', pairs_file,
            '--packet-volume-um3', str(vp),
            '--binning', str(binning),
            '--phase-capture-ticks', str(args.phase_capture_ticks),
            '--trap-density-scale', str(scale),
            '--exp-indep-charge-mode', args.exp_indep_charge_mode,
            '--clear-mode', clear_mode,
            '--exposure-order', exposure_order,
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

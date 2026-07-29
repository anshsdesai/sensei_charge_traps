"""Run a user-defined physical E/sigma trap-population campaign.

Defaults: 200 trials, central V_p=1 um^3, three-hour clear, and the headline
exposure-order pair.  Use --zero-exp-dep for the trap-only counterpart.
"""
import argparse
import os
import subprocess
import sys


def label_for(condition, exp_indep_charge_mode, clear_mode, exposure_order, zero_exp_dep):
    mode = 'expind_post' if exp_indep_charge_mode == 'post_readout' else 'expind_pre'
    clear = {'instantaneous': 'clearinstant', 'sequencer': 'clearseq',
             'three_hour': 'clear3h', 'binned_0h': 'bin0h'}[clear_mode]
    order = {'shuffled': 'shuf', 'ordered': 'ord'}[exposure_order]
    zedr = '_zedr' if zero_exp_dep else ''
    return f'{condition}_custom_vp1_{mode}_{clear}_{order}{zedr}_esigma'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population-file', default='trap_population_custom.npz')
    parser.add_argument('--out-base', default='campaign')
    parser.add_argument('--num-runs', type=int, default=200)
    parser.add_argument('--run-offset', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--base-seed', type=int, default=20260728)
    parser.add_argument('--temperature-K', type=float, default=135.0)
    parser.add_argument('--zero-exp-dep', action='store_true',
                        help='Zero injected exposure-dependent single-e dark current (trap-only).')
    parser.add_argument('--exp-indep-charge-mode', choices=['pre_readout', 'post_readout'],
                        default='pre_readout')
    parser.add_argument('--clear-modes', nargs='+',
                        choices=['instantaneous', 'sequencer', 'three_hour', 'binned_0h'],
                        default=['three_hour'])
    parser.add_argument('--exposure-order-policy', choices=['headline', 'all', 'none'],
                        default='headline')
    parser.add_argument('--only', default=None)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.population_file):
        parser.error(f'Custom population file not found: {args.population_file}')

    if args.exposure_order_policy == 'none':
        orders = ['shuffled']
    else:  # central V_p only: headline and all both include the ordered comparison
        orders = ['shuffled', 'ordered']

    scenarios = []
    for condition in ('minos', 'snolab'):
        for clear_mode in args.clear_modes:
            for order in orders:
                label = label_for(condition, args.exp_indep_charge_mode, clear_mode,
                                  order, args.zero_exp_dep)
                if args.only is None or args.only in label:
                    scenarios.append((label, condition, clear_mode, order))

    for label, *_ in scenarios:
        print(label)
    if args.list:
        return

    run_offset = 0 if args.run_offset is None else args.run_offset
    for label, condition, clear_mode, order in scenarios:
        outdir = os.path.join(args.out_base, label)
        cmd = [
            sys.executable, 'run_ccd_simulation.py',
            '--num_runs', str(args.num_runs),
            '--run-offset', str(run_offset),
            '--runconditions', condition,
            '--population-model', 'esigma',
            '--population-file', args.population_file,
            '--temperature-K', str(args.temperature_K),
            '--base-seed', str(args.base_seed),
            '--packet-volume-um3', '1.0',
            '--clear-mode', clear_mode,
            '--exposure-order', order,
            '--exp-indep-charge-mode', args.exp_indep_charge_mode,
            '--out', outdir,
        ]
        if args.zero_exp_dep:
            cmd.append('--zero-exp-dep-rate')
        if args.num_workers is not None:
            cmd += ['--num_workers', str(args.num_workers)]
        print(f'\n=== {label} ===\n' + ' '.join(cmd))
        result = subprocess.run(cmd)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
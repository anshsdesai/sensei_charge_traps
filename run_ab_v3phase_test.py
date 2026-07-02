#!/usr/bin/env python
"""A/B test for the V1/V3 phase-split kernel fix (code review 2026-07-01, F1/F8).

Runs the SAME scenario twice, differing only in --v3-phase-fraction:

  * arm A (control): 1.0  -- the pre-2026-07 all-V3 kernel (every trap's
    readout/clear emission faces a same-step recapture roll)
  * arm B (fixed):   0.5  -- Bernoulli-half V1/V3 phase split (V1 emissions
    always escape; V3 drain recapture-thinned)

Scenario (chosen to maximize sensitivity to the escaped-emission term F1
modifies): trap-only (--zero-exp-dep-rate, so trap emission is the ONLY
exposure-dependent single-e source), upper-limit population (weights the
large-tau / large-alpha traps that drive the UL slope), minos, central V_p,
shuffled exposures. Both arms share the same seed files (regenerated
2026-07-02 with the x0.972 energy-fit survival).

What to compare between arms: the fitted exposure-dependent slope and
intercept of the masked single-e excess, and the per-exposure raw excess.
With run-to-run CV ~8%, N=50 trials/arm resolves the A-B slope delta to
~1.6% of the mean excess. Decision rule: delta within the statistical band
(and small vs the V_p systematic band) -> existing campaign numbers stand
with a systematic note; significant delta -> full campaign rerun at 0.5.

Outputs go under ab_test_v3phase/ -- TEST DATA, not campaign results.
Resumable: run_ccd_simulation skips per-trial HDF5 files that already exist
with matching parameters.

Usage:
  python run_ab_v3phase_test.py                     # 50 trials/arm, cores//2
  python run_ab_v3phase_test.py --num-runs 30 --num-workers 6
  python run_ab_v3phase_test.py --clear-mode three_hour   # F8-sensitive variant
"""
import argparse
import datetime
import os
import subprocess
import sys

OUT_BASE = 'ab_test_v3phase'
ARMS = [
    ('v3frac1p0_control', 1.0),
    ('v3frac0p5_split', 0.5),
]

README = """# ab_test_v3phase -- TEST DATA, NOT CAMPAIGN RESULTS

Created {date} by run_ab_v3phase_test.py (code review 2026-07-01, finding F1/F8).

Purpose: measure whether the V1/V3 phase-split kernel fix moves the masked
single-e observables, before deciding on a full campaign rerun. The two arms
are identical except for --v3-phase-fraction:

  {clear_mode}_v3frac1p0_control/ : 1.0 = old all-V3 kernel (control)
  {clear_mode}_v3frac0p5_split/   : 0.5 = phase-split fix

Scenario: trap-only (--zero-exp-dep-rate), upper-limit population
(tau_at_135k_hist_minimal_caldet_upper_limit.npz, integral ~17,978 incl. the
x0.972 energy-fit survival), minos, central V_p=3 um^3, clear={clear_mode},
shuffled exposures, {num_runs} trials/arm.

Compare between arms: fitted exposure-dependent slope + intercept of the
masked excess; per-exposure raw excess. See notebook/code_review_2026-07-01.md
(F1) for the physics and the decision rule.

Safe to delete once the F1 A/B decision is recorded.
"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-runs', type=int, default=50,
                        help="Trials per arm (default 50 -> ~1.6%% resolution "
                             "on the A-B slope delta).")
    parser.add_argument('--num-workers', type=int, default=None,
                        help="Worker processes, forwarded to run_ccd_simulation "
                             "(default: cpu_count//2; ~2.3 GB RAM per worker).")
    parser.add_argument('--clear-mode', default='sequencer',
                        choices=['sequencer', 'three_hour', 'binned_0h'],
                        help="Clear mode for both arms (default sequencer = the "
                             "headline; three_hour additionally exercises the F8 "
                             "drain fix).")
    parser.add_argument('--out-base', default=OUT_BASE,
                        help=f"Base output directory (default {OUT_BASE}/).")
    args = parser.parse_args()

    os.makedirs(args.out_base, exist_ok=True)
    readme_path = os.path.join(args.out_base, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(README.format(date=datetime.date.today().isoformat(),
                              clear_mode=args.clear_mode,
                              num_runs=args.num_runs))
    print(f"Wrote {readme_path}")

    for label, frac in ARMS:
        outdir = os.path.join(args.out_base, f'{args.clear_mode}_{label}') + os.sep
        cmd = [
            sys.executable, 'run_ccd_simulation.py',
            '--num_runs', str(args.num_runs),
            '--runconditions', 'minos',
            '--tauhistfile', 'tau_at_135k_hist_minimal_caldet_upper_limit.npz',
            '--pairsfile', 'trap_tau135_sigma_pairs_minimal_caldet.npz',
            '--clear-mode', args.clear_mode,
            '--exposure-order', 'shuffled',
            '--zero-exp-dep-rate',
            '--v3-phase-fraction', str(frac),
            '--out', outdir,
        ]
        if args.num_workers is not None:
            cmd += ['--num_workers', str(args.num_workers)]
        print(f"\n=== arm {label} (v3_phase_fraction={frac}) ===\n{' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!!! arm {label} failed (exit {result.returncode}); stopping.")
            sys.exit(result.returncode)

    print("\nA/B run complete. Both arms under", args.out_base + os.sep)


if __name__ == '__main__':
    main()

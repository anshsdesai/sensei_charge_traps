#!/usr/bin/env python
"""Analyze the V1/V3 phase-split A/B test produced by run_ab_v3phase_test.py.

For each arm (v3_phase_fraction = 1.0 control vs 0.5 split) this script:
  1. reads every ccd_traps_run*.h5, checks the arms differ ONLY in
     v3_phase_fraction (same clear mode, seeds, flags),
  2. per trial and per exposure computes the masked single-e excess density
       Delta(h) = sum(counts_trap)/sum(unmasked_trap)
                - sum(counts_notrap)/sum(unmasked_notrap)
     under the headline mask (Halo+Bleed+HotColumn+HotPixel), plus the raw
     unmasked ('None') COUNT excess as the mechanism view (the masking
     denominator couples to the trap branch, so counts and density can
     disagree -- see physics.qmd deviation term F),
  3. fits Delta vs exposure hours per trial (OLS, 5 points) -> per-trial
     slope + intercept; arm summary = mean +/- SEM over trials,
  4. reports the A-B deltas with two-sample errors and z-scores -- the
     slope delta is THE decision number for finding F1 (UL slope bias),
  5. writes a JSON summary + a PNG (excess vs exposure, both arms) next to
     the arm directories.

Decision rule (see notebook/code_review_2026-07-01.md, F1): slope delta
consistent with zero at the few-percent level -> existing campaign numbers
stand with a systematic note; significant delta -> campaign rerun at 0.5.

Usage:
  python analyze_ab_v3phase_test.py                          # sequencer arms
  python analyze_ab_v3phase_test.py --clear-mode three_hour
  python analyze_ab_v3phase_test.py --dirs pathA/ pathB/     # explicit dirs
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import h5py

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


HEADLINE_MASK = 'Halo+Bleed+HotColumn+HotPixel'
RAW_MASK = 'None'
# Attrs that must be identical across the two arms for a valid A/B.
MUST_MATCH_ATTRS = [
    'clear_mode', 'exposure_order', 'exp_indep_charge_mode', 'runconditions',
    'zero_exp_dep_rate', 'packet_volume_um3', 'phase_capture_ticks',
    'binning', 'trap_transport_model', 'tauhistfile', 'pairsfile',
    'n_detected_traps',
]


def _attr(f, key, default=None):
    v = f.attrs.get(key, default)
    if isinstance(v, bytes):
        v = v.decode()
    return v


def load_arm(rundir):
    """Read one arm directory -> per-trial per-exposure sums + config."""
    files = sorted(glob.glob(os.path.join(rundir, 'ccd_traps_run*.h5')),
                   key=lambda p: int(re.search(r'run(\d+)\.h5$', p).group(1)))
    if not files:
        sys.exit(f"No ccd_traps_run*.h5 in {rundir}")

    trials = []
    config = None
    for path in files:
        with h5py.File(path, 'r') as f:
            if config is None:
                config = {k: _attr(f, k) for k in MUST_MATCH_ATTRS}
                config['v3_phase_fraction'] = float(_attr(f, 'v3_phase_fraction', np.nan))
            else:
                frac = float(_attr(f, 'v3_phase_fraction', np.nan))
                if not np.isclose(frac, config['v3_phase_fraction']):
                    sys.exit(f"{path}: v3_phase_fraction {frac} != "
                             f"{config['v3_phase_fraction']} within one arm")

            exposures_h = f['exposures'][:] / 3600.0
            hours = np.unique(exposures_h)
            trial = {'hours': hours, 'n_traps': int(f['trap_taus'].shape[0])}
            if 'trap_is_v3' in f:
                trial['v3_frac_realized'] = float(np.mean(f['trap_is_v3'][:]))
            if 'clear_occupied_traps_before' in f:
                occ = f['clear_occupied_traps_before'][:]
                trial['mean_occupied_before_clear'] = (
                    float(np.mean(occ)) if occ.size else np.nan)

            for mask, tag in ((HEADLINE_MASK, 'masked'), (RAW_MASK, 'raw')):
                ct = f['stats_trap'][mask]['counts'][:]
                cn = f['stats_notrap'][mask]['counts'][:]
                if mask == RAW_MASK:
                    # raw view: full quadrant pixel count, same for both branches
                    ut = un = np.full(ct.shape, 512 * 3072, dtype=float)
                else:
                    ut = f['stats_trap'][mask]['unmasked_pix'][:].astype(float)
                    un = f['stats_notrap'][mask]['unmasked_pix'][:].astype(float)
                dens, cnts = [], []
                for h in hours:
                    sel = np.isclose(exposures_h, h)
                    dens.append(ct[sel].sum() / ut[sel].sum()
                                - cn[sel].sum() / un[sel].sum())
                    cnts.append((ct[sel].sum() - cn[sel].sum()) / sel.sum())
                trial[f'excess_density_{tag}'] = np.asarray(dens)
                trial[f'excess_counts_per_img_{tag}'] = np.asarray(cnts)
            trials.append(trial)

    return trials, config, len(files)


def arm_summary(trials, key):
    """Per-trial OLS fits of excess vs hours -> arm means/SEMs."""
    hours = trials[0]['hours']
    Y = np.vstack([t[key] for t in trials])            # (n_trials, n_hours)
    slopes, intercepts = [], []
    for y in Y:
        m, b = np.polyfit(hours, y, 1)
        slopes.append(m)
        intercepts.append(b)
    slopes, intercepts = np.asarray(slopes), np.asarray(intercepts)

    def msem(a):
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))

    return {
        'hours': hours.tolist(),
        'n_trials': int(Y.shape[0]),
        'per_exposure_mean': Y.mean(axis=0).tolist(),
        'per_exposure_sem': (Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0])).tolist(),
        'slope_mean': msem(slopes)[0], 'slope_sem': msem(slopes)[1],
        'intercept_mean': msem(intercepts)[0], 'intercept_sem': msem(intercepts)[1],
    }


def delta_row(name, a, b, mkey, skey):
    d = b[mkey] - a[mkey]
    e = float(np.hypot(a[skey], b[skey]))
    z = d / e if e > 0 else np.inf
    rel = d / abs(a[mkey]) if a[mkey] != 0 else np.inf
    return {'name': name, 'control': a[mkey], 'split': b[mkey],
            'delta': d, 'err': e, 'z': z, 'rel': rel}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', default='ab_test_v3phase')
    parser.add_argument('--clear-mode', default='sequencer',
                        choices=['sequencer', 'three_hour', 'binned_0h'])
    parser.add_argument('--dirs', nargs=2, metavar=('CONTROL', 'SPLIT'),
                        default=None,
                        help="Explicit arm directories (control=1.0 first); "
                             "overrides --base/--clear-mode.")
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    if args.dirs:
        dir_a, dir_b = args.dirs
    else:
        dir_a = os.path.join(args.base, f'{args.clear_mode}_v3frac1p0_control')
        dir_b = os.path.join(args.base, f'{args.clear_mode}_v3frac0p5_split')

    print(f"control arm: {dir_a}")
    print(f"split   arm: {dir_b}")
    trials_a, cfg_a, n_a = load_arm(dir_a)
    trials_b, cfg_b, n_b = load_arm(dir_b)

    # --- validity checks -------------------------------------------------
    mismatched = [k for k in MUST_MATCH_ATTRS if cfg_a[k] != cfg_b[k]]
    if mismatched:
        sys.exit("!!! arms differ in more than v3_phase_fraction: "
                 + ", ".join(f"{k}: {cfg_a[k]!r} vs {cfg_b[k]!r}" for k in mismatched))
    print(f"config check OK: arms identical except v3_phase_fraction "
          f"({cfg_a['v3_phase_fraction']:g} vs {cfg_b['v3_phase_fraction']:g}); "
          f"{n_a} vs {n_b} trials; clear={cfg_a['clear_mode']}, "
          f"zero_exp_dep={cfg_a['zero_exp_dep_rate']}, "
          f"hist={cfg_a['tauhistfile']}")
    for label, trials in (('control', trials_a), ('split', trials_b)):
        fr = [t['v3_frac_realized'] for t in trials if 'v3_frac_realized' in t]
        nt = [t['n_traps'] for t in trials]
        occ = [t.get('mean_occupied_before_clear', np.nan) for t in trials]
        print(f"  {label}: n_traps {np.mean(nt):.0f}+/-{np.std(nt):.0f}, "
              f"realized V3 fraction {np.mean(fr):.3f}, "
              f"mean occupied before clear {np.nanmean(occ):.1f}")

    # --- headline numbers -------------------------------------------------
    results = {'control_dir': dir_a, 'split_dir': dir_b,
               'clear_mode': cfg_a['clear_mode'], 'summaries': {}, 'deltas': []}
    print()
    for key, label, unit in (
            ('excess_density_masked',
             'MASKED excess density (headline observable)', '/pix/img'),
            ('excess_counts_per_img_raw',
             'RAW count excess (mechanism view)', 'counts/img')):
        a = arm_summary(trials_a, key)
        b = arm_summary(trials_b, key)
        results['summaries'][key] = {'control': a, 'split': b}
        print(f"=== {label} ===")
        print(f"    {'exp':>4}  {'control':>13}  {'split':>13}  "
              f"{'delta':>13}  {'z':>6}")
        for i, h in enumerate(a['hours']):
            d = b['per_exposure_mean'][i] - a['per_exposure_mean'][i]
            e = np.hypot(a['per_exposure_sem'][i], b['per_exposure_sem'][i])
            print(f"    {int(h):>3}h  {a['per_exposure_mean'][i]:>13.4e}  "
                  f"{b['per_exposure_mean'][i]:>13.4e}  {d:>13.4e}  "
                  f"{d / e if e > 0 else float('inf'):>6.1f}")
        for nm, mkey, skey in (('slope [' + unit + '/h]', 'slope_mean', 'slope_sem'),
                               ('intercept [' + unit + ']', 'intercept_mean', 'intercept_sem')):
            row = delta_row(f'{key}:{nm}', a, b, mkey, skey)
            results['deltas'].append(row)
            print(f"  {nm:<28} control {row['control']:+.4e} +/- {a[skey]:.1e}"
                  f" | split {row['split']:+.4e} +/- {b[skey]:.1e}"
                  f" | delta {row['delta']:+.4e} +/- {row['err']:.1e}"
                  f"  (z={row['z']:+.1f}, {100 * row['rel']:+.1f}%)")
        print()

    # --- verdict helper ---------------------------------------------------
    slope = next(r for r in results['deltas']
                 if r['name'].startswith('excess_density_masked:slope'))
    print("F1 decision number -> masked-excess SLOPE delta (split - control): "
          f"{slope['delta']:+.4e} +/- {slope['err']:.1e} /pix/img/h "
          f"(z = {slope['z']:+.1f}, {100 * slope['rel']:+.1f}% of control)")
    if abs(slope['z']) < 2:
        print("  -> consistent with zero at 2 sigma: campaign numbers stand; "
              "record the delta as a systematic.")
    else:
        print("  -> SIGNIFICANT: the all-V3 kernel biased the slope; "
              "campaign rerun at v3_phase_fraction=0.5 is warranted.")

    # --- outputs -----------------------------------------------------------
    outbase = os.path.dirname(os.path.normpath(dir_a)) or '.'
    tag = cfg_a['clear_mode']
    json_path = os.path.join(outbase, f'ab_analysis_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nWrote {json_path}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a = results['summaries']['excess_density_masked']['control']
        b = results['summaries']['excess_density_masked']['split']
        fig, ax = plt.subplots(figsize=(9, 6))
        for s, lbl, c in ((a, 'control (all-V3, frac=1.0)', 'tab:gray'),
                          (b, 'phase split (frac=0.5)', 'tab:red')):
            h = np.asarray(s['hours'])
            y = np.asarray(s['per_exposure_mean'])
            e = np.asarray(s['per_exposure_sem'])
            ax.errorbar(h, y, yerr=e, fmt='o', capsize=3, label=lbl, color=c)
            ax.plot(h, s['slope_mean'] * h + s['intercept_mean'], '--',
                    color=c, alpha=0.7)
        ax.set_xlabel('exposure [h]')
        ax.set_ylabel('masked single-e excess density [/pix/img]')
        ax.set_title(f'F1 A/B: v3_phase_fraction 1.0 vs 0.5 ({tag}, trap-only)')
        ax.legend()
        ax.grid(alpha=0.3)
        png_path = os.path.join(outbase, f'ab_analysis_{tag}.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {png_path}")


if __name__ == '__main__':
    main()

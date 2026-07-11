"""Diagnose the two validation flags: legacy-trap attrition and the high-T Arrhenius systematic."""

from collections import Counter, defaultdict

import numpy as np
import h5py
from scipy.optimize import curve_fit

from dipole import log_energy_cross_section

LEGACY = 'fit_dipole_spectra_err_4.h5'
NEW = 'fit_dipole_spectra_signed_err_4.h5'


def load_catalog(path):
    traps = {}
    with h5py.File(path, 'r') as h5:
        for quad_name, quad_group in h5.items():
            quad = int(quad_name.split('_')[1])
            for dp_name, dp_group in quad_group.items():
                _, row, col = dp_name.split('_')
                key = (quad, int(row), int(col))
                attrs = dp_group.attrs
                good = {}
                fail_reasons = {}
                for temp_name in dp_group.keys():
                    if not temp_name.startswith('temp_'):
                        continue
                    t = int(temp_name.split('_')[1])
                    tg = dp_group[temp_name]
                    if bool(tg.attrs.get('GoodIntensityFit', False)):
                        good[t] = (float(tg.attrs['fit_tau']), float(tg.attrs['fit_tau_err']))
                    else:
                        if bool(tg.attrs.get('IntensityFitFailed', True)):
                            fail_reasons[t] = 'fit_failed'
                        elif not (float(tg.attrs.get('fit_p_value', 0)) > 0.05):
                            fail_reasons[t] = 'p_value'
                        elif float(tg.attrs.get('delta_chi2_vs_constant', np.inf)) < 11.83:
                            fail_reasons[t] = 'delta_chi2'
                        elif float(tg.attrs.get('amplitude_significance', np.inf)) < 3:
                            fail_reasons[t] = 'significance'
                        else:
                            fail_reasons[t] = 'tau_rel_err'
                traps[key] = {
                    'characterized': bool(attrs.get('WellBehavedTrap', False))
                    and not bool(attrs.get('EnergyFitFailed', True))
                    and bool(attrs.get('GoodEnergyFit', False)),
                    'well_behaved': bool(attrs.get('WellBehavedTrap', False)),
                    'n_good': len(good),
                    'good': good,
                    'fail_reasons': fail_reasons,
                }
    return traps


def main():
    legacy = load_catalog(LEGACY)
    new = load_catalog(NEW)

    legacy_char = {k for k, v in legacy.items() if v['characterized']}
    print(f'legacy characterized: {len(legacy_char)}')
    in_new = {k for k in legacy_char if k in new}
    # allow +-1 row coordinate slop
    shifted = {
        k for k in legacy_char - in_new
        if (k[0], k[1] - 1, k[2]) in new or (k[0], k[1] + 1, k[2]) in new
    }
    print(f'  found in new candidate list (exact coord): {len(in_new)}')
    print(f'  found at +-1 row: {len(shifted)}')
    print(f'  missing entirely: {len(legacy_char - in_new - shifted)}')

    fates = Counter()
    fail_mix = Counter()
    for k in in_new:
        v = new[k]
        if v['characterized']:
            fates['characterized'] += 1
        elif v['well_behaved']:
            fates['well_behaved_but_energy_fit_failed_or_bad'] += 1
        else:
            fates[f'n_good={v["n_good"]}'] += 1
            fail_mix.update(v['fail_reasons'].values())
    print('fate of legacy-characterized traps present in new list:')
    for fate, count in fates.most_common():
        print(f'  {fate}: {count}')
    print('per-temperature failure reasons for those with n_good < 4:')
    for reason, count in fail_mix.most_common():
        print(f'  {reason}: {count}')

    # Arrhenius systematic: median log10(tau_meas / tau_pred_from_lowT) by temperature.
    disc = defaultdict(list)
    for k, v in new.items():
        low = {t: x for t, x in v['good'].items() if t <= 160}
        high = {t: x for t, x in v['good'].items() if t > 160}
        if len(low) < 4 or not high:
            continue
        temps = np.array(sorted(low), dtype=float)
        taus = np.array([low[t][0] for t in sorted(low)])
        errs = np.array([low[t][1] for t in sorted(low)]) / taus
        try:
            popt, _ = curve_fit(
                log_energy_cross_section, temps, np.log(taus), sigma=errs,
                bounds=([0, -100], [2, -1]),
            )
        except Exception:
            continue
        for t, (tau, _err) in high.items():
            pred = float(log_energy_cross_section(np.array([float(t)]), *popt)[0])
            disc[t].append((np.log(tau) - pred) / np.log(10))
    print('\nmedian log10(tau_measured / tau_predicted_from_<=160K) by temperature:')
    for t in sorted(disc):
        arr = np.array(disc[t])
        print(f'  {t} K: n={arr.size:5d}  median {np.median(arr):+.2f}  '
              f'68% [{np.percentile(arr, 16):+.2f}, {np.percentile(arr, 84):+.2f}]')


if __name__ == '__main__':
    main()

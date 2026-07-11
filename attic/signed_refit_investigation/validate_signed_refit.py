"""Validate the signed-pipeline refit against the legacy analysis.

Checks:
 1. Per-temperature GoodIntensityFit rates: the GOF survival should now be
    roughly flat in temperature (the pedestal is modeled), unlike the legacy
    collapse above 165 K.
 2. Reduced chi-square of good fits ~ 1 at all temperatures (validates the
    temporal noise model).
 3. Arrhenius self-consistency: for traps with >= 4 good fits at T <= 160 K,
    fit E/sigma from the low-T points only and compare predicted tau(T) with
    the directly fitted tau at recovered high-T points (pull distribution).
 4. E / log(sigma) stability for traps characterized in both catalogs.
 5. Headline counts (dipoles, good fits, well-behaved traps) old vs new.
"""

import sys
from collections import defaultdict

import numpy as np
import h5py
from scipy.optimize import curve_fit

from dipole import log_energy_cross_section

LEGACY_FITS = 'fit_dipole_spectra_err_4.h5'
NEW_FITS = 'fit_dipole_spectra_signed_abssigma_err_4.h5'


def iter_traps(h5):
    for quad_name, quad_group in h5.items():
        quad = int(quad_name.split('_')[1])
        for dp_name, dp_group in quad_group.items():
            _, row, col = dp_name.split('_')
            yield quad, (int(row), int(col)), dp_group


def collect(path):
    per_temp_good = defaultdict(lambda: [0, 0])
    per_temp_chi2 = defaultdict(list)
    traps = {}
    n_dipoles = 0
    with h5py.File(path, 'r') as h5:
        for quad, coord, dp_group in iter_traps(h5):
            n_dipoles += 1
            good_temps = {}
            for temp_name in dp_group.keys():
                if not temp_name.startswith('temp_'):
                    continue
                temp = int(temp_name.split('_')[1])
                tg = dp_group[temp_name]
                good = bool(tg.attrs.get('GoodIntensityFit', False))
                per_temp_good[temp][0] += int(good)
                per_temp_good[temp][1] += 1
                if good:
                    per_temp_chi2[temp].append(float(tg.attrs.get('fit_reduced_chi_squared', np.nan)))
                    good_temps[temp] = (
                        float(tg.attrs.get('fit_tau', np.nan)),
                        float(tg.attrs.get('fit_tau_err', np.nan)),
                    )
            attrs = dp_group.attrs
            traps[(quad, coord)] = {
                'well_behaved': bool(attrs.get('WellBehavedTrap', False)),
                'good_energy_fit': bool(attrs.get('GoodEnergyFit', False))
                and not bool(attrs.get('EnergyFitFailed', True)),
                'E': float(attrs.get('energy_BestFitEnergy', np.nan)),
                'cs': float(attrs.get('energy_BestFitCrossSection', np.nan)),
                'good_temps': good_temps,
            }
    return per_temp_good, per_temp_chi2, traps, n_dipoles


def arrhenius_pulls(traps, low_t_max=160, min_low=4):
    pulls = []
    n_traps_tested = 0
    for key, trap in traps.items():
        low = {t: v for t, v in trap['good_temps'].items() if t <= low_t_max}
        high = {t: v for t, v in trap['good_temps'].items() if t > low_t_max}
        if len(low) < min_low or not high:
            continue
        temps = np.array(sorted(low))
        taus = np.array([low[t][0] for t in temps])
        tau_errs = np.array([low[t][1] for t in temps])
        logtaus = np.log(taus)
        logtau_err = tau_errs / taus
        try:
            popt, pcov = curve_fit(
                log_energy_cross_section, temps.astype(float), logtaus,
                sigma=logtau_err, bounds=([0, -100], [2, -1]),
            )
        except Exception:
            continue
        n_traps_tested += 1
        for t, (tau, tau_err) in high.items():
            pred_log = float(log_energy_cross_section(np.array([float(t)]), *popt)[0])
            meas_log = np.log(tau)
            # crude pull: measurement log-error only (fit extrapolation error ignored,
            # so this is an upper bound on the discrepancy significance)
            pull = (meas_log - pred_log) / max(tau_err / tau, 1e-3)
            pulls.append(pull)
    return np.array(pulls), n_traps_tested


def main():
    new_only = len(sys.argv) > 1 and sys.argv[1] == '--new-only'

    print('=== NEW (signed, offset, physical errors) ===')
    g_new, chi_new, traps_new, nd_new = collect(NEW_FITS)
    print(f'dipoles: {nd_new}')
    print(f'well-behaved: {sum(t["well_behaved"] for t in traps_new.values())}')
    print(f'characterized (well-behaved & good energy fit): '
          f'{sum(t["well_behaved"] and t["good_energy_fit"] for t in traps_new.values())}')
    print('T     good/total  frac    median chi2red(good)')
    for t in sorted(g_new):
        good, tot = g_new[t]
        med = np.nanmedian(chi_new[t]) if chi_new[t] else np.nan
        print(f'{t:4d}  {good:5d}/{tot:5d}  {good/tot:.3f}   {med:.2f}')

    pulls, n_tested = arrhenius_pulls(traps_new)
    if pulls.size:
        print(f'\nArrhenius consistency ({n_tested} traps, {pulls.size} recovered high-T points):')
        print(f'  median pull {np.median(pulls):+.2f}, central 68% '
              f'[{np.percentile(pulls, 16):+.2f}, {np.percentile(pulls, 84):+.2f}], '
              f'|pull|>5: {np.mean(np.abs(pulls) > 5):.3f}')

    if new_only:
        return

    print('\n=== LEGACY ===')
    g_old, chi_old, traps_old, nd_old = collect(LEGACY_FITS)
    print(f'dipoles: {nd_old}')
    print(f'well-behaved: {sum(t["well_behaved"] for t in traps_old.values())}')
    print(f'characterized: {sum(t["well_behaved"] and t["good_energy_fit"] for t in traps_old.values())}')

    both = [
        k for k in traps_new
        if k in traps_old
        and traps_new[k]['well_behaved'] and traps_new[k]['good_energy_fit']
        and traps_old[k]['well_behaved'] and traps_old[k]['good_energy_fit']
    ]
    if both:
        dE = np.array([traps_new[k]['E'] - traps_old[k]['E'] for k in both])
        dls = np.array([np.log(traps_new[k]['cs']) - np.log(traps_old[k]['cs']) for k in both])
        print(f'\nTraps characterized in both: {len(both)}')
        print(f'  delta E: median {np.median(dE):+.4f} eV, central 68% '
              f'[{np.percentile(dE, 16):+.4f}, {np.percentile(dE, 84):+.4f}]')
        print(f'  delta ln(sigma): median {np.median(dls):+.3f}, central 68% '
              f'[{np.percentile(dls, 16):+.3f}, {np.percentile(dls, 84):+.3f}]')


if __name__ == '__main__':
    main()

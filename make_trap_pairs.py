"""Generate the per-trap (tau_e at 135 K, sigma, E) pairs file used by the CCD
simulation's SRH capture/recapture model.

Refits each well-behaved trap's energy/cross-section from the per-temperature
tau_e measurements stored in the fit HDF5, using the current (corrected)
log_energy_cross_section in dipole.py. Because the fit family
log(tau) = const - log(sigma) - 2 log(kT) + E/kT is unchanged, the predicted
tau_e(135 K) values are identical to those in tau_at_135k_hist.npz; only the
quoted E and sigma move when the physical constants are corrected.

Usage:
    python make_trap_pairs.py [--fitfile fit_dipole_spectra_err_4.h5]
                              [--out trap_tau135_sigma_pairs.npz]
"""
import argparse

import h5py
import numpy as np
from scipy.optimize import curve_fit

from dipole import log_energy_cross_section


def make_pairs(fitfile='fit_dipole_spectra_err_4.h5', out='trap_tau135_sigma_pairs.npz'):
    energies, log_sigmas, tau135, tau135_stored = [], [], [], []
    n_skipped = 0

    with h5py.File(fitfile, 'r') as f:
        for quad_name, quad_grp in f.items():
            for dp_name, dp_grp in quad_grp.items():
                a = dp_grp.attrs
                if not (a.get('WellBehavedTrap', False) and a.get('GoodEnergyFit', False)):
                    continue
                try:
                    temps = np.asarray(dp_grp['energy_temperatures'], dtype=float)
                    taus = np.asarray(dp_grp['energy_taus'], dtype=float)
                    tau_errs = np.asarray(dp_grp['energy_tau_errs'], dtype=float)
                except KeyError:
                    n_skipped += 1
                    continue

                logtaus = np.log(taus)
                logtauerr = tau_errs / taus
                try:
                    popt, _ = curve_fit(
                        log_energy_cross_section, temps, logtaus,
                        sigma=logtauerr, bounds=([0, -100], [2, -1]),
                    )
                except Exception:
                    n_skipped += 1
                    continue

                energies.append(popt[0])
                log_sigmas.append(popt[1])
                tau135.append(np.exp(log_energy_cross_section(135.0, popt[0], popt[1])))
                # tau135 implied by the cached (old-constant) fit, for the invariance check
                if 'energy_BestFitEnergy' in a and 'energy_BestFitCrossSection' in a:
                    tau135_stored.append(np.exp(
                        log_energy_cross_section(135.0, float(a['energy_BestFitEnergy']),
                                                 np.log(float(a['energy_BestFitCrossSection'])))
                    ))
                else:
                    tau135_stored.append(np.nan)

    energies = np.array(energies)
    sigmas = np.exp(np.array(log_sigmas))
    tau135 = np.array(tau135)

    print(f'Refit {len(energies)} well-behaved traps ({n_skipped} skipped).')
    print(f'E   [eV] : median {np.median(energies):.3f}, 5-95% {np.percentile(energies, 5):.3f}-{np.percentile(energies, 95):.3f}')
    print(f'sigma[cm2]: median {np.median(sigmas):.3e}, 5-95% {np.percentile(sigmas, 5):.3e}-{np.percentile(sigmas, 95):.3e}')
    print(f'tau(135K)[s]: median {np.median(tau135):.3g}')

    np.savez(out, tau135=tau135, sigma=sigmas, energy=energies)
    print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fitfile', type=str, default='fit_dipole_spectra_err_4.h5')
    parser.add_argument('--out', type=str, default='trap_tau135_sigma_pairs.npz')
    args = parser.parse_args()
    make_pairs(fitfile=args.fitfile, out=args.out)


if __name__ == '__main__':
    main()

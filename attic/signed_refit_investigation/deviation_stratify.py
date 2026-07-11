"""Stratify the high-T Arrhenius deviation by (fitted tau, temperature)."""
import numpy as np
import h5py
from collections import defaultdict
from scipy.optimize import curve_fit
from dipole import log_energy_cross_section

traps = {}
with h5py.File('fit_dipole_spectra_signed_err_4.h5', 'r') as h5:
    for quad_name, quad_group in h5.items():
        for dp_name, dp_group in quad_group.items():
            good = {}
            for temp_name in dp_group.keys():
                if not temp_name.startswith('temp_'):
                    continue
                t = int(temp_name.split('_')[1])
                tg = dp_group[temp_name]
                if bool(tg.attrs.get('GoodIntensityFit', False)):
                    good[t] = (float(tg.attrs['fit_tau']), float(tg.attrs['fit_tau_err']))
            if len(good) >= 5:
                traps[(quad_name, dp_name)] = good

cells = defaultdict(list)
for key, good in traps.items():
    low = {t: v for t, v in good.items() if t <= 160}
    high = {t: v for t, v in good.items() if t > 160}
    if len(low) < 4 or not high:
        continue
    temps = np.array(sorted(low), dtype=float)
    taus = np.array([low[t][0] for t in sorted(low)])
    errs = np.array([low[t][1] for t in sorted(low)]) / taus
    try:
        popt, _ = curve_fit(log_energy_cross_section, temps, np.log(taus),
                            sigma=errs, bounds=([0, -100], [2, -1]))
    except Exception:
        continue
    for t, (tau, _e) in high.items():
        pred = float(log_energy_cross_section(np.array([float(t)]), *popt)[0])
        dlog = (np.log(tau) - pred) / np.log(10)
        tau_decade = int(np.floor(np.log10(tau)))
        tband = '165-180' if t <= 180 else ('183-195' if t <= 195 else '197-210')
        cells[(tau_decade, tband)].append(dlog)

print('median dlog10(tau_meas/tau_pred) [n]  -- rows: fitted tau decade; cols: temperature band')
bands = ['165-180', '183-195', '197-210']
print(f'{"tau decade":>12} ' + ' '.join(f'{b:>18}' for b in bands))
for dec in sorted({d for d, _ in cells}):
    row = f'{"1e"+str(dec):>12} '
    for b in bands:
        arr = np.array(cells.get((dec, b), []))
        row += f'   {np.median(arr):+6.2f} [{arr.size:5d}]  ' if arr.size >= 30 else f'   {"--":>6} [{arr.size:5d}]  '
    print(row)

"""Is the readout pedestal really constant in t_ph?

For traps whose fitted tau at a high temperature is small, every point with
t_ph > 10*tau is pure pedestal. Pool (I - fit_offset)/sigma per dtph index
across many such traps: a flat profile means constant pedestal; a trend means
the pedestal depends on t_ph (dark current accumulated during pumping), which
would explain the residual high-T model deviation.
"""
import numpy as np
import h5py
from collections import defaultdict

TEMP = 190
with h5py.File('fit_dipole_spectra_signed_err_4.h5', 'r') as h5:
    prof = defaultdict(list)
    n_used = 0
    for quad_name, quad_group in h5.items():
        for dp_name, dp_group in quad_group.items():
            tname = f'temp_{TEMP}'
            if tname not in dp_group:
                continue
            tg = dp_group[tname]
            if not bool(tg.attrs.get('GoodIntensityFit', False)):
                continue
            tau = float(tg.attrs.get('fit_tau', np.nan))
            off = float(tg.attrs.get('fit_offset', np.nan))
            if not (np.isfinite(tau) and np.isfinite(off)) or tau > 2e-3:
                continue
            seconds = np.asarray(tg['seconds'][()], dtype=float)
            I = np.asarray(tg['intensities'][()], dtype=float)
            err = np.asarray(tg['intensity_err'][()], dtype=float)
            sign = np.sign(off) if off != 0 else 1.0
            tail = seconds > 10 * tau
            if tail.sum() < 6:
                continue
            n_used += 1
            for s, i, e in zip(seconds[tail], I[tail], err[tail]):
                # normalized pedestal residual, oriented so positive = |pedestal| grows
                prof[round(float(s), 6)].append(sign * (i - off) / e)
    print(f'{n_used} traps at {TEMP} K with tau < 2e-3 used')
    print('t_ph [s]      n    median norm. residual (pedestal - fitted C)/sigma')
    for s in sorted(prof):
        arr = np.array(prof[s])
        if arr.size < 50:
            continue
        print(f'{s:10.4f} {arr.size:6d}   {np.median(arr):+7.3f}')

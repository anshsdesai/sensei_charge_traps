"""Is the high-T Arrhenius deviation an analysis artifact?

Inject 3-parameter curves (typical high-T conditions: pedestal -700 e-, noise
35 e-, real dtph grids) and measure the bias of the fitted tau among fits that
pass the full selection. If the fitted tau is biased low by ~0.1-0.2 dex, the
observed deviation is an artifact of the fit; if unbiased, it is physics (or a
selection effect) in the data.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

from dipole_new import intensity_function_offset, INTENSITY_SHAPE_PEAK, INTENSITY_SHAPE_PEAK_X

GRID_18 = np.array([750, 1200, 2000, 3000, 5000, 8000, 18000, 28000, 45000, 70000,
                    100000, 170000, 260000, 400000, 650000, 1000000, 1500000, 2500000]) / 15e6
GRID_25 = np.array([750, 1200, 2000, 3000, 5000, 8000, 18000, 28000, 45000, 70000,
                    100000, 170000, 260000, 400000, 650000, 1000000, 1500000, 1800000,
                    2000000, 2500000, 3500000, 6500000, 8500000, 10500000, 15500000]) / 15e6

NOISE = 35.0
N_REAL = 300


def fit_one(seconds, intensities, sigma):
    offset0 = float(np.median(intensities))
    dev = intensities - offset0
    k = int(np.argmax(np.abs(dev)))
    p0 = [dev[k] / (3000 * INTENSITY_SHAPE_PEAK), float(np.clip(seconds[k] / INTENSITY_SHAPE_PEAK_X, 1e-8, 1000)), offset0]
    try:
        popt, pcov = curve_fit(
            intensity_function_offset, seconds, intensities,
            sigma=np.full(seconds.size, sigma), p0=p0,
            bounds=([-np.inf, 1e-8, -np.inf], [np.inf, 1000, np.inf]), maxfev=20000,
            absolute_sigma=True,
        )
    except Exception:
        return None
    perr = np.sqrt(np.diag(pcov))
    resid = intensities - intensity_function_offset(seconds, *popt)
    chisq = float(np.sum((resid / sigma) ** 2))
    dof = seconds.size - 3
    p_value = 1 - chi2.cdf(chisq, dof)
    w = np.full(seconds.size, 1 / sigma**2)
    cbest = np.sum(intensities * w) / np.sum(w)
    delta = float(np.sum(((intensities - cbest) / sigma) ** 2)) - chisq
    passed = (
        p_value > 0.05
        and perr[0] > 0 and abs(popt[0]) / perr[0] >= 3
        and delta >= 11.83
        and perr[1] / popt[1] <= 0.5
    )
    return popt[1] if passed else None


def main():
    rng = np.random.default_rng(99)
    print("grid  amplitude  pedestal  tau_true   pass_frac  median log10(tau_fit/tau_true)  [16%, 84%]")
    for grid_name, seconds in [('18d', GRID_18), ('25d', GRID_25)]:
        for amp_e in [300.0, 600.0, 2000.0]:
            coeff = amp_e / (3000 * INTENSITY_SHAPE_PEAK)
            for tau in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]:
                fitted = []
                n_pass = 0
                for _ in range(N_REAL):
                    truth = intensity_function_offset(seconds, coeff, tau, -700.0)
                    noisy = truth + rng.normal(0, NOISE, seconds.size)
                    out = fit_one(seconds, noisy, NOISE)
                    if out is not None:
                        n_pass += 1
                        fitted.append(np.log10(out / tau))
                fitted = np.array(fitted)
                if fitted.size:
                    print(f"{grid_name}  {amp_e:8.0f}  -700      {tau:8.0e}   {n_pass/N_REAL:8.2f}   "
                          f"{np.median(fitted):+8.3f}                 [{np.percentile(fitted,16):+.3f}, {np.percentile(fitted,84):+.3f}]")
                else:
                    print(f"{grid_name}  {amp_e:8.0f}  -700      {tau:8.0e}   {n_pass/N_REAL:8.2f}   no passing fits")


if __name__ == '__main__':
    main()

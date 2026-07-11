"""Leave-one-temperature-out calibration test (codex5.5's decisive test).

For each target high T: build a cohort of traps that are BRIGHT + in-window + grid-easy
(grid p_det>0.9), with amplitude estimated ONLY from colder/intermediate good temps
(never the target T). Then compare the observed GoodIntensityFit rate at the withheld T
to the grid prediction. ~0.9 -> grid is right for good traps (H2). ~0.45-0.6 -> grid
misses real high-T degradation (H1). Also tally the failure cut among cohort misses.
"""
import sys
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import dipole
from trap_completeness_method3.src.single_curve_recovery import _fit_one_curve
from trap_completeness_method3.src.analysis_flavors import get_analysis_flavor, load_delta_chi2_thresholds

GRID = ROOT / "trap_completeness_method3" / "cache" / "08_pdet_grid_minimal_caldet_v1.h5"
CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
AMP_T_RANGE = (140, 165)   # estimate amplitude from these temps only (high recovery)
TARGETS = [175, 185, 190, 195, 200]

with h5py.File(GRID, "r") as f:
    Tg = [int(round(x)) for x in f["grid/temperature_K"][:]]
    tau = f["grid/tau_seconds"][:].astype(float)
    A = f["grid/amplitude_electrons"][:].astype(float)
    pdet = f["results/p_det"][:].astype(float)
logtau = np.log(tau)
interp = {Tg[i]: RegularGridInterpolator((logtau, A), pdet[i], bounds_error=False, fill_value=None)
          for i in range(len(Tg))}

def gridp(T, tau_val, amp):
    if tau_val < tau.min() or tau_val > tau.max():
        return 0.0
    v = float(interp[T]((np.log(tau_val), float(np.clip(amp, A.min(), A.max())))))
    return float(np.clip(v, 0.0, 1.0))

flavor = get_analysis_flavor("minimal_caldet")
thresh = load_delta_chi2_thresholds(flavor)

# Load traps
traps = []
with h5py.File(CAT, "r") as f:
    for q in f:
        if not isinstance(f[q], h5py.Group):
            continue
        for dp in f[q]:
            g = f[q][dp]
            if not isinstance(g, h5py.Group) or "energy_BestFitEnergy" not in g.attrs:
                continue
            xs = float(g.attrs["energy_BestFitCrossSection"])
            if not (xs > 0):
                continue
            E = float(g.attrs["energy_BestFitEnergy"]); logsig = float(np.log(xs))
            amp_other = {}; good = set(); curves = {}
            for name in g:
                tgrp = g[name]
                if not isinstance(tgrp, h5py.Group) or not name.startswith("temp_"):
                    continue
                T = int(name.split("_")[1])
                if bool(tgrp.attrs.get("GoodIntensityFit", False)):
                    good.add(T)
                    c = tgrp.attrs.get("fit_coeff")
                    if c is not None and np.isfinite(float(c)) and AMP_T_RANGE[0] <= T <= AMP_T_RANGE[1]:
                        amp_other[T] = abs(float(c)) * 3000.0
                if "seconds" in tgrp and "intensities" in tgrp:
                    curves[T] = (np.asarray(tgrp["seconds"][()], float),
                                 np.asarray(tgrp["intensities"][()], float),
                                 np.asarray(tgrp["intensity_err"][()], float),
                                 float(tgrp.attrs.get("image_sigma", np.nan)))
            traps.append(dict(E=E, logsig=logsig, good=good, amp_other=amp_other, curves=curves))

print(f"loaded {len(traps)} traps; amplitude estimated from good temps in {AMP_T_RANGE}")
print(f"{'T':>4} {'cohort_n':>8} {'grid_pred':>9} {'obs_rate':>8}   failure cuts among cohort misses")
for T in TARGETS:
    n = 0; obs = 0; preds = []; cuts = Counter()
    for tr in traps:
        if T not in tr["curves"]:        # must be measured at the withheld T
            continue
        amps = [v for k, v in tr["amp_other"].items() if k != T]
        if not amps:                     # need an independent (non-T) amplitude estimate
            continue
        A_proxy = float(np.median(amps))
        tau_T = float(np.exp(dipole.log_energy_cross_section(float(T), tr["E"], tr["logsig"])))
        p = gridp(T, tau_T, A_proxy)
        if p <= 0.9:                     # cohort = bright + in-window + grid-easy
            continue
        n += 1; preds.append(p)
        is_good = T in tr["good"]
        obs += int(is_good)
        if not is_good:
            s, i, e, isig = tr["curves"][T]
            fit = _fit_one_curve(s, i, e, isig, analysis_flavor=flavor.name, temperature_k=int(T),
                                 delta_chi2_threshold_by_temperature=thresh, _flavor=flavor)
            cuts[fit.get("controlling_failure_cut") or "unknown"] += 1
    if n:
        nfail = sum(cuts.values())
        top = ", ".join(f"{c}={v/nfail:.0%}" for c, v in cuts.most_common(3)) if nfail else "-"
        print(f"{T:>4} {n:>8} {np.mean(preds):>9.2f} {obs/n:>8.2f}   {top}")
    else:
        print(f"{T:>4} {0:>8} {'-':>9} {'-':>8}")

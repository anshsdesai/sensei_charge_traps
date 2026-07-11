"""Why does the intensity-fit p_value fail on bright in-window high-T curves?

For the same unbiased holdout cohort (bright via cold temps, in-window, grid p_det>0.9),
re-fit at the withheld high T and compare reduced_chi2 of the p_value FAILURES vs the
PASSES, and locate the p=0.05 threshold in reduced-chi2 terms.

  - failures with reduced_chi2 just above threshold (~visually fine) -> Wilks over-rejection
    of precise fits (fixable by a dof-insensitive reduced_chi2 cut; recovers the traps).
  - failures with reduced_chi2 >> threshold (>5,>10) -> genuine model/noise misfit (real loss).
  - passes reduced_chi2 ~1 -> noise model is calibrated (not a global underestimate).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import h5py
from scipy.stats import chi2 as chi2dist
from scipy.interpolate import RegularGridInterpolator
import dipole
from trap_completeness_method3.src.single_curve_recovery import _fit_one_curve
from trap_completeness_method3.src.analysis_flavors import get_analysis_flavor, load_delta_chi2_thresholds

GRID = ROOT / "trap_completeness_method3" / "cache" / "08_pdet_grid_minimal_caldet_v1.h5"
CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
AMP_T_RANGE = (140, 165)
TARGETS = [175, 185, 195]

with h5py.File(GRID, "r") as f:
    Tg = [int(round(x)) for x in f["grid/temperature_K"][:]]
    tau = f["grid/tau_seconds"][:].astype(float); A = f["grid/amplitude_electrons"][:].astype(float)
    pdet = f["results/p_det"][:].astype(float)
logtau = np.log(tau)
interp = {Tg[i]: RegularGridInterpolator((logtau, A), pdet[i], bounds_error=False, fill_value=None) for i in range(len(Tg))}
def gridp(T, tv, amp):
    if tv < tau.min() or tv > tau.max(): return 0.0
    return float(np.clip(interp[T]((np.log(tv), float(np.clip(amp, A.min(), A.max())))), 0, 1))

flavor = get_analysis_flavor("minimal_caldet"); thresh = load_delta_chi2_thresholds(flavor)

traps = []
with h5py.File(CAT, "r") as f:
    for q in f:
        if not isinstance(f[q], h5py.Group): continue
        for dp in f[q]:
            g = f[q][dp]
            if not isinstance(g, h5py.Group) or "energy_BestFitEnergy" not in g.attrs: continue
            xs = float(g.attrs["energy_BestFitCrossSection"])
            if not (xs > 0): continue
            E = float(g.attrs["energy_BestFitEnergy"]); logsig = float(np.log(xs))
            amp_other = {}; good = set(); curves = {}
            for name in g:
                tgrp = g[name]
                if not isinstance(tgrp, h5py.Group) or not name.startswith("temp_"): continue
                T = int(name.split("_")[1])
                if bool(tgrp.attrs.get("GoodIntensityFit", False)):
                    good.add(T); c = tgrp.attrs.get("fit_coeff")
                    if c is not None and np.isfinite(float(c)) and AMP_T_RANGE[0] <= T <= AMP_T_RANGE[1]:
                        amp_other[T] = abs(float(c)) * 3000.0
                if "seconds" in tgrp and "intensities" in tgrp:
                    curves[T] = (np.asarray(tgrp["seconds"][()], float), np.asarray(tgrp["intensities"][()], float),
                                 np.asarray(tgrp["intensity_err"][()], float), float(tgrp.attrs.get("image_sigma", np.nan)))
            traps.append(dict(E=E, logsig=logsig, good=good, amp_other=amp_other, curves=curves))

def pct(a, p): return float(np.percentile(a, p)) if len(a) else float("nan")
for T in TARGETS:
    pass_rc, pfail_rc, dofs = [], [], []
    n_pass = n_pfail = n_otherfail = 0
    for tr in traps:
        if T not in tr["curves"]: continue
        amps = [v for k, v in tr["amp_other"].items() if k != T]
        if not amps: continue
        tau_T = float(np.exp(dipole.log_energy_cross_section(float(T), tr["E"], tr["logsig"])))
        if gridp(T, tau_T, float(np.median(amps))) <= 0.9: continue
        s, i, e, isig = tr["curves"][T]
        fit = _fit_one_curve(s, i, e, isig, analysis_flavor=flavor.name, temperature_k=int(T),
                             delta_chi2_threshold_by_temperature=thresh, _flavor=flavor)
        rc = fit.get("fit_reduced_chi_squared")
        if fit["good_intensity_fit"]:
            n_pass += 1
            if rc is not None and np.isfinite(rc): pass_rc.append(rc)
        elif fit.get("controlling_failure_cut") == "p_value":
            n_pfail += 1; dofs.append(len(s) - 3)
            if rc is not None and np.isfinite(rc): pfail_rc.append(rc)
        else:
            n_otherfail += 1
    pfail_rc = np.array(pfail_rc); pass_rc = np.array(pass_rc)
    med_dof = int(np.median(dofs)) if dofs else 0
    rc_thresh = chi2dist.ppf(0.95, med_dof) / med_dof if med_dof else float("nan")
    print(f"\n=== T={T}K ===  cohort: pass={n_pass}  p_value-fail={n_pfail}  other-fail={n_otherfail}")
    print(f"  passes reduced_chi2:  median={pct(pass_rc,50):.2f}  (noise calibrated if ~1)")
    print(f"  p=0.05 reduced_chi2 threshold at median dof={med_dof}:  {rc_thresh:.2f}")
    if pfail_rc.size:
        print(f"  p_value-FAIL reduced_chi2: median={pct(pfail_rc,50):.2f}  p25={pct(pfail_rc,25):.2f}  "
              f"p75={pct(pfail_rc,75):.2f}  p95={pct(pfail_rc,95):.2f}")
        print(f"    frac < 3 (visually fine, Wilks) = {(pfail_rc<3).mean():.0%} | "
              f"3-5 = {((pfail_rc>=3)&(pfail_rc<5)).mean():.0%} | "
              f">5 = {(pfail_rc>=5).mean():.0%} | >10 (gross) = {(pfail_rc>=10).mean():.0%}")

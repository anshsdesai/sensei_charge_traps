"""Step 0: quantitative residual audit of high-T intensity-fit failures.

For the leakage-free bright/in-window/grid-easy holdout cohort, reconstruct each curve's
fit residuals from the STORED fit params (no re-fitting) and compute a per-curve taxonomy:
  - reduced_chi2
  - worst2_frac: share of chi2 from the worst 2 points (outlier-dominated)
  - resid_autocorr: lag-1 autocorrelation of residuals ordered by t_ph (coherent deviation)
  - tau_ratio_dex: |log10(fit_tau / SRH tau(T))|, tau(T) from the trap's E/sigma (target T was
    NOT a good temp for the failures, so this is leakage-free for them)
  - dchi2_const: delta chi2 vs best constant (bump evidence)
Then assign each FAILURE one primary mechanism label and report the composition.
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
from dipole_new import intensity_function_offset

GRID = ROOT / "trap_completeness_method3" / "cache" / "08_pdet_grid_minimal_caldet_v1.h5"
CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
AMP_T_RANGE = (140, 165)
TARGETS = [175, 185, 195]
DCHI2_BUMP = 11.83          # below this -> "no real bump"
TAU_OFF_DEX = 0.5           # |log10(fit/pred)| above this -> tau-inconsistent (>~3x)
OUTLIER_FRAC = 0.5          # worst-2 points > this share of chi2 -> outlier-dominated
AUTOCORR = 0.35            # lag-1 residual autocorr above this -> coherent deviation

with h5py.File(GRID, "r") as f:
    Tg = [int(round(x)) for x in f["grid/temperature_K"][:]]
    tau = f["grid/tau_seconds"][:].astype(float); A = f["grid/amplitude_electrons"][:].astype(float)
    pdet = f["results/p_det"][:].astype(float)
logtau = np.log(tau)
interp = {Tg[i]: RegularGridInterpolator((logtau, A), pdet[i], bounds_error=False, fill_value=None) for i in range(len(Tg))}
def gridp(T, tv, amp):
    if tv < tau.min() or tv > tau.max(): return 0.0
    return float(np.clip(interp[T]((np.log(tv), float(np.clip(amp, A.min(), A.max())))), 0, 1))

def curve_stats(seconds, inten, err, coeff, tauf, offset):
    order = np.argsort(seconds)
    s = seconds[order]; y = inten[order]; e = err[order]
    model = intensity_function_offset(s, coeff, tauf, offset)
    r = (y - model) / e
    chi2 = float(np.sum(r**2)); dof = max(len(s) - 3, 1)
    red = chi2 / dof
    sq = np.sort(r**2)
    worst2 = float(sq[-2:].sum() / chi2) if chi2 > 0 else np.nan
    ac = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 and np.std(r) > 0 else 0.0
    # delta chi2 vs best constant
    w = 1.0 / e**2; cstar = float(np.sum(y * w) / np.sum(w))
    chi2_const = float(np.sum(((y - cstar) / e)**2))
    return red, worst2, ac, chi2_const - chi2

# Load traps
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
            amp_other = {}; good = set(); cur = {}
            for name in g:
                tgrp = g[name]
                if not isinstance(tgrp, h5py.Group) or not name.startswith("temp_"): continue
                T = int(name.split("_")[1])
                isgood = bool(tgrp.attrs.get("GoodIntensityFit", False))
                if isgood:
                    good.add(T); c = tgrp.attrs.get("fit_coeff")
                    if c is not None and np.isfinite(float(c)) and AMP_T_RANGE[0] <= T <= AMP_T_RANGE[1]:
                        amp_other[T] = abs(float(c)) * 3000.0
                if "seconds" in tgrp and "intensities" in tgrp:
                    cur[T] = dict(s=np.asarray(tgrp["seconds"][()], float),
                                  y=np.asarray(tgrp["intensities"][()], float),
                                  e=np.asarray(tgrp["intensity_err"][()], float),
                                  coeff=float(tgrp.attrs.get("fit_coeff", np.nan)),
                                  tauf=float(tgrp.attrs.get("fit_tau", np.nan)),
                                  off=float(tgrp.attrs.get("fit_offset", 0.0)),
                                  good=isgood)
            traps.append(dict(E=E, logsig=logsig, good=good, amp_other=amp_other, cur=cur))

def classify(red, worst2, ac, dchi2, tau_dex):
    if dchi2 < DCHI2_BUMP:           return "no_bump_flat"
    if not np.isfinite(tau_dex) or tau_dex > TAU_OFF_DEX: return "tau_inconsistent"
    if worst2 > OUTLIER_FRAC:        return "outlier_dominated"
    if ac > AUTOCORR:                return "coherent_deviation"
    if red >= 5:                     return "severe_misfit"
    if red >= 3:                     return "moderate_misfit"
    return "mild_misfit"

for T in TARGETS:
    comp = Counter(); n_pass = 0; pass_red = []
    for tr in traps:
        if T not in tr["cur"]: continue
        amps = [v for k, v in tr["amp_other"].items() if k != T]
        if not amps: continue
        tau_pred = float(np.exp(dipole.log_energy_cross_section(float(T), tr["E"], tr["logsig"])))
        if gridp(T, tau_pred, float(np.median(amps))) <= 0.9: continue
        c = tr["cur"][T]
        if not (np.isfinite(c["coeff"]) and np.isfinite(c["tauf"]) and c["tauf"] > 0): continue
        red, worst2, ac, dchi2 = curve_stats(c["s"], c["y"], c["e"], c["coeff"], c["tauf"], c["off"])
        if c["good"]:
            n_pass += 1; pass_red.append(red); continue
        tau_dex = abs(np.log10(c["tauf"] / tau_pred)) if tau_pred > 0 else np.inf
        comp[classify(red, worst2, ac, dchi2, tau_dex)] += 1
    nf = sum(comp.values())
    print(f"\n=== T={T}K ===  cohort passes={n_pass} (med reduced_chi2={np.median(pass_red):.2f})  failures={nf}")
    print("  failure composition (primary mechanism):")
    for lab in ["no_bump_flat", "tau_inconsistent", "outlier_dominated", "coherent_deviation",
                "moderate_misfit", "severe_misfit", "mild_misfit"]:
        if comp.get(lab):
            print(f"    {lab:18s} {comp[lab]:5d}  {comp[lab]/nf:5.0%}")
    # recoverable vs not, rough
    recov = comp["outlier_dominated"] + comp["mild_misfit"]
    junk = comp["no_bump_flat"] + comp["tau_inconsistent"]
    realmisfit = comp["coherent_deviation"] + comp["moderate_misfit"] + comp["severe_misfit"]
    print(f"  -> potentially-recoverable ~{recov/nf:.0%} | should-not-count ~{junk/nf:.0%} | "
          f"genuine-misfit ~{realmisfit/nf:.0%}")

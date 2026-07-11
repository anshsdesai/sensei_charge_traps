"""Probe: how sensitive is the Method-3 efficiency curve / hidden-trap count to
(1) the red-dots-are-efficiency-shaped bias (Horvitz-Thompson 1/P reweighting), and
(2) the choice of slicing axis -- global-E-per-tau-slice (production) vs.
    same-sigma extrapolation (equivalent to doing the correction in (E, sigma) axes).

Read-only on cached method3 artifacts (minimal_caldet flavor, the live one).
"""
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gamma

repo = Path(r"C:\Users\Ansh\Projects\sensei_charge_traps")
cache = repo / "trap_completeness_method3" / "cache"

ENERGY_FIT_SURVIVAL = 0.972
KT_135 = 8.617333262e-5 * 135.0  # eV

with h5py.File(cache / "09_characterization_probability_minimal_caldet_v1.h5", "r") as h5:
    tau_grid = h5["grid/tau_135_seconds"][:]
    E_grid = h5["grid/E_eV"][:]
    p4 = h5["results/p_characterized_n_good_4"][:] * ENERGY_FIT_SURVIVAL

with open(cache / "01_records_minimal_caldet_ngood4.csv", newline="") as fh:
    recs = list(csv.DictReader(fh))
obs_E = np.array([float(r["E_eV"]) for r in recs])
obs_tau = np.array([float(r["tau_135_seconds"]) for r in recs])
n_dots = obs_E.size

logt_grid = np.log10(tau_grid)

print(f"grid: tau [{tau_grid[0]:.2e}, {tau_grid[-1]:.2e}] x{tau_grid.size}, "
      f"E [{E_grid[0]:.3f}, {E_grid[-1]:.3f}] x{E_grid.size}; dots: {n_dots}")

# --- How correlated are the dots in (E, log tau)? (tests the factorization assumption)
lt_obs = np.log10(obs_tau)
rho = np.corrcoef(lt_obs, obs_E)[0, 1]
slope, intercept = np.polyfit(lt_obs, obs_E, 1)
resid = obs_E - (slope * lt_obs + intercept)
print(f"\ndot correlation corr(E, log10 tau135) = {rho:.3f}")
print(f"ridge fit: E = {slope:.4f}*log10(tau) + {intercept:.4f}  "
      f"(same-sigma SRH slope would be kT*ln10 = {KT_135*np.log(10):.4f} eV/decade)")
print(f"E spread: global std = {obs_E.std():.4f} eV, residual-around-ridge std = {resid.std():.4f} eV")

# --- P at each observed dot (bilinear in (log tau, E)), for HT weights
def p_at_points(taus, Es):
    lt = np.log10(taus)
    it = np.clip(np.searchsorted(logt_grid, lt) - 1, 0, logt_grid.size - 2)
    wt = np.clip((lt - logt_grid[it]) / (logt_grid[it + 1] - logt_grid[it]), 0.0, 1.0)
    Ec = np.clip(Es, E_grid[0], E_grid[-1])
    lo = np.array([np.interp(e, E_grid, p4[i]) for i, e in zip(it, Ec)])
    hi = np.array([np.interp(e, E_grid, p4[i + 1]) for i, e in zip(it, Ec)])
    return (1 - wt) * lo + wt * hi

P_dots = p_at_points(obs_tau, obs_E)
print(f"\nP(char) at dots: min {P_dots.min():.3f}, median {np.median(P_dots):.3f}, "
      f"frac<0.5 {np.mean(P_dots < 0.5):.3%}, frac<0.8 {np.mean(P_dots < 0.8):.3%}")
w_ht = 1.0 / np.clip(P_dots, 0.01, None)

# --- The four efficiency curves on tau_grid
curve_default = np.array([np.interp(obs_E, E_grid, row).mean() for row in p4])
curve_ht = np.array([np.average(np.interp(obs_E, E_grid, row), weights=w_ht) for row in p4])

# same-sigma: shift each dot's E along its own constant-sigma SRH line to the slice tau
# log tau = C - log sigma + E/kT  =>  at fixed sigma, dE = kT * dln(tau)
curve_ss = np.empty_like(curve_default)
curve_ss_ht = np.empty_like(curve_default)
frac_clipped = np.empty_like(curve_default)
for j, tau_j in enumerate(tau_grid):
    E_shift = obs_E + KT_135 * np.log(tau_j / obs_tau)
    frac_clipped[j] = np.mean((E_shift < E_grid[0]) | (E_shift > E_grid[-1]))
    vals = np.interp(np.clip(E_shift, E_grid[0], E_grid[-1]), E_grid, p4[j])
    curve_ss[j] = vals.mean()
    curve_ss_ht[j] = np.average(vals, weights=w_ht)

# --- spot values
sel_taus = [1e2, 1e3, 1e4, 1e5, 3e5, 1e6, 1e7, tau_grid[-1]]
print("\n  tau135      eps_default  eps_HT    eps_sameSig  eps_sameSig_HT  fracEclip")
for t in sel_taus:
    if t < tau_grid[0] or t > tau_grid[-1]:
        continue
    def at(curve):
        return np.interp(np.log10(t), logt_grid, curve)
    print(f"  {t:9.2e}   {at(curve_default):9.4f}  {at(curve_ht):9.4f}  "
          f"{at(curve_ss):9.4f}    {at(curve_ss_ht):9.4f}     {np.interp(np.log10(t), logt_grid, frac_clipped):.2f}")

# --- corrected populations (mirror _completeness_correction + paper bounds)
with np.load(repo / "tau_at_135k_hist_minimal_caldet.npz") as data:
    tau_edges = data["bin_edges"]
    if "total_taus" in data:
        tau_hist, _ = np.histogram(data["total_taus"], bins=tau_edges)
    elif "tau_at_135s" in data:
        tau_hist, _ = np.histogram(data["tau_at_135s"], bins=tau_edges)
    else:
        tau_hist = data["hist"]

centers = np.sqrt(tau_edges[:-1] * tau_edges[1:])
MIN_EFF, TAU_MIN, TAU_MAX = 1e-3, 6e-5, 5e7
DRIVER = (3e5, 5e7)

print(f"\nraw trap count in histogram: {int(tau_hist.sum())}")
print("\n  curve            corrected  UL90-corrected   corrected(driver band)  UL90(driver band)")
for name, curve in [("default (paper)", curve_default), ("HT-reweighted", curve_ht),
                    ("same-sigma", curve_ss), ("same-sigma HT", curve_ss_ht)]:
    eff = np.interp(np.log10(centers), logt_grid, curve, left=np.nan, right=np.nan)
    valid = np.isfinite(eff) & (eff >= MIN_EFF) & (centers >= TAU_MIN) & (centers <= TAU_MAX)
    corr = np.where(valid, tau_hist / np.where(valid, eff, 1), 0.0)
    ul = np.where(valid, gamma.ppf(0.90, tau_hist + 1) / np.where(valid, eff, 1), 0.0)
    band = valid & (centers >= DRIVER[0]) & (centers <= DRIVER[1])
    print(f"  {name:<16} {corr.sum():9.0f}  {ul.sum():14.0f}   {corr[band].sum():21.0f}  {ul[band].sum():16.0f}"
          f"   (valid bins: {valid.sum()}, tau range [{centers[valid].min():.1e}, {centers[valid].max():.1e}])")

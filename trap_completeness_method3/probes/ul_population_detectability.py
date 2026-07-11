"""Probe 2: detection probability of the UL/effcorr populations AS THE SIMULATION
BUILDS THEM. Replicates ccd_simulation.CCD.__init__'s sigma assignment (nearest
K=20 measured (tau135, sigma) pairs in log tau), inverts SRH at 135 K for the
implied E, and looks up P(characterized | tau135, E) from the live stage-09 grid.
Read-only.
"""
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import gamma

repo = Path(r"C:\Users\Ansh\Projects\sensei_charge_traps")
sys.path.insert(0, str(repo))
from dipole import log_energy_cross_section  # noqa: E402

cache = repo / "trap_completeness_method3" / "cache"
ENERGY_FIT_SURVIVAL = 0.972

with h5py.File(cache / "09_characterization_probability_minimal_caldet_v1.h5", "r") as h5:
    tau_grid = h5["grid/tau_135_seconds"][:]
    E_grid = h5["grid/E_eV"][:]
    p4 = h5["results/p_characterized_n_good_4"][:] * ENERGY_FIT_SURVIVAL
logt_grid = np.log10(tau_grid)

with open(cache / "01_records_minimal_caldet_ngood4.csv", newline="") as fh:
    obs_E = np.array([float(r["E_eV"]) for r in csv.DictReader(fh)])

# production efficiency curve (global-E average), for building the UL/effcorr seeds
curve_default = np.array([np.interp(obs_E, E_grid, row).mean() for row in p4])

# measured (tau135, sigma) pairs exactly as run_campaign feeds run_ccd_simulation
pairs = np.load(repo / "trap_tau135_sigma_pairs_minimal_caldet.npz")
pair_tau135 = np.asarray(pairs["tau135"], dtype=float)
pair_sigma = np.asarray(pairs["sigma"], dtype=float)
order = np.argsort(pair_tau135)
sorted_logtau = np.log(pair_tau135[order])
sorted_sigma = pair_sigma[order]
K = min(20, len(sorted_sigma))
print(f"pairs: {pair_tau135.size}, tau range [{pair_tau135.min():.2e}, {pair_tau135.max():.2e}] s, "
      f"sigma range [{pair_sigma.min():.2e}, {pair_sigma.max():.2e}] cm^2")

def sim_sigma_candidates(tau):
    """The K sigma values CCD.__init__ draws uniformly from for a trap at tau."""
    ins = np.searchsorted(sorted_logtau, np.log(tau))
    lo = int(np.clip(ins - K // 2, 0, len(sorted_sigma) - K))
    return sorted_sigma[lo:lo + K]

def implied_E(tau, sigma):
    """Invert SRH at 135 K: E such that tau_e(135K; E, sigma) = tau."""
    lt_of_E = log_energy_cross_section(135.0, E_grid, np.log(sigma))  # ln tau, monotone in E
    return float(np.interp(np.log(tau), lt_of_E, E_grid))

def p_char(tau, E):
    """Bilinear lookup of P(characterized | tau135, E) on the stage-09 grid."""
    lt = np.log10(tau)
    i = int(np.clip(np.searchsorted(logt_grid, lt) - 1, 0, logt_grid.size - 2))
    w = np.clip((lt - logt_grid[i]) / (logt_grid[i + 1] - logt_grid[i]), 0.0, 1.0)
    Ec = np.clip(E, E_grid[0], E_grid[-1])
    return (1 - w) * np.interp(Ec, E_grid, p4[i]) + w * np.interp(Ec, E_grid, p4[i + 1])

# --- build the corrected populations exactly like _completeness_correction
with np.load(repo / "tau_at_135k_hist_minimal_caldet.npz") as data:
    tau_edges = data["bin_edges"]
    if "total_taus" in data:
        tau_hist, _ = np.histogram(data["total_taus"], bins=tau_edges)
    elif "tau_at_135s" in data:
        tau_hist, _ = np.histogram(data["tau_at_135s"], bins=tau_edges)
    else:
        tau_hist = data["hist"]

centers = np.sqrt(tau_edges[:-1] * tau_edges[1:])
eff = np.interp(np.log10(centers), logt_grid, curve_default, left=np.nan, right=np.nan)
valid = np.isfinite(eff) & (eff >= 1e-3) & (centers >= 6e-5) & (centers <= 5e7)
effcorr = np.where(valid, tau_hist / np.where(valid, eff, 1), 0.0)
ul = np.where(valid, gamma.ppf(0.90, tau_hist + 1) / np.where(valid, eff, 1), 0.0)

# --- per-bin detectability of the as-simulated population (mean over the K sigma draws)
p_sim = np.zeros_like(centers)
sig_lo = np.zeros_like(centers)
sig_hi = np.zeros_like(centers)
E_mid = np.zeros_like(centers)
for b in np.flatnonzero(valid):
    cands = sim_sigma_candidates(centers[b])
    ps = [p_char(centers[b], implied_E(centers[b], s)) for s in cands]
    p_sim[b] = float(np.mean(ps))
    sig_lo[b], sig_hi[b] = cands.min(), cands.max()
    E_mid[b] = implied_E(centers[b], float(np.median(cands)))

def report(label, lo, hi):
    m = valid & (centers >= lo) & (centers <= hi)
    if not m.any():
        return
    add_ul = ul[m].sum()
    add_ec = effcorr[m].sum()
    would_see_ul = (ul[m] * p_sim[m]).sum()
    would_see_ec = (effcorr[m] * p_sim[m]).sum()
    print(f"\n{label}  (tau in [{lo:.0e}, {hi:.0e}] s, {m.sum()} bins)")
    print(f"  observed counts in these bins:            {int(tau_hist[m].sum())}")
    print(f"  effcorr population placed here:           {add_ec:8.0f}  -> would be characterized: {would_see_ec:8.0f}")
    print(f"  UL population placed here:                {add_ul:8.0f}  -> would be characterized: {would_see_ul:8.0f}")
    print(f"  mean P(char) of as-simulated traps here:  {(ul[m]*p_sim[m]).sum()/max(add_ul,1e-9):.3f}")

print("\nper-bin detail (tail bins, tau >= 1e4 s):")
print("  tau_center   eps_prod   sim sigma range          implied E   P(char|as-simulated)  UL added")
for b in np.flatnonzero(valid & (centers >= 1e4)):
    print(f"  {centers[b]:9.2e}  {eff[b]:8.4f}  [{sig_lo[b]:.1e}, {sig_hi[b]:.1e}]  {E_mid[b]:8.3f}    {p_sim[b]:8.3f}          {ul[b]:8.1f}")

report("DRIVER BAND", 3e5, 5e7)
report("FULL TAIL", 1e4, 5e7)
report("WHOLE CORRECTED RANGE", 6e-5, 5e7)

"""Probe 3: the 2D per-trap inverse-probability (Horvitz-Thompson) correction.
Each observed trap i gets weight w_i = 1/P(char | tau_i, E_i) from the stage-09
grid. Totals and the weighted tau histogram are what a corrected seed would use.
Read-only.
"""
import csv
from pathlib import Path

import h5py
import numpy as np

repo = Path(r"C:\Users\Ansh\Projects\sensei_charge_traps")
cache = repo / "trap_completeness_method3" / "cache"
ENERGY_FIT_SURVIVAL = 0.972

with h5py.File(cache / "09_characterization_probability_minimal_caldet_v1.h5", "r") as h5:
    tau_grid = h5["grid/tau_135_seconds"][:]
    E_grid = h5["grid/E_eV"][:]
    p4 = h5["results/p_characterized_n_good_4"][:] * ENERGY_FIT_SURVIVAL
logt_grid = np.log10(tau_grid)

with open(cache / "01_records_minimal_caldet_ngood4.csv", newline="") as fh:
    recs = list(csv.DictReader(fh))
obs_E = np.array([float(r["E_eV"]) for r in recs])
obs_tau = np.array([float(r["tau_135_seconds"]) for r in recs])

lt = np.log10(obs_tau)
it = np.clip(np.searchsorted(logt_grid, lt) - 1, 0, logt_grid.size - 2)
wt = np.clip((lt - logt_grid[it]) / (logt_grid[it + 1] - logt_grid[it]), 0.0, 1.0)
Ec = np.clip(obs_E, E_grid[0], E_grid[-1])
plo = np.array([np.interp(e, E_grid, p4[i]) for i, e in zip(it, Ec)])
phi = np.array([np.interp(e, E_grid, p4[i + 1]) for i, e in zip(it, Ec)])
P = (1 - wt) * plo + wt * phi

w = 1.0 / P
print(f"dots: {P.size};  P min/median: {P.min():.3f}/{np.median(P):.3f};  "
      f"P<0.5: {(P < 0.5).sum()}, P<0.25: {(P < 0.25).sum()}  (no floor needed)")
print(f"\nHT-corrected total population:      {w.sum():8.1f}   (raw {P.size})")
for lo, hi, name in [(3e5, 5e7, "driver band"), (1e4, 5e7, "full tail"), (1e5, 5e7, "tau>1e5")]:
    m = (obs_tau >= lo) & (obs_tau <= hi)
    print(f"  {name:<12} raw {m.sum():5d}  ->  HT-corrected {w[m].sum():8.1f}")

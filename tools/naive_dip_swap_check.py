"""Lightweight check: recompute the naive measured-efficiency dip under the swapped per-temperature
GOF cut (reduced_chi2<X replacing p_value>0.05), holding the characterized-trap set fixed, to see
whether the OBSERVED dip rises from 0.414 toward Method-3 (pure 0.943 / hybrid 0.676).

Replicates naive_efficiency_closure.py's observed estimator: per-(trap, T_grid) point,
tau_e(T)=exp(LECS(T,E,logsig)); measured = good intensity fit at T; eff(bin)=measured/total;
dip = MIN eff over tau in (3e-3, 3e-2). Trap set = the 3798 records (held fixed).
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py, numpy as np
import dipole_new

CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
RECORDS = ROOT / "trap_completeness_method3" / "cache" / "01_records_minimal_caldet_ngood4.csv"
GRID = ROOT / "trap_completeness_method3" / "cache" / "08_pdet_grid_minimal_caldet_v1.h5"
THR = 11.83
DIP = (3e-3, 3e-2)

def ff(v, d=np.nan):
    try: return float(v)
    except (TypeError, ValueError): return d

def precuts(a):
    tau = ff(a.get("fit_tau")); te = ff(a.get("fit_tau_err"))
    rel = te/tau if (np.isfinite(tau) and tau != 0) else np.inf
    return ff(a.get("amplitude_significance")) >= 3 and ff(a.get("delta_chi2_vs_constant")) >= ff(a.get("delta_chi2_threshold"), THR) and rel <= 0.5

def good(a, rule, x=None):
    if not precuts(a): return False
    return ff(a.get("fit_p_value")) > 0.05 if rule == "baseline" else ff(a.get("fit_reduced_chi_squared")) < x

def main():
    with h5py.File(GRID, "r") as g:
        temps = g["grid/temperature_K"][:].astype(float)
    bins = np.geomspace(1e-7, 1e8, 75)
    centers = np.sqrt(bins[:-1] * bins[1:])
    dip_bins = (centers >= DIP[0]) & (centers <= DIP[1])

    recs = []
    with RECORDS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            recs.append((int(r["quadrant"]), int(r["row"]), int(r["col"]), float(r["E_eV"]), float(r["log_sigma"])))
    print(f"records (characterized traps): {len(recs)}; grid temps: {len(temps)}")

    Xs = [3.0, 5.0, 10.0]
    # tau_e(T) per (trap, T) and measured masks
    tau_pts = []
    meas = {"baseline": [], **{f"alt{X:g}": [] for X in Xs}}
    n_missing = 0
    with h5py.File(CAT, "r") as f:
        for q, row, col, E, logsig in recs:
            grp = f.get(f"quad_{q}/dp_{row}_{col}")
            taus = np.exp(dipole_new.log_energy_cross_section(temps, E, logsig))  # tau_e at each grid T
            tau_pts.append(taus)
            attrs_by_T = {}
            if grp is not None:
                for tn in grp:
                    if tn.startswith("temp_") and isinstance(grp[tn], h5py.Group):
                        attrs_by_T[float(tn.split("_")[1])] = {k: grp[tn].attrs[k] for k in grp[tn].attrs}
            else:
                n_missing += 1
            mb, ma = [], {X: [] for X in Xs}
            for T in temps:
                a = attrs_by_T.get(float(T))
                mb.append(bool(a) and good(a, "baseline"))
                for X in Xs:
                    ma[X].append(bool(a) and good(a, "alt", X))
            meas["baseline"].append(mb)
            for X in Xs:
                meas[f"alt{X:g}"].append(ma[X])
    tau_pts = np.array(tau_pts).reshape(-1)
    total, _ = np.histogram(tau_pts, bins=bins)

    def dip_eff(mask_key):
        m = np.array(meas[mask_key]).reshape(-1)
        num, _ = np.histogram(tau_pts[m], bins=bins)
        eff = np.divide(num, total, out=np.zeros(len(total)), where=total > 0)
        sel = dip_bins & (total > 0)
        return float(np.min(eff[sel])), float(np.mean(eff[sel]))

    print(f"missing catalog groups: {n_missing}")
    print(f"\n{'mask':>10} {'dip_min':>9} {'dip_mean':>9}")
    bmin, bmean = dip_eff("baseline")
    print(f"{'baseline':>10} {bmin:>9.3f} {bmean:>9.3f}   (reference observed dip_min = 0.414)")
    for X in Xs:
        mn, me = dip_eff(f"alt{X:g}")
        print(f"{'X=%g'%X:>10} {mn:>9.3f} {me:>9.3f}")
    print("\nMethod-3 reference: pure (uncond/cond) dip = 0.943, hybrid = 0.676")

    # figure: full efficiency curve baseline vs swapped, dip window shaded
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def curve(mask_key):
        m = np.array(meas[mask_key]).reshape(-1)
        num, _ = np.histogram(tau_pts[m], bins=bins)
        return np.divide(num, total, out=np.zeros(len(total)), where=total > 0)
    valid = total > 0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(DIP[0], DIP[1], color="0.9", label="dip window")
    ax.plot(centers[valid], curve("baseline")[valid], "k-o", ms=3, label="observed, p-value cut (dip=0.414)")
    ax.plot(centers[valid], curve("alt5")[valid], color="tab:green", lw=1.8, label="observed, reduced-chi2<5 (dip=0.596)")
    ax.plot(centers[valid], curve("alt10")[valid], color="tab:blue", lw=1.8, label="observed, reduced-chi2<10 (dip=0.642)")
    ax.axhline(0.676, color="tab:orange", ls="--", lw=1.2, label="hybrid M3xGOF (0.676)")
    ax.axhline(0.943, color="tab:red", ls=":", lw=1.2, label="pure Method 3 (0.943)")
    ax.set_xscale("log"); ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$\tau_e$ [s]"); ax.set_ylabel("Efficiency of measurement")
    ax.set_title("Naive measurement-efficiency dip under per-temperature GOF cut swap")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    out = ROOT / "figures" / "naive_dip_swap_check.png"
    fig.tight_layout(); fig.savefig(out, dpi=160)
    print("wrote", out)

if __name__ == "__main__":
    main()

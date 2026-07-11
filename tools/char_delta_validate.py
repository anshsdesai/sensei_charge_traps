"""Validate the gained/lost traps from the X=3 GOF-cut swap: plot each trap's Arrhenius
tau(T) showing which temps are good under baseline (p>0.05) vs alt (reduced_chi2<X), with
both energy-fit lines, so we can SEE whether gained traps are clean lines and lost traps
are broken by an added/removed temp.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dipole_new

CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
OUT = ROOT / "figures" / "char_delta_validate.png"
THR = 11.83
X = 3.0

def ff(v, d=np.nan):
    try: return float(v)
    except (TypeError, ValueError): return d

def precuts(a):
    tau = ff(a.get("fit_tau")); te = ff(a.get("fit_tau_err"))
    rel = te/tau if (np.isfinite(tau) and tau != 0) else np.inf
    return ff(a.get("amplitude_significance")) >= 3 and ff(a.get("delta_chi2_vs_constant")) >= ff(a.get("delta_chi2_threshold"), THR) and rel <= 0.5

def good(a, rule):
    if not precuts(a): return False
    return ff(a.get("fit_p_value")) > 0.05 if rule == "baseline" else ff(a.get("fit_reduced_chi_squared")) < X

def char(data):
    if len(data) < 4: return False, None
    T = np.array([d[0] for d in data], float); tau = np.array([d[1] for d in data], float)
    te = np.array([d[2] for d in data], float); sg = np.sign([d[3] for d in data])
    if not (np.all(np.isfinite(tau)) and np.all(tau > 0)): return False, None
    r = dipole_new.fit_energy_cross_section(T, tau, te, sg, wellBehavedThreshold=4, errors_are_absolute=True)
    ok = bool(r["WellBehavedTrap"]) and (r["EnergyFitFailed"] is False) and bool(r["GoodEnergyFit"])
    return ok, r

def collect():
    gained, lost = [], []
    with h5py.File(CAT, "r") as f:
        for q in f:
            if not isinstance(f[q], h5py.Group): continue
            for dp in f[q]:
                g = f[q][dp]
                if not isinstance(g, h5py.Group): continue
                rows = {}
                bd, ad = [], []
                for tn in g:
                    if not (tn.startswith("temp_") and isinstance(g[tn], h5py.Group)): continue
                    T = int(tn.split("_")[1]); a = {k: g[tn].attrs[k] for k in g[tn].attrs}
                    rows[T] = (ff(a.get("fit_tau")), ff(a.get("fit_tau_err")), ff(a.get("fit_coeff")),
                               good(a, "baseline"), good(a, "alt"))
                    tup = (T, ff(a.get("fit_tau")), ff(a.get("fit_tau_err")), ff(a.get("fit_coeff")))
                    if rows[T][3]: bd.append(tup)
                    if rows[T][4]: ad.append(tup)
                cb, rb = char(bd); ca, ra = char(ad)
                if ca and not cb: gained.append((f"{q}/{dp}", rows, rb, ra))
                if cb and not ca: lost.append((f"{q}/{dp}", rows, rb, ra))
    return gained, lost

def panel(ax, name, rows, rb, ra, kind):
    Ts = sorted(rows)
    for T in Ts:
        tau, te, coeff, gb, ga = rows[T]
        if not (np.isfinite(tau) and tau > 0): continue
        # color by membership: both=blue, baseline-only=gray(removed by alt), alt-only=red(added)
        if gb and ga: c, m = "tab:blue", "o"
        elif gb and not ga: c, m = "gray", "x"
        elif ga and not gb: c, m = "tab:red", "D"
        else: c, m = "lightgray", "."
        yerr = (te/tau)/np.log(10) if tau > 0 else 0
        ax.errorbar(T, np.log10(tau), yerr=yerr, fmt=m, color=c, ms=5, lw=0.8)
    # overlay energy-fit lines
    tgrid = np.linspace(min(Ts), max(Ts), 100)
    if ra is not None and ra.get("popt") is not None:
        ax.plot(tgrid, dipole_new.log_energy_cross_section(tgrid, *ra["popt"])/np.log(10), "tab:red", lw=1.2,
                label=f"alt fit rchi2={ra.get('reduced_chi_squared', float('nan')):.1f}")
    if rb is not None and rb.get("popt") is not None:
        ax.plot(tgrid, dipole_new.log_energy_cross_section(tgrid, *rb["popt"])/np.log(10), "tab:blue", lw=1.0, ls="--",
                label=f"base fit rchi2={rb.get('reduced_chi_squared', float('nan')):.1f}")
    ax.set_title(f"{kind}: {name}", fontsize=7); ax.tick_params(labelsize=6)
    ax.legend(fontsize=5); ax.set_xlabel("T [K]", fontsize=6); ax.set_ylabel("log10 tau", fontsize=6)

def main():
    gained, lost = collect()
    print(f"gained={len(gained)} lost={len(lost)}")
    rng = np.random.default_rng(3)
    gi = rng.choice(len(gained), size=min(4, len(gained)), replace=False)
    li = rng.choice(len(lost), size=min(4, len(lost)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for k, i in enumerate(gi): panel(axes[0][k], *gained[i][:1], gained[i][1], gained[i][2], gained[i][3], "GAINED")
    for k, i in enumerate(li): panel(axes[1][k], *lost[i][:1], lost[i][1], lost[i][2], lost[i][3], "LOST")
    fig.suptitle("X=3 census churn validation. blue=good both, red diamond=added by alt, gray x=removed by alt. Lines: energy fits.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT, dpi=140)
    print("wrote", OUT)
    print("\nGAINED sample (added temps make a clean Arrhenius line):")
    for i in gi: print(" ", gained[i][0], "alt_rchi2=%.2f" % gained[i][3]["reduced_chi_squared"])
    print("LOST sample (a removed/added temp breaks the fit or drops below 4):")
    for i in li:
        ra = lost[i][3]; rstr = "%.2f" % ra["reduced_chi_squared"] if (ra and ra.get("reduced_chi_squared") is not None) else "n/a(<4 or fail)"
        print(" ", lost[i][0], "alt_rchi2=", rstr)

if __name__ == "__main__":
    main()

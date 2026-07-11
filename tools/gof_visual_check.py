"""Visual sanity check for the GOF-cut recovery: plot real recovered casualties
across the reduced-chi2 range and decoy 'characterized' curves, with the stored
fit overlaid, so a human can judge whether the statistical 'good/bad' labels match
what the curves actually look like.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dipole_new import intensity_function_offset

CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
DEC = ROOT / "decoy_fit_signed.h5"
OUT = ROOT / "figures" / "gof_visual_check.png"

def precuts(a):
    amp = a.get("amplitude_significance", np.nan)
    dchi = a.get("delta_chi2_vs_constant", np.nan)
    thr = a.get("delta_chi2_threshold", 11.83)
    tau = a.get("fit_tau", np.nan); te = a.get("fit_tau_err", np.nan)
    rel = te/tau if (np.isfinite(tau) and tau != 0) else np.inf
    return (amp >= 3) and (dchi >= thr) and (rel <= 0.5)

def find_real_casualties(targets, per_bin=2, seed=7):
    """High-T points failing baseline p-value but passing pre-cuts; RANDOM sample within each
    target rchi2 band (+-15%), drawn from the full pool so panels are representative, not first-found."""
    pool = {t: [] for t in targets}
    with h5py.File(CAT, "r") as f:
        for q in f:
            if not isinstance(f[q], h5py.Group): continue
            for dp in f[q]:
                g = f[q][dp]
                if not isinstance(g, h5py.Group) or "energy_BestFitEnergy" not in g.attrs: continue
                for tn in g:
                    tg = g[tn]
                    if not (isinstance(tg, h5py.Group) and tn.startswith("temp_")): continue
                    T = int(tn.split("_")[1])
                    if T < 170 or "seconds" not in tg: continue
                    a = {k: tg.attrs[k] for k in tg.attrs}
                    if not precuts(a): continue
                    if not (a.get("fit_p_value", 1.0) <= 0.05): continue  # baseline casualty
                    rc = float(a.get("fit_reduced_chi_squared", np.nan))
                    if not np.isfinite(rc): continue
                    for t in targets:
                        if abs(rc - t) < t*0.15:
                            pool[t].append((f"{q}/{dp}", T))  # store id only; reread data after sampling
    rng = np.random.default_rng(seed)
    picks = {t: [] for t in targets}
    with h5py.File(CAT, "r") as f:
        for t in targets:
            cand = pool[t]
            idx = rng.choice(len(cand), size=min(per_bin, len(cand)), replace=False) if cand else []
            for i in idx:
                qdp, T = cand[i]; q, dp = qdp.split("/")
                tg = f[q][dp][f"temp_{T}"]; a = {k: tg.attrs[k] for k in tg.attrs}
                picks[t].append((qdp, T, float(a["fit_reduced_chi_squared"]),
                    np.array(tg["seconds"]), np.array(tg["intensities"]), np.array(tg["intensity_err"]),
                    a["fit_coeff"], a["fit_tau"], a["fit_offset"], a.get("fit_p_value")))
            print(f"  rchi2~{t}: pool={len(cand)} sampled={len(picks[t])}")
    return picks

def find_decoy_examples(n=2):
    """Decoy curves that pass pre-cuts at high-T (false-positive bumps) with good rchi2."""
    out = []
    with h5py.File(DEC, "r") as f:
        for q in f:
            if not isinstance(f[q], h5py.Group): continue
            for dp in f[q]:
                g = f[q][dp]
                if not isinstance(g, h5py.Group): continue
                for tn in g:
                    tg = g[tn]
                    if not (isinstance(tg, h5py.Group) and tn.startswith("temp_")): continue
                    T = int(tn.split("_")[1])
                    if T < 170: continue
                    a = {k: tg.attrs[k] for k in tg.attrs}
                    if precuts(a) and "seconds" in tg:
                        rc = float(a.get("fit_reduced_chi_squared", np.nan))
                        out.append((f"DECOY {q}/{dp}", T, rc,
                            np.array(tg["seconds"]), np.array(tg["intensities"]),
                            np.array(tg["intensity_err"]),
                            a["fit_coeff"], a["fit_tau"], a["fit_offset"], a.get("fit_p_value")))
                        if len(out) >= n: return out
    return out

def panel(ax, rec, color):
    name, T, rc, s, y, e, coeff, tau, off, pv = rec
    order = np.argsort(s)
    s, y, e = s[order], y[order], e[order]
    ax.errorbar(s, y, yerr=e, fmt="o", ms=3, color="k", alpha=0.6, lw=0.8)
    s_pos = s[s > 0]
    lo = s_pos.min() if s_pos.size else max(s.min(), 1e-4)
    xs = np.geomspace(lo, s.max(), 400)
    ax.plot(xs, intensity_function_offset(xs, coeff, tau, off), color=color, lw=1.6)
    ax.set_xscale("log")
    ax.set_title(f"{name}  T={T}K\n rchi2={rc:.1f}  p={pv:.1e}", fontsize=7)
    ax.tick_params(labelsize=6)

def main():
    targets = [3.0, 5.0, 10.0, 18.0]
    picks = find_real_casualties(targets, per_bin=2)
    decoys = find_decoy_examples(4)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    # row 0-1: real casualties across rchi2
    flat = []
    for t in targets:
        flat += picks[t][:2]
    for i, rec in enumerate(flat[:8]):
        panel(axes[i//4][i%4], rec, "tab:blue")
    # row 2: decoys
    for j, rec in enumerate(decoys[:4]):
        panel(axes[2][j], rec, "tab:red")
    axes[0][0].set_ylabel("REAL recovered\n(rchi2 ~3,5)", fontsize=8)
    axes[1][0].set_ylabel("REAL recovered\n(rchi2 ~10,18)", fontsize=8)
    axes[2][0].set_ylabel("DECOY false-pos\n(pass pre-cuts)", fontsize=8)
    fig.suptitle("Visual check: recovered real casualties (blue) vs decoy false positives (red), stored fit overlaid", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)
    print("\nReal casualty examples by target rchi2:")
    for t in targets:
        for p in picks[t][:2]:
            print(f"  rchi2~{t}: {p[0]} T={p[1]} rchi2={p[2]:.2f} p={p[9]:.2e} coeff={p[6]:.3g} tau={p[7]:.3g}")
    print("Decoy examples:")
    for d in decoys:
        print(f"  {d[0]} T={d[1]} rchi2={d[2]:.2f} p={d[9]:.2e} coeff={d[6]:.3g} tau={d[7]:.3g}")

if __name__ == "__main__":
    main()

"""Characterization delta: how many REAL traps become characterized when the per-temperature
GOF cut is swapped from p_value>0.05 to reduced_chi2<X, vs the baseline catalog.

Recomputes WellBehaved + dipole_new.fit_energy_cross_section (sign-aware, reduced_chi2<10 GOF)
under baseline and alt rules from stored per-temperature attrs. Validates the baseline replication
against the catalog's stored GoodEnergyFit before trusting the alt count.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import h5py
import numpy as np
import dipole_new

CAT = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
DEFAULT_THR = 11.83

def ff(v, d=np.nan):
    try: return float(v)
    except (TypeError, ValueError): return d

def precuts(a):
    amp = ff(a.get("amplitude_significance")); dchi = ff(a.get("delta_chi2_vs_constant"))
    thr = ff(a.get("delta_chi2_threshold"), DEFAULT_THR)
    tau = ff(a.get("fit_tau")); te = ff(a.get("fit_tau_err"))
    rel = te/tau if (np.isfinite(tau) and tau != 0) else np.inf
    return (amp >= 3) and (dchi >= thr) and (rel <= 0.5)

def good(a, rule, x):
    if not precuts(a): return False
    if rule == "baseline": return ff(a.get("fit_p_value")) > 0.05
    return ff(a.get("fit_reduced_chi_squared")) < x

def characterized(good_temps_data):
    """good_temps_data: list of (T, tau, tau_err, coeff). Returns (is_char, wellbehaved)."""
    if len(good_temps_data) < 4:
        return False, False
    T = np.array([d[0] for d in good_temps_data], float)
    tau = np.array([d[1] for d in good_temps_data], float)
    te = np.array([d[2] for d in good_temps_data], float)
    sgn = np.sign([d[3] for d in good_temps_data])
    if not (np.all(np.isfinite(tau)) and np.all(tau > 0) and np.all(np.isfinite(te))):
        return False, True
    res = dipole_new.fit_energy_cross_section(T, tau, te, sgn, wellBehavedThreshold=4, errors_are_absolute=True)
    is_char = bool(res["WellBehavedTrap"]) and (res["EnergyFitFailed"] is False) and bool(res["GoodEnergyFit"])
    return is_char, bool(res["WellBehavedTrap"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=10.0)
    ap.add_argument("--catalog", type=Path, default=CAT)
    args = ap.parse_args()

    n_traps = 0
    char_base = char_alt = 0
    stored_char = 0
    base_matches_stored = base_total_checkable = 0
    gained = []   # newly characterized under alt
    lost = []     # characterized at baseline, not at alt
    recovered_points_on_already_char = 0
    recovered_points_on_newly_char = 0
    recovered_points_total = 0

    with h5py.File(args.catalog, "r") as f:
        for q in f:
            if not isinstance(f[q], h5py.Group): continue
            for dp in f[q]:
                g = f[q][dp]
                if not isinstance(g, h5py.Group): continue
                temps = [(int(tn.split("_")[1]), g[tn]) for tn in g
                         if tn.startswith("temp_") and isinstance(g[tn], h5py.Group)]
                if not temps: continue
                n_traps += 1
                base_data, alt_data = [], []
                n_recov_here = 0
                for T, tg in temps:
                    a = {k: tg.attrs[k] for k in tg.attrs}
                    tup = (T, ff(a.get("fit_tau")), ff(a.get("fit_tau_err")), ff(a.get("fit_coeff")))
                    gb = good(a, "baseline", None); ga = good(a, "alt", args.x)
                    if gb: base_data.append(tup)
                    if ga: alt_data.append(tup)
                    if ga and not gb and T >= 170:
                        n_recov_here += 1
                cb, _ = characterized(base_data)
                ca, _ = characterized(alt_data)
                char_base += int(cb); char_alt += int(ca)
                # validate baseline vs stored
                if "GoodEnergyFit" in g.attrs:
                    base_total_checkable += 1
                    stored = (bool(g.attrs.get("WellBehavedTrap", False))
                              and (g.attrs.get("EnergyFitFailed", True) is np.False_ or g.attrs.get("EnergyFitFailed", True) == False)
                              and bool(g.attrs.get("GoodEnergyFit", False)))
                    stored_char += int(stored)
                    if stored == cb: base_matches_stored += 1
                if ca and not cb: gained.append(f"{q}/{dp}")
                if cb and not ca: lost.append(f"{q}/{dp}")
                recovered_points_total += n_recov_here
                if n_recov_here:
                    if cb: recovered_points_on_already_char += n_recov_here
                    elif ca: recovered_points_on_newly_char += n_recov_here

    print(f"traps (dp groups) scanned: {n_traps}")
    print(f"baseline replication vs stored GoodEnergyFit: {base_matches_stored}/{base_total_checkable} match "
          f"({100*base_matches_stored/max(base_total_checkable,1):.1f}%)")
    print(f"stored characterized (catalog attr): {stored_char}")
    print(f"characterized  BASELINE (replicated): {char_base}")
    print(f"characterized  ALT (reduced_chi2<{args.x:g}): {char_alt}")
    print(f"  -> NET delta: {char_alt - char_base:+d}   gained: {len(gained)}   lost: {len(lost)}")
    print(f"recovered high-T intensity points total: {recovered_points_total}")
    print(f"  on already-characterized traps: {recovered_points_on_already_char}")
    print(f"  on newly-characterized traps:   {recovered_points_on_newly_char}")
    print(f"  (rest are on traps still not characterized even after recovery)")
    print("sample newly-characterized trap ids:", gained[:8])
    print("sample lost trap ids:", lost[:8])

if __name__ == "__main__":
    main()

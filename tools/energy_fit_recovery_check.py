#!/usr/bin/env python
"""Lean energy-fit recovery check (the 'does the SRH GOF erode the intensity reach' test).

For each catalog trap with >=4 good intensity fits and a valid energy fit, treat its
fitted (E, ln sigma) as a CLEAN SRH trap. Re-inject the SRH-true tau(T) at the trap's
real measurement temperatures and delays, with the WS1-consistent flavor noise + pedestal,
recover tau_hat(T)/sigma_tau(T) via the live _fit_one_curve, and (when >=4 survive) run the
live SRH energy fit. Report:
  * P(>=4 good intensity | these >=4-good traps)         -- conditional reach under fresh noise
  * P(SRH GoodEnergyFit | >=4)                            -- the energy-fit survival (the question)
  * P(characterized | >=4)                               -- full clean-trap completeness factor
  * sigma_tau pull RMS, stratified by temperature & amplitude (the calibration gate)

Pilot scale: samples traps and uses modest realizations. Reuses live code; injects nothing
that stage-08 wouldn't.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dipole
import dipole_new
from dipole_new import intensity_function_offset
from trap_completeness_method3.src.single_curve_recovery import _fit_one_curve, N_PUMPS
from trap_completeness_method3.src.full_pdet_grid_pilot import (
    _minimal_intensity_err,
    _load_pair_noise_lookup,
    _load_pedestal_lookup,
)
from trap_completeness_method3.src.analysis_flavors import (
    get_analysis_flavor,
    load_delta_chi2_thresholds,
)

WELL_BEHAVED_THRESHOLD = 4
TAU_TRUE_CAP = 1.0e4  # s; beyond the measurement window the curve is flat anyway


def srh_tau_true(flavor_name, temperatures_k, energy, log_sigma):
    mod = dipole_new if flavor_name != "legacy" else dipole
    log_tau = mod.log_energy_cross_section(np.asarray(temperatures_k, float), energy, log_sigma)
    return np.clip(np.exp(log_tau), 1e-8, TAU_TRUE_CAP)


def run_one_trap(dp_group, quad, flavor, rng, n_real,
                 pair_lookup, pedestal_lookup, delta_thresh, srh_fit):
    E = float(dp_group.attrs["energy_BestFitEnergy"])
    log_sigma = float(np.log(float(dp_group.attrs["energy_BestFitCrossSection"])))

    # The trap's real good-intensity temperatures + their measured curves.
    temps, seconds_by_T, coeff_by_T, errpatch_by_T, imgsig_by_T = [], {}, {}, {}, {}
    for name in dp_group:
        g = dp_group[name]
        if not isinstance(g, h5py.Group) or not name.startswith("temp_"):
            continue
        if not bool(g.attrs.get("GoodIntensityFit", False)):
            continue
        T = int(name.split("_")[1])
        coeff = float(g.attrs.get("fit_coeff", np.nan))
        if not np.isfinite(coeff) or coeff == 0.0:
            continue
        temps.append(T)
        seconds_by_T[T] = np.asarray(g["seconds"][()], float)
        coeff_by_T[T] = coeff
        errpatch_by_T[T] = np.asarray(g["intensity_err"][()], float)  # patch sigma (legacy noise)
        imgsig_by_T[T] = float(g.attrs.get("image_sigma", np.nan))
    temps = sorted(temps)
    if len(temps) < WELL_BEHAVED_THRESHOLD:
        return None

    tau_true_by_T = {T: float(t) for T, t in
                     zip(temps, srh_tau_true(flavor.name, temps, E, log_sigma))}

    per_real = {"k": [], "good_energy": [], "characterized": []}
    pulls = []  # (T, amplitude, pull)

    for _ in range(n_real):
        good_T, taus, tauerrs, signs = [], [], [], []
        for T in temps:
            seconds = seconds_by_T[T]
            tau_true = tau_true_by_T[T]
            coeff_true = coeff_by_T[T]  # measured amplitude (signed minimal, +ve legacy)
            offset_pool = pedestal_lookup.get(int(T))
            offset_true = float(rng.choice(offset_pool)) if offset_pool is not None and offset_pool.size else 0.0
            true_int = intensity_function_offset(seconds, coeff_true, tau_true, offset_true)

            if flavor.name == "legacy":
                err = errpatch_by_T[T]
            else:
                err = _minimal_intensity_err(true_int, int(T), int(quad), pair_lookup)
            noisy = true_int + rng.normal(0.0, err)
            if flavor.name == "legacy":
                noisy = np.abs(noisy)

            fit = _fit_one_curve(
                seconds, noisy, err, imgsig_by_T[T],
                analysis_flavor=flavor.name, temperature_k=int(T),
                delta_chi2_threshold_by_temperature=delta_thresh, _flavor=flavor,
            )
            if fit["good_intensity_fit"]:
                tau_hat = fit["fit_tau_seconds"]
                sig_tau = fit["fit_tau_err_seconds"]
                good_T.append(T)
                taus.append(tau_hat)
                tauerrs.append(sig_tau)
                signs.append(np.sign(fit["fit_coeff"]))
                if np.isfinite(tau_hat) and np.isfinite(sig_tau) and tau_hat > 0 and sig_tau > 0:
                    pull = (np.log(tau_hat) - np.log(tau_true)) / (sig_tau / tau_hat)
                    pulls.append((T, abs(coeff_true) * N_PUMPS, pull))

        k = len(good_T)
        per_real["k"].append(k)
        good_energy = False
        characterized = False
        if k >= WELL_BEHAVED_THRESHOLD:
            taus = np.asarray(taus, float)
            tauerrs = np.asarray(tauerrs, float)
            tk = np.asarray(good_T, float)
            if flavor.name == "legacy":
                res = srh_fit(tk, taus, tauerrs, WELL_BEHAVED_THRESHOLD)
            else:
                res = srh_fit(tk, taus, tauerrs, np.asarray(signs, float),
                              WELL_BEHAVED_THRESHOLD, True)
            good_energy = bool(res.get("GoodEnergyFit"))
            orient_ok = bool(res.get("OrientationConsistent", True))
            characterized = (bool(res.get("WellBehavedTrap")) and orient_ok
                             and not bool(res.get("EnergyFitFailed")) and good_energy)
        per_real["good_energy"].append(good_energy)
        per_real["characterized"].append(characterized)

    return per_real, pulls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flavor", choices=["legacy", "minimal_caldet"], default="minimal_caldet")
    p.add_argument("--catalog", default=None)
    p.add_argument("--n-traps", type=int, default=400)
    p.add_argument("--n-real", type=int, default=25)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    catalog = args.catalog or (
        "fit_dipole_spectra_minimal_caldet_err_4.h5" if args.flavor != "legacy"
        else "fit_dipole_spectra_err_4.h5"
    )
    flavor = get_analysis_flavor(args.flavor)
    srh_fit = (dipole_new.fit_energy_cross_section if flavor.name != "legacy"
               else dipole.fit_energy_cross_section)
    delta_thresh = load_delta_chi2_thresholds(flavor)
    pair_lookup = _load_pair_noise_lookup(ROOT / "pair_noise_table_minimal.npz")
    pedestal_lookup = _load_pedestal_lookup(ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5")
    rng = np.random.default_rng(args.seed)

    # Collect eligible traps: WellBehavedTrap & energy fit ran (valid E, ln sigma).
    eligible = []
    with h5py.File(catalog, "r") as f:
        for qname in f:
            qg = f[qname]
            if not isinstance(qg, h5py.Group):
                continue
            quad = int(qname.split("_")[1])
            for dpname in qg:
                dg = qg[dpname]
                if not isinstance(dg, h5py.Group):
                    continue
                if not bool(dg.attrs.get("WellBehavedTrap", False)):
                    continue
                if bool(dg.attrs.get("EnergyFitFailed", True)):
                    continue
                if "energy_BestFitEnergy" not in dg.attrs:
                    continue
                eligible.append((qname, dpname, quad))

        n_total = len(eligible)
        idx = rng.permutation(n_total)[: args.n_traps]
        sample = [eligible[i] for i in idx]

        n_real = args.n_real
        K, GE, CH = [], [], []
        all_pulls = []
        catalog_good_energy = 0
        for qname, dpname, quad in sample:
            dg = f[qname][dpname]
            catalog_good_energy += int(bool(dg.attrs.get("GoodEnergyFit", False)))
            out = run_one_trap(dg, quad, flavor, rng, n_real,
                               pair_lookup, pedestal_lookup, delta_thresh, srh_fit)
            if out is None:
                continue
            per_real, pulls = out
            K.extend(per_real["k"])
            GE.extend(per_real["good_energy"])
            CH.extend(per_real["characterized"])
            all_pulls.extend(pulls)

    K = np.asarray(K)
    GE = np.asarray(GE, bool)
    CH = np.asarray(CH, bool)
    ge4 = K >= WELL_BEHAVED_THRESHOLD
    n = K.size

    print(f"\n==== energy-fit recovery check | flavor={flavor.name} ====")
    print(f"catalog={catalog}")
    print(f"eligible traps (WellBehaved & energy fit ran): {n_total}; sampled: {len(sample)}; "
          f"realizations/trap: {n_real}; total realizations: {n}")
    print(f"catalog GoodEnergyFit fraction among sampled traps: {catalog_good_energy/len(sample):.3f}")
    print("-" * 60)
    print(f"P(>=4 good intensity | fresh noise)      : {ge4.mean():.3f}")
    if ge4.sum():
        print(f"P(SRH GoodEnergyFit | >=4)               : {GE[ge4].mean():.3f}   <-- energy-fit survival")
        print(f"P(characterized | >=4)                   : {CH[ge4].mean():.3f}")
        print(f"P(characterized) overall                 : {CH.mean():.3f}")

    if all_pulls:
        pa = np.array([p[2] for p in all_pulls], float)
        pa = pa[np.isfinite(pa)]
        print("-" * 60)
        print(f"sigma_tau PULL  (target RMS ~ 1.0)")
        print(f"  N={pa.size}  mean={pa.mean():+.3f}  RMS={np.sqrt(np.mean(pa**2)):.3f}  "
              f"std={pa.std():.3f}")
        # stratify by temperature
        Tarr = np.array([p[0] for p in all_pulls], float)[np.isfinite([p[2] for p in all_pulls])]
        print("  by temperature:")
        for lo, hi in [(120, 145), (145, 165), (165, 185), (185, 215)]:
            m = (Tarr >= lo) & (Tarr < hi)
            if m.sum():
                print(f"    {lo}-{hi}K : N={m.sum():5d}  RMS={np.sqrt(np.mean(pa[m]**2)):.3f}  "
                      f"mean={pa[m].mean():+.3f}")
        # stratify by amplitude
        Aarr = np.array([p[1] for p in all_pulls], float)[np.isfinite([p[2] for p in all_pulls])]
        qs = np.quantile(Aarr, [0, 0.25, 0.5, 0.75, 1.0])
        print("  by amplitude (electrons, quartiles):")
        for i in range(4):
            m = (Aarr >= qs[i]) & (Aarr <= qs[i + 1] if i == 3 else Aarr < qs[i + 1])
            if m.sum():
                print(f"    {qs[i]:7.0f}-{qs[i+1]:7.0f} : N={m.sum():5d}  "
                      f"RMS={np.sqrt(np.mean(pa[m]**2)):.3f}  mean={pa[m].mean():+.3f}")


if __name__ == "__main__":
    main()

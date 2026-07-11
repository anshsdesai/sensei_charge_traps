"""Split high-temperature intensity-fit failures by recoverability.

This is the leakage-free high-T failure split described in
HIGH_T_FAILURE_SPLIT_PLAN.md sections 3-6.  It refits failed minimal-catalog
curves, computes fixed-tau and Fisher counterfactuals, and reports whether each
failure is recoverable/analysis-limited or genuinely undetectable in the
sampled contrast.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.stats import chi2

import dipole
from dipole_new import INTENSITY_SHAPE_PEAK, INTENSITY_SHAPE_PEAK_X, intensity_function_offset
from trap_completeness_method3.src.validation_sensitivity import load_stage05


CATALOG = ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"
STAGE05 = ROOT / "trap_completeness_method3" / "cache" / "05_amplitude_prior_minimal_caldet_v1.npz"
GRID = ROOT / "trap_completeness_method3" / "cache" / "08_pdet_grid_minimal_caldet_v1.h5"
OUT_CSV = ROOT / "trap_completeness_method3" / "cache" / "high_t_failure_split_points.csv"
OUT_JSON = ROOT / "trap_completeness_method3" / "cache" / "high_t_failure_split_summary.json"
OUT_FIG = ROOT / "figures" / "high_t_failure_split.png"

AMP_T_RANGE = (140, 165)
HIGH_T_MIN = 170
DIP_TAU_WINDOW = (3e-3, 3e-2)
COND_MAX = 1e12
SIG_THRESHOLD = 3.0
P_VALUE_THRESHOLD = 0.05
TAU_MATCH_DEX_MAX = 0.3
N_PUMPS = 3000.0


@dataclass
class CurveRecord:
    seconds: np.ndarray
    intensities: np.ndarray
    intensity_err: np.ndarray
    attrs: dict[str, Any]


@dataclass
class TrapRecord:
    q: int
    trap_name: str
    trap_id: str
    row: int | None
    col: int | None
    energy_ev: float
    log_sigma: float
    curves: dict[int, CurveRecord]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    return bool(value)


def shape_template(seconds: np.ndarray, tau: float) -> np.ndarray:
    seconds = np.asarray(seconds, dtype=float)
    return N_PUMPS * (np.exp(-seconds / tau) - np.exp(-8.0 * seconds / tau))


def jacobian_offset_model(seconds: np.ndarray, coeff: float, tau: float) -> np.ndarray:
    seconds = np.asarray(seconds, dtype=float)
    exp1 = np.exp(-seconds / tau)
    exp8 = np.exp(-8.0 * seconds / tau)
    d_coeff = N_PUMPS * (exp1 - exp8)
    d_tau = N_PUMPS * coeff * (
        (seconds / tau**2) * exp1 - 8.0 * (seconds / tau**2) * exp8
    )
    d_offset = np.ones_like(seconds, dtype=float)
    return np.column_stack([d_coeff, d_tau, d_offset])


def verify_tau_jacobian(seconds: np.ndarray, coeff: float, tau: float, offset: float) -> None:
    step = max(abs(tau) * 1e-6, 1e-8)
    if tau - step <= 0:
        step = tau * 0.25
    plus = intensity_function_offset(seconds, coeff, tau + step, offset)
    minus = intensity_function_offset(seconds, coeff, tau - step, offset)
    numeric = (plus - minus) / (2.0 * step)
    analytic = jacobian_offset_model(seconds, coeff, tau)[:, 1]
    scale = np.maximum(1.0, np.maximum(np.abs(numeric), np.abs(analytic)))
    max_rel = float(np.max(np.abs(numeric - analytic) / scale))
    if not max_rel < 1e-5:
        raise AssertionError(f"analytic d/dtau Jacobian failed finite-difference check: {max_rel:.3g}")


def fresh_refit(seconds: np.ndarray, y: np.ndarray, err: np.ndarray) -> dict[str, Any]:
    offset_estimate = float(np.median(y))
    deviations = y - offset_estimate
    peak_index = int(np.argmax(np.abs(deviations)))
    tau_estimate = float(np.clip(seconds[peak_index] / INTENSITY_SHAPE_PEAK_X, 1e-8, 1000.0))
    coeff_estimate = float(deviations[peak_index] / (N_PUMPS * INTENSITY_SHAPE_PEAK))
    popt, pcov = curve_fit(
        intensity_function_offset,
        seconds,
        y,
        sigma=err,
        p0=[coeff_estimate, tau_estimate, offset_estimate],
        bounds=([-np.inf, 1e-8, -np.inf], [np.inf, 1000.0, np.inf]),
        absolute_sigma=True,
        maxfev=20000,
    )
    model = intensity_function_offset(seconds, *popt)
    residuals = y - model
    chi_squared = float(np.sum(np.square(residuals / err)))
    dof = int(len(y) - len(popt))
    reduced_chi_squared = float(chi_squared / dof) if dof > 0 else math.nan
    p_value = float(1.0 - chi2.cdf(chi_squared, dof)) if dof > 0 else math.nan
    perr = np.sqrt(np.diag(pcov))
    weights = 1.0 / np.square(err)
    const_best = float(np.sum(y * weights) / np.sum(weights))
    chi2_const = float(np.sum(np.square((y - const_best) / err)))
    delta_chi2 = float(chi2_const - chi_squared)
    amplitude_significance = float(abs(popt[0]) / perr[0]) if perr[0] > 0 else 0.0
    return {
        "popt": popt,
        "pcov": pcov,
        "perr": perr,
        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "p_value": p_value,
        "delta_chi2_vs_constant": delta_chi2,
        "amplitude_significance": amplitude_significance,
    }


def fisher_counterfactuals(
    seconds: np.ndarray,
    err: np.ndarray,
    coeff: float,
    tau: float,
) -> dict[str, float | bool | str]:
    try:
        j = jacobian_offset_model(seconds, coeff, tau)
        jw = j / np.square(err)[:, None]
        fisher = j.T @ jw
        fisher2 = fisher[np.ix_([0, 1], [0, 1])]
        cond_f = float(np.linalg.cond(fisher))
        cond_f2 = float(np.linalg.cond(fisher2))
        if not np.isfinite(cond_f) or not np.isfinite(cond_f2) or cond_f > COND_MAX or cond_f2 > COND_MAX:
            return {"singular": True, "singular_reason": "ill_conditioned", "cond_fisher": cond_f, "cond_fisher2": cond_f2}
        cov = np.linalg.inv(fisher)
        cov2 = np.linalg.inv(fisher2)
        coeff_var_full = float(cov[0, 0])
        coeff_var_fixoff = float(cov2[0, 0])
        coeff_info_fixboth = float(fisher[0, 0])
        if coeff_var_full <= 0 or coeff_var_fixoff <= 0 or coeff_info_fixboth <= 0:
            return {
                "singular": True,
                "singular_reason": "nonpositive_variance",
                "cond_fisher": cond_f,
                "cond_fisher2": cond_f2,
            }
        sig_full = float(abs(coeff) / math.sqrt(coeff_var_full))
        sig_fixoff = float(abs(coeff) / math.sqrt(coeff_var_fixoff))
        sig_fixboth = float(abs(coeff) * math.sqrt(coeff_info_fixboth))
        tol1 = max(1e-9, 1e-9 * max(sig_full, sig_fixoff, sig_fixboth, 1.0))
        if sig_fixoff + tol1 < sig_full:
            raise AssertionError(f"sig_fixoff < sig_full: {sig_fixoff} < {sig_full}")
        if sig_fixboth + tol1 < sig_fixoff:
            raise AssertionError(f"sig_fixboth < sig_fixoff: {sig_fixboth} < {sig_fixoff}")
        return {
            "singular": False,
            "singular_reason": "",
            "cond_fisher": cond_f,
            "cond_fisher2": cond_f2,
            "sig_full": sig_full,
            "sig_fixoff": sig_fixoff,
            "sig_fixboth": sig_fixboth,
        }
    except AssertionError:
        raise
    except Exception as exc:
        return {
            "singular": True,
            "singular_reason": type(exc).__name__,
            "cond_fisher": math.nan,
            "cond_fisher2": math.nan,
        }


def forced_amplitude_fit(seconds: np.ndarray, y: np.ndarray, err: np.ndarray, tau_srh: float) -> dict[str, float]:
    g = shape_template(seconds, tau_srh)
    x = np.column_stack([g, np.ones_like(g)])
    w = 1.0 / np.square(err)
    xtw = x.T * w
    fisher = xtw @ x
    cov = np.linalg.inv(fisher)
    beta = cov @ (xtw @ y)
    return {
        "coeff_forced": float(beta[0]),
        "offset_forced": float(beta[1]),
        "coeff_forced_err": float(math.sqrt(cov[0, 0])),
        "forced_cond": float(np.linalg.cond(fisher)),
    }


def sampled_information(seconds: np.ndarray, err: np.ndarray, tau_srh: float) -> tuple[float, float, float]:
    g = shape_template(seconds, tau_srh)
    w = 1.0 / np.square(err)
    g_perp = g - float(np.sum(w * g) / np.sum(w))
    info = float(np.sum(w * np.square(g_perp)))
    return info, float(np.min(g_perp)), float(np.max(g_perp))


def contrast_stat(seconds: np.ndarray, y: np.ndarray, err: np.ndarray, tau_srh: float, sign: float) -> dict[str, float]:
    g = shape_template(seconds, tau_srh)
    w = 1.0 / np.square(err)
    g_perp = g - float(np.sum(w * g) / np.sum(w))
    abs_g = np.abs(g_perp)
    n = abs_g.size
    order = np.argsort(abs_g)
    k = max(1, n // 3)
    late_idx = order[:k]
    early_idx = order[-k:]

    def wmean_and_var(indices: np.ndarray) -> tuple[float, float]:
        ww = w[indices]
        yy = y[indices]
        weight_sum = float(np.sum(ww))
        mean = float(np.sum(ww * yy) / weight_sum)
        var = float(1.0 / weight_sum)
        return mean, var

    early, var_early = wmean_and_var(early_idx)
    late, var_late = wmean_and_var(late_idx)
    contrast = float(early - late)
    sigma = float(math.sqrt(var_early + var_late))
    return {
        "contrast": contrast,
        "contrast_err": sigma,
        "sig_C": float(sign * contrast / sigma) if sigma > 0 else math.nan,
    }


def parse_dp_position(name: str) -> tuple[int | None, int | None]:
    try:
        _, row, col = name.split("_", 2)
        return int(row), int(col)
    except Exception:
        return None, None


def load_traps(path: Path) -> list[TrapRecord]:
    traps: list[TrapRecord] = []
    with h5py.File(path, "r") as h5:
        for q_name in h5:
            q_group = h5[q_name]
            if not isinstance(q_group, h5py.Group) or not q_name.startswith("quad_"):
                continue
            q = int(q_name.split("_")[1])
            for dp_name in q_group:
                group = q_group[dp_name]
                if not isinstance(group, h5py.Group) or "energy_BestFitEnergy" not in group.attrs:
                    continue
                cross_section = as_float(group.attrs.get("energy_BestFitCrossSection"))
                if not (cross_section > 0.0):
                    continue
                curves: dict[int, CurveRecord] = {}
                for temp_name in group:
                    temp_group = group[temp_name]
                    if not isinstance(temp_group, h5py.Group) or not temp_name.startswith("temp_"):
                        continue
                    if not {"seconds", "intensities", "intensity_err"}.issubset(temp_group.keys()):
                        continue
                    temp = int(temp_name.split("_")[1])
                    curves[temp] = CurveRecord(
                        seconds=np.asarray(temp_group["seconds"][()], dtype=float),
                        intensities=np.asarray(temp_group["intensities"][()], dtype=float),
                        intensity_err=np.asarray(temp_group["intensity_err"][()], dtype=float),
                        attrs={key: temp_group.attrs[key] for key in temp_group.attrs},
                    )
                row, col = parse_dp_position(dp_name)
                traps.append(
                    TrapRecord(
                        q=q,
                        trap_name=dp_name,
                        trap_id=f"{q_name}/{dp_name}",
                        row=row,
                        col=col,
                        energy_ev=as_float(group.attrs["energy_BestFitEnergy"]),
                        log_sigma=float(np.log(cross_section)),
                        curves=curves,
                    )
                )
    return traps


def good_cold_coeff(trap: TrapRecord, target_t: int) -> tuple[float, float, int, str]:
    vals: list[tuple[int, float]] = []
    for temp, curve in trap.curves.items():
        if temp == target_t:
            continue
        if not (AMP_T_RANGE[0] <= temp <= AMP_T_RANGE[1]):
            continue
        if not as_bool(curve.attrs.get("GoodIntensityFit", False)):
            continue
        coeff = as_float(curve.attrs.get("fit_coeff"))
        if np.isfinite(coeff):
            vals.append((temp, coeff))
    if not vals:
        return math.nan, math.nan, 0, ""
    temps = np.asarray([v[0] for v in vals], dtype=float)
    coeffs = np.asarray([v[1] for v in vals], dtype=float)
    return float(np.median(coeffs)), float(np.median(temps)), int(coeffs.size), ";".join(str(int(t)) for t in temps)


def load_pc_interpolator(path: Path):
    stage05 = load_stage05(path)
    temps = np.asarray(stage05["temperatures"], dtype=float)
    pc = np.asarray(stage05["pc"], dtype=float)
    order = np.argsort(temps)
    temps = temps[order]
    pc = pc[order]

    def pc_at(temp: float) -> float:
        return float(np.interp(float(temp), temps, pc))

    return pc_at, {"temperatures": temps.tolist(), "pc": pc.tolist()}


def load_grid_interpolators(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        temps = [int(round(x)) for x in h5["grid/temperature_K"][:]]
        tau = h5["grid/tau_seconds"][:].astype(float)
        amp = h5["grid/amplitude_electrons"][:].astype(float)
        pdet = h5["results/p_det"][:].astype(float)
    log_tau = np.log(tau)
    interpolators = {
        temp: RegularGridInterpolator((log_tau, amp), pdet[i], bounds_error=False, fill_value=None)
        for i, temp in enumerate(temps)
    }
    return {"temps": temps, "tau": tau, "amp": amp, "interpolators": interpolators}


def grid_pdet(grid: dict[str, Any] | None, temp: int, tau_srh: float, amp_electrons: float) -> float:
    if grid is None or temp not in grid["interpolators"]:
        return math.nan
    tau_grid = grid["tau"]
    amp_grid = grid["amp"]
    if not (tau_grid[0] <= tau_srh <= tau_grid[-1]):
        return 0.0
    amp_eval = float(np.clip(abs(amp_electrons), amp_grid[0], amp_grid[-1]))
    val = float(grid["interpolators"][temp]((math.log(tau_srh), amp_eval)))
    return float(np.clip(val, 0.0, 1.0)) if np.isfinite(val) else math.nan


def safe_ratio_delta(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return math.nan
    return float(abs(a - b) / abs(b))


def classify_row(row: dict[str, Any]) -> None:
    tags: list[str] = []
    if row["singular_fisher"]:
        row["primary_class"] = "unclassified"
        row["leverage_class"] = "unclassified"
        row["leverage_class_rawcold"] = "unclassified"
        tags.append("unclassified")
        if row.get("singular_reason"):
            tags.append(str(row["singular_reason"]))
        row["tags"] = ";".join(tags)
        return

    shape_fail = (
        row["sig_full"] >= SIG_THRESHOLD
        and not (row["fresh_p_value"] > P_VALUE_THRESHOLD)
        and row["fresh_delta_chi2_vs_constant"] >= row["delta_chi2_threshold"]
        and row["tau_match_dex"] <= TAU_MATCH_DEX_MAX
    )
    pedestal_cost = row["sig_full"] < SIG_THRESHOLD and row["sig_fixoff"] >= SIG_THRESHOLD
    tau_cost = row["sig_fixoff"] < SIG_THRESHOLD and row["sig_fixboth"] >= SIG_THRESHOLD

    if shape_fail:
        tags.append("shape_fail")
    if pedestal_cost:
        tags.append("pedestal_cost")
    if tau_cost:
        tags.append("tau_cost")

    recoverable = (
        row["sig_forced"] >= SIG_THRESHOLD
        or row["sig_C"] >= SIG_THRESHOLD
        or pedestal_cost
        or tau_cost
        or shape_fail
    )

    if recoverable:
        row["primary_class"] = "recoverable_or_analysis_limited"
        row["leverage_class"] = "recoverable_or_analysis_limited"
        row["leverage_class_rawcold"] = "recoverable_or_analysis_limited"
    else:
        row["primary_class"] = "undetectable_sampled_contrast"
        if row["snr_exp_pc"] >= SIG_THRESHOLD:
            row["leverage_class"] = "genuine_fading"
        else:
            row["leverage_class"] = "design_compression_limited"
            tags.append("compression")
            if row["snr_exp_rawcold"] < SIG_THRESHOLD:
                tags.append("low_contrast")
        if row["snr_exp_rawcold"] >= SIG_THRESHOLD:
            row["leverage_class_rawcold"] = "genuine_fading"
        else:
            row["leverage_class_rawcold"] = "design_compression_limited"

    row["tags"] = ";".join(dict.fromkeys(tags))


def build_row(
    trap: TrapRecord,
    temp: int,
    curve: CurveRecord,
    pc_at,
    grid: dict[str, Any] | None,
    jacobian_check_state: dict[str, bool],
) -> dict[str, Any]:
    seconds = curve.seconds
    y = curve.intensities
    err = curve.intensity_err
    attrs = curve.attrs
    base: dict[str, Any] = {
        "trap_id": trap.trap_id,
        "q": trap.q,
        "trap_name": trap.trap_name,
        "row": trap.row if trap.row is not None else "",
        "col": trap.col if trap.col is not None else "",
        "T": temp,
        "E_eV": trap.energy_ev,
        "log_sigma": trap.log_sigma,
        "GoodIntensityFit": as_bool(attrs.get("GoodIntensityFit", False)),
        "stored_fit_coeff": as_float(attrs.get("fit_coeff")),
        "stored_fit_tau": as_float(attrs.get("fit_tau")),
        "stored_fit_offset": as_float(attrs.get("fit_offset")),
        "stored_fit_coeff_err": as_float(attrs.get("fit_coeff_err")),
        "stored_fit_tau_err": as_float(attrs.get("fit_tau_err")),
        "stored_fit_offset_err": as_float(attrs.get("fit_offset_err")),
        "stored_amplitude_significance": as_float(attrs.get("amplitude_significance")),
        "stored_delta_chi2_vs_constant": as_float(attrs.get("delta_chi2_vs_constant")),
        "delta_chi2_threshold": as_float(attrs.get("delta_chi2_threshold")),
        "stored_fit_p_value": as_float(attrs.get("fit_p_value")),
        "stored_fit_reduced_chi_squared": as_float(attrs.get("fit_reduced_chi_squared")),
    }

    cold_coeff, t_cold_ref, n_cold, cold_temps = good_cold_coeff(trap, temp)
    sign = float(np.sign(cold_coeff)) if np.isfinite(cold_coeff) and cold_coeff != 0 else 1.0
    tau_srh = float(np.exp(dipole.log_energy_cross_section(float(temp), trap.energy_ev, trap.log_sigma)))
    pc_t = pc_at(temp)
    pc_ref = pc_at(t_cold_ref) if np.isfinite(t_cold_ref) else math.nan
    pc_ratio = pc_t / pc_ref if np.isfinite(pc_ref) and pc_ref != 0 else math.nan
    a_exp_pc = cold_coeff * pc_ratio if np.isfinite(cold_coeff) and np.isfinite(pc_ratio) else math.nan
    a_exp_rawcold = cold_coeff

    base.update(
        {
            "tau_srh": tau_srh,
            "tau_in_dip_window": bool(DIP_TAU_WINDOW[0] <= tau_srh <= DIP_TAU_WINDOW[1]),
            "cold_coeff": cold_coeff,
            "cold_coeff_electrons": cold_coeff * N_PUMPS if np.isfinite(cold_coeff) else math.nan,
            "cold_t_ref": t_cold_ref,
            "cold_good_count": n_cold,
            "cold_good_temperatures": cold_temps,
            "pc_T": pc_t,
            "pc_ref": pc_ref,
            "pc_ratio": pc_ratio,
            "A_exp_pc_coeff": a_exp_pc,
            "A_exp_pc_electrons": a_exp_pc * N_PUMPS if np.isfinite(a_exp_pc) else math.nan,
            "A_exp_rawcold_coeff": a_exp_rawcold,
            "A_exp_rawcold_electrons": a_exp_rawcold * N_PUMPS if np.isfinite(a_exp_rawcold) else math.nan,
            "orientation_sign": sign,
            "primary_class": "unclassified",
            "leverage_class": "unclassified",
            "leverage_class_rawcold": "unclassified",
            "tags": "unclassified",
            "singular_fisher": True,
            "singular_reason": "",
            "bright_cohort": False,
        }
    )

    try:
        if seconds.size != y.size or y.size != err.size or seconds.size < 4:
            raise ValueError("invalid_curve_lengths")
        if not np.all(np.isfinite(seconds)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(err)):
            raise ValueError("nonfinite_curve")
        if not np.all(err > 0):
            raise ValueError("nonpositive_errors")
        if not (np.isfinite(cold_coeff) and np.isfinite(t_cold_ref) and n_cold > 0):
            raise ValueError("missing_cold_reference")
        if not (np.isfinite(tau_srh) and tau_srh > 0):
            raise ValueError("invalid_tau_srh")

        fit = fresh_refit(seconds, y, err)
        coeff, tau_fit, offset = [float(v) for v in fit["popt"]]
        perr = fit["perr"]
        if not jacobian_check_state["done"] and np.isfinite(coeff) and coeff != 0 and np.isfinite(tau_fit) and tau_fit > 0:
            verify_tau_jacobian(seconds, coeff, tau_fit, offset)
            jacobian_check_state["done"] = True

        fisher = fisher_counterfactuals(seconds, err, coeff, tau_fit)
        forced = forced_amplitude_fit(seconds, y, err, tau_srh)
        info, g_perp_min, g_perp_max = sampled_information(seconds, err, tau_srh)
        contrast = contrast_stat(seconds, y, err, tau_srh, sign)

        coeff_forced = forced["coeff_forced"]
        coeff_forced_err = forced["coeff_forced_err"]
        a_proj = sign * coeff_forced
        sig_forced = a_proj / coeff_forced_err if coeff_forced_err > 0 else math.nan
        ul95 = a_proj + 1.645 * coeff_forced_err if coeff_forced_err > 0 else math.nan
        snr_exp_pc = abs(a_exp_pc) * math.sqrt(info) if np.isfinite(a_exp_pc) and info >= 0 else math.nan
        snr_exp_rawcold = abs(a_exp_rawcold) * math.sqrt(info) if np.isfinite(a_exp_rawcold) and info >= 0 else math.nan
        tau_match_dex = abs(math.log10(tau_fit / tau_srh)) if tau_fit > 0 and tau_srh > 0 else math.inf
        pdet = grid_pdet(grid, temp, tau_srh, abs(cold_coeff) * N_PUMPS)

        base.update(
            {
                "fresh_fit_coeff": coeff,
                "fresh_fit_tau": tau_fit,
                "fresh_fit_offset": offset,
                "fresh_fit_coeff_err": float(perr[0]),
                "fresh_fit_tau_err": float(perr[1]),
                "fresh_fit_offset_err": float(perr[2]),
                "fresh_amplitude_significance": fit["amplitude_significance"],
                "fresh_delta_chi2_vs_constant": fit["delta_chi2_vs_constant"],
                "fresh_chi_squared": fit["chi_squared"],
                "fresh_reduced_chi_squared": fit["reduced_chi_squared"],
                "fresh_p_value": fit["p_value"],
                "tau_match_dex": tau_match_dex,
                "coeff_err_rel_diff": safe_ratio_delta(float(perr[0]), base["stored_fit_coeff_err"]),
                "amplitude_sig_rel_diff": safe_ratio_delta(
                    fit["amplitude_significance"], base["stored_amplitude_significance"]
                ),
                "sig_full": fisher.get("sig_full", math.nan),
                "sig_fixoff": fisher.get("sig_fixoff", math.nan),
                "sig_fixboth": fisher.get("sig_fixboth", math.nan),
                "cond_fisher": fisher.get("cond_fisher", math.nan),
                "cond_fisher2": fisher.get("cond_fisher2", math.nan),
                "singular_fisher": bool(fisher["singular"]),
                "singular_reason": fisher.get("singular_reason", ""),
                "coeff_forced": coeff_forced,
                "offset_forced": forced["offset_forced"],
                "coeff_forced_err": coeff_forced_err,
                "forced_cond": forced["forced_cond"],
                "A_proj_forced": a_proj,
                "sig_forced": sig_forced,
                "UL95_forced_coeff": ul95,
                "UL95_forced_electrons": ul95 * N_PUMPS if np.isfinite(ul95) else math.nan,
                "sampled_info": info,
                "g_perp_min": g_perp_min,
                "g_perp_max": g_perp_max,
                "snr_exp_pc": snr_exp_pc,
                "snr_exp_rawcold": snr_exp_rawcold,
                "contrast": contrast["contrast"],
                "contrast_err": contrast["contrast_err"],
                "sig_C": contrast["sig_C"],
                "grid_pdet_rawcold": pdet,
                "bright_cohort": bool(np.isfinite(pdet) and pdet > 0.9),
            }
        )
        classify_row(base)
        return base
    except Exception as exc:
        base.update(
            {
                "singular_fisher": True,
                "singular_reason": type(exc).__name__ if str(exc) == "" else str(exc),
                "primary_class": "unclassified",
                "leverage_class": "unclassified",
                "leverage_class_rawcold": "unclassified",
                "tags": f"unclassified;{type(exc).__name__}",
            }
        )
        return base


def fraction_dict(count: int, denominator: int) -> dict[str, float | int]:
    frac = float(count / denominator) if denominator else math.nan
    err = float(math.sqrt(frac * (1.0 - frac) / denominator)) if denominator and np.isfinite(frac) else math.nan
    return {"count": int(count), "denominator": int(denominator), "fraction": frac, "binomial_error": err}


def composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classified = [r for r in rows if r["primary_class"] != "unclassified"]
    primary = Counter(r["primary_class"] for r in rows)
    leverage = Counter(r["leverage_class"] for r in rows)
    tags = Counter()
    for row in rows:
        for tag in str(row.get("tags", "")).split(";"):
            if tag:
                tags[tag] += 1
    denom = len(classified)
    genuine_pc = sum(1 for r in classified if r["leverage_class"] == "genuine_fading")
    genuine_raw = sum(1 for r in classified if r["leverage_class_rawcold"] == "genuine_fading")
    return {
        "n_total": int(len(rows)),
        "n_classified": int(denom),
        "n_unclassified": int(primary.get("unclassified", 0)),
        "primary_counts": dict(primary),
        "primary_fractions_classified_denominator": {
            key: float(val / denom) if denom else math.nan for key, val in primary.items() if key != "unclassified"
        },
        "leverage_counts": dict(leverage),
        "tag_counts": dict(tags),
        "genuine_fading_fraction": fraction_dict(genuine_pc, denom),
        "genuine_fading_fraction_rawcold": fraction_dict(genuine_raw, denom),
    }


def assert_self_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AssertionError("no failed high-T points found")
    total = len(rows)
    bucket_total = sum(Counter(r["primary_class"] for r in rows).values())
    if bucket_total != total:
        raise AssertionError(f"bucket accounting failed: {bucket_total} != {total}")

    coeff_diffs = np.asarray([r.get("coeff_err_rel_diff", math.nan) for r in rows], dtype=float)
    amp_diffs = np.asarray([r.get("amplitude_sig_rel_diff", math.nan) for r in rows], dtype=float)
    coeff_diffs = coeff_diffs[np.isfinite(coeff_diffs)]
    amp_diffs = amp_diffs[np.isfinite(amp_diffs)]
    coeff_bulk = float(np.mean(coeff_diffs <= 0.05)) if coeff_diffs.size else 0.0
    amp_bulk = float(np.mean(amp_diffs <= 0.05)) if amp_diffs.size else 0.0
    coeff_median = float(np.median(coeff_diffs)) if coeff_diffs.size else math.nan
    amp_median = float(np.median(amp_diffs)) if amp_diffs.size else math.nan
    if coeff_diffs.size == 0 or coeff_median > 0.05 or coeff_bulk < 0.90:
        raise AssertionError(
            f"fresh fit coeff_err consistency failed: median={coeff_median:.4g}, "
            f"frac_within_5pct={coeff_bulk:.3f}"
        )
    if amp_diffs.size == 0 or amp_median > 0.05 or amp_bulk < 0.90:
        raise AssertionError(
            f"fresh amplitude significance consistency failed: median={amp_median:.4g}, "
            f"frac_within_5pct={amp_bulk:.3f}"
        )
    if not any(r.get("singular_fisher") is False for r in rows):
        raise AssertionError("no nonsingular Fisher rows found")
    return {
        "bucket_sum": int(bucket_total),
        "failed_high_t_points": int(total),
        "coeff_err_rel_diff_median": coeff_median,
        "coeff_err_rel_diff_fraction_within_5pct": coeff_bulk,
        "amplitude_sig_rel_diff_median": amp_median,
        "amplitude_sig_rel_diff_fraction_within_5pct": amp_bulk,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temps = sorted({int(r["T"]) for r in rows})
    categories = [
        ("recoverable_or_analysis_limited", "Recoverable / analysis-limited", "#4c78a8"),
        ("genuine_fading", "Genuine fading", "#e45756"),
        ("design_compression_limited", "Design / low contrast", "#f2cf5b"),
        ("unclassified", "Unclassified", "#9d9da1"),
    ]
    values = {key: [] for key, _, _ in categories}
    totals = []
    for temp in temps:
        subset = [r for r in rows if int(r["T"]) == temp]
        totals.append(len(subset))
        for key, _, _ in categories:
            if key == "recoverable_or_analysis_limited":
                count = sum(1 for r in subset if r["primary_class"] == key)
            elif key == "unclassified":
                count = sum(1 for r in subset if r["primary_class"] == key)
            else:
                count = sum(1 for r in subset if r["leverage_class"] == key)
            values[key].append(count / len(subset) if subset else 0.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottom = np.zeros(len(temps))
    x = np.arange(len(temps))
    for key, label, color in categories:
        arr = np.asarray(values[key], dtype=float)
        ax.bar(x, arr, bottom=bottom, label=label, color=color, width=0.78)
        bottom += arr
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temps], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of failed high-T points")
    ax.set_xlabel("Temperature [K]")
    ax.set_title("High-T intensity-fit failure split", pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    for xi, total in zip(x, totals):
        ax.text(xi, 0.985, str(total), ha="center", va="top", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--stage05", type=Path, default=STAGE05)
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-fig", type=Path, default=OUT_FIG)
    args = parser.parse_args()

    pc_at, pc_payload = load_pc_interpolator(args.stage05)
    grid = load_grid_interpolators(args.grid)
    traps = load_traps(args.catalog)
    rows: list[dict[str, Any]] = []
    jacobian_check_state = {"done": False}
    for trap in traps:
        for temp, curve in sorted(trap.curves.items()):
            if temp < HIGH_T_MIN:
                continue
            if as_bool(curve.attrs.get("GoodIntensityFit", False)):
                continue
            rows.append(build_row(trap, temp, curve, pc_at, grid, jacobian_check_state))

    if not jacobian_check_state["done"]:
        raise AssertionError("Jacobian finite-difference check did not run on any sample curve")

    checks = assert_self_consistency(rows)
    write_csv(args.out_csv, rows)
    write_figure(args.out_fig, rows)

    dip_rows = [r for r in rows if r["tau_in_dip_window"]]
    bright_rows = [r for r in rows if r["bright_cohort"]]
    summary = {
        "created_at": now_iso(),
        "inputs": {
            "catalog": str(args.catalog),
            "stage05": str(args.stage05),
            "grid": str(args.grid),
            "pc_loader_keys": ["pc", "temperatures"],
            "pc_payload": pc_payload,
        },
        "settings": {
            "high_t_min_K": HIGH_T_MIN,
            "dip_tau_window_seconds": list(DIP_TAU_WINDOW),
            "amp_t_range_K": list(AMP_T_RANGE),
            "condition_number_max": COND_MAX,
            "significance_threshold": SIG_THRESHOLD,
            "p_value_threshold": P_VALUE_THRESHOLD,
            "tau_match_dex_max": TAU_MATCH_DEX_MAX,
            "a_exp_headline": "cold_coeff * PC(T) / PC(T_cold_ref)",
            "a_exp_rawcold_context": "cold_coeff",
        },
        "n_characterized_traps": int(len(traps)),
        "overall_high_t": composition(rows),
        "dip_window_weighted": composition(dip_rows),
        "bright_cohort_grid_pdet_gt_0p9": composition(bright_rows),
        "per_temperature": {str(temp): composition([r for r in rows if int(r["T"]) == temp]) for temp in sorted({int(r["T"]) for r in rows})},
        "self_consistency": checks,
        "outputs": {"csv": str(args.out_csv), "summary_json": str(args.out_json), "figure": str(args.out_fig)},
    }
    summary["genuine_fading_fraction"] = summary["dip_window_weighted"]["genuine_fading_fraction"]
    summary["genuine_fading_fraction_rawcold"] = summary["dip_window_weighted"]["genuine_fading_fraction_rawcold"]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    dip_comp = summary["dip_window_weighted"]
    print("Dip-window-weighted composition (tau_SRH in 3e-3..3e-2 s):")
    print(json.dumps(dip_comp["primary_counts"], indent=2, sort_keys=True))
    print("Leverage split:")
    print(json.dumps(dip_comp["leverage_counts"], indent=2, sort_keys=True))
    print("genuine_fading_fraction (PC-scaled):", json.dumps(summary["genuine_fading_fraction"], sort_keys=True))
    print(
        "genuine_fading_fraction_rawcold:",
        json.dumps(summary["genuine_fading_fraction_rawcold"], sort_keys=True),
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_fig}")


if __name__ == "__main__":
    main()

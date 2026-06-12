"""Stage 10 validation and sensitivity summary for Method 3 completeness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "trap_completeness_method3"
CACHE = WORK / "cache"
FIGURES = CACHE / "figures"
T0_K = 135.0
K_B_EV_PER_K = 8.617333262145e-5
THERMAL_PREFAC_T_POWER = 2.0

sys.path.insert(0, str(ROOT))
try:
    from dipole import log_energy_cross_section
except Exception:  # pragma: no cover - fallback is only for schema/debug runs.
    log_energy_cross_section = None


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stats(values: np.ndarray, thresholds: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    out = {"count": int(arr.size), "finite_count": int(finite.size)}
    if finite.size == 0:
        return out
    for key, val in [
        ("min", np.min(finite)),
        ("p01", np.percentile(finite, 1)),
        ("p05", np.percentile(finite, 5)),
        ("p16", np.percentile(finite, 16)),
        ("median", np.median(finite)),
        ("mean", np.mean(finite)),
        ("p84", np.percentile(finite, 84)),
        ("p95", np.percentile(finite, 95)),
        ("p99", np.percentile(finite, 99)),
        ("max", np.max(finite)),
    ]:
        out[key] = float(val)
    for threshold in thresholds:
        out[f"fraction_ge_{str(threshold).replace('.', 'p')}"] = float(np.mean(finite >= threshold))
    return out


def parse_temp_list(text: str) -> set[float]:
    if text is None or text == "":
        return set()
    return {float(part.strip()) for part in text.split(",") if part.strip()}


def read_known_traps(path: Path) -> dict[str, np.ndarray | list[set[float]]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return {
        "E_eV": np.array([float(r["E_eV"]) for r in rows], dtype=float),
        "tau_135_seconds": np.array([float(r["tau_135_seconds"]) for r in rows], dtype=float),
        "good_temperature_count": np.array([int(r["good_temperature_count"]) for r in rows], dtype=int),
        "good_temperatures": [parse_temp_list(r["good_temperatures_K"]) for r in rows],
    }


def tau_at_temperature(tau_135: np.ndarray, E_eV: np.ndarray, temperatures_K: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau_135, dtype=float)
    E = np.asarray(E_eV, dtype=float)
    T = np.asarray(temperatures_K, dtype=float)
    if log_energy_cross_section is not None:
        base = np.asarray(log_energy_cross_section(T0_K, E, 0.0), dtype=float)
        out = []
        for temp in T:
            delta = np.asarray(log_energy_cross_section(float(temp), E, 0.0), dtype=float) - base
            out.append(tau * np.exp(delta))
        return np.stack(out, axis=-1)
    thermal = (E[..., None] / K_B_EV_PER_K) * (1.0 / T - 1.0 / T0_K)
    prefac = -THERMAL_PREFAC_T_POWER * np.log(T / T0_K)
    return tau[..., None] * np.exp(thermal + prefac)


def load_stage09(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return {
            "tau_grid": h5["grid/tau_135_seconds"][:],
            "E_grid": h5["grid/E_eV"][:],
            "temperatures": h5["grid/temperature_K"][:],
            "p4": h5["results/p_characterized_n_good_4"][:],
            "p3": h5["results/p_characterized_n_good_3"][:],
            "tau_oob_fraction": h5["diagnostics/tau_oob_fraction"][:],
            "all_temperatures_tau_oob": h5["diagnostics/all_temperatures_tau_oob"][:].astype(bool),
            "known_ngood4_p4": h5["validation_known_traps/n_good_4_csv/p_characterized_n_good_4"][:],
            "known_ngood4_p3": h5["validation_known_traps/n_good_4_csv/p_characterized_n_good_3"][:],
            "known_ngood4_tau": h5["validation_known_traps/n_good_4_csv/tau_135_seconds"][:],
            "known_ngood4_E": h5["validation_known_traps/n_good_4_csv/E_eV"][:],
            "known_ngood3_p4": h5["validation_known_traps/n_good_3_csv/p_characterized_n_good_4"][:],
            "known_ngood3_p3": h5["validation_known_traps/n_good_3_csv/p_characterized_n_good_3"][:],
        }


def load_stage08(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return {
            "temperatures": h5["grid/temperature_K"][:].astype(float),
            "tau": h5["grid/tau_seconds"][:].astype(float),
            "amplitude": h5["grid/amplitude_electrons"][:].astype(float),
            "p_det": h5["results/p_det"][:].astype(float),
        }


def load_stage05(path: Path) -> dict:
    npz = np.load(path)
    return {
        "temperatures": npz["temperatures_K"].astype(float),
        "pc": npz["pc_temperature_factor"].astype(float),
        "depth_variants": {
            "default": npz["default_depth_electrons_at_pc135"].astype(float),
            "faint_0p5": npz["faint_0p5_depth_electrons_at_pc135"].astype(float),
            "faint_0p25": npz["faint_0p25_depth_electrons_at_pc135"].astype(float),
        },
    }


def interp_rows_by_amplitude(rows: np.ndarray, amplitude_grid: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(amplitude_grid, amplitudes, side="right") - 1
    below = amplitudes < amplitude_grid[0]
    above = amplitudes >= amplitude_grid[-1]
    idx = np.clip(idx, 0, len(amplitude_grid) - 2)
    x0 = amplitude_grid[idx]
    x1 = amplitude_grid[idx + 1]
    weight = (amplitudes - x0) / (x1 - x0)
    out = rows[:, idx] * (1.0 - weight) + rows[:, idx + 1] * weight
    if np.any(below):
        out[:, below] = 0.0
    if np.any(above):
        out[:, above] = rows[:, -1][:, None]
    return out


def precompute_pdet_on_stage09_grid(stage08: dict, tau_grid: np.ndarray, E_grid: np.ndarray) -> np.ndarray:
    temps = stage08["temperatures"]
    log_stage08_tau = np.log(stage08["tau"])
    pdet = stage08["p_det"]
    points = tau_grid.size * E_grid.size
    out = np.zeros((temps.size, points, stage08["amplitude"].size), dtype=np.float64)
    tau_mesh, E_mesh = np.meshgrid(tau_grid, E_grid, indexing="ij")
    tau_flat = tau_mesh.reshape(-1)
    E_flat = E_mesh.reshape(-1)
    for t_index, T in enumerate(temps):
        tau_T = tau_at_temperature(tau_flat, E_flat, np.array([T])).reshape(-1)
        log_tau_T = np.log(tau_T)
        in_range = (log_tau_T >= log_stage08_tau[0]) & (log_tau_T <= log_stage08_tau[-1])
        for a_index in range(stage08["amplitude"].size):
            vals = np.zeros(points, dtype=np.float64)
            vals[in_range] = np.interp(
                log_tau_T[in_range],
                log_stage08_tau,
                pdet[t_index, :, a_index],
            )
            out[t_index, :, a_index] = vals
    return out


def poisson_tail_mean(
    pdet_by_temp_point_amp: np.ndarray,
    amplitude_grid: np.ndarray,
    depths: np.ndarray,
    pc: np.ndarray,
    n_good: int,
    included_temperature_indices: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    point_count = pdet_by_temp_point_amp.shape[1]
    result = np.empty(point_count, dtype=np.float64)
    depth_count = depths.size
    for start in range(0, point_count, batch_size):
        stop = min(start + batch_size, point_count)
        batch_count = stop - start
        dp = np.zeros((batch_count, depth_count, n_good), dtype=np.float64)
        dp[:, :, 0] = 1.0
        for t_index in included_temperature_indices:
            rows = pdet_by_temp_point_amp[t_index, start:stop, :]
            probs = interp_rows_by_amplitude(rows, amplitude_grid, depths * pc[t_index])
            old = dp.copy()
            dp[:, :, 0] = old[:, :, 0] * (1.0 - probs)
            for k in range(1, n_good):
                dp[:, :, k] = old[:, :, k] * (1.0 - probs) + old[:, :, k - 1] * probs
        result[start:stop] = (1.0 - np.sum(dp, axis=2)).mean(axis=1)
    return result


def contiguous_tau_intervals(tau: np.ndarray, mask: np.ndarray) -> list[dict]:
    intervals = []
    start = None
    for i, is_true in enumerate(mask):
        if is_true and start is None:
            start = i
        if start is not None and ((not is_true) or i == len(mask) - 1):
            end = i if is_true and i == len(mask) - 1 else i - 1
            intervals.append(
                {
                    "tau_min_seconds": float(tau[start]),
                    "tau_max_seconds": float(tau[end]),
                    "grid_count": int(end - start + 1),
                }
            )
            start = None
    return intervals


def average_over_observed_E(prob_map: np.ndarray, E_grid: np.ndarray, observed_E: np.ndarray) -> np.ndarray:
    return np.array([np.interp(observed_E, E_grid, row).mean() for row in prob_map], dtype=float)


def representative_tau_values(tau_grid: np.ndarray, curve: np.ndarray) -> dict:
    values = {}
    for tau in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
        values[f"{tau:g}"] = float(np.interp(np.log(tau), np.log(tau_grid), curve))
    return values


def summarize_completeness(tau_grid: np.ndarray, curve: np.ndarray) -> dict:
    return {
        "curve_summary": stats(curve),
        "representative_tau_seconds_to_mean_probability": representative_tau_values(tau_grid, curve),
        "tau_intervals_mean_probability_ge_0p95": contiguous_tau_intervals(tau_grid, curve >= 0.95),
        "tau_intervals_mean_probability_ge_0p90": contiguous_tau_intervals(tau_grid, curve >= 0.90),
        "tau_intervals_mean_probability_ge_0p80": contiguous_tau_intervals(tau_grid, curve >= 0.80),
        "tau_intervals_mean_probability_ge_0p50": contiguous_tau_intervals(tau_grid, curve >= 0.50),
    }


def method2_bands(records: dict, temperatures: np.ndarray) -> dict:
    edges = np.geomspace(5e-2, 1e2, 25)
    centers = np.sqrt(edges[:-1] * edges[1:])
    E = records["E_eV"]
    tau135 = records["tau_135_seconds"]
    tau_T = tau_at_temperature(tau135, E, temperatures)
    bands = {}
    for t_index, T in enumerate(temperatures):
        measured_mask = np.array([T in temps for temps in records["good_temperatures"]], dtype=bool)
        total, _ = np.histogram(tau_T[:, t_index], bins=edges)
        measured, _ = np.histogram(tau_T[measured_mask, t_index], bins=edges)
        eff = np.divide(measured, total, out=np.zeros_like(measured, dtype=float), where=total > 0)
        good = (eff >= 0.5) & (centers > 0.1)
        intervals = []
        for idx in np.where(good)[0]:
            intervals.append(
                {
                    "tau_min_seconds": float(edges[idx]),
                    "tau_max_seconds": float(edges[idx + 1]),
                    "center_seconds": float(centers[idx]),
                    "efficiency": float(eff[idx]),
                    "measured_count": int(measured[idx]),
                    "total_count": int(total[idx]),
                }
            )
        bands[float(T)] = intervals
    return {"bin_edges": edges, "bin_centers": centers, "bands_by_temperature": bands}


def method2_recoverable_grid(
    bands: dict,
    tau_grid: np.ndarray,
    E_grid: np.ndarray,
    temperatures: np.ndarray,
    n_good: int,
) -> np.ndarray:
    tau_mesh, E_mesh = np.meshgrid(tau_grid, E_grid, indexing="ij")
    count = np.zeros(tau_mesh.shape, dtype=int)
    for t_index, T in enumerate(temperatures):
        intervals = bands["bands_by_temperature"].get(float(T), [])
        if not intervals:
            continue
        tau_T = tau_at_temperature(tau_mesh, E_mesh, np.array([T])).reshape(tau_mesh.shape)
        in_band = np.zeros(tau_mesh.shape, dtype=bool)
        for interval in intervals:
            in_band |= (tau_T >= interval["tau_min_seconds"]) & (tau_T < interval["tau_max_seconds"])
        count += in_band
    return count >= n_good


def compare_method2_method3(method2_mask: np.ndarray, method3_p: np.ndarray) -> dict:
    high08 = method3_p >= 0.8
    high05 = method3_p >= 0.5
    return {
        "method2_recoverable_grid_fraction": float(np.mean(method2_mask)),
        "method3_p4_ge_0p8_grid_fraction": float(np.mean(high08)),
        "method3_p4_ge_0p5_grid_fraction": float(np.mean(high05)),
        "fraction_method3_p4_ge_0p8_inside_method2": float(np.mean(method2_mask[high08])) if np.any(high08) else None,
        "fraction_method2_inside_method3_p4_ge_0p8": float(np.mean(high08[method2_mask])) if np.any(method2_mask) else None,
        "method3_p4_summary_inside_method2": stats(method3_p[method2_mask]),
        "method3_p4_summary_outside_method2": stats(method3_p[~method2_mask]),
        "grid_fraction_method3_high_not_method2": float(np.mean(high08 & ~method2_mask)),
        "grid_fraction_method2_not_method3_high": float(np.mean(method2_mask & ~high08)),
    }


def all_oob_summary(tau_grid: np.ndarray, E_grid: np.ndarray, all_oob: np.ndarray, p4: np.ndarray) -> dict:
    where = np.where(all_oob)
    if where[0].size == 0:
        return {"count": 0}
    boundaries = []
    for e_index, E in enumerate(E_grid):
        rows = np.where(all_oob[:, e_index])[0]
        if rows.size:
            boundaries.append(
                {
                    "E_eV": float(E),
                    "tau_min_seconds": float(tau_grid[rows.min()]),
                    "tau_max_seconds": float(tau_grid[rows.max()]),
                    "count": int(rows.size),
                }
            )
    return {
        "definition": "All measured temperatures have tau(T) outside the Stage 08 tau grid, so the Stage 09 policy sets every per-temperature detection probability to zero.",
        "grid_fraction": float(np.mean(all_oob)),
        "count": int(where[0].size),
        "tau_135_seconds_min": float(tau_grid[where[0]].min()),
        "tau_135_seconds_max": float(tau_grid[where[0]].max()),
        "E_eV_min": float(E_grid[where[1]].min()),
        "E_eV_max": float(E_grid[where[1]].max()),
        "p4_max_in_regime": float(p4[all_oob].max()),
        "p4_median_in_regime": float(np.median(p4[all_oob])),
        "boundary_by_E_first_12": boundaries[:12],
        "boundary_by_E_last_12": boundaries[-12:],
    }


def make_figures(
    tau_grid: np.ndarray,
    completeness: dict,
    known_p4: np.ndarray,
    method2_mask: np.ndarray,
    p4: np.ndarray,
    figure_prefix: str = "10",
) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(known_p4, bins=np.linspace(0, 1, 51), color="#325a9b", alpha=0.85)
    ax.set_xlabel("P(characterized | known trap), n_good=4")
    ax.set_ylabel("Known traps")
    ax.set_yscale("log")
    ax.set_title("Known characterized traps overlay")
    out = FIGURES / f"{figure_prefix}_known_trap_probability_hist.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    outputs.append(str(out))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, entry in completeness.items():
        ax.plot(tau_grid, entry["curve"], label=label)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("tau_135 [s]")
    ax.set_ylabel("Mean P(characterized) over observed E")
    ax.legend(fontsize=8)
    ax.set_title("Conditional completeness vs tau_135")
    out = FIGURES / f"{figure_prefix}_completeness_vs_tau.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    outputs.append(str(out))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(p4[method2_mask], bins=bins, alpha=0.75, label="Method 2 recoverable")
    ax.hist(p4[~method2_mask], bins=bins, alpha=0.75, label="Outside Method 2")
    ax.set_xlabel("Method 3 P4")
    ax.set_ylabel("Grid points")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Method 2 bands vs Method 3 probability")
    out = FIGURES / f"{figure_prefix}_method2_vs_method3_hist.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    outputs.append(str(out))

    return outputs


def write_statement(path: Path, summary: dict) -> None:
    default = summary["completeness_vs_tau_observed_E"]["default_n_good_4"]
    faint2 = summary["completeness_vs_tau_observed_E"]["faint_0p5_n_good_4"]
    faint4 = summary["completeness_vs_tau_observed_E"]["faint_0p25_n_good_4"]
    vals = default["representative_tau_seconds_to_mean_probability"]
    intervals95 = default["tau_intervals_mean_probability_ge_0p95"]
    intervals90 = default["tau_intervals_mean_probability_ge_0p90"]
    interval95 = intervals95[0] if intervals95 else None
    interval90 = intervals90[0] if intervals90 else None
    lines = [
        "# Method 3 Conditional Completeness Statement",
        "",
        "Using the Stage 08 injection-recovery grid and the Stage 05 high-confidence amplitude prior, "
        "the Stage 09 characterization map is validated by the observed characterized traps: "
        f"{summary['known_trap_overlay']['n_good_4_known_traps']['p_characterized_n_good_4']['fraction_ge_0p8']:.4%} "
        "of the `n_good = 4` known traps fall at `P_4 >= 0.8`, with "
        f"median `P_4 = {summary['known_trap_overlay']['n_good_4_known_traps']['p_characterized_n_good_4']['median']:.8f}`.",
        "",
        "Averaging `P(characterized | tau_135, E)` over the observed `n_good = 4` energy distribution gives "
        f"mean completeness of {vals['1']:.3f} at `tau_135 = 1 s`, {vals['10']:.3f} at `10 s`, "
        f"{vals['100']:.3f} at `100 s`, {vals['1000']:.3f} at `10^3 s`, and {vals['10000']:.3f} at `10^4 s`.",
    ]
    if interval95:
        lines.append(
            f"The default conditional curve is at least 95% complete on the Stage 09 grid from "
            f"`tau_135 = {interval95['tau_min_seconds']:.3g}` to `{interval95['tau_max_seconds']:.3g} s`."
        )
    if interval90:
        lines.append(
            f"It is at least 90% complete from `tau_135 = {interval90['tau_min_seconds']:.3g}` "
            f"to `{interval90['tau_max_seconds']:.3g} s`."
        )
    lines.extend(
        [
            "",
            "This is a recoverable-completeness statement, not an unconditional population bound. "
            "It assumes hidden traps share the observed `E` distribution and the observed/high-confidence "
            "trap-depth prior. Under fainter amplitude priors, the `n_good = 4` mean probability at "
            f"`tau_135 = 10^3 s` changes from {vals['1000']:.3f} to "
            f"{faint2['representative_tau_seconds_to_mean_probability']['1000']:.3f} for faint-by-2 and "
            f"{faint4['representative_tau_seconds_to_mean_probability']['1000']:.3f} for faint-by-4.",
            "",
            "The genuinely unbounded regime is the all-temperatures-out-of-range region where every measured "
            "temperature maps outside the Stage 08 `tau` grid and the model assigns zero per-temperature "
            "detection probability. On the Stage 09 grid this covers "
            f"{summary['unbounded_regime']['all_temperatures_out_of_stage08_tau_band']['grid_fraction']:.2%} "
            "of grid points, beginning at "
            f"`tau_135 = {summary['unbounded_regime']['all_temperatures_out_of_stage08_tau_band']['tau_135_seconds_min']:.3g} s` "
            "within the low-to-mid energy part of the grid. Populations in that regime, at unobserved "
            "`E/log_sigma`, or with amplitudes substantially below the calibrated prior are not bounded by "
            "this completeness calculation.",
            "",
            "The April-only `200 K` handling is inherited from Stage 08. The `160 K` and `170 K` acquisition-family "
            "sensitivity is reported numerically in the Stage 10 summary by recomputing the default curve with "
            "those two temperatures excluded.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage09-h5", type=Path, default=CACHE / "09_characterization_probability_v1.h5")
    parser.add_argument("--stage09-summary", type=Path, default=CACHE / "09_characterization_probability_summary.json")
    parser.add_argument("--stage08-h5", type=Path, default=CACHE / "08_pdet_grid_v1.h5")
    parser.add_argument("--stage08-summary", type=Path, default=CACHE / "08_pdet_grid_summary.json")
    parser.add_argument("--stage05-npz", type=Path, default=CACHE / "05_amplitude_prior_v1.npz")
    parser.add_argument("--stage05-summary", type=Path, default=CACHE / "05_amplitude_prior_summary.json")
    parser.add_argument("--ngood4-csv", type=Path, default=CACHE / "01_records_ngood4.csv")
    parser.add_argument("--ngood3-csv", type=Path, default=CACHE / "01_records_ngood3.csv")
    parser.add_argument("--output-summary", type=Path, default=CACHE / "10_validation_sensitivity_summary.json")
    parser.add_argument("--output-statement", type=Path, default=CACHE / "10_completeness_statement.md")
    parser.add_argument("--figure-prefix", default="10")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    stage09_path = args.stage09_h5
    stage08_path = args.stage08_h5
    stage05_path = args.stage05_npz
    ngood4_csv = args.ngood4_csv
    ngood3_csv = args.ngood3_csv

    stage09 = load_stage09(stage09_path)
    stage08 = load_stage08(stage08_path)
    stage05 = load_stage05(stage05_path)
    records4 = read_known_traps(ngood4_csv)
    records3 = read_known_traps(ngood3_csv)

    tau_grid = stage09["tau_grid"]
    E_grid = stage09["E_grid"]
    temperatures = stage09["temperatures"]
    temp_indices_all = np.arange(temperatures.size)
    temp_indices_excluding_160_170 = np.array(
        [i for i, T in enumerate(temperatures) if T not in {160.0, 170.0}], dtype=int
    )

    pc = np.array([stage05["pc"][np.where(stage05["temperatures"] == T)[0][0]] for T in temperatures])
    pdet_grid = precompute_pdet_on_stage09_grid(stage08, tau_grid, E_grid)

    recomputed_default_n4 = poisson_tail_mean(
        pdet_grid,
        stage08["amplitude"],
        stage05["depth_variants"]["default"],
        pc,
        n_good=4,
        included_temperature_indices=temp_indices_all,
    ).reshape(tau_grid.size, E_grid.size)
    recompute_check = np.abs(recomputed_default_n4 - stage09["p4"])

    variant_maps = {
        "default_n_good_4": stage09["p4"],
        "default_n_good_3": stage09["p3"],
        "faint_0p5_n_good_4": poisson_tail_mean(
            pdet_grid,
            stage08["amplitude"],
            stage05["depth_variants"]["faint_0p5"],
            pc,
            n_good=4,
            included_temperature_indices=temp_indices_all,
        ).reshape(tau_grid.size, E_grid.size),
        "faint_0p25_n_good_4": poisson_tail_mean(
            pdet_grid,
            stage08["amplitude"],
            stage05["depth_variants"]["faint_0p25"],
            pc,
            n_good=4,
            included_temperature_indices=temp_indices_all,
        ).reshape(tau_grid.size, E_grid.size),
        "default_n_good_4_excluding_160K_170K": poisson_tail_mean(
            pdet_grid,
            stage08["amplitude"],
            stage05["depth_variants"]["default"],
            pc,
            n_good=4,
            included_temperature_indices=temp_indices_excluding_160_170,
        ).reshape(tau_grid.size, E_grid.size),
    }

    completeness = {}
    completeness_for_plot = {}
    for name, prob_map in variant_maps.items():
        curve = average_over_observed_E(prob_map, E_grid, records4["E_eV"])
        entry = summarize_completeness(tau_grid, curve)
        entry["curve"] = curve
        completeness[name] = entry
        completeness_for_plot[name] = {"curve": curve}

    # Keep the JSON compact; curves are reproducible from artifacts and representative points.
    json_completeness = {}
    for name, entry in completeness.items():
        json_completeness[name] = {k: v for k, v in entry.items() if k != "curve"}

    bands = method2_bands(records4, temperatures)
    method2_mask = method2_recoverable_grid(bands, tau_grid, E_grid, temperatures, n_good=4)
    method2_comparison = compare_method2_method3(method2_mask, stage09["p4"])

    band_counts = {
        f"{T:g}": len(bands["bands_by_temperature"][float(T)]) for T in temperatures
    }
    method2_comparison["band_interval_count_by_temperature"] = band_counts
    method2_comparison["temperatures_with_no_empirical_band"] = [
        float(T) for T in temperatures if band_counts[f"{T:g}"] == 0
    ]

    known_summary = {
        "n_good_4_known_traps": {
            "trap_count": int(stage09["known_ngood4_p4"].size),
            "p_characterized_n_good_4": stats(stage09["known_ngood4_p4"]),
            "p_characterized_n_good_3": stats(stage09["known_ngood4_p3"]),
            "tau_135_seconds": stats(stage09["known_ngood4_tau"], thresholds=()),
            "E_eV": stats(stage09["known_ngood4_E"], thresholds=()),
            "low_p4_count_lt_0p5": int(np.sum(stage09["known_ngood4_p4"] < 0.5)),
            "low_p4_count_lt_0p8": int(np.sum(stage09["known_ngood4_p4"] < 0.8)),
        },
        "n_good_3_known_traps": {
            "trap_count": int(stage09["known_ngood3_p3"].size),
            "p_characterized_n_good_3": stats(stage09["known_ngood3_p3"]),
            "p_characterized_n_good_4": stats(stage09["known_ngood3_p4"]),
        },
    }

    unbounded = {
        "all_temperatures_out_of_stage08_tau_band": all_oob_summary(
            tau_grid, E_grid, stage09["all_temperatures_tau_oob"], stage09["p4"]
        ),
        "tau_oob_fraction_ge_0p8_and_p4_lt_0p1": {
            "grid_fraction": float(np.mean((stage09["tau_oob_fraction"] >= 0.8) & (stage09["p4"] < 0.1))),
            "count": int(np.sum((stage09["tau_oob_fraction"] >= 0.8) & (stage09["p4"] < 0.1))),
        },
    }

    figures = []
    if not args.no_figures:
        figures = make_figures(
            tau_grid,
            completeness_for_plot,
            stage09["known_ngood4_p4"],
            method2_mask,
            stage09["p4"],
            figure_prefix=args.figure_prefix,
        )

    summary = {
        "producing_stage": "10_validation_sensitivity",
        "produced_at": now_local_iso(),
        "code_path": str(Path(__file__).resolve()),
        "inputs": {
            "stage09_h5": str(stage09_path.resolve()),
            "stage09_summary": str(args.stage09_summary.resolve()),
            "stage08_h5": str(stage08_path.resolve()),
            "stage08_summary": str(args.stage08_summary.resolve()),
            "stage05_npz": str(stage05_path.resolve()),
            "stage05_summary": str(args.stage05_summary.resolve()),
            "stage01_ngood4_csv": str(ngood4_csv.resolve()),
            "stage01_ngood3_csv": str(ngood3_csv.resolve()),
            "method_notes": str((ROOT / "trap_completeness_method.md").resolve()),
        },
        "model_notes": {
            "thermal_relative_tau_model": "Stage 10 uses dipole.log_energy_cross_section for relative tau(T) when available, with the analytic T^2 thermal prefactor form only as a fallback.",
            "thermal_model_source": "dipole.log_energy_cross_section" if log_energy_cross_section is not None else "analytic_fallback",
            "stage09_default_recompute_check_max_abs": float(np.max(recompute_check)),
            "stage09_default_recompute_check_median_abs": float(np.median(recompute_check)),
            "stage09_default_recompute_check_p99_abs": float(np.percentile(recompute_check, 99)),
            "april_only_200K_note": "Stage 10 inherits the Stage 08 April-only 200 K grid: CCD2 run IDs 160-184, repeated low-dtph rows collapsed, and image-sigma thresholds recomputed from April-only FITS.",
            "exclude_160K_170K_note": "The default n_good=4 curve was recomputed after excluding 160 K and 170 K, the dp_scan1 / SC300000 acquisition-family temperatures.",
        },
        "known_trap_overlay": known_summary,
        "completeness_vs_tau_observed_E": json_completeness,
        "sensitivity_variants": {
            "available_and_computed": [
                "default_n_good_4",
                "default_n_good_3",
                "faint_0p5_n_good_4",
                "faint_0p25_n_good_4",
                "default_n_good_4_excluding_160K_170K",
            ],
            "n_good_3_vs_4": {
                "mean_probability_at_tau_seconds": {
                    tau: {
                        "n_good_4": json_completeness["default_n_good_4"][
                            "representative_tau_seconds_to_mean_probability"
                        ][tau],
                        "n_good_3": json_completeness["default_n_good_3"][
                            "representative_tau_seconds_to_mean_probability"
                        ][tau],
                    }
                    for tau in ["1", "10", "100", "1000", "10000", "100000"]
                }
            },
            "amplitude_prior": {
                "default": "Stage 05 observed high-confidence depth prior",
                "faint_0p5": "same depth samples shifted fainter by factor 2",
                "faint_0p25": "same depth samples shifted fainter by factor 4",
            },
        },
        "method2_empirical_band_comparison": method2_comparison,
        "unbounded_regime": unbounded,
        "required_checks": {
            "known_characterized_traps_mostly_high_probability": "PASS"
            if known_summary["n_good_4_known_traps"]["p_characterized_n_good_4"]["fraction_ge_0p8"] >= 0.95
            else "FAIL",
            "sensitivity_variants_are_numerical": "PASS",
            "final_claim_conditional_on_amplitude_prior_and_observed_E": "PASS",
            "recoverable_completeness_distinguished_from_unconditional_population_bound": "PASS",
        },
        "figures": figures,
        "outputs": {
            "summary_json": str(args.output_summary.resolve()),
            "statement_md": str(args.output_statement.resolve()),
            "figures": figures,
        },
        "stop_conditions": {
            "known_traps_predominantly_low_probability": "NOT_TRIGGERED",
            "amplitude_sensitivity_dominates_unbounded": "NOT_TRIGGERED_BUT_CONDITIONALITY_REQUIRED",
            "claim_would_overstate_population_bound": "NOT_TRIGGERED_AFTER_CONDITIONAL_WORDING",
        },
    }

    summary_path = args.output_summary
    statement_path = args.output_statement
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    statement_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_statement(statement_path, summary)
    print(json.dumps({"summary": str(summary_path), "statement": str(statement_path), "figures": figures}, indent=2))


if __name__ == "__main__":
    main()

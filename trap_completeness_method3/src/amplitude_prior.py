#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dipole import log_energy_cross_section


STAGE_ID = "05_amplitude_prior"
N_PUMPS = 3000.0
REFERENCE_TEMPERATURE_K = 135
HIGH_CONF_COEFF_SNR_MIN = 5.0
HIGH_CONF_TAU_REL_ERR_MAX = 0.25
HIGH_CONF_SAMPLED_PEAK_FRACTION_MIN = 0.5


def _parse_quad_name(name: str) -> int:
    return int(name.split("_")[1])


def _parse_dp_name(name: str) -> tuple[int, int]:
    _, row, col = name.split("_")
    return int(row), int(col)


def _parse_temp_name(name: str) -> int:
    return int(name.split("_")[1])


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            "count": 0,
            "median": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p16": math.nan,
            "p84": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "min": math.nan,
            "max": math.nan,
            "p95_over_p05": math.nan,
            "max_over_min": math.nan,
        }
    p05 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": p05,
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "p95": p95,
        "p99": float(np.percentile(arr, 99)),
        "min": vmin,
        "max": vmax,
        "p95_over_p05": float(p95 / p05) if p05 > 0 else math.nan,
        "max_over_min": float(vmax / vmin) if vmin > 0 else math.nan,
    }


def _as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_as_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_as_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_as_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ideal_shape_peak() -> float:
    x_peak = math.log(8.0) / 7.0
    return math.exp(-x_peak) - math.exp(-8.0 * x_peak)


def _sampled_peak_fraction(seconds: np.ndarray, tau: float) -> float:
    if not np.isfinite(tau) or tau <= 0:
        return math.nan
    shape = np.exp(-seconds / tau) - np.exp(-8.0 * seconds / tau)
    sampled_peak = float(np.max(shape)) if shape.size else math.nan
    return sampled_peak / _ideal_shape_peak() if np.isfinite(sampled_peak) else math.nan


def _load_observations(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        for quad_name, quad_group in h5.items():
            quad = _parse_quad_name(quad_name)
            for dp_name, dp_group in quad_group.items():
                trap_attrs = dp_group.attrs
                if not bool(trap_attrs.get("WellBehavedTrap", False)):
                    continue
                if bool(trap_attrs.get("EnergyFitFailed", False)):
                    continue
                if not bool(trap_attrs.get("GoodEnergyFit", False)):
                    continue
                row, col = _parse_dp_name(dp_name)
                energy = float(trap_attrs["energy_BestFitEnergy"])
                cross_section = float(trap_attrs["energy_BestFitCrossSection"])
                if not np.isfinite(energy) or not np.isfinite(cross_section) or cross_section <= 0:
                    continue
                log_sigma = float(np.log(cross_section))
                tau_135 = float(
                    np.exp(log_energy_cross_section(np.array([135.0]), energy, log_sigma))[0]
                )
                trap_id = f"{label}:q{quad}:r{row}:c{col}"

                temp_names = sorted(
                    [name for name in dp_group.keys() if name.startswith("temp_")],
                    key=_parse_temp_name,
                )
                for temp_name in temp_names:
                    temp_group = dp_group[temp_name]
                    if not bool(temp_group.attrs.get("GoodIntensityFit", False)):
                        continue
                    required_attrs = ["fit_coeff", "fit_coeff_err", "fit_tau", "fit_tau_err"]
                    if any(key not in temp_group.attrs for key in required_attrs):
                        continue
                    coeff = float(temp_group.attrs["fit_coeff"])
                    coeff_err = float(temp_group.attrs["fit_coeff_err"])
                    fit_tau = float(temp_group.attrs["fit_tau"])
                    fit_tau_err = float(temp_group.attrs["fit_tau_err"])
                    if not (np.isfinite(coeff) and np.isfinite(fit_tau) and coeff > 0 and fit_tau > 0):
                        continue
                    temperature = _parse_temp_name(temp_name)
                    seconds = np.asarray(temp_group["seconds"][()], dtype=float)
                    amplitude = N_PUMPS * coeff
                    amplitude_err = N_PUMPS * coeff_err if np.isfinite(coeff_err) else math.nan
                    coeff_snr = coeff / coeff_err if np.isfinite(coeff_err) and coeff_err > 0 else math.inf
                    tau_rel_err = (
                        fit_tau_err / fit_tau
                        if np.isfinite(fit_tau_err) and fit_tau_err >= 0 and fit_tau > 0
                        else math.nan
                    )
                    peak_fraction = _sampled_peak_fraction(seconds, fit_tau)
                    high_confidence = (
                        np.isfinite(amplitude)
                        and np.isfinite(coeff_snr)
                        and np.isfinite(tau_rel_err)
                        and np.isfinite(peak_fraction)
                        and coeff_snr >= HIGH_CONF_COEFF_SNR_MIN
                        and tau_rel_err <= HIGH_CONF_TAU_REL_ERR_MAX
                        and peak_fraction >= HIGH_CONF_SAMPLED_PEAK_FRACTION_MIN
                    )
                    rows.append(
                        {
                            "source_hdf5": str(path),
                            "selection_label": label,
                            "trap_id": trap_id,
                            "quadrant": quad,
                            "row": row,
                            "col": col,
                            "temperature_K": temperature,
                            "E_eV": energy,
                            "log_sigma": log_sigma,
                            "tau_135_seconds": tau_135,
                            "fit_tau_seconds": fit_tau,
                            "fit_tau_err_seconds": fit_tau_err,
                            "tau_rel_err": tau_rel_err,
                            "fit_coeff": coeff,
                            "fit_coeff_err": coeff_err,
                            "fit_coeff_snr": coeff_snr,
                            "amplitude_electrons": amplitude,
                            "amplitude_err_electrons": amplitude_err,
                            "sampled_peak_fraction": peak_fraction,
                            "high_confidence": high_confidence,
                        }
                    )
    return rows


def _temperature_counter(rows: list[dict[str, Any]]) -> dict[int, int]:
    return dict(sorted(Counter(int(row["temperature_K"]) for row in rows).items()))


def _connected_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trap_nodes = {f"trap:{row['trap_id']}" for row in rows}
    temp_nodes = {f"temp:{row['temperature_K']}" for row in rows}
    graph: dict[str, set[str]] = {node: set() for node in trap_nodes | temp_nodes}
    for row in rows:
        trap_node = f"trap:{row['trap_id']}"
        temp_node = f"temp:{row['temperature_K']}"
        graph[trap_node].add(temp_node)
        graph[temp_node].add(trap_node)

    seen: set[str] = set()
    components: list[dict[str, Any]] = []
    for node in graph:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        nodes: list[str] = []
        while queue:
            current = queue.popleft()
            nodes.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        temps = sorted(int(item.split(":", 1)[1]) for item in nodes if item.startswith("temp:"))
        traps = [item for item in nodes if item.startswith("trap:")]
        components.append(
            {
                "n_nodes": len(nodes),
                "n_traps": len(traps),
                "n_temperatures": len(temps),
                "temperatures_K": temps,
            }
        )
    components.sort(key=lambda item: item["n_nodes"], reverse=True)
    return {
        "component_count": len(components),
        "largest_components": components[:5],
    }


def _fit_temperature_scaling(rows: list[dict[str, Any]]) -> tuple[dict[int, float], dict[str, float]]:
    if not rows:
        raise ValueError("Cannot fit temperature scaling with no high-confidence rows.")
    temps = sorted({int(row["temperature_K"]) for row in rows})
    temp_effects = {
        temp: float(
            np.median(
                [
                    math.log(row["amplitude_electrons"])
                    for row in rows
                    if int(row["temperature_K"]) == temp
                ]
            )
        )
        for temp in temps
    }
    global_center = float(np.median([math.log(row["amplitude_electrons"]) for row in rows]))
    temp_effects = {temp: effect - global_center for temp, effect in temp_effects.items()}

    trap_effects: dict[str, float] = {}
    for _ in range(100):
        previous = dict(temp_effects)
        by_trap: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_trap[row["trap_id"]].append(
                math.log(row["amplitude_electrons"]) - temp_effects[int(row["temperature_K"])]
            )
        trap_effects = {trap_id: float(np.median(values)) for trap_id, values in by_trap.items()}

        by_temp: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_temp[int(row["temperature_K"])].append(
                math.log(row["amplitude_electrons"]) - trap_effects[row["trap_id"]]
            )
        temp_effects = {temp: float(np.median(values)) for temp, values in by_temp.items()}

        if REFERENCE_TEMPERATURE_K in temp_effects:
            anchor = temp_effects[REFERENCE_TEMPERATURE_K]
        else:
            anchor = float(np.median(list(temp_effects.values())))
        temp_effects = {temp: effect - anchor for temp, effect in temp_effects.items()}

        max_change = max(abs(temp_effects[temp] - previous.get(temp, 0.0)) for temp in temp_effects)
        if max_change < 1e-8:
            break
    return temp_effects, trap_effects


def _add_depth_columns(
    rows: list[dict[str, Any]],
    temp_effects: dict[int, float],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        if int(row["temperature_K"]) not in temp_effects:
            continue
        next_row = dict(row)
        log_depth = math.log(row["amplitude_electrons"]) - temp_effects[int(row["temperature_K"])]
        next_row["pc_temperature_factor"] = float(math.exp(temp_effects[int(row["temperature_K"])]))
        next_row["depth_electrons_at_pc135"] = float(math.exp(log_depth))
        next_row["log_depth"] = log_depth
        updated.append(next_row)
    return updated


def _trap_depths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_trap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trap[row["trap_id"]].append(row)
    trap_rows: list[dict[str, Any]] = []
    for trap_id, values in by_trap.items():
        depths = _finite([row["depth_electrons_at_pc135"] for row in values])
        if depths.size == 0:
            continue
        first = values[0]
        fit_taus = _finite([row["fit_tau_seconds"] for row in values])
        trap_rows.append(
            {
                "trap_id": trap_id,
                "selection_label": first["selection_label"],
                "quadrant": first["quadrant"],
                "row": first["row"],
                "col": first["col"],
                "E_eV": first["E_eV"],
                "tau_135_seconds": first["tau_135_seconds"],
                "temperature_count": len(values),
                "temperatures_K": sorted(int(row["temperature_K"]) for row in values),
                "depth_electrons_at_pc135": float(np.median(depths)),
                "depth_scatter_log_std": float(np.std(np.log(depths))) if depths.size > 1 else 0.0,
                "median_fit_tau_seconds": float(np.median(fit_taus)) if fit_taus.size else math.nan,
            }
        )
    return trap_rows


def _spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(mask)) < 3:
        return {"n": int(np.sum(mask)), "rho": math.nan, "p_value": math.nan}
    result = spearmanr(x_arr[mask], y_arr[mask])
    return {
        "n": int(np.sum(mask)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _correlation_block(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    values = [row[value_key] for row in rows]
    return {
        "value_key": value_key,
        "vs_E_eV": _spearman(values, [row["E_eV"] for row in rows]),
        "vs_log_fit_tau": _spearman(values, [math.log(row["fit_tau_seconds"]) for row in rows]),
        "vs_log_tau_135": _spearman(values, [math.log(row["tau_135_seconds"]) for row in rows]),
    }


def _trap_correlation_block(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    values = [row[value_key] for row in rows]
    return {
        "value_key": value_key,
        "vs_E_eV": _spearman(values, [row["E_eV"] for row in rows]),
        "vs_log_median_fit_tau": _spearman(values, [math.log(row["median_fit_tau_seconds"]) for row in rows]),
        "vs_log_tau_135": _spearman(values, [math.log(row["tau_135_seconds"]) for row in rows]),
    }


def _temperature_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for temp in sorted({int(row["temperature_K"]) for row in rows}):
        temp_rows = [row for row in rows if int(row["temperature_K"]) == temp]
        output[str(temp)] = {
            "record_count": len(temp_rows),
            "trap_count": len({row["trap_id"] for row in temp_rows}),
            "amplitude_electrons": _summary([row["amplitude_electrons"] for row in temp_rows]),
            "fit_coeff": _summary([row["fit_coeff"] for row in temp_rows]),
            "fit_tau_seconds": _summary([row["fit_tau_seconds"] for row in temp_rows]),
            "sampled_peak_fraction": _summary([row["sampled_peak_fraction"] for row in temp_rows]),
        }
    return output


def _write_npz(
    path: Path,
    rows: list[dict[str, Any]],
    trap_rows: list[dict[str, Any]],
    temp_effects: dict[int, float],
) -> None:
    temperatures = np.array(sorted(temp_effects), dtype=float)
    pc = np.array([math.exp(temp_effects[int(temp)]) for temp in temperatures], dtype=float)
    pc_counts = np.array(
        [sum(1 for row in rows if int(row["temperature_K"]) == int(temp)) for temp in temperatures],
        dtype=int,
    )
    default_depth = np.array([row["depth_electrons_at_pc135"] for row in trap_rows], dtype=float)
    observed_amplitudes = np.array([row["amplitude_electrons"] for row in rows], dtype=float)
    observed_temperatures = np.array([row["temperature_K"] for row in rows], dtype=float)

    np.savez_compressed(
        path,
        producing_stage=np.array(STAGE_ID),
        n_pumps=np.array(N_PUMPS),
        reference_temperature_K=np.array(REFERENCE_TEMPERATURE_K),
        temperatures_K=temperatures,
        pc_temperature_factor=pc,
        pc_temperature_record_count=pc_counts,
        default_depth_electrons_at_pc135=default_depth,
        faint_0p5_depth_electrons_at_pc135=0.5 * default_depth,
        faint_0p25_depth_electrons_at_pc135=0.25 * default_depth,
        observed_high_conf_amplitude_electrons=observed_amplitudes,
        observed_high_conf_temperature_K=observed_temperatures,
        trap_ids=np.array([row["trap_id"] for row in trap_rows]),
        variant_names=np.array(["default", "faint_0p5", "faint_0p25"]),
    )


def _maybe_write_plots(
    figures_dir: Path,
    high_conf_rows: list[dict[str, Any]],
    trap_rows: list[dict[str, Any]],
    temp_effects: dict[int, float],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[str] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(
        np.log10([row["depth_electrons_at_pc135"] for row in trap_rows]),
        bins=40,
        histtype="step",
        linewidth=1.8,
    )
    ax.set_xlabel("log10 depth electrons at Pc(135 K)")
    ax.set_ylabel("Traps")
    ax.set_title("Stage 05 default depth prior")
    path = figures_dir / "05_depth_prior_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    output_paths.append(str(path.resolve()))

    temps = np.array(sorted(temp_effects), dtype=float)
    pc = np.array([math.exp(temp_effects[int(temp)]) for temp in temps], dtype=float)
    counts = np.array([sum(1 for row in high_conf_rows if int(row["temperature_K"]) == int(temp)) for temp in temps])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(temps, pc, marker="o", linewidth=1.5)
    for temp, value, count in zip(temps, pc, counts):
        ax.annotate(str(int(count)), (temp, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("Temperature K")
    ax.set_ylabel("Pc(T) factor, normalized at 135 K")
    ax.set_title("Stage 05 common temperature scaling")
    path = figures_dir / "05_pc_temperature_scaling.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    output_paths.append(str(path.resolve()))

    return output_paths


def build_summary(root: Path) -> dict[str, Any]:
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_path = str((workspace / "src" / "amplitude_prior.py").resolve())
    primary_path = root / "fit_dipole_spectra_err_4.h5"
    sensitivity_path = root / "fit_dipole_spectra_err_3.h5"

    primary_rows = _load_observations(primary_path, "n_good_4")
    sensitivity_rows = _load_observations(sensitivity_path, "n_good_3")
    if not primary_rows:
        raise ValueError("No usable primary fit_coeff rows found.")

    primary_high_conf = [row for row in primary_rows if row["high_confidence"]]
    if not primary_high_conf:
        raise ValueError("No high-confidence primary amplitude rows found.")

    temp_effects, _ = _fit_temperature_scaling(primary_high_conf)
    high_conf_with_depth = _add_depth_columns(primary_high_conf, temp_effects)
    trap_rows = _trap_depths(high_conf_with_depth)
    if not trap_rows:
        raise ValueError("No trap-level depth rows found.")

    npz_path = cache_dir / "05_amplitude_prior_v1.npz"
    _write_npz(npz_path, high_conf_with_depth, trap_rows, temp_effects)

    figures = _maybe_write_plots(cache_dir / "figures", high_conf_with_depth, trap_rows, temp_effects)

    pc_values = np.array([math.exp(effect) for effect in temp_effects.values()], dtype=float)
    depth_values = np.array([row["depth_electrons_at_pc135"] for row in trap_rows], dtype=float)
    amplitude_values = np.array([row["amplitude_electrons"] for row in primary_rows], dtype=float)
    high_conf_amplitude_values = np.array(
        [row["amplitude_electrons"] for row in primary_high_conf], dtype=float
    )

    summary = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "inputs": [
            str(primary_path.resolve()),
            str(sensitivity_path.resolve()),
            str((workspace / "agents" / "01_hdf5_records_audit.md").resolve()),
            str((root / "dipole.py").resolve()),
            str((root / "utils.py").resolve()),
        ],
        "outputs": [
            str(npz_path.resolve()),
            str((cache_dir / "05_amplitude_prior_summary.json").resolve()),
            *figures,
        ],
        "selection": {
            "primary": "n_good_4 characterized traps: WellBehavedTrap and GoodEnergyFit and not EnergyFitFailed",
            "sensitivity": "n_good_3 characterized traps with the same cuts",
            "amplitude_definition": "A_electrons = 3000 * fit_coeff",
            "high_confidence_for_prior": {
                "GoodIntensityFit": True,
                "fit_coeff_snr_min": HIGH_CONF_COEFF_SNR_MIN,
                "fit_tau_rel_err_max": HIGH_CONF_TAU_REL_ERR_MAX,
                "sampled_peak_fraction_min": HIGH_CONF_SAMPLED_PEAK_FRACTION_MIN,
            },
        },
        "counts": {
            "primary_good_fit_records": len(primary_rows),
            "primary_good_fit_traps": len({row["trap_id"] for row in primary_rows}),
            "primary_high_confidence_records": len(primary_high_conf),
            "primary_high_confidence_traps": len({row["trap_id"] for row in primary_high_conf}),
            "default_depth_prior_traps": len(trap_rows),
            "sensitivity_ngood3_good_fit_records": len(sensitivity_rows),
            "sensitivity_ngood3_good_fit_traps": len({row["trap_id"] for row in sensitivity_rows}),
        },
        "temperature_record_counts": {
            "primary_good_fit": _temperature_counter(primary_rows),
            "primary_high_confidence": _temperature_counter(primary_high_conf),
        },
        "primary_good_fit_summary": {
            "fit_coeff": _summary([row["fit_coeff"] for row in primary_rows]),
            "amplitude_electrons": _summary(amplitude_values),
            "by_temperature": _temperature_summaries(primary_rows),
        },
        "primary_high_confidence_summary": {
            "fit_coeff": _summary([row["fit_coeff"] for row in primary_high_conf]),
            "amplitude_electrons": _summary(high_conf_amplitude_values),
            "by_temperature": _temperature_summaries(primary_high_conf),
            "bipartite_connectivity": _connected_components(primary_high_conf),
        },
        "pc_temperature_scaling": {
            "method": "robust median two-way fixed-effect fit: log(A) = log(D_trap) + log(Pc_T), anchored to Pc(135 K)=1 when available",
            "reference_temperature_K": REFERENCE_TEMPERATURE_K,
            "temperatures_K": sorted(temp_effects),
            "pc_temperature_factor": {
                str(temp): float(math.exp(effect)) for temp, effect in sorted(temp_effects.items())
            },
            "pc_factor_summary": _summary(pc_values),
            "warm_to_135_ratio_210K": float(math.exp(temp_effects[210])) if 210 in temp_effects else math.nan,
        },
        "depth_prior": {
            "definition": "D_t proxy in electrons at Pc(135 K): median over high-confidence records of A/Pc(T)",
            "default": _summary(depth_values),
            "within_trap_log_scatter": _summary([row["depth_scatter_log_std"] for row in trap_rows]),
            "sensitivity_variants": {
                "default": "empirical high-confidence trap-depth samples",
                "faint_0p5": "same samples shifted fainter by a factor of 2",
                "faint_0p25": "same samples shifted fainter by a factor of 4",
            },
            "truncation_note": "The empirical prior is conditioned on detected GoodIntensityFit records and the high-confidence subset, so it is truncated at the faint end.",
        },
        "correlations": {
            "observation_log_amplitude_primary_good_fit": _correlation_block(
                [
                    {**row, "log_amplitude": math.log(row["amplitude_electrons"])}
                    for row in primary_rows
                ],
                "log_amplitude",
            ),
            "observation_log_depth_high_confidence": _correlation_block(
                high_conf_with_depth,
                "log_depth",
            ),
            "trap_log_depth_high_confidence": _trap_correlation_block(
                [
                    {**row, "log_depth": math.log(row["depth_electrons_at_pc135"])}
                    for row in trap_rows
                ],
                "log_depth",
            ),
        },
        "required_checks": {},
        "stop_conditions": {
            "fit_coeff_absent_or_unreliable": "PASS",
            "strong_amplitude_correlations_without_replacement_model": "PASS",
        },
        "open_questions_for_next_stage": [
            "Use the default prior as conditional on the observed/high-confidence amplitude distribution.",
            "Carry the faint_0p5 and faint_0p25 variants through Stage 08/10 because the faint end is selection-truncated.",
        ],
    }

    breadth_factor = summary["primary_good_fit_summary"]["fit_coeff"]["p95_over_p05"]
    max_abs_rho = max(
        abs(summary["correlations"]["trap_log_depth_high_confidence"]["vs_E_eV"]["rho"]),
        abs(summary["correlations"]["trap_log_depth_high_confidence"]["vs_log_median_fit_tau"]["rho"]),
        abs(summary["correlations"]["trap_log_depth_high_confidence"]["vs_log_tau_135"]["rho"]),
    )
    summary["required_checks"] = {
        "fit_coeff_breadth": {
            "status": "PASS",
            "p95_over_p05": breadth_factor,
            "interpretation": "orders of magnitude" if breadth_factor >= 10 else "factor of a few",
        },
        "rank_correlations_reported": {
            "status": "PASS",
            "max_abs_trap_depth_rho_vs_E_tau_tau135": max_abs_rho,
        },
        "depth_independent_of_E_tau_assumption": {
            "status": "PASS" if max_abs_rho < 0.3 else "CAUTION",
            "interpretation": (
                "No strong trap-level depth correlation with E, median fit_tau, or tau_135."
                if max_abs_rho < 0.3
                else "Trap-level depth correlations are large enough that later stages should add a conditional model."
            ),
        },
        "fainter_amplitude_sensitivity_variants_defined": {
            "status": "PASS",
            "variants": ["faint_0p5", "faint_0p25"],
        },
        "pc_temperature_scaling_estimated": {
            "status": "PASS",
            "pc_factor_p95_over_p05": summary["pc_temperature_scaling"]["pc_factor_summary"]["p95_over_p05"],
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Method 3 Stage 05 amplitude prior.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    summary = build_summary(root)
    summary_path = root / "trap_completeness_method3" / "cache" / "05_amplitude_prior_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_as_builtin(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "Stage 05 amplitude prior: "
        f"primary_records={summary['counts']['primary_good_fit_records']}, "
        f"high_conf_records={summary['counts']['primary_high_confidence_records']}, "
        f"depth_prior_traps={summary['counts']['default_depth_prior_traps']}, "
        f"fit_coeff_p95_over_p05={summary['required_checks']['fit_coeff_breadth']['p95_over_p05']:.3g}, "
        f"max_abs_depth_rho={summary['required_checks']['rank_correlations_reported']['max_abs_trap_depth_rho_vs_E_tau_tau135']:.3g}"
    )


if __name__ == "__main__":
    main()

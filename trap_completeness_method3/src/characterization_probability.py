"""Stage 09 characterization-probability map.

Combines the Stage 08 per-temperature detection grid with the Stage 05
conditional amplitude/depth prior to estimate
P(characterized | tau_135, E).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = REPO_ROOT / "trap_completeness_method3"
CACHE_DIR = METHOD_DIR / "cache"
FIGURES_DIR = CACHE_DIR / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dipole import log_energy_cross_section  # noqa: E402


DEFAULT_STAGE08_H5 = CACHE_DIR / "08_pdet_grid_v1.h5"
DEFAULT_STAGE08_SUMMARY = CACHE_DIR / "08_pdet_grid_summary.json"
DEFAULT_STAGE05_NPZ = CACHE_DIR / "05_amplitude_prior_v1.npz"
DEFAULT_STAGE05_SUMMARY = CACHE_DIR / "05_amplitude_prior_summary.json"
DEFAULT_STAGE01_CSV = CACHE_DIR / "01_records_ngood4.csv"
DEFAULT_STAGE01_NGOOD3_CSV = CACHE_DIR / "01_records_ngood3.csv"
DEFAULT_OUTPUT_H5 = CACHE_DIR / "09_characterization_probability_v1.h5"
DEFAULT_OUTPUT_SUMMARY = CACHE_DIR / "09_characterization_probability_summary.json"
DEFAULT_SMOKE_SUMMARY = CACHE_DIR / "09_characterization_probability_smoke_summary.json"


@dataclass(frozen=True)
class Stage08Grid:
    """Production p_det grid from Stage 08."""

    h5_path: Path
    summary_path: Path
    temperatures_K: np.ndarray
    tau_seconds: np.ndarray
    amplitude_electrons: np.ndarray
    p_det: np.ndarray
    summary: dict
    h5_metadata: dict


@dataclass(frozen=True)
class AmplitudePrior:
    """Stage 05 empirical depth prior."""

    npz_path: Path
    summary_path: Path
    variant: str
    depth_electrons_at_pc135: np.ndarray
    temperatures_K: np.ndarray
    pc_temperature_factor: np.ndarray
    trap_ids: np.ndarray
    summary: dict


@dataclass(frozen=True)
class InterpolationPolicy:
    """Explicit policy for Stage 08 grid boundaries."""

    tau_axis: str = "log_tau_seconds"
    amplitude_axis: str = "linear_electrons"
    tau_below_grid: str = "zero"
    tau_above_grid: str = "zero"
    amplitude_below_grid: str = "zero"
    amplitude_above_grid: str = "edge_clipped"
    temperature_axis: str = "exact_measured_temperatures_only"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_stage08_grid(
    h5_path: Path = DEFAULT_STAGE08_H5,
    summary_path: Path = DEFAULT_STAGE08_SUMMARY,
) -> Stage08Grid:
    h5_path = Path(h5_path)
    summary_path = Path(summary_path)
    summary = _load_json(summary_path)
    with h5py.File(h5_path, "r") as handle:
        temperatures = handle["grid/temperature_K"][:].astype(float)
        tau_seconds = handle["grid/tau_seconds"][:].astype(float)
        amplitude = handle["grid/amplitude_electrons"][:].astype(float)
        p_det = handle["results/p_det"][:].astype(float)
        metadata_raw = handle.attrs.get("metadata_json", "{}")
        if isinstance(metadata_raw, bytes):
            metadata_raw = metadata_raw.decode("utf-8")
        h5_metadata = json.loads(metadata_raw)

    expected_shape = (temperatures.size, tau_seconds.size, amplitude.size)
    if p_det.shape != expected_shape:
        raise ValueError(f"Stage 08 p_det shape {p_det.shape} != {expected_shape}")
    if np.any(np.diff(temperatures) <= 0):
        raise ValueError("Stage 08 temperatures must be strictly increasing.")
    if np.any(np.diff(tau_seconds) <= 0):
        raise ValueError("Stage 08 tau grid must be strictly increasing.")
    if np.any(np.diff(amplitude) <= 0):
        raise ValueError("Stage 08 amplitude grid must be strictly increasing.")
    if np.nanmin(p_det) < -1e-12 or np.nanmax(p_det) > 1 + 1e-12:
        raise ValueError("Stage 08 p_det values are outside [0, 1].")

    return Stage08Grid(
        h5_path=h5_path,
        summary_path=summary_path,
        temperatures_K=temperatures,
        tau_seconds=tau_seconds,
        amplitude_electrons=amplitude,
        p_det=np.clip(p_det, 0.0, 1.0),
        summary=summary,
        h5_metadata=h5_metadata,
    )


def load_amplitude_prior(
    npz_path: Path = DEFAULT_STAGE05_NPZ,
    summary_path: Path = DEFAULT_STAGE05_SUMMARY,
    variant: str = "default",
) -> AmplitudePrior:
    npz_path = Path(npz_path)
    summary_path = Path(summary_path)
    summary = _load_json(summary_path)
    dataset_by_variant = {
        "default": "default_depth_electrons_at_pc135",
        "faint_0p5": "faint_0p5_depth_electrons_at_pc135",
        "faint_0p25": "faint_0p25_depth_electrons_at_pc135",
    }
    if variant not in dataset_by_variant:
        raise ValueError(f"Unknown amplitude-prior variant: {variant}")

    with np.load(npz_path, allow_pickle=False) as npz:
        depth = npz[dataset_by_variant[variant]].astype(float)
        temperatures = npz["temperatures_K"].astype(float)
        pc_factor = npz["pc_temperature_factor"].astype(float)
        trap_ids = npz["trap_ids"].astype(str)

    if depth.ndim != 1 or depth.size == 0:
        raise ValueError("Amplitude-prior depth samples must be a non-empty 1D array.")
    if np.any(~np.isfinite(depth)) or np.any(depth <= 0):
        raise ValueError("Amplitude-prior depth samples must be finite and positive.")
    if temperatures.shape != pc_factor.shape:
        raise ValueError("Stage 05 temperatures and Pc factors have inconsistent shapes.")

    return AmplitudePrior(
        npz_path=npz_path,
        summary_path=summary_path,
        variant=variant,
        depth_electrons_at_pc135=depth,
        temperatures_K=temperatures,
        pc_temperature_factor=pc_factor,
        trap_ids=trap_ids,
        summary=summary,
    )


def align_pc_temperature_factor(stage08: Stage08Grid, prior: AmplitudePrior) -> np.ndarray:
    pc_by_temperature = {
        float(temp): float(pc)
        for temp, pc in zip(prior.temperatures_K, prior.pc_temperature_factor)
    }
    missing = [float(temp) for temp in stage08.temperatures_K if float(temp) not in pc_by_temperature]
    if missing:
        raise ValueError(f"Stage 05 Pc(T) is missing Stage 08 temperatures: {missing}")
    return np.array([pc_by_temperature[float(temp)] for temp in stage08.temperatures_K], dtype=float)


def infer_log_sigma(tau_135_seconds: np.ndarray, energy_eV: np.ndarray) -> np.ndarray:
    tau_135_seconds = np.asarray(tau_135_seconds, dtype=float)
    energy_eV = np.asarray(energy_eV, dtype=float)
    if np.any(tau_135_seconds <= 0):
        raise ValueError("tau_135 values must be positive.")
    return log_energy_cross_section(135.0, energy_eV, 0.0) - np.log(tau_135_seconds)


def implied_log_tau_by_temperature(
    tau_135_seconds: np.ndarray,
    energy_eV: np.ndarray,
    temperatures_K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tau_135_seconds = np.asarray(tau_135_seconds, dtype=float)
    energy_eV = np.asarray(energy_eV, dtype=float)
    temperatures_K = np.asarray(temperatures_K, dtype=float)
    log_sigma = infer_log_sigma(tau_135_seconds, energy_eV)
    log_tau = log_energy_cross_section(
        temperatures_K[:, None],
        energy_eV[None, :],
        log_sigma[None, :],
    )
    return np.asarray(log_tau, dtype=float), np.asarray(log_sigma, dtype=float)


def algebra_reconstruction_check(tau_135_seconds: np.ndarray, energy_eV: np.ndarray) -> dict:
    tau_135_seconds = np.asarray(tau_135_seconds, dtype=float)
    energy_eV = np.asarray(energy_eV, dtype=float)
    log_sigma = infer_log_sigma(tau_135_seconds, energy_eV)
    reconstructed_log_tau = log_energy_cross_section(135.0, energy_eV, log_sigma)
    target_log_tau = np.log(tau_135_seconds)
    delta = reconstructed_log_tau - target_log_tau
    return {
        "formula": "log_sigma = log_energy_cross_section(135, E, 0) - log(tau_135)",
        "max_abs_log_tau_error": float(np.max(np.abs(delta))),
        "median_abs_log_tau_error": float(np.median(np.abs(delta))),
        "passed": bool(np.max(np.abs(delta)) < 1e-12),
    }


def poisson_binomial_tail(probs: np.ndarray, n_good: int) -> np.ndarray:
    """Return P(sum Bernoulli(probs_i) >= n_good), along the last axis."""

    probs = np.asarray(probs, dtype=float)
    if np.any(probs < -1e-12) or np.any(probs > 1 + 1e-12):
        raise ValueError("Poisson-binomial probabilities must be in [0, 1].")
    probs = np.clip(probs, 0.0, 1.0)
    if n_good <= 0:
        return np.ones(probs.shape[:-1], dtype=float)
    if probs.shape[-1] == 0:
        return np.zeros(probs.shape[:-1], dtype=float)

    dist = np.zeros(probs.shape[:-1] + (n_good,), dtype=float)
    dist[..., 0] = 1.0
    for p in np.moveaxis(probs, -1, 0):
        q = 1.0 - p
        for count in range(n_good - 1, 0, -1):
            dist[..., count] = dist[..., count] * q + dist[..., count - 1] * p
        dist[..., 0] *= q
    return np.clip(1.0 - np.sum(dist, axis=-1), 0.0, 1.0)


def poisson_binomial_tail_bruteforce(probs: Iterable[float], n_good: int) -> float:
    probs = np.asarray(list(probs), dtype=float)
    if n_good <= 0:
        return 1.0
    total = 0.0
    for mask in product([0, 1], repeat=probs.size):
        count = sum(mask)
        if count < n_good:
            continue
        term = 1.0
        for bit, prob in zip(mask, probs):
            term *= prob if bit else (1.0 - prob)
        total += term
    return float(total)


def run_poisson_binomial_tests() -> dict:
    vectors = [
        np.array([0.1, 0.5, 0.9]),
        np.array([0.0, 0.2, 1.0, 0.7]),
        np.array([0.13, 0.31, 0.47, 0.59, 0.83]),
    ]
    results = []
    max_abs_error = 0.0
    for probs in vectors:
        for n_good in range(1, probs.size + 2):
            dp = float(poisson_binomial_tail(probs, n_good))
            brute = poisson_binomial_tail_bruteforce(probs, n_good)
            abs_error = abs(dp - brute)
            max_abs_error = max(max_abs_error, abs_error)
            results.append(
                {
                    "probs": probs.tolist(),
                    "n_good": n_good,
                    "dynamic_programming": dp,
                    "bruteforce": brute,
                    "abs_error": abs_error,
                }
            )
    return {
        "passed": bool(max_abs_error < 1e-12),
        "max_abs_error": float(max_abs_error),
        "cases": results,
    }


def _prepare_amplitude_interpolation(
    amplitude_grid: np.ndarray,
    amplitudes_by_temperature_depth: np.ndarray,
) -> dict:
    amp = amplitudes_by_temperature_depth
    n_amp = amplitude_grid.size
    idx = np.searchsorted(amplitude_grid, amp, side="right") - 1
    below = amp < amplitude_grid[0]
    above = amp > amplitude_grid[-1]
    finite = np.isfinite(amp)
    nonzero = finite & ~below

    idx = np.clip(idx, 0, n_amp - 2)
    weight = (amp - amplitude_grid[idx]) / (amplitude_grid[idx + 1] - amplitude_grid[idx])
    weight = np.clip(weight, 0.0, 1.0)

    if np.any(above):
        idx = idx.copy()
        weight = weight.copy()
        idx[above] = n_amp - 2
        weight[above] = 1.0

    return {
        "idx": idx.astype(np.int64),
        "weight": weight.astype(float),
        "below": below,
        "above": above,
        "nonzero": nonzero,
        "finite": finite,
    }


def _tau_interp_indices(log_tau_grid: np.ndarray, log_tau_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.searchsorted(log_tau_grid, log_tau_values, side="right") - 1
    below = log_tau_values < log_tau_grid[0]
    above = log_tau_values > log_tau_grid[-1]
    finite = np.isfinite(log_tau_values)
    valid = finite & ~below & ~above
    idx = np.clip(idx, 0, log_tau_grid.size - 2)
    weight = (log_tau_values - log_tau_grid[idx]) / (log_tau_grid[idx + 1] - log_tau_grid[idx])
    weight = np.clip(weight, 0.0, 1.0)
    return idx.astype(np.int64), weight.astype(float), valid, below, above


def _interp_temperature_pdet(
    p_det_T: np.ndarray,
    tau_idx: np.ndarray,
    tau_weight: np.ndarray,
    tau_valid: np.ndarray,
    amp_idx: np.ndarray,
    amp_weight: np.ndarray,
    amp_nonzero: np.ndarray,
) -> np.ndarray:
    low_tau_low_amp = p_det_T[tau_idx[:, None], amp_idx[None, :]]
    low_tau_high_amp = p_det_T[tau_idx[:, None], amp_idx[None, :] + 1]
    high_tau_low_amp = p_det_T[tau_idx[:, None] + 1, amp_idx[None, :]]
    high_tau_high_amp = p_det_T[tau_idx[:, None] + 1, amp_idx[None, :] + 1]

    amp_weight_2d = amp_weight[None, :]
    low_tau = low_tau_low_amp * (1.0 - amp_weight_2d) + low_tau_high_amp * amp_weight_2d
    high_tau = high_tau_low_amp * (1.0 - amp_weight_2d) + high_tau_high_amp * amp_weight_2d
    p_det = low_tau * (1.0 - tau_weight[:, None]) + high_tau * tau_weight[:, None]

    p_det[~tau_valid, :] = 0.0
    p_det[:, ~amp_nonzero] = 0.0
    return np.clip(p_det, 0.0, 1.0)


def _update_truncated_count_distribution(dist: np.ndarray, p: np.ndarray) -> None:
    q = 1.0 - p
    for count in range(dist.shape[-1] - 1, 0, -1):
        dist[..., count] = dist[..., count] * q + dist[..., count - 1] * p
    dist[..., 0] *= q


def _probability_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite_count": 0}
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p16": float(np.percentile(finite, 16)),
        "median": float(np.median(finite)),
        "p84": float(np.percentile(finite, 84)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "fraction_ge_0p5": float(np.mean(finite >= 0.5)),
        "fraction_ge_0p8": float(np.mean(finite >= 0.8)),
        "fraction_ge_0p9": float(np.mean(finite >= 0.9)),
    }


def summarize_point_diagnostics(diagnostics: dict, n_temperatures: int, n_depth_samples: int) -> dict:
    tau_below_count = diagnostics["tau_below_count"]
    tau_above_count = diagnostics["tau_above_count"]
    tau_oob_count = diagnostics["tau_oob_count"]
    amp_below_count_by_T = diagnostics["amplitude_below_count_by_temperature"]
    amp_above_count_by_T = diagnostics["amplitude_above_count_by_temperature"]
    n_points = tau_oob_count.size
    total_tau_queries = max(n_points * n_temperatures, 1)
    total_amp_queries = max(n_depth_samples * n_temperatures, 1)
    total_pdet_queries = max(n_points * n_depth_samples * n_temperatures, 1)

    zero_due_to_tau = int(np.sum(tau_oob_count) * n_depth_samples)
    zero_due_to_amp_below = 0
    clipped_due_to_amp_above = 0
    for temp_index in range(n_temperatures):
        in_tau_count = int(n_points - diagnostics["tau_oob_count_by_temperature"][temp_index])
        zero_due_to_amp_below += in_tau_count * int(amp_below_count_by_T[temp_index])
        clipped_due_to_amp_above += in_tau_count * int(amp_above_count_by_T[temp_index])

    return {
        "point_count": int(n_points),
        "temperature_count": int(n_temperatures),
        "depth_sample_count": int(n_depth_samples),
        "tau_below_grid_fraction": float(np.sum(tau_below_count) / total_tau_queries),
        "tau_above_grid_fraction": float(np.sum(tau_above_count) / total_tau_queries),
        "tau_out_of_grid_fraction": float(np.sum(tau_oob_count) / total_tau_queries),
        "all_temperatures_tau_out_of_grid_fraction": float(np.mean(tau_oob_count == n_temperatures)),
        "amplitude_below_grid_fraction": float(np.sum(amp_below_count_by_T) / total_amp_queries),
        "amplitude_above_grid_fraction": float(np.sum(amp_above_count_by_T) / total_amp_queries),
        "pdet_queries_zeroed_by_tau_fraction": float(zero_due_to_tau / total_pdet_queries),
        "pdet_queries_zeroed_by_amplitude_below_fraction": float(zero_due_to_amp_below / total_pdet_queries),
        "pdet_queries_edge_clipped_by_amplitude_above_fraction": float(clipped_due_to_amp_above / total_pdet_queries),
        "per_temperature": diagnostics["per_temperature"],
    }


def compute_probabilities_for_points(
    tau_135_seconds: np.ndarray,
    energy_eV: np.ndarray,
    stage08: Stage08Grid,
    prior: AmplitudePrior,
    n_good_values: tuple[int, ...] = (4, 3),
    chunk_size: int = 256,
) -> tuple[dict[int, np.ndarray], dict]:
    tau_135_seconds = np.asarray(tau_135_seconds, dtype=float).reshape(-1)
    energy_eV = np.asarray(energy_eV, dtype=float).reshape(-1)
    if tau_135_seconds.shape != energy_eV.shape:
        raise ValueError("tau_135 and E arrays must have the same shape.")
    if np.any(tau_135_seconds <= 0):
        raise ValueError("tau_135 values must be positive.")
    if np.any(~np.isfinite(energy_eV)):
        raise ValueError("E values must be finite.")

    n_points = tau_135_seconds.size
    n_temperatures = stage08.temperatures_K.size
    depth_samples = prior.depth_electrons_at_pc135
    n_depth = depth_samples.size
    pc_factor = align_pc_temperature_factor(stage08, prior)
    amplitudes_by_T_depth = pc_factor[:, None] * depth_samples[None, :]
    amp_interp = _prepare_amplitude_interpolation(stage08.amplitude_electrons, amplitudes_by_T_depth)
    log_tau_grid = np.log(stage08.tau_seconds)

    max_n_good = max(n_good_values)
    probabilities = {n_good: np.empty(n_points, dtype=float) for n_good in n_good_values}
    tau_below_count = np.zeros(n_points, dtype=np.int16)
    tau_above_count = np.zeros(n_points, dtype=np.int16)
    tau_oob_count = np.zeros(n_points, dtype=np.int16)
    tau_oob_count_by_temperature = np.zeros(n_temperatures, dtype=np.int64)
    tau_below_count_by_temperature = np.zeros(n_temperatures, dtype=np.int64)
    tau_above_count_by_temperature = np.zeros(n_temperatures, dtype=np.int64)

    amp_below_count_by_temperature = np.sum(amp_interp["below"], axis=1).astype(np.int64)
    amp_above_count_by_temperature = np.sum(amp_interp["above"], axis=1).astype(np.int64)

    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        tau_chunk = tau_135_seconds[start:stop]
        energy_chunk = energy_eV[start:stop]
        log_tau_by_T, _ = implied_log_tau_by_temperature(
            tau_chunk,
            energy_chunk,
            stage08.temperatures_K,
        )
        dist = np.zeros((stop - start, n_depth, max_n_good), dtype=float)
        dist[..., 0] = 1.0

        for temp_index in range(n_temperatures):
            tau_idx, tau_weight, tau_valid, tau_below, tau_above = _tau_interp_indices(
                log_tau_grid,
                log_tau_by_T[temp_index],
            )
            tau_oob = ~tau_valid
            tau_below_count[start:stop] += tau_below.astype(np.int16)
            tau_above_count[start:stop] += tau_above.astype(np.int16)
            tau_oob_count[start:stop] += tau_oob.astype(np.int16)
            tau_oob_count_by_temperature[temp_index] += int(np.sum(tau_oob))
            tau_below_count_by_temperature[temp_index] += int(np.sum(tau_below))
            tau_above_count_by_temperature[temp_index] += int(np.sum(tau_above))

            p_det = _interp_temperature_pdet(
                stage08.p_det[temp_index],
                tau_idx,
                tau_weight,
                tau_valid,
                amp_interp["idx"][temp_index],
                amp_interp["weight"][temp_index],
                amp_interp["nonzero"][temp_index],
            )
            _update_truncated_count_distribution(dist, p_det)

        for n_good in n_good_values:
            tail_by_depth = np.clip(1.0 - np.sum(dist[..., :n_good], axis=-1), 0.0, 1.0)
            probabilities[n_good][start:stop] = np.mean(tail_by_depth, axis=1)

    per_temperature = []
    for temp_index, temperature in enumerate(stage08.temperatures_K):
        per_temperature.append(
            {
                "temperature_K": float(temperature),
                "tau_below_grid_fraction": float(tau_below_count_by_temperature[temp_index] / max(n_points, 1)),
                "tau_above_grid_fraction": float(tau_above_count_by_temperature[temp_index] / max(n_points, 1)),
                "tau_out_of_grid_fraction": float(tau_oob_count_by_temperature[temp_index] / max(n_points, 1)),
                "amplitude_below_grid_fraction": float(amp_below_count_by_temperature[temp_index] / max(n_depth, 1)),
                "amplitude_above_grid_fraction": float(amp_above_count_by_temperature[temp_index] / max(n_depth, 1)),
            }
        )

    diagnostics = {
        "tau_below_count": tau_below_count,
        "tau_above_count": tau_above_count,
        "tau_oob_count": tau_oob_count,
        "tau_oob_count_by_temperature": tau_oob_count_by_temperature,
        "tau_below_count_by_temperature": tau_below_count_by_temperature,
        "tau_above_count_by_temperature": tau_above_count_by_temperature,
        "amplitude_below_count_by_temperature": amp_below_count_by_temperature,
        "amplitude_above_count_by_temperature": amp_above_count_by_temperature,
        "per_temperature": per_temperature,
        "summary": summarize_point_diagnostics(
            {
                "tau_below_count": tau_below_count,
                "tau_above_count": tau_above_count,
                "tau_oob_count": tau_oob_count,
                "tau_oob_count_by_temperature": tau_oob_count_by_temperature,
                "amplitude_below_count_by_temperature": amp_below_count_by_temperature,
                "amplitude_above_count_by_temperature": amp_above_count_by_temperature,
                "per_temperature": per_temperature,
            },
            n_temperatures,
            n_depth,
        ),
        "amplitudes_by_temperature_summary": {
            "min": float(np.min(amplitudes_by_T_depth)),
            "median": float(np.median(amplitudes_by_T_depth)),
            "max": float(np.max(amplitudes_by_T_depth)),
        },
    }
    return probabilities, diagnostics


def make_probability_map(
    tau_135_grid: np.ndarray,
    energy_grid: np.ndarray,
    stage08: Stage08Grid,
    prior: AmplitudePrior,
    n_good_values: tuple[int, ...] = (4, 3),
    chunk_size: int = 256,
) -> tuple[dict[int, np.ndarray], dict]:
    tau_mesh, energy_mesh = np.meshgrid(tau_135_grid, energy_grid, indexing="ij")
    probabilities, diagnostics = compute_probabilities_for_points(
        tau_mesh.ravel(),
        energy_mesh.ravel(),
        stage08,
        prior,
        n_good_values=n_good_values,
        chunk_size=chunk_size,
    )
    probability_maps = {
        n_good: values.reshape(tau_135_grid.size, energy_grid.size)
        for n_good, values in probabilities.items()
    }
    diagnostics["tau_oob_fraction_map"] = (
        diagnostics["tau_oob_count"].reshape(tau_135_grid.size, energy_grid.size)
        / stage08.temperatures_K.size
    )
    diagnostics["all_temperatures_tau_oob_map"] = (
        diagnostics["tau_oob_count"].reshape(tau_135_grid.size, energy_grid.size)
        == stage08.temperatures_K.size
    )
    return probability_maps, diagnostics


def default_tau_135_grid(
    n_tau: int,
    tau_min_seconds: float = 2e-5,
    tau_max_seconds: float = 1e8,
) -> np.ndarray:
    if n_tau < 2:
        raise ValueError("tau_135 grid needs at least two points.")
    if tau_min_seconds <= 0 or tau_max_seconds <= 0:
        raise ValueError("tau_135 grid limits must be positive.")
    if tau_min_seconds >= tau_max_seconds:
        raise ValueError("tau_135 min must be smaller than tau_135 max.")
    return np.geomspace(tau_min_seconds, tau_max_seconds, n_tau)


def default_energy_grid(n_energy: int) -> np.ndarray:
    return np.linspace(0.04, 0.70, n_energy)


def load_known_traps(csv_path: Path = DEFAULT_STAGE01_CSV) -> dict:
    import csv

    tau_values = []
    energy_values = []
    good_counts = []
    quadrants = []
    rows = []
    cols = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau_values.append(float(row["tau_135_seconds"]))
            energy_values.append(float(row["E_eV"]))
            good_counts.append(int(float(row["good_temperature_count"])))
            quadrants.append(int(float(row["quadrant"])))
            rows.append(int(float(row["row"])))
            cols.append(int(float(row["col"])))
    return {
        "tau_135_seconds": np.asarray(tau_values, dtype=float),
        "E_eV": np.asarray(energy_values, dtype=float),
        "good_temperature_count": np.asarray(good_counts, dtype=int),
        "quadrant": np.asarray(quadrants, dtype=int),
        "row": np.asarray(rows, dtype=int),
        "col": np.asarray(cols, dtype=int),
    }


def validate_known_traps(
    stage08: Stage08Grid,
    prior: AmplitudePrior,
    known_trap_csv: Path = DEFAULT_STAGE01_CSV,
    chunk_size: int = 256,
) -> tuple[dict, dict]:
    known = load_known_traps(known_trap_csv)
    probabilities, diagnostics = compute_probabilities_for_points(
        known["tau_135_seconds"],
        known["E_eV"],
        stage08,
        prior,
        n_good_values=(4, 3),
        chunk_size=chunk_size,
    )
    p4 = probabilities[4]
    p3 = probabilities[3]
    low_mask = p4 < 0.5
    very_low_mask = p4 < 0.2
    summary = {
        "known_trap_csv": str(known_trap_csv),
        "trap_count": int(p4.size),
        "n_good_4_probability": _probability_summary(p4),
        "n_good_3_probability": _probability_summary(p3),
        "fraction_p4_ge_0p5": float(np.mean(p4 >= 0.5)),
        "fraction_p4_ge_0p8": float(np.mean(p4 >= 0.8)),
        "fraction_p4_ge_0p9": float(np.mean(p4 >= 0.9)),
        "low_p4_count_lt_0p5": int(np.sum(low_mask)),
        "very_low_p4_count_lt_0p2": int(np.sum(very_low_mask)),
        "all_temperatures_tau_oob_count": int(np.sum(diagnostics["tau_oob_count"] == stage08.temperatures_K.size)),
        "median_tau_oob_fraction": float(np.median(diagnostics["tau_oob_count"] / stage08.temperatures_K.size)),
        "low_p4_examples": [],
    }
    low_indices = np.argsort(p4)[:10]
    for idx in low_indices:
        summary["low_p4_examples"].append(
            {
                "quadrant": int(known["quadrant"][idx]),
                "row": int(known["row"][idx]),
                "col": int(known["col"][idx]),
                "tau_135_seconds": float(known["tau_135_seconds"][idx]),
                "E_eV": float(known["E_eV"][idx]),
                "good_temperature_count": int(known["good_temperature_count"][idx]),
                "p_characterized_n_good_4": float(p4[idx]),
                "p_characterized_n_good_3": float(p3[idx]),
                "tau_oob_fraction": float(diagnostics["tau_oob_count"][idx] / stage08.temperatures_K.size),
            }
        )

    validation_arrays = {
        **known,
        "p_characterized_n_good_4": p4,
        "p_characterized_n_good_3": p3,
        "tau_oob_fraction": diagnostics["tau_oob_count"] / stage08.temperatures_K.size,
    }
    return summary, validation_arrays


def summarize_unbounded_regime(
    tau_135_grid: np.ndarray,
    energy_grid: np.ndarray,
    probability_map_n4: np.ndarray,
    tau_oob_fraction_map: np.ndarray,
    all_temperatures_oob_map: np.ndarray,
) -> dict:
    tau_mesh, energy_mesh = np.meshgrid(tau_135_grid, energy_grid, indexing="ij")
    all_mask = all_temperatures_oob_map.astype(bool)
    high_oob_low_prob = (tau_oob_fraction_map >= 0.8) & (probability_map_n4 < 0.1)

    def masked_range(mask: np.ndarray) -> dict:
        if not np.any(mask):
            return {"count": 0}
        return {
            "count": int(np.sum(mask)),
            "tau_135_seconds_min": float(np.min(tau_mesh[mask])),
            "tau_135_seconds_max": float(np.max(tau_mesh[mask])),
            "E_eV_min": float(np.min(energy_mesh[mask])),
            "E_eV_max": float(np.max(energy_mesh[mask])),
            "p4_min": float(np.min(probability_map_n4[mask])),
            "p4_median": float(np.median(probability_map_n4[mask])),
            "p4_max": float(np.max(probability_map_n4[mask])),
        }

    return {
        "definition": "Regimes are flagged by the fraction of measured temperatures whose implied tau(T) is outside the Stage 08 tau grid. Under the Stage 09 primary policy, those temperature detections are set to zero rather than extrapolated.",
        "all_temperatures_out_of_stage08_tau_band": masked_range(all_mask),
        "tau_oob_fraction_ge_0p8_and_p4_lt_0p1": masked_range(high_oob_low_prob),
        "all_temperatures_out_of_band_grid_fraction": float(np.mean(all_mask)),
        "max_p4_when_all_temperatures_out_of_band": float(np.max(probability_map_n4[all_mask])) if np.any(all_mask) else None,
    }


def write_hdf5(
    output_path: Path,
    tau_135_grid: np.ndarray,
    energy_grid: np.ndarray,
    stage08: Stage08Grid,
    prior: AmplitudePrior,
    probability_maps: dict[int, np.ndarray],
    diagnostics: dict,
    validation_arrays_by_label: dict[str, dict],
    metadata: dict,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        grid_group = handle.create_group("grid")
        grid_group.create_dataset("tau_135_seconds", data=tau_135_grid)
        grid_group.create_dataset("E_eV", data=energy_grid)
        grid_group.create_dataset("temperature_K", data=stage08.temperatures_K)
        grid_group.create_dataset("stage08_tau_seconds", data=stage08.tau_seconds)
        grid_group.create_dataset("stage08_amplitude_electrons", data=stage08.amplitude_electrons)

        prior_group = handle.create_group("amplitude_prior")
        prior_group.create_dataset("depth_electrons_at_pc135", data=prior.depth_electrons_at_pc135)
        prior_group.create_dataset("pc_temperature_factor_aligned_to_stage08", data=align_pc_temperature_factor(stage08, prior))
        prior_group.attrs["variant"] = prior.variant
        prior_group.attrs["conditioning_note"] = (
            "Empirical Stage 05 prior is conditional on observed high-confidence GoodIntensityFit traps."
        )

        results_group = handle.create_group("results")
        for n_good, probability_map in probability_maps.items():
            results_group.create_dataset(f"p_characterized_n_good_{n_good}", data=probability_map)

        diag_group = handle.create_group("diagnostics")
        diag_group.create_dataset("tau_oob_fraction", data=diagnostics["tau_oob_fraction_map"])
        diag_group.create_dataset("all_temperatures_tau_oob", data=diagnostics["all_temperatures_tau_oob_map"].astype(np.uint8))
        diag_group.create_dataset(
            "tau_oob_count",
            data=diagnostics["tau_oob_count"].reshape(tau_135_grid.size, energy_grid.size),
        )
        diag_group.create_dataset("amplitude_below_count_by_temperature", data=diagnostics["amplitude_below_count_by_temperature"])
        diag_group.create_dataset("amplitude_above_count_by_temperature", data=diagnostics["amplitude_above_count_by_temperature"])

        validation_group = handle.create_group("validation_known_traps")
        for label, validation_arrays in validation_arrays_by_label.items():
            label_group = validation_group.create_group(label)
            for key, values in validation_arrays.items():
                label_group.create_dataset(key, data=values)

        handle.attrs["metadata_json"] = json.dumps(_jsonable(metadata), sort_keys=True)


def make_figures(
    output_h5_path: Path,
    figures_dir: Path = FIGURES_DIR,
    figure_prefix: str = "09",
) -> list[str]:
    figure_paths = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with h5py.File(output_h5_path, "r") as handle:
            tau = handle["grid/tau_135_seconds"][:]
            energy = handle["grid/E_eV"][:]
            p4 = handle["results/p_characterized_n_good_4"][:]
            oob = handle["diagnostics/tau_oob_fraction"][:]

        figures_dir.mkdir(parents=True, exist_ok=True)
        for name, values, label, cmap in [
            (f"{figure_prefix}_characterization_probability_n4.png", p4, "P(characterized), n_good=4", "viridis"),
            (f"{figure_prefix}_tau_oob_fraction.png", oob, "Fraction of temperatures out of Stage 08 tau band", "magma"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            mesh = ax.pcolormesh(tau, energy, values.T, shading="auto", vmin=0.0, vmax=1.0, cmap=cmap)
            ax.set_xscale("log")
            ax.set_xlabel("tau_135 [s]")
            ax.set_ylabel("E [eV]")
            ax.set_title(label)
            fig.colorbar(mesh, ax=ax, label=label)
            fig.tight_layout()
            figure_path = figures_dir / name
            fig.savefig(figure_path, dpi=180)
            plt.close(fig)
            figure_paths.append(str(figure_path))
    except Exception as exc:  # pragma: no cover - figures are optional diagnostics.
        figure_paths.append(f"figure_generation_failed: {exc}")
    return figure_paths


def build_stage09_metadata(
    stage08: Stage08Grid,
    prior: AmplitudePrior,
    policy: InterpolationPolicy,
    tau_135_grid: np.ndarray,
    energy_grid: np.ndarray,
    n_good_values: tuple[int, ...],
    poisson_tests: dict,
    algebra_check: dict,
    runtime_seconds: float,
    output_h5: Path,
    output_summary: Path,
    smoke_summary: dict | None,
) -> dict:
    return {
        "producing_stage": "09_characterization_probability",
        "produced_at": _now_iso(),
        "code_path": str(Path(__file__).resolve()),
        "inputs": {
            "stage08_h5": str(stage08.h5_path),
            "stage08_summary": str(stage08.summary_path),
            "stage05_npz": str(prior.npz_path),
            "stage05_summary": str(prior.summary_path),
            "stage01_ngood4_known_traps_csv": str(DEFAULT_STAGE01_CSV),
            "stage01_ngood3_known_traps_csv": str(DEFAULT_STAGE01_NGOOD3_CSV),
            "dipole_py": str(REPO_ROOT / "dipole.py"),
        },
        "outputs": {
            "hdf5": str(output_h5),
            "summary_json": str(output_summary),
        },
        "stage08_production_provenance": {
            "produced_at": stage08.summary.get("produced_at"),
            "producing_stage": stage08.summary.get("producing_stage"),
            "shape": stage08.summary.get("shape"),
            "random_seed": stage08.summary.get("random_seed"),
            "realizations_per_grid_point": stage08.summary.get("realizations_per_grid_point"),
            "code_path": stage08.summary.get("code_path"),
            "model_choice": stage08.summary.get("model_choice"),
            "runtime": stage08.summary.get("runtime"),
        },
        "amplitude_prior": {
            "variant": prior.variant,
            "depth_sample_count": int(prior.depth_electrons_at_pc135.size),
            "pc_temperature_factor_source": str(prior.npz_path),
            "conditioning_note": prior.summary.get("depth_prior", {}).get("truncation_note"),
            "definition": prior.summary.get("depth_prior", {}).get("definition"),
        },
        "grid_definitions": {
            "tau_135_seconds": {
                "count": int(tau_135_grid.size),
                "min": float(np.min(tau_135_grid)),
                "max": float(np.max(tau_135_grid)),
                "spacing": "log",
            },
            "E_eV": {
                "count": int(energy_grid.size),
                "min": float(np.min(energy_grid)),
                "max": float(np.max(energy_grid)),
                "spacing": "linear",
            },
            "measured_temperatures_K": stage08.temperatures_K.tolist(),
        },
        "n_good_values": list(n_good_values),
        "primary_n_good": 4,
        "sensitivity_n_good": 3 if 3 in n_good_values else None,
        "interpolation_policy": policy.__dict__,
        "out_of_grid_policy": {
            "tau": "Values below or above the Stage 08 tau grid are assigned p_det=0 for that temperature.",
            "amplitude": "Values below the Stage 08 amplitude grid are assigned p_det=0; values above are clipped to the top amplitude bin.",
            "temperature": "No temperature interpolation is performed; Stage 09 uses the measured Stage 08 temperatures exactly.",
        },
        "algebra_check": algebra_check,
        "poisson_binomial_tests": poisson_tests,
        "runtime_seconds": runtime_seconds,
        "smoke_check": smoke_summary,
    }


def run_smoke(
    stage08: Stage08Grid,
    prior: AmplitudePrior,
    smoke_summary_path: Path = DEFAULT_SMOKE_SUMMARY,
    production_tau_count: int = 161,
    production_energy_count: int = 121,
    chunk_size: int = 256,
) -> dict:
    tau_grid = np.geomspace(1e-3, 1e5, 10)
    energy_grid = np.linspace(0.08, 0.62, 9)
    start = time.perf_counter()
    poisson_tests = run_poisson_binomial_tests()
    algebra_check = algebra_reconstruction_check(
        np.array([tau_grid[0], tau_grid[len(tau_grid) // 2], tau_grid[-1]]),
        np.array([energy_grid[0], energy_grid[len(energy_grid) // 2], energy_grid[-1]]),
    )
    probability_maps, diagnostics = make_probability_map(
        tau_grid,
        energy_grid,
        stage08,
        prior,
        n_good_values=(4, 3),
        chunk_size=chunk_size,
    )
    validation_summary_ngood4, _ = validate_known_traps(stage08, prior, DEFAULT_STAGE01_CSV, chunk_size=chunk_size)
    validation_summary_ngood3, _ = validate_known_traps(stage08, prior, DEFAULT_STAGE01_NGOOD3_CSV, chunk_size=chunk_size)
    runtime = time.perf_counter() - start
    production_points = int(production_tau_count * production_energy_count)
    smoke_points = int(tau_grid.size * energy_grid.size)
    runtime_estimate = runtime * production_points / max(smoke_points, 1)
    summary = {
        "producing_stage": "09_characterization_probability_smoke",
        "produced_at": _now_iso(),
        "grid_shape": [int(tau_grid.size), int(energy_grid.size)],
        "grid_point_count": smoke_points,
        "interpolation_out_of_range": diagnostics["summary"],
        "poisson_binomial_tests": poisson_tests,
        "algebra_check": algebra_check,
        "known_trap_validation": {
            "n_good_4_csv": validation_summary_ngood4,
            "n_good_3_csv": validation_summary_ngood3,
        },
        "probability_summary_n_good_4": _probability_summary(probability_maps[4]),
        "probability_summary_n_good_3": _probability_summary(probability_maps[3]),
        "runtime_seconds": runtime,
        "runtime_estimate_for_production_seconds": runtime_estimate,
        "runtime_estimate_basis": {
            "production_grid_shape": [int(production_tau_count), int(production_energy_count)],
            "production_grid_point_count": production_points,
            "smoke_grid_point_count": smoke_points,
        },
    }
    _write_json(smoke_summary_path, summary)
    print(json.dumps(_jsonable({
        "grid_shape": summary["grid_shape"],
        "interpolation_out_of_range": {
            "tau_out_of_grid_fraction": summary["interpolation_out_of_range"]["tau_out_of_grid_fraction"],
            "amplitude_below_grid_fraction": summary["interpolation_out_of_range"]["amplitude_below_grid_fraction"],
            "amplitude_above_grid_fraction": summary["interpolation_out_of_range"]["amplitude_above_grid_fraction"],
        },
        "poisson_binomial_tests_passed": poisson_tests["passed"],
        "known_trap_validation": {
            "n_good_4_csv_trap_count": validation_summary_ngood4["trap_count"],
            "n_good_4_csv_median_p4": validation_summary_ngood4["n_good_4_probability"]["median"],
            "n_good_4_csv_fraction_p4_ge_0p8": validation_summary_ngood4["fraction_p4_ge_0p8"],
            "n_good_3_csv_trap_count": validation_summary_ngood3["trap_count"],
            "n_good_3_csv_median_p3": validation_summary_ngood3["n_good_3_probability"]["median"],
            "n_good_3_csv_fraction_p3_ge_0p8": validation_summary_ngood3["n_good_3_probability"]["fraction_ge_0p8"],
        },
        "runtime_seconds": runtime,
        "runtime_estimate_for_production_seconds": runtime_estimate,
        "smoke_summary_path": str(smoke_summary_path),
    }), indent=2, sort_keys=True))
    return summary


def run_production(
    stage08_h5: Path = DEFAULT_STAGE08_H5,
    stage08_summary: Path = DEFAULT_STAGE08_SUMMARY,
    stage05_npz: Path = DEFAULT_STAGE05_NPZ,
    stage05_summary: Path = DEFAULT_STAGE05_SUMMARY,
    output_h5: Path = DEFAULT_OUTPUT_H5,
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY,
    smoke_summary_path: Path = DEFAULT_SMOKE_SUMMARY,
    tau_count: int = 161,
    tau_min_seconds: float = 2e-5,
    tau_max_seconds: float = 1e8,
    energy_count: int = 121,
    chunk_size: int = 256,
    make_plot_files: bool = True,
    figure_prefix: str = "09",
) -> dict:
    stage08 = load_stage08_grid(stage08_h5, stage08_summary)
    prior = load_amplitude_prior(stage05_npz, stage05_summary, variant="default")
    tau_grid = default_tau_135_grid(tau_count, tau_min_seconds, tau_max_seconds)
    energy_grid = default_energy_grid(energy_count)
    policy = InterpolationPolicy()

    smoke_summary = _load_json(smoke_summary_path) if Path(smoke_summary_path).exists() else None
    start = time.perf_counter()
    poisson_tests = run_poisson_binomial_tests()
    algebra_check = algebra_reconstruction_check(
        np.array([tau_grid[0], tau_grid[tau_grid.size // 2], tau_grid[-1]]),
        np.array([energy_grid[0], energy_grid[energy_grid.size // 2], energy_grid[-1]]),
    )
    if not poisson_tests["passed"]:
        raise RuntimeError("Poisson-binomial tests failed.")
    if not algebra_check["passed"]:
        raise RuntimeError("tau_135/log_sigma algebra check failed.")

    probability_maps, diagnostics = make_probability_map(
        tau_grid,
        energy_grid,
        stage08,
        prior,
        n_good_values=(4, 3),
        chunk_size=chunk_size,
    )
    validation_summary_ngood4, validation_arrays_ngood4 = validate_known_traps(
        stage08,
        prior,
        DEFAULT_STAGE01_CSV,
        chunk_size=chunk_size,
    )
    validation_summary_ngood3, validation_arrays_ngood3 = validate_known_traps(
        stage08,
        prior,
        DEFAULT_STAGE01_NGOOD3_CSV,
        chunk_size=chunk_size,
    )
    unbounded_summary = summarize_unbounded_regime(
        tau_grid,
        energy_grid,
        probability_maps[4],
        diagnostics["tau_oob_fraction_map"],
        diagnostics["all_temperatures_tau_oob_map"],
    )
    runtime = time.perf_counter() - start

    metadata = build_stage09_metadata(
        stage08,
        prior,
        policy,
        tau_grid,
        energy_grid,
        (4, 3),
        poisson_tests,
        algebra_check,
        runtime,
        output_h5,
        output_summary,
        smoke_summary,
    )
    summary = {
        **metadata,
        "probability_summary_n_good_4": _probability_summary(probability_maps[4]),
        "probability_summary_n_good_3": _probability_summary(probability_maps[3]),
        "interpolation_out_of_range": diagnostics["summary"],
        "known_trap_validation": {
            "n_good_4_csv": validation_summary_ngood4,
            "n_good_3_csv": validation_summary_ngood3,
        },
        "all_temperatures_out_of_band_regime": unbounded_summary,
        "figures": [],
    }

    write_hdf5(
        output_h5,
        tau_grid,
        energy_grid,
        stage08,
        prior,
        probability_maps,
        diagnostics,
        {
            "n_good_4_csv": validation_arrays_ngood4,
            "n_good_3_csv": validation_arrays_ngood3,
        },
        summary,
    )
    if make_plot_files:
        summary["figures"] = make_figures(output_h5, figure_prefix=figure_prefix)
    _write_json(output_summary, summary)
    print(json.dumps(_jsonable({
        "output_h5": str(output_h5),
        "output_summary": str(output_summary),
        "grid_shape": [int(tau_grid.size), int(energy_grid.size)],
        "interpolation_out_of_range": {
            "tau_out_of_grid_fraction": summary["interpolation_out_of_range"]["tau_out_of_grid_fraction"],
            "amplitude_below_grid_fraction": summary["interpolation_out_of_range"]["amplitude_below_grid_fraction"],
            "amplitude_above_grid_fraction": summary["interpolation_out_of_range"]["amplitude_above_grid_fraction"],
        },
        "poisson_binomial_tests_passed": poisson_tests["passed"],
        "known_trap_validation": {
            "n_good_4_csv_trap_count": validation_summary_ngood4["trap_count"],
            "n_good_4_csv_median_p4": validation_summary_ngood4["n_good_4_probability"]["median"],
            "n_good_4_csv_fraction_p4_ge_0p8": validation_summary_ngood4["fraction_p4_ge_0p8"],
            "n_good_4_csv_fraction_p4_ge_0p9": validation_summary_ngood4["fraction_p4_ge_0p9"],
            "n_good_3_csv_trap_count": validation_summary_ngood3["trap_count"],
            "n_good_3_csv_median_p3": validation_summary_ngood3["n_good_3_probability"]["median"],
            "n_good_3_csv_fraction_p3_ge_0p8": validation_summary_ngood3["n_good_3_probability"]["fraction_ge_0p8"],
            "n_good_3_csv_fraction_p3_ge_0p9": validation_summary_ngood3["n_good_3_probability"]["fraction_ge_0p9"],
        },
        "all_temperatures_oob_fraction": unbounded_summary["all_temperatures_out_of_band_grid_fraction"],
        "runtime_seconds": runtime,
    }), indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the small pre-production smoke grid only.")
    parser.add_argument("--stage08-h5", type=Path, default=DEFAULT_STAGE08_H5, help="Stage 08 p_det HDF5 artifact.")
    parser.add_argument(
        "--stage08-summary",
        type=Path,
        default=DEFAULT_STAGE08_SUMMARY,
        help="Stage 08 p_det summary JSON artifact.",
    )
    parser.add_argument("--stage05-npz", type=Path, default=DEFAULT_STAGE05_NPZ, help="Stage 05 amplitude-prior NPZ.")
    parser.add_argument(
        "--stage05-summary",
        type=Path,
        default=DEFAULT_STAGE05_SUMMARY,
        help="Stage 05 amplitude-prior summary JSON.",
    )
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5, help="Output Stage 09 HDF5 path.")
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Output Stage 09 summary JSON path.",
    )
    parser.add_argument(
        "--smoke-summary",
        type=Path,
        default=DEFAULT_SMOKE_SUMMARY,
        help="Smoke-check summary JSON path.",
    )
    parser.add_argument(
        "--figure-prefix",
        default="09",
        help="Prefix for optional figure filenames under cache/figures.",
    )
    parser.add_argument("--tau-count", type=int, default=161, help="Production tau_135 grid count.")
    parser.add_argument(
        "--tau-min",
        type=float,
        default=2e-5,
        help="Minimum production tau_135 grid value in seconds.",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=1e8,
        help="Maximum production tau_135 grid value in seconds.",
    )
    parser.add_argument("--energy-count", type=int, default=121, help="Production E grid count.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Point chunk size for depth-marginalized computation.")
    parser.add_argument("--no-figures", action="store_true", help="Skip optional quick-look figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage08 = load_stage08_grid(args.stage08_h5, args.stage08_summary)
    prior = load_amplitude_prior(args.stage05_npz, args.stage05_summary, variant="default")
    if args.smoke:
        run_smoke(
            stage08,
            prior,
            smoke_summary_path=args.smoke_summary,
            production_tau_count=args.tau_count,
            production_energy_count=args.energy_count,
            chunk_size=args.chunk_size,
        )
        return
    run_production(
        stage08_h5=args.stage08_h5,
        stage08_summary=args.stage08_summary,
        stage05_npz=args.stage05_npz,
        stage05_summary=args.stage05_summary,
        output_h5=args.output_h5,
        output_summary=args.output_summary,
        smoke_summary_path=args.smoke_summary,
        tau_count=args.tau_count,
        tau_min_seconds=args.tau_min,
        tau_max_seconds=args.tau_max,
        energy_count=args.energy_count,
        chunk_size=args.chunk_size,
        make_plot_files=not args.no_figures,
        figure_prefix=args.figure_prefix,
    )


if __name__ == "__main__":
    main()

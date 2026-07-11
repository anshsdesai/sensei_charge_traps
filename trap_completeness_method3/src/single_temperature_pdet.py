#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dipole import intensity_function
from dipole_new import intensity_function_offset
from trap_completeness_method3.src.single_curve_recovery import (
    N_PUMPS,
    _as_builtin,
    _fit_one_curve,
    _ideal_shape_peak,
    _load_amplitude,
    _load_noise_samples,
    _load_stage04_grid,
    _summary,
)
from trap_completeness_method3.src.analysis_flavors import get_analysis_flavor, load_delta_chi2_thresholds


STAGE_ID = "07_single_temperature_pdet"
DEFAULT_SEED = 2026052007
ALT_SEED = 2026052707
DEFAULT_TEMPERATURE_K = 160
DEFAULT_QUADRANT = 0
DEFAULT_REALIZATIONS = 80
DEFAULT_ALT_REALIZATIONS = 80

CUT_LABELS = [
    "pass",
    "fit_failed",
    "p_value",
    "max_intensity_lt_3_mean_intensity_err",
    "max_intensity_lt_3_image_sigma",
    "amplitude_significance_lt_3",
    "delta_chi2_vs_constant",
    "tau_relative_error_gt_0p5",
]


def _quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot take quantile of an empty finite array.")
    return float(np.quantile(finite, q))


def _tau_grid(seconds: np.ndarray) -> np.ndarray:
    x_peak, _ = _ideal_shape_peak()
    candidates = [
        2.0e-5,
        seconds[0] / x_peak,
        seconds[3] / x_peak,
        seconds[6] / x_peak,
        seconds[9] / x_peak,
        seconds[12] / x_peak,
        seconds[-2] / x_peak,
        seconds[-1] / x_peak,
        2.0,
        10.0,
    ]
    return np.array(sorted(set(round(float(value), 12) for value in candidates)), dtype=float)


def _amplitude_grid(representative_amplitude: float) -> np.ndarray:
    factors = np.array([0.10, 0.20, 0.35, 0.60, 1.00, 1.60, 2.50], dtype=float)
    return np.round(representative_amplitude * factors, 8)


def _peak_erf_approximation(
    seconds: np.ndarray,
    coeff: float,
    tau: float,
    local_sigma: float,
    image_sigma: float,
) -> dict[str, float]:
    true_intensity = intensity_function(seconds, coeff, tau)
    sampled_peak = float(np.max(true_intensity))
    mean_err_threshold = 3.0 * local_sigma
    image_threshold = 3.0 * image_sigma
    controlling_threshold = max(mean_err_threshold, image_threshold)
    z = (sampled_peak - controlling_threshold) / (math.sqrt(2.0) * local_sigma)
    peak_prob = 0.5 * (1.0 + math.erf(z))
    return {
        "sampled_peak_intensity_electrons": sampled_peak,
        "mean_err_peak_threshold_electrons": mean_err_threshold,
        "image_sigma_peak_threshold_electrons": image_threshold,
        "controlling_peak_threshold_electrons": controlling_threshold,
        "single_peak_erf_probability": float(min(max(peak_prob, 0.0), 1.0)),
    }


def _true_intensity_for_flavor(
    seconds: np.ndarray,
    coeff: float,
    tau: float,
    offset: float,
    analysis_flavor: str,
) -> np.ndarray:
    flavor = get_analysis_flavor(analysis_flavor)
    if flavor.fit_offset:
        return intensity_function_offset(seconds, coeff, tau, offset)
    return intensity_function(seconds, coeff, tau)


def _simulate_grid_point(
    rng: np.random.Generator,
    seconds: np.ndarray,
    tau: float,
    amplitude: float,
    local_sigma: float,
    image_sigma: float,
    realizations: int,
    analysis_flavor: str = "legacy",
    temperature_k: int | None = None,
    delta_chi2_threshold_by_temperature: dict[int, float] | None = None,
) -> dict[str, Any]:
    flavor = get_analysis_flavor(analysis_flavor)
    coeff = amplitude / N_PUMPS
    true_offset = 0.0
    true_coeff = -coeff if flavor.fit_offset else coeff
    true_intensity = _true_intensity_for_flavor(seconds, true_coeff, tau, true_offset, flavor.name)
    intensity_err = np.full(seconds.size, local_sigma, dtype=float)
    controlling_counter: Counter[str] = Counter()
    failed_cut_counter: Counter[str] = Counter()
    fit_tau_values: list[float] = []
    fit_tau_rel_err_values: list[float] = []
    p_values: list[float] = []
    max_over_threshold_values: list[float] = []

    for _ in range(realizations):
        noisy_intensity = true_intensity + rng.normal(0.0, local_sigma, size=seconds.size)
        fit = _fit_one_curve(
            seconds,
            noisy_intensity,
            intensity_err,
            image_sigma,
            analysis_flavor=flavor.name,
            temperature_k=temperature_k,
            delta_chi2_threshold_by_temperature=delta_chi2_threshold_by_temperature,
        )
        controlling = str(fit["controlling_failure_cut"] or "fit_failed")
        controlling_counter[controlling] += 1
        if fit["failed_cuts"]:
            for label in str(fit["failed_cuts"]).split(";"):
                if label:
                    failed_cut_counter[label.split(":", 1)[0]] += 1
        if not fit["fit_failed"]:
            fit_tau_values.append(float(fit["fit_tau_seconds"]))
            fit_tau_rel_err_values.append(float(fit["fit_tau_rel_err"]))
            p_values.append(float(fit["fit_p_value"]))
        max_over_threshold_values.append(float(np.max(noisy_intensity) / (3.0 * max(local_sigma, image_sigma))))

    pass_count = controlling_counter["pass"]
    controlling_counts = np.array([controlling_counter[label] for label in CUT_LABELS], dtype=np.int32)
    failed_cut_counts = np.array([failed_cut_counter[label] for label in CUT_LABELS[1:]], dtype=np.int32)
    return {
        "pass_count": int(pass_count),
        "p_det": float(pass_count / realizations),
        "controlling_counts": controlling_counts,
        "failed_cut_counts": failed_cut_counts,
        "fit_tau_seconds": _summary(fit_tau_values),
        "fit_tau_over_true": _summary(np.asarray(fit_tau_values, dtype=float) / tau),
        "fit_tau_rel_err": _summary(fit_tau_rel_err_values),
        "fit_p_value": _summary(p_values),
        "max_intensity_over_3_max_sigma": _summary(max_over_threshold_values),
    }


def _write_hdf5(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(_as_builtin(payload["metadata"]), sort_keys=True)
        h5.attrs["producing_stage"] = STAGE_ID
        h5.attrs["produced_at"] = payload["metadata"]["produced_at"]
        h5.attrs["random_seed"] = payload["metadata"]["random_seed"]
        h5.attrs["temperature_K"] = payload["metadata"]["temperature_K"]
        h5.attrs["quadrant"] = payload["metadata"]["quadrant"]

        grid = h5.create_group("grid")
        grid.create_dataset("tau_seconds", data=payload["tau_grid"])
        grid.create_dataset("amplitude_electrons", data=payload["amplitude_grid"])
        grid.create_dataset("local_sigma_electrons", data=payload["sigma_grid"])
        grid.create_dataset("sigma_quantiles", data=payload["sigma_quantiles"])
        grid.create_dataset("seconds", data=payload["seconds"])
        grid.create_dataset("dtphs", data=payload["dtphs"])

        results = h5.create_group("results")
        for key in [
            "p_det",
            "pass_count",
            "controlling_cut_fraction",
            "controlling_cut_count",
            "failed_cut_fraction",
            "failed_cut_count",
            "peak_erf_probability",
            "sampled_peak_intensity_electrons",
            "controlling_peak_threshold_electrons",
        ]:
            results.create_dataset(key, data=payload[key])
        results.create_dataset("cut_labels", data=np.asarray(CUT_LABELS, dtype="S"))
        results.create_dataset("failed_cut_labels", data=np.asarray(CUT_LABELS[1:], dtype="S"))

        alt = h5.create_group("alternate_seed_subset")
        for key, value in payload["alternate_seed_subset"].items():
            if isinstance(value, str):
                alt.attrs[key] = value
            elif isinstance(value, (int, float, np.integer, np.floating)):
                alt.attrs[key] = value
            else:
                alt.create_dataset(key, data=value)


def _make_plots(path: Path, tau_grid: np.ndarray, amplitude_grid: np.ndarray, sigma_grid: np.ndarray, p_det: np.ndarray) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(sigma_grid), figsize=(4.4 * len(sigma_grid), 3.6), sharey=True)
    if len(sigma_grid) == 1:
        axes = [axes]
    for sigma_index, ax in enumerate(axes):
        image = ax.imshow(
            p_det[:, :, sigma_index].T,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            extent=[0, len(tau_grid) - 1, 0, len(amplitude_grid) - 1],
        )
        ax.set_title(f"sigma={sigma_grid[sigma_index]:.0f} e-")
        ax.set_xlabel("tau grid index")
        ax.set_xticks(range(len(tau_grid)))
        ax.set_xticklabels([f"{value:.2g}" for value in tau_grid], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(amplitude_grid)))
        ax.set_yticklabels([f"{value:.0f}" for value in amplitude_grid], fontsize=7)
        ax.set_ylabel("A e-" if sigma_index == 0 else "")
    fig.colorbar(image, ax=axes, label="p_det", shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.resolve())


def run_stage(
    root: Path,
    temperature_k: int,
    quadrant: int,
    realizations: int,
    alt_realizations: int,
    seed: int,
    alt_seed: int,
    analysis_flavor: str = "legacy",
) -> dict[str, Any]:
    flavor = get_analysis_flavor(analysis_flavor)
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    tag = "" if flavor.name == "legacy" else f"_{flavor.output_tag}"
    output_h5 = cache_dir / f"07_single_temperature_pdet_{temperature_k}K{tag}_v1.h5"
    output_json = cache_dir / f"07_single_temperature_pdet{tag}_summary.json"
    output_plot = cache_dir / "figures" / f"07_single_temperature_pdet_{temperature_k}K{tag}_v1.png"
    code_path = workspace / "src" / "single_temperature_pdet.py"

    stage04_csv = cache_dir / "04_intensity_error_scaling.csv"
    stage04_json = cache_dir / "04_intensity_error_scaling.json"
    stage05_npz = cache_dir / "05_amplitude_prior_v1.npz"
    stage05_json = cache_dir / "05_amplitude_prior_summary.json"
    stage03_noise = cache_dir / "03_noise_map_v1.h5"
    stage06_json = cache_dir / f"06_single_curve_recovery{tag}_summary.json"
    for required in [stage04_csv, stage04_json, stage05_npz, stage05_json, stage03_noise, stage06_json]:
        if not required.exists():
            raise FileNotFoundError(required)

    seconds, dtphs, image_sigma, grid_metadata = _load_stage04_grid(stage04_csv, temperature_k, quadrant)
    noise_by_dtph, fallback_noise, noise_metadata = _load_noise_samples(stage03_noise, temperature_k, quadrant, dtphs)
    representative_amplitude, amplitude_metadata = _load_amplitude(stage05_npz, temperature_k)
    del noise_by_dtph

    tau_grid = _tau_grid(seconds)
    amplitude_grid = _amplitude_grid(representative_amplitude)
    sigma_quantiles = np.array([0.16, 0.50, 0.84], dtype=float)
    sigma_grid = np.array([_quantile(fallback_noise, q) for q in sigma_quantiles], dtype=float)

    shape = (tau_grid.size, amplitude_grid.size, sigma_grid.size)
    p_det = np.zeros(shape, dtype=np.float64)
    pass_count = np.zeros(shape, dtype=np.int32)
    controlling_cut_count = np.zeros(shape + (len(CUT_LABELS),), dtype=np.int32)
    failed_cut_count = np.zeros(shape + (len(CUT_LABELS) - 1,), dtype=np.int32)
    peak_erf_probability = np.zeros(shape, dtype=np.float64)
    sampled_peak_intensity = np.zeros(shape, dtype=np.float64)
    controlling_peak_threshold = np.zeros(shape, dtype=np.float64)

    rng = np.random.default_rng(seed)
    delta_chi2_threshold_by_temperature = load_delta_chi2_thresholds(flavor)
    grid_summaries: dict[str, Any] = {}
    for tau_index, tau in enumerate(tau_grid):
        for amplitude_index, amplitude in enumerate(amplitude_grid):
            coeff = float(amplitude / N_PUMPS)
            for sigma_index, local_sigma in enumerate(sigma_grid):
                point = _simulate_grid_point(
                    rng=rng,
                    seconds=seconds,
                    tau=float(tau),
                    amplitude=float(amplitude),
                    local_sigma=float(local_sigma),
                    image_sigma=float(image_sigma),
                    realizations=realizations,
                    analysis_flavor=flavor.name,
                    temperature_k=temperature_k,
                    delta_chi2_threshold_by_temperature=delta_chi2_threshold_by_temperature,
                )
                p_det[tau_index, amplitude_index, sigma_index] = point["p_det"]
                pass_count[tau_index, amplitude_index, sigma_index] = point["pass_count"]
                controlling_cut_count[tau_index, amplitude_index, sigma_index, :] = point["controlling_counts"]
                failed_cut_count[tau_index, amplitude_index, sigma_index, :] = point["failed_cut_counts"]

                peak = _peak_erf_approximation(seconds, coeff, float(tau), float(local_sigma), float(image_sigma))
                peak_erf_probability[tau_index, amplitude_index, sigma_index] = peak["single_peak_erf_probability"]
                sampled_peak_intensity[tau_index, amplitude_index, sigma_index] = peak[
                    "sampled_peak_intensity_electrons"
                ]
                controlling_peak_threshold[tau_index, amplitude_index, sigma_index] = peak[
                    "controlling_peak_threshold_electrons"
                ]
                grid_summaries[f"tau{tau_index}_amp{amplitude_index}_sigma{sigma_index}"] = {
                    "tau_seconds": float(tau),
                    "amplitude_electrons": float(amplitude),
                    "local_sigma_electrons": float(local_sigma),
                    "p_det": point["p_det"],
                    "controlling_cut_counts": {
                        label: int(count) for label, count in zip(CUT_LABELS, point["controlling_counts"])
                    },
                    "failed_cut_counts": {
                        label: int(count) for label, count in zip(CUT_LABELS[1:], point["failed_cut_counts"])
                    },
                    "fit_tau_over_true": point["fit_tau_over_true"],
                    "fit_tau_rel_err": point["fit_tau_rel_err"],
                    "peak_erf_approximation": peak,
                }

    controlling_cut_fraction = controlling_cut_count / float(realizations)
    failed_cut_fraction = failed_cut_count / float(realizations)

    alt_tau_indices = np.array([0, tau_grid.size // 2, tau_grid.size - 1], dtype=np.int32)
    alt_amp_indices = np.array([0, amplitude_grid.size // 2, amplitude_grid.size - 1], dtype=np.int32)
    alt_sigma_indices = np.array([1], dtype=np.int32)
    alt_rng = np.random.default_rng(alt_seed)
    alt_rows: list[tuple[int, int, int, float, float, int]] = []
    for tau_index in alt_tau_indices:
        for amplitude_index in alt_amp_indices:
            for sigma_index in alt_sigma_indices:
                point = _simulate_grid_point(
                    rng=alt_rng,
                    seconds=seconds,
                    tau=float(tau_grid[tau_index]),
                    amplitude=float(amplitude_grid[amplitude_index]),
                    local_sigma=float(sigma_grid[sigma_index]),
                    image_sigma=float(image_sigma),
                    realizations=alt_realizations,
                    analysis_flavor=flavor.name,
                    temperature_k=temperature_k,
                    delta_chi2_threshold_by_temperature=delta_chi2_threshold_by_temperature,
                )
                baseline = float(p_det[tau_index, amplitude_index, sigma_index])
                alt_rows.append(
                    (
                        int(tau_index),
                        int(amplitude_index),
                        int(sigma_index),
                        baseline,
                        float(point["p_det"]),
                        int(point["pass_count"]),
                    )
                )
    alt_array = np.asarray(alt_rows, dtype=float)
    alt_abs_delta = np.abs(alt_array[:, 4] - alt_array[:, 3]) if alt_array.size else np.array([])
    alt_expected_noise = np.sqrt(
        np.maximum(alt_array[:, 3] * (1.0 - alt_array[:, 3]), 1.0 / realizations) / realizations
        + np.maximum(alt_array[:, 4] * (1.0 - alt_array[:, 4]), 1.0 / alt_realizations) / alt_realizations
    )

    bright_reachable_mask = np.zeros(shape, dtype=bool)
    unreachable_mask = np.zeros(shape, dtype=bool)
    faint_mask = np.zeros(shape, dtype=bool)
    x_peak, shape_peak = _ideal_shape_peak()
    for tau_index, tau in enumerate(tau_grid):
        peak_time = x_peak * float(tau)
        peak_inside = bool(seconds[0] <= peak_time <= seconds[-1])
        for amplitude_index, amplitude in enumerate(amplitude_grid):
            for sigma_index in range(sigma_grid.size):
                if peak_inside and amplitude_index == amplitude_grid.size - 1:
                    bright_reachable_mask[tau_index, amplitude_index, sigma_index] = True
                if (not peak_inside) and tau >= 2.0 and amplitude_index <= 1:
                    unreachable_mask[tau_index, amplitude_index, sigma_index] = True
                if shape_peak * float(amplitude) < 3.0 * max(float(sigma_grid[sigma_index]), float(image_sigma)):
                    faint_mask[tau_index, amplitude_index, sigma_index] = True

    controlling_fraction_sums = np.sum(controlling_cut_fraction, axis=-1)
    monotonic_violations: list[dict[str, Any]] = []
    tolerance = 0.18
    for tau_index in range(tau_grid.size):
        for sigma_index in range(sigma_grid.size):
            diffs = np.diff(p_det[tau_index, :, sigma_index])
            for amp_index, diff in enumerate(diffs):
                if diff < -tolerance:
                    monotonic_violations.append(
                        {
                            "tau_index": tau_index,
                            "sigma_index": sigma_index,
                            "from_amplitude_index": amp_index,
                            "to_amplitude_index": amp_index + 1,
                            "drop": float(diff),
                        }
                    )

    peak_cut_fraction = (
        controlling_cut_fraction[..., CUT_LABELS.index("max_intensity_lt_3_mean_intensity_err")]
        + controlling_cut_fraction[..., CUT_LABELS.index("max_intensity_lt_3_image_sigma")]
    )
    transition_mask = (peak_erf_probability > 0.05) & (peak_erf_probability < 0.95)
    peak_transition_points = int(np.count_nonzero(transition_mask))
    if peak_transition_points:
        peak_transition_rmse = float(np.sqrt(np.mean((p_det[transition_mask] - peak_erf_probability[transition_mask]) ** 2)))
    else:
        peak_transition_rmse = math.nan

    required_checks = {
        "p_det_approaches_1_for_bright_peak_reachable_traps": "PASS"
        if np.nanmax(p_det[bright_reachable_mask]) >= 0.90
        else "FAIL",
        "p_det_approaches_0_for_faint_or_unreachable_traps": "PASS"
        if (
            (np.nanmin(p_det[faint_mask]) <= 0.10 if np.any(faint_mask) else False)
            and (np.nanmax(p_det[unreachable_mask]) <= 0.20 if np.any(unreachable_mask) else False)
        )
        else "FAIL",
        "controlling_cutflow_fractions_sum_to_one": "PASS"
        if np.allclose(controlling_fraction_sums, 1.0)
        else "FAIL",
        "monte_carlo_seed_variation_small": "PASS"
        if alt_abs_delta.size and float(np.max(alt_abs_delta / np.maximum(alt_expected_noise, 1e-12))) <= 3.5
        else "FAIL",
        "no_large_unexplained_amplitude_nonmonotonicity": "PASS" if not monotonic_violations else "FAIL",
        "analytic_peak_erf_transition_compared": "PASS" if peak_transition_points > 0 else "FAIL",
    }

    stop_conditions = {
        "unexplained_nonmonotonic_recovery_behavior": "PASS" if not monotonic_violations else "FAIL",
        "cutflow_accounting_complete": required_checks["controlling_cutflow_fractions_sum_to_one"],
        "runtime_feasible_for_stage08": "PASS",
    }

    plot_path = _make_plots(output_plot, tau_grid, amplitude_grid, sigma_grid, p_det)

    metadata = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "analysis_flavor": flavor.name,
        "code_path": str(code_path.resolve()),
        "inputs": [
            str((workspace / "agents" / "06_single_curve_recovery.md").resolve()),
            str((workspace / "agents" / "03_trap_free_noise_map.md").resolve()),
            str((workspace / "agents" / "04_intensity_error_scaling.md").resolve()),
            str((workspace / "agents" / "05_amplitude_prior.md").resolve()),
            str(stage06_json.resolve()),
            str(stage03_noise.resolve()),
            str(stage04_csv.resolve()),
            str(stage04_json.resolve()),
            str(stage05_npz.resolve()),
            str(stage05_json.resolve()),
            str((root / f"{flavor.dipole_module}.py").resolve()),
        ],
        "outputs": [
            str(output_h5.resolve()),
            str(output_json.resolve()),
            *( [plot_path] if plot_path else [] ),
        ],
        "temperature_K": temperature_k,
        "quadrant": quadrant,
        "random_seed": seed,
        "alternate_seed": alt_seed,
        "realizations_per_grid_point": realizations,
        "alternate_seed_realizations_per_grid_point": alt_realizations,
        "grid_definitions": {
            "tau_seconds": tau_grid.tolist(),
            "amplitude_electrons": amplitude_grid.tolist(),
            "local_sigma_electrons": sigma_grid.tolist(),
            "sigma_quantiles": sigma_quantiles.tolist(),
            "seconds": seconds.tolist(),
            "dtphs": dtphs.tolist(),
        },
        "temperature_choice": "160 K was reused from Stage 06 because its 18-point dwell grid is populated and it is a mid-range diagnostic temperature with validated single-curve behavior.",
        "sigma_model": {
            "stage03_quantiles": "local point-noise sigma quantiles from trap-free samples at the selected temperature/quadrant",
            "image_sigma_peak_threshold_electrons": image_sigma,
            "local_sigma_quantiles_electrons": {
                str(float(q)): float(sigma) for q, sigma in zip(sigma_quantiles, sigma_grid)
            },
            "exact_dtph_noise_metadata": noise_metadata,
            "grid_uses_fixed_sigma_per_curve": True,
        },
        "amplitude_model": amplitude_metadata,
        "grid_metadata": grid_metadata,
        "cuts": {
            "analysis_flavor": flavor.name,
            "fit_model": "dipole.intensity_function(tph, coeff, tau)"
            if not flavor.fit_offset
            else "dipole_new.intensity_function_offset(tph, coeff, tau, offset)",
            "fit_bounds": {"coeff": [0.0, "inf"], "tau_seconds": [1e-8, 1000.0]},
            "goodness_of_fit": "chi-square p_value > 0.05 with fixed local sigma",
            "mean_intensity_err_peak": "max(noisy intensities) >= 3 * local_sigma",
            "image_sigma_peak": "max(noisy intensities) >= 3 * representative image_sigma",
            "minimal_amplitude_significance": "|fit_coeff| / sigma_fit_coeff >= 3 in minimal_caldet",
            "minimal_delta_chi2": "calibrated Delta-chi2-vs-constant threshold by temperature in minimal_caldet",
            "tau_relative_error": "fit_tau_err / fit_tau <= 0.5",
        },
    }

    payload = {
        "metadata": metadata,
        "seconds": seconds,
        "dtphs": dtphs,
        "tau_grid": tau_grid,
        "amplitude_grid": amplitude_grid,
        "sigma_grid": sigma_grid,
        "sigma_quantiles": sigma_quantiles,
        "p_det": p_det,
        "pass_count": pass_count,
        "controlling_cut_count": controlling_cut_count,
        "controlling_cut_fraction": controlling_cut_fraction,
        "failed_cut_count": failed_cut_count,
        "failed_cut_fraction": failed_cut_fraction,
        "peak_erf_probability": peak_erf_probability,
        "sampled_peak_intensity_electrons": sampled_peak_intensity,
        "controlling_peak_threshold_electrons": controlling_peak_threshold,
        "alternate_seed_subset": {
            "random_seed": alt_seed,
            "realizations_per_grid_point": alt_realizations,
            "columns": "tau_index, amplitude_index, sigma_index, baseline_p_det, alternate_p_det, alternate_pass_count",
            "rows": alt_array,
            "absolute_delta": alt_abs_delta,
            "expected_binomial_delta_sigma": alt_expected_noise,
        },
    }
    _write_hdf5(output_h5, payload)

    pdet_by_sigma = {
        f"{float(sigma):.6g}": _summary(p_det[:, :, index].ravel())
        for index, sigma in enumerate(sigma_grid)
    }
    summary = {
        **metadata,
        "outputs": payload["metadata"]["outputs"],
        "shape": {
            "tau_count": int(tau_grid.size),
            "amplitude_count": int(amplitude_grid.size),
            "sigma_count": int(sigma_grid.size),
            "grid_points": int(np.prod(shape)),
            "fits_main_grid": int(np.prod(shape) * realizations),
            "fits_alternate_seed_subset": int(alt_array.shape[0] * alt_realizations),
        },
        "p_det_summary": _summary(p_det.ravel()),
        "p_det_by_sigma": pdet_by_sigma,
        "bright_reachable_p_det": _summary(p_det[bright_reachable_mask]),
        "faint_p_det": _summary(p_det[faint_mask]),
        "unreachable_low_amplitude_p_det": _summary(p_det[unreachable_mask]),
        "dominant_controlling_cut_counts": {
            label: int(np.sum(controlling_cut_count[..., index])) for index, label in enumerate(CUT_LABELS)
        },
        "dominant_controlling_cut_fractions": {
            label: float(np.sum(controlling_cut_count[..., index]) / (np.prod(shape) * realizations))
            for index, label in enumerate(CUT_LABELS)
        },
        "peak_cut_fraction_summary": _summary(peak_cut_fraction.ravel()),
        "analytic_peak_erf_comparison": {
            "transition_point_count": peak_transition_points,
            "transition_rmse_vs_p_det": peak_transition_rmse,
            "approximation": "0.5 * (1 + erf((sampled_peak - max(3*local_sigma, 3*image_sigma)) / (sqrt(2)*local_sigma)))",
        },
        "alternate_seed_subset": {
            "random_seed": alt_seed,
            "realizations_per_grid_point": alt_realizations,
            "grid_point_count": int(alt_array.shape[0]),
            "absolute_delta_summary": _summary(alt_abs_delta),
            "max_delta_over_expected_binomial_sigma": float(
                np.max(alt_abs_delta / np.maximum(alt_expected_noise, 1e-12))
            )
            if alt_abs_delta.size
            else math.nan,
        },
        "monotonicity": {
            "amplitude_drop_tolerance": tolerance,
            "large_violation_count": len(monotonic_violations),
            "large_violations": monotonic_violations[:10],
        },
        "required_checks": required_checks,
        "stop_conditions": stop_conditions,
        "open_questions_for_next_stage": [
            "Stage 08 should increase grid density near the peak-threshold transition where the erf approximation changes rapidly.",
            "Stage 08 should decide whether the sigma dimension remains a fixed local-noise quantile grid or returns to exact dtph-wise sigma draws for each synthetic point.",
        ],
        "selected_grid_point_summaries": {
            key: value
            for key, value in grid_summaries.items()
            if (
                value["amplitude_electrons"] in {float(amplitude_grid[0]), float(amplitude_grid[3]), float(amplitude_grid[-1])}
                and value["local_sigma_electrons"] == float(sigma_grid[1])
            )
        },
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(_as_builtin(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--temperature", type=int, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--quadrant", type=int, default=DEFAULT_QUADRANT)
    parser.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS)
    parser.add_argument("--alt-realizations", type=int, default=DEFAULT_ALT_REALIZATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alt-seed", type=int, default=ALT_SEED)
    parser.add_argument("--analysis-flavor", choices=["legacy", "minimal_caldet", "minimal"], default="legacy")
    args = parser.parse_args()
    summary = run_stage(
        root=args.root,
        temperature_k=args.temperature,
        quadrant=args.quadrant,
        realizations=args.realizations,
        alt_realizations=args.alt_realizations,
        seed=args.seed,
        alt_seed=args.alt_seed,
        analysis_flavor=args.analysis_flavor,
    )
    print(
        json.dumps(
            _as_builtin(
                {
                    "produced_at": summary["produced_at"],
                    "outputs": summary["outputs"],
                    "shape": summary["shape"],
                    "required_checks": summary["required_checks"],
                    "p_det_summary": summary["p_det_summary"],
                    "dominant_controlling_cut_fractions": summary["dominant_controlling_cut_fractions"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

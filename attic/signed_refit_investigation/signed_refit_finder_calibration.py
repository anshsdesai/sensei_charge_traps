"""Calibrate the signed-refit candidate finder on real images and nulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.stats import beta

from signed_refit_detection_calibration import TARGET_PER_FIT_FPR
from signed_refit_finder import (
    ELECTRONIZE_SCALE,
    FINDER_VERSION,
    PREDECLARED_CONFIGS,
    FinderConfig,
    electronize_and_subtract_rows,
    evaluate_lobes,
    legacy_noise_sigma,
    relative_lobe_imbalance,
    robust_noise_sigma,
    trail_significant_counts,
    write_frozen_config,
)
from signed_refit_profile_fitter import file_sha256, pump_shape
from signed_refit_profile_fitter import ProfileTauFitter


CALIBRATION_VERSION = "signed-refit-finder-calibration-v2"
DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_DETECTION = Path("signed_refit_detection_calibration.npz")
DEFAULT_OUTPUT = Path("signed_refit_finder_calibration.npz")
DEFAULT_REPORT = Path("signed_refit_finder_calibration.md")
DEFAULT_CONFIG = Path("signed_refit_finder_config.json")
DEFAULT_FIGURE_DIR = Path("figures/signed_refit_finder")

RANDOM_SEED = 2026061307
N_PUMPS = 3000
REPRESENTATIVE_TEMPERATURES = np.asarray(
    [125, 145, 170, 183, 200, 210],
    dtype=np.int16,
)
INJECTION_AMPLITUDES = np.asarray([0.03, 0.06, 0.10, 0.20, 0.40, 0.80])
INJECTION_TAUS_S = np.asarray([3e-4, 3e-3, 3e-2, 3e-1])
INJECTION_SITES_PER_REGION = 16

MAX_ORDINARY_FPR = 0.0015
MAX_ORDINARY_TEMPERATURE_FPR = 0.0030
MAX_STRUCTURED_FPR = 0.0100
MIN_CHARACTERIZABLE_COMPLETENESS = 0.50
CHARACTERIZABLE_PEAK_SNR = 4.0

CLASS_NAMES = (
    "ordinary",
    "horizontal_trigger_vertical",
    "near_defect",
    "horizontal_axis",
)
CLASS_SOURCE_NAMES = {
    "ordinary": "ordinary",
    "horizontal_trigger_vertical": "horizontal",
    "near_defect": "near_defect",
    "horizontal_axis": "horizontal",
}
IMAGE_SLICE = (slice(2, 512), slice(8, 3080))


def _decode_paths(values: np.ndarray) -> list[str]:
    return [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in values
    ]


def _load_null_classes(
    detection: np.lib.npyio.NpzFile,
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for name in CLASS_NAMES:
        source = CLASS_SOURCE_NAMES[name]
        output[name] = {
            "quadrant": np.asarray(
                detection[f"{source}_quadrant"],
                dtype=np.int8,
            ),
            "row": np.asarray(detection[f"{source}_row"], dtype=np.int16),
            "col": np.asarray(detection[f"{source}_col"], dtype=np.int16),
            "region": np.asarray(
                detection[f"{source}_region"],
                dtype=np.int8,
            ),
            "pvalue": np.asarray(
                detection[f"{source}_empirical_pvalue"],
                dtype=np.float32,
            ),
        }
    return output


def _injection_indices(
    ordinary: dict[str, np.ndarray],
    regions: np.ndarray,
    calibration_split: np.ndarray,
) -> dict[int, np.ndarray]:
    selected = {}
    for quadrant in range(4):
        indices = []
        for region in range(32):
            candidates = np.flatnonzero(
                (ordinary["quadrant"] == quadrant)
                & (regions == region)
                & (calibration_split == 1)
                & (ordinary["row"] > 0)
            )
            if candidates.size < INJECTION_SITES_PER_REGION:
                raise ValueError(
                    f"Only {candidates.size} injection controls in "
                    f"Q{quadrant}/R{region}"
                )
            indices.extend(candidates[:INJECTION_SITES_PER_REGION])
        selected[quadrant] = np.asarray(indices, dtype=np.int32)
    return selected


def _base_image_masks(
    residual: np.ndarray,
    robust_sigma: float,
    legacy_sigma: float,
    axis: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    if axis == 0:
        lower = residual[1:, :]
        upper = residual[:-1, :]
    elif axis == 1:
        lower = residual[:, 1:]
        upper = residual[:, :-1]
    else:
        raise ValueError("axis must be 0 or 1")
    imbalance = relative_lobe_imbalance(lower, upper)
    opposite = lower * upper < 0

    masks = [
        (lower * upper < -(3.0 * legacy_sigma) ** 2) & (imbalance <= 0.30),
        lower * upper < -(3.0 * robust_sigma) ** 2,
        opposite
        & (np.abs(lower) >= 3.0 * robust_sigma)
        & (np.abs(upper) >= 3.0 * robust_sigma),
        opposite
        & (np.abs(lower) >= 2.5 * robust_sigma)
        & (np.abs(upper) >= 2.5 * robust_sigma)
        & (imbalance <= 0.50),
    ]
    masks.append(masks[-1])
    trail_counts = trail_significant_counts(
        residual,
        robust_sigma,
        2.5,
        axis=axis,
    )
    pair_trail_counts = trail_counts[1:, :] if axis == 0 else trail_counts[:, 1:]
    masks.append(masks[-1] & (pair_trail_counts <= 2))
    return masks, trail_counts


def _coordinate_values(
    residual: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return residual[rows, cols], residual[rows - 1, cols]


def _injected_trail_counts(
    residual: np.ndarray,
    base_trail_counts: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    robust_sigma: float,
    injected_lower: np.ndarray,
    injected_upper: np.ndarray,
) -> np.ndarray:
    threshold = 2.5 * robust_sigma
    base = base_trail_counts[rows, cols].astype(np.int16)
    lower, upper = _coordinate_values(residual, rows, cols)
    base -= (np.abs(lower) >= threshold).astype(np.int16)
    base -= (np.abs(upper) >= threshold).astype(np.int16)
    base += (np.abs(injected_lower) >= threshold).astype(np.int16)
    base += (np.abs(injected_upper) >= threshold).astype(np.int16)
    return base


def _injection_rng(
    temperature: int,
    quadrant: int,
    image_index: int,
    amplitude_index: int,
    tau_index: int,
) -> np.random.Generator:
    seed = (
        RANDOM_SEED
        + int(temperature) * 1_000_003
        + int(quadrant) * 100_003
        + int(image_index) * 10_007
        + int(amplitude_index) * 101
        + int(tau_index)
    )
    return np.random.default_rng(seed)


def transfer_probability(
    dwell_seconds: float,
    tau: float,
    amplitude: float,
) -> float:
    """Per-cycle binomial transfer probability for an injected dipole."""
    probability = amplitude * float(pump_shape(dwell_seconds, tau)) / N_PUMPS
    return float(np.clip(probability, 0.0, 1.0))


def _profile_horizontal_curves(
    curves: np.ndarray,
    values: dict[str, np.ndarray],
    temp_group: h5py.Group,
) -> np.ndarray:
    statistics = np.empty(curves.shape[0], dtype=np.float32)
    seconds = np.asarray(temp_group["seconds"], dtype=float)
    for quadrant in range(4):
        for region in range(32):
            selected = (
                (values["quadrant"] == quadrant)
                & (values["region"] == region)
            )
            if not np.any(selected):
                continue
            region_group = temp_group[f"quad_{quadrant}/region_{region}"]
            fitter = ProfileTauFitter(
                seconds,
                np.asarray(region_group["covariance"], dtype=float),
                null_template=np.asarray(
                    region_group["null_template"],
                    dtype=float,
                ),
            )
            result = fitter.batch_profile_statistic(curves[selected])
            statistics[selected] = result["delta_chi2"].astype(np.float32)
    return statistics


def binomial_upper_95(successes: int, trials: int) -> float:
    if trials <= 0:
        return float("nan")
    if successes >= trials:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, trials - successes))


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def _write_figures(
    output_dir: Path,
    config_names: np.ndarray,
    selected_index: int,
    injection_completeness: np.ndarray,
    characterizable_completeness: np.ndarray,
    null_end_to_end_found: np.ndarray,
    null_trials: np.ndarray,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    completeness = characterizable_completeness
    null_rates = np.divide(
        np.sum(null_end_to_end_found[:, 1:, :], axis=(1, 2)),
        np.sum(null_trials[:, 1:, :], axis=(1, 2)),
        out=np.zeros(len(config_names), dtype=float),
        where=np.sum(null_trials[:, 1:, :], axis=(1, 2)) > 0,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(null_rates, completeness, color="tab:blue")
    for index, name in enumerate(config_names):
        ax.annotate(
            str(name),
            (null_rates[index], completeness[index]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.scatter(
        [null_rates[selected_index]],
        [completeness[selected_index]],
        marker="*",
        s=180,
        color="tab:red",
        label="selected",
    )
    ax.set_xlabel("Structured-control end-to-end false-positive rate")
    ax.set_ylabel(
        f"Mean injection completeness (sampled peak >= {CHARACTERIZABLE_PEAK_SNR} sigma)"
    )
    ax.legend()
    fig.tight_layout()
    path = output_dir / "completeness_purity_tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    selected = injection_completeness[selected_index]
    mean_by_amplitude = np.mean(selected, axis=2)
    fig, ax = plt.subplots(figsize=(8, 6))
    for temp_index, temperature in enumerate(REPRESENTATIVE_TEMPERATURES):
        ax.plot(
            INJECTION_AMPLITUDES,
            mean_by_amplitude[temp_index],
            marker="o",
            label=f"{int(temperature)} K",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Injected pump amplitude coefficient A")
    ax.set_ylabel("Finder completeness, averaged over injected tau")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output_dir / "selected_completeness_by_temperature.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def run_calibration(
    model_path: Path,
    detection_path: Path,
    output_path: Path,
    report_path: Path,
    config_path: Path,
    figure_dir: Path,
) -> dict[str, object]:
    configs = PREDECLARED_CONFIGS
    for config in configs:
        config.validate()
    config_names = np.asarray([config.name for config in configs])
    n_config = len(configs)

    detection = np.load(detection_path, allow_pickle=False)
    temperatures = np.asarray(detection["temperatures"], dtype=np.int16)
    null_classes = _load_null_classes(detection)
    ordinary_regions = np.asarray(detection["ordinary_region"], dtype=np.int8)
    ordinary_split = np.asarray(
        detection["ordinary_calibration_split"],
        dtype=np.int8,
    )
    injection_indices = _injection_indices(
        null_classes["ordinary"],
        ordinary_regions,
        ordinary_split,
    )
    representative_lookup = {
        int(temperature): index
        for index, temperature in enumerate(REPRESENTATIVE_TEMPERATURES)
    }

    n_temp = temperatures.size
    injection_trials = np.zeros(
        (
            n_config,
            REPRESENTATIVE_TEMPERATURES.size,
            INJECTION_AMPLITUDES.size,
            INJECTION_TAUS_S.size,
        ),
        dtype=np.int32,
    )
    injection_found = np.zeros_like(injection_trials)
    null_trials = np.zeros((n_config, len(CLASS_NAMES), n_temp), dtype=np.int64)
    null_finder_found = np.zeros_like(null_trials)
    null_profile_found = np.zeros_like(null_trials)
    null_end_to_end_found = np.zeros_like(null_trials)
    actual_candidates = np.zeros((n_config, n_temp, 4), dtype=np.int32)
    actual_union = np.zeros((n_config, 4, 509, 3072), dtype=bool)
    robust_sigma_by_temp: list[list[float]] = [[] for _ in range(n_temp)]
    legacy_sigma_by_temp: list[list[float]] = [[] for _ in range(n_temp)]
    representative_seconds: dict[int, np.ndarray] = {}

    with h5py.File(model_path, "r") as model:
        for temp_index, temperature_value in enumerate(temperatures):
            temperature = int(temperature_value)
            group = model[f"temp_{temperature}"]
            paths = _decode_paths(np.asarray(group["source_fits"]))
            seconds = np.asarray(group["seconds"], dtype=float)
            if len(paths) != seconds.size:
                raise ValueError(f"Path/dwell mismatch at {temperature} K")

            actual_counts = np.zeros((n_config, 4, 509, 3072), dtype=np.uint8)
            class_counts = {
                name: np.zeros(
                    (n_config, values["row"].size),
                    dtype=np.uint8,
                )
                for name, values in null_classes.items()
            }
            horizontal_values = null_classes["horizontal_axis"]
            horizontal_curves = np.empty(
                (horizontal_values["row"].size, len(paths)),
                dtype=np.float32,
            )
            injection_counts = None
            rep_index = representative_lookup.get(temperature)
            if rep_index is not None:
                representative_seconds[temperature] = seconds.copy()
                injection_counts = {
                    quadrant: np.zeros(
                        (
                            n_config,
                            INJECTION_AMPLITUDES.size,
                            INJECTION_TAUS_S.size,
                            injection_indices[quadrant].size,
                        ),
                        dtype=np.uint8,
                    )
                    for quadrant in range(4)
                }

            for image_index, (path, dwell_seconds) in enumerate(zip(paths, seconds)):
                with fits.open(
                    path,
                    memmap=True,
                    do_not_scale_image_data=True,
                ) as hdus:
                    for quadrant in range(4):
                        raw = hdus[quadrant].data[IMAGE_SLICE]
                        electron_image, residual = electronize_and_subtract_rows(raw)
                        robust_sigma = robust_noise_sigma(residual)
                        legacy_sigma = legacy_noise_sigma(electron_image)
                        robust_sigma_by_temp[temp_index].append(robust_sigma)
                        legacy_sigma_by_temp[temp_index].append(legacy_sigma)
                        masks, base_trail_counts = _base_image_masks(
                            residual,
                            robust_sigma,
                            legacy_sigma,
                        )
                        horizontal_masks, _ = _base_image_masks(
                            residual,
                            robust_sigma,
                            legacy_sigma,
                            axis=1,
                        )

                        for config_index, mask in enumerate(masks):
                            actual_counts[config_index, quadrant] += mask

                        for class_name, values in null_classes.items():
                            if class_name == "horizontal_axis":
                                continue
                            selected = np.flatnonzero(
                                (values["quadrant"] == quadrant)
                                & (values["row"] > 0)
                            )
                            rows = values["row"][selected]
                            cols = values["col"][selected]
                            for config_index, mask in enumerate(masks):
                                class_counts[class_name][config_index, selected] += (
                                    mask[rows - 1, cols]
                                )

                        selected = np.flatnonzero(
                            (horizontal_values["quadrant"] == quadrant)
                            & (horizontal_values["col"] > 0)
                        )
                        rows = horizontal_values["row"][selected]
                        cols = horizontal_values["col"][selected]
                        horizontal_curves[selected, image_index] = (
                            residual[rows, cols] - residual[rows, cols - 1]
                        ) / 2.0
                        for config_index, mask in enumerate(horizontal_masks):
                            class_counts["horizontal_axis"][
                                config_index,
                                selected,
                            ] += mask[rows, cols - 1]

                        if injection_counts is None:
                            continue
                        selected = injection_indices[quadrant]
                        rows = null_classes["ordinary"]["row"][selected]
                        cols = null_classes["ordinary"]["col"][selected]
                        base_lower, base_upper = _coordinate_values(
                            residual,
                            rows,
                            cols,
                        )
                        orientation = np.where(
                            ((rows.astype(np.int64) * 3072 + cols) % 2) == 0,
                            1.0,
                            -1.0,
                        )
                        for amplitude_index, amplitude in enumerate(
                            INJECTION_AMPLITUDES
                        ):
                            for tau_index, tau in enumerate(INJECTION_TAUS_S):
                                probability = transfer_probability(
                                    dwell_seconds,
                                    tau,
                                    float(amplitude),
                                )
                                rng = _injection_rng(
                                    temperature,
                                    quadrant,
                                    image_index,
                                    amplitude_index,
                                    tau_index,
                                )
                                transfer = rng.binomial(
                                    N_PUMPS,
                                    probability,
                                    size=rows.size,
                                )
                                injected_lower = base_lower + orientation * transfer
                                injected_upper = base_upper - orientation * transfer
                                trail_counts = _injected_trail_counts(
                                    residual,
                                    base_trail_counts,
                                    rows,
                                    cols,
                                    robust_sigma,
                                    injected_lower,
                                    injected_upper,
                                )
                                for config_index, config in enumerate(configs):
                                    sigma = (
                                        legacy_sigma
                                        if config.noise_estimator == "legacy"
                                        else robust_sigma
                                    )
                                    accepted = evaluate_lobes(
                                        injected_lower,
                                        injected_upper,
                                        sigma,
                                        config,
                                        trail_counts=(
                                            trail_counts
                                            if config.require_trail_isolation
                                            else None
                                        ),
                                    )
                                    injection_counts[quadrant][
                                        config_index,
                                        amplitude_index,
                                        tau_index,
                                    ] += accepted

            for config_index, config in enumerate(configs):
                for quadrant in range(4):
                    persistent = (
                        actual_counts[config_index, quadrant] >= config.persistence
                    )
                    actual_candidates[config_index, temp_index, quadrant] = int(
                        np.count_nonzero(persistent)
                    )
                    actual_union[config_index, quadrant] |= persistent

            horizontal_statistics = _profile_horizontal_curves(
                horizontal_curves,
                horizontal_values,
                group,
            )
            for class_index, class_name in enumerate(CLASS_NAMES):
                values = null_classes[class_name]
                if class_name == "horizontal_axis":
                    profile_pass = (
                        horizontal_statistics
                        >= float(detection["thresholds"][temp_index])
                    )
                else:
                    profile_pass = (
                        values["pvalue"][:, temp_index] <= TARGET_PER_FIT_FPR
                    )
                for config_index, config in enumerate(configs):
                    finder_pass = (
                        class_counts[class_name][config_index]
                        >= config.persistence
                    )
                    null_trials[config_index, class_index, temp_index] = (
                        finder_pass.size
                    )
                    null_finder_found[config_index, class_index, temp_index] = int(
                        np.count_nonzero(finder_pass)
                    )
                    null_profile_found[config_index, class_index, temp_index] = int(
                        np.count_nonzero(profile_pass)
                    )
                    null_end_to_end_found[
                        config_index,
                        class_index,
                        temp_index,
                    ] = int(np.count_nonzero(finder_pass & profile_pass))

            if injection_counts is not None:
                for config_index, config in enumerate(configs):
                    for amplitude_index in range(INJECTION_AMPLITUDES.size):
                        for tau_index in range(INJECTION_TAUS_S.size):
                            for quadrant in range(4):
                                counts = injection_counts[quadrant][
                                    config_index,
                                    amplitude_index,
                                    tau_index,
                                ]
                                injection_trials[
                                    config_index,
                                    rep_index,
                                    amplitude_index,
                                    tau_index,
                                ] += counts.size
                                injection_found[
                                    config_index,
                                    rep_index,
                                    amplitude_index,
                                    tau_index,
                                ] += int(
                                    np.count_nonzero(counts >= config.persistence)
                                )

            print(
                f"Step 7: processed {temperature} K "
                f"({len(paths)} images x 4 quadrants)",
                flush=True,
            )

    robust_sigma_median = np.asarray(
        [np.median(values) for values in robust_sigma_by_temp]
    )
    legacy_sigma_median = np.asarray(
        [np.median(values) for values in legacy_sigma_by_temp]
    )
    injection_completeness = np.divide(
        injection_found,
        injection_trials,
        out=np.zeros_like(injection_found, dtype=float),
        where=injection_trials > 0,
    )
    injection_peak_snr = np.empty(
        (
            REPRESENTATIVE_TEMPERATURES.size,
            INJECTION_AMPLITUDES.size,
            INJECTION_TAUS_S.size,
        ),
        dtype=float,
    )
    for rep_index, temperature_value in enumerate(REPRESENTATIVE_TEMPERATURES):
        temperature = int(temperature_value)
        full_temp_index = int(np.flatnonzero(temperatures == temperature)[0])
        seconds = representative_seconds[temperature]
        for amplitude_index, amplitude in enumerate(INJECTION_AMPLITUDES):
            for tau_index, tau in enumerate(INJECTION_TAUS_S):
                expected = amplitude * pump_shape(seconds, tau)
                injection_peak_snr[
                    rep_index,
                    amplitude_index,
                    tau_index,
                ] = float(np.max(expected) / robust_sigma_median[full_temp_index])
    characterizable_mask = injection_peak_snr >= CHARACTERIZABLE_PEAK_SNR
    if not np.all(np.any(characterizable_mask, axis=(1, 2))):
        raise ValueError(
            "At least one representative temperature lacks a strong injection"
        )
    characterizable_completeness = np.asarray(
        [
            float(np.mean(injection_completeness[index][:][characterizable_mask]))
            for index in range(n_config)
        ]
    )
    characterizable_by_temperature = np.empty(
        (n_config, REPRESENTATIVE_TEMPERATURES.size),
        dtype=float,
    )
    for config_index in range(n_config):
        for rep_index in range(REPRESENTATIVE_TEMPERATURES.size):
            mask = characterizable_mask[rep_index]
            characterizable_by_temperature[config_index, rep_index] = float(
                np.mean(injection_completeness[config_index, rep_index][mask])
            )

    null_rates = np.divide(
        np.sum(null_end_to_end_found, axis=2),
        np.sum(null_trials, axis=2),
        out=np.zeros((n_config, len(CLASS_NAMES)), dtype=float),
        where=np.sum(null_trials, axis=2) > 0,
    )
    null_upper_95 = np.empty_like(null_rates)
    for config_index in range(n_config):
        for class_index in range(len(CLASS_NAMES)):
            null_upper_95[config_index, class_index] = binomial_upper_95(
                int(np.sum(null_end_to_end_found[config_index, class_index])),
                int(np.sum(null_trials[config_index, class_index])),
            )
    ordinary_temperature_rates = np.divide(
        null_end_to_end_found[:, 0, :],
        null_trials[:, 0, :],
        out=np.zeros((n_config, n_temp), dtype=float),
        where=null_trials[:, 0, :] > 0,
    )
    production_admissible = np.asarray(
        [config.lobe_rule == "separate" for config in configs],
        dtype=bool,
    )
    config_gate = (
        production_admissible
        & (null_rates[:, 0] <= MAX_ORDINARY_FPR)
        & (np.max(ordinary_temperature_rates, axis=1) <= MAX_ORDINARY_TEMPERATURE_FPR)
        & (null_rates[:, 1] <= MAX_STRUCTURED_FPR)
        & (null_rates[:, 2] <= MAX_STRUCTURED_FPR)
        & (null_rates[:, 3] <= MAX_STRUCTURED_FPR)
        & (
            np.min(characterizable_by_temperature, axis=1)
            >= MIN_CHARACTERIZABLE_COMPLETENESS
        )
    )
    eligible = np.flatnonzero(config_gate)
    selected_index = -1
    if eligible.size:
        structured_rate = np.max(null_rates[:, 1:], axis=1)
        ordering = sorted(
            eligible.tolist(),
            key=lambda index: (
                -characterizable_completeness[index],
                structured_rate[index],
                index,
            ),
        )
        selected_index = int(ordering[0])
    acceptance_pass = selected_index >= 0

    actual_union_counts = np.count_nonzero(actual_union, axis=(2, 3))

    metadata = {
        "calibration_version": CALIBRATION_VERSION,
        "finder_version": FINDER_VERSION,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "detection_path": str(detection_path),
        "detection_sha256": file_sha256(detection_path),
        "electronize_scale_adu_per_e": ELECTRONIZE_SCALE,
        "gain_status": "provisionally accepted by analysis owner on 2026-06-13",
        "random_seed": RANDOM_SEED,
        "selection_rule": {
            "maximize": (
                f"mean injection completeness where the sampled expected peak "
                f"is >= {CHARACTERIZABLE_PEAK_SNR} robust image sigma"
            ),
            "gates": {
                "both_lobes_separately_significant": True,
                "ordinary_fpr": MAX_ORDINARY_FPR,
                "max_ordinary_temperature_fpr": MAX_ORDINARY_TEMPERATURE_FPR,
                "structured_fpr": MAX_STRUCTURED_FPR,
                "minimum_temperature_completeness": (
                    MIN_CHARACTERIZABLE_COMPLETENESS
                ),
            },
            "tie_break": "lower structured-control FPR, then predeclared order",
            "actual_candidate_count_used": False,
        },
        "acceptance_pass": bool(acceptance_pass),
    }
    output = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "config_names": config_names,
        "config_json": np.asarray(
            [json.dumps(config.to_dict(), sort_keys=True) for config in configs]
        ),
        "temperatures": temperatures,
        "representative_temperatures": REPRESENTATIVE_TEMPERATURES,
        "injection_amplitudes": INJECTION_AMPLITUDES,
        "injection_taus_s": INJECTION_TAUS_S,
        "injection_trials": injection_trials,
        "injection_found": injection_found,
        "injection_completeness": injection_completeness,
        "injection_peak_snr": injection_peak_snr,
        "characterizable_mask": characterizable_mask.astype(np.int8),
        "class_names": np.asarray(CLASS_NAMES),
        "null_trials": null_trials,
        "null_finder_found": null_finder_found,
        "null_profile_found": null_profile_found,
        "null_end_to_end_found": null_end_to_end_found,
        "null_end_to_end_rates": null_rates,
        "null_end_to_end_upper_95": null_upper_95,
        "ordinary_temperature_rates": ordinary_temperature_rates,
        "actual_candidates": actual_candidates,
        "actual_union_candidates": actual_union_counts,
        "robust_sigma_median": robust_sigma_median,
        "legacy_sigma_median": legacy_sigma_median,
        "characterizable_completeness": characterizable_completeness,
        "characterizable_completeness_by_temperature": (
            characterizable_by_temperature
        ),
        "production_admissible": production_admissible.astype(np.int8),
        "config_gate": config_gate.astype(np.int8),
        "selected_index": np.asarray(selected_index, dtype=np.int16),
        "acceptance_pass": np.asarray(acceptance_pass, dtype=np.int8),
    }
    np.savez_compressed(output_path, **output)

    figure_paths = []
    if acceptance_pass:
        selected = configs[selected_index]
        write_frozen_config(
            selected,
            config_path,
            metadata={
                "calibration_version": CALIBRATION_VERSION,
                "calibration_path": str(output_path),
                "calibration_sha256": file_sha256(output_path),
                "selection_did_not_use_candidate_count": True,
                "electronize_scale_adu_per_e": ELECTRONIZE_SCALE,
                "gain_status": metadata["gain_status"],
            },
        )
        figure_paths = _write_figures(
            figure_dir,
            config_names,
            selected_index,
            injection_completeness,
            characterizable_completeness,
            null_end_to_end_found,
            null_trials,
        )

    report_lines = [
        "# Signed Refit Finder Calibration",
        "",
        f"- Calibration version: `{CALIBRATION_VERSION}`",
        f"- Finder version: `{FINDER_VERSION}`",
        f"- Gain used: `{ELECTRONIZE_SCALE:.0f}` ADU/e- "
        "(provisionally accepted by the analysis owner on 2026-06-13)",
        f"- Acceptance status: **{'PASS' if acceptance_pass else 'FAIL'}**",
        "",
        "## Predeclared scan",
        "",
        "| Configuration | Noise | Lobe rule | Threshold | Balance | Persistence | Trail isolation |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for config in configs:
        balance = (
            "none"
            if config.max_relative_lobe_imbalance is None
            else f"{config.max_relative_lobe_imbalance:.2f}"
        )
        report_lines.append(
            f"| `{config.name}` | {config.noise_estimator} | {config.lobe_rule} | "
            f"{config.sigma_threshold:.1f} sigma | {balance} | "
            f"{config.persistence} | "
            f"{'yes' if config.require_trail_isolation else 'no'} |"
        )

    report_lines.extend(
        [
            "",
            "The scan was fixed before looking at the final candidate count. "
            "The operating point maximizes real-residual injection completeness "
            f"where the sampled expected peak is at least "
            f"{CHARACTERIZABLE_PEAK_SNR:.1f} robust image sigma, subject to "
            "ordinary and structured-null gates. Product-only rules are retained "
            "as historical comparisons but are not production-admissible because "
            "they can accept one sub-threshold lobe. Candidate "
            "count is reported only as a consequence, not as a selection input.",
            "",
            "A v1 pilot incorrectly multiplied the 3000-cycle pump shape by "
            "`N_PUMPS` a second time and therefore clipped injection probabilities "
            "to one. After that unit bug was fixed, its A>=0.10 completeness gate "
            "was also found to be physically impossible: A=0.10 peaks near 195 e- "
            "while the measured per-image thresholds are typically 420-1050 e-. "
            "Version v2 uses the dimensionless sampled peak-SNR rule above and "
            "adds A=0.80 so every representative temperature has strong injections.",
            "",
            "## Noise-estimator comparison",
            "",
            "| T (K) | Robust median sigma (e-) | Legacy median sigma (e-) | Legacy/robust |",
            "|---:|---:|---:|---:|",
        ]
    )
    for index, temperature in enumerate(temperatures):
        report_lines.append(
            f"| {int(temperature)} | {robust_sigma_median[index]:.2f} | "
            f"{legacy_sigma_median[index]:.2f} | "
            f"{legacy_sigma_median[index] / robust_sigma_median[index]:.3f} |"
        )

    report_lines.extend(
        [
            "",
            "## Completeness-purity tradeoff",
            "",
            "| Configuration | Strong-signal completeness | Ordinary E2E FPR | Horizontal-trigger vertical FPR | Near-defect FPR | Horizontal-axis FPR | Union candidates | Production gate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for config_index, config in enumerate(configs):
        report_lines.append(
            f"| `{config.name}` | "
            f"{_format_percent(characterizable_completeness[config_index])} | "
            f"{_format_percent(null_rates[config_index, 0])} | "
            f"{_format_percent(null_rates[config_index, 1])} | "
            f"{_format_percent(null_rates[config_index, 2])} | "
            f"{_format_percent(null_rates[config_index, 3])} | "
            f"{int(np.sum(actual_union_counts[config_index])):,} | "
            f"{'PASS' if config_gate[config_index] else 'FAIL'} |"
        )
    report_lines.extend(
        [
            "",
            "The ordinary controls were deliberately masked away from the "
            "preliminary candidate union in Step 2, so their zero rate is "
            "conditional and is not, by itself, a full-field purity estimate. "
            "The near-defect and horizontal controls provide the adversarial "
            "checks, and the quoted binomial upper bounds retain finite-sample "
            "uncertainty.",
            "",
            "The horizontal-axis control applies the vertical-pair covariance and "
            "Step 6 threshold to horizontal-pair curves. It is intentionally a "
            "stress test for non-pumping structure, not a claim that horizontal "
            "curves have independently calibrated p-values.",
        ]
    )

    if acceptance_pass:
        selected = configs[selected_index]
        report_lines.extend(
            [
                "",
                "## Selected operating point",
                "",
                f"Selected: `{selected.name}`.",
                "",
                f"It recovers {_format_percent(characterizable_completeness[selected_index])} "
                f"of injections whose sampled expected peak is >="
                f"{CHARACTERIZABLE_PEAK_SNR:.1f} robust image sigma. Its complete "
                "finder-plus-profile false-positive rates are "
                f"{_format_percent(null_rates[selected_index, 0])} on ordinary "
                f"controls, {_format_percent(null_rates[selected_index, 1])} at "
                "horizontal-trigger coordinates using the vertical pair, "
                f"{_format_percent(null_rates[selected_index, 2])} near defects, "
                f"and {_format_percent(null_rates[selected_index, 3])} for the "
                "true horizontal-axis negative control.",
                "",
                "For these aggregate rates, the corresponding one-sided 95% "
                "binomial upper bounds are "
                f"{_format_percent(null_upper_95[selected_index, 0])}, "
                f"{_format_percent(null_upper_95[selected_index, 1])}, "
                f"{_format_percent(null_upper_95[selected_index, 2])}, and "
                f"{_format_percent(null_upper_95[selected_index, 3])}.",
                "",
                "### Completeness by temperature and amplitude",
                "",
                "| T (K) | "
                + " | ".join(f"A={value:.2f}" for value in INJECTION_AMPLITUDES)
                + " |",
                "|---:|" + "---:|" * INJECTION_AMPLITUDES.size,
            ]
        )
        selected_mean = np.mean(injection_completeness[selected_index], axis=2)
        for temp_index, temperature in enumerate(REPRESENTATIVE_TEMPERATURES):
            report_lines.append(
                f"| {int(temperature)} | "
                + " | ".join(
                    _format_percent(value) for value in selected_mean[temp_index]
                )
                + " |"
            )

        report_lines.extend(
            [
                "",
                "### End-to-end null rate by temperature",
                "",
                "| T (K) | Ordinary | Horizontal-trigger vertical | Near defect | Horizontal axis |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        selected_temp_rates = np.divide(
            null_end_to_end_found[selected_index],
            null_trials[selected_index],
            out=np.zeros((len(CLASS_NAMES), n_temp), dtype=float),
            where=null_trials[selected_index] > 0,
        )
        for temp_index, temperature in enumerate(temperatures):
            report_lines.append(
                f"| {int(temperature)} | "
                f"{_format_percent(selected_temp_rates[0, temp_index])} | "
                f"{_format_percent(selected_temp_rates[1, temp_index])} | "
                f"{_format_percent(selected_temp_rates[2, temp_index])} | "
                f"{_format_percent(selected_temp_rates[3, temp_index])} |"
            )

        p2_index = 3
        p3_index = 4
        isolated_index = 5
        report_lines.extend(
            [
                "",
                "## Charge-balance and trail diagnostics",
                "",
                "The relaxed balance requirement limits the magnitude mismatch "
                "between opposite lobes to 50%. Requiring three dwell detections "
                f"changes the all-temperature union from "
                f"{int(np.sum(actual_union_counts[p2_index])):,} to "
                f"{int(np.sum(actual_union_counts[p3_index])):,} sites. Adding "
                "the 20-row trail-isolation diagnostic changes it further to "
                f"{int(np.sum(actual_union_counts[isolated_index])):,}.",
                "",
                "The isolation rule counts additional >=2.5-sigma pixels in the "
                "same column within 20 rows of the pair; it is a diagnostic for "
                "deferred trails and crowded defects, not a claim that every "
                "non-isolated site is nonphysical.",
                "",
                "## Acceptance gate",
                "",
                "- PASS: finder completeness was measured with binomial transfer "
                "injections placed on real, held-out residual image pairs.",
                "- PASS: the full finder decision was intersected with the stored "
                "Step 6 profile-tau empirical p-value for ordinary, horizontal-"
                "trigger, and near-defect null controls; a horizontal-axis "
                "finder-plus-profile negative control was added independently.",
                "- PASS: both lobes must be separately significant in the frozen "
                "production configuration.",
                "- PASS: the operating point obeys the predeclared completeness "
                "and false-positive gates.",
                "- PASS: final trap count was not an optimization target.",
                "",
                f"Frozen configuration: `{config_path}`.",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "## Acceptance gate",
                "",
                "- FAIL: no predeclared finder configuration satisfies all "
                "completeness and null-control gates.",
                "- No finder configuration was frozen.",
            ]
        )

    if figure_paths:
        report_lines.extend(
            [
                "",
                "## Figures",
                "",
                *[f"- `{path.as_posix()}`" for path in figure_paths],
            ]
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="ascii")
    detection.close()
    return {
        "acceptance_pass": acceptance_pass,
        "selected_index": selected_index,
        "selected_name": (
            configs[selected_index].name if acceptance_pass else None
        ),
        "output_sha256": file_sha256(output_path),
        "report_path": str(report_path),
        "config_path": str(config_path) if acceptance_pass else None,
    }


def validate_outputs(
    output_path: Path,
    report_path: Path,
    config_path: Path,
) -> dict[str, object]:
    data = np.load(output_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    if metadata["calibration_version"] != CALIBRATION_VERSION:
        raise ValueError("Calibration version mismatch")
    if data["config_names"].size != len(PREDECLARED_CONFIGS):
        raise ValueError("Unexpected finder configuration count")
    if data["injection_completeness"].shape != (
        len(PREDECLARED_CONFIGS),
        REPRESENTATIVE_TEMPERATURES.size,
        INJECTION_AMPLITUDES.size,
        INJECTION_TAUS_S.size,
    ):
        raise ValueError("Unexpected injection completeness shape")
    selected_index = int(data["selected_index"])
    acceptance_pass = bool(data["acceptance_pass"])
    if acceptance_pass:
        if selected_index < 0 or not bool(data["config_gate"][selected_index]):
            raise ValueError("Selected finder does not pass its gate")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config_values = json.loads(config_path.read_text(encoding="ascii"))
        if (
            config_values["selected_config"]["name"]
            != str(data["config_names"][selected_index])
        ):
            raise ValueError("Frozen finder does not match calibration selection")
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    return {
        "acceptance_pass": acceptance_pass,
        "selected_index": selected_index,
        "selected_name": (
            str(data["config_names"][selected_index])
            if selected_index >= 0
            else None
        ),
        "output_sha256": file_sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_outputs(args.output, args.report, args.config)
    else:
        result = run_calibration(
            args.model,
            args.detection,
            args.output,
            args.report,
            args.config,
            args.figure_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["acceptance_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

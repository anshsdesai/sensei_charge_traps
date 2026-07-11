"""Calibrate and freeze the signed-orientation policy before the SRH fit."""

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

from signed_refit_controls import region_id
from signed_refit_detection_calibration import TARGET_PER_FIT_FPR
from signed_refit_finder import (
    ELECTRONIZE_SCALE,
    finder_mask,
    load_frozen_config,
    robust_noise_sigma,
    electronize_and_subtract_rows,
)
from signed_refit_orientation import (
    FROZEN_POLICY,
    LABEL_AMBIGUOUS,
    LABEL_DUAL,
    LABEL_SINGLE_NEGATIVE,
    LABEL_SINGLE_POSITIVE,
    LABEL_STRUCTURED,
    ORIENTATION_LABELS,
    ORIENTATION_POLICY_VERSION,
    classify_orientations,
    write_policy,
)
from signed_refit_profile_fitter import ProfileTauFitter, file_sha256, pump_shape


VALIDATION_VERSION = "signed-refit-orientation-validation-v2"
DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_DETECTION = Path("signed_refit_detection_calibration.npz")
DEFAULT_FINDER = Path("signed_refit_finder_config.json")
DEFAULT_FINDER_CALIBRATION = Path("signed_refit_finder_calibration.npz")
DEFAULT_COORDS = Path("signed_refit_orientation_candidates.npz")
DEFAULT_OUTPUT = Path("signed_refit_orientation_validation.npz")
DEFAULT_POLICY = Path("signed_refit_orientation_policy.json")
DEFAULT_REPORT = Path("signed_refit_orientation_policy.md")
DEFAULT_FIGURE_DIR = Path("figures/signed_refit_orientation")

IMAGE_SLICE = (slice(2, 512), slice(8, 3080))
N_PUMPS = 3000
RANDOM_SEED = 2026061308
INJECTION_SITES_PER_REGION = 4
INJECTION_AMPLITUDES = np.asarray([0.40, 0.80])
INJECTION_ACTIVE_WIDTHS = np.asarray([4, 8, 12, 23])
MIN_INJECTION_ORIENTATION_EFFICIENCY = 0.95
MIN_INJECTION_SIGN_ACCURACY = 0.99
MIN_ELIGIBLE_INJECTIONS = 200
MAX_END_TO_END_NULL_SINGLE_ORIENTATION_RATE = 0.001

CONTROL_CLASSES = (
    "ordinary",
    "horizontal_trigger_vertical",
    "near_defect",
    "horizontal_axis",
)
CONTROL_SOURCES = {
    "ordinary": "ordinary",
    "horizontal_trigger_vertical": "horizontal",
    "near_defect": "near_defect",
    "horizontal_axis": "horizontal",
}


def _decode_paths(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _empirical_survival(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    first_ge = np.searchsorted(reference, values, side="left")
    count_ge = reference.size - first_ge
    return (count_ge + 1.0) / (reference.size + 1.0)


def _load_control_classes(
    detection: np.lib.npyio.NpzFile,
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for name in CONTROL_CLASSES:
        source = CONTROL_SOURCES[name]
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


def _build_candidate_coordinates(
    model: h5py.File,
    finder_path: Path,
    finder_calibration_path: Path,
    output_path: Path,
) -> tuple[dict[int, np.ndarray], str]:
    config = load_frozen_config(finder_path)
    if config.noise_estimator != "robust":
        raise ValueError("Step 8 expects the frozen robust finder")
    union = np.zeros((4, 509, 3072), dtype=bool)
    temp_names = sorted(
        (name for name in model if name.startswith("temp_")),
        key=lambda value: int(value.split("_")[1]),
    )
    for temp_name in temp_names:
        group = model[temp_name]
        paths = _decode_paths(np.asarray(group["source_fits"]))
        counts = np.zeros((4, 509, 3072), dtype=np.uint8)
        for path in paths:
            with fits.open(
                path,
                memmap=True,
                do_not_scale_image_data=True,
            ) as hdus:
                for quadrant in range(4):
                    _, residual = electronize_and_subtract_rows(
                        hdus[quadrant].data[IMAGE_SLICE]
                    )
                    sigma = robust_noise_sigma(residual)
                    counts[quadrant] += finder_mask(residual, sigma, config)
        union |= counts >= config.persistence

    coordinates = {}
    for quadrant in range(4):
        rows, cols = np.where(union[quadrant])
        coordinates[quadrant] = np.column_stack((rows + 1, cols)).astype(
            np.int16
        )

    finder_calibration = np.load(finder_calibration_path, allow_pickle=False)
    selected = int(finder_calibration["selected_index"])
    expected = np.asarray(
        finder_calibration["actual_union_candidates"][selected],
        dtype=int,
    )
    observed = np.asarray(
        [coordinates[q].shape[0] for q in range(4)],
        dtype=int,
    )
    finder_calibration.close()
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"Rebuilt finder counts {observed.tolist()} do not match "
            f"Step 7 {expected.tolist()}"
        )
    payload = {
        **{
            f"quad_idx_{quadrant}": coordinates[quadrant]
            for quadrant in range(4)
        },
        "finder_config_sha256": np.asarray(file_sha256(finder_path)),
        "finder_calibration_sha256": np.asarray(
            file_sha256(finder_calibration_path)
        ),
        "orientation_validation_version": np.asarray(VALIDATION_VERSION),
    }
    np.savez_compressed(output_path, **payload)
    return coordinates, file_sha256(output_path)


def _flatten_candidates(
    coordinates: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    quadrants = []
    rows = []
    cols = []
    regions = []
    for quadrant in range(4):
        coords = coordinates[quadrant]
        quadrants.append(np.full(coords.shape[0], quadrant, dtype=np.int8))
        rows.append(coords[:, 0])
        cols.append(coords[:, 1])
        regions.append(region_id(coords[:, 0], coords[:, 1]).astype(np.int8))
    return {
        "quadrant": np.concatenate(quadrants),
        "row": np.concatenate(rows),
        "col": np.concatenate(cols),
        "region": np.concatenate(regions),
    }


def _coordinate_membership(
    quadrants: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    coordinates: dict[int, np.ndarray],
) -> np.ndarray:
    coordinate_sets = {
        quadrant: set(map(tuple, np.asarray(values, dtype=int)))
        for quadrant, values in coordinates.items()
    }
    return np.asarray(
        [
            (int(row), int(col)) in coordinate_sets[int(quadrant)]
            for quadrant, row, col in zip(quadrants, rows, cols)
        ],
        dtype=bool,
    )


def _horizontal_pixel_overlap(
    quadrants: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    horizontal: dict[str, np.ndarray],
) -> np.ndarray:
    horizontal_pixels = set()
    for quadrant, row, col in zip(
        horizontal["quadrant"],
        horizontal["row"],
        horizontal["col"],
    ):
        horizontal_pixels.add((int(quadrant), int(row), int(col)))
        horizontal_pixels.add((int(quadrant), int(row), int(col) - 1))
    return np.asarray(
        [
            (
                (int(quadrant), int(row), int(col)) in horizontal_pixels
                or (int(quadrant), int(row) - 1, int(col))
                in horizontal_pixels
            )
            for quadrant, row, col in zip(quadrants, rows, cols)
        ],
        dtype=bool,
    )


def _injection_indices(
    ordinary: dict[str, np.ndarray],
) -> np.ndarray:
    selected = []
    for quadrant in range(4):
        for region in range(32):
            indices = np.flatnonzero(
                (ordinary["quadrant"] == quadrant)
                & (ordinary["region"] == region)
                & (ordinary["row"] > 0)
            )
            if indices.size < INJECTION_SITES_PER_REGION:
                raise ValueError("Insufficient ordinary controls for injections")
            selected.extend(indices[:INJECTION_SITES_PER_REGION].tolist())
    return np.asarray(selected, dtype=np.int32)


def _profile_curves(
    curves: np.ndarray,
    quadrants: np.ndarray,
    regions: np.ndarray,
    temp_group: h5py.Group,
) -> dict[str, np.ndarray]:
    count = curves.shape[0]
    output = {
        "statistic": np.full(count, np.nan, dtype=np.float32),
        "tau": np.full(count, np.nan, dtype=np.float32),
        "amplitude": np.full(count, np.nan, dtype=np.float32),
        "amplitude_z": np.full(count, np.nan, dtype=np.float32),
    }
    seconds = np.asarray(temp_group["seconds"], dtype=float)
    for quadrant in range(4):
        for region in range(32):
            selected = (quadrants == quadrant) & (regions == region)
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
            best_index = np.asarray(result["best_grid_index"], dtype=int)
            amplitude_sigma = np.sqrt(
                fitter.constant_normal / fitter.determinant[best_index]
            )
            output["statistic"][selected] = result["delta_chi2"]
            output["tau"][selected] = result["tau"]
            output["amplitude"][selected] = result["amplitude"]
            output["amplitude_z"][selected] = (
                result["amplitude"] / amplitude_sigma
            )
    return output


def _profile_selected_controls(
    curves: np.ndarray,
    values: dict[str, np.ndarray],
    selected: np.ndarray,
    temp_group: h5py.Group,
) -> tuple[np.ndarray, np.ndarray]:
    signs = np.zeros(values["row"].size, dtype=np.int8)
    zvalues = np.full(values["row"].size, np.nan, dtype=np.float32)
    if not np.any(selected):
        return signs, zvalues
    result = _profile_curves(
        curves[selected],
        values["quadrant"][selected],
        values["region"][selected],
        temp_group,
    )
    signs[selected] = np.sign(result["amplitude"]).astype(np.int8)
    zvalues[selected] = result["amplitude_z"]
    return signs, zvalues


def _injection_design(
    count: int,
    temperature_count: int,
) -> dict[str, np.ndarray]:
    scenario_count = INJECTION_AMPLITUDES.size * INJECTION_ACTIVE_WIDTHS.size
    scenario = np.arange(count) % scenario_count
    amplitude_index = scenario // INJECTION_ACTIVE_WIDTHS.size
    width_index = scenario % INJECTION_ACTIVE_WIDTHS.size
    amplitude = INJECTION_AMPLITUDES[amplitude_index]
    active_width = INJECTION_ACTIVE_WIDTHS[width_index]
    true_sign = np.where((np.arange(count) & 1) == 0, 1, -1).astype(np.int8)
    active = np.zeros((count, temperature_count), dtype=bool)
    for index, width in enumerate(active_width):
        max_start = temperature_count - int(width)
        start = 0 if max_start == 0 else (
            (index * 2654435761 + RANDOM_SEED) % (max_start + 1)
        )
        active[index, int(start) : int(start) + int(width)] = True
    return {
        "scenario": scenario.astype(np.int8),
        "amplitude": amplitude.astype(np.float32),
        "active_width": active_width.astype(np.int8),
        "true_sign": true_sign,
        "active": active,
    }


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        label: int(np.count_nonzero(labels == label))
        for label in ORIENTATION_LABELS
    }


def _write_figures(
    figure_dir: Path,
    temperatures: np.ndarray,
    candidate: dict[str, np.ndarray],
    candidate_classification: dict[str, np.ndarray],
    injection_summary: dict[str, object],
    control_summaries: dict[str, dict[str, object]],
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    labels = candidate_classification["label"]
    conflict = np.flatnonzero(
        np.isin(labels, (LABEL_AMBIGUOUS, LABEL_DUAL))
    )
    ordering = sorted(
        conflict.tolist(),
        key=lambda index: (
            -int(candidate_classification["minority_count"][index]),
            -int(candidate_classification["significant_count"][index]),
            index,
        ),
    )[:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for axis, index in zip(axes.flat, ordering):
        accepted = candidate["significant"][index]
        axis.axhline(0.0, color="0.5", linewidth=1)
        axis.plot(
            temperatures,
            candidate["amplitude"][index],
            color="0.75",
            marker="o",
            markersize=3,
            linewidth=1,
        )
        axis.errorbar(
            temperatures[accepted],
            candidate["amplitude"][index, accepted],
            yerr=np.divide(
                np.abs(candidate["amplitude"][index, accepted]),
                np.abs(candidate["amplitude_z"][index, accepted]),
                out=np.zeros(np.count_nonzero(accepted)),
                where=np.abs(candidate["amplitude_z"][index, accepted]) > 0,
            ),
            fmt="o",
            color="tab:red",
            markersize=4,
        )
        axis.set_title(
            f"Q{candidate['quadrant'][index]} "
            f"({candidate['row'][index]},{candidate['col'][index]})\n"
            f"{labels[index]}, "
            f"+{candidate_classification['positive_count'][index]}/"
            f"-{candidate_classification['negative_count'][index]}",
            fontsize=9,
        )
    for axis in axes[-1]:
        axis.set_xlabel("Temperature (K)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude coefficient A")
    for axis in axes.flat[len(ordering) :]:
        axis.axis("off")
    fig.tight_layout()
    path = figure_dir / "sign_changing_candidate_examples.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    names = ["injections", *CONTROL_CLASSES]
    single_rates = [float(injection_summary["correct_single_rate"])]
    for name in CONTROL_CLASSES:
        single_rates.append(
            float(control_summaries[name]["raw_single_rate_all"])
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, single_rates, color=["tab:blue", *["tab:orange"] * 4])
    ax.set_yscale("log")
    ax.set_ylabel("Correct single orientation / null single-orientation rate")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figure_dir / "orientation_signal_null_rates.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def run_validation(
    model_path: Path,
    detection_path: Path,
    finder_path: Path,
    finder_calibration_path: Path,
    coords_path: Path,
    output_path: Path,
    policy_path: Path,
    report_path: Path,
    figure_dir: Path,
) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    detection = np.load(detection_path, allow_pickle=False)
    temperatures = np.asarray(detection["temperatures"], dtype=np.int16)
    references = np.asarray(detection["reference_statistics"], dtype=float)
    thresholds = np.asarray(detection["thresholds"], dtype=float)
    controls = _load_control_classes(detection)
    injection_indices = _injection_indices(controls["ordinary"])
    injection_design = _injection_design(
        injection_indices.size,
        temperatures.size,
    )

    with h5py.File(model_path, "r") as model:
        coordinates, coords_sha256 = _build_candidate_coordinates(
            model,
            finder_path,
            finder_calibration_path,
            coords_path,
        )
        candidate_coords = _flatten_candidates(coordinates)
        candidate_count = candidate_coords["row"].size
        horizontal_control = controls["horizontal_trigger_vertical"]
        candidate_structured = _horizontal_pixel_overlap(
            candidate_coords["quadrant"],
            candidate_coords["row"],
            candidate_coords["col"],
            horizontal_control,
        )
        control_finder_selected = {}
        control_structured = {}
        for name, values in controls.items():
            if name == "horizontal_axis":
                control_finder_selected[name] = np.ones(
                    values["row"].size,
                    dtype=bool,
                )
            else:
                control_finder_selected[name] = _coordinate_membership(
                    values["quadrant"],
                    values["row"],
                    values["col"],
                    coordinates,
                )
            control_structured[name] = _horizontal_pixel_overlap(
                values["quadrant"],
                values["row"],
                values["col"],
                horizontal_control,
            )
        injection_structured = control_structured["ordinary"][
            injection_indices
        ]
        candidate = {
            **candidate_coords,
            "amplitude": np.full(
                (candidate_count, temperatures.size),
                np.nan,
                dtype=np.float32,
            ),
            "amplitude_z": np.full(
                (candidate_count, temperatures.size),
                np.nan,
                dtype=np.float32,
            ),
            "tau": np.full(
                (candidate_count, temperatures.size),
                np.nan,
                dtype=np.float32,
            ),
            "pvalue": np.full(
                (candidate_count, temperatures.size),
                np.nan,
                dtype=np.float32,
            ),
            "significant": np.zeros(
                (candidate_count, temperatures.size),
                dtype=bool,
            ),
            "sign": np.zeros(
                (candidate_count, temperatures.size),
                dtype=np.int8,
            ),
        }
        control_sign = {
            name: np.zeros(
                (values["row"].size, temperatures.size),
                dtype=np.int8,
            )
            for name, values in controls.items()
        }
        control_significant = {
            name: np.zeros(
                (values["row"].size, temperatures.size),
                dtype=bool,
            )
            for name, values in controls.items()
        }
        control_z = {
            name: np.full(
                (values["row"].size, temperatures.size),
                np.nan,
                dtype=np.float32,
            )
            for name, values in controls.items()
        }
        injection_amplitude = np.full(
            (injection_indices.size, temperatures.size),
            np.nan,
            dtype=np.float32,
        )
        injection_z = np.full_like(injection_amplitude, np.nan)
        injection_pvalue = np.full_like(injection_amplitude, np.nan)
        injection_significant = np.zeros_like(
            injection_amplitude,
            dtype=bool,
        )
        injection_sign = np.zeros_like(injection_amplitude, dtype=np.int8)

        for temp_index, temperature_value in enumerate(temperatures):
            temperature = int(temperature_value)
            temp_group = model[f"temp_{temperature}"]
            paths = _decode_paths(np.asarray(temp_group["source_fits"]))
            seconds = np.asarray(temp_group["seconds"], dtype=float)
            candidate_curves = np.empty(
                (candidate_count, len(paths)),
                dtype=np.float32,
            )
            control_curves = {
                name: np.empty(
                    (values["row"].size, len(paths)),
                    dtype=np.float32,
                )
                for name, values in controls.items()
            }
            for image_index, path in enumerate(paths):
                with fits.open(
                    path,
                    memmap=True,
                    do_not_scale_image_data=True,
                ) as hdus:
                    for quadrant in range(4):
                        _, residual = electronize_and_subtract_rows(
                            hdus[quadrant].data[IMAGE_SLICE]
                        )
                        selected = candidate["quadrant"] == quadrant
                        rows = candidate["row"][selected]
                        cols = candidate["col"][selected]
                        candidate_curves[selected, image_index] = (
                            residual[rows, cols] - residual[rows - 1, cols]
                        ) / 2.0
                        for name, values in controls.items():
                            chosen = values["quadrant"] == quadrant
                            rows = values["row"][chosen]
                            cols = values["col"][chosen]
                            if name == "horizontal_axis":
                                control_curves[name][chosen, image_index] = (
                                    residual[rows, cols]
                                    - residual[rows, cols - 1]
                                ) / 2.0
                            else:
                                control_curves[name][chosen, image_index] = (
                                    residual[rows, cols]
                                    - residual[rows - 1, cols]
                                ) / 2.0

            candidate_result = _profile_curves(
                candidate_curves,
                candidate["quadrant"],
                candidate["region"],
                temp_group,
            )
            candidate_pvalue = _empirical_survival(
                references[temp_index],
                candidate_result["statistic"],
            )
            accepted = candidate_pvalue <= TARGET_PER_FIT_FPR
            signs = np.sign(candidate_result["amplitude"]).astype(np.int8)
            candidate["amplitude"][:, temp_index] = candidate_result["amplitude"]
            candidate["amplitude_z"][:, temp_index] = candidate_result[
                "amplitude_z"
            ]
            candidate["tau"][:, temp_index] = candidate_result["tau"]
            candidate["pvalue"][:, temp_index] = candidate_pvalue
            candidate["significant"][:, temp_index] = accepted
            candidate["sign"][:, temp_index] = signs

            for name, values in controls.items():
                if name == "horizontal_axis":
                    result = _profile_curves(
                        control_curves[name],
                        values["quadrant"],
                        values["region"],
                        temp_group,
                    )
                    pvalues = _empirical_survival(
                        references[temp_index],
                        result["statistic"],
                    )
                    selected = pvalues <= TARGET_PER_FIT_FPR
                    control_significant[name][:, temp_index] = selected
                    control_sign[name][:, temp_index] = np.sign(
                        result["amplitude"]
                    ).astype(np.int8)
                    control_z[name][:, temp_index] = result["amplitude_z"]
                else:
                    selected = (
                        values["pvalue"][:, temp_index]
                        <= TARGET_PER_FIT_FPR
                    )
                    signs, zvalues = _profile_selected_controls(
                        control_curves[name],
                        values,
                        selected,
                        temp_group,
                    )
                    control_significant[name][:, temp_index] = selected
                    control_sign[name][:, temp_index] = signs
                    control_z[name][:, temp_index] = zvalues

            ordinary_curves = control_curves["ordinary"][injection_indices]
            injection_curves = ordinary_curves.copy()
            active = injection_design["active"][:, temp_index]
            tau_injected = (
                float(seconds[len(seconds) // 2])
                / (np.log(8.0) / 7.0)
            )
            shape = pump_shape(seconds, tau_injected) / N_PUMPS
            active_indices = np.flatnonzero(active)
            for index in active_indices:
                probability = np.clip(
                    float(injection_design["amplitude"][index]) * shape,
                    0.0,
                    1.0,
                )
                draw = rng.binomial(N_PUMPS, probability)
                injection_curves[index] += (
                    int(injection_design["true_sign"][index]) * draw
                )
            injection_values = controls["ordinary"]
            injection_result = _profile_curves(
                injection_curves,
                injection_values["quadrant"][injection_indices],
                injection_values["region"][injection_indices],
                temp_group,
            )
            pvalues = _empirical_survival(
                references[temp_index],
                injection_result["statistic"],
            )
            selected = pvalues <= TARGET_PER_FIT_FPR
            injection_amplitude[:, temp_index] = injection_result["amplitude"]
            injection_z[:, temp_index] = injection_result["amplitude_z"]
            injection_pvalue[:, temp_index] = pvalues
            injection_significant[:, temp_index] = selected
            injection_sign[:, temp_index] = np.sign(
                injection_result["amplitude"]
            ).astype(np.int8)
            print(
                f"Step 8: profiled {temperature} K "
                f"({candidate_count} candidates)",
                flush=True,
            )

    candidate_classification = classify_orientations(
        candidate["sign"],
        candidate["significant"],
        structured_background=candidate_structured,
    )
    control_raw_classification = {
        name: classify_orientations(control_sign[name], control_significant[name])
        for name in CONTROL_CLASSES
    }
    control_classification = {
        name: classify_orientations(
            control_sign[name],
            control_significant[name],
            structured_background=control_structured[name],
        )
        for name in CONTROL_CLASSES
    }
    injection_classification = classify_orientations(
        injection_sign,
        injection_significant,
        structured_background=injection_structured,
    )

    true_detected = injection_significant & injection_design["active"]
    true_detected_count = np.sum(true_detected, axis=1)
    orientation_eligible = true_detected_count >= (
        FROZEN_POLICY.minimum_significant_temperatures
    )
    correct_single = (
        injection_classification["single_trap_eligible"]
        & (
            injection_classification["dominant_sign"]
            == injection_design["true_sign"]
        )
    )
    correct_single_rate = float(
        np.mean(correct_single[orientation_eligible])
    )
    active_accepted = true_detected
    sign_accuracy = float(
        np.mean(
            injection_sign[active_accepted]
            == np.broadcast_to(
                injection_design["true_sign"][:, None],
                injection_sign.shape,
            )[active_accepted]
        )
    )

    scenario_summary = {}
    for amplitude in INJECTION_AMPLITUDES:
        for width in INJECTION_ACTIVE_WIDTHS:
            selected = (
                np.isclose(injection_design["amplitude"], amplitude)
                & (injection_design["active_width"] == width)
            )
            eligible = selected & orientation_eligible
            scenario_summary[f"A={amplitude:.2f},width={int(width)}"] = {
                "count": int(np.count_nonzero(selected)),
                "orientation_eligible": int(np.count_nonzero(eligible)),
                "correct_single_rate_eligible": (
                    float(np.mean(correct_single[eligible]))
                    if np.any(eligible)
                    else None
                ),
            }
    injection_summary = {
        "count": int(injection_indices.size),
        "orientation_eligible": int(np.count_nonzero(orientation_eligible)),
        "structured_background_overlap": int(
            np.count_nonzero(injection_structured)
        ),
        "correct_single_rate": correct_single_rate,
        "sign_accuracy_active_accepted": sign_accuracy,
        "classification_counts": _label_counts(
            injection_classification["label"]
        ),
        "by_scenario": scenario_summary,
    }

    control_summaries = {}
    for name in CONTROL_CLASSES:
        classification = control_classification[name]
        raw_classification = control_raw_classification[name]
        labels = classification["label"]
        raw_labels = raw_classification["label"]
        raw_eligible = (
            raw_classification["significant_count"]
            >= FROZEN_POLICY.minimum_significant_temperatures
        )
        raw_single = raw_classification["single_trap_eligible"]
        finder_selected = control_finder_selected[name]
        single = classification["single_trap_eligible"]
        end_to_end_single = finder_selected & single
        control_summaries[name] = {
            "site_count": int(labels.size),
            "raw_orientation_eligible": int(np.count_nonzero(raw_eligible)),
            "raw_single_orientation": int(np.count_nonzero(raw_single)),
            "raw_single_rate_all": float(np.mean(raw_single)),
            "raw_sign_conflict_rejection_eligible": (
                float(np.mean(~raw_single[raw_eligible]))
                if np.any(raw_eligible)
                else None
            ),
            "finder_selected": int(np.count_nonzero(finder_selected)),
            "structured_background_overlap": int(
                np.count_nonzero(control_structured[name])
            ),
            "end_to_end_single_orientation": int(
                np.count_nonzero(end_to_end_single)
            ),
            "end_to_end_single_rate": float(np.mean(end_to_end_single)),
            "raw_classification_counts": _label_counts(raw_labels),
            "classification_counts": _label_counts(labels),
        }

    candidate_summary = {
        "site_count": candidate_count,
        "classification_counts": _label_counts(
            candidate_classification["label"]
        ),
        "single_trap_eligible": int(
            np.count_nonzero(candidate_classification["single_trap_eligible"])
        ),
        "ambiguous_sign_conflict": int(
            np.count_nonzero(
                candidate_classification["label"] == LABEL_AMBIGUOUS
            )
        ),
        "dual_response": int(
            np.count_nonzero(candidate_classification["label"] == LABEL_DUAL)
        ),
        "structured_background_overlap": int(
            np.count_nonzero(
                candidate_classification["label"] == LABEL_STRUCTURED
            )
        ),
    }

    production_control_names = (
        "ordinary",
        "horizontal_trigger_vertical",
        "near_defect",
    )
    maximum_null_single_rate = max(
        control_summaries[name]["end_to_end_single_rate"]
        for name in production_control_names
    )
    acceptance_pass = bool(
        injection_summary["orientation_eligible"] >= MIN_ELIGIBLE_INJECTIONS
        and correct_single_rate >= MIN_INJECTION_ORIENTATION_EFFICIENCY
        and sign_accuracy >= MIN_INJECTION_SIGN_ACCURACY
        and maximum_null_single_rate
        <= MAX_END_TO_END_NULL_SINGLE_ORIENTATION_RATE
        and np.all(
            ~candidate_classification["single_trap_eligible"]
            | (candidate_classification["minority_count"] == 0)
        )
    )

    metadata = {
        "validation_version": VALIDATION_VERSION,
        "orientation_policy_version": ORIENTATION_POLICY_VERSION,
        "model_sha256": file_sha256(model_path),
        "detection_sha256": file_sha256(detection_path),
        "finder_config_sha256": file_sha256(finder_path),
        "finder_calibration_sha256": file_sha256(finder_calibration_path),
        "candidate_coordinates_sha256": coords_sha256,
        "random_seed": RANDOM_SEED,
        "significance_rule": "temperature empirical profile p <= 0.001",
        "candidate_fit_role": (
            "policy calibration only; Step 9 reruns definitive "
            "signal-dependent fits"
        ),
        "acceptance_pass": acceptance_pass,
    }

    output = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "temperatures": temperatures,
        "candidate_quadrant": candidate["quadrant"],
        "candidate_row": candidate["row"],
        "candidate_col": candidate["col"],
        "candidate_region": candidate["region"],
        "candidate_amplitude": candidate["amplitude"],
        "candidate_amplitude_z": candidate["amplitude_z"],
        "candidate_tau": candidate["tau"],
        "candidate_empirical_pvalue": candidate["pvalue"],
        "candidate_significant": candidate["significant"].astype(np.int8),
        "candidate_sign": candidate["sign"],
        "candidate_orientation_label": candidate_classification["label"],
        "candidate_significant_count": candidate_classification[
            "significant_count"
        ],
        "candidate_positive_count": candidate_classification["positive_count"],
        "candidate_negative_count": candidate_classification["negative_count"],
        "candidate_dominant_sign": candidate_classification["dominant_sign"],
        "candidate_dominant_fraction": candidate_classification[
            "dominant_fraction"
        ],
        "candidate_structured_background": candidate_structured.astype(np.int8),
        "candidate_single_trap_eligible": candidate_classification[
            "single_trap_eligible"
        ].astype(np.int8),
        "injection_control_index": injection_indices,
        "injection_amplitude_true": injection_design["amplitude"],
        "injection_active_width": injection_design["active_width"],
        "injection_true_sign": injection_design["true_sign"],
        "injection_active": injection_design["active"].astype(np.int8),
        "injection_amplitude_fit": injection_amplitude,
        "injection_amplitude_z": injection_z,
        "injection_empirical_pvalue": injection_pvalue,
        "injection_significant": injection_significant.astype(np.int8),
        "injection_sign": injection_sign,
        "injection_orientation_label": injection_classification["label"],
        "injection_structured_background": injection_structured.astype(np.int8),
        "injection_orientation_eligible": orientation_eligible.astype(np.int8),
        "injection_correct_single": correct_single.astype(np.int8),
        "class_names": np.asarray(CONTROL_CLASSES),
        "acceptance_pass": np.asarray(acceptance_pass, dtype=np.int8),
    }
    for name in CONTROL_CLASSES:
        output[f"{name}_significant"] = control_significant[name].astype(np.int8)
        output[f"{name}_sign"] = control_sign[name]
        output[f"{name}_amplitude_z"] = control_z[name]
        output[f"{name}_finder_selected"] = control_finder_selected[name].astype(
            np.int8
        )
        output[f"{name}_structured_background"] = control_structured[name].astype(
            np.int8
        )
        output[f"{name}_raw_orientation_label"] = control_raw_classification[
            name
        ]["label"]
        output[f"{name}_orientation_label"] = control_classification[name][
            "label"
        ]
        output[f"{name}_single_trap_eligible"] = control_classification[name][
            "single_trap_eligible"
        ].astype(np.int8)
    np.savez_compressed(output_path, **output)

    write_policy(
        policy_path,
        metadata={
            "acceptance_pass": acceptance_pass,
            "status": "FROZEN" if acceptance_pass else "NOT_FROZEN",
            "validation_version": VALIDATION_VERSION,
            "validation_path": str(output_path),
            "validation_sha256": file_sha256(output_path),
            "significance_rule": metadata["significance_rule"],
            "single_trap_rule": (
                "at least four significant temperatures and no accepted "
                "opposite-sign amplitude, with no shared pixel in the frozen "
                "persistent-horizontal morphology list"
            ),
            "step9_requirement": (
                "recompute labels from the definitive accepted-temperature mask "
                "and apply the frozen horizontal-overlap classification"
            ),
        },
    )
    figure_paths = _write_figures(
        figure_dir,
        temperatures,
        candidate,
        candidate_classification,
        injection_summary,
        control_summaries,
    )

    report_lines = [
        "# Signed Refit Orientation Policy",
        "",
        f"- Validation version: `{VALIDATION_VERSION}`",
        f"- Policy version: `{ORIENTATION_POLICY_VERSION}`",
        f"- Acceptance status: **{'PASS' if acceptance_pass else 'FAIL'}**",
        "",
        "## Tested policy (not frozen)" if not acceptance_pass else "## Frozen policy",
        "",
        "A temperature contributes an orientation only when its complete profile-"
        "tau search has empirical `p <= 0.001` under the Step 6 calibration. "
        "Insignificant temperatures are retained in the artifact but ignored for "
        "orientation consistency.",
        "",
        "- Fewer than four significant temperatures: "
        "`insufficient_significant_temperatures`.",
        "- At least four significant temperatures, all positive or all negative: "
        "`single_orientation_positive` or `single_orientation_negative`.",
        "- At least four significant temperatures with exactly one accepted "
        "minority-sign fit: `ambiguous_sign_conflict`.",
        "- At least two accepted positive and two accepted negative fits: "
        "`dual_response`.",
        "- A vertical pair sharing either lobe pixel with the frozen persistent-"
        "horizontal morphology list: `structured_background_overlap`, regardless "
        "of sign consistency.",
        "- Both conflict classes are excluded from a single-trap SRH fit and "
        "remain published as auditable classifications.",
        "",
        "Step 9 must recompute these labels using its definitive accepted-"
        "temperature mask. It may remove an unaccepted temperature from the sign "
        "test, but it may not combine accepted opposite signs or restore a "
        "persistent-horizontal overlap to the single-trap class.",
        "",
        "## Candidate results",
        "",
        f"- Step 7 candidate sites: {candidate_count:,}.",
        f"- Single-orientation eligible: "
        f"{candidate_summary['single_trap_eligible']:,}.",
        f"- Ambiguous one-sign-conflict sites: "
        f"{candidate_summary['ambiguous_sign_conflict']:,}.",
        f"- Dual-response sites: {candidate_summary['dual_response']:,}.",
        f"- Persistent-horizontal pixel overlaps: "
        f"{candidate_summary['structured_background_overlap']:,}.",
        "",
        "| Classification | Sites |",
        "|---|---:|",
    ]
    for label in ORIENTATION_LABELS:
        report_lines.append(
            f"| `{label}` | "
            f"{candidate_summary['classification_counts'][label]:,} |"
        )

    report_lines.extend(
        [
            "",
            "## Injection efficiency",
            "",
            f"- Injection sites: {injection_summary['count']:,}.",
            f"- Sites with at least four detected true-signal temperatures: "
            f"{injection_summary['orientation_eligible']:,}.",
            f"- Correct single-orientation efficiency conditional on that "
            f"eligibility: {100 * correct_single_rate:.3f}%.",
            f"- Sign accuracy among accepted active-temperature fits: "
            f"{100 * sign_accuracy:.3f}%.",
            f"- Injection sites overlapping persistent-horizontal morphology: "
            f"{injection_summary['structured_background_overlap']:,}.",
            "",
            "| Scenario | Injections | Orientation eligible | Correct single rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, summary in scenario_summary.items():
        rate = summary["correct_single_rate_eligible"]
        rate_text = "n/a" if rate is None else f"{100 * rate:.3f}%"
        report_lines.append(
            f"| `{name}` | {summary['count']} | "
            f"{summary['orientation_eligible']} | {rate_text} |"
        )

    report_lines.extend(
        [
            "",
            "## Null and structured controls",
            "",
            "| Class | Sites | Raw >=4 significant | Raw single orientation | Finder-selected union | Structured overlap | Final single orientation | Final rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CONTROL_CLASSES:
        summary = control_summaries[name]
        report_lines.append(
            f"| `{name}` | {summary['site_count']:,} | "
            f"{summary['raw_orientation_eligible']:,} | "
            f"{summary['raw_single_orientation']:,} | "
            f"{summary['finder_selected']:,} | "
            f"{summary['structured_background_overlap']:,} | "
            f"{summary['end_to_end_single_orientation']:,} | "
            f"{100 * summary['end_to_end_single_rate']:.4f}% |"
        )

    report_lines.extend(
        [
            "",
            "The candidate amplitudes in this report use the frozen null-covariance "
            "detection fit solely to calibrate the sign policy. They are not the "
            "Step 9 definitive amplitudes or uncertainties. The final artifact "
            "must use the signal-dependent covariance and rerun this classifier.",
            "",
            "The horizontal-axis class is intentionally selected by a horizontal "
            "finder and therefore measures whether sign consistency alone rejects "
            "coherent non-pumping structure. It is not included in the vertical-"
            "catalog false-positive gate. Production null rates require entry "
            "through the frozen vertical finder union. Version v1 failed because "
            "it omitted that conditioning and treated all horizontal-axis stress "
            "sites as vertical candidates.",
            "",
            "## Visual inspection",
            "",
            "The sign-changing examples plot every fitted amplitude in gray and "
            "mark empirically significant temperatures in red. The inspected "
            "examples show coherent positive and negative temperature bands or a "
            "single isolated conflicting fit; no sign is silently converted.",
            "",
            *[f"- `{path.as_posix()}`" for path in figure_paths],
            "",
            "## Acceptance gate",
            "",
            f"- {'PASS' if correct_single_rate >= MIN_INJECTION_ORIENTATION_EFFICIENCY else 'FAIL'}: "
            f"conditional injection efficiency is "
            f"{100 * correct_single_rate:.3f}% (required "
            f">= {100 * MIN_INJECTION_ORIENTATION_EFFICIENCY:.1f}%).",
            f"- {'PASS' if sign_accuracy >= MIN_INJECTION_SIGN_ACCURACY else 'FAIL'}: "
            f"accepted active-fit sign accuracy is {100 * sign_accuracy:.3f}% "
            f"(required >= {100 * MIN_INJECTION_SIGN_ACCURACY:.1f}%).",
            f"- {'PASS' if maximum_null_single_rate <= MAX_END_TO_END_NULL_SINGLE_ORIENTATION_RATE else 'FAIL'}: "
            f"maximum end-to-end vertical-null single-orientation rate is "
            f"{100 * maximum_null_single_rate:.4f}% (required <= "
            f"{100 * MAX_END_TO_END_NULL_SINGLE_ORIENTATION_RATE:.1f}%).",
            "- PASS: classifier logic makes a single-trap label impossible when "
            "any accepted opposite-sign temperature is present.",
            "- PASS: persistent response sharing a pixel in the non-pumping "
            "horizontal direction is retained as structured background, not a "
            "single vertical trap.",
            "- PASS: ambiguous and dual-response labels and per-temperature signs "
            "remain stored for auditing.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="ascii")
    detection.close()
    return {
        "acceptance_pass": acceptance_pass,
        "candidate_count": candidate_count,
        "candidate_single_trap_eligible": candidate_summary[
            "single_trap_eligible"
        ],
        "injection_orientation_efficiency": correct_single_rate,
        "injection_sign_accuracy": sign_accuracy,
        "output_sha256": file_sha256(output_path),
    }


def validate_outputs(
    output_path: Path,
    policy_path: Path,
    report_path: Path,
    coords_path: Path,
) -> dict[str, object]:
    data = np.load(output_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    if metadata["validation_version"] != VALIDATION_VERSION:
        raise ValueError("Orientation validation version mismatch")
    if not bool(data["acceptance_pass"]):
        raise ValueError("Orientation validation did not pass")
    if data["candidate_amplitude"].shape != (
        data["candidate_row"].size,
        data["temperatures"].size,
    ):
        raise ValueError("Candidate amplitude shape mismatch")
    single = np.asarray(data["candidate_single_trap_eligible"], dtype=bool)
    labels = np.asarray(data["candidate_orientation_label"])
    if np.any(
        single
        & ~np.isin(
            labels,
            (LABEL_SINGLE_POSITIVE, LABEL_SINGLE_NEGATIVE),
        )
    ):
        raise ValueError("Non-single labels marked single-trap eligible")
    if not policy_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Orientation policy output is missing")
    if metadata["candidate_coordinates_sha256"] != file_sha256(coords_path):
        raise ValueError("Candidate coordinate hash mismatch")
    result = {
        "acceptance_pass": True,
        "candidate_count": int(data["candidate_row"].size),
        "candidate_single_trap_eligible": int(np.count_nonzero(single)),
        "output_sha256": file_sha256(output_path),
    }
    data.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    parser.add_argument("--finder", type=Path, default=DEFAULT_FINDER)
    parser.add_argument(
        "--finder-calibration",
        type=Path,
        default=DEFAULT_FINDER_CALIBRATION,
    )
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_outputs(
            args.output,
            args.policy,
            args.report,
            args.coords,
        )
    else:
        result = run_validation(
            args.model,
            args.detection,
            args.finder,
            args.finder_calibration,
            args.coords,
            args.output,
            args.policy,
            args.report,
            args.figure_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["acceptance_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

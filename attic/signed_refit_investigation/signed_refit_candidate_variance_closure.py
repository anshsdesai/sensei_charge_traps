"""Measure signal-dependent residual closure on real dipole candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from astropy.io import fits
from scipy.stats import chi2
import h5py
import numpy as np

from signed_refit_controls import ROW_REGIONS, COL_REGIONS, region_id
from signed_refit_detection_calibration import TARGET_PER_FIT_FPR
from signed_refit_noise_model import ELECTRONIZE_SCALE, NOISE_MODEL_VERSION
from signed_refit_orientation import classify_orientations
from signed_refit_profile_fitter import (
    ProfileTauFitter,
    SignalDependentProfileTauFitter,
    intensity_model,
)
from signed_refit_variance_model import VARIANCE_MODEL_VERSION
from signed_refit_variance_validation import (
    DEFAULT_DETECTION,
    DEFAULT_MODEL,
    file_sha256,
)


CLOSURE_VERSION = "signed-refit-real-candidate-variance-closure-v2"
DEFAULT_COORDS = Path("signed_refit_orientation_candidates.npz")
DEFAULT_ORIENTATION = Path("signed_refit_orientation_validation.npz")
DEFAULT_VARIANCE = Path("signed_refit_variance_validation.npz")
DEFAULT_OUTPUT = Path("signed_refit_candidate_variance_closure_v2.npz")
DEFAULT_REPORT = Path("signed_refit_candidate_variance_closure_v2.md")
RANDOM_SEED = 2026061308
SAMPLES_PER_TEMPERATURE_SPLIT_QUARTILE = 16
MAX_CALIBRATION_ROOT_SAMPLE = 1200
MAX_OVERDISPERSION = 100.0
OVERDISPERSION_BISECTION_STEPS = 7
AMPLITUDE_BIN_COUNT = 4
LOBE_ORDER_CONTRACT = "I=(image[row,col]-image[row-1,col])/2"
EVALUATION_WIDTH_RANGE = (0.88, 1.12)
MAX_WIDTH_SPREAD = 0.12
MAX_EVALUATION_P05 = 0.12
MIN_EVALUATION_FITS = 500


def _decode_paths(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def load_null_scales(path: Path) -> dict[int, float]:
    data = np.load(path, allow_pickle=False)
    return {
        int(temperature): float(scale)
        for temperature, scale in zip(
            np.asarray(data["temperatures"], dtype=int),
            np.asarray(data["null_scales"], dtype=float),
        )
    }


def load_detection_references(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        int(temperature): np.sort(np.asarray(reference, dtype=float))
        for temperature, reference in zip(
            np.asarray(data["temperatures"], dtype=int),
            np.asarray(data["reference_statistics"], dtype=float),
        )
    }


def empirical_survival(
    reference: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    first_ge = np.searchsorted(reference, values, side="left")
    count_ge = reference.size - first_ge
    return (count_ge + 1.0) / (reference.size + 1.0)


def coordinate_split(
    quadrant: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    # Use an avalanched 64-bit hash. Taking the low bit of the previous linear
    # expression reduced exactly to row+col+quadrant parity because every
    # multiplier was odd, creating a detector checkerboard rather than a
    # pseudorandom coordinate split.
    value = (
        rows.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
        ^ cols.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ np.uint64(int(quadrant) + 1) * np.uint64(0x165667B19E3779F9)
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return (value & np.uint64(1)).astype(np.int8)


def extract_candidate_curves(
    temp_group: h5py.Group,
    quadrant: int,
    coords: np.ndarray,
    control_regions: np.ndarray,
    control_splits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract signed curves and excess pair-shot variance.

    The lobe order is pinned to ``I=(row-row_minus_one)/2``. Pair charge uses
    the lobe sum, for which an ideal transfer cancels. The regional training
    control background supplies the matched illumination reference.
    """
    paths = _decode_paths(np.asarray(temp_group["source_fits"]))
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    curves = np.empty((coords.shape[0], len(paths)), dtype=np.float32)
    extra_shot = np.empty_like(curves)
    background = np.asarray(temp_group[f"quad_{quadrant}/background"], dtype=float)
    reference = np.empty((ROW_REGIONS * COL_REGIONS, len(paths)), dtype=float)
    for region in range(ROW_REGIONS * COL_REGIONS):
        selected = (control_regions == region) & (control_splits == 0)
        reference[region] = 2.0 * np.median(background[selected], axis=0)
    candidate_regions = region_id(rows, cols).astype(int)

    for delay_index, path in enumerate(paths):
        with fits.open(
            path,
            memmap=True,
            do_not_scale_image_data=True,
        ) as hdus:
            image = np.rint(
                hdus[quadrant].data[2:512, 8:3080] / ELECTRONIZE_SCALE
            )
        row_median = np.median(image, axis=1)
        centered = image - row_median[:, None]
        curves[:, delay_index] = (
            centered[rows, cols] - centered[rows - 1, cols]
        ) / 2.0
        pair_charge = image[rows, cols] + image[rows - 1, cols]
        excess_charge = np.maximum(
            pair_charge - reference[candidate_regions, delay_index],
            0.0,
        )
        extra_shot[:, delay_index] = excess_charge / 4.0
    return curves, extra_shot


def stratified_sample(
    selected_indices: np.ndarray,
    amplitudes: np.ndarray,
    splits: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if selected_indices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    abs_amplitude = np.abs(amplitudes[selected_indices])
    edges = np.unique(np.quantile(abs_amplitude, [0.25, 0.5, 0.75]))
    quartiles = np.digitize(abs_amplitude, edges)
    output = []
    for split_value in (0, 1):
        chosen_split = []
        for quartile in np.unique(quartiles):
            pool = selected_indices[
                (splits[selected_indices] == split_value)
                & (quartiles == quartile)
            ]
            if pool.size > SAMPLES_PER_TEMPERATURE_SPLIT_QUARTILE:
                pool = rng.choice(
                    pool,
                    SAMPLES_PER_TEMPERATURE_SPLIT_QUARTILE,
                    replace=False,
                )
            chosen_split.extend(pool.tolist())
        output.append(np.asarray(sorted(chosen_split), dtype=int))
    return output[0], output[1]


def collect_samples(
    model_path: Path,
    coords_path: Path,
    orientation_path: Path,
    detection_references: dict[int, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, int]]:
    rng = np.random.default_rng(RANDOM_SEED)
    coords_npz = np.load(coords_path, allow_pickle=False)
    orientation_npz = np.load(orientation_path, allow_pickle=False)
    orientation_temperatures = np.asarray(
        orientation_npz["temperatures"], dtype=int
    )
    temperature_index = {
        int(temperature): index
        for index, temperature in enumerate(orientation_temperatures)
    }
    orientation_quadrant = np.asarray(
        orientation_npz["candidate_quadrant"], dtype=np.int8
    )
    orientation_rows = np.asarray(
        orientation_npz["candidate_row"], dtype=np.int16
    )
    orientation_cols = np.asarray(
        orientation_npz["candidate_col"], dtype=np.int16
    )
    orientation_single = np.asarray(
        orientation_npz["candidate_single_trap_eligible"], dtype=bool
    )
    orientation_amplitude = np.asarray(
        orientation_npz["candidate_amplitude"], dtype=float
    )
    orientation_significant = np.asarray(
        orientation_npz["candidate_significant"], dtype=bool
    )
    orientation_sign = np.asarray(
        orientation_npz["candidate_sign"], dtype=np.int8
    )
    orientation_structured = np.asarray(
        orientation_npz["candidate_structured_background"], dtype=bool
    )
    physical_significant = (
        orientation_significant
        & np.isfinite(orientation_amplitude)
        & (np.abs(orientation_amplitude) <= 1.0)
    )
    physical_classification = classify_orientations(
        orientation_sign,
        physical_significant,
        structured_background=orientation_structured,
    )
    physical_single = np.asarray(
        physical_classification["single_trap_eligible"], dtype=bool
    )
    orientation_count = np.asarray(
        orientation_npz["candidate_significant_count"], dtype=np.int16
    )
    orientation_fraction = np.asarray(
        orientation_npz["candidate_dominant_fraction"], dtype=float
    )
    orientation_indices = {}
    for quadrant in range(4):
        indices = np.flatnonzero(orientation_quadrant == quadrant)
        coords = np.asarray(coords_npz[f"quad_idx_{quadrant}"], dtype=np.int16)
        expected = np.column_stack(
            (orientation_rows[indices], orientation_cols[indices])
        ).astype(np.int16)
        if not np.array_equal(coords, expected):
            raise ValueError(
                f"Candidate coordinates/order differ from Step 8 in quadrant "
                f"{quadrant}"
            )
        orientation_indices[quadrant] = indices
    samples = []
    selection_summary = {}
    with h5py.File(model_path, "r") as model:
        if str(model.attrs.get("version", "")) != NOISE_MODEL_VERSION:
            raise ValueError("Unexpected noise model version")
        global_quadrants = np.asarray(model["controls/quadrant"], dtype=np.int8)
        global_regions = np.asarray(model["controls/region"], dtype=np.int8)
        global_splits = np.asarray(model["controls/split"], dtype=np.int8)
        temp_names = sorted(
            (name for name in model if name.startswith("temp_")),
            key=lambda value: int(value.split("_")[1]),
        )
        payloads = []
        for temp_name in temp_names:
            temperature = int(temp_name.split("_")[1])
            temp_group = model[temp_name]
            seconds = np.asarray(temp_group["seconds"], dtype=float)
            for quadrant in range(4):
                coords = np.asarray(
                    coords_npz[f"quad_idx_{quadrant}"],
                    dtype=np.int16,
                )
                rows = coords[:, 0]
                cols = coords[:, 1]
                regions = region_id(rows, cols).astype(np.int8)
                splits = coordinate_split(quadrant, rows, cols)
                qsel = global_quadrants == quadrant
                curves, extra_shot = extract_candidate_curves(
                    temp_group,
                    quadrant,
                    coords,
                    global_regions[qsel],
                    global_splits[qsel],
                )
                statistic = np.empty(coords.shape[0], dtype=float)
                amplitude = np.empty(coords.shape[0], dtype=float)
                tau = np.empty(coords.shape[0], dtype=float)
                for region in range(ROW_REGIONS * COL_REGIONS):
                    selected = regions == region
                    if not np.any(selected):
                        continue
                    rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                    fitter = ProfileTauFitter(
                        seconds,
                        np.asarray(rgroup["covariance"], dtype=float),
                        null_template=np.asarray(rgroup["null_template"], dtype=float),
                    )
                    result = fitter.batch_profile_statistic(curves[selected])
                    statistic[selected] = result["delta_chi2"]
                    amplitude[selected] = result["amplitude"]
                    tau[selected] = result["tau"]
                detected = np.flatnonzero(
                    (
                        empirical_survival(
                            detection_references[temperature],
                            statistic,
                        )
                        <= TARGET_PER_FIT_FPR
                    )
                    & np.isfinite(amplitude)
                    & (np.abs(amplitude) <= 1.0)
                )
                payloads.append(
                    {
                        "temperature": temperature,
                        "temp_name": temp_name,
                        "quadrant": quadrant,
                        "seconds": seconds,
                        "coords": coords,
                        "rows": rows,
                        "cols": cols,
                        "regions": regions,
                        "splits": splits,
                        "curves": curves,
                        "extra_shot": extra_shot,
                        "statistic": statistic,
                        "amplitude": amplitude,
                        "tau": tau,
                        "detected": detected,
                        "orientation_indices": orientation_indices[quadrant],
                        "orientation_temperature_index": temperature_index[
                            temperature
                        ],
                    }
                )

        for item in payloads:
            temperature = item["temperature"]
            quadrant = item["quadrant"]
            detected = item["detected"]
            global_indices = item["orientation_indices"]
            temp_index = item["orientation_temperature_index"]
            strict_mask = (
                physical_single[global_indices]
                & physical_significant[global_indices, temp_index]
            )
            stable_current = np.flatnonzero(strict_mask)
            recomputed_mask = np.zeros(item["coords"].shape[0], dtype=bool)
            recomputed_mask[detected] = True
            if np.any(strict_mask & ~recomputed_mask):
                missing = np.flatnonzero(strict_mask & ~recomputed_mask)
                first = int(missing[0])
                global_index = int(global_indices[first])
                raise ValueError(
                    f"Step 8 accepted candidates do not reproduce at "
                    f"{temperature} K Q{quadrant}: {missing.size} missing; "
                    f"first local/global={first}/{global_index}, "
                    f"stored/current amplitude="
                    f"{orientation_amplitude[global_index, temp_index]:.9g}/"
                    f"{item['amplitude'][first]:.9g}, statistic/"
                    f"{item['statistic'][first]:.9g}/"
                    f"empirical-p="
                    f"{empirical_survival(detection_references[temperature], item['statistic'][first]):.9g}"
                )
            calibration, evaluation = stratified_sample(
                stable_current,
                item["amplitude"],
                item["splits"],
                rng,
            )
            summary = selection_summary.setdefault(
                str(temperature),
                {
                    "detected_physical_amplitude": 0,
                    "stable_orientation_selected": 0,
                    "calibration_sample": 0,
                    "evaluation_sample": 0,
                },
            )
            summary["detected_physical_amplitude"] += int(detected.size)
            summary["stable_orientation_selected"] += int(stable_current.size)
            summary["calibration_sample"] += int(calibration.size)
            summary["evaluation_sample"] += int(evaluation.size)
            temp_group = model[item["temp_name"]]
            for split_value, indices in enumerate((calibration, evaluation)):
                for index in indices:
                    region = int(item["regions"][index])
                    rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                    samples.append(
                        {
                            "temperature": temperature,
                            "quadrant": quadrant,
                            "region": region,
                            "row": int(item["rows"][index]),
                            "col": int(item["cols"][index]),
                            "split": split_value,
                            "seconds": item["seconds"].copy(),
                            "curve": item["curves"][index].astype(float),
                            "extra_shot": item["extra_shot"][index].astype(float),
                            "covariance": np.asarray(
                                rgroup["covariance"], dtype=float
                            ),
                            "template": np.asarray(
                                rgroup["null_template"], dtype=float
                            ),
                            "initial_amplitude": float(
                                item["amplitude"][index]
                            ),
                            "initial_tau": float(item["tau"][index]),
                            "detection_statistic": float(
                                item["statistic"][index]
                            ),
                            "significant_temperature_count": int(
                                orientation_count[global_indices[index]]
                            ),
                            "orientation_fraction": float(
                                orientation_fraction[global_indices[index]]
                            ),
                        }
                    )
        for temperature in sorted(int(key) for key in selection_summary):
            summary = selection_summary[str(temperature)]
            print(
                f"{temperature} K: selected "
                f"{summary['detected_physical_amplitude']}; stable "
                f"{summary['stable_orientation_selected']}; sampled "
                f"{summary['calibration_sample']}/"
                f"{summary['evaluation_sample']}"
            )
    population_summary = {
        "step8_single_trap_count": int(np.count_nonzero(orientation_single)),
        "physical_single_trap_count": int(np.count_nonzero(physical_single)),
    }
    return samples, selection_summary, population_summary


def fit_sample(
    sample: dict[str, object],
    null_scale: float,
    overdispersion: float,
) -> dict[str, object]:
    fitter = SignalDependentProfileTauFitter(
        sample["seconds"],
        sample["covariance"],
        null_template=sample["template"],
        null_scale=null_scale,
        pump_overdispersion=overdispersion,
        extra_pair_shot=sample["extra_shot"],
    )
    result = fitter.fit(sample["curve"])
    model = intensity_model(
        sample["seconds"],
        result["amplitude"],
        result["tau"],
        result["offset"],
        sample["template"],
    )
    return {
        **{
            key: sample[key]
            for key in (
                "temperature",
                "quadrant",
                "region",
                "row",
                "col",
                "split",
            )
        },
        "amplitude": float(result["amplitude"]),
        "tau": float(result["tau"]),
        "chi2": float(result["chi2"]),
        "dof": int(result["dof"]),
        "p_value": float(result["fit_p_value"]),
        "boundary_limited": bool(result["boundary_limited"]),
        "multimodal": bool(result["multimodal"]),
        "variance_converged": bool(result["variance_converged"]),
        "residual": np.asarray(sample["curve"] - model, dtype=float),
        "null_component": (
            null_scale * np.asarray(sample["covariance"], dtype=float)
            + np.diag(np.asarray(sample["extra_shot"], dtype=float))
        ),
        "pump_component": np.diag(
            np.asarray(result["pumping_variance"], dtype=float)
        ),
    }


def eligible_fits(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        item
        for item in records
        if not item["boundary_limited"]
        and not item["multimodal"]
        and item["variance_converged"]
    ]


def fit_width(records: list[dict[str, object]]) -> float:
    selected = eligible_fits(records)
    return float(
        np.sqrt(
            np.sum([item["chi2"] for item in selected])
            / np.sum([item["dof"] for item in selected])
        )
    )


def fit_samples_at_factor(
    samples: list[dict[str, object]],
    null_scales: dict[int, float],
    factor: float,
) -> list[dict[str, object]]:
    return [
        fit_sample(item, null_scales[item["temperature"]], factor)
        for item in samples
    ]


def estimate_overdispersion_bin(
    samples: list[dict[str, object]],
    null_scales: dict[int, float],
) -> dict[str, object]:
    one_fits = fit_samples_at_factor(samples, null_scales, 1.0)
    width_at_one = fit_width(one_fits)
    if width_at_one <= 1.0:
        factor = 1.0
        final_fits = one_fits
        width_at_max = fit_width(
            fit_samples_at_factor(samples, null_scales, MAX_OVERDISPERSION)
        )
        hit_upper_bound = False
    else:
        max_fits = fit_samples_at_factor(
            samples,
            null_scales,
            MAX_OVERDISPERSION,
        )
        width_at_max = fit_width(max_fits)
        if width_at_max > 1.0:
            factor = MAX_OVERDISPERSION
            final_fits = max_fits
            hit_upper_bound = True
        else:
            lower = 1.0
            upper = MAX_OVERDISPERSION
            final_fits = max_fits
            for _ in range(OVERDISPERSION_BISECTION_STEPS):
                factor = 0.5 * (lower + upper)
                trial_fits = fit_samples_at_factor(
                    samples,
                    null_scales,
                    factor,
                )
                width = fit_width(trial_fits)
                final_fits = trial_fits
                if width > 1.0:
                    lower = factor
                else:
                    upper = factor
            factor = 0.5 * (lower + upper)
            final_fits = fit_samples_at_factor(
                samples,
                null_scales,
                factor,
            )
            hit_upper_bound = False
    selected = eligible_fits(final_fits)
    return {
        "input_count": len(samples),
        "fit_count": len(selected),
        "closure_width_at_one": width_at_one,
        "closure_width_at_max": width_at_max,
        "estimated_factor": float(factor),
        "final_refit_width": fit_width(final_fits),
        "hit_upper_bound": hit_upper_bound,
    }


def estimate_amplitude_stratified_overdispersion(
    calibration_samples: list[dict[str, object]],
    null_scales: dict[int, float],
) -> dict[str, object]:
    amplitudes = np.abs(
        np.asarray([item["initial_amplitude"] for item in calibration_samples])
    )
    edges = np.quantile(
        amplitudes,
        np.arange(1, AMPLITUDE_BIN_COUNT) / AMPLITUDE_BIN_COUNT,
    )
    bins = np.digitize(amplitudes, edges)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    per_bin_limit = MAX_CALIBRATION_ROOT_SAMPLE // AMPLITUDE_BIN_COUNT
    summaries = []
    for bin_index in range(AMPLITUDE_BIN_COUNT):
        indices = np.flatnonzero(bins == bin_index)
        if indices.size > per_bin_limit:
            indices = rng.choice(indices, per_bin_limit, replace=False)
        selected = [calibration_samples[index] for index in sorted(indices)]
        print(
            f"Calibrating amplitude bin {bin_index + 1}/"
            f"{AMPLITUDE_BIN_COUNT} "
            f"with {len(selected)} candidates",
            flush=True,
        )
        summary = estimate_overdispersion_bin(selected, null_scales)
        summary["amplitude_min"] = (
            0.0 if bin_index == 0 else float(edges[bin_index - 1])
        )
        summary["amplitude_max"] = (
            1.0
            if bin_index == AMPLITUDE_BIN_COUNT - 1
            else float(edges[bin_index])
        )
        summary["amplitude_median"] = float(np.median(amplitudes[indices]))
        summaries.append(summary)
    return {
        "amplitude_edges": [float(value) for value in edges],
        "bins": summaries,
        "estimated_factors": [
            float(item["estimated_factor"]) for item in summaries
        ],
        "hit_upper_bound": bool(
            any(item["hit_upper_bound"] for item in summaries)
        ),
        "fit_count": int(sum(item["fit_count"] for item in summaries)),
    }


def factor_for_sample(
    sample: dict[str, object],
    calibration: dict[str, object],
) -> float:
    index = int(
        np.digitize(
            abs(float(sample["initial_amplitude"])),
            np.asarray(calibration["amplitude_edges"], dtype=float),
        )
    )
    return float(calibration["estimated_factors"][index])


def summarize_evaluation(records: list[dict[str, object]]) -> dict[str, object]:
    eligible = [
        item
        for item in records
        if not item["boundary_limited"]
        and not item["multimodal"]
        and item["variance_converged"]
    ]
    amplitudes = np.abs(np.asarray([item["amplitude"] for item in eligible]))
    edges = np.quantile(amplitudes, [0.25, 0.5, 0.75])
    quartiles = np.digitize(amplitudes, edges)
    summary = {}
    widths = []
    for quartile in range(4):
        selected = [
            item
            for index, item in enumerate(eligible)
            if quartiles[index] == quartile
        ]
        width = float(
            np.sqrt(
                np.sum([item["chi2"] for item in selected])
                / np.sum([item["dof"] for item in selected])
            )
        )
        widths.append(width)
        summary[str(quartile + 1)] = {
            "count": len(selected),
            "amplitude_median": float(
                np.median(amplitudes[quartiles == quartile])
            ),
            "closure_width": width,
            "p05_rate": float(
                np.mean([item["p_value"] < 0.05 for item in selected])
            ),
        }
    return {
        "input_count": len(records),
        "eligible_count": len(eligible),
        "eligible_rate": float(len(eligible) / len(records)),
        "aggregate_width": float(
            np.sqrt(
                np.sum([item["chi2"] for item in eligible])
                / np.sum([item["dof"] for item in eligible])
            )
        ),
        "aggregate_p05_rate": float(
            np.mean([item["p_value"] < 0.05 for item in eligible])
        ),
        "amplitude_quartiles": summary,
        "width_spread": float(max(widths) - min(widths)),
    }


def evaluate_acceptance(
    overdispersion: dict[str, object],
    evaluation: dict[str, object],
) -> dict[str, object]:
    failures = []
    if overdispersion["hit_upper_bound"]:
        failures.append("overdispersion hit the calibration upper bound")
    if evaluation["eligible_count"] < MIN_EVALUATION_FITS:
        failures.append(
            f"only {evaluation['eligible_count']} eligible evaluation fits"
        )
    widths = []
    for quartile, item in evaluation["amplitude_quartiles"].items():
        widths.append(item["closure_width"])
        if not EVALUATION_WIDTH_RANGE[0] <= item["closure_width"] <= EVALUATION_WIDTH_RANGE[1]:
            failures.append(
                f"amplitude quartile {quartile} width "
                f"{item['closure_width']:.3f}"
            )
        if item["p05_rate"] > MAX_EVALUATION_P05:
            failures.append(
                f"amplitude quartile {quartile} p05 "
                f"{item['p05_rate']:.3f}"
            )
    if max(widths) - min(widths) > MAX_WIDTH_SPREAD:
        failures.append(
            f"amplitude width spread {max(widths) - min(widths):.3f}"
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def run_closure(
    model_path: Path,
    coords_path: Path,
    orientation_path: Path,
    variance_path: Path,
    detection_path: Path,
) -> dict[str, object]:
    detection_references = load_detection_references(detection_path)
    null_scales = load_null_scales(variance_path)
    samples, selection, population = collect_samples(
        model_path,
        coords_path,
        orientation_path,
        detection_references,
    )
    fold_metrics = []
    evaluation_fits = []
    for held_out_split in (0, 1):
        calibration_samples = [
            item for item in samples if item["split"] != held_out_split
        ]
        evaluation_samples = [
            item for item in samples if item["split"] == held_out_split
        ]
        print(
            f"Cross-fit fold {held_out_split + 1}/2: calibrating "
            f"{len(calibration_samples)} candidates"
        )
        fold = estimate_amplitude_stratified_overdispersion(
            calibration_samples,
            null_scales,
        )
        fold["held_out_split"] = held_out_split
        fold["evaluation_input_count"] = len(evaluation_samples)
        fold_metrics.append(fold)
        print(
            f"Cross-fit fold {held_out_split + 1}/2 factors: "
            + ", ".join(
                f"{value:.4f}" for value in fold["estimated_factors"]
            )
        )
        print(
            f"Cross-fit fold {held_out_split + 1}/2: fitting "
            f"{len(evaluation_samples)} held-out candidates"
        )
        evaluation_fits.extend(
            fit_sample(
                item,
                null_scales[item["temperature"]],
                factor_for_sample(item, fold),
            )
            for item in evaluation_samples
        )
    factor_metrics = {
        "crossfit_folds": fold_metrics,
        "amplitude_edges": np.mean(
            [fold["amplitude_edges"] for fold in fold_metrics],
            axis=0,
        ).tolist(),
        "estimated_factors": np.exp(
            np.mean(
                np.log(
                    [fold["estimated_factors"] for fold in fold_metrics]
                ),
                axis=0,
            )
        ).tolist(),
        "hit_upper_bound": bool(
            any(fold["hit_upper_bound"] for fold in fold_metrics)
        ),
        "fit_count": int(sum(fold["fit_count"] for fold in fold_metrics)),
        "method": (
            "two-fold coordinate cross-fit; reported final edges are arithmetic "
            "fold means and final factors are geometric fold means"
        ),
    }
    evaluation = summarize_evaluation(evaluation_fits)
    metrics = {
        "closure_version": CLOSURE_VERSION,
        "variance_model_version": VARIANCE_MODEL_VERSION,
        "noise_model_sha256": file_sha256(model_path),
        "candidate_coords_sha256": file_sha256(coords_path),
        "orientation_validation_sha256": file_sha256(orientation_path),
        "variance_validation_sha256": file_sha256(variance_path),
        "detection_calibration_sha256": file_sha256(detection_path),
        "random_seed": RANDOM_SEED,
        "lobe_order_contract": LOBE_ORDER_CONTRACT,
        "electronize_scale": ELECTRONIZE_SCALE,
        "selection_by_temperature": selection,
        **population,
        "calibration": factor_metrics,
        "evaluation": evaluation,
    }
    metrics["acceptance"] = evaluate_acceptance(factor_metrics, evaluation)
    return metrics


def write_outputs(
    metrics: dict[str, object],
    output_path: Path,
    report_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(metrics, sort_keys=True)),
    )
    calibration = metrics["calibration"]
    evaluation = metrics["evaluation"]
    lines = [
        "# Signed Refit Real-Candidate Variance Closure",
        "",
        f"- Closure version: `{CLOSURE_VERSION}`",
        f"- Noise-model SHA-256: `{metrics['noise_model_sha256']}`",
        f"- Candidate-coordinate SHA-256: `{metrics['candidate_coords_sha256']}`",
        f"- Orientation-validation SHA-256: "
        f"`{metrics['orientation_validation_sha256']}`",
        f"- Lobe-order contract: `{LOBE_ORDER_CONTRACT}`",
        f"- Electronization scale: {ELECTRONIZE_SCALE:.1f} ADU/e- (global).",
        f"- Acceptance status: **{metrics['acceptance']['status']}**",
        f"- Step 8 orientation-only single-trap sites: "
        f"{metrics['step8_single_trap_count']:,}.",
        f"- Physical-amplitude single-trap sites used here: "
        f"{metrics['physical_single_trap_count']:,}.",
        "",
        "## Split and selection",
        "",
        "Candidate coordinates are split by a fixed coordinate hash before their "
        "curves are examined. The sample is restricted to the frozen Step 8 "
        "single-trap policy after requiring the paper-model coefficient "
        "`|D_t P_c| <= 1`: at least four empirically significant physical "
        "temperatures, no accepted opposite-sign temperature, and no persistent-"
        "horizontal pixel overlap. No residual-goodness criterion enters this "
        "selection. "
        "Each split is sampled uniformly across fitted-amplitude quartiles at "
        "every temperature.",
        "",
        f"- Calibration fits used for overdispersion: {calibration['fit_count']}.",
        f"- Evaluation input/eligible fits: {evaluation['input_count']}/"
        f"{evaluation['eligible_count']} ({evaluation['eligible_rate']:.2%}).",
        "",
        "## Pumping overdispersion",
        "",
        "- Null-fit amplitude edges: "
        + ", ".join(
            f"{value:.4f}" for value in calibration["amplitude_edges"]
        )
        + ".",
        "- Measured multipliers `phi`: "
        + ", ".join(
            f"{value:.4f}" for value in calibration["estimated_factors"]
        )
        + ".",
        f"- Any upper-bound hit: {calibration['hit_upper_bound']}.",
        "",
        "The frozen amplitude-bin multiplier applies only to `3000 q(1-q)`. "
        "The bin is chosen from the null-covariance detection amplitude before "
        "the signal-dependent refit. Null covariance scale and excess pair-shot "
        "variance remain separate terms.",
        "",
        f"- Method: {calibration['method']}.",
        "",
        "| Held-out split | Null-fit amplitude bin | Median |A| | Fits | phi | Refit width |",
        "|---:|---:|---:|---:|---:|---:|",
        "",
        "## Evaluation closure",
        "",
        f"- Aggregate residual width: {evaluation['aggregate_width']:.3f}.",
        f"- Aggregate nominal `p<0.05` rate: "
        f"{evaluation['aggregate_p05_rate']:.2%}.",
        f"- Fitted-amplitude width spread: {evaluation['width_spread']:.3f}.",
        "",
        "| Fitted-amplitude quartile | Median |A| | Fits | Closure width | p<0.05 |",
        "|---:|---:|---:|---:|---:|",
    ]
    calibration_rows = []
    for fold in calibration["crossfit_folds"]:
        for index, item in enumerate(fold["bins"], start=1):
            calibration_rows.append(
                f"| {fold['held_out_split']} | {index} | "
                f"{item['amplitude_median']:.4f} | "
                f"{item['fit_count']} | {item['estimated_factor']:.4f} | "
                f"{item['final_refit_width']:.3f} |"
            )
    evaluation_header = lines.index("## Evaluation closure")
    lines[evaluation_header:evaluation_header] = calibration_rows + [""]
    for quartile, item in evaluation["amplitude_quartiles"].items():
        lines.append(
            f"| {quartile} | {item['amplitude_median']:.4f} | "
            f"{item['count']} | {item['closure_width']:.3f} | "
            f"{item['p05_rate']:.2%} |"
        )
    lines.extend(["", "## Acceptance gate", ""])
    if metrics["acceptance"]["status"] == "PASS":
        lines.extend(
            [
                "- PASS: real-candidate residual width closes in every fitted-amplitude quartile.",
                "- PASS: no material amplitude trend remains after the measured pumping term.",
                "- PASS: amplitude-stratified overdispersion was fixed on disjoint coordinates.",
            ]
        )
    else:
        lines.append("- **FAIL:** " + "; ".join(metrics["acceptance"]["failures"]))
    lines.extend(
        [
            "",
            "Nominal chi-square tails remain a goodness-of-fit diagnostic; Step 6 "
            "empirical calibration continues to define detection significance.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def validate_output(
    output_path: Path,
    model_path: Path,
    coords_path: Path,
    orientation_path: Path,
    variance_path: Path,
    detection_path: Path,
) -> dict[str, object]:
    metrics = json.loads(
        str(np.load(output_path, allow_pickle=False)["metadata_json"])
    )
    errors = []
    expected = {
        "noise_model_sha256": file_sha256(model_path),
        "candidate_coords_sha256": file_sha256(coords_path),
        "orientation_validation_sha256": file_sha256(orientation_path),
        "variance_validation_sha256": file_sha256(variance_path),
        "detection_calibration_sha256": file_sha256(detection_path),
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            errors.append(f"{key} mismatch")
    if metrics.get("closure_version") != CLOSURE_VERSION:
        errors.append("closure version mismatch")
    if metrics.get("acceptance", {}).get("status") != "PASS":
        errors.append(
            "acceptance failed: "
            + "; ".join(metrics.get("acceptance", {}).get("failures", []))
        )
    if errors:
        raise ValueError("Candidate variance closure failed:\n- " + "\n- ".join(errors))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument(
        "--orientation", type=Path, default=DEFAULT_ORIENTATION
    )
    parser.add_argument("--variance", type=Path, default=DEFAULT_VARIANCE)
    parser.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        metrics = validate_output(
            args.output,
            args.model,
            args.coords,
            args.orientation,
            args.variance,
            args.detection,
        )
    else:
        metrics = run_closure(
            args.model,
            args.coords,
            args.orientation,
            args.variance,
            args.detection,
        )
        write_outputs(metrics, args.output, args.report)
        if metrics["acceptance"]["status"] != "PASS":
            raise RuntimeError(
                "Candidate variance closure failed: "
                + "; ".join(metrics["acceptance"]["failures"])
            )
    print(
        f"{metrics['acceptance']['status']}: phi "
        + ",".join(
            f"{value:.4f}"
            for value in metrics["calibration"]["estimated_factors"]
        )
        + "; "
        f"evaluation width {metrics['evaluation']['aggregate_width']:.3f}"
    )


if __name__ == "__main__":
    main()

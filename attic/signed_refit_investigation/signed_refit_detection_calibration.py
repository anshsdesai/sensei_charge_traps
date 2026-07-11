"""Empirically calibrate the signed profile-tau detection statistic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.stats import beta

from finder_null_test import find_pairs
from signed_refit_controls import (
    IMAGE_SHAPE,
    ROW_REGIONS,
    COL_REGIONS,
    CONTROL_VERSION,
    region_id,
)
from signed_refit_noise_model import ELECTRONIZE_SCALE, NOISE_MODEL_VERSION
from signed_refit_profile_fitter import (
    PROFILE_FITTER_VERSION,
    ProfileTauFitter,
    file_sha256,
)


DETECTION_CALIBRATION_VERSION = "signed-refit-detection-calibration-v1"
DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_CONTROLS = Path("signed_refit_control_pairs.npz")
DEFAULT_OUTPUT = Path("signed_refit_detection_calibration.npz")
DEFAULT_REPORT = Path("signed_refit_detection_calibration.md")

RANDOM_SEED = 2026061306
TARGET_PER_FIT_FPR = 0.001
LEGACY_DELTA_CHI2 = 11.83
PRELIMINARY_CANDIDATE_COUNT = 9333
PRODUCTION_TEMPERATURE_COUNT = 23
NEAR_DEFECT_DILATION = 5
MAX_NEAR_DEFECT_PER_QUADRANT = 4096

MAX_AGGREGATE_EVALUATION_FPR = 0.0015
MAX_TEMPERATURE_EVALUATION_FPR = 0.0030
MAX_QUADRANT_EVALUATION_FPR = 0.0020
MAX_REGION_EVALUATION_FPR = 0.0040
MAX_STRUCTURED_FPR = 0.0100


def _decode_paths(values: np.ndarray) -> list[str]:
    return [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in values
    ]


def _empirical_survival(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    first_ge = np.searchsorted(reference, values, side="left")
    count_ge = reference.size - first_ge
    return (count_ge + 1.0) / (reference.size + 1.0)


def empirical_threshold(reference: np.ndarray, target_p: float) -> float:
    """Lowest observed statistic whose add-one empirical p meets target_p."""
    reference = np.sort(np.asarray(reference, dtype=float))
    unique = np.unique(reference)
    pvalues = _empirical_survival(reference, unique)
    valid = unique[pvalues <= target_p]
    if valid.size == 0:
        raise ValueError(
            f"Target p={target_p} is below the attainable finite-sample "
            f"resolution 1/(n+1)={1.0 / (reference.size + 1):.6g}"
        )
    return float(valid[0])


def empirical_p_value(
    statistic: float | np.ndarray,
    temperature: int,
    calibration_path: Path | str = DEFAULT_OUTPUT,
) -> np.ndarray:
    """Return the finite-sample empirical profile-search p-value."""
    data = np.load(calibration_path, allow_pickle=False)
    temperatures = np.asarray(data["temperatures"], dtype=int)
    matches = np.flatnonzero(temperatures == int(temperature))
    if matches.size != 1:
        raise ValueError(f"Temperature {temperature} is not calibrated")
    reference = np.asarray(data["reference_statistics"][matches[0]], dtype=float)
    return _empirical_survival(reference, np.asarray(statistic))


def passes_detection_threshold(
    statistic: float | np.ndarray,
    temperature: int,
    calibration_path: Path | str = DEFAULT_OUTPUT,
) -> np.ndarray:
    data = np.load(calibration_path, allow_pickle=False)
    temperatures = np.asarray(data["temperatures"], dtype=int)
    matches = np.flatnonzero(temperatures == int(temperature))
    if matches.size != 1:
        raise ValueError(f"Temperature {temperature} is not calibrated")
    threshold = float(data["thresholds"][matches[0]])
    return np.asarray(statistic) >= threshold


def _control_split_within_region(
    quadrants: np.ndarray,
    regions: np.ndarray,
    original_split: np.ndarray,
) -> np.ndarray:
    """Split the Step 4 validation sites 64/64 without using their curves."""
    output = np.full(original_split.size, -1, dtype=np.int8)
    for quadrant in range(4):
        for region in range(ROW_REGIONS * COL_REGIONS):
            indices = np.flatnonzero(
                (quadrants == quadrant)
                & (regions == region)
                & (original_split == 1)
            )
            if indices.size != 128:
                raise ValueError(
                    f"Expected 128 validation controls in Q{quadrant}/R{region}, "
                    f"found {indices.size}"
                )
            output[indices[:64]] = 0
            output[indices[64:]] = 1
    return output


def _structured_coordinates(
    model: h5py.File,
    controls: np.lib.npyio.NpzFile,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build frozen horizontal-trigger and near-defect coordinate classes."""
    horizontal_sets = [set() for _ in range(4)]
    for temp_name in sorted(
        (name for name in model if name.startswith("temp_")),
        key=lambda value: int(value.split("_")[1]),
    ):
        group = model[temp_name]
        delays = np.asarray(group["dtph"], dtype=int)
        paths = _decode_paths(np.asarray(group["source_fits"]))
        counts = [defaultdict(set) for _ in range(4)]
        for delay, path in zip(delays, paths):
            with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
                for quadrant in range(4):
                    image = np.rint(
                        hdus[quadrant].data[2:512, 8:3080] / ELECTRONIZE_SCALE
                    )
                    for coordinate in find_pairs(image, axis=1):
                        counts[quadrant][coordinate].add(int(delay))
        for quadrant in range(4):
            valid = controls[f"valid_pair_mask_q{quadrant}"]
            horizontal_sets[quadrant].update(
                coordinate
                for coordinate, seen_delays in counts[quadrant].items()
                if len(seen_delays) > 1
                and 0 <= coordinate[0] < IMAGE_SHAPE[0]
                and 0 <= coordinate[1] < IMAGE_SHAPE[1]
                and valid[coordinate]
            )

    horizontal = {
        quadrant: np.asarray(sorted(values), dtype=np.int16).reshape(-1, 2)
        for quadrant, values in enumerate(horizontal_sets)
    }

    rng = np.random.default_rng(RANDOM_SEED)
    near_defect = {}
    for quadrant in range(4):
        valid = np.asarray(controls[f"valid_pair_mask_q{quadrant}"], dtype=bool)
        defect = np.asarray(controls[f"defect_mask_q{quadrant}"], dtype=bool)
        near = valid & ndimage.binary_dilation(
            defect,
            iterations=NEAR_DEFECT_DILATION,
        )
        coordinates = np.argwhere(near)
        rng.shuffle(coordinates)
        coordinates = coordinates[:MAX_NEAR_DEFECT_PER_QUADRANT]
        near_defect[quadrant] = coordinates.astype(np.int16)
    return horizontal, near_defect


def _flatten_coordinate_class(
    coordinates: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        np.concatenate(quadrants),
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(regions),
    )


def _profile_by_region(
    curves: np.ndarray,
    coordinate_regions: np.ndarray,
    temp_group: h5py.Group,
    quadrant: int,
) -> np.ndarray:
    statistics = np.empty(curves.shape[0], dtype=np.float32)
    for region in np.unique(coordinate_regions):
        selected = coordinate_regions == region
        region_group = temp_group[f"quad_{quadrant}/region_{int(region)}"]
        fitter = ProfileTauFitter(
            np.asarray(temp_group["seconds"], dtype=float),
            np.asarray(region_group["covariance"], dtype=float),
            null_template=np.asarray(region_group["null_template"], dtype=float),
        )
        result = fitter.batch_profile_statistic(curves[selected])
        statistics[selected] = result["delta_chi2"].astype(np.float32)
    return statistics


def _extract_structured_statistics(
    model: h5py.File,
    coordinates: dict[int, np.ndarray],
    temperatures: np.ndarray,
) -> np.ndarray:
    q_offsets = np.cumsum(
        [0] + [coordinates[quadrant].shape[0] for quadrant in range(4)]
    )
    statistics = np.full(
        (int(q_offsets[-1]), temperatures.size),
        np.nan,
        dtype=np.float32,
    )
    coordinate_regions = {
        quadrant: region_id(
            coordinates[quadrant][:, 0],
            coordinates[quadrant][:, 1],
        )
        for quadrant in range(4)
    }

    for temp_index, temperature in enumerate(temperatures):
        temp_group = model[f"temp_{int(temperature)}"]
        paths = _decode_paths(np.asarray(temp_group["source_fits"]))
        for quadrant in range(4):
            coords = coordinates[quadrant]
            if coords.size == 0:
                continue
            curves = np.empty((coords.shape[0], len(paths)), dtype=np.float32)
            rows = coords[:, 0]
            cols = coords[:, 1]
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
                    image = image - row_median[:, None]
                    curves[:, delay_index] = (
                        image[rows, cols] - image[rows - 1, cols]
                    ) / 2.0
            start, stop = q_offsets[quadrant : quadrant + 2]
            statistics[start:stop, temp_index] = _profile_by_region(
                curves,
                coordinate_regions[quadrant],
                temp_group,
                quadrant,
            )
    return statistics


def _ordinary_statistics(
    model: h5py.File,
    controls: np.lib.npyio.NpzFile,
    temperatures: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    all_quadrants = np.asarray(controls["quadrant"], dtype=np.int8)
    all_regions = np.asarray(controls["region"], dtype=np.int8)
    all_rows = np.asarray(controls["row"], dtype=np.int16)
    all_cols = np.asarray(controls["col"], dtype=np.int16)
    original_split = np.asarray(controls["split"], dtype=np.int8)
    sub_split = _control_split_within_region(
        all_quadrants,
        all_regions,
        original_split,
    )
    validation = original_split == 1
    quadrants = all_quadrants[validation]
    regions = all_regions[validation]
    rows = all_rows[validation]
    cols = all_cols[validation]
    split = sub_split[validation]
    statistics = np.empty((rows.size, temperatures.size), dtype=np.float32)

    q_validation_indices = {
        quadrant: np.flatnonzero(validation & (all_quadrants == quadrant))
        for quadrant in range(4)
    }
    output_q_indices = {
        quadrant: np.flatnonzero(quadrants == quadrant)
        for quadrant in range(4)
    }
    for temp_index, temperature in enumerate(temperatures):
        temp_group = model[f"temp_{int(temperature)}"]
        for quadrant in range(4):
            curves = np.asarray(
                temp_group[f"quad_{quadrant}/intensity"]
            )
            qsel_all = all_quadrants == quadrant
            qsplit = original_split[qsel_all]
            validation_curves = curves[qsplit == 1]
            qregions = all_regions[q_validation_indices[quadrant]]
            statistics[output_q_indices[quadrant], temp_index] = (
                _profile_by_region(
                    validation_curves,
                    qregions,
                    temp_group,
                    quadrant,
                )
            )
    return statistics, quadrants, regions, rows, cols, split


def _binomial_upper_95(successes: int, trials: int) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, trials - successes))


def _rate_summary(
    accepted: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    return {
        str(int(label)): float(np.mean(accepted[labels == label]))
        for label in np.unique(labels)
    }


def build_calibration(
    model_path: Path,
    controls_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    controls = np.load(controls_path, allow_pickle=False)
    if str(controls["version"]) != CONTROL_VERSION:
        raise ValueError("Control artifact version mismatch")

    with h5py.File(model_path, "r") as model:
        if str(model.attrs.get("version", "")) != NOISE_MODEL_VERSION:
            raise ValueError("Noise-model version mismatch")
        temperatures = np.asarray(
            sorted(
                int(name.split("_")[1])
                for name in model
                if name.startswith("temp_")
            ),
            dtype=np.int16,
        )
        horizontal_coords, near_coords = _structured_coordinates(model, controls)
        (
            ordinary_stats,
            ordinary_quadrants,
            ordinary_regions,
            ordinary_rows,
            ordinary_cols,
            ordinary_split,
        ) = _ordinary_statistics(model, controls, temperatures)
        horizontal_stats = _extract_structured_statistics(
            model,
            horizontal_coords,
            temperatures,
        )
        near_stats = _extract_structured_statistics(
            model,
            near_coords,
            temperatures,
        )

    horizontal_q, horizontal_r, horizontal_c, horizontal_region = (
        _flatten_coordinate_class(horizontal_coords)
    )
    near_q, near_r, near_c, near_region = _flatten_coordinate_class(near_coords)

    calibration_mask = ordinary_split == 0
    evaluation_mask = ordinary_split == 1
    reference_statistics = ordinary_stats[calibration_mask].T
    thresholds = np.asarray(
        [
            empirical_threshold(reference, TARGET_PER_FIT_FPR)
            for reference in reference_statistics
        ],
        dtype=np.float64,
    )
    ordinary_pvalues = np.column_stack(
        [
            _empirical_survival(reference, ordinary_stats[:, index])
            for index, reference in enumerate(reference_statistics)
        ]
    ).astype(np.float32)
    horizontal_pvalues = np.column_stack(
        [
            _empirical_survival(reference, horizontal_stats[:, index])
            for index, reference in enumerate(reference_statistics)
        ]
    ).astype(np.float32)
    near_pvalues = np.column_stack(
        [
            _empirical_survival(reference, near_stats[:, index])
            for index, reference in enumerate(reference_statistics)
        ]
    ).astype(np.float32)
    ordinary_accepted = ordinary_stats >= thresholds[None, :]
    horizontal_accepted = horizontal_stats >= thresholds[None, :]
    near_accepted = near_stats >= thresholds[None, :]
    legacy_ordinary = ordinary_stats >= LEGACY_DELTA_CHI2

    evaluation_acceptance = ordinary_accepted[evaluation_mask]
    evaluation_legacy = legacy_ordinary[evaluation_mask]
    evaluation_quadrants = ordinary_quadrants[evaluation_mask]
    evaluation_regions = ordinary_regions[evaluation_mask]

    temperature_rates = {
        str(int(temperature)): float(np.mean(evaluation_acceptance[:, index]))
        for index, temperature in enumerate(temperatures)
    }
    quadrant_rates = _rate_summary(
        evaluation_acceptance.ravel(),
        np.repeat(evaluation_quadrants, temperatures.size),
    )
    region_rates = _rate_summary(
        evaluation_acceptance.ravel(),
        np.repeat(evaluation_regions, temperatures.size),
    )
    aggregate_rate = float(np.mean(evaluation_acceptance))
    horizontal_rate = float(np.mean(horizontal_accepted))
    near_rate = float(np.mean(near_accepted))

    ordinary_site_counts = np.sum(evaluation_acceptance, axis=1)
    horizontal_site_counts = np.sum(horizontal_accepted, axis=1)
    near_site_counts = np.sum(near_accepted, axis=1)
    ordinary_four_or_more = int(np.count_nonzero(ordinary_site_counts >= 4))
    ordinary_site_rate = ordinary_four_or_more / ordinary_site_counts.size
    ordinary_site_upper = _binomial_upper_95(
        ordinary_four_or_more,
        ordinary_site_counts.size,
    )

    failures = []
    if aggregate_rate > MAX_AGGREGATE_EVALUATION_FPR:
        failures.append(f"ordinary aggregate FPR {aggregate_rate:.4%}")
    if max(temperature_rates.values()) > MAX_TEMPERATURE_EVALUATION_FPR:
        failures.append(
            f"maximum temperature FPR {max(temperature_rates.values()):.4%}"
        )
    if max(quadrant_rates.values()) > MAX_QUADRANT_EVALUATION_FPR:
        failures.append(
            f"maximum quadrant FPR {max(quadrant_rates.values()):.4%}"
        )
    if max(region_rates.values()) > MAX_REGION_EVALUATION_FPR:
        failures.append(f"maximum region FPR {max(region_rates.values()):.4%}")
    if horizontal_rate > MAX_STRUCTURED_FPR:
        failures.append(f"horizontal-trigger FPR {horizontal_rate:.4%}")
    if near_rate > MAX_STRUCTURED_FPR:
        failures.append(f"near-defect FPR {near_rate:.4%}")

    metadata = {
        "version": DETECTION_CALIBRATION_VERSION,
        "profile_fitter_version": PROFILE_FITTER_VERSION,
        "noise_model_version": NOISE_MODEL_VERSION,
        "noise_model_sha256": file_sha256(model_path),
        "controls_version": CONTROL_VERSION,
        "controls_sha256": file_sha256(controls_path),
        "random_seed": RANDOM_SEED,
        "statistic": (
            "maximum generalized-least-squares chi-square improvement over the "
            "constant model across the frozen 801-point log-tau profile"
        ),
        "target_per_fit_fpr": TARGET_PER_FIT_FPR,
        "threshold_policy": (
            "temperature-specific lowest observed statistic with add-one "
            "finite-sample empirical p <= 0.001 from 8,192 ordinary-null "
            "calibration curves; independent 8,192 ordinary-null curves "
            "evaluate each temperature"
        ),
        "candidate_temperature_test_count": (
            PRELIMINARY_CANDIDATE_COUNT * PRODUCTION_TEMPERATURE_COUNT
        ),
        "catalog_false_temperature_fit_budget": (
            PRELIMINARY_CANDIDATE_COUNT
            * PRODUCTION_TEMPERATURE_COUNT
            * TARGET_PER_FIT_FPR
        ),
        "ordinary": {
            "calibration_sites": int(np.count_nonzero(calibration_mask)),
            "evaluation_sites": int(np.count_nonzero(evaluation_mask)),
            "aggregate_evaluation_fpr": aggregate_rate,
            "maximum_temperature_fpr": max(temperature_rates.values()),
            "temperature_fpr": temperature_rates,
            "quadrant_fpr": quadrant_rates,
            "region_fpr": region_rates,
            "legacy_11p83_evaluation_fpr": float(np.mean(evaluation_legacy)),
            "sites_with_at_least_four": ordinary_four_or_more,
            "site_fpr_at_least_four": ordinary_site_rate,
            "site_fpr_at_least_four_upper_95": ordinary_site_upper,
            "projected_sites_at_least_four_upper_95": (
                ordinary_site_upper * PRELIMINARY_CANDIDATE_COUNT
            ),
        },
        "horizontal_trigger": {
            "site_count": int(horizontal_stats.shape[0]),
            "curve_count": int(horizontal_stats.size),
            "fpr": horizontal_rate,
            "sites_with_at_least_four": int(
                np.count_nonzero(horizontal_site_counts >= 4)
            ),
        },
        "near_defect": {
            "site_count": int(near_stats.shape[0]),
            "curve_count": int(near_stats.size),
            "fpr": near_rate,
            "sites_with_at_least_four": int(
                np.count_nonzero(near_site_counts >= 4)
            ),
            "dilation_pixels": NEAR_DEFECT_DILATION,
        },
        "thresholds": {
            str(int(temperature)): float(threshold)
            for temperature, threshold in zip(temperatures, thresholds)
        },
        "acceptance_limits": {
            "aggregate_evaluation_fpr": MAX_AGGREGATE_EVALUATION_FPR,
            "temperature_evaluation_fpr": MAX_TEMPERATURE_EVALUATION_FPR,
            "quadrant_evaluation_fpr": MAX_QUADRANT_EVALUATION_FPR,
            "region_evaluation_fpr": MAX_REGION_EVALUATION_FPR,
            "structured_fpr": MAX_STRUCTURED_FPR,
        },
        "acceptance": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
        },
    }

    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        temperatures=temperatures,
        thresholds=thresholds,
        reference_statistics=reference_statistics.astype(np.float32),
        ordinary_statistics=ordinary_stats,
        ordinary_empirical_pvalue=ordinary_pvalues,
        ordinary_quadrant=ordinary_quadrants,
        ordinary_region=ordinary_regions,
        ordinary_row=ordinary_rows,
        ordinary_col=ordinary_cols,
        ordinary_calibration_split=ordinary_split,
        horizontal_statistics=horizontal_stats,
        horizontal_empirical_pvalue=horizontal_pvalues,
        horizontal_quadrant=horizontal_q,
        horizontal_region=horizontal_region,
        horizontal_row=horizontal_r,
        horizontal_col=horizontal_c,
        near_defect_statistics=near_stats,
        near_defect_empirical_pvalue=near_pvalues,
        near_defect_quadrant=near_q,
        near_defect_region=near_region,
        near_defect_row=near_r,
        near_defect_col=near_c,
    )
    write_report(report_path, metadata)
    return metadata


def write_report(path: Path, metadata: dict[str, object]) -> None:
    ordinary = metadata["ordinary"]
    horizontal = metadata["horizontal_trigger"]
    near = metadata["near_defect"]
    lines = [
        "# Signed Refit Detection-Significance Calibration",
        "",
        f"- Calibration version: `{DETECTION_CALIBRATION_VERSION}`",
        f"- Profile fitter: `{PROFILE_FITTER_VERSION}`",
        f"- Noise-model SHA-256: `{metadata['noise_model_sha256']}`",
        f"- Controls SHA-256: `{metadata['controls_sha256']}`",
        f"- Acceptance status: **{metadata['acceptance']['status']}**",
        "",
        "## Statistic and decision rule",
        "",
        "The ranking statistic is the largest generalized-least-squares "
        "`delta chi-square` improvement over a constant curve while scanning the "
        "complete frozen 801-point log-`tau` grid. Because `tau` is undefined when "
        "the amplitude is zero, no Wilks-theorem chi-square interpretation is used.",
        "",
        f"A candidate-temperature fit passes when its finite-sample empirical "
        f"`p <= {TARGET_PER_FIT_FPR}` against the calibration controls for that "
        "temperature. Equivalent temperature-specific statistic thresholds are:",
        "",
        "| T (K) | Delta chi-square threshold | Independent evaluation FPR |",
        "|---:|---:|---:|",
    ]
    for temperature, threshold in metadata["thresholds"].items():
        lines.append(
            f"| {temperature} | {threshold:.3f} | "
            f"{ordinary['temperature_fpr'][temperature]:.3%} |"
        )
    lines.extend(
        [
            "",
            "Each threshold is the lowest observed calibration statistic whose "
            "add-one finite-sample tail probability is at most 0.001. With 8,192 "
            "references this is normally the seventh-largest value. A disjoint "
            "8,192 curves at each temperature are used only for evaluation.",
            "",
            "## False-positive budget",
            "",
            f"- Preliminary candidate-temperature tests: "
            f"{metadata['candidate_temperature_test_count']:,} "
            f"({PRELIMINARY_CANDIDATE_COUNT:,} sites x "
            f"{PRODUCTION_TEMPERATURE_COUNT} temperatures).",
            f"- Target ordinary-null budget: at most "
            f"{metadata['catalog_false_temperature_fit_budget']:.1f} false "
            "temperature fits in that complete preliminary set.",
            "- This is an intensity-fit budget, not a final false-trap claim. Step 7 "
            "must include finder selection, and later steps require multiple "
            "temperature fits and sign/SRH consistency.",
            "",
            "## Independent ordinary-null evaluation",
            "",
            f"- Aggregate FPR: {ordinary['aggregate_evaluation_fpr']:.4%}.",
            f"- Maximum temperature FPR: {ordinary['maximum_temperature_fpr']:.4%}.",
            "- Quadrant FPRs: "
            + ", ".join(
                f"Q{key}={value:.4%}"
                for key, value in ordinary["quadrant_fpr"].items()
            )
            + ".",
            "- Region FPR range: "
            f"{min(ordinary['region_fpr'].values()):.4%}-"
            f"{max(ordinary['region_fpr'].values()):.4%}.",
            f"- Sites passing at least four temperatures: "
            f"{ordinary['sites_with_at_least_four']} of "
            f"{ordinary['evaluation_sites']:,}.",
            "- 95% upper bound projected to 9,333 sites for at least four "
            f"temperature passes: "
            f"{ordinary['projected_sites_at_least_four_upper_95']:.2f} sites.",
            "",
            "## Look-elsewhere effect",
            "",
            f"- The old `delta_chi2 >= {LEGACY_DELTA_CHI2}` rule accepts "
            f"{ordinary['legacy_11p83_evaluation_fpr']:.3%} of independent "
            "ordinary nulls.",
            f"- The calibrated target is {TARGET_PER_FIT_FPR:.3%}.",
            "- The empirical thresholds are therefore substantially higher and vary "
            "with the actual dwell grid and acquisition condition.",
            "",
            "## Structured controls",
            "",
            f"- Persistent horizontal-trigger sites: {horizontal['site_count']} "
            f"sites / {horizontal['curve_count']:,} curves; FPR "
            f"{horizontal['fpr']:.3%}; sites with >=4 passes "
            f"{horizontal['sites_with_at_least_four']}.",
            f"- Near-defect vertical sites: {near['site_count']:,} sites / "
            f"{near['curve_count']:,} curves; FPR {near['fpr']:.3%}; "
            f"sites with >=4 passes {near['sites_with_at_least_four']}.",
            "",
            "Horizontal-trigger coordinates were selected by persistence in at "
            "least two dwell images but evaluated using the normal vertical-pair "
            "profile fit. Near-defect coordinates lie outside all v2 masks but "
            f"within {NEAR_DEFECT_DILATION} pixels of a persistent-defect mask.",
            "",
            "## Acceptance gate",
            "",
        ]
    )
    if metadata["acceptance"]["status"] == "PASS":
        lines.extend(
            [
                "- PASS: the independent ordinary-null FPR meets the stated budget.",
                "- PASS: rates remain within predefined temperature, quadrant, and "
                "region stability limits.",
                "- PASS: horizontal-trigger and near-defect stress controls remain "
                "below the predefined 1% ceiling.",
                "- PASS: thresholds and empirical p-values include the full tau "
                "look-elsewhere search without a Wilks-theorem claim.",
            ]
        )
    else:
        lines.append("- **FAIL:** " + "; ".join(metadata["acceptance"]["failures"]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_calibration(
    output_path: Path,
    model_path: Path,
    controls_path: Path,
) -> dict[str, object]:
    data = np.load(output_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    errors = []
    if metadata.get("version") != DETECTION_CALIBRATION_VERSION:
        errors.append("calibration version mismatch")
    if metadata.get("profile_fitter_version") != PROFILE_FITTER_VERSION:
        errors.append("profile-fitter version mismatch")
    if metadata.get("noise_model_sha256") != file_sha256(model_path):
        errors.append("noise-model hash mismatch")
    if metadata.get("controls_sha256") != file_sha256(controls_path):
        errors.append("controls hash mismatch")
    if data["reference_statistics"].shape != (23, 8192):
        errors.append(
            f"unexpected reference shape {data['reference_statistics'].shape}"
        )
    temperatures = np.asarray(data["temperatures"], dtype=int)
    thresholds = np.asarray(data["thresholds"], dtype=float)
    target = float(metadata.get("target_per_fit_fpr", np.nan))
    for temperature, threshold in zip(temperatures, thresholds):
        pvalue = float(
            empirical_p_value(threshold, int(temperature), output_path)
        )
        if pvalue > target:
            errors.append(
                f"{temperature} K threshold empirical p {pvalue} exceeds {target}"
            )
    if metadata.get("acceptance", {}).get("status") != "PASS":
        errors.append(
            "acceptance failed: "
            + "; ".join(metadata.get("acceptance", {}).get("failures", []))
        )
    if errors:
        raise ValueError(
            "Detection calibration validation failed:\n- " + "\n- ".join(errors)
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        metadata = validate_calibration(args.output, args.model, args.controls)
    else:
        metadata = build_calibration(
            args.model,
            args.controls,
            args.output,
            args.report,
        )
        if metadata["acceptance"]["status"] != "PASS":
            raise RuntimeError(
                "Step 6 acceptance failed: "
                + "; ".join(metadata["acceptance"]["failures"])
            )
    print(
        f"PASS: independent ordinary-null FPR "
        f"{metadata['ordinary']['aggregate_evaluation_fpr']:.4%}; "
        f"horizontal {metadata['horizontal_trigger']['fpr']:.4%}; "
        f"near-defect {metadata['near_defect']['fpr']:.4%}"
    )


if __name__ == "__main__":
    main()

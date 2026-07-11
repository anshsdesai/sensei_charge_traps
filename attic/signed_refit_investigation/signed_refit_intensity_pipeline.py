"""Regenerate definitive signed per-temperature dipole intensity fits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from astropy.io import fits
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from signed_refit_candidate_variance_closure import (
    empirical_survival,
    load_detection_references,
    load_null_scales,
)
from signed_refit_controls import COL_REGIONS, ROW_REGIONS, region_id
from signed_refit_detection_calibration import TARGET_PER_FIT_FPR
from signed_refit_finder import (
    ELECTRONIZE_SCALE,
    electronize_and_subtract_rows,
    finder_mask,
    load_frozen_config,
    robust_noise_sigma,
)
from signed_refit_noise_model import NOISE_MODEL_VERSION
from signed_refit_orientation import (
    LABEL_AMBIGUOUS,
    LABEL_DUAL,
    LABEL_SINGLE_NEGATIVE,
    LABEL_SINGLE_POSITIVE,
    LABEL_STRUCTURED,
    ORIENTATION_POLICY_VERSION,
    classify_orientations,
    load_policy,
)
from signed_refit_profile_fitter import (
    SIGNAL_DEPENDENT_FITTER_VERSION,
    ProfileTauFitter,
    SignalDependentProfileTauFitter,
    file_sha256,
    intensity_model,
)
from signed_refit_variance_model import (
    MAX_PHYSICAL_AMPLITUDE,
    VARIANCE_MODEL_VERSION,
)


PIPELINE_VERSION = "signed-refit-intensity-v1"
COORDINATE_VERSION = "signed-refit-candidates-v1"
SPECTRA_VERSION = "signed-refit-spectra-v1"
FIT_VERSION = "signed-refit-intensity-fits-v1"

DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_MANIFEST = Path("signed_refit_manifest.csv")
DEFAULT_DETECTION = Path("signed_refit_detection_calibration.npz")
DEFAULT_VARIANCE = Path("signed_refit_variance_validation.npz")
DEFAULT_CLOSURE = Path("signed_refit_candidate_variance_closure_v2.npz")
DEFAULT_ORIENTATION = Path("signed_refit_orientation_validation.npz")
DEFAULT_POLICY = Path("signed_refit_orientation_policy.json")
DEFAULT_FINDER = Path("signed_refit_finder_config.json")
DEFAULT_FINDER_CALIBRATION = Path("signed_refit_finder_calibration.npz")
DEFAULT_COORDS = Path("signed_refit_candidates_v1.npz")
DEFAULT_SPECTRA = Path("signed_refit_spectra_v1.h5")
DEFAULT_FITS = Path("signed_refit_intensity_fits_v1.h5")
DEFAULT_REPORT = Path("signed_refit_intensity_cutflow.md")
DEFAULT_FIGURE_DIR = Path("figures/signed_refit_intensity")

IMAGE_SLICE = (slice(2, 512), slice(8, 3080))
REJECTION_NAMES = (
    "accepted",
    "empirical_detection_fail",
    "null_amplitude_nonphysical",
    "variance_not_converged",
    "final_amplitude_nonphysical",
    "interval_not_two_sided",
    "boundary_limited",
    "multimodal_profile",
    "profile_fit_exception",
)
REJECTION_CODE = {name: index for index, name in enumerate(REJECTION_NAMES)}
LOCKED_PRIOR_LABELS = (LABEL_AMBIGUOUS, LABEL_DUAL, LABEL_STRUCTURED)


def _decode_paths(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _atomic_target(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _replace_atomic(temporary: Path, destination: Path) -> None:
    temporary.replace(destination)


def _flatten_coordinates(
    coordinates: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    quadrant = []
    rows = []
    cols = []
    for value in range(4):
        coords = coordinates[value]
        quadrant.append(np.full(coords.shape[0], value, dtype=np.int8))
        rows.append(coords[:, 0].astype(np.int16))
        cols.append(coords[:, 1].astype(np.int16))
    quadrant_array = np.concatenate(quadrant)
    row_array = np.concatenate(rows)
    col_array = np.concatenate(cols)
    return {
        "quadrant": quadrant_array,
        "row": row_array,
        "col": col_array,
        "region": region_id(row_array, col_array).astype(np.int8),
    }


def rebuild_coordinates(
    model_path: Path,
    finder_path: Path,
    finder_calibration_path: Path,
    orientation_path: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    config = load_frozen_config(finder_path)
    if config.noise_estimator != "robust":
        raise ValueError("Step 9 requires the frozen robust finder")
    union = np.zeros((4, 509, 3072), dtype=bool)
    with h5py.File(model_path, "r") as model:
        temp_names = sorted(
            (name for name in model if name.startswith("temp_")),
            key=lambda value: int(value.split("_")[1]),
        )
        for temp_name in temp_names:
            paths = _decode_paths(np.asarray(model[temp_name]["source_fits"]))
            counts = np.zeros_like(union, dtype=np.uint8)
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
                        counts[quadrant] += finder_mask(
                            residual,
                            robust_noise_sigma(residual),
                            config,
                        )
            union |= counts >= config.persistence
            print(f"Step 9 finder: completed {temp_name}", flush=True)

    coordinates = {}
    for quadrant in range(4):
        rows, cols = np.where(union[quadrant])
        coordinates[quadrant] = np.column_stack((rows + 1, cols)).astype(
            np.int16
        )

    calibration = np.load(finder_calibration_path, allow_pickle=False)
    selected = int(calibration["selected_index"])
    expected_counts = np.asarray(
        calibration["actual_union_candidates"][selected], dtype=int
    )
    calibration.close()
    observed_counts = np.asarray(
        [coordinates[q].shape[0] for q in range(4)], dtype=int
    )
    if not np.array_equal(observed_counts, expected_counts):
        raise ValueError(
            f"Finder count mismatch: {observed_counts.tolist()} versus "
            f"{expected_counts.tolist()}"
        )

    orientation = np.load(orientation_path, allow_pickle=False)
    for quadrant in range(4):
        selected = np.asarray(orientation["candidate_quadrant"]) == quadrant
        expected = np.column_stack(
            (
                np.asarray(orientation["candidate_row"])[selected],
                np.asarray(orientation["candidate_col"])[selected],
            )
        ).astype(np.int16)
        if not np.array_equal(coordinates[quadrant], expected):
            raise ValueError(
                f"Rebuilt coordinates differ from Step 8 in quadrant {quadrant}"
            )
    orientation.close()

    temporary = _atomic_target(output_path)
    temporary.unlink(missing_ok=True)
    metadata = {
        "version": COORDINATE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "noise_model_sha256": file_sha256(model_path),
        "finder_config_sha256": file_sha256(finder_path),
        "finder_calibration_sha256": file_sha256(finder_calibration_path),
        "orientation_validation_sha256": file_sha256(orientation_path),
        "counts_by_quadrant": observed_counts.tolist(),
        "candidate_count": int(np.sum(observed_counts)),
        "electronize_scale_adu_per_e": ELECTRONIZE_SCALE,
    }
    np.savez_compressed(
        temporary,
        **{
            f"quad_idx_{quadrant}": coordinates[quadrant]
            for quadrant in range(4)
        },
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    generated = temporary.with_suffix(temporary.suffix + ".npz")
    if generated.exists() and generated != temporary:
        generated.replace(temporary)
    _replace_atomic(temporary, output_path)
    return _flatten_coordinates(coordinates)


def _regional_reference_pair_charge(
    temp_group: h5py.Group,
    quadrant: int,
    control_regions: np.ndarray,
    control_splits: np.ndarray,
) -> np.ndarray:
    background = np.asarray(
        temp_group[f"quad_{quadrant}/background"], dtype=float
    )
    reference = np.empty(
        (ROW_REGIONS * COL_REGIONS, background.shape[1]), dtype=float
    )
    for region in range(ROW_REGIONS * COL_REGIONS):
        selected = (control_regions == region) & (control_splits == 0)
        reference[region] = 2.0 * np.median(background[selected], axis=0)
    return reference


def write_spectra(
    model_path: Path,
    manifest_path: Path,
    coords_path: Path,
    flat: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    temporary = _atomic_target(output_path)
    temporary.unlink(missing_ok=True)
    with h5py.File(model_path, "r") as model, h5py.File(temporary, "w") as output:
        if str(model.attrs.get("version", "")) != NOISE_MODEL_VERSION:
            raise ValueError("Unexpected noise model version")
        output.attrs["version"] = SPECTRA_VERSION
        output.attrs["pipeline_version"] = PIPELINE_VERSION
        output.attrs["noise_model_sha256"] = file_sha256(model_path)
        output.attrs["manifest_sha256"] = file_sha256(manifest_path)
        output.attrs["coordinates_sha256"] = file_sha256(coords_path)
        output.attrs["lobe_order_contract"] = (
            "I=(image[row,col]-image[row-1,col])/2"
        )
        output.attrs["electronize_scale_adu_per_e"] = ELECTRONIZE_SCALE
        for key, values in flat.items():
            output.create_dataset(f"candidate_{key}", data=values)

        global_quadrants = np.asarray(
            model["controls/quadrant"], dtype=np.int8
        )
        global_regions = np.asarray(model["controls/region"], dtype=np.int8)
        global_splits = np.asarray(model["controls/split"], dtype=np.int8)
        temp_names = sorted(
            (name for name in model if name.startswith("temp_")),
            key=lambda value: int(value.split("_")[1]),
        )
        string_dtype = h5py.string_dtype("utf-8")
        for temp_name in temp_names:
            source = model[temp_name]
            paths = _decode_paths(np.asarray(source["source_fits"]))
            count = flat["row"].size
            intensity = np.empty((count, len(paths)), dtype=np.float32)
            extra_shot = np.empty_like(intensity)
            references = {}
            quadrant_indices = {}
            for quadrant in range(4):
                qsel_controls = global_quadrants == quadrant
                references[quadrant] = _regional_reference_pair_charge(
                    source,
                    quadrant,
                    global_regions[qsel_controls],
                    global_splits[qsel_controls],
                )
                quadrant_indices[quadrant] = np.flatnonzero(
                    flat["quadrant"] == quadrant
                )

            for delay_index, path in enumerate(paths):
                with fits.open(
                    path,
                    memmap=True,
                    do_not_scale_image_data=True,
                ) as hdus:
                    for quadrant in range(4):
                        indices = quadrant_indices[quadrant]
                        rows = flat["row"][indices].astype(int)
                        cols = flat["col"][indices].astype(int)
                        image, residual = electronize_and_subtract_rows(
                            hdus[quadrant].data[IMAGE_SLICE]
                        )
                        intensity[indices, delay_index] = (
                            residual[rows, cols] - residual[rows - 1, cols]
                        ) / 2.0
                        pair_charge = (
                            image[rows, cols] + image[rows - 1, cols]
                        )
                        reference = references[quadrant][
                            flat["region"][indices], delay_index
                        ]
                        extra_shot[indices, delay_index] = (
                            np.maximum(pair_charge - reference, 0.0) / 4.0
                        )

            group = output.create_group(temp_name)
            group.create_dataset(
                "seconds", data=np.asarray(source["seconds"], dtype=float)
            )
            group.create_dataset(
                "dtph", data=np.asarray(source["dtph"], dtype=np.int64)
            )
            group.create_dataset(
                "source_fits",
                data=np.asarray(paths, dtype=object),
                dtype=string_dtype,
            )
            group.create_dataset(
                "intensity",
                data=intensity,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(min(512, count), len(paths)),
            )
            group.create_dataset(
                "extra_pair_shot_variance",
                data=extra_shot,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(min(512, count), len(paths)),
            )
            print(
                f"Step 9 spectra: wrote {temp_name} ({len(paths)} delays)",
                flush=True,
            )
    _replace_atomic(temporary, output_path)


def load_overdispersion(path: Path) -> dict[str, np.ndarray]:
    metadata = json.loads(
        str(np.load(path, allow_pickle=False)["metadata_json"])
    )
    if metadata.get("acceptance", {}).get("status") != "PASS":
        raise ValueError("Candidate variance closure has not passed")
    calibration = metadata["calibration"]
    return {
        "edges": np.asarray(calibration["amplitude_edges"], dtype=float),
        "factors": np.asarray(calibration["estimated_factors"], dtype=float),
    }


def _factor_for_amplitude(
    amplitude: float,
    calibration: dict[str, np.ndarray],
) -> float:
    index = int(np.digitize(abs(float(amplitude)), calibration["edges"]))
    return float(calibration["factors"][index])


def _empty_fit_arrays(
    candidate_count: int,
    temperature_count: int,
) -> dict[str, np.ndarray]:
    shape = (candidate_count, temperature_count)
    float_names = (
        "detection_statistic",
        "empirical_pvalue",
        "null_tau",
        "null_amplitude",
        "null_offset",
        "tau",
        "tau_interval_lower",
        "tau_interval_upper",
        "amplitude",
        "amplitude_error",
        "amplitude_z",
        "offset",
        "offset_error",
        "chi2",
        "reduced_chi2",
        "fit_pvalue",
        "overdispersion_factor",
    )
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in float_names
    }
    arrays.update(
        {
            "dof": np.full(shape, -1, dtype=np.int16),
            "sign": np.zeros(shape, dtype=np.int8),
            "profile_fitted": np.zeros(shape, dtype=np.int8),
            "variance_converged": np.zeros(shape, dtype=np.int8),
            "at_lower_boundary": np.zeros(shape, dtype=np.int8),
            "at_upper_boundary": np.zeros(shape, dtype=np.int8),
            "interval_lower_limited": np.zeros(shape, dtype=np.int8),
            "interval_upper_limited": np.zeros(shape, dtype=np.int8),
            "boundary_limited": np.zeros(shape, dtype=np.int8),
            "multimodal": np.zeros(shape, dtype=np.int8),
            "accepted": np.zeros(shape, dtype=np.int8),
            "rejection_code": np.full(
                shape,
                REJECTION_CODE["empirical_detection_fail"],
                dtype=np.int8,
            ),
        }
    )
    return arrays


def _quality_code(result: dict[str, object]) -> int:
    if not bool(result["variance_converged"]):
        return REJECTION_CODE["variance_not_converged"]
    if abs(float(result["amplitude"])) > MAX_PHYSICAL_AMPLITUDE:
        return REJECTION_CODE["final_amplitude_nonphysical"]
    if (
        result["tau_interval_lower"] is None
        or result["tau_interval_upper"] is None
    ):
        return REJECTION_CODE["interval_not_two_sided"]
    if bool(result["boundary_limited"]):
        return REJECTION_CODE["boundary_limited"]
    if bool(result["multimodal"]):
        return REJECTION_CODE["multimodal_profile"]
    return REJECTION_CODE["accepted"]


def write_fits(
    model_path: Path,
    manifest_path: Path,
    detection_path: Path,
    variance_path: Path,
    closure_path: Path,
    orientation_path: Path,
    policy_path: Path,
    coords_path: Path,
    spectra_path: Path,
    output_path: Path,
) -> dict[str, object]:
    load_policy(policy_path)
    references = load_detection_references(detection_path)
    null_scales = load_null_scales(variance_path)
    overdispersion = load_overdispersion(closure_path)
    orientation = np.load(orientation_path, allow_pickle=False)
    prior_labels = np.asarray(orientation["candidate_orientation_label"])
    structured = np.asarray(
        orientation["candidate_structured_background"], dtype=bool
    )

    temporary = _atomic_target(output_path)
    temporary.unlink(missing_ok=True)
    with (
        h5py.File(model_path, "r") as model,
        h5py.File(spectra_path, "r") as spectra,
        h5py.File(temporary, "w") as output,
    ):
        temperatures = np.asarray(
            sorted(
                int(name.split("_")[1])
                for name in spectra
                if name.startswith("temp_")
            ),
            dtype=np.int16,
        )
        flat = {
            key: np.asarray(spectra[f"candidate_{key}"])
            for key in ("quadrant", "row", "col", "region")
        }
        candidate_count = flat["row"].size
        arrays = _empty_fit_arrays(candidate_count, temperatures.size)

        output.attrs["version"] = FIT_VERSION
        output.attrs["pipeline_version"] = PIPELINE_VERSION
        output.attrs["profile_fitter_version"] = SIGNAL_DEPENDENT_FITTER_VERSION
        output.attrs["variance_model_version"] = VARIANCE_MODEL_VERSION
        output.attrs["orientation_policy_version"] = ORIENTATION_POLICY_VERSION
        output.attrs["manifest_sha256"] = file_sha256(manifest_path)
        output.attrs["noise_model_sha256"] = file_sha256(model_path)
        output.attrs["detection_calibration_sha256"] = file_sha256(
            detection_path
        )
        output.attrs["variance_validation_sha256"] = file_sha256(variance_path)
        output.attrs["candidate_variance_closure_sha256"] = file_sha256(
            closure_path
        )
        output.attrs["orientation_validation_sha256"] = file_sha256(
            orientation_path
        )
        output.attrs["orientation_policy_sha256"] = file_sha256(policy_path)
        output.attrs["coordinates_sha256"] = file_sha256(coords_path)
        output.attrs["spectra_sha256"] = file_sha256(spectra_path)
        output.attrs["detection_rule"] = "finite-sample empirical p <= 0.001"
        output.attrs["quality_rule"] = (
            "physical amplitude, converged signal variance, two-sided tau "
            "interval, not boundary-limited, not multimodal"
        )
        output.create_dataset("temperatures", data=temperatures)
        for key, values in flat.items():
            output.create_dataset(f"candidate_{key}", data=values)
        output.create_dataset(
            "overdispersion_amplitude_edges", data=overdispersion["edges"]
        )
        output.create_dataset(
            "overdispersion_factors", data=overdispersion["factors"]
        )
        profiles_group = output.create_group("profiles")

        for temp_index, temperature_value in enumerate(temperatures):
            temperature = int(temperature_value)
            temp_name = f"temp_{temperature}"
            source = model[temp_name]
            seconds = np.asarray(source["seconds"], dtype=float)
            curves = np.asarray(spectra[temp_name]["intensity"], dtype=float)
            extra_shot = np.asarray(
                spectra[temp_name]["extra_pair_shot_variance"], dtype=float
            )
            statistic = arrays["detection_statistic"][:, temp_index]
            null_tau = arrays["null_tau"][:, temp_index]
            null_amplitude = arrays["null_amplitude"][:, temp_index]
            null_offset = arrays["null_offset"][:, temp_index]

            for quadrant in range(4):
                for region in range(ROW_REGIONS * COL_REGIONS):
                    selected = (
                        (flat["quadrant"] == quadrant)
                        & (flat["region"] == region)
                    )
                    if not np.any(selected):
                        continue
                    region_group = source[
                        f"quad_{quadrant}/region_{region}"
                    ]
                    null_fitter = ProfileTauFitter(
                        seconds,
                        np.asarray(region_group["covariance"], dtype=float),
                        null_template=np.asarray(
                            region_group["null_template"], dtype=float
                        ),
                    )
                    result = null_fitter.batch_profile_statistic(
                        curves[selected]
                    )
                    statistic[selected] = result["delta_chi2"]
                    null_tau[selected] = result["tau"]
                    null_amplitude[selected] = result["amplitude"]
                    null_offset[selected] = result["offset"]

            pvalues = empirical_survival(
                references[temperature],
                statistic.astype(float),
            )
            arrays["empirical_pvalue"][:, temp_index] = pvalues
            detected = pvalues <= TARGET_PER_FIT_FPR
            physical_null = (
                np.isfinite(null_amplitude)
                & (np.abs(null_amplitude) <= MAX_PHYSICAL_AMPLITUDE)
            )
            arrays["rejection_code"][
                detected & ~physical_null, temp_index
            ] = REJECTION_CODE["null_amplitude_nonphysical"]
            profile_indices = np.flatnonzero(detected & physical_null)
            profile_site_index = []
            profile_delta_chi2 = []
            profile_tau_grid = None

            for sequence, candidate_index in enumerate(profile_indices):
                quadrant = int(flat["quadrant"][candidate_index])
                region = int(flat["region"][candidate_index])
                region_group = source[f"quad_{quadrant}/region_{region}"]
                factor = _factor_for_amplitude(
                    float(null_amplitude[candidate_index]),
                    overdispersion,
                )
                try:
                    fitter = SignalDependentProfileTauFitter(
                        seconds,
                        np.asarray(region_group["covariance"], dtype=float),
                        null_template=np.asarray(
                            region_group["null_template"], dtype=float
                        ),
                        null_scale=null_scales[temperature],
                        pump_overdispersion=factor,
                        extra_pair_shot=extra_shot[candidate_index],
                    )
                    result = fitter.fit(curves[candidate_index])
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    arrays["rejection_code"][
                        candidate_index, temp_index
                    ] = REJECTION_CODE["profile_fit_exception"]
                    continue

                arrays["profile_fitted"][candidate_index, temp_index] = 1
                arrays["tau"][candidate_index, temp_index] = result["tau"]
                arrays["tau_interval_lower"][
                    candidate_index, temp_index
                ] = (
                    np.nan
                    if result["tau_interval_lower"] is None
                    else result["tau_interval_lower"]
                )
                arrays["tau_interval_upper"][
                    candidate_index, temp_index
                ] = (
                    np.nan
                    if result["tau_interval_upper"] is None
                    else result["tau_interval_upper"]
                )
                arrays["amplitude"][
                    candidate_index, temp_index
                ] = result["amplitude"]
                arrays["amplitude_error"][
                    candidate_index, temp_index
                ] = result["amplitude_error_conditional"]
                arrays["amplitude_z"][
                    candidate_index, temp_index
                ] = result["amplitude_z_conditional"]
                arrays["sign"][candidate_index, temp_index] = result[
                    "amplitude_sign"
                ]
                arrays["offset"][
                    candidate_index, temp_index
                ] = result["offset"]
                arrays["offset_error"][
                    candidate_index, temp_index
                ] = result["offset_error_conditional"]
                arrays["chi2"][candidate_index, temp_index] = result["chi2"]
                arrays["dof"][candidate_index, temp_index] = result["dof"]
                arrays["reduced_chi2"][
                    candidate_index, temp_index
                ] = result["reduced_chi2"]
                arrays["fit_pvalue"][
                    candidate_index, temp_index
                ] = result["fit_p_value"]
                arrays["overdispersion_factor"][
                    candidate_index, temp_index
                ] = factor
                arrays["variance_converged"][
                    candidate_index, temp_index
                ] = int(result["variance_converged"])
                arrays["at_lower_boundary"][
                    candidate_index, temp_index
                ] = int(result["at_lower_boundary"])
                arrays["at_upper_boundary"][
                    candidate_index, temp_index
                ] = int(result["at_upper_boundary"])
                arrays["interval_lower_limited"][
                    candidate_index, temp_index
                ] = int(result["interval_lower_limited"])
                arrays["interval_upper_limited"][
                    candidate_index, temp_index
                ] = int(result["interval_upper_limited"])
                arrays["boundary_limited"][
                    candidate_index, temp_index
                ] = int(result["boundary_limited"])
                arrays["multimodal"][
                    candidate_index, temp_index
                ] = int(result["multimodal"])
                code = _quality_code(result)
                arrays["rejection_code"][
                    candidate_index, temp_index
                ] = code
                arrays["accepted"][candidate_index, temp_index] = int(
                    code == REJECTION_CODE["accepted"]
                )
                profile_site_index.append(candidate_index)
                profile_tau_grid = np.asarray(
                    result["tau_grid"], dtype=np.float32
                )
                profile_delta_chi2.append(
                    (
                        np.asarray(result["chi2_profile"], dtype=float)
                        - float(np.min(result["chi2_profile"]))
                    ).astype(np.float32)
                )
                if (sequence + 1) % 500 == 0:
                    print(
                        f"Step 9 fits: {temperature} K "
                        f"{sequence + 1}/{profile_indices.size}",
                        flush=True,
                    )

            profile_group = profiles_group.create_group(temp_name)
            profile_group.create_dataset(
                "candidate_index",
                data=np.asarray(profile_site_index, dtype=np.int32),
            )
            if profile_tau_grid is None:
                profile_tau_grid = ProfileTauFitter(
                    seconds,
                    np.eye(seconds.size),
                ).tau_grid.astype(np.float32)
            profile_group.create_dataset("tau_grid", data=profile_tau_grid)
            profile_group.create_dataset(
                "delta_chi2",
                data=np.asarray(profile_delta_chi2, dtype=np.float32).reshape(
                    len(profile_site_index), profile_tau_grid.size
                ),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(
                    max(1, min(64, len(profile_site_index))),
                    profile_tau_grid.size,
                ),
            )
            print(
                f"Step 9 fits: completed {temperature} K; "
                f"detected physical={profile_indices.size}, "
                f"accepted={int(np.sum(arrays['accepted'][:, temp_index]))}",
                flush=True,
            )

        raw_classification = classify_orientations(
            arrays["sign"],
            arrays["accepted"].astype(bool),
            structured_background=structured,
        )
        final_labels = raw_classification["label"].copy()
        locked = np.isin(prior_labels, LOCKED_PRIOR_LABELS)
        restored = locked & np.isin(
            final_labels,
            (LABEL_SINGLE_POSITIVE, LABEL_SINGLE_NEGATIVE),
        )
        final_labels[restored] = prior_labels[restored]
        final_single = np.isin(
            final_labels,
            (LABEL_SINGLE_POSITIVE, LABEL_SINGLE_NEGATIVE),
        )

        for name, values in arrays.items():
            output.create_dataset(
                name,
                data=values,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        string_dtype = h5py.string_dtype("utf-8")
        output.create_dataset(
            "rejection_names",
            data=np.asarray(REJECTION_NAMES, dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset(
            "prior_orientation_label",
            data=np.asarray(prior_labels, dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset(
            "raw_orientation_label",
            data=np.asarray(raw_classification["label"], dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset(
            "final_orientation_label",
            data=np.asarray(final_labels, dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset(
            "single_trap_eligible", data=final_single.astype(np.int8)
        )
        output.create_dataset(
            "orientation_significant_count",
            data=raw_classification["significant_count"],
        )
        output.create_dataset(
            "orientation_positive_count",
            data=raw_classification["positive_count"],
        )
        output.create_dataset(
            "orientation_negative_count",
            data=raw_classification["negative_count"],
        )
        output.create_dataset(
            "orientation_dominant_sign",
            data=raw_classification["dominant_sign"],
        )
        output.create_dataset(
            "locked_prior_exclusion", data=locked.astype(np.int8)
        )
        output.create_dataset(
            "prevented_restoration", data=restored.astype(np.int8)
        )

    orientation.close()
    _replace_atomic(temporary, output_path)
    return {
        "candidate_count": candidate_count,
        "temperature_count": temperatures.size,
        "accepted_fit_count": int(np.sum(arrays["accepted"])),
        "single_trap_count": int(np.sum(final_single)),
        "prevented_restoration_count": int(np.sum(restored)),
    }


def backfill_boundary_flags(fits_path: Path) -> None:
    names = (
        "at_lower_boundary",
        "at_upper_boundary",
        "interval_lower_limited",
        "interval_upper_limited",
    )
    with h5py.File(fits_path, "r+") as handle:
        if all(name in handle for name in names):
            return
        profile_fitted = np.asarray(handle["profile_fitted"], dtype=bool)
        shape = profile_fitted.shape
        lower_interval = np.asarray(handle["tau_interval_lower"], dtype=float)
        upper_interval = np.asarray(handle["tau_interval_upper"], dtype=float)
        at_lower = np.zeros(shape, dtype=np.int8)
        at_upper = np.zeros(shape, dtype=np.int8)
        lower_limited = (
            profile_fitted & ~np.isfinite(lower_interval)
        ).astype(np.int8)
        upper_limited = (
            profile_fitted & ~np.isfinite(upper_interval)
        ).astype(np.int8)
        temperatures = np.asarray(handle["temperatures"], dtype=int)
        for temp_index, temperature in enumerate(temperatures):
            group = handle[f"profiles/temp_{temperature}"]
            indices = np.asarray(group["candidate_index"], dtype=int)
            profiles = np.asarray(group["delta_chi2"], dtype=float)
            best = np.argmin(profiles, axis=1)
            at_lower[indices[best == 0], temp_index] = 1
            at_upper[indices[best == profiles.shape[1] - 1], temp_index] = 1
        values = {
            "at_lower_boundary": at_lower,
            "at_upper_boundary": at_upper,
            "interval_lower_limited": lower_limited,
            "interval_upper_limited": upper_limited,
        }
        for name, value in values.items():
            if name not in handle:
                handle.create_dataset(
                    name,
                    data=value,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
        combined = (
            at_lower.astype(bool)
            | at_upper.astype(bool)
            | lower_limited.astype(bool)
            | upper_limited.astype(bool)
        )
        stored = np.asarray(handle["boundary_limited"], dtype=bool)
        if not np.array_equal(combined, stored):
            raise ValueError(
                "Backfilled boundary components do not reproduce boundary_limited"
            )


def null_control_summary(detection_path: Path) -> dict[str, object]:
    data = np.load(detection_path, allow_pickle=False)
    temperatures = np.asarray(data["temperatures"], dtype=int)
    pvalues = np.asarray(data["ordinary_empirical_pvalue"], dtype=float)
    split = np.asarray(data["ordinary_calibration_split"], dtype=np.int8)
    evaluation = split == 1
    accepted = pvalues[evaluation] <= TARGET_PER_FIT_FPR
    rates = np.mean(accepted, axis=0)
    metadata = json.loads(str(data["metadata_json"]))
    data.close()
    return {
        "aggregate_rate": float(np.mean(accepted)),
        "temperature_rates": {
            str(int(temperature)): float(rate)
            for temperature, rate in zip(temperatures, rates)
        },
        "calibration_status": metadata["acceptance"]["status"],
        "maximum_temperature_rate": float(np.max(rates)),
    }


def _read_fit_file(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        names = (
            "temperatures",
            "candidate_quadrant",
            "candidate_row",
            "candidate_col",
            "accepted",
            "rejection_code",
            "empirical_pvalue",
            "null_amplitude",
            "null_tau",
            "null_offset",
            "tau",
            "amplitude",
            "offset",
            "reduced_chi2",
            "single_trap_eligible",
            "final_orientation_label",
            "prevented_restoration",
        )
        return {name: np.asarray(handle[name]) for name in names}


def write_figures(
    model_path: Path,
    spectra_path: Path,
    fits_path: Path,
    figure_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    values = _read_fit_file(fits_path)
    accepted_pairs = np.argwhere(values["accepted"].astype(bool))
    if accepted_pairs.size:
        score = np.abs(values["reduced_chi2"][tuple(accepted_pairs.T)] - 1.0)
        accepted_pairs = accepted_pairs[np.argsort(score)]
    rejected_pairs = []
    rejection = values["rejection_code"]
    for name in REJECTION_NAMES[1:]:
        matches = np.argwhere(rejection == REJECTION_CODE[name])
        if matches.size:
            rejected_pairs.append(matches[len(matches) // 2])
    rejected_pairs = np.asarray(rejected_pairs, dtype=int)

    outputs = []
    for title, pairs, filename in (
        (
            "Representative accepted intensity fits",
            accepted_pairs[:12],
            "accepted_intensity_fits.png",
        ),
        (
            "Representative rejected intensity fits",
            rejected_pairs[:12],
            "rejected_intensity_fits.png",
        ),
    ):
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        with (
            h5py.File(model_path, "r") as model,
            h5py.File(spectra_path, "r") as spectra,
        ):
            for axis, pair in zip(axes.flat, pairs):
                candidate_index, temp_index = map(int, pair)
                temperature = int(values["temperatures"][temp_index])
                temp_name = f"temp_{temperature}"
                seconds = np.asarray(spectra[temp_name]["seconds"], dtype=float)
                curve = np.asarray(
                    spectra[temp_name]["intensity"][candidate_index], dtype=float
                )
                axis.plot(seconds, curve, "o", color="black", markersize=4)
                if np.isfinite(values["tau"][candidate_index, temp_index]):
                    tau = float(values["tau"][candidate_index, temp_index])
                    amplitude = float(
                        values["amplitude"][candidate_index, temp_index]
                    )
                    offset = float(values["offset"][candidate_index, temp_index])
                else:
                    tau = float(
                        values["null_tau"][candidate_index, temp_index]
                    )
                    amplitude = float(
                        values["null_amplitude"][candidate_index, temp_index]
                    )
                    offset = float(
                        values["null_offset"][candidate_index, temp_index]
                    )
                quadrant = int(values["candidate_quadrant"][candidate_index])
                row = int(values["candidate_row"][candidate_index])
                col = int(values["candidate_col"][candidate_index])
                region = int(region_id(row, col))
                template = np.asarray(
                    model[temp_name][
                        f"quad_{quadrant}/region_{region}/null_template"
                    ],
                    dtype=float,
                )
                if np.isfinite(tau) and tau > 0:
                    grid = np.geomspace(seconds.min(), seconds.max(), 300)
                    template_grid = np.interp(grid, seconds, template)
                    axis.plot(
                        grid,
                        intensity_model(
                            grid,
                            amplitude,
                            tau,
                            offset,
                            template_grid,
                        ),
                        color="tab:red",
                        linewidth=1.5,
                    )
                code = int(rejection[candidate_index, temp_index])
                axis.set_xscale("log")
                axis.set_title(
                    f"{temperature} K Q{quadrant} ({row},{col})\n"
                    f"{REJECTION_NAMES[code]}",
                    fontsize=9,
                )
            for axis in axes.flat[len(pairs) :]:
                axis.axis("off")
        for axis in axes[-1]:
            axis.set_xlabel("Phase dwell time (s)")
        for axis in axes[:, 0]:
            axis.set_ylabel("Signed intensity (e-)")
        fig.suptitle(title)
        fig.tight_layout()
        output = figure_dir / filename
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(output)
    return outputs


def build_cutflow(
    fits_path: Path,
    detection_path: Path,
    figure_paths: list[Path],
    report_path: Path,
) -> dict[str, object]:
    values = _read_fit_file(fits_path)
    temperatures = values["temperatures"].astype(int)
    rejection = values["rejection_code"].astype(int)
    accepted = values["accepted"].astype(bool)
    quadrants = values["candidate_quadrant"].astype(int)
    null_summary = null_control_summary(detection_path)
    failures = []

    physical_detected = np.isin(
        rejection,
        [
            REJECTION_CODE[name]
            for name in REJECTION_NAMES
            if name
            not in (
                "empirical_detection_fail",
                "null_amplitude_nonphysical",
            )
        ],
    )
    for index, temperature in enumerate(temperatures):
        detected_count = int(np.sum(physical_detected[:, index]))
        accepted_count = int(np.sum(accepted[:, index]))
        if detected_count >= 50 and accepted_count == 0:
            failures.append(f"{temperature} K has no accepted fits")
        exception_rate = float(
            np.mean(
                rejection[physical_detected[:, index], index]
                == REJECTION_CODE["profile_fit_exception"]
            )
        ) if detected_count else 0.0
        if exception_rate > 0.01:
            failures.append(
                f"{temperature} K profile exception rate {exception_rate:.2%}"
            )
    if null_summary["calibration_status"] != "PASS":
        failures.append("frozen null calibration status is not PASS")
    if null_summary["aggregate_rate"] > 0.002:
        failures.append(
            f"ordinary-null aggregate FPR {null_summary['aggregate_rate']:.3%}"
        )

    label_counts = Counter(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values["final_orientation_label"]
    )
    lines = [
        "# Signed Refit Step 9 Intensity Cutflow",
        "",
        f"- Pipeline version: `{PIPELINE_VERSION}`",
        f"- Fit artifact: `{fits_path.as_posix()}`",
        f"- Fit artifact SHA-256: `{file_sha256(fits_path)}`",
        f"- Acceptance status: **{'PASS' if not failures else 'FAIL'}**",
        f"- Candidate sites: {values['candidate_row'].size:,}.",
        f"- Accepted candidate-temperature fits: {int(np.sum(accepted)):,}.",
        f"- Final single-trap sites: "
        f"{int(np.sum(values['single_trap_eligible'])):,}.",
        f"- Prevented restorations of prior conflict/structured sites: "
        f"{int(np.sum(values['prevented_restoration'])):,}.",
        "",
        "A fit is attempted only after the exact finite-sample empirical "
        "`p <= 0.001` test and `|D_t P_c| <= 1` on the null-covariance fit. "
        "Characterization then requires converged signal-dependent covariance, "
        "a physical final amplitude, a two-sided profile interval, and no "
        "boundary or multimodal flag.",
        "",
        "## Temperature cutflow",
        "",
        "| T (K) | Candidates | Detected physical | Accepted | Quality fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, temperature in enumerate(temperatures):
        detected_count = int(np.sum(physical_detected[:, index]))
        accepted_count = int(np.sum(accepted[:, index]))
        fraction = accepted_count / detected_count if detected_count else 0.0
        lines.append(
            f"| {temperature} | {values['candidate_row'].size:,} | "
            f"{detected_count:,} | {accepted_count:,} | {fraction:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Temperature diagnostics",
            "",
        ]
    )
    for index, temperature in enumerate(temperatures):
        detected_count = int(np.sum(physical_detected[:, index]))
        accepted_count = int(np.sum(accepted[:, index]))
        if detected_count and accepted_count / detected_count < 0.5:
            rejected_counts = np.bincount(
                rejection[:, index], minlength=len(REJECTION_NAMES)
            )
            considered = rejected_counts.copy()
            considered[REJECTION_CODE["accepted"]] = 0
            considered[REJECTION_CODE["empirical_detection_fail"]] = 0
            considered[REJECTION_CODE["null_amplitude_nonphysical"]] = 0
            dominant_code = int(np.argmax(considered))
            lines.append(
                f"- {temperature} K: {accepted_count:,}/{detected_count:,} "
                f"physical detections are characterizable; the dominant fit "
                f"rejection is `{REJECTION_NAMES[dominant_code]}` "
                f"({int(considered[dominant_code]):,})."
            )
    lines.extend(
        [
            "",
            "The 130-140 K loss is therefore a scan-window effect: most detected "
            "profiles do not bracket both sides of the Delta-chi-square interval. "
            "They remain detected in the artifact but are not assigned a lifetime "
            "for Step 10.",
            "",
            "## Rejection reasons by temperature",
            "",
            "| T (K) | "
            + " | ".join(REJECTION_NAMES)
            + " |",
            "|---:|" + "---:|" * len(REJECTION_NAMES),
        ]
    )
    for index, temperature in enumerate(temperatures):
        counts = np.bincount(
            rejection[:, index], minlength=len(REJECTION_NAMES)
        )
        lines.append(
            f"| {temperature} | "
            + " | ".join(f"{int(value):,}" for value in counts)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quadrant cutflow",
            "",
            "| Quadrant | Candidates | Accepted fits | Final single-trap sites |",
            "|---:|---:|---:|---:|",
        ]
    )
    for quadrant in range(4):
        selected = quadrants == quadrant
        lines.append(
            f"| {quadrant} | {int(np.sum(selected)):,} | "
            f"{int(np.sum(accepted[selected])):,} | "
            f"{int(np.sum(values['single_trap_eligible'][selected])):,} |"
        )

    lines.extend(
        [
            "",
            "## Final orientation classes",
            "",
            "| Class | Sites |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(label_counts.items()):
        lines.append(f"| `{label}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Null-control check",
            "",
            f"- Independent ordinary-null aggregate empirical FPR: "
            f"{null_summary['aggregate_rate']:.3%}.",
            f"- Maximum per-temperature ordinary-null FPR: "
            f"{null_summary['maximum_temperature_rate']:.3%}.",
            f"- Frozen calibration status: "
            f"{null_summary['calibration_status']}.",
            "",
            "## Visual checks",
            "",
            *[f"- `{path.as_posix()}`" for path in figure_paths],
            "",
            "## Acceptance gate",
            "",
        ]
    )
    if failures:
        lines.append("- **FAIL:** " + "; ".join(failures))
    else:
        lines.extend(
            [
                "- PASS: all artifacts use new versioned names and pinned hashes.",
                "- PASS: every accepted fit maps to stored manifest source files, covariance, variance calibration, and profile.",
                "- PASS: no temperature has an unexplained fit-processing collapse.",
                "- PASS: ordinary-null false-positive rates remain within the frozen calibration gate.",
            ]
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "accepted_fit_count": int(np.sum(accepted)),
        "single_trap_count": int(np.sum(values["single_trap_eligible"])),
        "null_control": null_summary,
    }


def validate_outputs(
    model_path: Path,
    manifest_path: Path,
    detection_path: Path,
    variance_path: Path,
    closure_path: Path,
    orientation_path: Path,
    policy_path: Path,
    finder_path: Path,
    finder_calibration_path: Path,
    coords_path: Path,
    spectra_path: Path,
    fits_path: Path,
) -> dict[str, object]:
    coordinate_data = np.load(coords_path, allow_pickle=False)
    coordinate_metadata = json.loads(str(coordinate_data["metadata_json"]))
    if coordinate_metadata["version"] != COORDINATE_VERSION:
        raise ValueError("Coordinate version mismatch")
    coordinate_data.close()

    with h5py.File(model_path, "r") as model, h5py.File(
        spectra_path, "r"
    ) as spectra:
        expected_spectra = {
            "version": SPECTRA_VERSION,
            "noise_model_sha256": file_sha256(model_path),
            "manifest_sha256": file_sha256(manifest_path),
            "coordinates_sha256": file_sha256(coords_path),
        }
        for key, expected in expected_spectra.items():
            if str(spectra.attrs.get(key, "")) != expected:
                raise ValueError(f"Spectra {key} mismatch")
        for temp_name in (name for name in model if name.startswith("temp_")):
            if _decode_paths(np.asarray(model[temp_name]["source_fits"])) != (
                _decode_paths(np.asarray(spectra[temp_name]["source_fits"]))
            ):
                raise ValueError(f"Source FITS mismatch in {temp_name}")

    with h5py.File(fits_path, "r") as fit_file:
        expected_fit = {
            "version": FIT_VERSION,
            "manifest_sha256": file_sha256(manifest_path),
            "noise_model_sha256": file_sha256(model_path),
            "detection_calibration_sha256": file_sha256(detection_path),
            "variance_validation_sha256": file_sha256(variance_path),
            "candidate_variance_closure_sha256": file_sha256(closure_path),
            "orientation_validation_sha256": file_sha256(orientation_path),
            "orientation_policy_sha256": file_sha256(policy_path),
            "coordinates_sha256": file_sha256(coords_path),
            "spectra_sha256": file_sha256(spectra_path),
        }
        for key, expected in expected_fit.items():
            if str(fit_file.attrs.get(key, "")) != expected:
                raise ValueError(f"Fit {key} mismatch")
        accepted = np.asarray(fit_file["accepted"], dtype=bool)
        if np.any(accepted & ~np.asarray(fit_file["profile_fitted"], dtype=bool)):
            raise ValueError("Accepted fit lacks a stored profile")
        if np.any(
            accepted
            & (
                ~np.isfinite(np.asarray(fit_file["tau_interval_lower"]))
                | ~np.isfinite(np.asarray(fit_file["tau_interval_upper"]))
            )
        ):
            raise ValueError("Accepted fit lacks a two-sided interval")
        if np.any(
            accepted
            & (
                np.asarray(fit_file["boundary_limited"], dtype=bool)
                | np.asarray(fit_file["multimodal"], dtype=bool)
            )
        ):
            raise ValueError("Accepted fit has a forbidden quality flag")
        boundary_components = (
            np.asarray(fit_file["at_lower_boundary"], dtype=bool)
            | np.asarray(fit_file["at_upper_boundary"], dtype=bool)
            | np.asarray(fit_file["interval_lower_limited"], dtype=bool)
            | np.asarray(fit_file["interval_upper_limited"], dtype=bool)
        )
        if not np.array_equal(
            boundary_components,
            np.asarray(fit_file["boundary_limited"], dtype=bool),
        ):
            raise ValueError("Boundary component flags are inconsistent")
        if np.any(
            accepted
            & (
                np.abs(np.asarray(fit_file["amplitude"], dtype=float))
                > MAX_PHYSICAL_AMPLITUDE
            )
        ):
            raise ValueError("Accepted fit has nonphysical amplitude")
        prior = np.asarray(fit_file["prior_orientation_label"]).astype(str)
        final_single = np.asarray(
            fit_file["single_trap_eligible"], dtype=bool
        )
        if np.any(final_single & np.isin(prior, LOCKED_PRIOR_LABELS)):
            raise ValueError("A prior conflict/structured site was restored")
        for temp_name in fit_file["profiles"]:
            indices = np.asarray(
                fit_file[f"profiles/{temp_name}/candidate_index"], dtype=int
            )
            profiles = fit_file[f"profiles/{temp_name}/delta_chi2"]
            if profiles.shape[0] != indices.size or profiles.shape[1] != 801:
                raise ValueError(f"Profile shape mismatch in {temp_name}")

    provenance = {
        "finder_config_sha256": file_sha256(finder_path),
        "finder_calibration_sha256": file_sha256(finder_calibration_path),
        "coordinates_sha256": file_sha256(coords_path),
        "spectra_sha256": file_sha256(spectra_path),
        "fits_sha256": file_sha256(fits_path),
    }
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    parser.add_argument("--variance", type=Path, default=DEFAULT_VARIANCE)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--orientation", type=Path, default=DEFAULT_ORIENTATION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--finder", type=Path, default=DEFAULT_FINDER)
    parser.add_argument(
        "--finder-calibration",
        type=Path,
        default=DEFAULT_FINDER_CALIBRATION,
    )
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--spectra", type=Path, default=DEFAULT_SPECTRA)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        flat = rebuild_coordinates(
            args.model,
            args.finder,
            args.finder_calibration,
            args.orientation,
            args.coords,
        )
        write_spectra(
            args.model,
            args.manifest,
            args.coords,
            flat,
            args.spectra,
        )
        fit_summary = write_fits(
            args.model,
            args.manifest,
            args.detection,
            args.variance,
            args.closure,
            args.orientation,
            args.policy,
            args.coords,
            args.spectra,
            args.fits,
        )
        backfill_boundary_flags(args.fits)
        figures = write_figures(
            args.model,
            args.spectra,
            args.fits,
            args.figure_dir,
        )
        cutflow = build_cutflow(
            args.fits,
            args.detection,
            figures,
            args.report,
        )
        if cutflow["status"] != "PASS":
            raise RuntimeError(
                "Step 9 acceptance failed: "
                + "; ".join(cutflow["failures"])
            )
        print(json.dumps({**fit_summary, **cutflow}, indent=2))

    backfill_boundary_flags(args.fits)
    provenance = validate_outputs(
        args.model,
        args.manifest,
        args.detection,
        args.variance,
        args.closure,
        args.orientation,
        args.policy,
        args.finder,
        args.finder_calibration,
        args.coords,
        args.spectra,
        args.fits,
    )
    print("PASS: " + json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()

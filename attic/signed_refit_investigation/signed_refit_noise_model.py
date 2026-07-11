"""Calibrate regional empirical null covariance for signed dipole curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from scipy.stats import spearmanr

from signed_refit_controls import (
    COL_REGIONS,
    CONTROL_VERSION,
    ROW_REGIONS,
    file_sha256,
)
from signed_refit_manifest import load_selected_image_files


NOISE_MODEL_VERSION = "signed-refit-noise-v2"
DEFAULT_CONTROLS = Path("signed_refit_control_pairs.npz")
DEFAULT_OUTPUT = Path("signed_refit_noise_model.h5")
DEFAULT_SUMMARY = Path("signed_refit_noise_model_report.md")
ELECTRONIZE_SCALE = 400.0
WINSOR_SIGMA = 5.0
EIGENVALUE_FLOOR_FRACTION = 1e-8
MAX_ACCEPTED_CONDITION = 1e8
MAX_TEMPLATE_PUMP_DELTA_CHI2 = 1.0
MAX_TEMPLATE_PUMP_ABS_Z = 1.0


def group_manifest_files(image_files: list[str]) -> dict[int, list[tuple[int, str]]]:
    groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for path in image_files:
        temperature = int(re.search(r"_(\d+)k_", path).group(1))
        dtph = int(re.search(r"dtph(\d+)_", path).group(1))
        groups[temperature].append((dtph, path))
    for temperature in groups:
        groups[temperature].sort()
    return dict(sorted(groups.items()))


def robust_scale_axis0(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    default = float(np.median(positive)) if positive.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, default)
    return center, scale


def oas_covariance(values: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x, axis=0)
    n_samples, n_features = x.shape
    empirical = x.T @ x / n_samples
    mu = float(np.trace(empirical) / n_features)
    alpha = float(np.mean(empirical**2))
    denominator = (n_samples + 1.0) * (alpha - mu**2 / n_features)
    if denominator <= 0:
        shrinkage = 1.0
    else:
        shrinkage = min((alpha + mu**2) / denominator, 1.0)
    covariance = (1.0 - shrinkage) * empirical
    covariance.flat[:: n_features + 1] += shrinkage * mu
    return covariance, float(shrinkage)


def positive_definite_covariance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    max_eigenvalue = float(np.max(eigenvalues))
    floor = max(max_eigenvalue * EIGENVALUE_FLOOR_FRACTION, 1e-9)
    eigenvalues = np.maximum(eigenvalues, floor)
    regularized = (eigenvectors * eigenvalues) @ eigenvectors.T
    regularized = 0.5 * (regularized + regularized.T)
    condition = float(np.max(eigenvalues) / np.min(eigenvalues))
    return regularized, eigenvalues, condition


def template_pump_projection_guard(path: Path | str) -> dict[str, float | int]:
    """Project every null template onto the pump family without subtracting it."""
    delta_values = []
    z_values = []
    with h5py.File(path, "r") as h5:
        for temp_name in sorted(
            (name for name in h5 if name.startswith("temp_")),
            key=lambda value: int(value.split("_")[1]),
        ):
            temp_group = h5[temp_name]
            seconds = np.asarray(temp_group["seconds"], dtype=float)
            tau_grid = np.geomspace(
                float(np.min(seconds)) / 10.0,
                float(np.max(seconds)) * 10.0,
                801,
            )
            shapes = 3000.0 * (
                np.exp(-seconds[None, :] / tau_grid[:, None])
                - np.exp(-8.0 * seconds[None, :] / tau_grid[:, None])
            )
            ones = np.ones(seconds.size)
            for quadrant in range(4):
                for region in range(ROW_REGIONS * COL_REGIONS):
                    rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                    precision = np.asarray(rgroup["precision"], dtype=float)
                    template = np.asarray(rgroup["null_template"], dtype=float)
                    p1 = precision @ ones
                    one_normal = float(ones @ p1)
                    corrected_template = (
                        template - ones * float(template @ p1) / one_normal
                    )
                    corrected_shapes = (
                        shapes
                        - np.outer((shapes @ p1) / one_normal, ones)
                    )
                    pshape = corrected_shapes @ precision
                    denominator = np.einsum(
                        "gi,gi->g",
                        pshape,
                        corrected_shapes,
                    )
                    numerator = pshape @ corrected_template
                    delta = np.divide(
                        numerator**2,
                        denominator,
                        out=np.zeros_like(numerator),
                        where=denominator > 0,
                    )
                    maximum = float(np.max(delta))
                    delta_values.append(maximum)
                    z_values.append(float(np.sqrt(maximum)))
    return {
        "count": len(delta_values),
        "delta_chi2_max": float(np.max(delta_values)),
        "delta_chi2_p95": float(np.percentile(delta_values, 95)),
        "abs_z_max": float(np.max(z_values)),
        "abs_z_p95": float(np.percentile(z_values, 95)),
    }


def extract_temperature_curves(
    files: list[tuple[int, str]],
    control_data: dict[str, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    intensity = {}
    background = {}
    for quadrant in range(4):
        qsel = control_data["quadrant"] == quadrant
        n_controls = int(np.count_nonzero(qsel))
        intensity[quadrant] = np.empty((n_controls, len(files)), dtype=np.float32)
        background[quadrant] = np.empty((n_controls, len(files)), dtype=np.float32)

    for delay_index, (_, path) in enumerate(files):
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
            for quadrant in range(4):
                qsel = control_data["quadrant"] == quadrant
                rows = control_data["row"][qsel]
                cols = control_data["col"][qsel]
                image = np.rint(hdus[quadrant].data[2:512, 8:3080] / ELECTRONIZE_SCALE)
                row_median = np.median(image, axis=1)
                image = image - row_median[:, None]
                intensity[quadrant][:, delay_index] = (
                    image[rows, cols] - image[rows - 1, cols]
                ) / 2.0
                background[quadrant][:, delay_index] = (
                    row_median[rows] + row_median[rows - 1]
                ) / 2.0
    return intensity, background


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = spearmanr(x, y, nan_policy="omit")
    value = float(result.statistic)
    return value if np.isfinite(value) else 0.0


def calibrate_region(
    curves: np.ndarray,
    backgrounds: np.ndarray,
    static_pair: np.ndarray,
) -> dict[str, object]:
    pair_offset = np.median(curves, axis=1)
    pair_centered = curves - pair_offset[:, None]
    null_template = np.median(pair_centered, axis=0)
    residual = pair_centered - null_template[None, :]

    robust_center, robust_scale = robust_scale_axis0(residual)
    clipped = np.clip(
        residual,
        robust_center - WINSOR_SIGMA * robust_scale,
        robust_center + WINSOR_SIGMA * robust_scale,
    )
    clipped -= np.mean(clipped, axis=0)

    sample_covariance = np.cov(residual, rowvar=False, ddof=1)
    robust_covariance = np.cov(clipped, rowvar=False, ddof=1)
    oas, shrinkage = oas_covariance(clipped)
    covariance, eigenvalues, condition = positive_definite_covariance(oas)
    precision = np.linalg.inv(covariance)
    sigma = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sigma, sigma)

    residual_rms = np.sqrt(np.mean(residual**2, axis=1))
    median_background = np.median(backgrounds, axis=1)
    sample_norm = float(np.linalg.norm(sample_covariance, ord="fro"))
    robust_difference = float(
        np.linalg.norm(robust_covariance - sample_covariance, ord="fro")
        / sample_norm
    ) if sample_norm > 0 else 0.0

    return {
        "null_template": null_template.astype(np.float32),
        "sample_covariance": sample_covariance,
        "robust_covariance": robust_covariance,
        "covariance": covariance,
        "precision": precision,
        "correlation": correlation,
        "eigenvalues": eigenvalues,
        "sigma": sigma,
        "shrinkage": shrinkage,
        "condition": condition,
        "robust_vs_classical_fractional_frobenius": robust_difference,
        "residual_rms_median": float(np.median(residual_rms)),
        "residual_rms_p95": float(np.percentile(residual_rms, 95)),
        "brightness_rms_spearman": safe_spearman(np.abs(static_pair), residual_rms),
        "background_rms_spearman": safe_spearman(median_background, residual_rms),
        "median_background": float(np.median(median_background)),
    }


def build_noise_model(
    manifest_path: Path,
    controls_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    image_files, manifest_sha256 = load_selected_image_files(manifest_path)
    control_npz = np.load(controls_path)
    control_metadata = json.loads(str(control_npz["metadata_json"]))
    if str(control_npz["version"]) != CONTROL_VERSION:
        raise ValueError(f"Unexpected control version: {control_npz['version']}")
    if str(control_npz["manifest_sha256"]) != manifest_sha256:
        raise ValueError("Control artifact does not match the frozen manifest")

    control_data = {
        key: control_npz[key]
        for key in (
            "quadrant",
            "row",
            "col",
            "region",
            "row_region",
            "col_region",
            "split",
            "static_pair_intensity",
        )
    }
    file_groups = group_manifest_files(image_files)
    summary_records = []

    with h5py.File(output_path, "w") as h5:
        h5.attrs["version"] = NOISE_MODEL_VERSION
        h5.attrs["manifest_sha256"] = manifest_sha256
        h5.attrs["controls_sha256"] = file_sha256(controls_path)
        h5.attrs["controls_version"] = CONTROL_VERSION
        h5.attrs["estimator"] = "5-MAD winsorized OAS covariance with eigenvalue floor"
        h5.attrs["winsor_sigma"] = WINSOR_SIGMA
        h5.attrs["eigenvalue_floor_fraction"] = EIGENVALUE_FLOOR_FRACTION
        h5.attrs["mapping"] = (
            "Assign a candidate to its 4x8 cropped-quadrant detector region; "
            "use covariance[temp, quadrant, region] on that scan's sorted dtph grid."
        )

        controls_group = h5.create_group("controls")
        for key, values in control_data.items():
            controls_group.create_dataset(key, data=values, compression="gzip")

        for temperature, files in file_groups.items():
            print(f"Extracting {temperature} K: {len(files)} dwell images")
            intensity, background = extract_temperature_curves(files, control_data)
            temp_group = h5.create_group(f"temp_{temperature}")
            dtph = np.asarray([item[0] for item in files], dtype=np.int64)
            temp_group.create_dataset("dtph", data=dtph)
            temp_group.create_dataset("seconds", data=dtph / 15e6)
            string_dtype = h5py.string_dtype("utf-8")
            temp_group.create_dataset(
                "source_fits",
                data=np.asarray([item[1] for item in files], dtype=object),
                dtype=string_dtype,
            )

            for quadrant in range(4):
                qsel = control_data["quadrant"] == quadrant
                qgroup = temp_group.create_group(f"quad_{quadrant}")
                qgroup.create_dataset(
                    "intensity",
                    data=intensity[quadrant],
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                qgroup.create_dataset(
                    "background",
                    data=background[quadrant],
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
                q_regions = control_data["region"][qsel]
                q_splits = control_data["split"][qsel]
                q_static = control_data["static_pair_intensity"][qsel]

                for region in range(ROW_REGIONS * COL_REGIONS):
                    train = (q_regions == region) & (q_splits == 0)
                    validation = (q_regions == region) & (q_splits == 1)
                    result = calibrate_region(
                        intensity[quadrant][train],
                        background[quadrant][train],
                        q_static[train],
                    )
                    rgroup = qgroup.create_group(f"region_{region}")
                    rgroup.attrs["n_train"] = int(np.count_nonzero(train))
                    rgroup.attrs["n_validation"] = int(np.count_nonzero(validation))
                    rgroup.attrs["shrinkage"] = result["shrinkage"]
                    rgroup.attrs["condition"] = result["condition"]
                    rgroup.attrs["robust_vs_classical_fractional_frobenius"] = result[
                        "robust_vs_classical_fractional_frobenius"
                    ]
                    rgroup.attrs["residual_rms_median"] = result["residual_rms_median"]
                    rgroup.attrs["residual_rms_p95"] = result["residual_rms_p95"]
                    rgroup.attrs["brightness_rms_spearman"] = result[
                        "brightness_rms_spearman"
                    ]
                    rgroup.attrs["background_rms_spearman"] = result[
                        "background_rms_spearman"
                    ]
                    rgroup.attrs["median_background"] = result["median_background"]
                    for key in (
                        "null_template",
                        "sample_covariance",
                        "robust_covariance",
                        "covariance",
                        "precision",
                        "correlation",
                        "eigenvalues",
                        "sigma",
                    ):
                        rgroup.create_dataset(key, data=result[key], compression="gzip")
                    summary_records.append(
                        {
                            "temperature": temperature,
                            "quadrant": quadrant,
                            "region": region,
                            "n_dtph": len(files),
                            "n_train": int(np.count_nonzero(train)),
                            "n_validation": int(np.count_nonzero(validation)),
                            "condition": result["condition"],
                            "shrinkage": result["shrinkage"],
                            "sigma_min": float(np.min(result["sigma"])),
                            "sigma_median": float(np.median(result["sigma"])),
                            "sigma_max": float(np.max(result["sigma"])),
                            "max_abs_offdiag_correlation": float(
                                np.max(
                                    np.abs(
                                        result["correlation"]
                                        - np.eye(result["correlation"].shape[0])
                                    )
                                )
                            ),
                            "robust_difference": result[
                                "robust_vs_classical_fractional_frobenius"
                            ],
                            "brightness_correlation": result["brightness_rms_spearman"],
                            "background_correlation": result["background_rms_spearman"],
                            "median_background": result["median_background"],
                        }
                    )

        metadata = {
            "version": NOISE_MODEL_VERSION,
            "manifest_sha256": manifest_sha256,
            "controls_sha256": file_sha256(controls_path),
            "controls_version": CONTROL_VERSION,
            "temperature_count": len(file_groups),
            "covariance_count": len(summary_records),
            "region_grid": [ROW_REGIONS, COL_REGIONS],
            "estimator": {
                "pair_offset": "per-control median across dtph",
                "null_template": "per-dtph median of pair-centered training curves",
                "robustification": f"per-dtph winsorization at {WINSOR_SIGMA} robust sigma",
                "shrinkage": "Oracle Approximating Shrinkage toward scaled identity",
                "positive_definite_floor_fraction": EIGENVALUE_FLOOR_FRACTION,
            },
            "mapping": (
                "Candidate cropped coordinates map to one of 4 row x 8 column regions. "
                "Use the exact temperature/quadrant/region covariance and dtph grid."
            ),
            "control_selection_metadata": control_metadata,
        }
        h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

    validation = validate_noise_model(output_path, manifest_path, controls_path)
    write_summary(summary_path, output_path, validation, summary_records)
    return validation


def validate_noise_model(
    path: Path,
    manifest_path: Path,
    controls_path: Path,
) -> dict[str, object]:
    _, manifest_sha256 = load_selected_image_files(manifest_path)
    errors = []
    conditions = []
    shrinkages = []
    minimum_eigenvalue = np.inf
    covariance_count = 0
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("version", "") != NOISE_MODEL_VERSION:
            errors.append(f"Unexpected noise model version: {h5.attrs.get('version')}")
        if h5.attrs.get("manifest_sha256", "") != manifest_sha256:
            errors.append("Noise model manifest hash mismatch")
        if h5.attrs.get("controls_sha256", "") != file_sha256(controls_path):
            errors.append("Noise model control artifact hash mismatch")
        temperature_groups = sorted(key for key in h5 if key.startswith("temp_"))
        for temp_name in temperature_groups:
            temp_group = h5[temp_name]
            n_dtph = len(temp_group["dtph"])
            if len(np.unique(temp_group["dtph"][()])) != n_dtph:
                errors.append(f"{temp_name} has duplicate dtph values")
            for quadrant in range(4):
                qgroup = temp_group[f"quad_{quadrant}"]
                if qgroup["intensity"].shape[1] != n_dtph:
                    errors.append(f"{temp_name}/quad_{quadrant} intensity grid mismatch")
                for region in range(ROW_REGIONS * COL_REGIONS):
                    rgroup = qgroup[f"region_{region}"]
                    covariance = rgroup["covariance"][()]
                    precision = rgroup["precision"][()]
                    eigenvalues = rgroup["eigenvalues"][()]
                    if covariance.shape != (n_dtph, n_dtph):
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} covariance shape mismatch"
                        )
                    if not np.all(np.isfinite(covariance)):
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} nonfinite covariance"
                        )
                    if np.min(eigenvalues) <= 0:
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} nonpositive eigenvalue"
                        )
                    identity_error = float(
                        np.max(np.abs(covariance @ precision - np.eye(n_dtph)))
                    )
                    if identity_error > 1e-6:
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} precision mismatch "
                            f"{identity_error}"
                        )
                    condition = float(rgroup.attrs["condition"])
                    if condition > MAX_ACCEPTED_CONDITION:
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} condition {condition}"
                        )
                    if int(rgroup.attrs["n_train"]) != 384:
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} wrong train count"
                        )
                    if int(rgroup.attrs["n_validation"]) != 128:
                        errors.append(
                            f"{temp_name}/quad_{quadrant}/region_{region} wrong validation count"
                        )
                    conditions.append(condition)
                    shrinkages.append(float(rgroup.attrs["shrinkage"]))
                    minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(eigenvalues)))
                    covariance_count += 1
    projection = template_pump_projection_guard(path)
    if projection["delta_chi2_max"] > MAX_TEMPLATE_PUMP_DELTA_CHI2:
        errors.append(
            "null-template pump projection delta chi2 "
            f"{projection['delta_chi2_max']}"
        )
    if projection["abs_z_max"] > MAX_TEMPLATE_PUMP_ABS_Z:
        errors.append(
            f"null-template pump projection |z| {projection['abs_z_max']}"
        )
    if errors:
        raise ValueError("Noise-model validation failed:\n- " + "\n- ".join(errors[:50]))
    return {
        "covariance_count": covariance_count,
        "condition_min": float(np.min(conditions)),
        "condition_median": float(np.median(conditions)),
        "condition_max": float(np.max(conditions)),
        "shrinkage_min": float(np.min(shrinkages)),
        "shrinkage_median": float(np.median(shrinkages)),
        "shrinkage_max": float(np.max(shrinkages)),
        "minimum_eigenvalue": minimum_eigenvalue,
        "noise_model_sha256": file_sha256(path),
        "template_projection": projection,
    }


def percentile_summary(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    return (
        f"min {np.min(array):.3g}, median {np.median(array):.3g}, "
        f"p95 {np.percentile(array, 95):.3g}, max {np.max(array):.3g}"
    )


def write_summary(
    path: Path,
    output_path: Path,
    validation: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    lines = [
        "# Signed Refit Empirical Noise Model",
        "",
        f"- Noise-model version: `{NOISE_MODEL_VERSION}`",
        f"- Artifact: `{output_path}`",
        f"- Artifact SHA-256: `{validation['noise_model_sha256']}`",
        f"- Covariance matrices: {validation['covariance_count']}",
        "- Mapping: exact temperature and quadrant, then the candidate's 4 x 8 "
        "cropped-detector region.",
        "- Training controls per covariance: 384.",
        "- Held-out validation controls per covariance: 128; not used here.",
        "",
        "## Estimator",
        "",
        "1. Remove each control pair's median across the dwell scan.",
        "2. Remove the training ensemble's median null template at each dwell point.",
        f"3. Winsorize each dwell coordinate at {WINSOR_SIGMA} robust sigma.",
        "4. Estimate Oracle Approximating Shrinkage covariance toward a scaled identity.",
        f"5. Apply a relative eigenvalue floor of {EIGENVALUE_FLOOR_FRACTION:g}.",
        "",
        "Classical and robust sample covariance matrices are also stored for audit.",
        "",
        "## Numerical validation",
        "",
        f"- Condition number: min {validation['condition_min']:.3g}, "
        f"median {validation['condition_median']:.3g}, "
        f"max {validation['condition_max']:.3g}.",
        f"- OAS shrinkage: min {validation['shrinkage_min']:.3g}, "
        f"median {validation['shrinkage_median']:.3g}, "
        f"max {validation['shrinkage_max']:.3g}.",
        f"- Minimum regularized eigenvalue: "
        f"{validation['minimum_eigenvalue']:.6g} (electrons)^2.",
        "- Null-template pump projection delta-chi-square max/p95: "
        f"{validation['template_projection']['delta_chi2_max']:.3f}/"
        f"{validation['template_projection']['delta_chi2_p95']:.3f}.",
        "- Null-template conditional |z| max/p95: "
        f"{validation['template_projection']['abs_z_max']:.3f}/"
        f"{validation['template_projection']['abs_z_p95']:.3f}.",
        "- Every covariance is finite, positive definite, invertible, and matched to "
        "its scan's unique sorted dtph grid.",
        "",
        "## Observed dependence",
        "",
        f"- Per-dwell sigma (e-): {percentile_summary([r['sigma_median'] for r in records])}.",
        f"- Maximum absolute off-diagonal correlation: "
        f"{percentile_summary([r['max_abs_offdiag_correlation'] for r in records])}.",
        f"- Robust/classical covariance fractional Frobenius difference: "
        f"{percentile_summary([r['robust_difference'] for r in records])}.",
        f"- Spearman residual-RMS versus static pair brightness: "
        f"{percentile_summary([r['brightness_correlation'] for r in records])}.",
        f"- Spearman residual-RMS versus generated-charge background: "
        f"{percentile_summary([r['background_correlation'] for r in records])}.",
        "",
        "The HDF5 file stores per-temperature, quadrant, region, and dwell sigma/correlation "
        "values so temperature, quadrant, region, and dtph dependence remain explicit.",
        "",
        "## Acceptance gate",
        "",
        "- PASS: every production scan has an invertible regional covariance model.",
        "- PASS: condition numbers and shrinkage strengths are recorded.",
        "- PASS: detector-region and acquisition dependence are retained rather than "
        "collapsed into one scalar.",
        "- PASS: the empirical null template has negligible projection onto every "
        "scan's pump-shape family.",
        "- PASS: the model was frozen without examining candidate acceptance changes.",
        "",
        "Held-out residual closure is intentionally deferred to Step 4.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("signed_refit_manifest.csv"))
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        result = validate_noise_model(args.output, args.manifest, args.controls)
    else:
        result = build_noise_model(args.manifest, args.controls, args.output, args.summary)
    print(
        f"PASS: {result['covariance_count']} covariance matrices; "
        f"condition max {result['condition_max']:.3g}; "
        f"SHA-256 {result['noise_model_sha256']}"
    )


if __name__ == "__main__":
    main()

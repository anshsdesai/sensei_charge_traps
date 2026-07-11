"""Validate null scaling and candidate variance on held-out real residual curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from signed_refit_controls import COL_REGIONS, ROW_REGIONS, file_sha256
from signed_refit_noise_model import ELECTRONIZE_SCALE, NOISE_MODEL_VERSION
from signed_refit_profile_fitter import (
    ProfileTauFitter,
    SIGNAL_DEPENDENT_FITTER_VERSION,
    SignalDependentProfileTauFitter,
    intensity_model,
)
from signed_refit_variance_model import (
    N_PUMPS,
    VARIANCE_MODEL_VERSION,
    transfer_probability,
)


VALIDATION_VERSION = "signed-refit-candidate-variance-validation-v1"
DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_MANIFEST = Path("signed_refit_manifest.csv")
DEFAULT_DETECTION = Path("signed_refit_detection_calibration.npz")
DEFAULT_OUTPUT = Path("signed_refit_variance_validation.npz")
DEFAULT_REPORT = Path("signed_refit_variance_validation.md")
RANDOM_SEED = 2026061307

AMPLITUDES = (0.03, 0.10, 0.30)
EXCESS_PAIR_CHARGE = (0.0, 500.0, 2000.0)
SCENARIOS = (
    ("cold-short", 125, 0, 0, 3e-4, 1),
    ("cold-mid", 145, 1, 10, 3e-3, -1),
    ("lever-mid", 170, 2, 20, 1e-2, 1),
    ("warm-mid", 183, 2, 20, 3e-2, -1),
    ("warm-long", 197, 3, 7, 1e-1, 1),
    ("hot-short", 210, 3, 15, 1e-3, -1),
)

NULL_EVALUATION_WIDTH_RANGE = (0.94, 1.06)
AGGREGATE_COVERAGE_RANGE = (0.62, 0.74)
AMPLITUDE_COVERAGE_RANGE = (0.60, 0.76)
MAX_ABS_AMPLITUDE_MEDIAN_BIAS_DEX = 0.04
AMPLITUDE_CLOSURE_WIDTH_RANGE = (0.85, 1.12)
MAX_AMPLITUDE_WIDTH_SPREAD = 0.08
MAX_TEMPLATE_DELTA_CHI2 = 1.0
MAX_TEMPLATE_ABS_Z = 1.0
MIN_IDENTIFIABLE_RATE_FOR_COVERAGE_GATE = 0.80


def _validation_indices(
    quadrants: np.ndarray,
    regions: np.ndarray,
    split: np.ndarray,
    quadrant: int,
    region: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(
        (quadrants == quadrant) & (regions == region) & (split == 1)
    )
    if indices.size != 128:
        raise ValueError(
            f"Expected 128 held-out controls in Q{quadrant}/R{region}, "
            f"found {indices.size}"
        )
    return indices[:64], indices[64:]


def _cell_curves(
    model: h5py.File,
    temp_group: h5py.Group,
    quadrant: int,
    global_indices: np.ndarray,
) -> np.ndarray:
    global_quadrant = np.asarray(model["controls/quadrant"], dtype=np.int8)
    q_global = np.flatnonzero(global_quadrant == quadrant)
    positions = np.searchsorted(q_global, global_indices)
    if np.any(q_global[positions] != global_indices):
        raise ValueError("Control-index mapping to quadrant intensity failed")
    return np.asarray(temp_group[f"quad_{quadrant}/intensity"][positions], dtype=float)


def calibrate_null_scales(model: h5py.File) -> dict[int, dict[str, float]]:
    quadrants = np.asarray(model["controls/quadrant"], dtype=np.int8)
    regions = np.asarray(model["controls/region"], dtype=np.int8)
    split = np.asarray(model["controls/split"], dtype=np.int8)
    output = {}
    for temp_name in sorted(
        (name for name in model if name.startswith("temp_")),
        key=lambda value: int(value.split("_")[1]),
    ):
        temperature = int(temp_name.split("_")[1])
        temp_group = model[temp_name]
        calibration_z = []
        evaluation_z = []
        evaluation_pvalues = []
        for quadrant in range(4):
            for region in range(ROW_REGIONS * COL_REGIONS):
                calibration_indices, evaluation_indices = _validation_indices(
                    quadrants,
                    regions,
                    split,
                    quadrant,
                    region,
                )
                rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                covariance = np.asarray(rgroup["covariance"], dtype=float)
                template = np.asarray(rgroup["null_template"], dtype=float)
                cholesky = np.linalg.cholesky(covariance)
                for target, indices in (
                    (calibration_z, calibration_indices),
                    (evaluation_z, evaluation_indices),
                ):
                    curves = _cell_curves(model, temp_group, quadrant, indices)
                    centered = curves - np.median(curves, axis=1, keepdims=True)
                    residual = centered - template[None, :]
                    target.append(
                        solve_triangular(cholesky, residual.T, lower=True).T.ravel()
                    )

        calibration_values = np.concatenate(calibration_z)
        raw_calibration_width = float(np.std(calibration_values, ddof=1))
        scale = max(raw_calibration_width**2, 1.0)
        evaluation_values = np.concatenate(evaluation_z) / np.sqrt(scale)

        # Constant-fit tails remain diagnostic only; detection uses Step 6.
        for quadrant in range(4):
            for region in range(ROW_REGIONS * COL_REGIONS):
                _, evaluation_indices = _validation_indices(
                    quadrants,
                    regions,
                    split,
                    quadrant,
                    region,
                )
                rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                covariance = scale * np.asarray(rgroup["covariance"], dtype=float)
                template = np.asarray(rgroup["null_template"], dtype=float)
                curves = _cell_curves(model, temp_group, quadrant, evaluation_indices)
                fitter = ProfileTauFitter(
                    np.asarray(temp_group["seconds"], dtype=float),
                    covariance,
                    null_template=template,
                )
                corrected = curves - template[None, :]
                precision = fitter.precision
                p1 = fitter.precision_ones
                offsets = corrected @ p1 / fitter.constant_normal
                residual = corrected - offsets[:, None]
                values = np.einsum("ni,ij,nj->n", residual, precision, residual)
                evaluation_pvalues.append(
                    chi2.sf(values, temp_group["seconds"].size - 1)
                )
        pvalues = np.concatenate(evaluation_pvalues)
        output[temperature] = {
            "calibration_width_raw": raw_calibration_width,
            "null_scale": float(scale),
            "evaluation_width_scaled": float(np.std(evaluation_values, ddof=1)),
            "evaluation_p05_rate": float(np.mean(pvalues < 0.05)),
            "evaluation_p01_rate": float(np.mean(pvalues < 0.01)),
        }
    return output


def template_projection_and_correlation(model: h5py.File) -> dict[str, object]:
    quadrants = np.asarray(model["controls/quadrant"], dtype=np.int8)
    regions = np.asarray(model["controls/region"], dtype=np.int8)
    split = np.asarray(model["controls/split"], dtype=np.int8)
    template_delta = []
    template_abs_z = []
    stored_max = []
    validation_at_stored_max = []
    high_count = 0
    high_same_sign = 0
    high_validation_half = 0
    matrix_correlations = []

    for temp_name in sorted(name for name in model if name.startswith("temp_")):
        temp_group = model[temp_name]
        seconds = np.asarray(temp_group["seconds"], dtype=float)
        for quadrant in range(4):
            for region in range(ROW_REGIONS * COL_REGIONS):
                rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                covariance = np.asarray(rgroup["covariance"], dtype=float)
                template = np.asarray(rgroup["null_template"], dtype=float)
                fitter = ProfileTauFitter(
                    seconds,
                    covariance,
                    null_template=np.zeros(seconds.size),
                )
                projection = fitter.batch_profile_statistic(template)
                best_index = int(projection["best_grid_index"][0])
                amplitude_sigma = np.sqrt(
                    fitter.constant_normal / fitter.determinant[best_index]
                )
                template_delta.append(float(projection["delta_chi2"][0]))
                template_abs_z.append(
                    abs(float(projection["amplitude"][0])) / amplitude_sigma
                )

                stored = np.asarray(rgroup["correlation"], dtype=float)
                offdiag = np.abs(stored - np.eye(stored.shape[0]))
                i, j = np.unravel_index(np.argmax(offdiag), offdiag.shape)
                stored_value = float(stored[i, j])
                stored_max.append(abs(stored_value))

                _, evaluation_indices = _validation_indices(
                    quadrants,
                    regions,
                    split,
                    quadrant,
                    region,
                )
                curves = _cell_curves(model, temp_group, quadrant, evaluation_indices)
                centered = curves - np.median(curves, axis=1, keepdims=True)
                residual = centered - template[None, :]
                validation = np.corrcoef(residual, rowvar=False)
                validation_value = float(validation[i, j])
                validation_at_stored_max.append(validation_value)

                tri = np.triu_indices(seconds.size, 1)
                finite = np.isfinite(validation[tri])
                if np.count_nonzero(finite) >= 3:
                    matrix_correlations.append(
                        float(
                            np.corrcoef(
                                stored[tri][finite],
                                validation[tri][finite],
                            )[0, 1]
                        )
                    )
                if abs(stored_value) >= 0.8:
                    high_count += 1
                    high_same_sign += int(np.sign(stored_value) == np.sign(validation_value))
                    high_validation_half += int(abs(validation_value) >= 0.5)

    return {
        "template_projection": {
            "count": len(template_delta),
            "delta_chi2_max": float(np.max(template_delta)),
            "delta_chi2_p95": float(np.percentile(template_delta, 95)),
            "abs_z_max": float(np.max(template_abs_z)),
            "abs_z_p95": float(np.percentile(template_abs_z, 95)),
        },
        "offdiagonal_correlation": {
            "stored_max": float(np.max(stored_max)),
            "stored_p95": float(np.percentile(stored_max, 95)),
            "validation_at_stored_max_median_abs": float(
                np.median(np.abs(validation_at_stored_max))
            ),
            "matrix_train_validation_correlation_median": float(
                np.nanmedian(matrix_correlations)
            ),
            "stored_abs_ge_0p8_count": int(high_count),
            "same_sign_fraction_for_stored_abs_ge_0p8": (
                float(high_same_sign / high_count) if high_count else None
            ),
            "validation_abs_ge_0p5_fraction_for_stored_abs_ge_0p8": (
                float(high_validation_half / high_count) if high_count else None
            ),
        },
    }


def run_real_residual_injections(
    model: h5py.File,
    null_scales: dict[int, dict[str, float]],
    detection_thresholds: dict[int, float],
) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    quadrants = np.asarray(model["controls/quadrant"], dtype=np.int8)
    regions = np.asarray(model["controls/region"], dtype=np.int8)
    split = np.asarray(model["controls/split"], dtype=np.int8)
    records = []

    for scenario_name, temperature, quadrant, region, tau_true, sign in SCENARIOS:
        temp_group = model[f"temp_{temperature}"]
        seconds = np.asarray(temp_group["seconds"], dtype=float)
        rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
        covariance = np.asarray(rgroup["covariance"], dtype=float)
        template = np.asarray(rgroup["null_template"], dtype=float)
        detection_fitter = ProfileTauFitter(
            seconds,
            covariance,
            null_template=template,
        )
        _, evaluation_indices = _validation_indices(
            quadrants,
            regions,
            split,
            quadrant,
            region,
        )
        null_curves = _cell_curves(
            model,
            temp_group,
            quadrant,
            evaluation_indices,
        )
        for amplitude_abs, excess_charge in zip(AMPLITUDES, EXCESS_PAIR_CHARGE):
            amplitude_true = float(sign * amplitude_abs)
            _, probability = transfer_probability(
                seconds,
                amplitude_true,
                tau_true,
            )
            expected_transfer = N_PUMPS * probability
            extra_shot_variance = np.full(seconds.size, excess_charge / 4.0)
            fitter = SignalDependentProfileTauFitter(
                seconds,
                covariance,
                null_template=template,
                null_scale=null_scales[temperature]["null_scale"],
                pump_overdispersion=1.0,
                extra_pair_shot=extra_shot_variance,
            )
            for null_curve in null_curves:
                transfer_draw = rng.binomial(N_PUMPS, probability)
                transfer_fluctuation = sign * (
                    transfer_draw - expected_transfer
                )
                if excess_charge > 0:
                    lobe_mean = excess_charge / 2.0
                    shot_fluctuation = (
                        rng.poisson(lobe_mean, seconds.size)
                        - rng.poisson(lobe_mean, seconds.size)
                    ) / 2.0
                else:
                    shot_fluctuation = np.zeros(seconds.size)
                observed = (
                    null_curve
                    + intensity_model(
                        seconds,
                        amplitude_true,
                        tau_true,
                        0.0,
                    )
                    + transfer_fluctuation
                    + shot_fluctuation
                )
                detection_statistic = float(
                    detection_fitter.batch_profile_statistic(observed)[
                        "delta_chi2"
                    ][0]
                )
                result = fitter.fit(observed)
                lower = (
                    fitter._fitter(result["effective_covariance"]).tau_bounds[0]
                    if result["tau_interval_lower"] is None
                    else result["tau_interval_lower"]
                )
                upper = (
                    fitter._fitter(result["effective_covariance"]).tau_bounds[1]
                    if result["tau_interval_upper"] is None
                    else result["tau_interval_upper"]
                )
                records.append(
                    {
                        "scenario": scenario_name,
                        "temperature": temperature,
                        "amplitude_true": amplitude_true,
                        "amplitude_fit": float(result["amplitude"]),
                        "tau_true": tau_true,
                        "tau_fit": float(result["tau"]),
                        "covered": bool(lower <= tau_true <= upper),
                        "chi2": float(result["chi2"]),
                        "dof": int(result["dof"]),
                        "boundary_limited": bool(result["boundary_limited"]),
                        "multimodal": bool(result["multimodal"]),
                        "two_sided_interval": bool(
                            result["tau_interval_lower"] is not None
                            and result["tau_interval_upper"] is not None
                        ),
                        "detection_statistic": detection_statistic,
                        "passes_detection": bool(
                            detection_statistic >= detection_thresholds[temperature]
                        ),
                        "variance_converged": bool(result["variance_converged"]),
                        "excess_pair_charge": excess_charge,
                    }
                )

    for item in records:
        item["characterization_eligible"] = bool(
            item["passes_detection"]
            and item["two_sided_interval"]
            and not item["boundary_limited"]
            and not item["multimodal"]
        )

    amplitude_summary = {}
    for amplitude_abs in AMPLITUDES:
        selected = [
            item
            for item in records
            if np.isclose(abs(item["amplitude_true"]), amplitude_abs)
        ]
        bias = np.asarray(
            [np.log10(item["tau_fit"] / item["tau_true"]) for item in selected]
        )
        eligible = [item for item in selected if item["characterization_eligible"]]
        eligible_bias = np.asarray(
            [np.log10(item["tau_fit"] / item["tau_true"]) for item in eligible]
        )
        amplitude_summary[f"{amplitude_abs:.2f}"] = {
            "count": len(selected),
            "coverage": float(np.mean([item["covered"] for item in selected])),
            "median_bias_dex": float(np.median(bias)),
            "closure_width": float(
                np.sqrt(
                    np.sum([item["chi2"] for item in selected])
                    / np.sum([item["dof"] for item in selected])
                )
            ),
            "boundary_rate": float(
                np.mean([item["boundary_limited"] for item in selected])
            ),
            "detection_rate": float(
                np.mean([item["passes_detection"] for item in selected])
            ),
            "identifiable_rate": float(
                np.mean([item["characterization_eligible"] for item in selected])
            ),
            "eligible_count": len(eligible),
            "eligible_coverage": (
                float(np.mean([item["covered"] for item in eligible]))
                if eligible
                else None
            ),
            "eligible_median_bias_dex": (
                float(np.median(eligible_bias)) if eligible else None
            ),
        }

    eligible_records = [
        item for item in records if item["characterization_eligible"]
    ]
    fitted_amplitude = np.abs(
        np.asarray([item["amplitude_fit"] for item in eligible_records], dtype=float)
    )
    quartile_edges = np.quantile(fitted_amplitude, [0.25, 0.5, 0.75])
    quartile_index = np.digitize(fitted_amplitude, quartile_edges)
    fitted_quartiles = {}
    for quartile in range(4):
        selected = [
            item
            for index, item in enumerate(eligible_records)
            if quartile_index[index] == quartile
        ]
        fitted_quartiles[str(quartile + 1)] = {
            "count": len(selected),
            "amplitude_median": float(
                np.median(
                    [
                        fitted_amplitude[index]
                        for index in np.flatnonzero(quartile_index == quartile)
                    ]
                )
            ),
            "coverage": float(np.mean([item["covered"] for item in selected])),
            "closure_width": float(
                np.sqrt(
                    np.sum([item["chi2"] for item in selected])
                    / np.sum([item["dof"] for item in selected])
                )
            ),
        }

    all_bias = np.asarray(
        [np.log10(item["tau_fit"] / item["tau_true"]) for item in records]
    )
    eligible_bias = np.asarray(
        [
            np.log10(item["tau_fit"] / item["tau_true"])
            for item in eligible_records
        ]
    )
    return {
        "record_count": len(records),
        "aggregate": {
            "coverage": float(np.mean([item["covered"] for item in records])),
            "median_bias_dex": float(np.median(all_bias)),
            "bias_p16_dex": float(np.percentile(all_bias, 16)),
            "bias_p84_dex": float(np.percentile(all_bias, 84)),
            "variance_convergence_rate": float(
                np.mean([item["variance_converged"] for item in records])
            ),
        },
        "eligible": {
            "count": len(eligible_records),
            "rate": float(len(eligible_records) / len(records)),
            "coverage": float(
                np.mean([item["covered"] for item in eligible_records])
            ),
            "median_bias_dex": float(np.median(eligible_bias)),
        },
        "by_true_amplitude": amplitude_summary,
        "by_fitted_amplitude_quartile": fitted_quartiles,
    }


def audit_gain_sidecars(manifest_path: Path) -> dict[str, object]:
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["selected"] == "1":
                rows.append(row)
    csv_missing = 0
    csv_empty = 0
    xml_missing = 0
    fitted_gains = []
    defaults = []
    for row in rows:
        stem = Path(row["file_name"]).stem.replace("proc_", "cal_", 1)
        csv_path = manifest_path.parent / "cal" / f"{stem}.csv"
        xml_path = manifest_path.parent / "cal" / f"{stem}.xml"
        if not csv_path.exists():
            csv_missing += 1
        else:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                calibration = next(csv.DictReader(handle), None)
            if calibration is None:
                csv_empty += 1
            else:
                for key in ("gain1", "gain2", "gain3", "gain4"):
                    if calibration.get(key):
                        fitted_gains.append(float(calibration[key]))
        if not xml_path.exists():
            xml_missing += 1
        else:
            root = ET.parse(xml_path).getroot()
            node = root.find("./calibration/default")
            if node is not None and node.get("gain"):
                defaults.append(float(node.get("gain")))
    return {
        "selected_image_count": len(rows),
        "missing_csv_count": csv_missing,
        "empty_csv_count": csv_empty,
        "missing_xml_count": xml_missing,
        "fitted_gain_count": len(fitted_gains),
        "fitted_gain_min": min(fitted_gains) if fitted_gains else None,
        "fitted_gain_max": max(fitted_gains) if fitted_gains else None,
        "xml_default_count": len(defaults),
        "xml_default_unique": sorted(set(defaults)),
        "pipeline_global_scale": float(ELECTRONIZE_SCALE),
        "status": (
            "CONFIRMED_FROM_IMAGE_CALIBRATIONS"
            if fitted_gains
            and np.allclose(fitted_gains, ELECTRONIZE_SCALE, rtol=0.02)
            else "REQUIRES_EXTERNAL_GLOBAL_CALIBRATION_CONFIRMATION"
        ),
    }


def evaluate_acceptance(metrics: dict[str, object]) -> dict[str, object]:
    failures = []
    null_scales = metrics["null_scale_by_temperature"]
    for temperature, item in null_scales.items():
        width = item["evaluation_width_scaled"]
        if not NULL_EVALUATION_WIDTH_RANGE[0] <= width <= NULL_EVALUATION_WIDTH_RANGE[1]:
            failures.append(f"{temperature} K scaled null width {width:.3f}")

    injection = metrics["real_residual_injection"]
    coverage = injection["eligible"]["coverage"]
    if not AGGREGATE_COVERAGE_RANGE[0] <= coverage <= AGGREGATE_COVERAGE_RANGE[1]:
        failures.append(f"eligible real-residual coverage {coverage:.3f}")
    widths = []
    for amplitude, item in injection["by_true_amplitude"].items():
        if item["identifiable_rate"] >= MIN_IDENTIFIABLE_RATE_FOR_COVERAGE_GATE:
            if not AMPLITUDE_COVERAGE_RANGE[0] <= item["eligible_coverage"] <= AMPLITUDE_COVERAGE_RANGE[1]:
                failures.append(
                    f"A={amplitude} eligible coverage "
                    f"{item['eligible_coverage']:.3f}"
                )
            if (
                abs(item["eligible_median_bias_dex"])
                > MAX_ABS_AMPLITUDE_MEDIAN_BIAS_DEX
            ):
                failures.append(
                    f"A={amplitude} eligible median bias "
                    f"{item['eligible_median_bias_dex']:.3f} dex"
                )
        if not AMPLITUDE_CLOSURE_WIDTH_RANGE[0] <= item["closure_width"] <= AMPLITUDE_CLOSURE_WIDTH_RANGE[1]:
            failures.append(
                f"A={amplitude} closure width {item['closure_width']:.3f}"
            )
        widths.append(item["closure_width"])
    fitted_widths = [
        item["closure_width"]
        for item in injection["by_fitted_amplitude_quartile"].values()
    ]
    if max(fitted_widths) - min(fitted_widths) > MAX_AMPLITUDE_WIDTH_SPREAD:
        failures.append(
            "fitted-amplitude closure-width spread "
            f"{max(fitted_widths) - min(fitted_widths):.3f}"
        )

    projection = metrics["template_projection"]
    if projection["delta_chi2_max"] > MAX_TEMPLATE_DELTA_CHI2:
        failures.append(
            f"template projection delta chi2 max {projection['delta_chi2_max']:.3f}"
        )
    if projection["abs_z_max"] > MAX_TEMPLATE_ABS_Z:
        failures.append(f"template projection |z| max {projection['abs_z_max']:.3f}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def load_detection_thresholds(path: Path) -> dict[int, float]:
    data = np.load(path, allow_pickle=False)
    return {
        int(temperature): float(threshold)
        for temperature, threshold in zip(
            np.asarray(data["temperatures"], dtype=int),
            np.asarray(data["thresholds"], dtype=float),
        )
    }


def run_validation(
    model_path: Path,
    manifest_path: Path,
    detection_path: Path,
) -> dict[str, object]:
    detection_thresholds = load_detection_thresholds(detection_path)
    with h5py.File(model_path, "r") as model:
        if str(model.attrs.get("version", "")) != NOISE_MODEL_VERSION:
            raise ValueError("Unexpected noise model version")
        null_scales = calibrate_null_scales(model)
        diagnostics = template_projection_and_correlation(model)
        injection = run_real_residual_injections(
            model,
            null_scales,
            detection_thresholds,
        )
    metrics = {
        "validation_version": VALIDATION_VERSION,
        "variance_model_version": VARIANCE_MODEL_VERSION,
        "signal_dependent_fitter_version": SIGNAL_DEPENDENT_FITTER_VERSION,
        "noise_model_version": NOISE_MODEL_VERSION,
        "noise_model_sha256": file_sha256(model_path),
        "detection_calibration_sha256": file_sha256(detection_path),
        "random_seed": RANDOM_SEED,
        "null_scale_by_temperature": null_scales,
        **diagnostics,
        "real_residual_injection": injection,
        "gain_audit": audit_gain_sidecars(manifest_path),
    }
    metrics["acceptance"] = evaluate_acceptance(metrics)
    return metrics


def write_outputs(
    metrics: dict[str, object],
    output_path: Path,
    report_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        temperatures=np.asarray(
            sorted(int(key) for key in metrics["null_scale_by_temperature"]),
            dtype=np.int16,
        ),
        null_scales=np.asarray(
            [
                metrics["null_scale_by_temperature"][temperature]["null_scale"]
                for temperature in sorted(
                    int(key) for key in metrics["null_scale_by_temperature"]
                )
            ],
            dtype=float,
        ),
        metadata_json=np.asarray(json.dumps(metrics, sort_keys=True)),
    )
    injection = metrics["real_residual_injection"]
    projection = metrics["template_projection"]
    correlation = metrics["offdiagonal_correlation"]
    gain = metrics["gain_audit"]
    lines = [
        "# Signed Refit Candidate-Variance Validation",
        "",
        f"- Validation version: `{VALIDATION_VERSION}`",
        f"- Variance model: `{VARIANCE_MODEL_VERSION}`",
        f"- Signal-dependent fitter: `{SIGNAL_DEPENDENT_FITTER_VERSION}`",
        f"- Noise-model SHA-256: `{metrics['noise_model_sha256']}`",
        f"- Acceptance status: **{metrics['acceptance']['status']}**",
        "",
        "## Physical covariance",
        "",
        "For the paper model, the per-cycle transfer probability is "
        "`q=|A|[exp(-t/tau)-exp(-8t/tau)]`. The added pumping variance is "
        "`3000 q(1-q)`, because a transferred charge contributes directly to "
        "`I=(a-b)/2`. The null covariance is retained for read/background noise. "
        "An optional measured excess pair-charge term adds "
        "`max((a+b)_candidate-(a+b)_control,0)/4` to the diagonal.",
        "",
        "## Null covariance scale",
        "",
        "The first 64 held-out controls in every region calibrate one conservative "
        "temperature scale; the other 64 evaluate it. Scale factors are never "
        "estimated from candidates.",
        "",
        "| T (K) | Calibration width | Covariance scale | Evaluation width | p<0.05 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for temperature in sorted(
        int(key) for key in metrics["null_scale_by_temperature"]
    ):
        item = metrics["null_scale_by_temperature"][temperature]
        lines.append(
            f"| {temperature} | {item['calibration_width_raw']:.3f} | "
            f"{item['null_scale']:.4f} | {item['evaluation_width_scaled']:.3f} | "
            f"{item['evaluation_p05_rate']:.3%} |"
        )
    lines.extend(
        [
            "",
            "The scale corrects the mild variance deficit from the robust finite-sample "
            "covariance estimate. Heavy analytical tails remain visible and still "
            "require the empirical Step 6 detection calibration.",
            "",
            "## Real-residual injection",
            "",
            f"- Evaluation fits: {injection['record_count']}.",
            f"- Aggregate 68% interval coverage: "
            f"{injection['aggregate']['coverage']:.2%}.",
            f"- Characterization-eligible fits: {injection['eligible']['count']} "
            f"({injection['eligible']['rate']:.2%}); eligible coverage "
            f"{injection['eligible']['coverage']:.2%}.",
            f"- Aggregate median tau bias: "
            f"{injection['aggregate']['median_bias_dex']:+.4f} dex.",
            f"- Variance-iteration convergence: "
            f"{injection['aggregate']['variance_convergence_rate']:.2%}.",
            "",
            "| |A| true | Fits | Detection | Identifiable | All coverage | Eligible coverage | Eligible bias (dex) | Closure width |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for amplitude, item in injection["by_true_amplitude"].items():
        lines.append(
            f"| {amplitude} | {item['count']} | {item['detection_rate']:.2%} | "
            f"{item['identifiable_rate']:.2%} | {item['coverage']:.2%} | "
            f"{item['eligible_coverage']:.2%} | "
            f"{item['eligible_median_bias_dex']:+.4f} | "
            f"{item['closure_width']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Closure binned by fitted amplitude:",
            "",
            "| Fitted-amplitude quartile | Median |A_fit| | Fits | Coverage | Closure width |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for quartile, item in injection["by_fitted_amplitude_quartile"].items():
        lines.append(
            f"| {quartile} | {item['amplitude_median']:.4f} | {item['count']} | "
            f"{item['coverage']:.2%} | {item['closure_width']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Signals are injected onto untouched evaluation-half control curves. "
            "Each injection also draws the binomial transfer count and, where "
            "specified, independent lobe shot noise. This replaces the circular "
            "Gaussian-only Step 5 coverage claim.",
            "",
            "Coverage is accepted only for fits that pass the frozen Step 6 "
            "detection threshold, have a two-sided interval, and are neither "
            "boundary-limited nor multimodal. Weak curves remain in the diagnostic "
            "table but their tau values are classified as non-characterizable and "
            "must not enter Step 10.",
            "",
            "## Template regression guard",
            "",
            f"- Template pump projection delta-chi-square max/p95: "
            f"{projection['delta_chi2_max']:.3f}/"
            f"{projection['delta_chi2_p95']:.3f}.",
            f"- Conditional template |z| max/p95: {projection['abs_z_max']:.3f}/"
            f"{projection['abs_z_p95']:.3f}.",
            "",
            "## Off-diagonal correlation diagnosis",
            "",
            f"- Stored maximum absolute correlation: {correlation['stored_max']:.3f}.",
            "- Median absolute held-out correlation at each stored maximum pair: "
            f"{correlation['validation_at_stored_max_median_abs']:.3f}.",
            "- Median full off-diagonal train/evaluation matrix correlation: "
            f"{correlation['matrix_train_validation_correlation_median']:.3f}.",
            "- For stored |rho|>=0.8, same-sign held-out fraction: "
            f"{correlation['same_sign_fraction_for_stored_abs_ge_0p8']}; "
            "held-out |rho|>=0.5 fraction: "
            f"{correlation['validation_abs_ge_0p5_fraction_for_stored_abs_ge_0p8']}.",
            "",
            "A per-image scalar common mode is removed by the regional dwell template "
            "and cannot create covariance across control coordinates after centering. "
            "Split-coordinate reproducibility instead tests whether the large modes "
            "are persistent detector-coordinate/row-response structure.",
            "",
            "## Gain provenance",
            "",
            f"- Selected images: {gain['selected_image_count']}.",
            f"- Sidecars missing: CSV={gain['missing_csv_count']}, "
            f"XML={gain['missing_xml_count']}.",
            f"- Empty calibration CSVs: {gain['empty_csv_count']}.",
            f"- Populated per-image gain fits: {gain['fitted_gain_count']}.",
            f"- XML default gains: {gain['xml_default_unique']}.",
            f"- Pipeline global electronization scale: "
            f"{gain['pipeline_global_scale']:.1f}.",
            f"- Status: **{gain['status']}**.",
            "",
            "The sidecars exist but do not contain fitted per-image gains for this "
            "manifest; their XML value is a fallback, not an image-by-image "
            "measurement. The established MINOS global scale is therefore retained "
            "pending external calibration provenance.",
            "",
            "## Acceptance gate",
            "",
        ]
    )
    if metrics["acceptance"]["status"] == "PASS":
        lines.extend(
            [
                "- PASS: independently scaled held-out null widths close.",
                "- PASS: real-residual injection tau bias and coverage close across "
                "the characterization-eligible amplitude range.",
                "- PASS: fitted-amplitude residual widths do not develop a signal trend.",
                "- PASS: the null-template pump projection remains negligible.",
            ]
        )
    else:
        lines.append("- **FAIL:** " + "; ".join(metrics["acceptance"]["failures"]))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_output(output_path: Path, model_path: Path) -> dict[str, object]:
    data = np.load(output_path, allow_pickle=False)
    metrics = json.loads(str(data["metadata_json"]))
    errors = []
    if metrics.get("validation_version") != VALIDATION_VERSION:
        errors.append("validation version mismatch")
    if metrics.get("noise_model_sha256") != file_sha256(model_path):
        errors.append("noise-model hash mismatch")
    if metrics.get("acceptance", {}).get("status") != "PASS":
        errors.append(
            "acceptance failed: "
            + "; ".join(metrics.get("acceptance", {}).get("failures", []))
        )
    if errors:
        raise ValueError("Variance validation failed:\n- " + "\n- ".join(errors))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detection", type=Path, default=DEFAULT_DETECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        metrics = validate_output(args.output, args.model)
    else:
        metrics = run_validation(args.model, args.manifest, args.detection)
        write_outputs(metrics, args.output, args.report)
        if metrics["acceptance"]["status"] != "PASS":
            raise RuntimeError(
                "Variance validation failed: "
                + "; ".join(metrics["acceptance"]["failures"])
            )
    print(
        f"{metrics['acceptance']['status']}: coverage "
        f"{metrics['real_residual_injection']['eligible']['coverage']:.3%}; "
        f"template max delta {metrics['template_projection']['delta_chi2_max']:.3f}"
    )


if __name__ == "__main__":
    main()

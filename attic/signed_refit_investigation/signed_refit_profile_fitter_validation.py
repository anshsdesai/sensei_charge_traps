"""Monte Carlo validation of the signed covariance-aware profile-tau fitter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from dipole_new import intensity_function_offset
from signed_refit_profile_fitter import (
    PROFILE_FITTER_VERSION,
    ProfileTauFitter,
    file_sha256,
    intensity_model,
    load_scan_calibration,
)


VALIDATION_VERSION = "signed-refit-profile-validation-v1"
DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_OUTPUT = Path("signed_refit_profile_fitter_validation.npz")
DEFAULT_REPORT = Path("signed_refit_profile_fitter_validation.md")
RANDOM_SEED = 2026061305
N_REALIZATIONS = 300
N_LEGACY_COMPARISONS = 30

MAX_ABS_MEDIAN_BIAS_DEX = 0.03
SCENARIO_COVERAGE_RANGE = (0.58, 0.78)
AGGREGATE_COVERAGE_RANGE = (0.64, 0.72)
MIN_TWO_SIDED_INTERVAL_RATE = 0.90
MIN_SIGN_RECOVERY_RATE = 0.98
MAX_GRID_STABILITY_DEX = 2e-4


@dataclass(frozen=True)
class Scenario:
    name: str
    temperature: int
    quadrant: int
    region: int
    tau: float
    amplitude: float
    offset: float


SCENARIOS = (
    Scenario("short-positive", 125, 0, 0, 3e-4, 0.18, -50.0),
    Scenario("short-negative", 145, 1, 10, 3e-3, -0.15, 100.0),
    Scenario("mid-positive", 183, 2, 20, 3e-2, 0.14, -250.0),
    Scenario("long-negative", 190, 3, 7, 3e-1, -0.16, -500.0),
    Scenario("warm-positive", 203, 0, 24, 1e-3, 0.12, -700.0),
    Scenario("warm-negative", 210, 3, 15, 1e-1, -0.14, -900.0),
)


def _legacy_curve_fit(
    seconds: np.ndarray,
    corrected: np.ndarray,
    sigma: np.ndarray,
    tau_bounds: tuple[float, float],
    initial_tau: float,
) -> float | None:
    offset_guess = float(np.median(corrected))
    deviations = corrected - offset_guess
    peak_index = int(np.argmax(np.abs(deviations)))
    amplitude_guess = float(deviations[peak_index] / (3000.0 * 0.650))
    try:
        fitted, _ = curve_fit(
            intensity_function_offset,
            seconds,
            corrected,
            sigma=sigma,
            absolute_sigma=True,
            p0=[amplitude_guess, initial_tau, offset_guess],
            bounds=(
                [-np.inf, tau_bounds[0], -np.inf],
                [np.inf, tau_bounds[1], np.inf],
            ),
            maxfev=20000,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    return float(fitted[1])


def run_validation(model_path: Path) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    scenario_results = []
    all_covered = []
    grid_differences = []
    legacy_full_differences = []
    legacy_diagonal_differences = []
    legacy_initial_spreads = []
    legacy_failure_count = 0
    legacy_fit_count = 0

    for scenario in SCENARIOS:
        calibration = load_scan_calibration(
            model_path,
            scenario.temperature,
            scenario.quadrant,
            region=scenario.region,
        )
        fitter = ProfileTauFitter.from_calibration(calibration)
        coarse_fitter = ProfileTauFitter.from_calibration(
            calibration, grid_size=401
        )
        fine_fitter = ProfileTauFitter.from_calibration(
            calibration, grid_size=1201
        )
        truth = intensity_model(
            calibration.seconds,
            scenario.amplitude,
            scenario.tau,
            scenario.offset,
            calibration.null_template,
        )
        biases = []
        covered = []
        two_sided = []
        signs = []
        boundary_flags = []
        multimodal_flags = []
        reduced_chi2 = []

        for realization in range(N_REALIZATIONS):
            observed = truth + rng.multivariate_normal(
                np.zeros(calibration.seconds.size),
                calibration.covariance,
            )
            result = fitter.fit(observed)
            biases.append(np.log10(result["tau"] / scenario.tau))
            lower = (
                fitter.tau_bounds[0]
                if result["tau_interval_lower"] is None
                else result["tau_interval_lower"]
            )
            upper = (
                fitter.tau_bounds[1]
                if result["tau_interval_upper"] is None
                else result["tau_interval_upper"]
            )
            covered.append(lower <= scenario.tau <= upper)
            two_sided.append(
                result["tau_interval_lower"] is not None
                and result["tau_interval_upper"] is not None
            )
            signs.append(result["amplitude_sign"] == int(np.sign(scenario.amplitude)))
            boundary_flags.append(result["boundary_limited"])
            multimodal_flags.append(result["multimodal"])
            reduced_chi2.append(result["reduced_chi2"])

            if realization < 20:
                coarse = coarse_fitter.fit(observed)
                fine = fine_fitter.fit(observed)
                grid_differences.append(
                    abs(np.log10(coarse["tau"] / fine["tau"]))
                )

            if realization < N_LEGACY_COMPARISONS:
                corrected = observed - calibration.null_template
                tau_guesses = (
                    fitter.tau_bounds[0] * 2.0,
                    np.sqrt(fitter.tau_bounds[0] * fitter.tau_bounds[1]),
                    fitter.tau_bounds[1] / 2.0,
                )
                full_fits = [
                    _legacy_curve_fit(
                        calibration.seconds,
                        corrected,
                        calibration.covariance,
                        fitter.tau_bounds,
                        guess,
                    )
                    for guess in tau_guesses
                ]
                diagonal_fits = [
                    _legacy_curve_fit(
                        calibration.seconds,
                        corrected,
                        np.sqrt(np.diag(calibration.covariance)),
                        fitter.tau_bounds,
                        guess,
                    )
                    for guess in tau_guesses
                ]
                legacy_fit_count += len(full_fits) + len(diagonal_fits)
                legacy_failure_count += sum(item is None for item in full_fits)
                legacy_failure_count += sum(item is None for item in diagonal_fits)
                finite_full = np.asarray(
                    [item for item in full_fits if item is not None]
                )
                finite_diagonal = np.asarray(
                    [item for item in diagonal_fits if item is not None]
                )
                if finite_full.size:
                    best_full = finite_full[
                        np.argmin(np.abs(np.log(finite_full / result["tau"])))
                    ]
                    legacy_full_differences.append(
                        abs(np.log10(best_full / result["tau"]))
                    )
                    legacy_initial_spreads.append(
                        np.ptp(np.log10(finite_full))
                    )
                if finite_diagonal.size:
                    best_diagonal = finite_diagonal[
                        np.argmin(np.abs(np.log(finite_diagonal / result["tau"])))
                    ]
                    legacy_diagonal_differences.append(
                        np.log10(best_diagonal / result["tau"])
                    )

        biases = np.asarray(biases)
        covered = np.asarray(covered)
        all_covered.extend(covered.tolist())
        scenario_results.append(
            {
                "name": scenario.name,
                "temperature": scenario.temperature,
                "quadrant": scenario.quadrant,
                "region": scenario.region,
                "tau_true": scenario.tau,
                "amplitude_true": scenario.amplitude,
                "offset_true": scenario.offset,
                "n": N_REALIZATIONS,
                "median_bias_dex": float(np.median(biases)),
                "bias_p16_dex": float(np.percentile(biases, 16)),
                "bias_p84_dex": float(np.percentile(biases, 84)),
                "coverage": float(np.mean(covered)),
                "two_sided_interval_rate": float(np.mean(two_sided)),
                "sign_recovery_rate": float(np.mean(signs)),
                "boundary_rate": float(np.mean(boundary_flags)),
                "multimodal_rate": float(np.mean(multimodal_flags)),
                "median_reduced_chi2": float(np.median(reduced_chi2)),
            }
        )

    boundary_calibration = load_scan_calibration(model_path, 183, 0, region=0)
    boundary_fitter = ProfileTauFitter.from_calibration(boundary_calibration)
    boundary_truth = intensity_model(
        boundary_calibration.seconds,
        0.20,
        boundary_fitter.tau_bounds[1] * 100.0,
        -200.0,
        boundary_calibration.null_template,
    )
    boundary_result = boundary_fitter.fit(boundary_truth)

    multimodal_seconds = np.geomspace(5e-5, 0.2, 25)
    multimodal_covariance = np.diag(np.full(25, 35.0**2))
    multimodal_fitter = ProfileTauFitter(
        multimodal_seconds,
        multimodal_covariance,
        tau_bounds=(5e-6, 2.0),
    )
    multimodal_observed = np.random.default_rng(5).normal(0.0, 35.0, 25)
    multimodal_result = multimodal_fitter.fit(multimodal_observed)

    failures = []
    for result in scenario_results:
        if abs(result["median_bias_dex"]) > MAX_ABS_MEDIAN_BIAS_DEX:
            failures.append(
                f"{result['name']} median bias {result['median_bias_dex']:.3f} dex"
            )
        if not SCENARIO_COVERAGE_RANGE[0] <= result["coverage"] <= SCENARIO_COVERAGE_RANGE[1]:
            failures.append(
                f"{result['name']} coverage {result['coverage']:.3f}"
            )
        if result["two_sided_interval_rate"] < MIN_TWO_SIDED_INTERVAL_RATE:
            failures.append(
                f"{result['name']} two-sided interval rate "
                f"{result['two_sided_interval_rate']:.3f}"
            )
        if result["sign_recovery_rate"] < MIN_SIGN_RECOVERY_RATE:
            failures.append(
                f"{result['name']} sign recovery {result['sign_recovery_rate']:.3f}"
            )

    aggregate_coverage = float(np.mean(all_covered))
    if not AGGREGATE_COVERAGE_RANGE[0] <= aggregate_coverage <= AGGREGATE_COVERAGE_RANGE[1]:
        failures.append(f"aggregate coverage {aggregate_coverage:.3f}")
    grid_max = float(np.max(grid_differences))
    if grid_max > MAX_GRID_STABILITY_DEX:
        failures.append(f"grid-refinement difference {grid_max:.3g} dex")
    if not boundary_result["boundary_limited"]:
        failures.append("out-of-window tau was not boundary flagged")
    if not multimodal_result["multimodal"]:
        failures.append("low-signal competing minima were not flagged")

    return {
        "validation_version": VALIDATION_VERSION,
        "profile_fitter_version": PROFILE_FITTER_VERSION,
        "noise_model_sha256": file_sha256(model_path),
        "random_seed": RANDOM_SEED,
        "n_realizations_per_scenario": N_REALIZATIONS,
        "scenario_results": scenario_results,
        "aggregate": {
            "coverage": aggregate_coverage,
            "max_abs_median_bias_dex": float(
                max(abs(item["median_bias_dex"]) for item in scenario_results)
            ),
            "minimum_sign_recovery_rate": float(
                min(item["sign_recovery_rate"] for item in scenario_results)
            ),
            "minimum_two_sided_interval_rate": float(
                min(item["two_sided_interval_rate"] for item in scenario_results)
            ),
            "grid_stability_max_dex": grid_max,
        },
        "boundary_test": {
            "tau_true": boundary_fitter.tau_bounds[1] * 100.0,
            "tau_fit": boundary_result["tau"],
            "boundary_limited": boundary_result["boundary_limited"],
            "at_upper_boundary": boundary_result["at_upper_boundary"],
            "upper_interval_limited": boundary_result["interval_upper_limited"],
        },
        "multimodal_test": {
            "multimodal": multimodal_result["multimodal"],
            "competitive_mode_count": len(multimodal_result["competitive_modes"]),
            "boundary_limited": multimodal_result["boundary_limited"],
            "delta_chi2_vs_constant": multimodal_result["delta_chi2_vs_constant"],
        },
        "legacy_comparison": {
            "fit_attempts": legacy_fit_count,
            "failure_count": legacy_failure_count,
            "failure_rate": legacy_failure_count / legacy_fit_count,
            "full_covariance_best_start_median_abs_difference_dex": float(
                np.median(legacy_full_differences)
            ),
            "full_covariance_initial_start_spread_p95_dex": float(
                np.percentile(legacy_initial_spreads, 95)
            ),
            "diagonal_minus_profile_median_dex": float(
                np.median(legacy_diagonal_differences)
            ),
            "diagonal_minus_profile_p95_abs_dex": float(
                np.percentile(np.abs(legacy_diagonal_differences), 95)
            ),
        },
        "acceptance_thresholds": {
            "max_abs_median_bias_dex": MAX_ABS_MEDIAN_BIAS_DEX,
            "scenario_coverage_range": SCENARIO_COVERAGE_RANGE,
            "aggregate_coverage_range": AGGREGATE_COVERAGE_RANGE,
            "min_two_sided_interval_rate": MIN_TWO_SIDED_INTERVAL_RATE,
            "min_sign_recovery_rate": MIN_SIGN_RECOVERY_RATE,
            "max_grid_stability_dex": MAX_GRID_STABILITY_DEX,
        },
        "acceptance": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
        },
    }


def write_outputs(
    metrics: dict[str, object],
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(metrics, sort_keys=True)),
    )
    aggregate = metrics["aggregate"]
    legacy = metrics["legacy_comparison"]
    lines = [
        "# Signed Refit Profile-Tau Fitter Validation",
        "",
        f"- Validation version: `{VALIDATION_VERSION}`",
        f"- Profile fitter version: `{PROFILE_FITTER_VERSION}`",
        f"- Noise-model SHA-256: `{metrics['noise_model_sha256']}`",
        f"- Random seed: `{RANDOM_SEED}`",
        f"- Realizations per scenario: {N_REALIZATIONS}",
        f"- Acceptance status: **{metrics['acceptance']['status']}**",
        "",
        "## Model-conditional synthetic bias and coverage",
        "",
        "| Scenario | T (K) | Q/R | tau true (s) | A true | Median bias (dex) | 68% coverage | Two-sided | Sign | Boundary | Multimodal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["scenario_results"]:
        lines.append(
            f"| {item['name']} | {item['temperature']} | "
            f"{item['quadrant']}/{item['region']} | {item['tau_true']:.3g} | "
            f"{item['amplitude_true']:+.3f} | {item['median_bias_dex']:+.4f} | "
            f"{item['coverage']:.1%} | {item['two_sided_interval_rate']:.1%} | "
            f"{item['sign_recovery_rate']:.1%} | {item['boundary_rate']:.1%} | "
            f"{item['multimodal_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "- Aggregate 68% profile coverage under the assumed Gaussian covariance: "
            f"{aggregate['coverage']:.2%}.",
            "- Maximum absolute scenario median bias: "
            f"{aggregate['max_abs_median_bias_dex']:.4f} dex.",
            "- Minimum sign recovery: "
            f"{aggregate['minimum_sign_recovery_rate']:.2%}.",
            "- Minimum two-sided interval rate: "
            f"{aggregate['minimum_two_sided_interval_rate']:.2%}.",
            "- Maximum 401-versus-1201-grid fitted-tau difference after continuous "
            f"refinement: {aggregate['grid_stability_max_dex']:.2e} dex.",
            "",
            "This is an algebra/model-conditional test: draws come from the same "
            "Gaussian covariance used by the fitter. Empirical coverage against the "
            "observed heavy-tailed PDF is tested separately in "
            "`signed_refit_variance_validation.md` using real held-out residuals.",
            "",
            "The tested tau values span 0.0003-0.3 s, both amplitude signs, warm and "
            "cold scans, 18- and 25-point dwell grids, nonzero pedestals, and exact "
            "regional covariance/null templates from the frozen v2 noise model.",
            "",
            "## Boundary and multimodal behavior",
            "",
            "- Out-of-window long-tau injection: "
            f"fit tau={metrics['boundary_test']['tau_fit']:.3g} s; "
            f"boundary_limited={metrics['boundary_test']['boundary_limited']}; "
            f"upper_interval_limited={metrics['boundary_test']['upper_interval_limited']}.",
            "- Low-signal noise realization: "
            f"multimodal={metrics['multimodal_test']['multimodal']}; "
            f"competitive modes={metrics['multimodal_test']['competitive_mode_count']}; "
            f"delta chi-square={metrics['multimodal_test']['delta_chi2_vs_constant']:.3f}.",
            "",
            "These cases retain profile limits and flags instead of receiving a "
            "symmetric Gaussian tau error.",
            "",
            "## Nonlinear curve-fit comparison",
            "",
            f"- Nonlinear attempts: {legacy['fit_attempts']}; failure rate "
            f"{legacy['failure_rate']:.2%}.",
            "- Best-start full-covariance nonlinear versus profile median absolute "
            f"difference: {legacy['full_covariance_best_start_median_abs_difference_dex']:.2e} dex.",
            "- Full-covariance nonlinear initial-start spread p95: "
            f"{legacy['full_covariance_initial_start_spread_p95_dex']:.3f} dex.",
            "- Current diagonal-error nonlinear minus covariance-profile median: "
            f"{legacy['diagonal_minus_profile_median_dex']:+.4f} dex; p95 absolute "
            f"difference {legacy['diagonal_minus_profile_p95_abs_dex']:.3f} dex.",
            "",
            "The profile fitter has no initial tau guess: it evaluates the complete "
            "log-tau grid and refines the global minimum. The full-covariance "
            "nonlinear comparison checks numerical agreement when its local "
            "optimizer reaches the same minimum; the diagonal comparison illustrates "
            "the effect of the legacy error treatment.",
            "",
            "## Acceptance gate",
            "",
        ]
    )
    if metrics["acceptance"]["status"] == "PASS":
        lines.extend(
            [
                "- PASS: synthetic tau bias is below the predefined 0.03 dex limit.",
                "- PASS: profile intervals have nominal coverage when the assumed "
                "Gaussian covariance is the data-generating model.",
                "- PASS: the solution is independent of nonlinear initial guesses and "
                "stable under profile-grid refinement.",
                "- PASS: boundary-limited and competing-minimum cases are explicitly flagged.",
            ]
        )
    else:
        lines.append("- **FAIL:** " + "; ".join(metrics["acceptance"]["failures"]))
    lines.extend(
        [
            "",
            "Step 5 does not assign a dipole-detection threshold. The empirical "
            "significance calibration remains Step 6.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def validate_output(output_path: Path, model_path: Path) -> dict[str, object]:
    data = np.load(output_path, allow_pickle=False)
    metrics = json.loads(str(data["metadata_json"]))
    errors = []
    if metrics.get("validation_version") != VALIDATION_VERSION:
        errors.append("validation version mismatch")
    if metrics.get("profile_fitter_version") != PROFILE_FITTER_VERSION:
        errors.append("profile fitter version mismatch")
    if metrics.get("noise_model_sha256") != file_sha256(model_path):
        errors.append("noise-model hash mismatch")
    if metrics.get("acceptance", {}).get("status") != "PASS":
        errors.append(
            "acceptance failed: "
            + "; ".join(metrics.get("acceptance", {}).get("failures", []))
        )
    if errors:
        raise ValueError("Profile-fitter validation failed:\n- " + "\n- ".join(errors))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        metrics = validate_output(args.output, args.model)
    else:
        metrics = run_validation(args.model)
        write_outputs(metrics, args.output, args.report)
        if metrics["acceptance"]["status"] != "PASS":
            raise RuntimeError(
                "Step 5 acceptance failed: "
                + "; ".join(metrics["acceptance"]["failures"])
            )
    print(
        f"PASS: aggregate coverage {metrics['aggregate']['coverage']:.3%}; "
        f"max median bias {metrics['aggregate']['max_abs_median_bias_dex']:.4f} dex"
    )


if __name__ == "__main__":
    main()

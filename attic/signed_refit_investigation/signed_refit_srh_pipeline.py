"""Fit the simple p-channel SRH model to definitive Step 9 tau profiles."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2, spearmanr

from signed_refit_profile_fitter import file_sha256


PIPELINE_VERSION = "signed-refit-srh-v1"
DEFAULT_INTENSITY = Path("signed_refit_intensity_fits_v1.h5")
DEFAULT_LEGACY = Path("fit_dipole_spectra_err_4.h5")
DEFAULT_OUTPUT = Path("signed_refit_srh_fits_v1.h5")
DEFAULT_REPORT = Path("signed_refit_srh_validation.md")
DEFAULT_FIGURE_DIR = Path("figures/signed_refit_srh")

MIN_TEMPERATURES = 4
FAMILY_TEMPERATURES = (160, 170)
SRH_PVALUE_THRESHOLD = 0.05
ENERGY_BOUNDS = (0.0, 2.0)
LOG_SIGMA_BOUNDS = (-100.0, -1.0)
PROFILE_INTERVAL_DELTA = 1.0
BOOTSTRAP_SEED = 2026061410
BOOTSTRAP_DRAWS = 5000
FAMILY_RESIDUAL_THRESHOLD_DEX = 0.10
HIGH_TEMPERATURE_MIN = 175

M_COND_HOLE = 0.41
M_DENS_HOLE = 0.94
KB_EV_K = 8.617333262e-5
PLANCK_EV_S = 4.135667696e-15
ELECTRON_MASS_EV = 0.510998950e6
LIGHT_SPEED_CM_S = 2.99792458e10

STATUS_NOT_STEP9_SINGLE = 0
STATUS_INSUFFICIENT_POINTS = 1
STATUS_OPTIMIZER_FAILED = 2
STATUS_PREDICTION_OUTSIDE_PROFILE = 3
STATUS_PARAMETER_BOUNDARY = 4
STATUS_NON_SRH = 5
STATUS_SRH_CONSISTENT = 6
STATUS_NAMES = (
    "not_step9_single",
    "insufficient_points",
    "optimizer_failed",
    "prediction_outside_profile",
    "parameter_boundary",
    "fit_success_non_srh",
    "srh_consistent",
)


def srh_log_tau(
    temperatures: np.ndarray,
    energy_ev: float,
    log_sigma_cm2: float,
) -> np.ndarray:
    """Natural log of the p-channel SRH hole-emission lifetime in seconds."""
    temperatures = np.asarray(temperatures, dtype=float)
    denominator = (
        2
        * np.sqrt(3.0)
        * (2.0 * np.pi) ** 1.5
        * (M_DENS_HOLE * ELECTRON_MASS_EV) ** 1.5
        / np.sqrt(M_COND_HOLE * ELECTRON_MASS_EV)
    )
    scaling = PLANCK_EV_S**3 * LIGHT_SPEED_CM_S**2 / denominator
    kb_t = KB_EV_K * temperatures
    return (
        np.log(scaling)
        - float(log_sigma_cm2)
        - 2.0 * np.log(kb_t)
        + float(energy_ev) / kb_t
    )


def weighted_linear_initial(
    temperatures: np.ndarray,
    log_tau: np.ndarray,
    sigma_log_tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = srh_log_tau(temperatures, 0.0, 0.0)
    design = np.column_stack(
        (-np.ones(temperatures.size), 1.0 / (KB_EV_K * temperatures))
    )
    weights = 1.0 / np.maximum(sigma_log_tau, 1e-6) ** 2
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    beta = covariance @ (design.T @ (weights * (log_tau - base)))
    log_sigma = float(
        np.clip(beta[0], LOG_SIGMA_BOUNDS[0], LOG_SIGMA_BOUNDS[1])
    )
    energy = float(np.clip(beta[1], ENERGY_BOUNDS[0], ENERGY_BOUNDS[1]))
    return np.asarray([energy, log_sigma]), covariance


def local_log_tau_sigma(
    tau: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_tau = np.log(tau)
    lower_sigma = log_tau - np.log(lower)
    upper_sigma = np.log(upper) - log_tau
    mean_sigma = 0.5 * (lower_sigma + upper_sigma)
    return lower_sigma, upper_sigma, mean_sigma


def interpolate_profile(
    log_tau_grid: np.ndarray,
    delta_chi2: np.ndarray,
    predicted_log_tau: float,
) -> tuple[float, bool]:
    lower = float(log_tau_grid[0])
    upper = float(log_tau_grid[-1])
    if predicted_log_tau < lower or predicted_log_tau > upper:
        return 1e6 + 1e4 * min(
            (predicted_log_tau - lower) ** 2,
            (predicted_log_tau - upper) ** 2,
        ), True
    return float(
        np.interp(predicted_log_tau, log_tau_grid, delta_chi2)
    ), False


def profile_objective(
    parameters: np.ndarray,
    temperatures: np.ndarray,
    log_tau_grids: list[np.ndarray],
    profiles: list[np.ndarray],
) -> float:
    energy, log_sigma = map(float, parameters)
    if not ENERGY_BOUNDS[0] <= energy <= ENERGY_BOUNDS[1]:
        return 1e12
    if not LOG_SIGMA_BOUNDS[0] <= log_sigma <= LOG_SIGMA_BOUNDS[1]:
        return 1e12
    predicted = srh_log_tau(temperatures, energy, log_sigma)
    total = 0.0
    for value, grid, profile in zip(predicted, log_tau_grids, profiles):
        contribution, _ = interpolate_profile(grid, profile, float(value))
        total += contribution
    return float(total)


def profile_parameter_interval(
    best_parameters: np.ndarray,
    best_value: float,
    parameter_index: int,
    bounds: tuple[float, float],
    temperatures: np.ndarray,
    log_tau_grids: list[np.ndarray],
    profiles: list[np.ndarray],
    initial_step: float,
) -> tuple[float, float]:
    nuisance_index = 1 - parameter_index
    nuisance_bounds = (
        LOG_SIGMA_BOUNDS if nuisance_index == 1 else ENERGY_BOUNDS
    )
    cache: dict[float, float] = {}
    best_nuisance = float(best_parameters[nuisance_index])

    def profiled(fixed_value: float) -> float:
        key = float(fixed_value)
        if key in cache:
            return cache[key]

        def objective(nuisance: float) -> float:
            parameters = best_parameters.copy()
            parameters[parameter_index] = fixed_value
            parameters[nuisance_index] = nuisance
            return profile_objective(
                parameters,
                temperatures,
                log_tau_grids,
                profiles,
            )

        result = minimize(
            lambda value: objective(float(value[0])),
            np.asarray([best_nuisance]),
            method="Powell",
            bounds=(nuisance_bounds,),
            options={"xtol": 1e-7, "ftol": 1e-9, "maxiter": 200},
        )
        # Retain the fitted nuisance value as a candidate. This prevents a
        # non-parabolic profile from jumping to a worse local branch.
        cache[key] = min(
            float(result.fun),
            objective(best_nuisance),
        )
        return cache[key]

    target = best_value + PROFILE_INTERVAL_DELTA
    center = float(best_parameters[parameter_index])
    outputs = []
    for direction in (-1.0, 1.0):
        inner = center
        inner_value = best_value - target
        step = initial_step
        crossing = np.nan
        for _ in range(30):
            outer = float(
                np.clip(center + direction * step, bounds[0], bounds[1])
            )
            if outer == inner:
                break
            outer_value = profiled(outer) - target
            if outer_value >= 0 and inner_value <= 0:
                lo, hi = sorted((inner, outer))
                lo_value = profiled(lo) - target
                hi_value = profiled(hi) - target
                if lo_value == 0.0:
                    crossing = lo
                elif hi_value == 0.0:
                    crossing = hi
                elif lo_value * hi_value < 0.0:
                    crossing = float(
                        brentq(
                            lambda value: profiled(value) - target,
                            lo,
                            hi,
                            xtol=1e-7,
                        )
                    )
                if np.isfinite(crossing):
                    break
            inner = outer
            inner_value = outer_value
            if outer in bounds:
                crossing = outer
                break
            step *= 1.8
        outputs.append(crossing)
    return float(outputs[0]), float(outputs[1])


def weighted_leverage(
    temperatures: np.ndarray,
    sigma_log_tau: np.ndarray,
) -> np.ndarray:
    design = np.column_stack(
        (-np.ones(temperatures.size), 1.0 / (KB_EV_K * temperatures))
    )
    weights = 1.0 / np.maximum(sigma_log_tau, 1e-6) ** 2
    covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
    return weights * np.einsum("ij,jk,ik->i", design, covariance, design)


def load_profiles(
    intensity_path: Path,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    with h5py.File(intensity_path, "r") as handle:
        site = {
            name: np.asarray(handle[name])
            for name in (
                "temperatures",
                "candidate_quadrant",
                "candidate_row",
                "candidate_col",
                "accepted",
                "tau",
                "tau_interval_lower",
                "tau_interval_upper",
                "amplitude",
                "offset",
                "fit_pvalue",
                "single_trap_eligible",
                "final_orientation_label",
            )
        }
        profiles = {}
        candidate_count = site["candidate_row"].size
        for temperature_value in site["temperatures"]:
            temperature = int(temperature_value)
            group = handle[f"profiles/temp_{temperature}"]
            candidate_index = np.asarray(group["candidate_index"], dtype=int)
            inverse = np.full(candidate_count, -1, dtype=np.int32)
            inverse[candidate_index] = np.arange(
                candidate_index.size, dtype=np.int32
            )
            profiles[temperature] = {
                "inverse": inverse,
                "log_tau_grid": np.log(
                    np.asarray(group["tau_grid"], dtype=float)
                ),
                "delta_chi2": np.asarray(
                    group["delta_chi2"], dtype=np.float32
                ),
            }
    return site, profiles


def legacy_membership(
    legacy_path: Path,
    quadrant: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_count = rows.size
    key_to_index = {
        (int(q), int(row), int(col)): index
        for index, (q, row, col) in enumerate(zip(quadrant, rows, cols))
    }
    output = {
        "legacy_coordinate": np.zeros(candidate_count, dtype=np.int8),
        "legacy_well_behaved": np.zeros(candidate_count, dtype=np.int8),
        "legacy_good_energy": np.zeros(candidate_count, dtype=np.int8),
        "legacy_energy": np.full(candidate_count, np.nan, dtype=np.float32),
        "legacy_log_sigma": np.full(candidate_count, np.nan, dtype=np.float32),
    }
    with h5py.File(legacy_path, "r") as handle:
        for quadrant_name, quadrant_group in handle.items():
            q = int(quadrant_name.split("_")[1])
            for dipole_name, dipole_group in quadrant_group.items():
                _, row_text, col_text = dipole_name.split("_")
                index = key_to_index.get((q, int(row_text), int(col_text)))
                if index is None:
                    continue
                output["legacy_coordinate"][index] = 1
                attrs = dipole_group.attrs
                if bool(attrs.get("WellBehavedTrap", False)):
                    output["legacy_well_behaved"][index] = 1
                if bool(attrs.get("GoodEnergyFit", False)):
                    output["legacy_good_energy"][index] = 1
                    output["legacy_energy"][index] = float(
                        attrs["energy_BestFitEnergy"]
                    )
                    output["legacy_log_sigma"][index] = float(
                        np.log(attrs["energy_BestFitCrossSection"])
                    )
    return output


def empty_variant_arrays(
    candidate_count: int,
    temperature_count: int,
) -> dict[str, np.ndarray]:
    return {
        "status": np.full(
            candidate_count, STATUS_NOT_STEP9_SINGLE, dtype=np.int8
        ),
        "point_count": np.zeros(candidate_count, dtype=np.int8),
        "energy": np.full(candidate_count, np.nan, dtype=np.float32),
        "energy_interval_lower": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "energy_interval_upper": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "log_sigma": np.full(candidate_count, np.nan, dtype=np.float32),
        "log_sigma_interval_lower": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "log_sigma_interval_upper": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "deviance": np.full(candidate_count, np.nan, dtype=np.float32),
        "dof": np.full(candidate_count, -1, dtype=np.int16),
        "pvalue": np.full(candidate_count, np.nan, dtype=np.float32),
        "max_leverage": np.full(candidate_count, np.nan, dtype=np.float32),
        "max_leverage_temperature": np.full(
            candidate_count, -1, dtype=np.int16
        ),
        "fit_temperature_min": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "fit_temperature_max": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "prediction_outside_profile": np.zeros(
            candidate_count, dtype=np.int8
        ),
        "parameter_boundary": np.zeros(candidate_count, dtype=np.int8),
        "used_temperature": np.zeros(
            (candidate_count, temperature_count), dtype=np.int8
        ),
        "predicted_log_tau": np.full(
            (candidate_count, temperature_count), np.nan, dtype=np.float32
        ),
        "residual_dex": np.full(
            (candidate_count, temperature_count), np.nan, dtype=np.float32
        ),
        "signed_profile_z": np.full(
            (candidate_count, temperature_count), np.nan, dtype=np.float32
        ),
        "leverage": np.full(
            (candidate_count, temperature_count), np.nan, dtype=np.float32
        ),
        "peak_in_window": np.zeros(
            (candidate_count, temperature_count), dtype=np.int8
        ),
    }


def fit_variant(
    site: dict[str, np.ndarray],
    profiles_by_temperature: dict[int, dict[str, np.ndarray]],
    excluded_temperatures: set[int],
) -> dict[str, np.ndarray]:
    temperatures_all = np.asarray(site["temperatures"], dtype=int)
    candidate_count = site["candidate_row"].size
    output = empty_variant_arrays(candidate_count, temperatures_all.size)
    single = np.asarray(site["single_trap_eligible"], dtype=bool)
    accepted = np.asarray(site["accepted"], dtype=bool)

    for candidate_index in np.flatnonzero(single):
        selected_indices = np.flatnonzero(
            accepted[candidate_index]
            & ~np.isin(temperatures_all, list(excluded_temperatures))
        )
        output["point_count"][candidate_index] = selected_indices.size
        if selected_indices.size < MIN_TEMPERATURES:
            output["status"][candidate_index] = STATUS_INSUFFICIENT_POINTS
            continue

        temperatures = temperatures_all[selected_indices].astype(float)
        tau = np.asarray(site["tau"][candidate_index, selected_indices], float)
        lower = np.asarray(
            site["tau_interval_lower"][candidate_index, selected_indices],
            float,
        )
        upper = np.asarray(
            site["tau_interval_upper"][candidate_index, selected_indices],
            float,
        )
        lower_sigma, upper_sigma, mean_sigma = local_log_tau_sigma(
            tau, lower, upper
        )
        initial, linear_covariance = weighted_linear_initial(
            temperatures, np.log(tau), mean_sigma
        )
        log_tau_grids = []
        profile_rows = []
        for temperature_index in selected_indices:
            temperature = int(temperatures_all[temperature_index])
            values = profiles_by_temperature[temperature]
            row = int(values["inverse"][candidate_index])
            if row < 0:
                raise ValueError(
                    f"Accepted candidate {candidate_index} lacks "
                    f"{temperature} K profile"
                )
            log_tau_grids.append(values["log_tau_grid"])
            profile_rows.append(values["delta_chi2"][row])

        result = minimize(
            profile_objective,
            initial,
            args=(temperatures, log_tau_grids, profile_rows),
            method="Powell",
            bounds=(ENERGY_BOUNDS, LOG_SIGMA_BOUNDS),
            options={"xtol": 1e-7, "ftol": 1e-7, "maxiter": 500},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            output["status"][candidate_index] = STATUS_OPTIMIZER_FAILED
            continue
        best = np.asarray(result.x, dtype=float)
        best_value = float(result.fun)
        predicted = srh_log_tau(temperatures, best[0], best[1])
        outside = np.asarray(
            [
                value < grid[0] or value > grid[-1]
                for value, grid in zip(predicted, log_tau_grids)
            ],
            dtype=bool,
        )
        parameter_boundary = bool(
            np.isclose(best[0], ENERGY_BOUNDS[0], atol=1e-5)
            or np.isclose(best[0], ENERGY_BOUNDS[1], atol=1e-5)
            or np.isclose(best[1], LOG_SIGMA_BOUNDS[0], atol=1e-4)
            or np.isclose(best[1], LOG_SIGMA_BOUNDS[1], atol=1e-4)
        )
        output["prediction_outside_profile"][candidate_index] = int(
            np.any(outside)
        )
        output["parameter_boundary"][candidate_index] = int(parameter_boundary)

        energy_linear_error = float(
            np.sqrt(max(linear_covariance[1, 1], 1e-8))
        )
        log_sigma_linear_error = float(
            np.sqrt(max(linear_covariance[0, 0], 1e-8))
        )
        energy_lower, energy_upper = profile_parameter_interval(
            best,
            best_value,
            0,
            ENERGY_BOUNDS,
            temperatures,
            log_tau_grids,
            profile_rows,
            max(energy_linear_error, 0.002),
        )
        sigma_lower, sigma_upper = profile_parameter_interval(
            best,
            best_value,
            1,
            LOG_SIGMA_BOUNDS,
            temperatures,
            log_tau_grids,
            profile_rows,
            max(log_sigma_linear_error, 0.05),
        )

        leverage = weighted_leverage(temperatures, mean_sigma)
        observed_log_tau = np.log(tau)
        residual = observed_log_tau - predicted
        signed_z = np.empty(selected_indices.size, dtype=float)
        for point_index, (
            observed,
            predicted_value,
            grid,
            profile,
        ) in enumerate(
            zip(observed_log_tau, predicted, log_tau_grids, profile_rows)
        ):
            delta, _ = interpolate_profile(
                grid, profile, float(predicted_value)
            )
            signed_z[point_index] = np.sign(
                observed - predicted_value
            ) * np.sqrt(max(delta, 0.0))

        dof = int(selected_indices.size - 2)
        pvalue = float(chi2.sf(best_value, dof))
        if np.any(outside):
            status = STATUS_PREDICTION_OUTSIDE_PROFILE
        elif parameter_boundary:
            status = STATUS_PARAMETER_BOUNDARY
        elif pvalue >= SRH_PVALUE_THRESHOLD:
            status = STATUS_SRH_CONSISTENT
        else:
            status = STATUS_NON_SRH

        output["status"][candidate_index] = status
        output["energy"][candidate_index] = best[0]
        output["energy_interval_lower"][candidate_index] = energy_lower
        output["energy_interval_upper"][candidate_index] = energy_upper
        output["log_sigma"][candidate_index] = best[1]
        output["log_sigma_interval_lower"][candidate_index] = sigma_lower
        output["log_sigma_interval_upper"][candidate_index] = sigma_upper
        output["deviance"][candidate_index] = best_value
        output["dof"][candidate_index] = dof
        output["pvalue"][candidate_index] = pvalue
        output["max_leverage"][candidate_index] = float(np.max(leverage))
        max_index = int(np.argmax(leverage))
        output["max_leverage_temperature"][candidate_index] = int(
            temperatures[max_index]
        )
        output["fit_temperature_min"][candidate_index] = float(
            np.min(temperatures)
        )
        output["fit_temperature_max"][candidate_index] = float(
            np.max(temperatures)
        )
        output["used_temperature"][candidate_index, selected_indices] = 1
        output["predicted_log_tau"][
            candidate_index, selected_indices
        ] = predicted
        output["residual_dex"][
            candidate_index, selected_indices
        ] = residual / np.log(10.0)
        output["signed_profile_z"][
            candidate_index, selected_indices
        ] = signed_z
        output["leverage"][candidate_index, selected_indices] = leverage
        seconds_peak = tau * np.log(8.0) / 7.0
        for local_index, temperature_index in enumerate(selected_indices):
            temperature = int(temperatures_all[temperature_index])
            tau_grid = np.exp(
                profiles_by_temperature[temperature]["log_tau_grid"]
            )
            # The original scan range is tau bounds divided by the fitter's
            # 0.1/10 extension factors.
            dwell_min = float(tau_grid.min() / 0.1)
            dwell_max = float(tau_grid.max() / 10.0)
            output["peak_in_window"][
                candidate_index, temperature_index
            ] = int(dwell_min <= seconds_peak[local_index] <= dwell_max)

        if (candidate_index + 1) % 250 == 0:
            print(
                f"Step 10 fit: processed candidate index {candidate_index + 1}",
                flush=True,
            )
    return output


def bootstrap_median_ci(
    values_by_site: dict[int, np.ndarray],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    site_ids = np.asarray(sorted(values_by_site), dtype=int)
    observed = np.concatenate([values_by_site[index] for index in site_ids])
    median = float(np.median(observed))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(site_ids, size=site_ids.size, replace=True)
        boot[draw] = np.median(
            np.concatenate([values_by_site[index] for index in sampled])
        )
    lower, upper = np.quantile(boot, [0.005, 0.995])
    return median, float(lower), float(upper)


def bootstrap_population_median(
    values: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.empty(draws, dtype=float)
    for draw in range(draws):
        medians[draw] = np.median(
            rng.choice(values, size=values.size, replace=True)
        )
    median = float(np.median(values))
    low, high = np.quantile(medians, [0.16, 0.84])
    sigma = float(0.5 * (high - low))
    return median, sigma


def family_systematic(
    site: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    no_family: dict[str, np.ndarray],
) -> dict[str, object]:
    temperatures = np.asarray(site["temperatures"], dtype=int)
    family_indices = np.flatnonzero(
        np.isin(temperatures, FAMILY_TEMPERATURES)
    )
    successful_full = np.isin(
        full["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    residuals_by_site = {}
    for candidate_index in np.flatnonzero(successful_full):
        values = full["residual_dex"][candidate_index, family_indices]
        values = values[np.isfinite(values)]
        if values.size:
            residuals_by_site[candidate_index] = values
    residual_median, residual_lower, residual_upper = bootstrap_median_ci(
        residuals_by_site
    )
    residual_trigger = bool(
        abs(residual_median) > FAMILY_RESIDUAL_THRESHOLD_DEX
        and not (residual_lower <= 0.0 <= residual_upper)
    )

    paired = np.isin(
        full["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    ) & np.isin(
        no_family["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    full_median, full_sigma = bootstrap_population_median(
        full["energy"][paired], seed=BOOTSTRAP_SEED + 1
    )
    no_median, no_sigma = bootstrap_population_median(
        no_family["energy"][paired], seed=BOOTSTRAP_SEED + 2
    )
    median_shift = float(no_median - full_median)
    combined_sigma = float(np.hypot(full_sigma, no_sigma))
    population_trigger = bool(abs(median_shift) > combined_sigma)
    promote = residual_trigger or population_trigger
    return {
        "family_temperatures": list(FAMILY_TEMPERATURES),
        "residual_site_count": len(residuals_by_site),
        "pooled_residual_median_dex": residual_median,
        "pooled_residual_bootstrap_99_lower_dex": residual_lower,
        "pooled_residual_bootstrap_99_upper_dex": residual_upper,
        "residual_threshold_dex": FAMILY_RESIDUAL_THRESHOLD_DEX,
        "residual_trigger": residual_trigger,
        "paired_population_count": int(np.sum(paired)),
        "full_population_median_energy_ev": full_median,
        "full_population_median_energy_sigma_ev": full_sigma,
        "no_family_population_median_energy_ev": no_median,
        "no_family_population_median_energy_sigma_ev": no_sigma,
        "population_median_delta_energy_ev": median_shift,
        "population_combined_sigma_ev": combined_sigma,
        "population_trigger": population_trigger,
        "promote_no_family_primary": promote,
        "primary_variant": "no_160_170" if promote else "full",
    }


def point_diagnostics(
    site: dict[str, np.ndarray],
    variant: dict[str, np.ndarray],
) -> dict[str, object]:
    temperatures = np.asarray(site["temperatures"], dtype=int)
    successful = np.isin(
        variant["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    temperature_summary = {}
    for temp_index, temperature in enumerate(temperatures):
        selected = successful & np.isfinite(
            variant["residual_dex"][:, temp_index]
        )
        values = variant["residual_dex"][selected, temp_index]
        if values.size:
            temperature_summary[str(int(temperature))] = {
                "count": int(values.size),
                "median_residual_dex": float(np.median(values)),
                "p16_residual_dex": float(np.quantile(values, 0.16)),
                "p84_residual_dex": float(np.quantile(values, 0.84)),
            }

    point_mask = successful[:, None] & np.isfinite(variant["residual_dex"])
    residual = variant["residual_dex"][point_mask]
    amplitude = np.abs(np.asarray(site["amplitude"], dtype=float))[point_mask]
    pedestal = np.abs(np.asarray(site["offset"], dtype=float))[point_mask]
    intensity_p = np.asarray(site["fit_pvalue"], dtype=float)[point_mask]

    def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        result = spearmanr(x, y, nan_policy="omit")
        return {
            "rho": float(result.statistic),
            "pvalue": float(result.pvalue),
        }

    quadrant_summary = {}
    orientation_summary = {}
    labels = np.asarray(site["final_orientation_label"]).astype(str)
    for quadrant in range(4):
        selected = (
            point_mask
            & (np.asarray(site["candidate_quadrant"])[:, None] == quadrant)
        )
        values = variant["residual_dex"][selected]
        quadrant_summary[str(quadrant)] = {
            "count": int(values.size),
            "median_residual_dex": float(np.median(values)),
        }
    for label in np.unique(labels):
        selected = point_mask & (labels[:, None] == label)
        values = variant["residual_dex"][selected]
        if values.size:
            orientation_summary[label] = {
                "count": int(values.size),
                "median_residual_dex": float(np.median(values)),
            }
    return {
        "temperature": temperature_summary,
        "residual_vs_abs_amplitude": correlation(residual, amplitude),
        "residual_vs_abs_pedestal": correlation(residual, pedestal),
        "residual_vs_intensity_fit_pvalue": correlation(
            residual, intensity_p
        ),
        "quadrant": quadrant_summary,
        "orientation": orientation_summary,
    }


def systematic_deltas(
    full: dict[str, np.ndarray],
    no_family: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidate_count = full["status"].size
    output = {
        "paired_fit_success": np.zeros(candidate_count, dtype=np.int8),
        "delta_energy_no160170_minus_full": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "delta_log_sigma_no160170_minus_full": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "delta_pvalue_no160170_minus_full": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
        "delta_max_leverage_no160170_minus_full": np.full(
            candidate_count, np.nan, dtype=np.float32
        ),
    }
    paired = np.isin(
        full["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    ) & np.isin(
        no_family["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    output["paired_fit_success"][paired] = 1
    for output_name, source_name in (
        ("delta_energy_no160170_minus_full", "energy"),
        ("delta_log_sigma_no160170_minus_full", "log_sigma"),
        ("delta_pvalue_no160170_minus_full", "pvalue"),
        ("delta_max_leverage_no160170_minus_full", "max_leverage"),
    ):
        output[output_name][paired] = (
            no_family[source_name][paired] - full[source_name][paired]
        )
    return output


def write_artifact(
    intensity_path: Path,
    legacy_path: Path,
    output_path: Path,
) -> dict[str, object]:
    site, profiles = load_profiles(intensity_path)
    legacy = legacy_membership(
        legacy_path,
        site["candidate_quadrant"],
        site["candidate_row"],
        site["candidate_col"],
    )
    print("Step 10: fitting all accepted temperatures", flush=True)
    full = fit_variant(site, profiles, set())
    print("Step 10: fitting with 160/170 K removed", flush=True)
    no_family = fit_variant(site, profiles, set(FAMILY_TEMPERATURES))
    family = family_systematic(site, full, no_family)
    systematic = systematic_deltas(full, no_family)
    primary_name = family["primary_variant"]
    primary = no_family if primary_name == "no_160_170" else full
    diagnostics = point_diagnostics(site, primary)

    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as output:
        output.attrs["version"] = PIPELINE_VERSION
        output.attrs["intensity_sha256"] = file_sha256(intensity_path)
        output.attrs["legacy_sha256"] = file_sha256(legacy_path)
        output.attrs["model"] = (
            "tau=exp(E/kT)/(sigma*v_th*N_v), p-channel holes, "
            "energy above valence edge"
        )
        output.attrs["minimum_temperatures"] = MIN_TEMPERATURES
        output.attrs["srh_pvalue_threshold"] = SRH_PVALUE_THRESHOLD
        output.attrs["intrinsic_scatter_dex"] = 0.0
        output.attrs["primary_variant"] = primary_name
        output.attrs["family_systematic_json"] = json.dumps(
            family, sort_keys=True
        )
        output.attrs["diagnostics_json"] = json.dumps(
            diagnostics, sort_keys=True
        )
        output.create_dataset(
            "status_names",
            data=np.asarray(STATUS_NAMES, dtype=object),
            dtype=string_dtype,
        )
        for name in (
            "temperatures",
            "candidate_quadrant",
            "candidate_row",
            "candidate_col",
            "single_trap_eligible",
            "final_orientation_label",
        ):
            values = site[name]
            if values.dtype.kind in ("U", "O"):
                output.create_dataset(
                    name,
                    data=np.asarray(values, dtype=object),
                    dtype=string_dtype,
                )
            else:
                output.create_dataset(name, data=values)
        for name, values in legacy.items():
            output.create_dataset(name, data=values)
        for variant_name, values in (
            ("full", full),
            ("no_160_170", no_family),
        ):
            group = output.create_group(variant_name)
            for name, array in values.items():
                group.create_dataset(
                    name,
                    data=array,
                    compression="gzip" if array.ndim > 1 else None,
                    compression_opts=4 if array.ndim > 1 else None,
                    shuffle=array.ndim > 1,
                )
        systematic_group = output.create_group("family_systematic")
        for name, array in systematic.items():
            systematic_group.create_dataset(name, data=array)
        primary_group = output.create_group("primary")
        primary_group.attrs["source_variant"] = primary_name
        for name, array in primary.items():
            primary_group[name] = h5py.SoftLink(f"/{primary_name}/{name}")
    temporary.replace(output_path)
    return {
        "site": site,
        "legacy": legacy,
        "full": full,
        "no_family": no_family,
        "systematic": systematic,
        "primary": primary,
        "family": family,
        "diagnostics": diagnostics,
    }


def status_counts(values: np.ndarray) -> dict[str, int]:
    counts = np.bincount(values, minlength=len(STATUS_NAMES))
    return {
        name: int(counts[index]) for index, name in enumerate(STATUS_NAMES)
    }


def legacy_comparison(payload: dict[str, object]) -> dict[str, object]:
    site = payload["site"]
    legacy = payload["legacy"]
    primary = payload["primary"]
    labels = np.asarray(site["final_orientation_label"]).astype(str)
    legacy_well = np.asarray(legacy["legacy_well_behaved"], dtype=bool)
    legacy_good = np.asarray(legacy["legacy_good_energy"], dtype=bool)
    current_single = np.asarray(site["single_trap_eligible"], dtype=bool)
    current_consistent = primary["status"] == STATUS_SRH_CONSISTENT
    common = legacy_good & current_consistent
    energy_delta = (
        primary["energy"][common] - legacy["legacy_energy"][common]
    )
    log_sigma_delta = (
        primary["log_sigma"][common] - legacy["legacy_log_sigma"][common]
    )
    return {
        "legacy_well_behaved_count": int(np.sum(legacy_well)),
        "legacy_good_energy_count": int(np.sum(legacy_good)),
        "legacy_well_behaved_step9_single_count": int(
            np.sum(legacy_well & current_single)
        ),
        "legacy_well_behaved_step9_excluded_count": int(
            np.sum(legacy_well & ~current_single)
        ),
        "legacy_well_behaved_exclusion_labels": dict(
            Counter(labels[legacy_well & ~current_single])
        ),
        "legacy_good_energy_current_srh_consistent_count": int(np.sum(common)),
        "common_median_delta_energy_ev": (
            float(np.median(energy_delta)) if energy_delta.size else np.nan
        ),
        "common_median_delta_log_sigma": (
            float(np.median(log_sigma_delta))
            if log_sigma_delta.size
            else np.nan
        ),
    }


def write_figures(
    payload: dict[str, object],
    figure_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    site = payload["site"]
    primary = payload["primary"]
    full = payload["full"]
    no_family = payload["no_family"]
    legacy = payload["legacy"]
    temperatures = np.asarray(site["temperatures"], dtype=int)
    outputs = []

    successful = np.isin(
        primary["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    medians = []
    p16 = []
    p84 = []
    for temp_index, temperature in enumerate(temperatures):
        values = primary["residual_dex"][
            successful
            & np.isfinite(primary["residual_dex"][:, temp_index]),
            temp_index,
        ]
        medians.append(np.median(values) if values.size else np.nan)
        p16.append(np.quantile(values, 0.16) if values.size else np.nan)
        p84.append(np.quantile(values, 0.84) if values.size else np.nan)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.fill_between(temperatures, p16, p84, color="0.85")
    axis.plot(temperatures, medians, "o-", color="tab:blue")
    axis.axvspan(159, 171, color="tab:orange", alpha=0.15)
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel("Observed - SRH predicted log10(tau)")
    axis.set_title(
        f"Primary SRH residuals ({payload['family']['primary_variant']})"
    )
    fig.tight_layout()
    path = figure_dir / "srh_residuals_by_temperature.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    paired = np.isin(
        full["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    ) & np.isin(
        no_family["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        full["energy"][paired],
        no_family["energy"][paired],
        s=7,
        alpha=0.35,
    )
    limits = [0.0, 1.0]
    axis.plot(limits, limits, color="black", linewidth=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Full-fit energy (eV)")
    axis.set_ylabel("No-160/170 K energy (eV)")
    axis.set_title("Acquisition-family energy sensitivity")
    fig.tight_layout()
    path = figure_dir / "family_energy_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    common = (
        np.asarray(legacy["legacy_good_energy"], dtype=bool)
        & (primary["status"] == STATUS_SRH_CONSISTENT)
    )
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        legacy["legacy_energy"][common],
        primary["energy"][common],
        s=7,
        alpha=0.35,
    )
    axis.plot(limits, limits, color="black", linewidth=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Legacy energy (eV)")
    axis.set_ylabel("Current profile-SRH energy (eV)")
    axis.set_title("Common characterized population")
    fig.tight_layout()
    path = figure_dir / "legacy_current_energy_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)
    return outputs


def write_report(
    payload: dict[str, object],
    output_path: Path,
    report_path: Path,
    figure_paths: list[Path],
) -> dict[str, object]:
    site = payload["site"]
    full = payload["full"]
    no_family = payload["no_family"]
    primary = payload["primary"]
    family = payload["family"]
    diagnostics = payload["diagnostics"]
    systematic = payload["systematic"]
    legacy = legacy_comparison(payload)
    primary_success = np.isin(
        primary["status"], (STATUS_NON_SRH, STATUS_SRH_CONSISTENT)
    )
    primary_consistent = primary["status"] == STATUS_SRH_CONSISTENT
    paired_systematic = np.asarray(
        systematic["paired_fit_success"], dtype=bool
    )
    primary_points = primary["used_temperature"].astype(bool)
    primary_peak_window = primary["peak_in_window"].astype(bool)
    peak_window_count = int(np.sum(primary_points))
    peak_outside_count = int(
        np.sum(primary_points & ~primary_peak_window)
    )
    primary_leverage = primary["max_leverage"][primary_success]
    high_temperature_flags = []
    for temperature_text, values in diagnostics["temperature"].items():
        temperature = int(temperature_text)
        if (
            temperature >= HIGH_TEMPERATURE_MIN
            and abs(values["median_residual_dex"]) > 0.10
        ):
            high_temperature_flags.append(
                {
                    "temperature": temperature,
                    "median_residual_dex": values["median_residual_dex"],
                }
            )

    failures = []
    if np.any(
        primary_consistent
        & (
            ~np.isfinite(primary["energy_interval_lower"])
            | ~np.isfinite(primary["energy_interval_upper"])
            | ~np.isfinite(primary["log_sigma_interval_lower"])
            | ~np.isfinite(primary["log_sigma_interval_upper"])
        )
    ):
        failures.append("consistent SRH fits lack profile intervals")
    if payload["family"]["primary_variant"] not in ("full", "no_160_170"):
        failures.append("family primary variant is undefined")
    if np.any(primary_consistent & primary["prediction_outside_profile"].astype(bool)):
        failures.append("consistent SRH fit predicts outside a tau profile")
    status = "PASS" if not failures else "FAIL"

    lines = [
        "# Signed Refit Step 10 SRH Validation",
        "",
        f"- Pipeline version: `{PIPELINE_VERSION}`",
        f"- Artifact: `{output_path.as_posix()}`",
        f"- Artifact SHA-256: `{file_sha256(output_path)}`",
        f"- Acceptance status: **{status}**",
        f"- Primary variant: `{family['primary_variant']}`.",
        "- Model: simple p-channel SRH emission from a level above the valence edge.",
        "- Intrinsic scatter: fixed to zero; no outlier rejection.",
        "- SRH-consistent classification: profile deviance `p>=0.05`.",
        "",
        "## Population cutflow",
        "",
        f"- Step 9 single-trap inputs: "
        f"{int(np.sum(site['single_trap_eligible'])):,}.",
        f"- Primary fit successes: {int(np.sum(primary_success)):,}.",
        f"- Primary SRH-consistent sites: {int(np.sum(primary_consistent)):,}.",
        "",
        "| Status | Full | No 160/170 K | Primary |",
        "|---|---:|---:|---:|",
    ]
    full_counts = status_counts(full["status"])
    no_counts = status_counts(no_family["status"])
    primary_counts = status_counts(primary["status"])
    for name in STATUS_NAMES:
        lines.append(
            f"| `{name}` | {full_counts[name]:,} | "
            f"{no_counts[name]:,} | {primary_counts[name]:,} |"
        )

    lines.extend(
        [
            "",
            "## Acquisition-family systematic",
            "",
            f"- 160/170 K pooled residual sites: "
            f"{family['residual_site_count']:,}.",
            f"- Pooled residual median: "
            f"{family['pooled_residual_median_dex']:+.4f} dex.",
            f"- Site-bootstrap 99% interval: "
            f"[{family['pooled_residual_bootstrap_99_lower_dex']:+.4f}, "
            f"{family['pooled_residual_bootstrap_99_upper_dex']:+.4f}] dex.",
            f"- Residual trigger: {family['residual_trigger']}.",
            f"- Paired fit population: {family['paired_population_count']:,}.",
            f"- Full/no-family median energy: "
            f"{family['full_population_median_energy_ev']:.4f}/"
            f"{family['no_family_population_median_energy_ev']:.4f} eV.",
            f"- Median energy shift: "
            f"{family['population_median_delta_energy_ev']:+.4f} eV; "
            f"combined one-sigma uncertainty "
            f"{family['population_combined_sigma_ev']:.4f} eV.",
            f"- Paired median no-family-minus-full ln(sigma): "
            f"{np.median(systematic['delta_log_sigma_no160170_minus_full'][paired_systematic]):+.4f}.",
            f"- Population trigger: {family['population_trigger']}.",
            f"- Primary decision: `{family['primary_variant']}`.",
            "",
            "## Residual diagnostics",
            "",
            f"- Primary fitted points outside the measured pump-peak window: "
            f"{peak_outside_count:,}/{peak_window_count:,} "
            f"({peak_outside_count / max(peak_window_count, 1):.2%}).",
            f"- Primary median/maximum per-site leverage: "
            f"{np.median(primary_leverage):.3f}/"
            f"{np.max(primary_leverage):.3f}.",
            f"- Residual versus |amplitude| Spearman rho: "
            f"{diagnostics['residual_vs_abs_amplitude']['rho']:+.3f} "
            f"(p={diagnostics['residual_vs_abs_amplitude']['pvalue']:.3g}).",
            f"- Residual versus |pedestal| Spearman rho: "
            f"{diagnostics['residual_vs_abs_pedestal']['rho']:+.3f} "
            f"(p={diagnostics['residual_vs_abs_pedestal']['pvalue']:.3g}).",
            f"- Residual versus intensity-fit p-value Spearman rho: "
            f"{diagnostics['residual_vs_intensity_fit_pvalue']['rho']:+.3f} "
            f"(p={diagnostics['residual_vs_intensity_fit_pvalue']['pvalue']:.3g}).",
            f"- High-temperature median residuals exceeding 0.10 dex: "
            f"{len(high_temperature_flags)}.",
        ]
    )
    for item in high_temperature_flags:
        lines.append(
            f"  - {item['temperature']} K: "
            f"{item['median_residual_dex']:+.3f} dex."
        )

    lines.extend(
        [
            "",
            "No high-temperature point is removed or assigned intrinsic scatter. "
            "Any listed deviation remains a documented simple-SRH failure mode.",
            "",
            "## Legacy comparison",
            "",
            f"- Legacy well-behaved sites: "
            f"{legacy['legacy_well_behaved_count']:,}.",
            f"- Legacy well-behaved entering current Step 10: "
            f"{legacy['legacy_well_behaved_step9_single_count']:,}.",
            f"- Legacy well-behaved excluded before Step 10: "
            f"{legacy['legacy_well_behaved_step9_excluded_count']:,}.",
            f"- Legacy good-energy sites: "
            f"{legacy['legacy_good_energy_count']:,}.",
            f"- Legacy good-energy and current primary SRH-consistent: "
            f"{legacy['legacy_good_energy_current_srh_consistent_count']:,}.",
            f"- Common-population median current-minus-legacy energy: "
            f"{legacy['common_median_delta_energy_ev']:+.4f} eV.",
            f"- Common-population median current-minus-legacy ln(sigma): "
            f"{legacy['common_median_delta_log_sigma']:+.4f}.",
            "",
            "Legacy well-behaved Step 9 exclusions:",
            "",
            "| Current classification | Sites |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(
        legacy["legacy_well_behaved_exclusion_labels"].items()
    ):
        lines.append(f"| `{label}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Figures",
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
                "- PASS: energy and cross-section intervals profile the calibrated tau likelihoods.",
                "- PASS: one fixed simple-SRH consistency criterion is used without intrinsic scatter or outlier removal.",
                "- PASS: acquisition-family promotion was evaluated before selecting the primary variant.",
                "- PASS: high-temperature deviations are retained and reported rather than tuned away.",
                "- PASS: non-SRH and failed sites remain explicit classifications.",
            ]
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": status,
        "failures": failures,
        "primary_variant": family["primary_variant"],
        "primary_fit_success": int(np.sum(primary_success)),
        "primary_srh_consistent": int(np.sum(primary_consistent)),
        "legacy": legacy,
    }


def validate_artifact(
    intensity_path: Path,
    legacy_path: Path,
    output_path: Path,
) -> dict[str, object]:
    with h5py.File(output_path, "r") as handle:
        errors = []
        if str(handle.attrs.get("version", "")) != PIPELINE_VERSION:
            errors.append("version mismatch")
        if str(handle.attrs.get("intensity_sha256", "")) != file_sha256(
            intensity_path
        ):
            errors.append("intensity SHA mismatch")
        if str(handle.attrs.get("legacy_sha256", "")) != file_sha256(
            legacy_path
        ):
            errors.append("legacy SHA mismatch")
        if float(handle.attrs.get("intrinsic_scatter_dex", np.nan)) != 0.0:
            errors.append("intrinsic scatter is not zero")
        primary_name = str(handle.attrs.get("primary_variant", ""))
        if primary_name not in ("full", "no_160_170"):
            errors.append("invalid primary variant")
        if "family_systematic" not in handle:
            errors.append("family-systematic deltas are missing")
        else:
            systematic = handle["family_systematic"]
            required_deltas = (
                "delta_energy_no160170_minus_full",
                "delta_log_sigma_no160170_minus_full",
                "delta_pvalue_no160170_minus_full",
                "delta_max_leverage_no160170_minus_full",
            )
            for name in required_deltas:
                if name not in systematic:
                    errors.append(f"{name} is missing")
        primary = handle["primary"]
        consistent = (
            np.asarray(primary["status"], dtype=int) == STATUS_SRH_CONSISTENT
        )
        for name in (
            "energy_interval_lower",
            "energy_interval_upper",
            "log_sigma_interval_lower",
            "log_sigma_interval_upper",
        ):
            if np.any(consistent & ~np.isfinite(np.asarray(primary[name]))):
                errors.append(f"{name} missing for consistent fits")
        if np.any(
            consistent
            & np.asarray(
                primary["prediction_outside_profile"], dtype=bool
            )
        ):
            errors.append("consistent fit outside profile support")
        if errors:
            raise ValueError("SRH validation failed:\n- " + "\n- ".join(errors))
        return {
            "primary_variant": primary_name,
            "primary_srh_consistent": int(np.sum(consistent)),
            "output_sha256": file_sha256(output_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intensity", type=Path, default=DEFAULT_INTENSITY)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        payload = write_artifact(args.intensity, args.legacy, args.output)
        figures = write_figures(payload, args.figure_dir)
        report = write_report(payload, args.output, args.report, figures)
        if report["status"] != "PASS":
            raise RuntimeError(
                "Step 10 acceptance failed: "
                + "; ".join(report["failures"])
            )
        print(json.dumps(report, indent=2, default=str))
    validation = validate_artifact(args.intensity, args.legacy, args.output)
    print("PASS: " + json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()

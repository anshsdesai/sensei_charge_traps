"""Covariance-aware profile-likelihood fitter for signed dipole intensities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import chi2

from signed_refit_controls import COL_REGIONS, IMAGE_SHAPE, ROW_REGIONS
from signed_refit_noise_model import NOISE_MODEL_VERSION
from signed_refit_variance_model import (
    N_PUMPS,
    VARIANCE_MODEL_VERSION,
    candidate_covariance,
)


PROFILE_FITTER_VERSION = "signed-refit-profile-tau-v1"
SIGNAL_DEPENDENT_FITTER_VERSION = "signed-refit-profile-tau-signal-variance-v1"
DEFAULT_GRID_SIZE = 801
DEFAULT_DELTA_CHI2 = 1.0
DEFAULT_TAU_LOWER_FACTOR = 0.1
DEFAULT_TAU_UPPER_FACTOR = 10.0
COMPETING_MODE_DELTA_CHI2 = 4.0
MIN_MODE_SEPARATION_DEX = 0.25


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pump_shape(seconds: np.ndarray, tau: float) -> np.ndarray:
    """Return the signed unit-coefficient pocket-pumping shape."""
    seconds = np.asarray(seconds, dtype=float)
    return 3000.0 * (
        np.exp(-seconds / tau) - np.exp(-8.0 * seconds / tau)
    )


def intensity_model(
    seconds: np.ndarray,
    amplitude: float,
    tau: float,
    offset: float,
    null_template: np.ndarray | None = None,
) -> np.ndarray:
    model = amplitude * pump_shape(seconds, tau) + offset
    if null_template is not None:
        model = model + np.asarray(null_template, dtype=float)
    return model


def detector_region(row: int, col: int) -> int:
    if not (0 <= row < IMAGE_SHAPE[0] and 0 <= col < IMAGE_SHAPE[1]):
        raise ValueError(f"Coordinate {(row, col)} is outside {IMAGE_SHAPE}")
    row_region = min(row * ROW_REGIONS // IMAGE_SHAPE[0], ROW_REGIONS - 1)
    col_region = min(col * COL_REGIONS // IMAGE_SHAPE[1], COL_REGIONS - 1)
    return int(row_region * COL_REGIONS + col_region)


@dataclass(frozen=True)
class ScanCalibration:
    temperature: int
    quadrant: int
    region: int
    seconds: np.ndarray
    covariance: np.ndarray
    null_template: np.ndarray
    noise_model_version: str
    noise_model_sha256: str


def load_scan_calibration(
    model_path: Path | str,
    temperature: int,
    quadrant: int,
    *,
    region: int | None = None,
    row: int | None = None,
    col: int | None = None,
) -> ScanCalibration:
    """Load the exact scan covariance/template for a candidate."""
    if region is None:
        if row is None or col is None:
            raise ValueError("Supply either region or both row and col")
        region = detector_region(row, col)
    if not 0 <= quadrant < 4:
        raise ValueError("quadrant must be in [0, 3]")
    if not 0 <= region < ROW_REGIONS * COL_REGIONS:
        raise ValueError("region is outside the configured detector grid")

    model_path = Path(model_path)
    with h5py.File(model_path, "r") as handle:
        version = str(handle.attrs.get("version", ""))
        if version != NOISE_MODEL_VERSION:
            raise ValueError(
                f"Expected noise model {NOISE_MODEL_VERSION}, found {version}"
            )
        temp_group = handle[f"temp_{int(temperature)}"]
        region_group = temp_group[f"quad_{quadrant}/region_{region}"]
        seconds = np.asarray(temp_group["seconds"], dtype=float)
        covariance = np.asarray(region_group["covariance"], dtype=float)
        null_template = np.asarray(region_group["null_template"], dtype=float)

    return ScanCalibration(
        temperature=int(temperature),
        quadrant=int(quadrant),
        region=int(region),
        seconds=seconds,
        covariance=covariance,
        null_template=null_template,
        noise_model_version=version,
        noise_model_sha256=file_sha256(model_path),
    )


def _profile_crossing(
    log_tau: np.ndarray,
    chi2_profile: np.ndarray,
    best_log_tau: float,
    best_chi2: float,
    target_delta: float,
    evaluate,
    direction: int,
) -> float | None:
    augmented = np.unique(np.append(log_tau, best_log_tau))
    if direction < 0:
        points = augmented[augmented < best_log_tau][::-1]
    else:
        points = augmented[augmented > best_log_tau]

    inner_log = best_log_tau
    inner_value = -target_delta
    for outer_log in points:
        outer_value = evaluate(float(outer_log))[0] - best_chi2 - target_delta
        if np.isfinite(outer_value) and outer_value >= 0 and inner_value <= 0:
            lo, hi = sorted((inner_log, float(outer_log)))
            root = brentq(
                lambda value: evaluate(value)[0] - best_chi2 - target_delta,
                lo,
                hi,
            )
            return float(np.exp(root))
        if np.isfinite(outer_value):
            inner_log = float(outer_log)
            inner_value = float(outer_value)
    return None


def profile_mode_indices(chi2_profile: np.ndarray) -> np.ndarray:
    """Return discrete local-minimum indices, including finite boundaries."""
    values = np.asarray(chi2_profile, dtype=float)
    finite = np.isfinite(values)
    working = np.where(finite, values, np.inf)
    indices = list(find_peaks(-working)[0])
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size:
        first = int(finite_indices[0])
        last = int(finite_indices[-1])
        if first == last or working[first] <= working[min(first + 1, last)]:
            indices.append(first)
        if last != first and working[last] <= working[max(last - 1, first)]:
            indices.append(last)
    return np.asarray(sorted(set(indices)), dtype=int)


class ProfileTauFitter:
    """Profile over tau while solving signed amplitude and offset exactly."""

    def __init__(
        self,
        seconds: np.ndarray,
        covariance: np.ndarray,
        *,
        null_template: np.ndarray | None = None,
        tau_bounds: tuple[float, float] | None = None,
        grid_size: int = DEFAULT_GRID_SIZE,
    ):
        self.seconds = np.asarray(seconds, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        if self.seconds.ndim != 1 or self.seconds.size < 4:
            raise ValueError("seconds must be a one-dimensional scan with >=4 points")
        if np.any(~np.isfinite(self.seconds)) or np.any(self.seconds <= 0):
            raise ValueError("seconds must be finite and positive")
        if self.covariance.shape != (self.seconds.size, self.seconds.size):
            raise ValueError("covariance shape does not match the dwell grid")
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-10, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        np.linalg.cholesky(self.covariance)
        self.precision = np.linalg.inv(self.covariance)
        self.null_template = (
            np.zeros(self.seconds.size, dtype=float)
            if null_template is None
            else np.asarray(null_template, dtype=float)
        )
        if self.null_template.shape != self.seconds.shape:
            raise ValueError("null_template shape does not match the dwell grid")
        if grid_size < 101:
            raise ValueError("grid_size must be at least 101")

        if tau_bounds is None:
            tau_bounds = (
                float(np.min(self.seconds)) * DEFAULT_TAU_LOWER_FACTOR,
                float(np.max(self.seconds)) * DEFAULT_TAU_UPPER_FACTOR,
            )
        self.tau_bounds = tuple(map(float, tau_bounds))
        if (
            self.tau_bounds[0] <= 0
            or self.tau_bounds[1] <= self.tau_bounds[0]
        ):
            raise ValueError("tau_bounds must be positive and increasing")

        self.log_tau_grid = np.linspace(
            np.log(self.tau_bounds[0]),
            np.log(self.tau_bounds[1]),
            int(grid_size),
        )
        self.tau_grid = np.exp(self.log_tau_grid)
        self.shape_grid = np.asarray(
            [pump_shape(self.seconds, tau) for tau in self.tau_grid]
        )
        self.shape_precision = self.shape_grid @ self.precision
        self.ones = np.ones(self.seconds.size)
        self.precision_ones = self.precision @ self.ones
        self.constant_normal = float(self.ones @ self.precision_ones)
        self.shape_normal = np.einsum(
            "gi,gi->g", self.shape_precision, self.shape_grid
        )
        self.shape_constant = self.shape_grid @ self.precision_ones
        self.determinant = (
            self.shape_normal * self.constant_normal - self.shape_constant**2
        )
        scale = np.maximum(
            self.shape_normal * self.constant_normal,
            np.finfo(float).tiny,
        )
        self.valid_grid = (
            np.isfinite(self.determinant)
            & (self.shape_normal > 0)
            & (self.determinant > 1e-12 * scale)
        )
        if np.count_nonzero(self.valid_grid) < 3:
            raise ValueError("tau bounds contain too few identifiable profile points")

    @classmethod
    def from_calibration(
        cls,
        calibration: ScanCalibration,
        **kwargs,
    ) -> "ProfileTauFitter":
        return cls(
            calibration.seconds,
            calibration.covariance,
            null_template=calibration.null_template,
            **kwargs,
        )

    def _solve_shape(
        self,
        corrected: np.ndarray,
        shape: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        pshape = self.precision @ shape
        a = float(shape @ pshape)
        b = float(shape @ self.precision_ones)
        c = self.constant_normal
        determinant = a * c - b * b
        if (
            not np.isfinite(determinant)
            or a <= 0
            or determinant <= 1e-12 * max(a * c, np.finfo(float).tiny)
        ):
            return np.inf, np.nan, np.nan, np.full((2, 2), np.nan)
        u = float(pshape @ corrected)
        v = float(self.precision_ones @ corrected)
        amplitude = (c * u - b * v) / determinant
        offset = (a * v - b * u) / determinant
        ypy = float(corrected @ self.precision @ corrected)
        fit_chi2 = max(ypy - amplitude * u - offset * v, 0.0)
        parameter_covariance = np.asarray(
            [[c, -b], [-b, a]], dtype=float
        ) / determinant
        return fit_chi2, amplitude, offset, parameter_covariance

    def fit(
        self,
        intensities: np.ndarray,
        *,
        interval_delta_chi2: float = DEFAULT_DELTA_CHI2,
    ) -> dict[str, object]:
        intensities = np.asarray(intensities, dtype=float)
        if intensities.shape != self.seconds.shape:
            raise ValueError("intensities shape does not match the dwell grid")
        if np.any(~np.isfinite(intensities)):
            raise ValueError("intensities must be finite")
        corrected = intensities - self.null_template

        ypy = float(corrected @ self.precision @ corrected)
        u = self.shape_precision @ corrected
        v = float(self.precision_ones @ corrected)
        amplitude = np.full(self.tau_grid.size, np.nan)
        offset = np.full(self.tau_grid.size, np.nan)
        chi2_profile = np.full(self.tau_grid.size, np.inf)
        amplitude_sigma = np.full(self.tau_grid.size, np.nan)
        valid = self.valid_grid
        determinant = self.determinant[valid]
        amplitude[valid] = (
            self.constant_normal * u[valid] - self.shape_constant[valid] * v
        ) / determinant
        offset[valid] = (
            self.shape_normal[valid] * v - self.shape_constant[valid] * u[valid]
        ) / determinant
        chi2_profile[valid] = np.maximum(
            ypy - amplitude[valid] * u[valid] - offset[valid] * v,
            0.0,
        )
        amplitude_sigma[valid] = np.sqrt(self.constant_normal / determinant)

        best_grid_index = int(np.nanargmin(chi2_profile))

        def evaluate(log_tau: float):
            tau = float(np.exp(log_tau))
            return self._solve_shape(corrected, pump_shape(self.seconds, tau))

        if 0 < best_grid_index < self.tau_grid.size - 1:
            refinement = minimize_scalar(
                lambda value: evaluate(float(value))[0],
                bounds=(
                    float(self.log_tau_grid[best_grid_index - 1]),
                    float(self.log_tau_grid[best_grid_index + 1]),
                ),
                method="bounded",
                options={"xatol": 1e-12},
            )
            best_log_tau = float(refinement.x)
        else:
            best_log_tau = float(self.log_tau_grid[best_grid_index])
        best_chi2, best_amplitude, best_offset, best_linear_covariance = evaluate(
            best_log_tau
        )
        best_tau = float(np.exp(best_log_tau))

        lower = _profile_crossing(
            self.log_tau_grid,
            chi2_profile,
            best_log_tau,
            best_chi2,
            interval_delta_chi2,
            evaluate,
            -1,
        )
        upper = _profile_crossing(
            self.log_tau_grid,
            chi2_profile,
            best_log_tau,
            best_chi2,
            interval_delta_chi2,
            evaluate,
            1,
        )

        constant_offset = v / self.constant_normal
        constant_chi2 = max(
            ypy - constant_offset * v,
            0.0,
        )
        amplitude_error = float(np.sqrt(best_linear_covariance[0, 0]))
        offset_error = float(np.sqrt(best_linear_covariance[1, 1]))

        mode_summaries = []
        for index in profile_mode_indices(chi2_profile):
            if 0 < index < self.tau_grid.size - 1:
                refined = minimize_scalar(
                    lambda value: evaluate(float(value))[0],
                    bounds=(
                        float(self.log_tau_grid[index - 1]),
                        float(self.log_tau_grid[index + 1]),
                    ),
                    method="bounded",
                )
                mode_log_tau = float(refined.x)
                mode_chi2 = float(refined.fun)
            else:
                mode_log_tau = float(self.log_tau_grid[index])
                mode_chi2 = float(chi2_profile[index])
            mode_summaries.append(
                {
                    "tau": float(np.exp(mode_log_tau)),
                    "chi2": mode_chi2,
                    "delta_chi2": max(mode_chi2 - best_chi2, 0.0),
                }
            )
        mode_summaries.sort(key=lambda item: item["chi2"])
        competitive_modes = []
        for mode in mode_summaries:
            if mode["delta_chi2"] > COMPETING_MODE_DELTA_CHI2:
                continue
            if all(
                abs(np.log10(mode["tau"] / existing["tau"]))
                >= MIN_MODE_SEPARATION_DEX
                for existing in competitive_modes
            ):
                competitive_modes.append(mode)

        lower_limited = lower is None
        upper_limited = upper is None
        at_lower_boundary = best_grid_index == int(np.flatnonzero(valid)[0])
        at_upper_boundary = best_grid_index == int(np.flatnonzero(valid)[-1])
        boundary_limited = (
            lower_limited
            or upper_limited
            or at_lower_boundary
            or at_upper_boundary
        )
        dof = self.seconds.size - 3

        return {
            "version": PROFILE_FITTER_VERSION,
            "seconds": self.seconds.copy(),
            "tau_bounds": self.tau_bounds,
            "tau_grid": self.tau_grid.copy(),
            "chi2_profile": chi2_profile,
            "amplitude_profile": amplitude,
            "offset_profile": offset,
            "amplitude_sigma_profile": amplitude_sigma,
            "tau": best_tau,
            "tau_interval_lower": lower,
            "tau_interval_upper": upper,
            "tau_error_lower": None if lower is None else best_tau - lower,
            "tau_error_upper": None if upper is None else upper - best_tau,
            "interval_delta_chi2": float(interval_delta_chi2),
            "amplitude": float(best_amplitude),
            "amplitude_error_conditional": amplitude_error,
            "amplitude_z_conditional": float(best_amplitude / amplitude_error),
            "amplitude_sign": int(np.sign(best_amplitude)),
            "offset": float(best_offset),
            "offset_error_conditional": offset_error,
            "linear_parameter_covariance": best_linear_covariance,
            "chi2": float(best_chi2),
            "dof": int(dof),
            "reduced_chi2": float(best_chi2 / dof),
            "fit_p_value": float(chi2.sf(best_chi2, dof)),
            "constant_offset": float(constant_offset),
            "constant_chi2": float(constant_chi2),
            "delta_chi2_vs_constant": float(constant_chi2 - best_chi2),
            "at_lower_boundary": bool(at_lower_boundary),
            "at_upper_boundary": bool(at_upper_boundary),
            "interval_lower_limited": bool(lower_limited),
            "interval_upper_limited": bool(upper_limited),
            "boundary_limited": bool(boundary_limited),
            "profile_modes": mode_summaries,
            "competitive_modes": competitive_modes,
            "multimodal": len(competitive_modes) > 1,
            "initial_guess_used": False,
            "null_template_applied": bool(np.any(self.null_template != 0)),
        }

    def batch_profile_statistic(
        self,
        intensities: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Evaluate the complete frozen tau grid for many curves at once.

        This is the Step 6 ranking statistic. It intentionally uses every point
        on the same 801-point grid as ``fit``; the Step 5 validation showed that
        subsequent continuous refinement changes fitted tau by less than
        6e-7 dex and has negligible effect on the null calibration.
        """
        intensities = np.asarray(intensities, dtype=float)
        if intensities.ndim == 1:
            intensities = intensities[None, :]
        if intensities.ndim != 2 or intensities.shape[1] != self.seconds.size:
            raise ValueError("intensities must have shape (n_curves, n_dtph)")
        if np.any(~np.isfinite(intensities)):
            raise ValueError("intensities must be finite")

        corrected = intensities - self.null_template[None, :]
        ypy = np.einsum(
            "ni,ij,nj->n",
            corrected,
            self.precision,
            corrected,
        )
        u = corrected @ self.shape_precision.T
        v = corrected @ self.precision_ones
        determinant = self.determinant
        amplitude = np.full_like(u, np.nan)
        offset = np.full_like(u, np.nan)
        chi2_profile = np.full_like(u, np.inf)
        valid = self.valid_grid
        amplitude[:, valid] = (
            self.constant_normal * u[:, valid]
            - v[:, None] * self.shape_constant[valid][None, :]
        ) / determinant[valid][None, :]
        offset[:, valid] = (
            v[:, None] * self.shape_normal[valid][None, :]
            - u[:, valid] * self.shape_constant[valid][None, :]
        ) / determinant[valid][None, :]
        chi2_profile[:, valid] = np.maximum(
            ypy[:, None]
            - amplitude[:, valid] * u[:, valid]
            - offset[:, valid] * v[:, None],
            0.0,
        )
        best_index = np.argmin(chi2_profile, axis=1)
        rows = np.arange(intensities.shape[0])
        best_chi2 = chi2_profile[rows, best_index]
        constant_offset = v / self.constant_normal
        constant_chi2 = np.maximum(ypy - constant_offset * v, 0.0)
        return {
            "delta_chi2": np.maximum(constant_chi2 - best_chi2, 0.0),
            "tau": self.tau_grid[best_index],
            "amplitude": amplitude[rows, best_index],
            "offset": offset[rows, best_index],
            "chi2": best_chi2,
            "constant_chi2": constant_chi2,
            "best_grid_index": best_index,
        }


class SignalDependentProfileTauFitter:
    """Iteratively fit with candidate variance evaluated at the fitted signal.

    The covariance dependence is handled as feasible generalized least squares:
    fit with the current covariance, update the physical variance from the
    fitted ``(A, tau)``, and repeat. The returned profile interval is therefore
    conditional on the converged plug-in covariance. Its frequentist coverage
    is validated with held-out real-residual injections.
    """

    def __init__(
        self,
        seconds: np.ndarray,
        null_covariance: np.ndarray,
        *,
        null_template: np.ndarray | None = None,
        null_scale: float = 1.0,
        pump_overdispersion: float = 1.0,
        extra_pair_shot: np.ndarray | float = 0.0,
        n_pumps: int = N_PUMPS,
        tau_bounds: tuple[float, float] | None = None,
        grid_size: int = DEFAULT_GRID_SIZE,
        max_iterations: int = 8,
        convergence_log_tau: float = 1e-4,
        convergence_amplitude: float = 1e-4,
    ):
        self.seconds = np.asarray(seconds, dtype=float)
        self.null_covariance = np.asarray(null_covariance, dtype=float)
        self.null_template = (
            np.zeros(self.seconds.size, dtype=float)
            if null_template is None
            else np.asarray(null_template, dtype=float)
        )
        self.null_scale = float(null_scale)
        self.pump_overdispersion = float(pump_overdispersion)
        self.extra_pair_shot = np.broadcast_to(
            np.asarray(extra_pair_shot, dtype=float),
            self.seconds.shape,
        ).astype(float, copy=True)
        self.n_pumps = int(n_pumps)
        self.tau_bounds = tau_bounds
        self.grid_size = int(grid_size)
        self.max_iterations = int(max_iterations)
        self.convergence_log_tau = float(convergence_log_tau)
        self.convergence_amplitude = float(convergence_amplitude)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.null_template.shape != self.seconds.shape:
            raise ValueError("null_template shape does not match seconds")
        if self.null_covariance.shape != (self.seconds.size, self.seconds.size):
            raise ValueError("null covariance shape does not match seconds")
        if np.any(self.extra_pair_shot < 0):
            raise ValueError("extra_pair_shot must be nonnegative")

    @classmethod
    def from_calibration(
        cls,
        calibration: ScanCalibration,
        **kwargs,
    ) -> "SignalDependentProfileTauFitter":
        return cls(
            calibration.seconds,
            calibration.covariance,
            null_template=calibration.null_template,
            **kwargs,
        )

    def _base_covariance(self) -> np.ndarray:
        covariance = self.null_scale * self.null_covariance.copy()
        covariance.flat[:: self.seconds.size + 1] += self.extra_pair_shot
        return 0.5 * (covariance + covariance.T)

    def _fitter(self, covariance: np.ndarray) -> ProfileTauFitter:
        return ProfileTauFitter(
            self.seconds,
            covariance,
            null_template=self.null_template,
            tau_bounds=self.tau_bounds,
            grid_size=self.grid_size,
        )

    def fit(
        self,
        intensities: np.ndarray,
        *,
        interval_delta_chi2: float = DEFAULT_DELTA_CHI2,
    ) -> dict[str, object]:
        covariance = self._base_covariance()
        previous_amplitude = None
        previous_log_tau = None
        converged = False
        result = None

        for iteration in range(1, self.max_iterations + 1):
            result = self._fitter(covariance).fit(
                intensities,
                interval_delta_chi2=interval_delta_chi2,
            )
            updated_covariance, _ = candidate_covariance(
                self.null_covariance,
                self.seconds,
                result["amplitude"],
                result["tau"],
                null_scale=self.null_scale,
                pump_overdispersion=self.pump_overdispersion,
                extra_pair_shot=self.extra_pair_shot,
                n_pumps=self.n_pumps,
            )
            current_log_tau = float(np.log(result["tau"]))
            if previous_amplitude is not None:
                converged = (
                    abs(current_log_tau - previous_log_tau)
                    <= self.convergence_log_tau
                    and abs(float(result["amplitude"]) - previous_amplitude)
                    <= self.convergence_amplitude
                )
            covariance = updated_covariance
            previous_amplitude = float(result["amplitude"])
            previous_log_tau = current_log_tau
            if converged:
                break

        # The reported fit is evaluated with exactly the reported covariance.
        covariance = updated_covariance
        result = self._fitter(covariance).fit(
            intensities,
            interval_delta_chi2=interval_delta_chi2,
        )
        next_covariance, variance_metadata = candidate_covariance(
            self.null_covariance,
            self.seconds,
            result["amplitude"],
            result["tau"],
            null_scale=self.null_scale,
            pump_overdispersion=self.pump_overdispersion,
            extra_pair_shot=self.extra_pair_shot,
            n_pumps=self.n_pumps,
        )
        covariance_change = float(
            np.linalg.norm(next_covariance - covariance, ord="fro")
            / np.linalg.norm(covariance, ord="fro")
        )

        result["base_profile_fitter_version"] = result["version"]
        result["version"] = SIGNAL_DEPENDENT_FITTER_VERSION
        result["variance_model_version"] = VARIANCE_MODEL_VERSION
        result["variance_iterations"] = int(iteration)
        result["variance_converged"] = bool(converged or covariance_change <= 1e-5)
        result["next_covariance_fractional_change"] = covariance_change
        result["effective_covariance"] = covariance
        result["null_covariance"] = self.null_covariance.copy()
        result.update(variance_metadata)
        return result


def fit_with_noise_model(
    intensities: np.ndarray,
    model_path: Path | str,
    temperature: int,
    quadrant: int,
    *,
    region: int | None = None,
    row: int | None = None,
    col: int | None = None,
    tau_bounds: tuple[float, float] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, object]:
    calibration = load_scan_calibration(
        model_path,
        temperature,
        quadrant,
        region=region,
        row=row,
        col=col,
    )
    fitter = ProfileTauFitter.from_calibration(
        calibration,
        tau_bounds=tau_bounds,
        grid_size=grid_size,
    )
    result = fitter.fit(intensities)
    result["temperature"] = calibration.temperature
    result["quadrant"] = calibration.quadrant
    result["region"] = calibration.region
    result["noise_model_version"] = calibration.noise_model_version
    result["noise_model_sha256"] = calibration.noise_model_sha256
    return result

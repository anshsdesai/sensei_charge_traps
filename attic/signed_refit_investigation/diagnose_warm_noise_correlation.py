"""Diagnose the warm-temperature residual correlations in Step 4.

This script is intentionally read-only: it compares the frozen covariance
against training-derived alternatives without modifying any audit artifact.
"""

from __future__ import annotations

from collections import defaultdict

import h5py
import numpy as np

from signed_refit_noise_model import positive_definite_covariance


MODEL_PATH = "signed_refit_noise_model.h5"
CONTROLS_PATH = "signed_refit_control_pairs.npz"
WARM_TEMPERATURES = (197, 200, 203, 207, 210)


def whiten(residuals: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    return np.linalg.solve(chol, residuals.T).T


def max_correlation(values: np.ndarray) -> tuple[float, tuple[int, int]]:
    corr = np.corrcoef(values, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    index = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    return float(abs(corr[index])), (int(index[0]), int(index[1]))


def covariance_from_residuals(residuals: np.ndarray) -> np.ndarray:
    center = np.median(residuals, axis=0)
    mad = 1.4826 * np.median(np.abs(residuals - center), axis=0)
    std = np.std(residuals, axis=0, ddof=1)
    scale = np.where(mad > 0, mad, std)
    scale = np.where(scale > 0, scale, 1.0)
    clipped = np.clip(residuals, center - 5.0 * scale, center + 5.0 * scale)
    return positive_definite_covariance(np.cov(clipped, rowvar=False, ddof=1))


def main() -> None:
    controls = np.load(CONTROLS_PATH, allow_pickle=False)
    quadrants = controls["quadrant"]
    region_ids = controls["region"]
    split_codes = controls["split"]
    control_rows = controls["row"]
    control_cols = controls["col"]
    candidate_distance = controls["candidate_distance"]

    by_method: dict[str, dict[int, list[np.ndarray]]] = {
        name: defaultdict(list)
        for name in ("frozen", "regional_robust", "regional_sample", "pooled_q")
    }
    by_method_q: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {
        name: defaultdict(list)
        for name in ("frozen", "regional_robust", "regional_sample", "pooled_q")
    }
    frozen_by_cell: dict[tuple[int, int, int], np.ndarray] = {}
    validation_metadata_by_cell: dict[tuple[int, int, int], np.ndarray] = {}
    shrinkage_by_temperature: dict[int, list[float]] = defaultdict(list)
    seconds_by_temperature: dict[int, np.ndarray] = {}

    with h5py.File(MODEL_PATH, "r") as handle:
        for temperature in WARM_TEMPERATURES:
            temperature_group = handle[f"temp_{temperature}"]
            dtph = np.asarray(temperature_group["seconds"])
            seconds_by_temperature[temperature] = dtph
            for quadrant in range(4):
                quadrant_mask = quadrants == quadrant
                quadrant_regions = region_ids[quadrant_mask]
                quadrant_splits = split_codes[quadrant_mask]
                quadrant_rows = control_rows[quadrant_mask]
                quadrant_cols = control_cols[quadrant_mask]
                quadrant_distance = candidate_distance[quadrant_mask]
                quadrant_group = temperature_group[f"quad_{quadrant}"]
                curves = np.asarray(quadrant_group["intensity"])
                cell_data = []
                pooled_training = []
                for region in range(32):
                    train_mask = (
                        (quadrant_regions == region) & (quadrant_splits == 0)
                    )
                    validation_mask = (
                        (quadrant_regions == region) & (quadrant_splits == 1)
                    )
                    group = quadrant_group[f"region_{region}"]
                    template = np.asarray(group["null_template"])
                    train_residual = (
                        curves[train_mask]
                        - np.median(curves[train_mask], axis=1, keepdims=True)
                        - template
                    )
                    validation_residual = (
                        curves[validation_mask]
                        - np.median(
                            curves[validation_mask], axis=1, keepdims=True
                        )
                        - template
                    )
                    pooled_training.append(train_residual)
                    cell_data.append(
                        (
                            region,
                            validation_residual,
                            np.asarray(group["covariance"]),
                            np.asarray(group["robust_covariance"]),
                            np.asarray(group["sample_covariance"]),
                        )
                    )
                    shrinkage_by_temperature[temperature].append(
                        float(group.attrs["shrinkage"])
                    )

                pooled_covariance, _, _ = covariance_from_residuals(
                    np.concatenate(pooled_training)
                )
                for (
                    region,
                    validation_residual,
                    frozen_covariance,
                    robust_covariance,
                    sample_covariance,
                ) in cell_data:
                    covariances = {
                        "frozen": frozen_covariance,
                        "regional_robust": positive_definite_covariance(
                            robust_covariance
                        )[0],
                        "regional_sample": positive_definite_covariance(
                            sample_covariance
                        )[0],
                        "pooled_q": pooled_covariance,
                    }
                    for method, covariance in covariances.items():
                        z = whiten(validation_residual, covariance)
                        by_method[method][temperature].append(z)
                        by_method_q[method][(temperature, quadrant)].append(z)
                        if method == "frozen":
                            frozen_by_cell[(temperature, quadrant, region)] = z
                            validation_metadata_by_cell[
                                (temperature, quadrant, region)
                            ] = np.column_stack(
                                (
                                    quadrant_rows[validation_mask],
                                    quadrant_cols[validation_mask],
                                    quadrant_distance[validation_mask],
                                )
                            )

    print(f"dtph_seconds={dtph.tolist()}")
    for temperature in WARM_TEMPERATURES:
        dtph = seconds_by_temperature[temperature]
        shrinkage = np.asarray(shrinkage_by_temperature[temperature])
        print(
            f"\n{temperature} K OAS shrinkage: "
            f"median={np.median(shrinkage):.4f}, "
            f"range=[{np.min(shrinkage):.4f}, {np.max(shrinkage):.4f}]"
        )
        for method in by_method:
            values = np.concatenate(by_method[method][temperature])
            max_corr, pair = max_correlation(values)
            row_norm = np.sqrt(np.sum(values**2, axis=1))
            trimmed = values[row_norm <= np.quantile(row_norm, 0.99)]
            trimmed_corr, _ = max_correlation(trimmed)
            clipped_corr, _ = max_correlation(np.clip(values, -5.0, 5.0))
            print(
                f"  {method:17s} width={np.std(values):.4f} "
                f"max|corr|={max_corr:.4f} at "
                f"({pair[0]}, {pair[1]}) = "
                f"({dtph[pair[0]]:.6g}, {dtph[pair[1]]:.6g}) s; "
                f"trim1%={trimmed_corr:.4f}, clip5={clipped_corr:.4f}"
            )
        for distance_cut in (8, 12, 16, 20, 24, 32):
            filtered = []
            for key, z in frozen_by_cell.items():
                cell_temperature, _, _ = key
                if cell_temperature != temperature:
                    continue
                distances = validation_metadata_by_cell[key][:, 2]
                filtered.append(z[distances > distance_cut])
            filtered_values = np.concatenate(filtered)
            filtered_corr, _ = max_correlation(filtered_values)
            print(
                f"    candidate distance > {distance_cut:2d}: "
                f"n={filtered_values.shape[0]:5d}, "
                f"max|corr|={filtered_corr:.4f}"
            )
        for quadrant in range(4):
            values = np.concatenate(
                by_method_q["frozen"][(temperature, quadrant)]
            )
            max_corr, pair = max_correlation(values)
            cell_centered = np.concatenate(
                [
                    z - np.mean(z, axis=0, keepdims=True)
                    for (cell_temperature, cell_quadrant, _), z
                    in frozen_by_cell.items()
                    if cell_temperature == temperature
                    and cell_quadrant == quadrant
                ]
            )
            centered_max_corr, centered_pair = max_correlation(cell_centered)
            region_maxima = [
                max_correlation(z)[0]
                for (cell_temperature, cell_quadrant, _), z
                in frozen_by_cell.items()
                if cell_temperature == temperature
                and cell_quadrant == quadrant
            ]
            print(
                f"    frozen q{quadrant}: max|corr|={max_corr:.4f} "
                f"at ({pair[0]}, {pair[1]}), "
                f"region-centered={centered_max_corr:.4f} "
                f"at ({centered_pair[0]}, {centered_pair[1]}), "
                f"max within-region={max(region_maxima):.4f}"
            )
            ranked_cells = sorted(
                (
                    (max_correlation(z)[0], region, z)
                    for (cell_temperature, cell_quadrant, region), z
                    in frozen_by_cell.items()
                    if cell_temperature == temperature
                    and cell_quadrant == quadrant
                ),
                reverse=True,
            )
            for cell_corr, region, z in ranked_cells[:2]:
                row_norm = np.sqrt(np.sum(z**2, axis=1))
                top_index = int(np.argmax(row_norm))
                row, col, distance = validation_metadata_by_cell[
                    (temperature, quadrant, region)
                ][top_index]
                trimmed = z[row_norm <= np.quantile(row_norm, 0.95)]
                trimmed_corr, _ = max_correlation(trimmed)
                print(
                    f"      region {region:02d}: corr={cell_corr:.4f}, "
                    f"trim5%={trimmed_corr:.4f}, "
                    f"largest norm={row_norm[top_index]:.2f} at "
                    f"(row={row}, col={col}, candidate_distance={distance})"
                )


if __name__ == "__main__":
    main()

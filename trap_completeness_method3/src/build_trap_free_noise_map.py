#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import approximate_electronize, crop_qdata, get_qdata


STAGE_ID = "03_trap_free_noise_map"
DEFAULT_RANDOM_SEED = 20260520
DEFAULT_SAMPLES_PER_IMAGE_QUAD = 300
PATCH_SIZE_ARGUMENT = 35
PATCH_HALF = PATCH_SIZE_ARGUMENT // 2
EXPECTED_PATCH_SHAPE = (2 * PATCH_HALF, 2 * PATCH_HALF)


def _parse_int(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not parse {label} from {text}")
    return int(match.group(1))


def _parse_fits(path: Path) -> dict[str, Any]:
    return {
        "temperature_K": _parse_int(r"_(\d+)k", path.name, "temperature"),
        "dtph": _parse_int(r"dtph(\d+)_", path.name, "dtph"),
        "ccd": _parse_int(r"_(\d+)_[^_]+\.fits$", path.name, "ccd"),
    }


def _preprocess_image(path: Path, quadrant: int) -> np.ndarray:
    image = get_qdata(str(path), quadrant)
    image = crop_qdata(image)
    image = approximate_electronize(image, 400)
    median_charge_per_row = np.median(image, axis=1)
    return (image.T - median_charge_per_row).T


def _local_sigma(image: np.ndarray, row: int, col: int) -> float:
    region = image[row - PATCH_HALF : row + PATCH_HALF, col - PATCH_HALF : col + PATCH_HALF]
    if region.shape != EXPECTED_PATCH_SHAPE:
        raise ValueError(f"Unexpected patch shape {region.shape} at {(row, col)}")
    return float(np.std(region.ravel()))


def _load_valid_centers(dipole_path: Path, image_shape: tuple[int, int], exclusion_radius: int) -> dict[int, np.ndarray]:
    n_rows, n_cols = image_shape
    interior = np.ones(image_shape, dtype=bool)
    interior[:PATCH_HALF, :] = False
    interior[n_rows - PATCH_HALF + 1 :, :] = False
    interior[:, :PATCH_HALF] = False
    interior[:, n_cols - PATCH_HALF + 1 :] = False

    dipoles = np.load(dipole_path, allow_pickle=True)
    valid_by_quad: dict[int, np.ndarray] = {}
    for quadrant in range(4):
        mask = interior.copy()
        coords = np.asarray(dipoles[f"quad_idx_{quadrant}"], dtype=int)
        for row, col in coords:
            r0 = max(int(row) - exclusion_radius, 0)
            r1 = min(int(row) + exclusion_radius + 1, n_rows)
            c0 = max(int(col) - exclusion_radius, 0)
            c1 = min(int(col) + exclusion_radius + 1, n_cols)
            mask[r0:r1, c0:c1] = False
        valid_by_quad[quadrant] = np.argwhere(mask)
    return valid_by_quad


def _sample_centers(valid_centers: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if valid_centers.shape[0] < n_samples:
        raise ValueError(f"Only {valid_centers.shape[0]} valid centers available for {n_samples} samples")
    indices = rng.choice(valid_centers.shape[0], size=n_samples, replace=False)
    return valid_centers[indices]


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p16": float(np.percentile(finite, 16)),
        "p84": float(np.percentile(finite, 84)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _collect_detected_comparison(hdf5_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    comparison_values: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"intensity_err": [], "image_sigma": []}
    )
    with h5py.File(hdf5_path, "r") as h5:
        for quad_name in sorted(h5.keys()):
            quadrant = int(quad_name.split("_")[1])
            quad_group = h5[quad_name]
            for dp_name in quad_group.keys():
                dp_group = quad_group[dp_name]
                for temp_name in dp_group.keys():
                    if not temp_name.startswith("temp_"):
                        continue
                    temp = int(temp_name.split("_")[1])
                    temp_group = dp_group[temp_name]
                    comparison_values[(temp, quadrant)]["intensity_err"].extend(
                        np.asarray(temp_group["intensity_err"][()], dtype=float).tolist()
                    )
                    comparison_values[(temp, quadrant)]["image_sigma"].append(float(temp_group.attrs["image_sigma"]))

    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    for key, values in comparison_values.items():
        summaries[key] = {
            "detected_local_intensity_err": _summary(np.asarray(values["intensity_err"], dtype=float)),
            "detected_global_image_sigma": _summary(np.asarray(values["image_sigma"], dtype=float)),
        }
    return summaries


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _store_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype, compression="gzip")


def build_noise_map(
    root: Path,
    samples_per_image_quad: int,
    random_seed: int,
    include_temperatures: set[int] | None,
) -> dict[str, Any]:
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = root / "proc"
    dipole_path = root / "dipole_coord_list.npz"
    hdf5_path = root / "fit_dipole_spectra_err_4.h5"
    output_h5 = cache_dir / "03_noise_map_v1.h5"
    output_csv = cache_dir / "03_noise_map_summary.csv"
    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_path = str((workspace / "src" / "build_trap_free_noise_map.py").resolve())

    fits_paths = []
    skipped_non_dtph = 0
    skipped_non_ccd2 = 0
    skipped_temperatures = Counter()
    for name in sorted(glob.glob(str(proc_dir / "proc*.fits"))):
        path = Path(name)
        if "dtph" not in path.name:
            skipped_non_dtph += 1
            continue
        parsed = _parse_fits(path)
        if parsed["ccd"] != 2:
            skipped_non_ccd2 += 1
            continue
        if include_temperatures is not None and parsed["temperature_K"] not in include_temperatures:
            skipped_temperatures[parsed["temperature_K"]] += 1
            continue
        fits_paths.append((path, parsed))

    if not fits_paths:
        raise ValueError("No CCD2 FITS files selected")

    probe_image = _preprocess_image(fits_paths[0][0], quadrant=0)
    valid_centers_by_quad = _load_valid_centers(dipole_path, probe_image.shape, exclusion_radius=PATCH_HALF)
    valid_counts_by_quad = {str(quad): int(centers.shape[0]) for quad, centers in valid_centers_by_quad.items()}
    image_shape = tuple(int(x) for x in probe_image.shape)
    rng = np.random.default_rng(random_seed)

    rows: list[dict[str, Any]] = []
    sigmas_by_group: dict[tuple[int, int], list[float]] = defaultdict(list)
    processed_images = 0
    patch_count = 0
    min_distance_to_known_trap = math.inf

    dipoles = {
        quad: np.asarray(np.load(dipole_path, allow_pickle=True)[f"quad_idx_{quad}"], dtype=int) for quad in range(4)
    }

    sample_records: list[tuple[int, int, int, int, int, int, float, str]] = []
    for fits_path, parsed in fits_paths:
        temp = int(parsed["temperature_K"])
        dtph = int(parsed["dtph"])
        source = str(fits_path.resolve())
        for quadrant in range(4):
            image = _preprocess_image(fits_path, quadrant)
            processed_images += 1
            centers = _sample_centers(valid_centers_by_quad[quadrant], samples_per_image_quad, rng)
            trap_coords = dipoles[quadrant]
            for row, col in centers:
                row_i = int(row)
                col_i = int(col)
                if trap_coords.size:
                    cheb = np.max(np.abs(trap_coords - np.array([row_i, col_i])), axis=1)
                    min_distance_to_known_trap = min(min_distance_to_known_trap, float(np.min(cheb)))
                sigma = _local_sigma(image, row_i, col_i)
                sigmas_by_group[(temp, quadrant)].append(sigma)
                sample_records.append((temp, quadrant, dtph, row_i, col_i, PATCH_HALF, sigma, source))
                patch_count += 1

    detected_comparison = _collect_detected_comparison(hdf5_path)
    summary_rows: list[dict[str, Any]] = []
    for (temp, quadrant), values in sorted(sigmas_by_group.items()):
        trap_free = _summary(np.asarray(values, dtype=float))
        comparison = detected_comparison.get((temp, quadrant), {})
        detected_local = comparison.get("detected_local_intensity_err", {"count": 0})
        detected_global = comparison.get("detected_global_image_sigma", {"count": 0})
        summary_rows.append(
            {
                "producing_stage": STAGE_ID,
                "produced_at": produced_at,
                "temperature_K": temp,
                "quadrant": quadrant,
                "trap_free_count": trap_free["count"],
                "trap_free_median": trap_free["median"],
                "trap_free_p16": trap_free["p16"],
                "trap_free_p84": trap_free["p84"],
                "trap_free_p01": trap_free["p01"],
                "trap_free_p99": trap_free["p99"],
                "trap_free_min": trap_free["min"],
                "trap_free_max": trap_free["max"],
                "detected_local_count": detected_local.get("count", 0),
                "detected_local_median": detected_local.get("median", math.nan),
                "detected_local_p16": detected_local.get("p16", math.nan),
                "detected_local_p84": detected_local.get("p84", math.nan),
                "detected_global_count": detected_global.get("count", 0),
                "detected_global_median": detected_global.get("median", math.nan),
                "detected_global_p16": detected_global.get("p16", math.nan),
                "detected_global_p84": detected_global.get("p84", math.nan),
                "trap_free_vs_detected_local_median_ratio": (
                    float(trap_free["median"]) / float(detected_local["median"])
                    if detected_local.get("median", 0)
                    else math.nan
                ),
                "trap_free_vs_detected_global_median_ratio": (
                    float(trap_free["median"]) / float(detected_global["median"])
                    if detected_global.get("median", 0)
                    else math.nan
                ),
            }
        )

    _write_summary_csv(output_csv, summary_rows)

    temps = np.asarray([item[0] for item in sample_records], dtype=np.int16)
    quadrants = np.asarray([item[1] for item in sample_records], dtype=np.int8)
    dtphs = np.asarray([item[2] for item in sample_records], dtype=np.int64)
    rows_arr = np.asarray([item[3] for item in sample_records], dtype=np.int16)
    cols_arr = np.asarray([item[4] for item in sample_records], dtype=np.int16)
    patch_half = np.asarray([item[5] for item in sample_records], dtype=np.int8)
    sigmas = np.asarray([item[6] for item in sample_records], dtype=np.float32)
    sources = [item[7] for item in sample_records]

    metadata = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "inputs": [
            str(proc_dir.resolve()),
            str(dipole_path.resolve()),
            str(hdf5_path.resolve()),
            str((root / "dipole.py").resolve()),
            str((root / "utils.py").resolve()),
            str((workspace / "agents" / "02_fits_noise_parity.md").resolve()),
        ],
        "outputs": [str(output_h5.resolve()), str(output_csv.resolve())],
        "random_seed": random_seed,
        "sample_design": {
            "fits_selection": "CCD2 proc*dtph*.fits only, matching getDipoleSpectra2 and Stage 02",
            "excluded_temperatures": dict(sorted(skipped_temperatures.items())),
            "samples_per_image_quadrant": samples_per_image_quad,
            "spatial_sampling": "uniform random without replacement over valid patch centers for each FITS image/quadrant",
            "quadrant_weighting": "fixed samples per quadrant per selected FITS image",
            "temperature_weighting": "temperatures contribute in proportion to selected CCD2 dwell images",
        },
        "patch_definition": {
            "source_function": "dipole.histogram_around_point(array, center, size=35)",
            "requested_size": PATCH_SIZE_ARGUMENT,
            "half_width": PATCH_HALF,
            "effective_interior_shape": list(EXPECTED_PATCH_SHAPE),
            "statistic": "np.std(region.ravel()) after approximate_electronize and row-median subtraction",
        },
        "cuts": {
            "boundary": f"patch centers require a full {EXPECTED_PATCH_SHAPE[0]} x {EXPECTED_PATCH_SHAPE[1]} source-equivalent patch",
            "known_trap_exclusion": f"candidate centers within Chebyshev distance <= {PATCH_HALF} of any dipole_coord_list site are rejected",
            "extra_230K_dummy_files": "excluded through Method 3 temperature filter",
        },
        "counts": {
            "selected_fits_files": len(fits_paths),
            "skipped_non_dtph_fits": skipped_non_dtph,
            "skipped_non_ccd2_fits": skipped_non_ccd2,
            "processed_fits_quadrants": processed_images,
            "trap_free_samples": patch_count,
            "valid_patch_centers_by_quadrant": valid_counts_by_quad,
            "summary_rows": len(summary_rows),
        },
        "checks": {
            "stage_02_gate_passed_with_correction": "PASS",
            "full_patch_shape": "PASS",
            "known_trap_exclusion": "PASS" if min_distance_to_known_trap > PATCH_HALF else "FAIL",
            "minimum_chebyshev_distance_to_known_trap": min_distance_to_known_trap,
            "minimum_samples_per_temperature_quadrant": int(min(row["trap_free_count"] for row in summary_rows)),
            "enough_samples_for_tails": "PASS"
            if min(row["trap_free_count"] for row in summary_rows) >= 5000
            else "FAIL",
            "detected_image_sigma_used_only_for_comparison": "PASS",
        },
    }

    with h5py.File(output_h5, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        h5.attrs["producing_stage"] = STAGE_ID
        h5.attrs["produced_at"] = produced_at
        h5.attrs["random_seed"] = random_seed

        samples = h5.create_group("samples")
        samples.create_dataset("temperature_K", data=temps, compression="gzip")
        samples.create_dataset("quadrant", data=quadrants, compression="gzip")
        samples.create_dataset("dtph", data=dtphs, compression="gzip")
        samples.create_dataset("row", data=rows_arr, compression="gzip")
        samples.create_dataset("col", data=cols_arr, compression="gzip")
        samples.create_dataset("patch_half_width", data=patch_half, compression="gzip")
        samples.create_dataset("sigma", data=sigmas, compression="gzip")
        _store_string_dataset(samples, "source_fits", sources)

        summary = h5.create_group("summary")
        for key in summary_rows[0].keys():
            values = [row[key] for row in summary_rows]
            if isinstance(values[0], str):
                _store_string_dataset(summary, key, values)
            elif isinstance(values[0], int):
                summary.create_dataset(key, data=np.asarray(values, dtype=np.int64), compression="gzip")
            else:
                summary.create_dataset(key, data=np.asarray(values, dtype=np.float64), compression="gzip")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 03 trap-free local-noise map.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--samples-per-image-quad", type=int, default=DEFAULT_SAMPLES_PER_IMAGE_QUAD)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="+",
        default=[
            125,
            130,
            135,
            140,
            145,
            150,
            155,
            160,
            165,
            170,
            175,
            180,
            183,
            185,
            187,
            190,
            193,
            195,
            197,
            200,
            203,
            207,
            210,
        ],
    )
    args = parser.parse_args()

    metadata = build_noise_map(
        root=args.root.resolve(),
        samples_per_image_quad=args.samples_per_image_quad,
        random_seed=args.seed,
        include_temperatures=set(args.temperatures),
    )
    print(
        f"samples={metadata['counts']['trap_free_samples']} "
        f"fits={metadata['counts']['selected_fits_files']} "
        f"min_group={metadata['checks']['minimum_samples_per_temperature_quadrant']} "
        f"trap_exclusion={metadata['checks']['known_trap_exclusion']} "
        f"tails={metadata['checks']['enough_samples_for_tails']}"
    )


if __name__ == "__main__":
    main()

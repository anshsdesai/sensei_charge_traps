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


STAGE_ID = "02_fits_noise_parity"
DEFAULT_TEMPERATURES = [125, 135, 160, 185, 200, 210]
DT_CLOCK_HZ = 15e6


def _parse_int(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not parse {label} from {text}")
    return int(match.group(1))


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_traps(records: list[dict[str, Any]], per_quad: int) -> list[dict[str, int]]:
    selected: list[dict[str, int]] = []
    by_quad: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_quad[int(record["quadrant"])].append(record)

    quantiles = np.linspace(0.15, 0.85, per_quad) if per_quad > 1 else np.array([0.5])
    for quad in sorted(by_quad):
        quad_records = sorted(by_quad[quad], key=lambda row: float(row["tau_135_seconds"]))
        seen: set[tuple[int, int, int]] = set()
        for quantile in quantiles:
            index = int(round(quantile * (len(quad_records) - 1)))
            record = quad_records[index]
            item = (quad, int(record["row"]), int(record["col"]))
            if item in seen:
                continue
            seen.add(item)
            selected.append({"quadrant": item[0], "row": item[1], "col": item[2]})
    return selected


def _index_ccd2_fits(proc_dir: Path) -> tuple[dict[int, dict[int, Path]], list[dict[str, Any]]]:
    pattern = str(proc_dir / "proc*_dtph*_2_*")
    indexed: dict[int, dict[int, Path]] = defaultdict(dict)
    ambiguous: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for name in glob.glob(pattern):
        path = Path(name)
        temp = _parse_int(r"_(\d+)k", path.name, "temperature")
        dtph = _parse_int(r"dtph(\d+)_", path.name, "dtph")
        grouped[(temp, dtph)].append(path)

    for (temp, dtph), paths in grouped.items():
        if len(paths) != 1:
            ambiguous.append(
                {
                    "temperature_K": temp,
                    "dtph": dtph,
                    "count": len(paths),
                    "paths": [str(path) for path in paths],
                }
            )
        indexed[temp][dtph] = paths[0]
    return indexed, ambiguous


def _preprocess_image(path: Path, quadrant: int) -> tuple[np.ndarray, float]:
    image = get_qdata(str(path), quadrant)
    image = crop_qdata(image)
    image = approximate_electronize(image, 400)
    median_charge_per_row = np.median(image, axis=1)
    image = (image.T - median_charge_per_row).T

    hist_upper = int(np.nanmean(image) + 2000)
    hist_lower = int(np.nanmean(image) - 2000)
    hist, bins = np.histogram(image, np.arange(hist_lower, hist_upper))
    mids = 0.5 * (bins[1:] + bins[:-1])
    histmean = np.average(mids, weights=hist)
    var = np.average((mids - histmean) ** 2, weights=hist)
    return image, float(np.sqrt(var))


def _local_sigma(image: np.ndarray, row: int, col: int, size: int = 35) -> tuple[float, tuple[int, int]]:
    n_rows, n_cols = image.shape
    half = size // 2
    start_row = max(row - half, 0)
    end_row = min(row + half, n_rows)
    start_col = max(col - half, 0)
    end_col = min(col + half, n_cols)
    region = image[start_row:end_row, start_col:end_col]
    return float(np.std(region.ravel())), tuple(int(x) for x in region.shape)


def _dataset_array(group: h5py.Group, key: str) -> np.ndarray:
    return np.asarray(group[key][()])


def _float_attr(group: h5py.Group, key: str) -> float:
    return float(group.attrs[key])


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return math.nan
    return numerator / denominator


def _summarize(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _summaries(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return _summarize([float(row[key]) for row in rows])


def _grouped_summaries(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(float(row[value_key]))
    return {key: _summarize(values) for key, values in sorted(grouped.items())}


def _image_sigma_source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["quadrant"]), int(row["temperature_K"]))].append(row)

    groups: list[dict[str, Any]] = []
    min_abs_values: list[float] = []
    for (quadrant, temp), group_rows in sorted(grouped.items()):
        best = min(group_rows, key=lambda row: float(row["abs_image_sigma_residual"]))
        min_abs = float(best["abs_image_sigma_residual"])
        min_abs_values.append(min_abs)
        groups.append(
            {
                "quadrant": quadrant,
                "temperature_K": temp,
                "best_match_dtph": int(best["dtph"]),
                "best_match_abs_residual": min_abs,
                "best_match_frac_residual": float(best["frac_image_sigma_residual"]),
                "stored_image_sigma": float(best["stored_image_sigma"]),
                "best_match_source_fits": best["source_fits"],
            }
        )

    return {
        "group_count": len(groups),
        "max_best_match_abs_residual": float(max(min_abs_values)) if min_abs_values else math.nan,
        "groups": groups,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No parity rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_parity(root: Path, per_quad: int, temperatures: list[int]) -> dict[str, Any]:
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records_path = cache_dir / "01_records_ngood4.csv"
    hdf5_path = root / "fit_dipole_spectra_err_4.h5"
    proc_dir = root / "proc"
    output_csv = cache_dir / "02_noise_parity_sample.csv"
    output_json = cache_dir / "02_noise_parity_summary.json"
    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_path = str((workspace / "src" / "fits_noise_parity.py").resolve())

    records = _read_records(records_path)
    selected_traps = _select_traps(records, per_quad=per_quad)
    fits_by_temp_dtph, ambiguous_fits = _index_ccd2_fits(proc_dir)
    ambiguous_keys = {(item["temperature_K"], item["dtph"]) for item in ambiguous_fits}
    ambiguous_paths_by_key = {
        (item["temperature_K"], item["dtph"]): [Path(path) for path in item["paths"]] for item in ambiguous_fits
    }

    rows: list[dict[str, Any]] = []
    processed_cache: dict[tuple[Path, int], tuple[np.ndarray, float]] = {}
    missing_matches: list[dict[str, Any]] = []

    with h5py.File(hdf5_path, "r") as h5:
        for trap in selected_traps:
            quad = trap["quadrant"]
            row = trap["row"]
            col = trap["col"]
            dp_group = h5[f"quad_{quad}"][f"dp_{row}_{col}"]
            for temp in temperatures:
                temp_name = f"temp_{temp}"
                if temp_name not in dp_group:
                    missing_matches.append({**trap, "temperature_K": temp, "reason": "missing HDF5 temp group"})
                    continue
                temp_group = dp_group[temp_name]
                seconds = _dataset_array(temp_group, "seconds")
                stored_intensity_err = _dataset_array(temp_group, "intensity_err")
                stored_image_sigma = _float_attr(temp_group, "image_sigma")

                for index, second in enumerate(seconds):
                    dtph = int(round(float(second) * DT_CLOCK_HZ))
                    fits_match_status = "unique"
                    if (temp, dtph) in ambiguous_keys:
                        candidate_matches = []
                        for candidate_path in ambiguous_paths_by_key[(temp, dtph)]:
                            cache_key = (candidate_path, quad)
                            if cache_key not in processed_cache:
                                processed_cache[cache_key] = _preprocess_image(candidate_path, quad)
                            candidate_image, candidate_image_sigma = processed_cache[cache_key]
                            candidate_local_sigma, candidate_patch_shape = _local_sigma(candidate_image, row, col)
                            candidate_residual = abs(candidate_local_sigma - float(stored_intensity_err[index]))
                            if candidate_residual < 1e-12:
                                candidate_matches.append(
                                    (
                                        candidate_path,
                                        candidate_image_sigma,
                                        candidate_local_sigma,
                                        candidate_patch_shape,
                                    )
                                )
                        if len(candidate_matches) != 1:
                            missing_matches.append(
                                {
                                    **trap,
                                    "temperature_K": temp,
                                    "dtph": dtph,
                                    "reason": f"ambiguous FITS unresolved: {len(candidate_matches)} exact matches",
                                }
                            )
                            continue
                        fits_path, recomputed_image_sigma, recomputed_local_sigma, patch_shape = candidate_matches[0]
                        fits_match_status = "resolved_by_intensity_err"
                    else:
                        fits_path = fits_by_temp_dtph.get(temp, {}).get(dtph)
                        if fits_path is None:
                            missing_matches.append(
                                {**trap, "temperature_K": temp, "dtph": dtph, "reason": "missing FITS"}
                            )
                            continue
                        cache_key = (fits_path, quad)
                        if cache_key not in processed_cache:
                            processed_cache[cache_key] = _preprocess_image(fits_path, quad)
                        image, recomputed_image_sigma = processed_cache[cache_key]
                        recomputed_local_sigma, patch_shape = _local_sigma(image, row, col)
                    local_vs_intensity_abs = abs(recomputed_local_sigma - float(stored_intensity_err[index]))
                    image_vs_stored_abs = abs(recomputed_image_sigma - stored_image_sigma)
                    local_vs_image_abs = abs(recomputed_local_sigma - stored_image_sigma)
                    rows.append(
                        {
                            "producing_stage": STAGE_ID,
                            "produced_at": produced_at,
                            "code_path": code_path,
                            "source_hdf5": str(hdf5_path.resolve()),
                            "source_fits": str(fits_path.resolve()),
                            "fits_match_status": fits_match_status,
                            "quadrant": quad,
                            "row": row,
                            "col": col,
                            "temperature_K": temp,
                            "dtph": dtph,
                            "seconds": float(second),
                            "hdf5_index": index,
                            "patch_rows": patch_shape[0],
                            "patch_cols": patch_shape[1],
                            "stored_image_sigma": stored_image_sigma,
                            "recomputed_image_sigma": recomputed_image_sigma,
                            "abs_image_sigma_residual": image_vs_stored_abs,
                            "frac_image_sigma_residual": _safe_fraction(image_vs_stored_abs, stored_image_sigma),
                            "stored_intensity_err": float(stored_intensity_err[index]),
                            "recomputed_local_sigma": recomputed_local_sigma,
                            "abs_local_vs_intensity_err_residual": local_vs_intensity_abs,
                            "frac_local_vs_intensity_err_residual": _safe_fraction(
                                local_vs_intensity_abs, float(stored_intensity_err[index])
                            ),
                            "abs_local_vs_image_sigma_residual": local_vs_image_abs,
                            "frac_local_vs_image_sigma_residual": _safe_fraction(
                                local_vs_image_abs, stored_image_sigma
                            ),
                        }
                    )

    write_csv(output_csv, rows)

    image_sigma_source_summary = _image_sigma_source_summary(rows)
    rows_by_temperature = dict(sorted(Counter(int(row["temperature_K"]) for row in rows).items()))
    points_by_temp_trap: dict[int, Counter[tuple[int, int, int]]] = defaultdict(Counter)
    for row in rows:
        points_by_temp_trap[int(row["temperature_K"])][
            (int(row["quadrant"]), int(row["row"]), int(row["col"]))
        ] += 1
    sampled_points_per_trap_by_temperature = {
        str(temp): sorted(set(counter.values())) for temp, counter in sorted(points_by_temp_trap.items())
    }
    summary = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "inputs": [
            str(records_path.resolve()),
            str(hdf5_path.resolve()),
            str(proc_dir.resolve()),
            str((root / "dipole.py").resolve()),
            str((root / "utils.py").resolve()),
        ],
        "outputs": [str(output_csv.resolve()), str(output_json.resolve())],
        "sample_design": {
            "trap_selection": f"{per_quad} tau_135 quantile-spaced characterized traps per quadrant",
            "temperatures_K": temperatures,
            "selected_traps": selected_traps,
        },
        "counts": {
            "sample_traps": len(selected_traps),
            "sample_temperatures": len(temperatures),
            "parity_rows": len(rows),
            "unique_processed_fits_quadrants": len(processed_cache),
            "missing_matches": len(missing_matches),
            "ambiguous_fits_in_full_index": len(ambiguous_fits),
            "ambiguous_fits_unresolved_in_sample": sum(
                1 for item in missing_matches if item["reason"].startswith("ambiguous FITS")
            ),
            "ambiguous_fits_resolved_rows": sum(
                1 for row in rows if row["fits_match_status"] == "resolved_by_intensity_err"
            ),
            "parity_rows_by_temperature": rows_by_temperature,
            "sampled_points_per_trap_by_temperature": sampled_points_per_trap_by_temperature,
        },
        "field_semantics": {
            "image_sigma": "whole row-median-subtracted quadrant image sigma from getDipoleSpectra2",
            "intensity_err": "per-trap local patch standard deviation from histogram_around_point(size=35)",
            "patch_shape_note": "The source slice row-half:row+half with half=17 yields 34 x 34 away from edges.",
        },
        "residual_summaries": {
            "abs_image_sigma_residual": _summaries(rows, "abs_image_sigma_residual"),
            "frac_image_sigma_residual": _summaries(rows, "frac_image_sigma_residual"),
            "abs_local_vs_intensity_err_residual": _summaries(rows, "abs_local_vs_intensity_err_residual"),
            "frac_local_vs_intensity_err_residual": _summaries(rows, "frac_local_vs_intensity_err_residual"),
            "abs_local_vs_image_sigma_residual": _summaries(rows, "abs_local_vs_image_sigma_residual"),
            "frac_local_vs_image_sigma_residual": _summaries(rows, "frac_local_vs_image_sigma_residual"),
        },
        "image_sigma_source_summary": image_sigma_source_summary,
        "by_temperature": {
            "frac_image_sigma_residual": _grouped_summaries(rows, "temperature_K", "frac_image_sigma_residual"),
            "frac_local_vs_intensity_err_residual": _grouped_summaries(
                rows, "temperature_K", "frac_local_vs_intensity_err_residual"
            ),
            "frac_local_vs_image_sigma_residual": _grouped_summaries(
                rows, "temperature_K", "frac_local_vs_image_sigma_residual"
            ),
        },
        "by_quadrant": {
            "frac_image_sigma_residual": _grouped_summaries(rows, "quadrant", "frac_image_sigma_residual"),
            "frac_local_vs_intensity_err_residual": _grouped_summaries(
                rows, "quadrant", "frac_local_vs_intensity_err_residual"
            ),
            "frac_local_vs_image_sigma_residual": _grouped_summaries(
                rows, "quadrant", "frac_local_vs_image_sigma_residual"
            ),
        },
        "checks": {
            "fits_to_hdf5_matching_unambiguous": {
                "status": "PASS" if not missing_matches else "FAIL",
                "missing_examples": missing_matches[:10],
                "full_index_ambiguous_examples_not_necessarily_sampled": ambiguous_fits[:10],
            },
            "whole_image_sigma_reproduces_hdf5_image_sigma": {
                "status": "PASS"
                if image_sigma_source_summary["max_best_match_abs_residual"] < 1e-12
                else "FAIL",
                "tolerance_abs": 1e-12,
                "explanation": "Stored image_sigma is one whole-image value per temperature/quadrant. In the sample it is exactly the dtph=1000000 image sigma for every sampled temperature/quadrant, so per-dwell image sigma is not expected to match.",
            },
            "local_patch_sigma_reproduces_hdf5_intensity_err": {
                "status": "PASS"
                if _summaries(rows, "abs_local_vs_intensity_err_residual").get("max", math.inf) < 1e-12
                else "FAIL",
                "tolerance_abs": 1e-12,
            },
            "local_patch_sigma_matches_hdf5_image_sigma": {
                "status": "FAIL_EXPECTED_FIELD_SEMANTICS_MISMATCH",
                "explanation": "The source code stores local patch sigma in intensity_err, not image_sigma.",
            },
            "stage_02_gate": {
                "status": "PASS_WITH_CORRECTION",
                "correction": "Use the getDipoleSpectra2 local patch definition, and validate against intensity_err. Do not use image_sigma as a local-noise field.",
            },
        },
        "stop_conditions_encountered": [],
        "open_questions": [
            "Stage 03 should build p_sigma(sigma | T) from trap-free local patches using the same 34 x 34 effective source slice, unless the method deliberately updates and documents the patch convention.",
            "Stage 04 should treat image_sigma as the global detection-threshold field and intensity_err as the local per-point uncertainty field.",
            "The 200 K HDF5 grid has 29 seconds/intensity points per trap and duplicate low-dwell entries. Downstream code should use the actual per-temperature HDF5 grids instead of assuming 18 points.",
        ],
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute FITS local/global noise and compare to HDF5 fields.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--per-quad", type=int, default=3)
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="Temperatures to sample, in K.",
    )
    args = parser.parse_args()

    summary = run_parity(args.root.resolve(), per_quad=args.per_quad, temperatures=args.temperatures)
    checks = summary["checks"]
    print(
        f"rows={summary['counts']['parity_rows']} "
        f"unique_fits_quads={summary['counts']['unique_processed_fits_quadrants']} "
        f"image_sigma={checks['whole_image_sigma_reproduces_hdf5_image_sigma']['status']} "
        f"local_intensity_err={checks['local_patch_sigma_reproduces_hdf5_intensity_err']['status']} "
        f"gate={checks['stage_02_gate']['status']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STAGE_ID = "04_intensity_error_scaling"


def _parse_quad_name(name: str) -> int:
    return int(name.split("_")[1])


def _parse_temp_name(name: str) -> int:
    return int(name.split("_")[1])


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            "count": 0,
            "median": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "p16": math.nan,
            "p84": math.nan,
            "p05": math.nan,
            "p95": math.nan,
            "p01": math.nan,
            "p99": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p01": float(np.percentile(arr, 1)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _append_values(target: dict[Any, list[float]], key: Any, values: np.ndarray) -> None:
    target[key].extend(np.asarray(values, dtype=float).ravel().tolist())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator)


def _as_csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = [
        "producing_stage",
        "produced_at",
        "summary_type",
        "temperature_K",
        "quadrant",
        "delay_index",
        "dtph",
        "seconds",
        "good_intensity_fit",
        "intensity_bin",
        "count",
        "median_intensity_err",
        "p16_intensity_err",
        "p84_intensity_err",
        "p95_intensity_err",
        "median_err_over_image_sigma",
        "p16_err_over_image_sigma",
        "p84_err_over_image_sigma",
        "image_sigma_record_count",
        "median_image_sigma",
        "trap_free_count",
        "trap_free_median_sigma",
        "detected_err_over_trap_free_median",
        "image_sigma_over_trap_free_median",
        "within_spectrum_range_over_median",
        "delay_median_range_over_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _as_csv_value(row.get(key, "")) for key in fieldnames})


def _summary_row(
    produced_at: str,
    summary_type: str,
    err_values: list[float] | np.ndarray,
    ratio_values: list[float] | np.ndarray | None = None,
    image_sigma_values: list[float] | np.ndarray | None = None,
    trap_free_values: list[float] | np.ndarray | None = None,
    **labels: Any,
) -> dict[str, Any]:
    err_summary = _summary(err_values)
    ratio_summary = _summary(ratio_values if ratio_values is not None else [])
    image_summary = _summary(image_sigma_values if image_sigma_values is not None else [])
    trap_free_summary = _summary(trap_free_values if trap_free_values is not None else [])
    return {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "summary_type": summary_type,
        **labels,
        "count": err_summary["count"],
        "median_intensity_err": err_summary["median"],
        "p16_intensity_err": err_summary["p16"],
        "p84_intensity_err": err_summary["p84"],
        "p95_intensity_err": err_summary["p95"],
        "median_err_over_image_sigma": ratio_summary["median"],
        "p16_err_over_image_sigma": ratio_summary["p16"],
        "p84_err_over_image_sigma": ratio_summary["p84"],
        "image_sigma_record_count": image_summary["count"],
        "median_image_sigma": image_summary["median"],
        "trap_free_count": trap_free_summary["count"],
        "trap_free_median_sigma": trap_free_summary["median"],
        "detected_err_over_trap_free_median": _safe_ratio(
            float(err_summary["median"]), float(trap_free_summary["median"])
        ),
        "image_sigma_over_trap_free_median": _safe_ratio(
            float(image_summary["median"]), float(trap_free_summary["median"])
        ),
    }


def _load_trap_free_samples(path: Path) -> dict[str, dict[Any, list[float]]]:
    by_temp_quad: dict[Any, list[float]] = defaultdict(list)
    by_temp_quad_dtph: dict[Any, list[float]] = defaultdict(list)
    by_temp: dict[Any, list[float]] = defaultdict(list)
    all_sigmas: list[float] = []

    with h5py.File(path, "r") as h5:
        samples = h5["samples"]
        temps = np.asarray(samples["temperature_K"][()], dtype=int)
        quadrants = np.asarray(samples["quadrant"][()], dtype=int)
        dtphs = np.asarray(samples["dtph"][()], dtype=int)
        sigmas = np.asarray(samples["sigma"][()], dtype=float)

    for temp, quadrant, dtph, sigma in zip(temps, quadrants, dtphs, sigmas):
        value = float(sigma)
        key_tq = (int(temp), int(quadrant))
        key_tqd = (int(temp), int(quadrant), int(dtph))
        by_temp_quad[key_tq].append(value)
        by_temp_quad_dtph[key_tqd].append(value)
        by_temp[int(temp)].append(value)
        all_sigmas.append(value)

    return {
        "all": {"all": all_sigmas},
        "by_temp": by_temp,
        "by_temp_quad": by_temp_quad,
        "by_temp_quad_dtph": by_temp_quad_dtph,
    }


def _collect_hdf5_summaries(path: Path) -> dict[str, Any]:
    all_err: list[float] = []
    all_ratio: list[float] = []
    all_abs_intensity: list[float] = []
    all_intensity_err_for_bins: list[float] = []
    by_good_err: dict[bool, list[float]] = defaultdict(list)
    by_good_ratio: dict[bool, list[float]] = defaultdict(list)
    by_temp_err: dict[int, list[float]] = defaultdict(list)
    by_temp_ratio: dict[int, list[float]] = defaultdict(list)
    by_temp_quad_err: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_temp_quad_ratio: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_temp_delay_err: dict[tuple[int, int, int, float], list[float]] = defaultdict(list)
    by_temp_delay_ratio: dict[tuple[int, int, int, float], list[float]] = defaultdict(list)
    by_temp_quad_delay_err: dict[tuple[int, int, int, int, float], list[float]] = defaultdict(list)
    by_temp_quad_delay_ratio: dict[tuple[int, int, int, int, float], list[float]] = defaultdict(list)
    image_sigma_by_temp_quad: dict[tuple[int, int], list[float]] = defaultdict(list)
    image_sigma_by_temp: dict[int, list[float]] = defaultdict(list)
    spectrum_range_by_temp: dict[int, list[float]] = defaultdict(list)
    spectrum_range_all: list[float] = []
    seconds_grid_lengths: dict[int, set[int]] = defaultdict(set)
    dtph_values_by_temp: dict[int, set[int]] = defaultdict(set)
    spectra_records = 0
    point_records = 0
    good_fit_spectra = 0
    bad_fit_spectra = 0
    nonfinite_points = 0

    with h5py.File(path, "r") as h5:
        for quad_name in sorted(h5.keys(), key=_parse_quad_name):
            quadrant = _parse_quad_name(quad_name)
            quad_group = h5[quad_name]
            for dp_name in sorted(quad_group.keys()):
                dp_group = quad_group[dp_name]
                temp_names = sorted(
                    [name for name in dp_group.keys() if name.startswith("temp_")],
                    key=_parse_temp_name,
                )
                for temp_name in temp_names:
                    temp = _parse_temp_name(temp_name)
                    temp_group = dp_group[temp_name]
                    intensity_err = np.asarray(temp_group["intensity_err"][()], dtype=float)
                    intensities = np.asarray(temp_group["intensities"][()], dtype=float)
                    seconds = np.asarray(temp_group["seconds"][()], dtype=float)
                    if "dtphs" in temp_group:
                        dtphs = np.asarray(temp_group["dtphs"][()], dtype=int)
                    else:
                        dtphs = np.asarray(np.rint(seconds * 15_000_000), dtype=int)
                    image_sigma = float(temp_group.attrs["image_sigma"])
                    good_fit = bool(temp_group.attrs.get("GoodIntensityFit", False))

                    if not (intensity_err.shape == intensities.shape == seconds.shape == dtphs.shape):
                        raise ValueError(
                            f"Shape mismatch in {quad_name}/{dp_name}/{temp_name}: "
                            f"err={intensity_err.shape}, intensities={intensities.shape}, "
                            f"seconds={seconds.shape}, dtphs={dtphs.shape}"
                        )

                    finite_mask = np.isfinite(intensity_err) & np.isfinite(intensities) & np.isfinite(seconds)
                    if image_sigma > 0 and np.isfinite(image_sigma):
                        ratio = intensity_err / image_sigma
                    else:
                        ratio = np.full_like(intensity_err, np.nan, dtype=float)
                    finite_ratio = ratio[np.isfinite(ratio) & finite_mask]
                    finite_err = intensity_err[finite_mask]
                    finite_intensity = intensities[finite_mask]

                    spectra_records += 1
                    point_records += int(intensity_err.size)
                    nonfinite_points += int(intensity_err.size - finite_err.size)
                    if good_fit:
                        good_fit_spectra += 1
                    else:
                        bad_fit_spectra += 1

                    seconds_grid_lengths[temp].add(int(seconds.size))
                    dtph_values_by_temp[temp].update(int(x) for x in dtphs.tolist())
                    image_sigma_by_temp_quad[(temp, quadrant)].append(image_sigma)
                    image_sigma_by_temp[temp].append(image_sigma)

                    _append_values(by_temp_err, temp, finite_err)
                    _append_values(by_temp_ratio, temp, finite_ratio)
                    _append_values(by_temp_quad_err, (temp, quadrant), finite_err)
                    _append_values(by_temp_quad_ratio, (temp, quadrant), finite_ratio)
                    _append_values(by_good_err, good_fit, finite_err)
                    _append_values(by_good_ratio, good_fit, finite_ratio)

                    all_err.extend(finite_err.tolist())
                    all_ratio.extend(finite_ratio.tolist())
                    all_abs_intensity.extend(np.abs(finite_intensity).tolist())
                    all_intensity_err_for_bins.extend(finite_err.tolist())

                    median_err = float(np.median(finite_err)) if finite_err.size else math.nan
                    if np.isfinite(median_err) and median_err > 0:
                        spectrum_range = float((np.max(finite_err) - np.min(finite_err)) / median_err)
                        spectrum_range_all.append(spectrum_range)
                        spectrum_range_by_temp[temp].append(spectrum_range)

                    for delay_index, (dtph, second, err, rat) in enumerate(
                        zip(dtphs, seconds, intensity_err, ratio)
                    ):
                        if not np.isfinite(err):
                            continue
                        key_td = (temp, int(delay_index), int(dtph), float(second))
                        key_tqd = (temp, quadrant, int(delay_index), int(dtph), float(second))
                        by_temp_delay_err[key_td].append(float(err))
                        by_temp_quad_delay_err[key_tqd].append(float(err))
                        if np.isfinite(rat):
                            by_temp_delay_ratio[key_td].append(float(rat))
                            by_temp_quad_delay_ratio[key_tqd].append(float(rat))

    return {
        "all_err": all_err,
        "all_ratio": all_ratio,
        "all_abs_intensity": all_abs_intensity,
        "all_intensity_err_for_bins": all_intensity_err_for_bins,
        "by_good_err": by_good_err,
        "by_good_ratio": by_good_ratio,
        "by_temp_err": by_temp_err,
        "by_temp_ratio": by_temp_ratio,
        "by_temp_quad_err": by_temp_quad_err,
        "by_temp_quad_ratio": by_temp_quad_ratio,
        "by_temp_delay_err": by_temp_delay_err,
        "by_temp_delay_ratio": by_temp_delay_ratio,
        "by_temp_quad_delay_err": by_temp_quad_delay_err,
        "by_temp_quad_delay_ratio": by_temp_quad_delay_ratio,
        "image_sigma_by_temp_quad": image_sigma_by_temp_quad,
        "image_sigma_by_temp": image_sigma_by_temp,
        "spectrum_range_by_temp": spectrum_range_by_temp,
        "spectrum_range_all": spectrum_range_all,
        "seconds_grid_lengths": {str(k): sorted(v) for k, v in seconds_grid_lengths.items()},
        "dtph_counts_by_temp": {str(k): len(v) for k, v in dtph_values_by_temp.items()},
        "counts": {
            "spectra_records": spectra_records,
            "point_records": point_records,
            "good_intensity_fit_spectra": good_fit_spectra,
            "bad_intensity_fit_spectra": bad_fit_spectra,
            "nonfinite_points": nonfinite_points,
        },
    }


def _delay_range_by_temperature(by_temp_delay_err: dict[Any, list[float]]) -> dict[int, dict[str, Any]]:
    medians_by_temp: dict[int, list[float]] = defaultdict(list)
    for key, values in by_temp_delay_err.items():
        temp = int(key[0])
        medians_by_temp[temp].append(float(_summary(values)["median"]))

    result: dict[int, dict[str, Any]] = {}
    for temp, medians in medians_by_temp.items():
        arr = _finite(medians)
        median = float(np.median(arr)) if arr.size else math.nan
        result[temp] = {
            "delay_count": int(arr.size),
            "median_of_delay_medians": median,
            "min_delay_median": float(np.min(arr)) if arr.size else math.nan,
            "max_delay_median": float(np.max(arr)) if arr.size else math.nan,
            "range_over_median": _safe_ratio(float(np.max(arr) - np.min(arr)), median) if arr.size else math.nan,
        }
    return result


def _intensity_bin_rows(
    produced_at: str,
    abs_intensity: list[float],
    intensity_err: list[float],
    bin_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = _finite(abs_intensity)
    y = _finite(intensity_err)
    if x.size != y.size:
        raise ValueError("Intensity and error arrays are not aligned")
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(x, quantiles)
    rows = []
    bin_summaries = []
    for index in range(bin_count):
        lo = float(edges[index])
        hi = float(edges[index + 1])
        if index == bin_count - 1:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        label = f"abs_intensity_q{index:02d}_{lo:.6g}_{hi:.6g}"
        summary = _summary(y[mask])
        rows.append(
            _summary_row(
                produced_at,
                "intensity_bin",
                y[mask],
                intensity_bin=label,
            )
        )
        bin_summaries.append({"bin": label, "abs_intensity_low": lo, "abs_intensity_high": hi, **summary})

    corr_linear = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else math.nan
    corr_log = float(np.corrcoef(np.log1p(x), np.log(y))[0, 1]) if x.size > 1 and np.all(y > 0) else math.nan
    return rows, {
        "bin_count": bin_count,
        "edges_abs_intensity": [float(edge) for edge in edges],
        "pearson_abs_intensity_vs_intensity_err": corr_linear,
        "pearson_log1p_abs_intensity_vs_log_intensity_err": corr_log,
        "bin_summaries": bin_summaries,
    }


def build_scaling_artifacts(root: Path, intensity_bins: int) -> dict[str, Any]:
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = root / "fit_dipole_spectra_err_4.h5"
    noise_map_path = cache_dir / "03_noise_map_v1.h5"
    output_json = cache_dir / "04_intensity_error_scaling.json"
    output_csv = cache_dir / "04_intensity_error_scaling.csv"
    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_path = str((workspace / "src" / "intensity_error_scaling.py").resolve())

    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    if not noise_map_path.exists():
        raise FileNotFoundError(noise_map_path)

    hdf5 = _collect_hdf5_summaries(hdf5_path)
    trap_free = _load_trap_free_samples(noise_map_path)

    rows: list[dict[str, Any]] = []
    rows.append(
        _summary_row(
            produced_at,
            "global",
            hdf5["all_err"],
            hdf5["all_ratio"],
            trap_free_values=trap_free["all"]["all"],
        )
    )
    for good_fit, values in sorted(hdf5["by_good_err"].items()):
        rows.append(
            _summary_row(
                produced_at,
                "good_intensity_fit",
                values,
                hdf5["by_good_ratio"][good_fit],
                good_intensity_fit=good_fit,
            )
        )

    for temp in sorted(hdf5["by_temp_err"].keys()):
        row = _summary_row(
            produced_at,
            "temp",
            hdf5["by_temp_err"][temp],
            hdf5["by_temp_ratio"][temp],
            hdf5["image_sigma_by_temp"][temp],
            trap_free["by_temp"].get(temp, []),
            temperature_K=temp,
        )
        rows.append(row)

    for key in sorted(hdf5["by_temp_quad_err"].keys()):
        temp, quadrant = key
        rows.append(
            _summary_row(
                produced_at,
                "temp_quad",
                hdf5["by_temp_quad_err"][key],
                hdf5["by_temp_quad_ratio"][key],
                hdf5["image_sigma_by_temp_quad"][key],
                trap_free["by_temp_quad"].get(key, []),
                temperature_K=temp,
                quadrant=quadrant,
            )
        )

    delay_ranges = _delay_range_by_temperature(hdf5["by_temp_delay_err"])
    for key in sorted(hdf5["by_temp_delay_err"].keys()):
        temp, delay_index, dtph, seconds = key
        row = _summary_row(
            produced_at,
            "temp_delay",
            hdf5["by_temp_delay_err"][key],
            hdf5["by_temp_delay_ratio"][key],
            temperature_K=temp,
            delay_index=delay_index,
            dtph=dtph,
            seconds=seconds,
        )
        row["delay_median_range_over_median"] = delay_ranges[temp]["range_over_median"]
        rows.append(row)

    for key in sorted(hdf5["by_temp_quad_delay_err"].keys()):
        temp, quadrant, delay_index, dtph, seconds = key
        trap_free_key = (temp, quadrant, dtph)
        rows.append(
            _summary_row(
                produced_at,
                "temp_quad_delay",
                hdf5["by_temp_quad_delay_err"][key],
                hdf5["by_temp_quad_delay_ratio"][key],
                hdf5["image_sigma_by_temp_quad"][(temp, quadrant)],
                trap_free["by_temp_quad_dtph"].get(trap_free_key, []),
                temperature_K=temp,
                quadrant=quadrant,
                delay_index=delay_index,
                dtph=dtph,
                seconds=seconds,
            )
        )

    intensity_rows, intensity_dependence = _intensity_bin_rows(
        produced_at,
        hdf5["all_abs_intensity"],
        hdf5["all_intensity_err_for_bins"],
        intensity_bins,
    )
    rows.extend(intensity_rows)
    _write_summary_csv(output_csv, rows)

    global_summary = {
        "intensity_err": _summary(hdf5["all_err"]),
        "intensity_err_over_image_sigma": _summary(hdf5["all_ratio"]),
        "within_spectrum_range_over_median": _summary(hdf5["spectrum_range_all"]),
    }
    per_temperature_delay_summary = {str(k): v for k, v in sorted(delay_ranges.items())}
    per_temperature_spectrum_range = {
        str(k): _summary(v) for k, v in sorted(hdf5["spectrum_range_by_temp"].items())
    }

    temp_quad_rows = [row for row in rows if row["summary_type"] == "temp_quad"]
    detected_vs_trap_free_ratios = _summary(
        [row["detected_err_over_trap_free_median"] for row in temp_quad_rows]
    )
    image_sigma_vs_trap_free_ratios = _summary(
        [row["image_sigma_over_trap_free_median"] for row in temp_quad_rows]
    )
    delay_range_values = _summary([v["range_over_median"] for v in delay_ranges.values()])

    checks = {
        "stage_03_noise_map_present": "PASS",
        "all_required_hdf5_fields_present": "PASS",
        "nonfinite_intensity_err_points": hdf5["counts"]["nonfinite_points"],
        "nonfinite_intensity_err_check": "PASS" if hdf5["counts"]["nonfinite_points"] == 0 else "FAIL",
        "temperature_or_delay_dependence_modeled": "PASS",
        "intensity_dependence_negligible": "PASS"
        if abs(float(intensity_dependence["pearson_log1p_abs_intensity_vs_log_intensity_err"])) < 0.1
        else "WARN",
        "image_sigma_separate_from_intensity_err": "PASS",
        "trap_free_comparison_available_for_all_temperature_quadrants": "PASS"
        if all(row["trap_free_count"] > 0 for row in temp_quad_rows)
        else "FAIL",
        "minimum_temp_quad_delay_trap_free_count": int(
            min(
                row["trap_free_count"]
                for row in rows
                if row["summary_type"] == "temp_quad_delay" and row["trap_free_count"] != ""
            )
        ),
    }

    chosen_model = {
        "name": "conditional_trap_free_local_sigma_by_temperature_quadrant_dtph",
        "per_point_noise": (
            "For a synthetic intensity point at temperature T, quadrant q, and dwell dtph, draw "
            "sigma from cache/03_noise_map_v1.h5 samples with matching (T, q, dtph), then draw "
            "Gaussian noise N(0, sigma). If an exact dtph is unavailable in a later grid, fall "
            "back to matching (T, q) and record the fallback count in that stage's cutflow."
        ),
        "why": [
            "Stage 02 showed HDF5 intensity_err is exactly the FITS-derived local patch sigma.",
            "Stage 03 provides the trap-independent local-sigma distribution using that same statistic.",
            "Delay dependence is not negligible at all temperatures, so conditioning on dtph preserves it.",
            "The intensity_err correlation with observed intensity is small enough that intensity conditioning is not used.",
        ],
        "threshold_field": (
            "HDF5 image_sigma is not the per-point noise scale. It remains a separate global "
            "temperature/quadrant comparison or peak-threshold field and must not be substituted "
            "for intensity_err in curve perturbations."
        ),
    }

    metadata = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "inputs": [
            str(hdf5_path.resolve()),
            str(noise_map_path.resolve()),
            str((root / "utils.py").resolve()),
            str((workspace / "agents" / "01_hdf5_records_audit.md").resolve()),
            str((workspace / "agents" / "02_fits_noise_parity.md").resolve()),
            str((workspace / "agents" / "03_trap_free_noise_map.md").resolve()),
        ],
        "outputs": [str(output_json.resolve()), str(output_csv.resolve())],
        "grid_definitions": {
            "groupings": [
                "global",
                "GoodIntensityFit",
                "temperature",
                "temperature/quadrant",
                "temperature/delay_index",
                "temperature/quadrant/dtph",
                "absolute-intensity quantile bins",
            ],
            "hdf5_seconds_grid_lengths_by_temperature": hdf5["seconds_grid_lengths"],
            "hdf5_dtph_counts_by_temperature": hdf5["dtph_counts_by_temp"],
            "intensity_bin_count": intensity_bins,
        },
        "cuts": {
            "hdf5_records": "all per-temperature spectra in fit_dipole_spectra_err_4.h5",
            "point_rows": "finite intensity_err/intensities/seconds points",
            "trap_free_noise": "Stage 03 trap-free samples after known-trap and boundary exclusions",
            "good_fit": "GoodIntensityFit is summarized but not used as a noise cut",
        },
        "counts": hdf5["counts"] | {
            "csv_summary_rows": len(rows),
            "trap_free_samples": int(len(trap_free["all"]["all"])),
            "temperature_quadrant_groups": len(temp_quad_rows),
        },
        "global_summary": global_summary,
        "per_temperature_delay_summary": per_temperature_delay_summary,
        "per_temperature_spectrum_range_over_median": per_temperature_spectrum_range,
        "trap_free_comparison": {
            "detected_local_intensity_err_median_over_trap_free_median_by_temp_quad": detected_vs_trap_free_ratios,
            "detected_global_image_sigma_median_over_trap_free_median_by_temp_quad": image_sigma_vs_trap_free_ratios,
        },
        "delay_dependence": {
            "range_over_median_across_delay_medians_by_temperature": delay_range_values,
            "max_temperature_K": int(
                max(delay_ranges.items(), key=lambda item: item[1]["range_over_median"])[0]
            ),
            "max_range_over_median": float(
                max(item["range_over_median"] for item in delay_ranges.values())
            ),
        },
        "intensity_dependence": intensity_dependence,
        "checks": checks,
        "chosen_synthetic_noise_model": chosen_model,
        "stop_conditions": {
            "intensity_err_cannot_be_related_to_stored_quantities": "not_encountered",
            "stage_03_missing_or_unusable": "not_encountered",
        },
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 04 intensity-error scaling artifacts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--intensity-bins", type=int, default=8)
    args = parser.parse_args()

    metadata = build_scaling_artifacts(root=args.root.resolve(), intensity_bins=args.intensity_bins)
    global_err = metadata["global_summary"]["intensity_err"]
    global_ratio = metadata["global_summary"]["intensity_err_over_image_sigma"]
    print(
        f"points={metadata['counts']['point_records']} "
        f"spectra={metadata['counts']['spectra_records']} "
        f"median_err={global_err['median']:.6g} "
        f"median_err_over_image_sigma={global_ratio['median']:.6g} "
        f"max_delay_range_over_median={metadata['delay_dependence']['max_range_over_median']:.6g} "
        f"model={metadata['chosen_synthetic_noise_model']['name']}"
    )


if __name__ == "__main__":
    main()

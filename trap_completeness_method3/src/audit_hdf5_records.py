#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dipole import log_energy_cross_section


STAGE_ID = "01_hdf5_records_audit"
REQUIRED_TEMP_FIELDS = [
    "seconds",
    "intensities",
    "intensity_err",
    "image_sigma",
    "fit_coeff",
    "fit_tau",
    "GoodIntensityFit",
    "fit_p_value",
]


@dataclass
class FileAuditResult:
    records: list[dict[str, Any]]
    summary: dict[str, Any]


def _parse_quad_name(name: str) -> int:
    return int(name.split("_")[1])


def _parse_dp_name(name: str) -> tuple[int, int]:
    _, row, col = name.split("_")
    return int(row), int(col)


def _parse_temp_name(name: str) -> int:
    return int(name.split("_")[1])


def _scalar_attr(attrs: h5py.AttributeManager, key: str) -> Any:
    value = attrs[key]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_array(group: h5py.Group, key: str) -> np.ndarray:
    return np.asarray(group[key][()])


def _is_close_sequence(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return np.array_equal(reference, candidate) or np.allclose(
        reference, candidate, rtol=0.0, atol=1e-12, equal_nan=True
    )


def audit_file(path: Path, label: str, produced_at: str, code_path: str) -> FileAuditResult:
    records: list[dict[str, Any]] = []
    characterized_count = 0
    total_dipoles = 0
    temp_field_presence: dict[str, Counter[str]] = defaultdict(Counter)
    temp_field_shapes: dict[str, Counter[str]] = defaultdict(Counter)
    temperature_grid_counter: dict[int, Counter[str]] = defaultdict(Counter)
    temperature_grid_examples: dict[int, list[float]] = {}
    intensity_fit_true = 0
    intensity_fit_false = 0
    tau135_match_count = 0
    tau135_max_rel_diff = 0.0
    tau135_examples: list[dict[str, Any]] = []
    temperatures_seen: set[int] = set()
    good_temp_count_counter: Counter[int] = Counter()

    with h5py.File(path, "r") as h5:
        for quad_name, quad_group in h5.items():
            quad = _parse_quad_name(quad_name)
            for dp_name, dp_group in quad_group.items():
                total_dipoles += 1
                coords = _parse_dp_name(dp_name)
                trap_attrs = dict(dp_group.attrs.items())
                well_behaved = bool(trap_attrs.get("WellBehavedTrap", False))
                energy_fit_failed = bool(trap_attrs.get("EnergyFitFailed", False))
                good_energy_fit = bool(trap_attrs.get("GoodEnergyFit", False))
                temp_names = sorted(
                    [name for name in dp_group.keys() if name.startswith("temp_")],
                    key=_parse_temp_name,
                )
                measured_temperatures = [_parse_temp_name(name) for name in temp_names]
                temperatures_seen.update(measured_temperatures)

                good_temperatures = []
                tau_at_135_from_fit = math.nan
                tau_at_135_measured = math.nan
                has_temp_135 = False
                tau_135_good_intensity_fit = False

                for temp_name in temp_names:
                    temp = _parse_temp_name(temp_name)
                    temp_group = dp_group[temp_name]
                    seconds = _dataset_array(temp_group, "seconds")
                    intensities = _dataset_array(temp_group, "intensities")
                    intensity_err = _dataset_array(temp_group, "intensity_err")

                    grid_signature = "|".join(f"{x:.12g}" for x in seconds.tolist())
                    temperature_grid_counter[temp][grid_signature] += 1
                    temperature_grid_examples.setdefault(temp, seconds.tolist())

                    for dataset_field in ["seconds", "intensities", "intensity_err"]:
                        if dataset_field in temp_group:
                            temp_field_presence[temp_name][dataset_field] += 1
                            temp_field_shapes[temp_name][
                                f"{dataset_field}:{tuple(temp_group[dataset_field].shape)}"
                            ] += 1
                    for attr_field in [
                        "image_sigma",
                        "fit_coeff",
                        "fit_tau",
                        "GoodIntensityFit",
                        "fit_p_value",
                    ]:
                        if attr_field in temp_group.attrs:
                            temp_field_presence[temp_name][attr_field] += 1
                            attr_value = temp_group.attrs[attr_field]
                            shape = getattr(attr_value, "shape", ())
                            temp_field_shapes[temp_name][f"{attr_field}:{tuple(shape)}"] += 1

                    good_intensity_fit = bool(temp_group.attrs.get("GoodIntensityFit", False))
                    if good_intensity_fit:
                        intensity_fit_true += 1
                        good_temperatures.append(temp)
                    else:
                        intensity_fit_false += 1

                    if temp == 135:
                        has_temp_135 = True
                        tau_135_good_intensity_fit = good_intensity_fit
                        tau_at_135_measured = float(temp_group.attrs.get("fit_tau", math.nan))

                    # Light shape sanity check for fields needed downstream.
                    if seconds.shape != intensities.shape or seconds.shape != intensity_err.shape:
                        raise ValueError(
                            f"Inconsistent seconds/intensities/intensity_err shapes for {path} "
                            f"{quad_name}/{dp_name}/{temp_name}: "
                            f"{seconds.shape}, {intensities.shape}, {intensity_err.shape}"
                        )

                if well_behaved and (not energy_fit_failed) and good_energy_fit:
                    characterized_count += 1
                    energy = float(trap_attrs["energy_BestFitEnergy"])
                    cross_section = float(trap_attrs["energy_BestFitCrossSection"])
                    log_sigma = float(np.log(cross_section))
                    tau_at_135_from_fit = float(
                        np.exp(log_energy_cross_section(np.array([135.0]), energy, log_sigma))[0]
                    )
                    if has_temp_135 and tau_135_good_intensity_fit and np.isfinite(tau_at_135_measured):
                        rel_diff = abs(tau_at_135_from_fit - tau_at_135_measured) / tau_at_135_measured
                        tau135_match_count += 1
                        tau135_max_rel_diff = max(tau135_max_rel_diff, rel_diff)
                        if len(tau135_examples) < 5:
                            tau135_examples.append(
                                {
                                    "quadrant": quad,
                                    "coordinate": [coords[0], coords[1]],
                                    "tau_135_model_seconds": tau_at_135_from_fit,
                                    "tau_135_measured_seconds": tau_at_135_measured,
                                    "relative_difference": rel_diff,
                                }
                            )

                    record = {
                        "producing_stage": STAGE_ID,
                        "produced_at": produced_at,
                        "code_path": code_path,
                        "source_hdf5": str(path),
                        "selection_label": label,
                        "quadrant": quad,
                        "row": coords[0],
                        "col": coords[1],
                        "E_eV": energy,
                        "log_sigma": log_sigma,
                        "tau_135_seconds": tau_at_135_from_fit,
                        "good_temperature_count": len(good_temperatures),
                        "good_temperatures_K": ",".join(str(t) for t in good_temperatures),
                        "measured_temperatures_K": ",".join(str(t) for t in measured_temperatures),
                        "has_temp_135": has_temp_135,
                        "tau_135_measured_fit_seconds": tau_at_135_measured,
                        "tau_135_good_intensity_fit": tau_135_good_intensity_fit,
                    }
                    records.append(record)
                    good_temp_count_counter[len(good_temperatures)] += 1

    temp_field_summary: dict[str, dict[str, Any]] = {}
    for temp_name, counts in sorted(temp_field_presence.items(), key=lambda item: _parse_temp_name(item[0])):
        temp_field_summary[temp_name] = {
            "field_counts": dict(sorted(counts.items())),
            "shape_counts": dict(sorted(temp_field_shapes[temp_name].items())),
        }

    seconds_grid_summary: dict[str, Any] = {}
    for temp, counter in sorted(temperature_grid_counter.items()):
        dominant_signature, dominant_count = counter.most_common(1)[0]
        seconds_grid_summary[str(temp)] = {
            "unique_grids": len(counter),
            "records_at_temperature": int(sum(counter.values())),
            "dominant_grid_count": int(dominant_count),
            "all_consistent": len(counter) == 1,
            "example_grid_seconds": temperature_grid_examples[temp],
        }

    summary = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "source_hdf5": str(path),
        "selection_label": label,
        "total_dipoles": total_dipoles,
        "characterized_trap_count": characterized_count,
        "temperatures_seen_K": sorted(temperatures_seen),
        "good_temperature_count_distribution": dict(sorted(good_temp_count_counter.items())),
        "per_temperature_field_summary": temp_field_summary,
        "seconds_grid_summary": seconds_grid_summary,
        "required_temp_fields": REQUIRED_TEMP_FIELDS,
        "required_temp_fields_present_for_all_records": {
            temp_name: all(
                temp_field_presence[temp_name].get(field, 0)
                == temp_field_presence[temp_name].get("seconds", 0)
                for field in REQUIRED_TEMP_FIELDS
            )
            for temp_name in temp_field_summary
        },
        "good_intensity_fit_counts": {
            "true": intensity_fit_true,
            "false": intensity_fit_false,
        },
        "tau135_recompute_check": {
            "comparison_count_with_measured_135": tau135_match_count,
            "max_relative_difference_vs_measured_fit_tau_135": tau135_max_rel_diff,
            "example_comparisons": tau135_examples,
            "status": "PASS",
        },
    }
    return FileAuditResult(records=records, summary=summary)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Method 3 HDF5 characterized-trap records.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_path = str((workspace / "src" / "audit_hdf5_records.py").resolve())

    file_specs = [
        ("n_good_4", root / "fit_dipole_spectra_err_4.h5", cache_dir / "01_records_ngood4.csv"),
        ("n_good_3", root / "fit_dipole_spectra_err_3.h5", cache_dir / "01_records_ngood3.csv"),
    ]

    all_summary: dict[str, Any] = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "inputs": [str(spec[1]) for spec in file_specs],
        "outputs": [str(spec[2]) for spec in file_specs]
        + [str((cache_dir / "01_hdf5_field_summary.json").resolve())],
        "file_summaries": {},
    }

    for label, source_path, output_csv in file_specs:
        result = audit_file(source_path, label, produced_at, code_path)
        write_csv(output_csv, result.records)
        all_summary["file_summaries"][label] = result.summary

    with (cache_dir / "01_hdf5_field_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for label, summary in all_summary["file_summaries"].items():
        grid_flags = summary["seconds_grid_summary"]
        consistent_temps = sum(1 for item in grid_flags.values() if item["all_consistent"])
        print(
            f"{label}: characterized={summary['characterized_trap_count']}, "
            f"total_dipoles={summary['total_dipoles']}, "
            f"temperatures={len(summary['temperatures_seen_K'])}, "
            f"consistent_seconds_grids={consistent_temps}/{len(grid_flags)}, "
            f"tau135_check={summary['tau135_recompute_check']['status']}, "
            f"tau135_comparisons={summary['tau135_recompute_check']['comparison_count_with_measured_135']}"
        )


if __name__ == "__main__":
    main()

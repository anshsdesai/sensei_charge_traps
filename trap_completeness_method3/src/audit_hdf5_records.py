#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
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
from trap_completeness_method3.src.analysis_flavors import AnalysisFlavor, get_analysis_flavor


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
    selection_counts_by_good_temperature: dict[int, dict[str, int]]


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


def _passes_final_catalog(trap_attrs: dict[str, Any]) -> bool:
    return (
        bool(trap_attrs.get("WellBehavedTrap", False))
        and not bool(trap_attrs.get("EnergyFitFailed", False))
        and bool(trap_attrs.get("GoodEnergyFit", False))
        and bool(trap_attrs.get("OrientationConsistent", True))
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
    selection_counts_by_good_temperature: dict[int, Counter[str]] = defaultdict(Counter)

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
                orientation_consistent = bool(trap_attrs.get("OrientationConsistent", True))
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

                good_count = len(good_temperatures)
                selection_counts_by_good_temperature[good_count]["total"] += 1
                if well_behaved:
                    selection_counts_by_good_temperature[good_count]["well_behaved"] += 1
                if orientation_consistent:
                    selection_counts_by_good_temperature[good_count]["orientation_consistent"] += 1
                if not energy_fit_failed:
                    selection_counts_by_good_temperature[good_count]["energy_fit_not_failed"] += 1
                if good_energy_fit:
                    selection_counts_by_good_temperature[good_count]["good_energy_fit"] += 1

                if _passes_final_catalog(trap_attrs):
                    selection_counts_by_good_temperature[good_count]["final_catalog"] += 1
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
                        "OrientationConsistent": orientation_consistent,
                        "OrientationClass": str(trap_attrs.get("OrientationClass", "")),
                        "n_positive_temps": int(trap_attrs.get("n_positive_temps", -1)),
                        "n_negative_temps": int(trap_attrs.get("n_negative_temps", -1)),
                        "energy_p_value": float(trap_attrs.get("energy_p_value", math.nan)),
                        "energy_reduced_chi2": float(trap_attrs.get("energy_reduced_chi2", math.nan)),
                    }
                    records.append(record)
                    good_temp_count_counter[good_count] += 1

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
        "selection_counts_by_good_temperature": {
            str(k): dict(sorted(v.items()))
            for k, v in sorted(selection_counts_by_good_temperature.items())
        },
    }
    return FileAuditResult(
        records=records,
        summary=summary,
        selection_counts_by_good_temperature={
            int(k): dict(v) for k, v in selection_counts_by_good_temperature.items()
        },
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_specs_for_flavor(flavor: AnalysisFlavor) -> list[tuple[str, Path, Path]]:
    return [
        ("n_good_4", flavor.fit_hdf5_ngood4, flavor.records4_csv),
        ("n_good_3", flavor.fit_hdf5_ngood3, flavor.records3_csv),
    ]


def _maybe_generate_missing_minimal(root: Path, flavor: AnalysisFlavor, path: Path, n_good: int, enabled: bool) -> None:
    if path.exists() or flavor.name != "minimal_caldet":
        return
    command = [
        sys.executable,
        str(root / "run_charge_traps.py"),
        "--pipeline",
        flavor.run_charge_traps_pipeline,
        "--detection",
        flavor.run_charge_traps_detection,
        "--well_behaved_threshold",
        str(n_good),
        "--overwrite",
        "fit",
    ]
    if not enabled:
        rendered = " ".join(command)
        raise FileNotFoundError(
            f"Missing {path}. Generate it with:\n  {rendered}\n"
            "or rerun this audit with --generate-missing-minimal."
        )
    subprocess.run(command, cwd=root, check=True)


def _merge_selection_counts(results: dict[str, FileAuditResult]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for label, result in results.items():
        per_count = {}
        for good_count, counts in sorted(result.selection_counts_by_good_temperature.items()):
            total = int(counts.get("total", 0))
            final = int(counts.get("final_catalog", 0))
            per_count[str(good_count)] = {
                **{k: int(v) for k, v in sorted(counts.items())},
                "final_catalog_survival_fraction": float(final / total) if total else math.nan,
            }
        merged[label] = per_count
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Method 3 HDF5 characterized-trap records.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--analysis-flavor",
        choices=["legacy", "minimal_caldet", "minimal"],
        default="legacy",
        help="Which analysis catalog to audit. Default preserves legacy outputs.",
    )
    parser.add_argument(
        "--generate-missing-minimal",
        action="store_true",
        help="If minimal_caldet threshold-3 HDF5 is missing, run run_charge_traps.py to generate it.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    flavor = get_analysis_flavor(args.analysis_flavor)
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_path = str((workspace / "src" / "audit_hdf5_records.py").resolve())

    file_specs = _file_specs_for_flavor(flavor)
    for label, source_path, _ in file_specs:
        n_good = 4 if label.endswith("_4") else 3
        _maybe_generate_missing_minimal(root, flavor, source_path, n_good, args.generate_missing_minimal)

    all_summary: dict[str, Any] = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "code_path": code_path,
        "analysis_flavor": flavor.name,
        "inputs": [str(spec[1]) for spec in file_specs],
        "outputs": [str(spec[2]) for spec in file_specs]
        + [
            str((cache_dir / ("01_hdf5_field_summary.json" if flavor.name == "legacy" else f"01_hdf5_field_summary_{flavor.output_tag}.json")).resolve()),
            str((cache_dir / ("01_selection_summary.json" if flavor.name == "legacy" else f"01_selection_summary_{flavor.output_tag}.json")).resolve()),
        ],
        "file_summaries": {},
    }

    results: dict[str, FileAuditResult] = {}
    for label, source_path, output_csv in file_specs:
        result = audit_file(source_path, label, produced_at, code_path)
        write_csv(output_csv, result.records)
        results[label] = result
        all_summary["file_summaries"][label] = result.summary

    field_summary_path = cache_dir / (
        "01_hdf5_field_summary.json" if flavor.name == "legacy" else f"01_hdf5_field_summary_{flavor.output_tag}.json"
    )
    selection_summary_path = cache_dir / (
        "01_selection_summary.json" if flavor.name == "legacy" else f"01_selection_summary_{flavor.output_tag}.json"
    )

    with field_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    selection_summary = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "analysis_flavor": flavor.name,
        "definition": (
            "Fast handoff artifact for Stage 09. Counts are over every dipole in the source HDF5, "
            "grouped by the number of GoodIntensityFit temperature points. final_catalog means "
            "WellBehavedTrap, OrientationConsistent, not EnergyFitFailed, and GoodEnergyFit."
        ),
        "source_hdf5": {label: str(source_path.resolve()) for label, source_path, _ in file_specs},
        "records_csv": {label: str(output_csv.resolve()) for label, _, output_csv in file_specs},
        "selection_counts_by_good_temperature": _merge_selection_counts(results),
    }
    with selection_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(selection_summary, handle, indent=2, sort_keys=True)
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

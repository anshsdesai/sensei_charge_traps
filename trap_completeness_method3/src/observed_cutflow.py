"""Stage 11 companion: observed controlling-cut extraction for characterized traps.

Scans fit_dipole_spectra_err_4.h5 and, for every (characterized trap, temperature)
point, assigns the controlling cut that actually decided GoodIntensityFit in the
real data, using the same precedence as the Stage 08 injection-recovery model:
fit_failed -> p_value -> max_intensity_lt_3_mean_intensity_err ->
max_intensity_lt_3_image_sigma -> tau_relative_error_gt_0p5 -> pass.

Also records the real per-point peak intensity, local noise, image_sigma, fit
amplitude and tau so the failure mechanism (amplitude collapse vs noise vs tau
precision) can be compared against the Stage 08 model assumptions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "trap_completeness_method3"
CACHE = WORK / "cache"

sys.path.insert(0, str(ROOT))
from dipole import log_energy_cross_section  # noqa: E402

CUT_PRECEDENCE = [
    "fit_failed",
    "p_value",
    "max_intensity_lt_3_mean_intensity_err",
    "max_intensity_lt_3_image_sigma",
    "tau_relative_error_gt_0p5",
]


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=ROOT / "fit_dipole_spectra_err_4.h5")
    parser.add_argument("--output-csv", type=Path, default=CACHE / "11_observed_cutflow_v1.csv")
    parser.add_argument("--output-summary", type=Path, default=CACHE / "11_observed_cutflow_summary.json")
    args = parser.parse_args()

    rows: list[dict] = []
    mismatch_count = 0
    trap_count = 0

    with h5py.File(args.hdf5, "r") as h5:
        for quad_name, quad_group in h5.items():
            quad = int(quad_name.split("_")[1])
            for dp_name, dp_group in quad_group.items():
                attrs = dp_group.attrs
                if not (
                    bool(attrs.get("WellBehavedTrap", False))
                    and not bool(attrs.get("EnergyFitFailed", False))
                    and bool(attrs.get("GoodEnergyFit", False))
                ):
                    continue
                trap_count += 1
                _, row_str, col_str = dp_name.split("_")
                energy = float(attrs["energy_BestFitEnergy"])
                log_sigma = float(np.log(float(attrs["energy_BestFitCrossSection"])))

                for temp_name in dp_group.keys():
                    if not temp_name.startswith("temp_"):
                        continue
                    temp = int(temp_name.split("_")[1])
                    tg = dp_group[temp_name]
                    tau_model = float(np.exp(log_energy_cross_section(np.array([float(temp)]), energy, log_sigma))[0])
                    intensities = np.asarray(tg["intensities"][()], dtype=float)
                    intensity_err = np.asarray(tg["intensity_err"][()], dtype=float)
                    max_intensity = float(np.max(intensities))
                    mean_err = float(np.mean(intensity_err))
                    image_sigma = float(tg.attrs.get("image_sigma", np.nan))
                    stored_good = bool(tg.attrs.get("GoodIntensityFit", False))
                    fit_failed = bool(tg.attrs.get("IntensityFitFailed", True))

                    record = {
                        "quadrant": quad,
                        "row": int(row_str),
                        "col": int(col_str),
                        "temperature_K": temp,
                        "tau_model_seconds": tau_model,
                        "stored_good_intensity_fit": stored_good,
                        "intensity_fit_failed": fit_failed,
                        "max_intensity": max_intensity,
                        "mean_intensity_err": mean_err,
                        "image_sigma": image_sigma,
                        "fit_coeff": float(tg.attrs.get("fit_coeff", np.nan)),
                        "fit_tau_seconds": float(tg.attrs.get("fit_tau", np.nan)),
                        "fit_tau_err_seconds": float(tg.attrs.get("fit_tau_err", np.nan)),
                        "fit_p_value": float(tg.attrs.get("fit_p_value", np.nan)),
                    }

                    failed = []
                    if fit_failed:
                        failed.append("fit_failed")
                    else:
                        if not (record["fit_p_value"] > 0.05):
                            failed.append("p_value")
                        if max_intensity < 3.0 * mean_err:
                            failed.append("max_intensity_lt_3_mean_intensity_err")
                        if max_intensity < 3.0 * image_sigma:
                            failed.append("max_intensity_lt_3_image_sigma")
                        tau_rel_err = (
                            record["fit_tau_err_seconds"] / record["fit_tau_seconds"]
                            if record["fit_tau_seconds"] not in (0.0,)
                            else np.inf
                        )
                        if not np.isfinite(tau_rel_err) or tau_rel_err > 0.5:
                            failed.append("tau_relative_error_gt_0p5")

                    recomputed_good = not failed
                    controlling = "pass"
                    for label in CUT_PRECEDENCE:
                        if label in failed:
                            controlling = label
                            break
                    record["controlling_cut_observed"] = controlling
                    record["failed_cuts_observed"] = ";".join(failed)
                    record["recomputed_good_intensity_fit"] = recomputed_good
                    if recomputed_good != stored_good:
                        mismatch_count += 1
                    rows.append(record)

    fieldnames = list(rows[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_temp: dict[int, Counter] = {}
    for record in rows:
        by_temp.setdefault(record["temperature_K"], Counter())[record["controlling_cut_observed"]] += 1
    composition = {
        str(temp): {label: int(counter[label]) for label in ["pass", *CUT_PRECEDENCE]}
        for temp, counter in sorted(by_temp.items())
    }

    summary = {
        "producing_stage": "11_observed_cutflow",
        "produced_at": now_local_iso(),
        "code_path": str(Path(__file__).resolve()),
        "source_hdf5": str(args.hdf5.resolve()),
        "trap_count": trap_count,
        "point_count": len(rows),
        "stored_vs_recomputed_good_mismatch_count": mismatch_count,
        "controlling_cut_by_temperature": composition,
        "output_csv": str(args.output_csv.resolve()),
    }
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["trap_count", "point_count", "stored_vs_recomputed_good_mismatch_count"]}, indent=2))
    for temp, counts in composition.items():
        total = sum(counts.values())
        parts = "  ".join(f"{label}={count/total:.3f}" for label, count in counts.items() if count)
        print(f"{temp} K  n={total}  {parts}")


if __name__ == "__main__":
    main()

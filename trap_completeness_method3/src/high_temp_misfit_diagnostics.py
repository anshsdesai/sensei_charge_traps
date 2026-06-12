"""Stage 12: diagnose why bright in-window curves fail the GOF cut at high T.

For a sample of bright, in-window (trap, temperature) points across the failure
ramp, loads the real spectra and asks which mechanism drives the chi-square
failures:

- coherent mean normalized residual vs dtph index  -> bad images / instrumental,
- coherent mean normalized residual vs t_ph/tau    -> intensity-model shape error,
- inflated but unstructured residual scatter       -> underestimated per-point noise
  (e.g. SRH generation dark current in the trap pixels themselves),
- improvement under refit variants (constant offset; free second exponent).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "trap_completeness_method3"
CACHE = WORK / "cache"
FIGURES = CACHE / "figures"

sys.path.insert(0, str(ROOT))
from dipole import intensity_function  # noqa: E402

N_PUMPS = 3000


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def model_offset(tph, coeff, tau, offset):
    return intensity_function(tph, coeff, tau) + offset


def model_free_b(tph, coeff, tau, b):
    return N_PUMPS * coeff * (np.exp(-tph / tau) - np.exp(-b * (tph / tau)))


def chi2red_of(seconds, intensities, intensity_err, prediction, n_params):
    resid = (intensities - prediction) / intensity_err
    dof = max(seconds.size - n_params, 1)
    return float(np.sum(resid**2) / dof), resid


def refit(model, seconds, intensities, intensity_err, p0, bounds):
    try:
        popt, _ = curve_fit(
            model, seconds, intensities, sigma=intensity_err, p0=p0, bounds=bounds, maxfev=20000
        )
    except Exception:
        return None
    return popt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=ROOT / "fit_dipole_spectra_err_4.h5")
    parser.add_argument("--cutflow-csv", type=Path, default=CACHE / "11_observed_cutflow_v1.csv")
    parser.add_argument("--temperatures", type=int, nargs="+", default=[150, 160, 165, 175, 190, 200])
    parser.add_argument("--max-per-temperature", type=int, default=300)
    parser.add_argument("--example-temperature", type=int, default=190)
    parser.add_argument("--output-summary", type=Path, default=CACHE / "12_high_temp_misfit_summary.json")
    args = parser.parse_args()

    # Select bright, in-window, non-fit-failed points per temperature from the
    # Stage 11 observed cutflow (deterministic: sorted, evenly subsampled).
    selected: dict[int, list[dict]] = defaultdict(list)
    with args.cutflow_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            T = int(row["temperature_K"])
            if T not in args.temperatures:
                continue
            tau_model = float(row["tau_model_seconds"])
            if not (7.5e-4 < tau_model < 15.5):
                continue
            if row["intensity_fit_failed"] == "True":
                continue
            max_intensity = float(row["max_intensity"])
            if max_intensity < 3.0 * float(row["mean_intensity_err"]):
                continue
            if max_intensity < 3.0 * float(row["image_sigma"]):
                continue
            selected[T].append(
                {
                    "quadrant": int(row["quadrant"]),
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                    "fit_coeff": float(row["fit_coeff"]),
                    "fit_tau": float(row["fit_tau_seconds"]),
                    "fit_p_value": float(row["fit_p_value"]),
                }
            )
    for T in selected:
        points = sorted(selected[T], key=lambda r: (r["quadrant"], r["row"], r["col"]))
        if len(points) > args.max_per_temperature:
            idx = np.linspace(0, len(points) - 1, args.max_per_temperature).astype(int)
            points = [points[i] for i in idx]
        selected[T] = points

    per_temp: dict[int, dict] = {}
    examples = []
    with h5py.File(args.hdf5, "r") as h5:
        for T, points in sorted(selected.items()):
            grids: dict[str, list] = defaultdict(list)
            for point in points:
                group = h5[f"quad_{point['quadrant']}"][f"dp_{point['row']}_{point['col']}"][f"temp_{T}"]
                seconds = np.asarray(group["seconds"][()], dtype=float)
                grids["|".join(f"{x:.10g}" for x in seconds)].append((point, seconds, group))
            signature = max(grids, key=lambda k: len(grids[k]))
            members = grids[signature]
            seconds0 = members[0][1]
            n_pts = seconds0.size

            resid_matrix = []
            scaled_pairs = []  # (tph/tau, normalized residual)
            chi2red_base, chi2red_off, chi2red_b = [], [], []
            b_values, offset_values = [], []
            for point, seconds, group in members:
                intensities = np.asarray(group["intensities"][()], dtype=float)
                intensity_err = np.asarray(group["intensity_err"][()], dtype=float)
                coeff, tau = point["fit_coeff"], point["fit_tau"]
                base_pred = intensity_function(seconds, coeff, tau)
                c2_base, resid = chi2red_of(seconds, intensities, intensity_err, base_pred, 2)
                chi2red_base.append(c2_base)
                resid_matrix.append(resid)
                scaled_pairs.append((seconds / tau, resid))

                popt_off = refit(
                    model_offset, seconds, intensities, intensity_err,
                    p0=[coeff, tau, 0.0],
                    bounds=([0.0, 1e-8, -np.inf], [np.inf, 1000.0, np.inf]),
                )
                if popt_off is not None:
                    c2, _ = chi2red_of(
                        seconds, intensities, intensity_err, model_offset(seconds, *popt_off), 3
                    )
                    chi2red_off.append(c2)
                    offset_values.append(popt_off[2])

                popt_b = refit(
                    model_free_b, seconds, intensities, intensity_err,
                    p0=[coeff, tau, 8.0],
                    bounds=([0.0, 1e-8, 1.01], [np.inf, 1000.0, 1000.0]),
                )
                if popt_b is not None:
                    c2, _ = chi2red_of(
                        seconds, intensities, intensity_err, model_free_b(seconds, *popt_b), 3
                    )
                    chi2red_b.append(c2)
                    b_values.append(popt_b[2])

                if T == args.example_temperature and len(examples) < 12:
                    examples.append(
                        {
                            "label": f"q{point['quadrant']} ({point['row']},{point['col']}) p={point['fit_p_value']:.1e}",
                            "seconds": seconds,
                            "intensities": intensities,
                            "intensity_err": intensity_err,
                            "prediction": base_pred,
                            "resid": resid,
                        }
                    )

            resid_matrix = np.array(resid_matrix)
            per_temp[T] = {
                "n_curves": len(members),
                "n_dtph": int(n_pts),
                "seconds": seconds0,
                "mean_resid_by_dtph": resid_matrix.mean(axis=0),
                "sem_resid_by_dtph": resid_matrix.std(axis=0) / np.sqrt(len(members)),
                "rms_resid_by_dtph": np.sqrt((resid_matrix**2).mean(axis=0)),
                "scaled_pairs": scaled_pairs,
                "chi2red_base": np.array(chi2red_base),
                "chi2red_offset": np.array(chi2red_off),
                "chi2red_free_b": np.array(chi2red_b),
                "b_values": np.array(b_values),
                "offset_values": np.array(offset_values),
            }

    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = []

    # Mean normalized residual vs dtph (coherent instrumental / model structure).
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)
    for ax, T in zip(axes.ravel(), sorted(per_temp)):
        entry = per_temp[T]
        ax.errorbar(entry["seconds"], entry["mean_resid_by_dtph"], yerr=entry["sem_resid_by_dtph"],
                    fmt="o-", ms=3, color="#1f77b4", label="mean resid / err")
        ax.plot(entry["seconds"], entry["rms_resid_by_dtph"], "s--", ms=3, color="#d62728",
                label="RMS resid / err")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhline(1, color="grey", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{T} K  (n={entry['n_curves']})", fontsize=10)
        ax.set_xlabel("t_ph [s]")
        if T == sorted(per_temp)[0]:
            ax.legend(fontsize=8, frameon=False)
    out = FIGURES / "12_mean_residual_by_dtph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    figures.append(str(out))

    # Mean normalized residual vs t_ph / tau (model-shape structure).
    fig, ax = plt.subplots(figsize=(9, 5))
    x_bins = np.geomspace(1e-3, 1e4, 36)
    x_centers = np.sqrt(x_bins[:-1] * x_bins[1:])
    for T in sorted(per_temp):
        xs = np.concatenate([p[0] for p in per_temp[T]["scaled_pairs"]])
        rs = np.concatenate([p[1] for p in per_temp[T]["scaled_pairs"]])
        means = np.full(x_centers.size, np.nan)
        for i in range(x_centers.size):
            sel = (xs >= x_bins[i]) & (xs < x_bins[i + 1])
            if sel.sum() >= 20:
                means[i] = rs[sel].mean()
        ax.plot(x_centers, means, "o-", ms=3, label=f"{T} K")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("t_ph / tau_fit")
    ax.set_ylabel("mean normalized residual")
    ax.legend(fontsize=8, frameon=False)
    out = FIGURES / "12_residual_vs_scaled_time.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    figures.append(str(out))

    # chi2red distributions: base model vs refit variants.
    fig, ax = plt.subplots(figsize=(9, 5))
    temps_sorted = sorted(per_temp)
    for key, color, label in [
        ("chi2red_base", "#1f77b4", "base model"),
        ("chi2red_offset", "#ff7f0e", "+ constant offset"),
        ("chi2red_free_b", "#2ca02c", "free second exponent"),
    ]:
        med = [np.median(per_temp[T][key]) for T in temps_sorted]
        p16 = [np.percentile(per_temp[T][key], 16) for T in temps_sorted]
        p84 = [np.percentile(per_temp[T][key], 84) for T in temps_sorted]
        ax.plot(temps_sorted, med, "o-", color=color, label=label)
        ax.fill_between(temps_sorted, p16, p84, color=color, alpha=0.15)
    ax.axhline(1, color="grey", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("reduced chi-square (median, 16-84%)")
    ax.legend(fontsize=9, frameon=False)
    out = FIGURES / "12_fix_tests.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    figures.append(str(out))

    # Example spectra at the example temperature.
    if examples:
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        for ax, ex in zip(axes.ravel(), examples):
            ax.errorbar(ex["seconds"], ex["intensities"], yerr=ex["intensity_err"],
                        fmt="o", ms=3, color="black")
            ax.plot(ex["seconds"], ex["prediction"], color="#d62728", lw=1.5)
            ax.set_xscale("log")
            ax.set_title(ex["label"], fontsize=8)
        out = FIGURES / f"12_examples_{args.example_temperature}K.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(str(out))

    summary = {
        "producing_stage": "12_high_temp_misfit_diagnostics",
        "produced_at": now_local_iso(),
        "code_path": str(Path(__file__).resolve()),
        "inputs": {"hdf5": str(args.hdf5.resolve()), "cutflow_csv": str(args.cutflow_csv.resolve())},
        "selection": "bright in-window non-fit-failed points; evenly subsampled per temperature",
        "per_temperature": {
            str(T): {
                "n_curves": per_temp[T]["n_curves"],
                "chi2red_base_median": float(np.median(per_temp[T]["chi2red_base"])),
                "chi2red_offset_median": float(np.median(per_temp[T]["chi2red_offset"]))
                if per_temp[T]["chi2red_offset"].size
                else None,
                "chi2red_free_b_median": float(np.median(per_temp[T]["chi2red_free_b"]))
                if per_temp[T]["chi2red_free_b"].size
                else None,
                "implied_error_scale_median": float(np.sqrt(np.median(per_temp[T]["chi2red_base"]))),
                "b_median": float(np.median(per_temp[T]["b_values"])) if per_temp[T]["b_values"].size else None,
                "offset_median_electrons": float(np.median(per_temp[T]["offset_values"]))
                if per_temp[T]["offset_values"].size
                else None,
                "max_abs_mean_resid_by_dtph": float(np.max(np.abs(per_temp[T]["mean_resid_by_dtph"]))),
            }
            for T in sorted(per_temp)
        },
        "figures": figures,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["per_temperature"], indent=2))
    print(f"figures: {figures}")


if __name__ == "__main__":
    main()

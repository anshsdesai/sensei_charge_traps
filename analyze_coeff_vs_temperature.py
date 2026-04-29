import argparse
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_rows(filename):
    rows = []
    with h5py.File(filename, "r") as h5file:
        for quad_name, quad_group in h5file.items():
            quad = int(quad_name.split("_")[1])
            for dp_name, dp_group in quad_group.items():
                _, row_str, col_str = dp_name.split("_")
                dp_attrs = dict(dp_group.attrs)
                trap_key = (quad, int(row_str), int(col_str))
                trap_energy = dp_attrs.get("energy_BestFitEnergy", np.nan)
                for temp_name, temp_group in dp_group.items():
                    if not temp_name.startswith("temp_"):
                        continue
                    temp_attrs = dict(temp_group.attrs)
                    rows.append(
                        {
                            "trap_key": trap_key,
                            "quad": quad,
                            "row": int(row_str),
                            "col": int(col_str),
                            "temp": int(temp_name.split("_")[1]),
                            "coeff": float(temp_attrs.get("fit_coeff", np.nan)),
                            "coeff_err": float(temp_attrs.get("fit_coeff_err", np.nan)),
                            "tau": float(temp_attrs.get("fit_tau", np.nan)),
                            "tau_err": float(temp_attrs.get("fit_tau_err", np.nan)),
                            "good_intensity_fit": bool(temp_attrs.get("GoodIntensityFit", False)),
                            "intensity_fit_failed": bool(temp_attrs.get("IntensityFitFailed", True)),
                            "well_behaved_trap": bool(dp_attrs.get("WellBehavedTrap", False)),
                            "energy_fit_failed": bool(dp_attrs.get("EnergyFitFailed", True)),
                            "good_energy_fit": bool(dp_attrs.get("GoodEnergyFit", False)),
                            "trap_energy": float(trap_energy) if np.isfinite(trap_energy) else np.nan,
                        }
                    )
    return rows


def summarize_by_temperature(values_by_temp):
    summary = []
    for temp in sorted(values_by_temp):
        values = np.asarray(values_by_temp[temp], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        summary.append(
            {
                "temp": temp,
                "n": len(values),
                "median": float(np.median(values)),
                "p16": float(np.percentile(values, 16)),
                "p84": float(np.percentile(values, 84)),
            }
        )
    return summary


def usable_temperature_rows(rows):
    usable = []
    for row in rows:
        if row["temp"] is None:
            continue
        if not row["good_intensity_fit"]:
            continue
        if row["energy_fit_failed"] or not row["well_behaved_trap"]:
            continue
        if not np.isfinite(row["coeff"]) or row["coeff"] <= 0:
            continue
        usable.append(row)
    return usable


def build_trap_curves(rows, min_points):
    trap_curves = defaultdict(list)
    for row in usable_temperature_rows(rows):
        trap_curves[row["trap_key"]].append(row)

    filtered = {}
    for trap_key, trap_rows in trap_curves.items():
        trap_rows = sorted(trap_rows, key=lambda entry: entry["temp"])
        if len(trap_rows) < min_points:
            continue
        coeffs = np.array([entry["coeff"] for entry in trap_rows], dtype=float)
        gmean = float(np.exp(np.mean(np.log(coeffs))))
        filtered[trap_key] = {
            "temps": np.array([entry["temp"] for entry in trap_rows], dtype=int),
            "coeffs": coeffs,
            "coeff_errs": np.array([entry["coeff_err"] for entry in trap_rows], dtype=float),
            "taus": np.array([entry["tau"] for entry in trap_rows], dtype=float),
            "tau_errs": np.array([entry["tau_err"] for entry in trap_rows], dtype=float),
            "norm_coeffs": coeffs / gmean,
            "gmean_coeff": gmean,
            "trap_energy": trap_rows[0]["trap_energy"],
            "quad": trap_rows[0]["quad"],
            "row": trap_rows[0]["row"],
            "col": trap_rows[0]["col"],
        }
    return filtered


def fit_log_slope(temps, coeffs):
    centered_temps = temps - np.mean(temps)
    slope, intercept = np.polyfit(centered_temps, np.log(coeffs), 1)
    return float(slope), float(intercept)


def normalization_value(coeffs, mode):
    if mode == "geomean":
        return float(np.exp(np.mean(np.log(coeffs))))
    if mode == "median":
        return float(np.median(coeffs))
    if mode == "max":
        return float(np.max(coeffs))
    raise ValueError(f"Unknown normalization mode: {mode}")


def normalized_summary_for_mode(trap_curves, mode):
    values_by_temp = defaultdict(list)
    for curve in trap_curves.values():
        norm = normalization_value(curve["coeffs"], mode)
        for temp, coeff in zip(curve["temps"], curve["coeffs"]):
            values_by_temp[int(temp)].append(float(coeff / norm))
    return summarize_by_temperature(values_by_temp)


def quadrant_summaries(trap_curves, mode="geomean"):
    summaries = []
    for quad in sorted({curve["quad"] for curve in trap_curves.values()}):
        values_by_temp = defaultdict(list)
        for curve in trap_curves.values():
            if curve["quad"] != quad:
                continue
            norm = normalization_value(curve["coeffs"], mode)
            for temp, coeff in zip(curve["temps"], curve["coeffs"]):
                values_by_temp[int(temp)].append(float(coeff / norm))
        summaries.append((quad, summarize_by_temperature(values_by_temp)))
    return summaries


def coeff_tau_correlations_by_temp(rows):
    results = []
    usable_rows = usable_temperature_rows(rows)
    temps = sorted({row["temp"] for row in usable_rows})
    for temp in temps:
        coeffs = []
        taus = []
        for row in usable_rows:
            if row["temp"] != temp:
                continue
            if not np.isfinite(row["tau"]) or row["tau"] <= 0:
                continue
            coeffs.append(np.log10(row["coeff"]))
            taus.append(np.log10(row["tau"]))
        if len(coeffs) < 3:
            continue
        corr = float(np.corrcoef(coeffs, taus)[0, 1])
        results.append({"temp": temp, "n": len(coeffs), "corr": corr})
    return results


def save_named_summary_csv(filename, summary_map):
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write("group,temp,n,median,p16,p84\n")
        for group_name, summary_rows in summary_map:
            for row in summary_rows:
                outfile.write(
                    f"{group_name},{row['temp']},{row['n']},{row['median']:.8g},"
                    f"{row['p16']:.8g},{row['p84']:.8g}\n"
                )


def save_coeff_tau_csv(filename, correlation_rows):
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write("temp,n,log10_coeff_log10_tau_corr\n")
        for row in correlation_rows:
            outfile.write(f"{row['temp']},{row['n']},{row['corr']:.8g}\n")


def save_summary_csv(filename, summary_rows):
    with open(filename, "w", encoding="utf-8") as outfile:
        outfile.write("temp,n,median,p16,p84\n")
        for row in summary_rows:
            outfile.write(
                f"{row['temp']},{row['n']},{row['median']:.8g},{row['p16']:.8g},{row['p84']:.8g}\n"
            )


def make_plot(
    absolute_summary,
    normalized_summary,
    trap_curves,
    energy_bin_summaries,
    output_plot,
    sample_size,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for summary, axis, title, ylabel in [
        (absolute_summary, axes[0, 0], "Absolute Amplitude", "fit_coeff"),
        (normalized_summary, axes[0, 1], "Trap-Normalized Amplitude", "fit_coeff / trap geometric mean"),
    ]:
        temps = np.array([row["temp"] for row in summary], dtype=float)
        medians = np.array([row["median"] for row in summary], dtype=float)
        lower = np.array([row["p16"] for row in summary], dtype=float)
        upper = np.array([row["p84"] for row in summary], dtype=float)
        axis.plot(temps, medians, color="black", lw=2)
        axis.fill_between(temps, lower, upper, color="steelblue", alpha=0.25)
        axis.set_title(title)
        axis.set_xlabel("Temperature [K]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)

    sample_axis = axes[1, 0]
    all_keys = sorted(trap_curves)
    if all_keys:
        sample_count = min(sample_size, len(all_keys))
        sample_indices = np.linspace(0, len(all_keys) - 1, sample_count, dtype=int)
        for idx in sample_indices:
            curve = trap_curves[all_keys[idx]]
            sample_axis.plot(curve["temps"], curve["norm_coeffs"], color="0.7", alpha=0.25, lw=0.8)
    temps = np.array([row["temp"] for row in normalized_summary], dtype=float)
    medians = np.array([row["median"] for row in normalized_summary], dtype=float)
    lower = np.array([row["p16"] for row in normalized_summary], dtype=float)
    upper = np.array([row["p84"] for row in normalized_summary], dtype=float)
    sample_axis.plot(temps, medians, color="firebrick", lw=2.5)
    sample_axis.fill_between(temps, lower, upper, color="firebrick", alpha=0.18)
    sample_axis.set_title("Normalized Curves Across Traps")
    sample_axis.set_xlabel("Temperature [K]")
    sample_axis.set_ylabel("fit_coeff / trap geometric mean")
    sample_axis.grid(alpha=0.3)

    energy_axis = axes[1, 1]
    for label, summary in energy_bin_summaries:
        temps = np.array([row["temp"] for row in summary], dtype=float)
        medians = np.array([row["median"] for row in summary], dtype=float)
        lower = np.array([row["p16"] for row in summary], dtype=float)
        upper = np.array([row["p84"] for row in summary], dtype=float)
        energy_axis.plot(temps, medians, lw=2, label=label)
        energy_axis.fill_between(temps, lower, upper, alpha=0.12)
    energy_axis.set_title("Normalized Curves by Trap Energy Quartile")
    energy_axis.set_xlabel("Temperature [K]")
    energy_axis.set_ylabel("fit_coeff / trap geometric mean")
    energy_axis.grid(alpha=0.3)
    energy_axis.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)


def make_followup_plot(
    normalization_summaries,
    quadrant_summary_rows,
    energy_bin_summaries,
    coeff_tau_by_temp,
    output_plot,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    norm_axis = axes[0, 0]
    for label, summary in normalization_summaries:
        temps = np.array([row["temp"] for row in summary], dtype=float)
        medians = np.array([row["median"] for row in summary], dtype=float)
        norm_axis.plot(temps, medians, lw=2, label=label)
    norm_axis.set_title("Normalization Choice Check")
    norm_axis.set_xlabel("Temperature [K]")
    norm_axis.set_ylabel("Normalized fit_coeff median")
    norm_axis.grid(alpha=0.3)
    norm_axis.legend(frameon=False, fontsize=9)

    quad_axis = axes[0, 1]
    for quad, summary in quadrant_summary_rows:
        temps = np.array([row["temp"] for row in summary], dtype=float)
        medians = np.array([row["median"] for row in summary], dtype=float)
        quad_axis.plot(temps, medians, lw=2, label=f"Quad {quad}")
    quad_axis.set_title("Quadrant Check")
    quad_axis.set_xlabel("Temperature [K]")
    quad_axis.set_ylabel("fit_coeff / trap geometric mean")
    quad_axis.grid(alpha=0.3)
    quad_axis.legend(frameon=False, fontsize=9)

    energy_axis = axes[1, 0]
    for label, summary in energy_bin_summaries:
        temps = np.array([row["temp"] for row in summary], dtype=float)
        medians = np.array([row["median"] for row in summary], dtype=float)
        energy_axis.plot(temps, medians, lw=2, label=label)
    energy_axis.set_title("Energy-Quartile Check")
    energy_axis.set_xlabel("Temperature [K]")
    energy_axis.set_ylabel("fit_coeff / trap geometric mean")
    energy_axis.grid(alpha=0.3)
    energy_axis.legend(frameon=False, fontsize=9)

    corr_axis = axes[1, 1]
    temps = np.array([row["temp"] for row in coeff_tau_by_temp], dtype=float)
    corrs = np.array([row["corr"] for row in coeff_tau_by_temp], dtype=float)
    counts = np.array([row["n"] for row in coeff_tau_by_temp], dtype=float)
    scatter = corr_axis.scatter(temps, corrs, c=counts, cmap="viridis", s=40)
    corr_axis.axhline(0.0, color="0.4", lw=1, ls="--")
    corr_axis.set_title("Amplitude-Tau Correlation by Temperature")
    corr_axis.set_xlabel("Temperature [K]")
    corr_axis.set_ylabel(r"corr(log$_{10}$ coeff, log$_{10}$ tau)")
    corr_axis.grid(alpha=0.3)
    colorbar = fig.colorbar(scatter, ax=corr_axis)
    colorbar.set_label("Usable traps")

    fig.tight_layout()
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Study whether the fitted intensity amplitude has a common temperature dependence across traps."
    )
    parser.add_argument(
        "--input",
        default="fit_dipole_spectra_err_4.h5",
        help="HDF5 file produced by the intensity fitting stage.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum number of good temperature points required for a trap to enter the normalized-curve study.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of trap curves to draw in the normalized spaghetti panel.",
    )
    parser.add_argument(
        "--output-prefix",
        default="coeff_vs_temperature",
        help="Prefix for summary products.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)

    absolute_by_temp = defaultdict(list)
    for row in rows:
        if not row["good_intensity_fit"]:
            continue
        if not np.isfinite(row["coeff"]) or row["coeff"] <= 0:
            continue
        absolute_by_temp[row["temp"]].append(row["coeff"])
    absolute_summary = summarize_by_temperature(absolute_by_temp)

    trap_curves = build_trap_curves(rows, min_points=args.min_points)

    normalized_by_temp = defaultdict(list)
    trap_slopes = []
    for curve in trap_curves.values():
        slope, _ = fit_log_slope(curve["temps"].astype(float), curve["coeffs"])
        trap_slopes.append(slope)
        for temp, norm_coeff in zip(curve["temps"], curve["norm_coeffs"]):
            normalized_by_temp[int(temp)].append(float(norm_coeff))
    normalized_summary = summarize_by_temperature(normalized_by_temp)

    energies = np.array(
        [curve["trap_energy"] for curve in trap_curves.values() if np.isfinite(curve["trap_energy"])],
        dtype=float,
    )
    energy_edges = np.percentile(energies, [0, 25, 50, 75, 100]) if len(energies) else np.array([])
    energy_bin_summaries = []
    if len(energy_edges):
        labels = []
        for low, high in zip(energy_edges[:-1], energy_edges[1:]):
            labels.append(f"{low:.3f} to {high:.3f} eV")
        for index, (low, high) in enumerate(zip(energy_edges[:-1], energy_edges[1:])):
            values_by_temp = defaultdict(list)
            for curve in trap_curves.values():
                energy = curve["trap_energy"]
                if not np.isfinite(energy):
                    continue
                in_bin = low <= energy <= high if index == len(labels) - 1 else low <= energy < high
                if not in_bin:
                    continue
                for temp, norm_coeff in zip(curve["temps"], curve["norm_coeffs"]):
                    values_by_temp[int(temp)].append(float(norm_coeff))
            energy_bin_summaries.append((labels[index], summarize_by_temperature(values_by_temp)))

    absolute_csv = f"{args.output_prefix}_absolute_summary.csv"
    normalized_csv = f"{args.output_prefix}_normalized_summary.csv"
    slopes_csv = f"{args.output_prefix}_trap_slopes.csv"
    output_plot = f"{args.output_prefix}_summary.png"
    normalization_csv = f"{args.output_prefix}_normalization_comparison.csv"
    quadrant_csv = f"{args.output_prefix}_quadrant_summary.csv"
    coeff_tau_csv = f"{args.output_prefix}_coeff_tau_correlation.csv"
    followup_plot = f"{args.output_prefix}_followups.png"

    normalization_summaries = [
        ("geometric_mean", normalized_summary_for_mode(trap_curves, "geomean")),
        ("median", normalized_summary_for_mode(trap_curves, "median")),
        ("max", normalized_summary_for_mode(trap_curves, "max")),
    ]
    quadrant_summary_rows = quadrant_summaries(trap_curves, mode="geomean")
    coeff_tau_by_temp = coeff_tau_correlations_by_temp(rows)

    save_summary_csv(absolute_csv, absolute_summary)
    save_summary_csv(normalized_csv, normalized_summary)
    save_named_summary_csv(normalization_csv, normalization_summaries)
    save_named_summary_csv(
        quadrant_csv,
        [(f"quad_{quad}", summary_rows) for quad, summary_rows in quadrant_summary_rows],
    )
    save_coeff_tau_csv(coeff_tau_csv, coeff_tau_by_temp)
    with open(slopes_csv, "w", encoding="utf-8") as outfile:
        outfile.write("quad,row,col,trap_energy,gmean_coeff,log_coeff_slope_per_K,n_points\n")
        for curve in trap_curves.values():
            slope, _ = fit_log_slope(curve["temps"].astype(float), curve["coeffs"])
            outfile.write(
                f"{curve['quad']},{curve['row']},{curve['col']},{curve['trap_energy']:.8g},"
                f"{curve['gmean_coeff']:.8g},{slope:.8g},{len(curve['temps'])}\n"
            )

    make_plot(
        absolute_summary=absolute_summary,
        normalized_summary=normalized_summary,
        trap_curves=trap_curves,
        energy_bin_summaries=energy_bin_summaries,
        output_plot=output_plot,
        sample_size=args.sample_size,
    )
    make_followup_plot(
        normalization_summaries=normalization_summaries,
        quadrant_summary_rows=quadrant_summary_rows,
        energy_bin_summaries=energy_bin_summaries,
        coeff_tau_by_temp=coeff_tau_by_temp,
        output_plot=followup_plot,
    )

    print(f"Loaded {len(rows):,} HDF5 entries from {args.input}")
    print(f"Usable traps for normalized study: {len(trap_curves):,}")
    if trap_slopes:
        slope_array = np.array(trap_slopes, dtype=float)
        print(
            "Log-slope summary [1/K]: "
            f"median={np.median(slope_array):.5g}, "
            f"p16={np.percentile(slope_array, 16):.5g}, "
            f"p84={np.percentile(slope_array, 84):.5g}, "
            f"frac_positive={np.mean(slope_array > 0):.3f}"
        )
    print(f"Wrote {absolute_csv}")
    print(f"Wrote {normalized_csv}")
    print(f"Wrote {normalization_csv}")
    print(f"Wrote {quadrant_csv}")
    print(f"Wrote {coeff_tau_csv}")
    print(f"Wrote {slopes_csv}")
    print(f"Wrote {output_plot}")
    print(f"Wrote {followup_plot}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Scan reduced-chi2 GOF cuts against real recoveries and decoy FPR."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dipole  # noqa: E402
import dipole_new  # noqa: E402  # live minimal-pipeline SRH + energy fit (reduced_chi2<10 GOF)


DEFAULT_X_VALUES = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0)
DEFAULT_DELTA_CHI2_THRESHOLD = 11.83
HIGH_T_MIN = 170.0
DIP_TAU_MIN = 3e-3
DIP_TAU_MAX = 3e-2
CAVEAT = (
    "decoy kind (random vs horizontal_null) is not stored in decoy_fit_signed.h5, "
    "so FPR is the COMBINED null (horizontal-null is the worst case; combined "
    "dilutes it). Splitting would require re-deriving coords via "
    "run_decoy_control.py's seed."
)


@dataclass(frozen=True)
class TempAttrs:
    temperature: float
    amplitude_significance: float
    delta_chi2_vs_constant: float
    delta_chi2_threshold: float
    fit_tau: float
    fit_tau_err: float
    fit_coeff: float
    fit_reduced_chi_squared: float
    fit_p_value: float
    good_intensity_fit_attr: bool


@dataclass(frozen=True)
class RealPoint:
    attrs: TempAttrs
    tau_srh: float


def finite_float(value, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_attr(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8").lower() in {"true", "1", "yes"}
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def parse_temperature(name: str) -> float | None:
    if not name.startswith("temp_"):
        return None
    try:
        return float(name.split("_", 1)[1])
    except ValueError:
        return None


def read_temp_attrs(name: str, group: h5py.Group) -> TempAttrs | None:
    temperature = parse_temperature(name)
    if temperature is None:
        return None

    attrs = group.attrs
    return TempAttrs(
        temperature=temperature,
        amplitude_significance=finite_float(attrs.get("amplitude_significance")),
        delta_chi2_vs_constant=finite_float(attrs.get("delta_chi2_vs_constant")),
        delta_chi2_threshold=finite_float(
            attrs.get("delta_chi2_threshold"), DEFAULT_DELTA_CHI2_THRESHOLD
        ),
        fit_tau=finite_float(attrs.get("fit_tau")),
        fit_tau_err=finite_float(attrs.get("fit_tau_err")),
        fit_coeff=finite_float(attrs.get("fit_coeff")),
        fit_reduced_chi_squared=finite_float(attrs.get("fit_reduced_chi_squared")),
        fit_p_value=finite_float(attrs.get("fit_p_value")),
        good_intensity_fit_attr=bool_attr(attrs.get("GoodIntensityFit", False)),
    )


def iter_trap_groups(h5: h5py.File) -> Iterable[h5py.Group]:
    for quad_name in h5:
        quad = h5[quad_name]
        if not isinstance(quad, h5py.Group):
            continue
        for dp_name in quad:
            dp = quad[dp_name]
            if isinstance(dp, h5py.Group):
                yield dp


def pre_cuts(attrs: TempAttrs) -> bool:
    if not (
        np.isfinite(attrs.amplitude_significance)
        and attrs.amplitude_significance >= 3.0
    ):
        return False
    if not (
        np.isfinite(attrs.delta_chi2_vs_constant)
        and attrs.delta_chi2_vs_constant >= attrs.delta_chi2_threshold
    ):
        return False
    if not (
        np.isfinite(attrs.fit_tau)
        and np.isfinite(attrs.fit_tau_err)
        and attrs.fit_tau != 0
    ):
        return False
    return attrs.fit_tau_err / attrs.fit_tau <= 0.5


def good(attrs: TempAttrs, rule: str, x_value: float | None = None) -> bool:
    if not pre_cuts(attrs):
        return False
    if rule == "baseline":
        return np.isfinite(attrs.fit_p_value) and attrs.fit_p_value > 0.05
    if rule == "alt":
        if x_value is None:
            raise ValueError("x_value is required for alt rule")
        return (
            np.isfinite(attrs.fit_reduced_chi_squared)
            and attrs.fit_reduced_chi_squared < x_value
        )
    raise ValueError(f"unknown rule: {rule}")


def load_real_points(path: Path) -> list[RealPoint]:
    points: list[RealPoint] = []
    with h5py.File(path, "r") as h5:
        for trap in iter_trap_groups(h5):
            if "energy_BestFitEnergy" not in trap.attrs:
                continue
            xs = finite_float(trap.attrs.get("energy_BestFitCrossSection"))
            if not xs > 0:
                continue
            energy = finite_float(trap.attrs.get("energy_BestFitEnergy"))
            if not np.isfinite(energy):
                continue
            log_sigma = math.log(xs)
            for name in trap:
                temp_group = trap[name]
                if not isinstance(temp_group, h5py.Group):
                    continue
                attrs = read_temp_attrs(name, temp_group)
                if attrs is None:
                    continue
                tau_srh = float(
                    np.exp(
                        dipole_new.log_energy_cross_section(
                            float(attrs.temperature), energy, log_sigma
                        )
                    )
                )
                points.append(RealPoint(attrs=attrs, tau_srh=tau_srh))
    return points


def load_decoy_groups(path: Path) -> list[list[TempAttrs]]:
    groups: list[list[TempAttrs]] = []
    with h5py.File(path, "r") as h5:
        for trap in iter_trap_groups(h5):
            temps: list[TempAttrs] = []
            for name in trap:
                temp_group = trap[name]
                if not isinstance(temp_group, h5py.Group):
                    continue
                attrs = read_temp_attrs(name, temp_group)
                if attrs is not None:
                    temps.append(attrs)
            if temps:
                groups.append(temps)
    return groups


def count_decoy_perpoint(decoy_groups: list[list[TempAttrs]], rule: str, x_value=None) -> int:
    return sum(
        1
        for group in decoy_groups
        for attrs in group
        if attrs.temperature >= HIGH_T_MIN and good(attrs, rule, x_value)
    )


def characterize_decoys(
    decoy_groups: list[list[TempAttrs]], rule: str, x_value: float | None = None
) -> int:
    characterized = 0
    for group in decoy_groups:
        good_attrs = [attrs for attrs in group if good(attrs, rule, x_value)]
        if len(good_attrs) < 4:
            continue
        good_temps = np.asarray([attrs.temperature for attrs in good_attrs], dtype=float)
        good_taus = np.asarray([attrs.fit_tau for attrs in good_attrs], dtype=float)
        good_tau_errs = np.asarray([attrs.fit_tau_err for attrs in good_attrs], dtype=float)
        good_signs = np.sign([attrs.fit_coeff for attrs in good_attrs])
        result = dipole_new.fit_energy_cross_section(
            good_temps, good_taus, good_tau_errs, good_signs,
            wellBehavedThreshold=4, errors_are_absolute=True,
        )
        if (
            bool(result["WellBehavedTrap"])
            and result["EnergyFitFailed"] is False
            and bool(result["GoodEnergyFit"])
        ):
            characterized += 1
    return characterized


def recovery_counts(real_points: list[RealPoint], x_value: float) -> tuple[int, int]:
    recovered = [
        point
        for point in real_points
        if point.attrs.temperature >= HIGH_T_MIN
        and not good(point.attrs, "baseline")
        and good(point.attrs, "alt", x_value)
    ]
    dip_recovered = [
        point for point in recovered if DIP_TAU_MIN <= point.tau_srh <= DIP_TAU_MAX
    ]
    return len(recovered), len(dip_recovered)


def monotonic_non_decreasing(values: list[int]) -> bool:
    return all(next_value >= value for value, next_value in zip(values, values[1:]))


def choose_recommended(results: list[dict], baseline_characterized: int) -> tuple[float, str]:
    qualifying = [
        result
        for result in results
        if result["decoy_characterized"] <= baseline_characterized
    ]
    if qualifying:
        return float(qualifying[-1]["X"]), "largest X with no net new false characterizations"
    best = min(results, key=lambda result: (result["decoy_characterized"], result["X"]))
    return (
        float(best["X"]),
        "no X had decoy_characterized <= baseline; chose X minimizing decoy_characterized",
    )


def write_figure(
    results: list[dict],
    baseline_characterized: int,
    recommended_x: float,
    out_path: Path,
) -> None:
    x_values = [result["X"] for result in results]
    recovery_overall = [result["recovery_overall"] for result in results]
    recovery_dip = [result["recovery_dip"] for result in results]
    decoy_characterized = [result["decoy_characterized"] for result in results]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax_recovery = plt.subplots(figsize=(8.0, 5.0))
    ax_fpr = ax_recovery.twinx()

    ax_recovery.plot(x_values, recovery_overall, marker="o", label="Real recovered")
    ax_recovery.plot(x_values, recovery_dip, marker="s", label="Real recovered, dip window")
    ax_fpr.plot(
        x_values,
        decoy_characterized,
        color="tab:red",
        marker="^",
        label="Decoy characterized",
    )
    ax_fpr.axhline(
        baseline_characterized,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="Baseline decoy characterized",
    )
    ax_recovery.axvline(
        recommended_x,
        color="0.25",
        linestyle=":",
        linewidth=1.2,
        label=f"Recommended X={recommended_x:g}",
    )

    ax_recovery.set_xlabel("Reduced chi-squared threshold X")
    ax_recovery.set_ylabel("Recovered real high-T points")
    ax_fpr.set_ylabel("Characterized decoy groups")
    ax_recovery.set_title("GOF Cut FPR Scan")
    ax_recovery.grid(True, alpha=0.25)

    lines, labels = ax_recovery.get_legend_handles_labels()
    lines2, labels2 = ax_fpr.get_legend_handles_labels()
    ax_recovery.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_table(results: list[dict]) -> None:
    print()
    print("Sweep:")
    print(
        "X      recovery  dip_recovery  recovery_frac  decoy_point  "
        "decoy_rate  decoy_char"
    )
    for result in results:
        print(
            f"{result['X']:>4g}  "
            f"{result['recovery_overall']:>8d}  "
            f"{result['recovery_dip']:>12d}  "
            f"{result['recovery_overall_frac']:>13.4f}  "
            f"{result['decoy_perpoint_pass']:>11d}  "
            f"{result['decoy_perpoint_rate']:>10.4f}  "
            f"{result['decoy_characterized']:>10d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan reduced-chi2 intensity-fit GOF thresholds against real high-T "
            "recoveries and combined decoy-null FPR."
        )
    )
    parser.add_argument(
        "--real-catalog",
        type=Path,
        default=REPO_ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5",
        help="Real minimal-caldet fit catalog HDF5.",
    )
    parser.add_argument(
        "--decoy-catalog",
        type=Path,
        default=REPO_ROOT / "decoy_fit_signed.h5",
        help="Signed decoy-null fit catalog HDF5.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT
        / "trap_completeness_method3"
        / "cache"
        / "gof_cut_fpr_scan_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=REPO_ROOT / "figures" / "gof_cut_fpr_scan.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--x-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_X_VALUES),
        help="Reduced-chi2 thresholds to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_values = [float(x) for x in args.x_values]

    print(f"Loading real catalog attrs: {args.real_catalog}")
    real_points = load_real_points(args.real_catalog)
    high_t_real_points = [
        point for point in real_points if point.attrs.temperature >= HIGH_T_MIN
    ]
    n_high_t_real_failed = sum(
        1 for point in high_t_real_points if not point.attrs.good_intensity_fit_attr
    )

    print(f"Loading decoy catalog attrs: {args.decoy_catalog}")
    decoy_groups = load_decoy_groups(args.decoy_catalog)
    n_decoy_high_t_points = sum(
        1
        for group in decoy_groups
        for attrs in group
        if attrs.temperature >= HIGH_T_MIN
    )

    baseline_decoy_perpoint = count_decoy_perpoint(decoy_groups, "baseline")
    baseline_decoy_perpoint_rate = (
        baseline_decoy_perpoint / n_decoy_high_t_points
        if n_decoy_high_t_points
        else 0.0
    )
    baseline_decoy_characterized = characterize_decoys(decoy_groups, "baseline")

    lower_sanity = 42 * 0.70
    upper_sanity = 42 * 1.30
    assert lower_sanity <= baseline_decoy_characterized <= upper_sanity, (
        "baseline decoy characterized count failed sanity check: "
        f"{baseline_decoy_characterized}"
    )

    results: list[dict] = []
    for x_value in x_values:
        recovery_overall, recovery_dip = recovery_counts(real_points, x_value)
        decoy_perpoint = count_decoy_perpoint(decoy_groups, "alt", x_value)
        decoy_perpoint_rate = (
            decoy_perpoint / n_decoy_high_t_points if n_decoy_high_t_points else 0.0
        )
        decoy_characterized = characterize_decoys(decoy_groups, "alt", x_value)
        results.append(
            {
                "X": x_value,
                "recovery_overall": recovery_overall,
                "recovery_dip": recovery_dip,
                "recovery_overall_frac": (
                    recovery_overall / n_high_t_real_failed
                    if n_high_t_real_failed
                    else 0.0
                ),
                "decoy_perpoint_pass": decoy_perpoint,
                "decoy_perpoint_rate": decoy_perpoint_rate,
                "decoy_characterized": decoy_characterized,
            }
        )

    assert monotonic_non_decreasing(
        [result["recovery_overall"] for result in results]
    ), "recovery_overall is not monotonically non-decreasing in X"
    assert monotonic_non_decreasing(
        [result["decoy_perpoint_pass"] for result in results]
    ), "decoy_perpoint_pass is not monotonically non-decreasing in X"
    assert results[-1]["recovery_overall"] > 0 and (
        results[0]["recovery_overall"] <= 0.5 * results[-1]["recovery_overall"]
    ), (
        "recovery_overall at X=1.5 is not small relative to X=20: "
        f"{results[0]['recovery_overall']} vs {results[-1]['recovery_overall']}"
    )

    recommended_x, recommended_reason = choose_recommended(
        results, baseline_decoy_characterized
    )
    recommended = next(result for result in results if result["X"] == recommended_x)

    summary = {
        "caveat": CAVEAT,
        "n_highT_real_failed": n_high_t_real_failed,
        "n_decoy_highT_points": n_decoy_high_t_points,
        "n_decoy_groups": len(decoy_groups),
        "recommended_X": recommended_x,
        "recommended_reason": recommended_reason,
        "baseline": {
            "recovery": 0,
            "decoy_perpoint_pass": baseline_decoy_perpoint,
            "decoy_perpoint_rate": baseline_decoy_perpoint_rate,
            "decoy_characterized": baseline_decoy_characterized,
        },
        "sweep": results,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    write_figure(results, baseline_decoy_characterized, recommended_x, args.figure)

    print()
    print(f"Caveat: {CAVEAT}")
    print(f"Baseline decoy characterized count: {baseline_decoy_characterized}")
    print(
        f"Recommended X={recommended_x:g}: recovered dip-window count="
        f"{recommended['recovery_dip']}, decoy characterized count="
        f"{recommended['decoy_characterized']}"
    )
    if "no X had" in recommended_reason:
        print(f"Recommendation note: {recommended_reason}")
    print_table(results)
    print()
    print(f"Wrote JSON: {args.summary_json}")
    print(f"Wrote figure: {args.figure}")


if __name__ == "__main__":
    main()

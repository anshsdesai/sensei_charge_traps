"""Stage 11: closure test of the naive measured/extrapolated efficiency curve.

Forward-models the pooled per-(trap, temperature) efficiency estimator from
charge_trap_figures.ipynb through the Stage 08 extended-tau p_det grid and the
Stage 05 amplitude prior, both unconditionally and conditioned on the trap
having been characterized (n_good >= 4), and attributes each tau bin to
temperatures and controlling cuts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "trap_completeness_method3"
CACHE = WORK / "cache"
FIGURES = CACHE / "figures"

sys.path.insert(0, str(WORK / "src"))
from validation_sensitivity import (  # noqa: E402
    interp_rows_by_amplitude,
    load_stage05,
    read_known_traps,
    tau_at_temperature,
)

N_GOOD = 4
OOB_CUT_LABEL = "tau_outside_stage08_grid"
MODEL_P_VALUE_SURVIVAL = 0.95  # Stage 08 injects the true model, so GOF fails only at the nominal rate.


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_stage08_extended(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return {
            "temperatures": h5["grid/temperature_K"][:].astype(float),
            "tau": h5["grid/tau_seconds"][:].astype(float),
            "amplitude": h5["grid/amplitude_electrons"][:].astype(float),
            "p_det": h5["results/p_det"][:].astype(float),
            "controlling_cut_fraction": h5["results/controlling_cut_fraction"][:].astype(float),
            "cut_labels": [s.decode() for s in h5["results/cut_labels"][:]],
            "random_seed": int(h5.attrs["random_seed"]),
        }


def stratified_depths(depth_samples: np.ndarray, count: int) -> np.ndarray:
    """Deterministic quantile-stratified draws from the depth prior."""
    quantiles = (np.arange(count) + 0.5) / count
    return np.quantile(np.sort(depth_samples), quantiles)


def interp_grid_rows(
    values_tau_amp: np.ndarray, log_tau_grid: np.ndarray, log_tau_points: np.ndarray
) -> np.ndarray:
    """Interpolate a (tau, amplitude) grid slab along log-tau for each point.

    Returns (n_points, n_amplitude); rows for points outside the tau grid are zero.
    """
    n_points = log_tau_points.size
    n_amp = values_tau_amp.shape[1]
    rows = np.zeros((n_points, n_amp), dtype=np.float64)
    in_range = (log_tau_points >= log_tau_grid[0]) & (log_tau_points <= log_tau_grid[-1])
    for a_index in range(n_amp):
        rows[in_range, a_index] = np.interp(
            log_tau_points[in_range], log_tau_grid, values_tau_amp[:, a_index]
        )
    return rows


def leave_one_out_tail(p: np.ndarray, exclude_index: int, tail_from: int) -> np.ndarray:
    """P(N_{-t} >= tail_from) per (trap, sample) for Bernoulli matrix p (T, N, S)."""
    n_temp, n_trap, n_samp = p.shape
    dp = np.zeros((n_trap, n_samp, tail_from), dtype=np.float64)
    dp[:, :, 0] = 1.0
    for t_index in range(n_temp):
        if t_index == exclude_index:
            continue
        q = p[t_index]
        old = dp.copy()
        dp[:, :, 0] = old[:, :, 0] * (1.0 - q)
        for k in range(1, tail_from):
            dp[:, :, k] = old[:, :, k] * (1.0 - q) + old[:, :, k - 1] * q
    return 1.0 - dp.sum(axis=2)


def characterization_probability(p: np.ndarray, n_good: int) -> np.ndarray:
    """P(N >= n_good) per (trap, sample) for Bernoulli matrix p (T, N, S)."""
    n_temp, n_trap, n_samp = p.shape
    dp = np.zeros((n_trap, n_samp, n_good), dtype=np.float64)
    dp[:, :, 0] = 1.0
    for t_index in range(n_temp):
        q = p[t_index]
        old = dp.copy()
        dp[:, :, 0] = old[:, :, 0] * (1.0 - q)
        for k in range(1, n_good):
            dp[:, :, k] = old[:, :, k] * (1.0 - q) + old[:, :, k - 1] * q
    return 1.0 - dp.sum(axis=2)


def observed_pvalue_survival(cutflow_csv: Path, temperatures: np.ndarray) -> np.ndarray:
    """Empirical P(p_value > 0.05) per temperature among bright, in-window real curves.

    Restricted to points with tau(T) inside the dtph window and passing both 3-sigma
    peak cuts, so the survival isolates the goodness-of-fit failure ramp from
    amplitude effects.
    """
    import csv as csv_module

    passed: dict[float, int] = {float(T): 0 for T in temperatures}
    totals: dict[float, int] = {float(T): 0 for T in temperatures}
    with cutflow_csv.open(newline="") as handle:
        for row in csv_module.DictReader(handle):
            tau = float(row["tau_model_seconds"])
            if not (7.5e-4 < tau < 15.5):
                continue
            if row["intensity_fit_failed"] == "True":
                continue
            max_intensity = float(row["max_intensity"])
            if max_intensity < 3.0 * float(row["mean_intensity_err"]):
                continue
            if max_intensity < 3.0 * float(row["image_sigma"]):
                continue
            T = float(row["temperature_K"])
            totals[T] += 1
            if float(row["fit_p_value"]) > 0.05:
                passed[T] += 1
    survival = np.array(
        [passed[float(T)] / totals[float(T)] if totals[float(T)] > 0 else np.nan for T in temperatures]
    )
    return survival


def binned_efficiency(
    tau_points: np.ndarray, weights: np.ndarray | None, bins: np.ndarray, totals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    numerator, _ = np.histogram(tau_points, bins=bins, weights=weights)
    eff = np.divide(numerator, totals, out=np.zeros_like(numerator, dtype=float), where=totals > 0)
    return numerator, eff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngood4-csv", type=Path, default=CACHE / "01_records_ngood4.csv")
    parser.add_argument("--stage08-h5", type=Path, default=CACHE / "08_pdet_grid_tau1000_v1.h5")
    parser.add_argument("--stage05-npz", type=Path, default=CACHE / "05_amplitude_prior_v1.npz")
    parser.add_argument("--depth-variant", default="default", choices=["default", "faint_0p5", "faint_0p25"])
    parser.add_argument(
        "--observed-cutflow-csv",
        type=Path,
        default=CACHE / "11_observed_cutflow_v1.csv",
        help="If present, adds a hybrid prediction with the empirical per-temperature GOF survival.",
    )
    parser.add_argument("--amplitude-samples", type=int, default=1024)
    parser.add_argument("--bin-min", type=float, default=1e-7)
    parser.add_argument("--bin-max", type=float, default=1e8)
    parser.add_argument("--bin-edges", type=int, default=75)
    parser.add_argument("--min-total-for-pull", type=int, default=20)
    parser.add_argument("--output-tag", default="v1")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    records = read_known_traps(args.ngood4_csv)
    stage08 = load_stage08_extended(args.stage08_h5)
    stage05 = load_stage05(args.stage05_npz)

    temperatures = stage08["temperatures"]
    n_temp = temperatures.size
    n_trap = records["E_eV"].size
    pc = np.array(
        [stage05["pc"][np.where(stage05["temperatures"] == T)[0][0]] for T in temperatures]
    )
    depths = stratified_depths(stage05["depth_variants"][args.depth_variant], args.amplitude_samples)
    n_samp = depths.size

    # tau(T) trajectories and observed measured/extrapolated tags.
    tau_T = tau_at_temperature(records["tau_135_seconds"], records["E_eV"], temperatures)  # (N, T)
    measured = np.zeros((n_trap, n_temp), dtype=bool)
    for i, good_set in enumerate(records["good_temperatures"]):
        for t_index, T in enumerate(temperatures):
            measured[i, t_index] = float(T) in good_set

    bins = np.geomspace(args.bin_min, args.bin_max, args.bin_edges)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    n_bins = bin_centers.size
    tau_flat = tau_T.reshape(-1)

    total_counts, _ = np.histogram(tau_flat, bins=bins)
    obs_counts, eff_obs = binned_efficiency(tau_T[measured], None, bins, total_counts)
    with np.errstate(invalid="ignore"):
        eff_obs_err = np.where(
            total_counts > 0, np.sqrt(eff_obs * (1.0 - eff_obs) / np.maximum(total_counts, 1)), 0.0
        )

    # Per-(temperature, trap, sample) detection probabilities from the Stage 08 grid.
    log_tau_grid = np.log(stage08["tau"])
    p = np.zeros((n_temp, n_trap, n_samp), dtype=np.float64)
    oob = np.zeros((n_trap, n_temp), dtype=bool)
    cutfrac = np.zeros((n_temp, n_trap, len(stage08["cut_labels"])), dtype=np.float64)
    for t_index in range(n_temp):
        log_tau_points = np.log(tau_T[:, t_index])
        oob[:, t_index] = (log_tau_points < log_tau_grid[0]) | (log_tau_points > log_tau_grid[-1])
        rows = interp_grid_rows(stage08["p_det"][t_index], log_tau_grid, log_tau_points)
        p[t_index] = interp_rows_by_amplitude(rows, stage08["amplitude"], depths * pc[t_index])
        # Amplitude-marginalized controlling-cut composition for attribution.
        for c_index in range(len(stage08["cut_labels"])):
            cut_rows = interp_grid_rows(
                stage08["controlling_cut_fraction"][t_index, :, :, c_index],
                log_tau_grid,
                log_tau_points,
            )
            cutfrac[t_index, :, c_index] = interp_rows_by_amplitude(
                cut_rows, stage08["amplitude"], depths * pc[t_index]
            ).mean(axis=1)

    # Off-grid points: all probability mass goes to the synthetic out-of-band category.
    cut_labels = stage08["cut_labels"] + [OOB_CUT_LABEL]
    oob_frac = oob.T.astype(float)  # (T, N)
    cutfrac = cutfrac * (1.0 - oob_frac[:, :, None])
    cutfrac = np.concatenate([cutfrac, oob_frac[:, :, None]], axis=2)

    # Unconditional prediction.
    w_unc = p.mean(axis=2)  # (T, N)

    # Conditional-on-characterization prediction via leave-one-out Poisson binomial.
    p_char = characterization_probability(p, N_GOOD)  # (N, S)
    den = p_char.mean(axis=1)  # (N,)
    w_cond = np.zeros_like(w_unc)
    for t_index in range(n_temp):
        tail = leave_one_out_tail(p, t_index, N_GOOD - 1)  # (N, S)
        w_cond[t_index] = (p[t_index] * tail).mean(axis=1)
    den_floor = 1e-6
    bad_den = den < den_floor
    safe_den = np.where(bad_den, 1.0, den)
    w_cond = w_cond / safe_den[None, :]
    w_cond[:, bad_den] = w_unc[:, bad_den]
    w_cond = np.clip(w_cond, 0.0, 1.0)

    pred_unc_counts, eff_unc = binned_efficiency(tau_flat, w_unc.T.reshape(-1), bins, total_counts)
    pred_cond_counts, eff_cond = binned_efficiency(tau_flat, w_cond.T.reshape(-1), bins, total_counts)

    # Hybrid: rescale the model's nominal GOF survival to the empirically observed
    # per-temperature survival of bright in-window curves (one scalar per T; all
    # tau structure still comes from the Stage 08 model).
    gof_survival = None
    w_hyb = None
    eff_hyb = np.full(n_bins, np.nan)
    pred_hyb_counts = np.zeros(n_bins)
    if args.observed_cutflow_csv.exists():
        gof_survival = observed_pvalue_survival(args.observed_cutflow_csv, temperatures)
        scale = np.where(np.isfinite(gof_survival), gof_survival / MODEL_P_VALUE_SURVIVAL, 1.0)
        w_hyb = np.clip(w_unc * scale[:, None], 0.0, 1.0)
        pred_hyb_counts, eff_hyb = binned_efficiency(tau_flat, w_hyb.T.reshape(-1), bins, total_counts)

    # Per-temperature decomposition (n_bins, n_temp).
    obs_by_T = np.zeros((n_bins, n_temp))
    pred_cond_by_T = np.zeros((n_bins, n_temp))
    pred_hyb_by_T = np.zeros((n_bins, n_temp))
    total_by_T = np.zeros((n_bins, n_temp))
    for t_index in range(n_temp):
        total_by_T[:, t_index], _ = np.histogram(tau_T[:, t_index], bins=bins)
        obs_by_T[:, t_index], _ = np.histogram(tau_T[measured[:, t_index], t_index], bins=bins)
        pred_cond_by_T[:, t_index], _ = np.histogram(
            tau_T[:, t_index], bins=bins, weights=w_cond[t_index]
        )
        if w_hyb is not None:
            pred_hyb_by_T[:, t_index], _ = np.histogram(
                tau_T[:, t_index], bins=bins, weights=w_hyb[t_index]
            )

    # Controlling-cut attribution per bin (count-weighted mean composition).
    cut_by_bin = np.zeros((n_bins, len(cut_labels)))
    for c_index in range(len(cut_labels)):
        weighted, _ = np.histogram(tau_flat, bins=bins, weights=cutfrac[:, :, c_index].T.reshape(-1))
        cut_by_bin[:, c_index] = np.divide(
            weighted, total_counts, out=np.zeros(n_bins), where=total_counts > 0
        )

    # Closure metrics over well-populated bins.
    variants = [("unconditional", eff_unc), ("conditional", eff_cond)]
    if gof_survival is not None:
        variants.append(("hybrid", eff_hyb))

    pull_mask = total_counts >= args.min_total_for_pull
    pulls = {}
    for name, eff_pred in variants:
        var = eff_pred * (1.0 - eff_pred) / np.maximum(total_counts, 1) + 1e-9
        pull = (eff_obs - eff_pred) / np.sqrt(var)
        pulls[name] = pull
    feature_windows = {
        "plateau_1e-4_1e-2": (1e-4, 1e-2),
        "dip_3e-3_3e-2": (3e-3, 3e-2),
        "peak_1e-1_2e0": (1e-1, 2.0),
        "falloff_2e0_3e1": (2.0, 30.0),
    }

    def window_summary(curve: np.ndarray, lo: float, hi: float, agg) -> float:
        sel = (bin_centers >= lo) & (bin_centers <= hi) & (total_counts > 0)
        return float(agg(curve[sel])) if np.any(sel) else float("nan")

    features = {}
    for window_name, (lo, hi) in feature_windows.items():
        agg = np.min if window_name.startswith("dip") else np.mean
        features[window_name] = {
            "observed": window_summary(eff_obs, lo, hi, agg),
            **{
                f"predicted_{name}": window_summary(eff_pred, lo, hi, agg)
                for name, eff_pred in variants
            },
        }

    pass_index = cut_labels.index("pass") if "pass" in cut_labels else None
    pass_consistency = (
        float(np.max(np.abs(cut_by_bin[pull_mask, pass_index] - eff_unc[pull_mask])))
        if pass_index is not None
        else None
    )

    output_h5 = CACHE / f"11_naive_efficiency_closure_{args.output_tag}.h5"
    summary_path = CACHE / f"11_naive_efficiency_closure_summary.json"
    metadata = {
        "producing_stage": "11_naive_efficiency_closure",
        "produced_at": now_local_iso(),
        "code_path": str(Path(__file__).resolve()),
        "inputs": {
            "ngood4_csv": str(args.ngood4_csv.resolve()),
            "stage08_h5": str(args.stage08_h5.resolve()),
            "stage05_npz": str(args.stage05_npz.resolve()),
            "observed_cutflow_csv": str(args.observed_cutflow_csv.resolve())
            if args.observed_cutflow_csv.exists()
            else None,
        },
        "hybrid_model": None
        if gof_survival is None
        else {
            "definition": "p_det scaled by observed bright in-window GOF survival per temperature over the nominal model survival",
            "model_p_value_survival": MODEL_P_VALUE_SURVIVAL,
            "observed_gof_survival_by_temperature": {
                f"{T:g}": float(s) for T, s in zip(temperatures, gof_survival)
            },
        },
        "depth_variant": args.depth_variant,
        "amplitude_samples": int(n_samp),
        "amplitude_sampling": "deterministic quantile-stratified midpoints of the Stage 05 depth prior",
        "stage08_random_seed": stage08["random_seed"],
        "bins": {"min": args.bin_min, "max": args.bin_max, "edge_count": args.bin_edges},
        "n_good": N_GOOD,
        "trap_count": int(n_trap),
        "conditional_denominator_floor": den_floor,
        "conditional_denominator_fallback_count": int(np.sum(bad_den)),
    }

    with h5py.File(output_h5, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(metadata)
        grid = h5.create_group("grid")
        grid.create_dataset("bin_edges_seconds", data=bins)
        grid.create_dataset("bin_centers_seconds", data=bin_centers)
        grid.create_dataset("temperature_K", data=temperatures)
        grid.create_dataset("cut_labels", data=np.array(cut_labels, dtype="S"))
        res = h5.create_group("results")
        res.create_dataset("total_counts", data=total_counts)
        res.create_dataset("observed_measured_counts", data=obs_counts)
        res.create_dataset("efficiency_observed", data=eff_obs)
        res.create_dataset("efficiency_observed_err", data=eff_obs_err)
        res.create_dataset("predicted_measured_unconditional", data=pred_unc_counts)
        res.create_dataset("predicted_measured_conditional", data=pred_cond_counts)
        res.create_dataset("efficiency_predicted_unconditional", data=eff_unc)
        res.create_dataset("efficiency_predicted_conditional", data=eff_cond)
        res.create_dataset("pull_unconditional", data=pulls["unconditional"])
        res.create_dataset("pull_conditional", data=pulls["conditional"])
        res.create_dataset("pull_mask_total_ge_threshold", data=pull_mask)
        if gof_survival is not None:
            res.create_dataset("predicted_measured_hybrid", data=pred_hyb_counts)
            res.create_dataset("efficiency_predicted_hybrid", data=eff_hyb)
            res.create_dataset("pull_hybrid", data=pulls["hybrid"])
            res.create_dataset("observed_gof_survival_by_temperature", data=gof_survival)
        dec = h5.create_group("decomposition")
        dec.create_dataset("total_counts_by_temperature", data=total_by_T)
        dec.create_dataset("observed_measured_by_temperature", data=obs_by_T)
        dec.create_dataset("predicted_conditional_by_temperature", data=pred_cond_by_T)
        if gof_survival is not None:
            dec.create_dataset("predicted_hybrid_by_temperature", data=pred_hyb_by_T)
        dec.create_dataset("controlling_cut_fraction_by_bin", data=cut_by_bin)
        per_trap = h5.create_group("per_trap")
        per_trap.create_dataset("tau_at_temperature_seconds", data=tau_T)
        per_trap.create_dataset("measured", data=measured)
        per_trap.create_dataset("weight_unconditional", data=w_unc.T)
        per_trap.create_dataset("weight_conditional", data=w_cond.T)
        per_trap.create_dataset("p_characterized_mean", data=den)
        per_trap.create_dataset("tau_outside_grid", data=oob)

    figures = []
    if not args.no_figures:
        FIGURES.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        valid = total_counts > 0
        ax.errorbar(
            bin_centers[valid], eff_obs[valid], yerr=eff_obs_err[valid],
            fmt="o", color="black", markersize=3.5, capsize=3, label="Observed (naive estimator)",
        )
        ax.plot(bin_centers[valid], eff_unc[valid], color="#d62728", lw=1.8,
                label="Method 3 prediction (unconditional)")
        ax.plot(bin_centers[valid], eff_cond[valid], color="#1f77b4", lw=1.8,
                label="Method 3 prediction (conditional on characterization)")
        if gof_survival is not None:
            ax.plot(bin_centers[valid], eff_hyb[valid], color="#2ca02c", lw=2.2,
                    label="Hybrid: Method 3 x empirical GOF survival per T")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel(r"$\tau_e$ [s]")
        ax.set_ylabel("Efficiency of Measurement")
        ax.legend(frameon=False)
        out = FIGURES / "11_closure_overlay.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(str(out))

        fig, ax = plt.subplots(figsize=(11, 3.5))
        colors = {"unconditional": "#d62728", "conditional": "#1f77b4", "hybrid": "#2ca02c"}
        for name, _ in variants:
            ax.plot(bin_centers[pull_mask], pulls[name][pull_mask], "o-", ms=3,
                    color=colors[name], label=name)
        ax.axhline(0, color="grey", lw=0.8)
        for level in (-3, 3):
            ax.axhline(level, color="grey", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau_e$ [s]")
        ax.set_ylabel("(obs - pred) / binomial err")
        ax.legend(frameon=False, fontsize=8)
        out = FIGURES / "11_closure_pulls.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(str(out))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        extent = None
        for ax, (matrix, title) in zip(
            axes,
            [
                (total_by_T, "Total points"),
                (obs_by_T, "Observed measured"),
                (pred_hyb_by_T if gof_survival is not None else pred_cond_by_T,
                 "Predicted measured (hybrid)" if gof_survival is not None else "Predicted measured (conditional)"),
            ],
        ):
            shown = np.log10(np.maximum(matrix, 0.5))
            im = ax.pcolormesh(bins, np.arange(n_temp + 1), shown.T, cmap="viridis")
            ax.set_xscale("log")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(r"$\tau_e$ [s]")
            fig.colorbar(im, ax=ax, label="log10 counts")
        axes[0].set_yticks(np.arange(n_temp) + 0.5)
        axes[0].set_yticklabels([f"{T:g}" for T in temperatures], fontsize=7)
        axes[0].set_ylabel("Temperature [K]")
        out = FIGURES / "11_temperature_decomposition.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(str(out))

        fig, ax = plt.subplots(figsize=(11, 5.5))
        order = np.argsort([-cut_by_bin[:, i].max() for i in range(len(cut_labels))])
        bottom = np.zeros(n_bins)
        cmap = plt.get_cmap("tab10")
        for rank, c_index in enumerate(order):
            ax.fill_between(
                bin_centers, bottom, bottom + cut_by_bin[:, c_index],
                step="mid", color=cmap(rank % 10), label=cut_labels[c_index], alpha=0.85,
            )
            bottom = bottom + cut_by_bin[:, c_index]
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"$\tau_e$ [s]")
        ax.set_ylabel("Controlling-outcome fraction")
        ax.legend(frameon=False, fontsize=7, loc="center left")
        out = FIGURES / "11_cut_attribution.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        figures.append(str(out))

    summary = {
        **metadata,
        "closure_metrics": {
            name: {
                "bins_used": int(np.sum(pull_mask)),
                "chi2": float(np.sum(pulls[name][pull_mask] ** 2)),
                "chi2_per_bin": float(np.mean(pulls[name][pull_mask] ** 2)),
                "max_abs_pull": float(np.max(np.abs(pulls[name][pull_mask]))),
                "mean_abs_eff_difference": float(
                    np.mean(np.abs((eff_obs - eff)[pull_mask]))
                ),
            }
            for name, eff in variants
        },
        "feature_windows": features,
        "pass_fraction_vs_unconditional_max_abs_diff": pass_consistency,
        "observed_curve_check": {
            "total_points": int(tau_flat.size),
            "measured_points": int(measured.sum()),
            "peak_efficiency": float(np.max(eff_obs)),
            "peak_tau_seconds": float(bin_centers[int(np.argmax(eff_obs))]),
        },
        "outputs": {"h5": str(output_h5), "figures": figures},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["closure_metrics"], indent=2))
    print(json.dumps(summary["feature_windows"], indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

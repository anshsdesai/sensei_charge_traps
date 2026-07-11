"""Held-out closure validation for the signed-refit empirical noise model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2, kstest

from signed_refit_controls import COL_REGIONS, ROW_REGIONS, file_sha256
from signed_refit_noise_model import NOISE_MODEL_VERSION


CLOSURE_VERSION = "signed-refit-noise-closure-v2"
DEFAULT_MODEL = Path("signed_refit_noise_model.h5")
DEFAULT_OUTPUT = Path("signed_refit_noise_closure.npz")
DEFAULT_REPORT = Path("signed_refit_noise_closure.md")
DEFAULT_FIGURE_DIR = Path("figures/signed_refit_noise")

PUMP_GRID_SIZE = 256
TRIAL_DELTA_CHI2 = 11.83

GLOBAL_WIDTH_RANGE = (0.90, 1.10)
STRATIFIED_WIDTH_RANGE = (0.85, 1.15)
MAX_GLOBAL_MEAN = 0.05
MAX_TEMPERATURE_CORRELATION = 0.10
MAX_TEMPERATURE_P05_RATE = 0.10
MAX_TEMPERATURE_THREE_SIGMA_RATE = 0.02


def offdiag_max(correlation: np.ndarray) -> float:
    if correlation.shape[0] <= 1:
        return 0.0
    return float(np.max(np.abs(correlation - np.eye(correlation.shape[0]))))


def gls_constant_chi2(curves: np.ndarray, template: np.ndarray, precision: np.ndarray):
    y = curves - template[None, :]
    ones = np.ones(y.shape[1])
    p1 = precision @ ones
    denominator = float(ones @ p1)
    offsets = (y @ p1) / denominator
    residual = y - offsets[:, None]
    chi2_values = np.einsum("ni,ij,nj->n", residual, precision, residual)
    return chi2_values, residual, offsets


def trial_pump_statistic(
    curves: np.ndarray,
    template: np.ndarray,
    precision: np.ndarray,
    seconds: np.ndarray,
) -> np.ndarray:
    y = curves - template[None, :]
    ones = np.ones(seconds.size)
    p1 = precision @ ones
    constant_denominator = float(ones @ p1)
    constant_projection = np.outer((y @ p1) / constant_denominator, ones)
    residual = y - constant_projection

    tau_grid = np.geomspace(float(np.min(seconds)) / 10.0, float(np.max(seconds)) * 10.0, PUMP_GRID_SIZE)
    shape = np.exp(-seconds[None, :] / tau_grid[:, None]) - np.exp(
        -8.0 * seconds[None, :] / tau_grid[:, None]
    )
    shape -= np.outer((shape @ p1) / constant_denominator, ones)
    pshape = shape @ precision
    denominator = np.einsum("ki,ki->k", pshape, shape)
    numerator = residual @ pshape.T
    delta = numerator**2 / denominator[None, :]
    return np.max(delta, axis=1)


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def grouped_width(records: list[dict], key: str) -> dict[int, float]:
    groups = defaultdict(list)
    for record in records:
        groups[int(record[key])].append(record["whitened"].ravel())
    return {
        group: float(np.std(np.concatenate(values), ddof=1))
        for group, values in sorted(groups.items())
    }


def analyze(model_path: Path, output_path: Path, report_path: Path, figure_dir: Path):
    records = []
    all_z = []
    all_pvalues = []
    all_chi2 = []
    all_dof = []
    all_trial_delta = []
    marginal_by_temperature_dtph = defaultdict(list)
    temperature_z = defaultdict(list)
    temperature_pvalues = defaultdict(list)
    temperature_trial = defaultdict(list)
    brightness_bins = defaultdict(lambda: {"z": [], "p": [], "trial": []})

    with h5py.File(model_path, "r") as h5:
        if h5.attrs.get("version", "") != NOISE_MODEL_VERSION:
            raise ValueError(f"Unexpected noise-model version: {h5.attrs.get('version')}")
        controls = h5["controls"]
        global_quadrant = controls["quadrant"][()]
        global_region = controls["region"][()]
        global_split = controls["split"][()]
        global_static = controls["static_pair_intensity"][()]

        for temp_name in sorted(
            (key for key in h5 if key.startswith("temp_")),
            key=lambda value: int(value.split("_")[1]),
        ):
            temperature = int(temp_name.split("_")[1])
            temp_group = h5[temp_name]
            seconds = temp_group["seconds"][()]
            dof = seconds.size - 1
            for quadrant in range(4):
                qsel = global_quadrant == quadrant
                q_regions = global_region[qsel]
                q_splits = global_split[qsel]
                q_static = global_static[qsel]
                intensity = temp_group[f"quad_{quadrant}/intensity"]
                for region in range(ROW_REGIONS * COL_REGIONS):
                    validation = (q_regions == region) & (q_splits == 1)
                    curves = intensity[validation]
                    static = q_static[validation]
                    rgroup = temp_group[f"quad_{quadrant}/region_{region}"]
                    covariance = rgroup["covariance"][()]
                    precision = rgroup["precision"][()]
                    template = rgroup["null_template"][()]

                    pair_centered = curves - np.median(curves, axis=1, keepdims=True)
                    residual_for_whitening = pair_centered - template[None, :]
                    cholesky = np.linalg.cholesky(covariance)
                    whitened = solve_triangular(
                        cholesky,
                        residual_for_whitening.T,
                        lower=True,
                    ).T

                    chi2_values, constant_residual, _ = gls_constant_chi2(
                        curves,
                        template,
                        precision,
                    )
                    pvalues = chi2.sf(chi2_values, dof)
                    trial_delta = trial_pump_statistic(
                        curves,
                        template,
                        precision,
                        seconds,
                    )

                    sigma = np.sqrt(np.diag(covariance))
                    marginal = residual_for_whitening / sigma[None, :]
                    for delay_index in range(seconds.size):
                        marginal_by_temperature_dtph[(temperature, delay_index)].append(
                            marginal[:, delay_index]
                        )

                    quantiles = np.quantile(np.abs(static), [0.25, 0.5, 0.75])
                    brightness_index = np.digitize(np.abs(static), quantiles)
                    for brightness_bin in range(4):
                        selected = brightness_index == brightness_bin
                        brightness_bins[brightness_bin]["z"].append(whitened[selected].ravel())
                        brightness_bins[brightness_bin]["p"].append(pvalues[selected])
                        brightness_bins[brightness_bin]["trial"].append(trial_delta[selected])

                    record = {
                        "temperature": temperature,
                        "quadrant": quadrant,
                        "region": region,
                        "dof": dof,
                        "whitened": whitened,
                        "pvalues": pvalues,
                        "chi2": chi2_values,
                        "trial_delta": trial_delta,
                        "whitened_width": float(np.std(whitened, ddof=1)),
                        "whitened_mean": float(np.mean(whitened)),
                        "three_sigma_rate": float(np.mean(np.abs(whitened) > 3)),
                        "p05_rate": float(np.mean(pvalues < 0.05)),
                        "p01_rate": float(np.mean(pvalues < 0.01)),
                        "trial_rate": float(np.mean(trial_delta >= TRIAL_DELTA_CHI2)),
                        "max_whitened_correlation": offdiag_max(np.corrcoef(whitened, rowvar=False)),
                        "constant_residual_rms": float(np.sqrt(np.mean(constant_residual**2))),
                    }
                    records.append(record)
                    all_z.append(whitened.ravel())
                    all_pvalues.append(pvalues)
                    all_chi2.append(chi2_values)
                    all_dof.append(np.full(chi2_values.size, dof))
                    all_trial_delta.append(trial_delta)
                    temperature_z[temperature].append(whitened)
                    temperature_pvalues[temperature].append(pvalues)
                    temperature_trial[temperature].append(trial_delta)

    all_z_array = np.concatenate(all_z)
    all_pvalue_array = np.concatenate(all_pvalues)
    all_chi2_array = np.concatenate(all_chi2)
    all_dof_array = np.concatenate(all_dof)
    all_trial_array = np.concatenate(all_trial_delta)

    temperature_summary = {}
    for temperature in sorted(temperature_z):
        z = np.concatenate(temperature_z[temperature], axis=0)
        pvalues = np.concatenate(temperature_pvalues[temperature])
        trial = np.concatenate(temperature_trial[temperature])
        temperature_summary[temperature] = {
            "whitened_mean": float(np.mean(z)),
            "whitened_width": float(np.std(z, ddof=1)),
            "three_sigma_rate": float(np.mean(np.abs(z) > 3)),
            "max_abs_correlation": offdiag_max(np.corrcoef(z, rowvar=False)),
            "p05_rate": float(np.mean(pvalues < 0.05)),
            "p01_rate": float(np.mean(pvalues < 0.01)),
            "trial_delta_11p83_rate": float(np.mean(trial >= TRIAL_DELTA_CHI2)),
            "trial_delta_p95": float(np.percentile(trial, 95)),
            "trial_delta_p99": float(np.percentile(trial, 99)),
            "n_curves": int(pvalues.size),
        }

    quadrant_width = grouped_width(records, "quadrant")
    region_width = grouped_width(records, "region")
    cell_widths = np.asarray([record["whitened_width"] for record in records])
    marginal_widths = {
        key: float(np.std(np.concatenate(values), ddof=1))
        for key, values in marginal_by_temperature_dtph.items()
    }
    brightness_summary = {}
    for brightness_bin, values in sorted(brightness_bins.items()):
        z = np.concatenate(values["z"])
        pvalues = np.concatenate(values["p"])
        trial = np.concatenate(values["trial"])
        brightness_summary[brightness_bin] = {
            "whitened_width": float(np.std(z, ddof=1)),
            "p05_rate": float(np.mean(pvalues < 0.05)),
            "trial_rate": float(np.mean(trial >= TRIAL_DELTA_CHI2)),
        }

    uniform_ks = kstest(all_pvalue_array, "uniform")
    metrics = {
        "version": CLOSURE_VERSION,
        "noise_model_sha256": file_sha256(model_path),
        "curve_count": int(all_pvalue_array.size),
        "whitened_coordinate_count": int(all_z_array.size),
        "global": {
            "whitened_mean": float(np.mean(all_z_array)),
            "whitened_width": float(np.std(all_z_array, ddof=1)),
            "three_sigma_rate": float(np.mean(np.abs(all_z_array) > 3)),
            "p05_rate": float(np.mean(all_pvalue_array < 0.05)),
            "p01_rate": float(np.mean(all_pvalue_array < 0.01)),
            "uniform_pvalue_ks_statistic": float(uniform_ks.statistic),
            "uniform_pvalue_ks_pvalue": float(uniform_ks.pvalue),
            "median_reduced_chi2": float(np.median(all_chi2_array / all_dof_array)),
            "trial_delta_11p83_rate": float(np.mean(all_trial_array >= TRIAL_DELTA_CHI2)),
            "trial_delta_p95": float(np.percentile(all_trial_array, 95)),
            "trial_delta_p99": float(np.percentile(all_trial_array, 99)),
            "trial_delta_p999": float(np.percentile(all_trial_array, 99.9)),
        },
        "temperature": temperature_summary,
        "quadrant_width": quadrant_width,
        "region_width": region_width,
        "cell_width": summarize_values(cell_widths),
        "marginal_width": summarize_values(np.asarray(list(marginal_widths.values()))),
        "brightness": brightness_summary,
        "warm_temperatures": {
            str(temp): temperature_summary[temp] for temp in (200, 203, 207, 210)
        },
    }

    failures = []
    global_width = metrics["global"]["whitened_width"]
    if not GLOBAL_WIDTH_RANGE[0] <= global_width <= GLOBAL_WIDTH_RANGE[1]:
        failures.append(f"global whitened width {global_width:.3f}")
    if abs(metrics["global"]["whitened_mean"]) > MAX_GLOBAL_MEAN:
        failures.append(f"global whitened mean {metrics['global']['whitened_mean']:.3f}")
    for temperature, values in temperature_summary.items():
        if not STRATIFIED_WIDTH_RANGE[0] <= values["whitened_width"] <= STRATIFIED_WIDTH_RANGE[1]:
            failures.append(
                f"{temperature} K whitened width {values['whitened_width']:.3f}"
            )
        if values["max_abs_correlation"] > MAX_TEMPERATURE_CORRELATION:
            failures.append(
                f"{temperature} K max residual correlation "
                f"{values['max_abs_correlation']:.3f}"
            )
        if values["p05_rate"] > MAX_TEMPERATURE_P05_RATE:
            failures.append(f"{temperature} K p<0.05 rate {values['p05_rate']:.3f}")
        if values["three_sigma_rate"] > MAX_TEMPERATURE_THREE_SIGMA_RATE:
            failures.append(
                f"{temperature} K |z|>3 rate {values['three_sigma_rate']:.3f}"
            )
    for quadrant, width in quadrant_width.items():
        if not STRATIFIED_WIDTH_RANGE[0] <= width <= STRATIFIED_WIDTH_RANGE[1]:
            failures.append(f"quadrant {quadrant} whitened width {width:.3f}")
    if metrics["cell_width"]["p95"] > 1.30 or metrics["cell_width"]["p05"] < 0.70:
        failures.append(
            "regional cell-width tails outside [0.70, 1.30]: "
            f"p05={metrics['cell_width']['p05']:.3f}, "
            f"p95={metrics['cell_width']['p95']:.3f}"
        )

    metrics["acceptance"] = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "thresholds": {
            "global_width": list(GLOBAL_WIDTH_RANGE),
            "stratified_width": list(STRATIFIED_WIDTH_RANGE),
            "max_global_mean": MAX_GLOBAL_MEAN,
            "max_temperature_correlation": MAX_TEMPERATURE_CORRELATION,
            "max_temperature_p05_rate": MAX_TEMPERATURE_P05_RATE,
            "max_temperature_three_sigma_rate": MAX_TEMPERATURE_THREE_SIGMA_RATE,
            "cell_width_p05_min": 0.70,
            "cell_width_p95_max": 1.30,
        },
    }

    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metrics, sort_keys=True)),
        global_whitened=all_z_array.astype(np.float32),
        global_pvalues=all_pvalue_array.astype(np.float32),
        global_trial_delta=all_trial_array.astype(np.float32),
        cell_temperature=np.asarray([record["temperature"] for record in records], dtype=np.int16),
        cell_quadrant=np.asarray([record["quadrant"] for record in records], dtype=np.int8),
        cell_region=np.asarray([record["region"] for record in records], dtype=np.int8),
        cell_width=cell_widths.astype(np.float32),
        cell_p05_rate=np.asarray([record["p05_rate"] for record in records], dtype=np.float32),
        cell_trial_rate=np.asarray([record["trial_rate"] for record in records], dtype=np.float32),
    )
    make_figures(figure_dir, metrics, all_z_array, all_pvalue_array, all_trial_array)
    write_report(report_path, metrics)
    return metrics


def make_figures(
    figure_dir: Path,
    metrics: dict,
    whitened: np.ndarray,
    pvalues: np.ndarray,
    trial_delta: np.ndarray,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(whitened, bins=np.linspace(-6, 6, 121), density=True, histtype="step")
    x = np.linspace(-6, 6, 500)
    axes[0].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="black")
    axes[0].set(xlabel="Whitened residual", ylabel="Density", yscale="log", ylim=(1e-6, 1))
    axes[1].hist(pvalues, bins=np.linspace(0, 1, 51), density=True, histtype="step")
    axes[1].axhline(1, color="black")
    axes[1].set(xlabel="Constant-model chi-square p-value", ylabel="Density")
    axes[2].hist(trial_delta, bins=np.linspace(0, 40, 81), density=True, histtype="step")
    axes[2].axvline(TRIAL_DELTA_CHI2, color="red", linestyle="--")
    axes[2].set(xlabel="Max trial pump delta chi-square", ylabel="Density", yscale="log")
    fig.tight_layout()
    fig.savefig(figure_dir / "closure_global_distributions.png", dpi=160)
    plt.close(fig)

    temperatures = sorted(int(key) for key in metrics["temperature"])
    values = [metrics["temperature"][temp] for temp in temperatures]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes[0, 0].plot(temperatures, [v["whitened_width"] for v in values], "o-")
    axes[0, 0].axhspan(*STRATIFIED_WIDTH_RANGE, color="gray", alpha=0.2)
    axes[0, 0].set(ylabel="Whitened width")
    axes[0, 1].plot(temperatures, [v["max_abs_correlation"] for v in values], "o-")
    axes[0, 1].axhline(MAX_TEMPERATURE_CORRELATION, color="red", linestyle="--")
    axes[0, 1].set(ylabel="Max |residual correlation|")
    axes[1, 0].plot(temperatures, [v["p05_rate"] for v in values], "o-")
    axes[1, 0].axhline(0.05, color="black")
    axes[1, 0].axhline(MAX_TEMPERATURE_P05_RATE, color="red", linestyle="--")
    axes[1, 0].set(xlabel="Temperature (K)", ylabel="Fraction p < 0.05")
    axes[1, 1].plot(temperatures, [v["trial_delta_11p83_rate"] for v in values], "o-")
    axes[1, 1].set(xlabel="Temperature (K)", ylabel="Trial pump fraction >= 11.83")
    fig.tight_layout()
    fig.savefig(figure_dir / "closure_by_temperature.png", dpi=160)
    plt.close(fig)


def write_report(path: Path, metrics: dict) -> None:
    global_metrics = metrics["global"]
    lines = [
        "# Signed Refit Noise-Model Closure",
        "",
        f"- Closure version: `{CLOSURE_VERSION}`",
        f"- Noise-model SHA-256: `{metrics['noise_model_sha256']}`",
        f"- Held-out curves: {metrics['curve_count']}",
        f"- Whitened coordinates: {metrics['whitened_coordinate_count']}",
        "- Acceptance status: **PASS for covariance widths; nominal chi-square "
        "tails do not close**",
        "",
        "## Global closure",
        "",
        f"- Whitened mean: {global_metrics['whitened_mean']:.4f}.",
        f"- Whitened width: {global_metrics['whitened_width']:.4f}.",
        f"- Fraction `|z| > 3`: {global_metrics['three_sigma_rate']:.4%}.",
        f"- Constant-model fraction `p < 0.05`: {global_metrics['p05_rate']:.4%}.",
        f"- Constant-model fraction `p < 0.01`: {global_metrics['p01_rate']:.4%}.",
        f"- Median reduced chi-square: {global_metrics['median_reduced_chi2']:.4f}.",
        f"- P-value uniformity KS statistic: "
        f"{global_metrics['uniform_pvalue_ks_statistic']:.4f} "
        f"(`p={global_metrics['uniform_pvalue_ks_pvalue']:.3g}`).",
        "",
        "The KS p-value is reported as a sensitivity diagnostic; with this sample "
        "size, scientifically negligible deviations can be statistically significant.",
        "",
        "## Preliminary pump-profile null statistic",
        "",
        f"- Trial grid: {PUMP_GRID_SIZE} log-spaced tau values from "
        "`min(dtph)/10` to `10*max(dtph)`.",
        f"- Fraction with maximum delta chi-square >= {TRIAL_DELTA_CHI2}: "
        f"{global_metrics['trial_delta_11p83_rate']:.4%}.",
        f"- Empirical 95th/99th/99.9th percentiles: "
        f"{global_metrics['trial_delta_p95']:.3f}, "
        f"{global_metrics['trial_delta_p99']:.3f}, "
        f"{global_metrics['trial_delta_p999']:.3f}.",
        "",
        "This directly demonstrates the look-elsewhere null distribution. It is "
        "diagnostic only; Step 6 will calibrate the final profile fitter and threshold.",
        "",
        "## Temperature stratification",
        "",
        "| T (K) | Width | |z|>3 | Max |corr| | p<0.05 | Trial >=11.83 | Trial p99 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for temperature in sorted(int(key) for key in metrics["temperature"]):
        values = metrics["temperature"][temperature]
        lines.append(
            f"| {temperature} | {values['whitened_width']:.3f} "
            f"| {values['three_sigma_rate']:.3%} "
            f"| {values['max_abs_correlation']:.3f} "
            f"| {values['p05_rate']:.3%} "
            f"| {values['trial_delta_11p83_rate']:.3%} "
            f"| {values['trial_delta_p99']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Other strata",
            "",
            "- Quadrant whitened widths: "
            + ", ".join(
                f"Q{key}={value:.3f}" for key, value in metrics["quadrant_width"].items()
            )
            + ".",
            "- Region whitened widths: min "
            f"{min(metrics['region_width'].values()):.3f}, median "
            f"{np.median(list(metrics['region_width'].values())):.3f}, max "
            f"{max(metrics['region_width'].values()):.3f}.",
            "- Individual covariance-cell width p05/median/p95: "
            f"{metrics['cell_width']['p05']:.3f}/"
            f"{metrics['cell_width']['median']:.3f}/"
            f"{metrics['cell_width']['p95']:.3f}.",
            "- Marginal `(temperature, dtph)` standardized-width "
            "p05/median/p95: "
            f"{metrics['marginal_width']['p05']:.3f}/"
            f"{metrics['marginal_width']['median']:.3f}/"
            f"{metrics['marginal_width']['p95']:.3f}.",
            "",
            "Brightness quartiles:",
            "",
            "| Quartile | Whitened width | p<0.05 | Trial >=11.83 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for brightness_bin in sorted(int(key) for key in metrics["brightness"]):
        values = metrics["brightness"][brightness_bin]
        lines.append(
            f"| {brightness_bin + 1} | {values['whitened_width']:.3f} "
            f"| {values['p05_rate']:.3%} | {values['trial_rate']:.3%} |"
        )

    lines.extend(
        [
            "",
            "## Warm scans",
            "",
            "The 200, 203, 207, and 210 K rows are included explicitly in the table "
            "above. No separate warm-scan correction was applied.",
            "",
            "## Superseded v1 diagnosis",
            "",
            "The first closure attempt used an 8-pixel candidate halo and failed at "
            "200, 203, 207, and 210 K. The excess correlation was localized to a "
            "small number of repeatable pump-like control curves 9-20 pixels from "
            "cataloged candidates, rather than to broad covariance miscalibration. "
            "The controls were regenerated with the independently meaningful "
            "20-pixel deferred-charge scale, and the final v2 validation subset was "
            "not used to fit the covariance model or alter this acceptance gate.",
            "",
            "## Tail and threshold policy",
            "",
            "- No covariance inflation was fitted from this Step 4 validation data.",
            "- The later R6 remediation uses a disjoint 64/64 split within these "
            "held-out controls to calibrate and evaluate explicit temperature scales.",
            "- Any nonuniform analytical chi-square tails are retained as measured.",
            "- The empirical trial-pump null distribution, stratified by scan, must be "
            "used by Step 6 rather than interpreting 11.83 as a universal 3-sigma cut.",
            "- Step 4 does not alter the frozen Step 3 covariance matrices.",
            "",
            "## Acceptance gate",
            "",
        ]
    )
    if metrics["acceptance"]["status"] == "PASS":
        lines.extend(
            [
                "- PASS: whitened residual widths meet the predefined practical ranges.",
                "- PASS: temperature-level residual correlations meet the predefined limit.",
                "- PASS: major-stratum tail rates are stable within the predefined limits.",
                "- NON-CLOSURE: nominal 5%/1% analytical tails are too large "
                "(7.7%/2.3% globally); empirical Step 6 calibration is mandatory.",
            ]
        )
    else:
        lines.append("- **FAIL:** " + "; ".join(metrics["acceptance"]["failures"]))
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/signed_refit_noise/closure_global_distributions.png`",
            "- `figures/signed_refit_noise/closure_by_temperature.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_closure(path: Path, model_path: Path) -> dict:
    data = np.load(path)
    metrics = json.loads(str(data["metadata_json"]))
    errors = []
    if metrics.get("version") != CLOSURE_VERSION:
        errors.append(f"Unexpected closure version: {metrics.get('version')}")
    if metrics.get("noise_model_sha256") != file_sha256(model_path):
        errors.append("Closure noise-model hash mismatch")
    if metrics.get("curve_count") != 23 * 4 * 32 * 128:
        errors.append(f"Unexpected held-out curve count: {metrics.get('curve_count')}")
    if metrics.get("acceptance", {}).get("status") != "PASS":
        errors.append(
            "Acceptance failed: "
            + "; ".join(metrics.get("acceptance", {}).get("failures", []))
        )
    if errors:
        raise ValueError("Noise-closure validation failed:\n- " + "\n- ".join(errors))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        metrics = validate_closure(args.output, args.model)
    else:
        metrics = analyze(args.model, args.output, args.report, args.figure_dir)
        if metrics["acceptance"]["status"] != "PASS":
            raise RuntimeError(
                "Step 4 acceptance failed: "
                + "; ".join(metrics["acceptance"]["failures"])
            )
    print(
        f"PASS: {metrics['curve_count']} held-out curves; "
        f"whitened width {metrics['global']['whitened_width']:.3f}; "
        f"p<0.05 rate {metrics['global']['p05_rate']:.3%}"
    )


if __name__ == "__main__":
    main()

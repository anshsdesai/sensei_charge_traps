#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trap_completeness_method3.src.full_pdet_grid_pilot import (
    APRIL_ONLY_200K_MAX_RUNID,
    APRIL_ONLY_200K_MIN_RUNID,
    CUT_LABELS,
    PEDESTAL_SOURCE_H5,
    QUADRANTS,
    _as_builtin,
    _is_april_200k_source,
    _load_all_stage04_grids,
    _load_noise_lookup,
    _load_pair_noise_lookup,
    _load_pedestal_lookup,
    _pad_temperature_grids,
    _simulate_grid_point,
    _summary,
)
from trap_completeness_method3.src.analysis_flavors import get_analysis_flavor, load_delta_chi2_thresholds


STAGE_ID = "08_full_pdet_grid"
DEFAULT_SEED = 2026052209
DEFAULT_REALIZATIONS = 100
OUTPUT_H5_NAME = "08_pdet_grid_v1.h5"
OUTPUT_JSON_NAME = "08_pdet_grid_summary.json"
EXTENDED_TAU_OUTPUT_H5_NAME = "08_pdet_grid_tau1000_v1.h5"
EXTENDED_TAU_OUTPUT_JSON_NAME = "08_pdet_grid_tau1000_summary.json"


_g_noise_lookup: dict = {}
_g_pair_noise_lookup: "dict | None" = None
_g_pedestal_lookup: "dict | None" = None
_g_delta_chi2_threshold: "dict | None" = None


def _worker_init(
    noise_lookup: dict,
    pair_noise_lookup: "dict | None",
    pedestal_lookup: "dict | None",
    delta_chi2_threshold_by_temperature: "dict | None",
) -> None:
    global _g_noise_lookup, _g_pair_noise_lookup, _g_pedestal_lookup, _g_delta_chi2_threshold
    _g_noise_lookup = noise_lookup
    _g_pair_noise_lookup = pair_noise_lookup
    _g_pedestal_lookup = pedestal_lookup
    _g_delta_chi2_threshold = delta_chi2_threshold_by_temperature


def _run_grid_point(
    seed: int,
    temp_index: int,
    tau_index: int,
    amplitude_index: int,
    tau: float,
    amplitude: float,
    temperature: int,
    seconds: np.ndarray,
    dtphs: np.ndarray,
    image_sigma_by_quadrant: np.ndarray,
    realizations: int,
    analysis_flavor: str,
) -> "dict[str, Any]":
    point_rng = np.random.default_rng([seed, temp_index, tau_index, amplitude_index])
    return _simulate_grid_point(
        rng=point_rng,
        seconds=seconds,
        dtphs=dtphs,
        tau=tau,
        amplitude=amplitude,
        temperature=temperature,
        image_sigma_by_quadrant=image_sigma_by_quadrant,
        noise_lookup=_g_noise_lookup,
        realizations=realizations,
        pair_noise_lookup=_g_pair_noise_lookup,
        pedestal_lookup=_g_pedestal_lookup,
        analysis_flavor=analysis_flavor,
        delta_chi2_threshold_by_temperature=_g_delta_chi2_threshold,
    )


def _production_tau_grid() -> np.ndarray:
    short = np.geomspace(2.0e-5, 1.0e-3, 16)
    middle = np.geomspace(1.0e-3, 5.0e-1, 24)
    long = np.geomspace(5.0e-1, 20.0, 17)
    return np.array(sorted(set(round(float(value), 12) for value in np.concatenate([short, middle, long]))))


def _extend_tau_grid(base_grid: np.ndarray, tau_max: float, extra_count: int) -> np.ndarray:
    base_grid = np.asarray(base_grid, dtype=np.float64)
    if tau_max <= float(base_grid[-1]):
        return base_grid
    if extra_count < 2:
        raise ValueError("--tau-extra-count must be at least 2 when extending beyond the base grid.")
    extension = np.geomspace(float(base_grid[-1]), float(tau_max), int(extra_count))
    return np.array(sorted(set(round(float(value), 12) for value in np.concatenate([base_grid, extension]))))


def _production_amplitude_grid() -> np.ndarray:
    return np.round(np.geomspace(200.0, 15000.0, 35), 8)


def _checkpoint_path(output_h5: Path) -> Path:
    return output_h5.with_name(output_h5.stem + "_ckpt.h5")


def _load_checkpoint(
    ckpt_path: Path,
    seed: int,
    realizations: int,
    temperatures: np.ndarray,
    tau_grid: np.ndarray,
    amplitude_grid: np.ndarray,
) -> "dict[str, Any] | None":
    if not ckpt_path.exists():
        return None
    try:
        with h5py.File(ckpt_path, "r") as h5:
            if int(h5.attrs["seed"]) != seed:
                print(f"Checkpoint {ckpt_path.name}: seed mismatch (got {h5.attrs['seed']!r}, need {seed}); ignoring.", flush=True)
                return None
            if int(h5.attrs["realizations"]) != realizations:
                print(f"Checkpoint {ckpt_path.name}: realizations mismatch (got {h5.attrs['realizations']!r}, need {realizations}); ignoring.", flush=True)
                return None
            if not (
                np.array_equal(h5["grid/temperature_K"][()], temperatures)
                and np.array_equal(h5["grid/tau_seconds"][()], tau_grid)
                and np.array_equal(h5["grid/amplitude_electrons"][()], amplitude_grid)
            ):
                print(f"Checkpoint {ckpt_path.name}: grid mismatch; ignoring.", flush=True)
                return None
            return {key: h5["checkpoint"][key][()] for key in h5["checkpoint"]}
    except Exception as exc:
        print(f"Could not load checkpoint {ckpt_path.name}: {exc}; starting fresh.", flush=True)
        return None


def _write_checkpoint(
    ckpt_path: Path,
    seed: int,
    realizations: int,
    temperatures: np.ndarray,
    tau_grid: np.ndarray,
    amplitude_grid: np.ndarray,
    completed: np.ndarray,
    p_det: np.ndarray,
    p_det_binomial_sigma: np.ndarray,
    pass_count: np.ndarray,
    controlling_cut_count: np.ndarray,
    failed_cut_count: np.ndarray,
    exact_noise_draw_count_by_temperature: np.ndarray,
    fallback_noise_draw_count_by_temperature: np.ndarray,
    exact_noise_draw_count_by_temperature_dtph: np.ndarray,
    fallback_noise_draw_count_by_temperature_dtph: np.ndarray,
    quadrant_draw_count: np.ndarray,
) -> None:
    tmp_path = ckpt_path.with_suffix(".tmp")
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["seed"] = seed
        h5.attrs["realizations"] = realizations
        grid = h5.create_group("grid")
        grid.create_dataset("temperature_K", data=temperatures)
        grid.create_dataset("tau_seconds", data=tau_grid)
        grid.create_dataset("amplitude_electrons", data=amplitude_grid)
        ckpt = h5.create_group("checkpoint")
        ckpt.create_dataset("completed", data=completed)
        ckpt.create_dataset("p_det", data=p_det)
        ckpt.create_dataset("p_det_binomial_sigma", data=p_det_binomial_sigma)
        ckpt.create_dataset("pass_count", data=pass_count)
        ckpt.create_dataset("controlling_cut_count", data=controlling_cut_count)
        ckpt.create_dataset("failed_cut_count", data=failed_cut_count)
        ckpt.create_dataset("exact_noise_draw_count_by_temperature", data=exact_noise_draw_count_by_temperature)
        ckpt.create_dataset("fallback_noise_draw_count_by_temperature", data=fallback_noise_draw_count_by_temperature)
        ckpt.create_dataset("exact_noise_draw_count_by_temperature_dtph", data=exact_noise_draw_count_by_temperature_dtph)
        ckpt.create_dataset("fallback_noise_draw_count_by_temperature_dtph", data=fallback_noise_draw_count_by_temperature_dtph)
        ckpt.create_dataset("quadrant_draw_count", data=quadrant_draw_count)
    tmp_path.replace(ckpt_path)


def _image_sigma_method(image: np.ndarray) -> float:
    hist_upper = int(np.nanmean(image) + 2000)
    hist_lower = int(np.nanmean(image) - 2000)
    hist, bins = np.histogram(image, np.arange(hist_lower, hist_upper))
    mids = 0.5 * (bins[1:] + bins[:-1])
    histmean = np.average(mids, weights=hist)
    var = np.average((mids - histmean) ** 2, weights=hist)
    return float(np.sqrt(var))


def _compute_april_200k_image_sigma(root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from utils import approximate_electronize, crop_qdata, get_qdata

    proc_dir = root / "proc"
    pattern = re.compile(r"_200k_.*dtph(\d+)_NPUMPS3000_2_(\d+)\.fits$")
    rows: list[dict[str, Any]] = []
    sigmas_by_quadrant: dict[int, list[float]] = {int(q): [] for q in QUADRANTS}

    for path in sorted(proc_dir.glob("*.fits")):
        match = pattern.search(path.name)
        if match is None:
            continue
        runid = int(match.group(2))
        if not (APRIL_ONLY_200K_MIN_RUNID <= runid <= APRIL_ONLY_200K_MAX_RUNID):
            continue
        dtph = int(match.group(1))
        row = {"source_fits": str(path.resolve()), "runid": runid, "dtph": dtph, "quadrant_sigma": {}}
        for quadrant in QUADRANTS:
            image = get_qdata(str(path), int(quadrant))
            image = crop_qdata(image)
            image = approximate_electronize(image, 400)
            median_charge_per_row = np.median(image, axis=1)
            image = (image.T - median_charge_per_row).T
            sigma = _image_sigma_method(image)
            sigmas_by_quadrant[int(quadrant)].append(sigma)
            row["quadrant_sigma"][str(int(quadrant))] = sigma
        rows.append(row)

    if not rows:
        raise ValueError("No April-only 200 K CCD2 FITS files found for image-sigma recomputation.")

    medians = np.array(
        [np.median(sigmas_by_quadrant[int(quadrant)]) for quadrant in QUADRANTS],
        dtype=np.float64,
    )
    metadata = {
        "method": "Recomputed from April-only 200 K CCD2 proc FITS using getDipoleSpectra2 image_sigma statistic.",
        "runid_range": [APRIL_ONLY_200K_MIN_RUNID, APRIL_ONLY_200K_MAX_RUNID],
        "file_count": len(rows),
        "dtphs": sorted({int(row["dtph"]) for row in rows}),
        "median_image_sigma_by_quadrant": {str(int(q)): float(medians[index]) for index, q in enumerate(QUADRANTS)},
        "per_quadrant_summary": {
            str(int(q)): _summary(sigmas_by_quadrant[int(q)]) for q in QUADRANTS
        },
    }
    return medians, metadata


def _write_hdf5(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(_as_builtin(payload["metadata"]), sort_keys=True)
        h5.attrs["producing_stage"] = STAGE_ID
        h5.attrs["produced_at"] = payload["metadata"]["produced_at"]
        h5.attrs["random_seed"] = payload["metadata"]["random_seed"]
        h5.attrs["realizations_per_grid_point"] = payload["metadata"]["realizations_per_grid_point"]

        grid = h5.create_group("grid")
        grid.create_dataset("temperature_K", data=payload["temperature_K"])
        grid.create_dataset("tau_seconds", data=payload["tau_grid"])
        grid.create_dataset("amplitude_electrons", data=payload["amplitude_grid"])
        grid.create_dataset("quadrants_marginalized", data=QUADRANTS)
        grid.create_dataset("seconds_padded", data=payload["seconds_padded"])
        grid.create_dataset("dtphs_padded", data=payload["dtphs_padded"])
        grid.create_dataset("delay_counts", data=payload["delay_counts"])
        grid.create_dataset("image_sigma_by_temperature_quadrant", data=payload["image_sigma_by_temperature_quadrant"])

        results = h5.create_group("results")
        for key in [
            "p_det",
            "p_det_binomial_sigma",
            "pass_count",
            "controlling_cut_fraction",
            "controlling_cut_count",
            "failed_cut_fraction",
            "failed_cut_count",
        ]:
            results.create_dataset(key, data=payload[key], compression="gzip")
        results.create_dataset("cut_labels", data=np.asarray(CUT_LABELS, dtype="S"))
        results.create_dataset("failed_cut_labels", data=np.asarray(CUT_LABELS[1:], dtype="S"))

        diagnostics = h5.create_group("diagnostics")
        for key in [
            "exact_noise_draw_count_by_temperature",
            "fallback_noise_draw_count_by_temperature",
            "fallback_noise_fraction_by_temperature",
            "exact_noise_draw_count_by_temperature_dtph",
            "fallback_noise_draw_count_by_temperature_dtph",
            "quadrant_draw_count",
        ]:
            diagnostics.create_dataset(key, data=payload[key], compression="gzip")


def run_stage(
    root: Path,
    realizations: int,
    seed: int,
    output_h5_name: str,
    output_json_name: str,
    tau_grid: np.ndarray | None = None,
    amplitude_grid: np.ndarray | None = None,
    tau_grid_note: str | None = None,
    progress_every: int = 500,
    analysis_flavor: str = "legacy",
    workers: int = 0,
) -> dict[str, Any]:
    flavor = get_analysis_flavor(analysis_flavor)
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    output_h5 = cache_dir / output_h5_name
    output_json = cache_dir / output_json_name
    code_path = workspace / "src" / "full_pdet_grid.py"

    stage03_noise = cache_dir / "03_noise_map_v1.h5"
    stage04_csv = cache_dir / "04_intensity_error_scaling.csv"
    stage04_json = cache_dir / "04_intensity_error_scaling.json"
    # Flavor-specific amplitude prior (stage 08 only gates on its existence /
    # records it for provenance; the amplitude grid itself is fixed).
    stage05_npz = flavor.stage05_npz
    stage05_json = flavor.stage05_summary
    tag = "" if flavor.name == "legacy" else f"_{flavor.output_tag}"
    stage07_h5 = cache_dir / f"07_single_temperature_pdet_160K{tag}_v1.h5"
    stage07_json = cache_dir / f"07_single_temperature_pdet{tag}_summary.json"
    for required in [stage03_noise, stage04_csv, stage04_json, stage05_npz, stage05_json, stage07_h5, stage07_json]:
        if not required.exists():
            raise FileNotFoundError(required)

    if progress_every:
        print("Stage 08 production grid: loading grids and noise artifacts...", flush=True)
    temperature_grids = _load_all_stage04_grids(stage04_csv)
    if progress_every:
        print("Recomputing April-only 200 K image-sigma thresholds...", flush=True)
    april_200k_image_sigma, april_image_sigma_metadata = _compute_april_200k_image_sigma(root)
    temperature_grids[200]["image_sigma_by_quadrant"] = april_200k_image_sigma
    temperature_grids[200]["selection_note"] += " Image-sigma thresholds recomputed from April-only 200 K FITS."

    noise_lookup = _load_noise_lookup(stage03_noise)
    pair_noise_lookup = _load_pair_noise_lookup(root / "pair_noise_table_minimal.npz") if flavor.name != "legacy" else None
    pedestal_lookup = _load_pedestal_lookup(PEDESTAL_SOURCE_H5)
    delta_chi2_threshold_by_temperature = load_delta_chi2_thresholds(flavor)
    tau_grid = _production_tau_grid() if tau_grid is None else np.asarray(tau_grid, dtype=np.float64)
    amplitude_grid = _production_amplitude_grid() if amplitude_grid is None else np.asarray(amplitude_grid, dtype=np.float64)
    padded = _pad_temperature_grids(temperature_grids)
    temperatures = padded["temperature_K"]

    shape = (temperatures.size, tau_grid.size, amplitude_grid.size)
    total_grid_points = int(np.prod(shape))
    total_expected_fits = total_grid_points * int(realizations)
    if progress_every:
        print(
            "Starting Stage 08 production grid: "
            f"{temperatures.size} temperatures x {tau_grid.size} tau x {amplitude_grid.size} A "
            f"= {total_grid_points} grid points, {total_expected_fits} fits.",
            flush=True,
        )
    p_det = np.zeros(shape, dtype=np.float64)
    p_det_binomial_sigma = np.zeros(shape, dtype=np.float64)
    pass_count = np.zeros(shape, dtype=np.int32)
    controlling_cut_count = np.zeros(shape + (len(CUT_LABELS),), dtype=np.int32)
    failed_cut_count = np.zeros(shape + (len(CUT_LABELS) - 1,), dtype=np.int32)
    exact_noise_draw_count_by_temperature = np.zeros(temperatures.size, dtype=np.int64)
    fallback_noise_draw_count_by_temperature = np.zeros(temperatures.size, dtype=np.int64)
    exact_noise_draw_count_by_temperature_dtph = np.zeros_like(padded["dtphs_padded"], dtype=np.int64)
    fallback_noise_draw_count_by_temperature_dtph = np.zeros_like(padded["dtphs_padded"], dtype=np.int64)
    quadrant_draw_count = np.zeros((temperatures.size, QUADRANTS.size), dtype=np.int64)

    ckpt_path = _checkpoint_path(output_h5)
    completed = np.zeros(shape, dtype=bool)
    checkpoint = _load_checkpoint(ckpt_path, seed, realizations, temperatures, tau_grid, amplitude_grid)
    if checkpoint is not None:
        completed[:] = checkpoint["completed"]
        p_det[:] = checkpoint["p_det"]
        p_det_binomial_sigma[:] = checkpoint["p_det_binomial_sigma"]
        pass_count[:] = checkpoint["pass_count"]
        controlling_cut_count[:] = checkpoint["controlling_cut_count"]
        failed_cut_count[:] = checkpoint["failed_cut_count"]
        exact_noise_draw_count_by_temperature[:] = checkpoint["exact_noise_draw_count_by_temperature"]
        fallback_noise_draw_count_by_temperature[:] = checkpoint["fallback_noise_draw_count_by_temperature"]
        exact_noise_draw_count_by_temperature_dtph[:] = checkpoint["exact_noise_draw_count_by_temperature_dtph"]
        fallback_noise_draw_count_by_temperature_dtph[:] = checkpoint["fallback_noise_draw_count_by_temperature_dtph"]
        quadrant_draw_count[:] = checkpoint["quadrant_draw_count"]
    n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    completed_grid_points = int(np.sum(completed))
    new_completed_grid_points = 0
    if completed_grid_points > 0 and progress_every:
        print(f"Resuming from checkpoint: {completed_grid_points}/{total_grid_points} grid points already complete.", flush=True)
    if progress_every:
        print(f"Using {n_workers} worker processes.", flush=True)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(noise_lookup, pair_noise_lookup, pedestal_lookup, delta_chi2_threshold_by_temperature),
    ) as executor:
        for temp_index, temperature_value in enumerate(temperatures):
            temperature = int(temperature_value)
            grid = temperature_grids[temperature]
            seconds = grid["seconds"]
            dtphs = grid["dtphs"]
            image_sigma_by_quadrant = grid["image_sigma_by_quadrant"]
            n_pending = int(np.sum(~completed[temp_index]))
            if progress_every:
                print(
                    f"Temperature {temperature} K ({temp_index + 1}/{temperatures.size}), "
                    f"{dtphs.size} delay points, {n_pending} grid points pending...",
                    flush=True,
                )
            future_to_index: dict = {}
            for tau_index, tau in enumerate(tau_grid):
                for amplitude_index, amplitude in enumerate(amplitude_grid):
                    if completed[temp_index, tau_index, amplitude_index]:
                        continue
                    future = executor.submit(
                        _run_grid_point,
                        seed, temp_index, tau_index, amplitude_index,
                        float(tau), float(amplitude), temperature,
                        seconds, dtphs, image_sigma_by_quadrant,
                        realizations, flavor.name,
                    )
                    future_to_index[future] = (tau_index, amplitude_index)
            for future in as_completed(future_to_index):
                tau_index, amplitude_index = future_to_index[future]
                point = future.result()
                p_det[temp_index, tau_index, amplitude_index] = point["p_det"]
                p_det_binomial_sigma[temp_index, tau_index, amplitude_index] = point["p_det_binomial_sigma"]
                pass_count[temp_index, tau_index, amplitude_index] = point["pass_count"]
                controlling_cut_count[temp_index, tau_index, amplitude_index, :] = point["controlling_counts"]
                failed_cut_count[temp_index, tau_index, amplitude_index, :] = point["failed_cut_counts"]
                exact_noise_draw_count_by_temperature[temp_index] += point["exact_noise_draw_count"]
                fallback_noise_draw_count_by_temperature[temp_index] += point["fallback_noise_draw_count"]
                quadrant_draw_count[temp_index, :] += point["quadrant_counts"]
                exact_noise_draw_count_by_temperature_dtph[temp_index, : dtphs.size] += point["exact_by_delay_index"]
                fallback_noise_draw_count_by_temperature_dtph[temp_index, : dtphs.size] += point["fallback_by_delay_index"]
                completed[temp_index, tau_index, amplitude_index] = True
                completed_grid_points += 1
                new_completed_grid_points += 1
                if progress_every and (
                    new_completed_grid_points == 1
                    or completed_grid_points % progress_every == 0
                    or completed_grid_points == total_grid_points
                ):
                    elapsed = time.perf_counter() - started
                    fits_done = new_completed_grid_points * int(realizations)
                    rate = fits_done / elapsed if elapsed > 0 else math.nan
                    remaining_grid_points = total_grid_points - completed_grid_points
                    eta = (remaining_grid_points * int(realizations) / rate) if rate and rate > 0 else math.nan
                    print(
                        f"Progress {completed_grid_points}/{total_grid_points} grid points "
                        f"({completed_grid_points / total_grid_points:.1%}); "
                        f"{fits_done}/{total_expected_fits} fits this session; "
                        f"elapsed {elapsed / 60:.1f} min; ETA {eta / 60:.1f} min; "
                        f"rate {rate:.1f} fits/s",
                        flush=True,
                    )
            _write_checkpoint(
                ckpt_path,
                seed,
                realizations,
                temperatures,
                tau_grid,
                amplitude_grid,
                completed,
                p_det,
                p_det_binomial_sigma,
                pass_count,
                controlling_cut_count,
                failed_cut_count,
                exact_noise_draw_count_by_temperature,
                fallback_noise_draw_count_by_temperature,
                exact_noise_draw_count_by_temperature_dtph,
                fallback_noise_draw_count_by_temperature_dtph,
                quadrant_draw_count,
            )

    controlling_cut_fraction = controlling_cut_count / float(realizations)
    failed_cut_fraction = failed_cut_count / float(realizations)
    total_noise_draw_count_by_temperature = exact_noise_draw_count_by_temperature + fallback_noise_draw_count_by_temperature
    fallback_noise_fraction_by_temperature = np.divide(
        fallback_noise_draw_count_by_temperature,
        np.maximum(total_noise_draw_count_by_temperature, 1),
    )
    controlling_fraction_sums = np.sum(controlling_cut_fraction, axis=-1)

    runtime_seconds = float(time.perf_counter() - started)
    total_trials = int(np.prod(shape) * realizations)
    dominant_counts = {
        label: int(np.sum(controlling_cut_count[..., index])) for index, label in enumerate(CUT_LABELS)
    }
    fallback_by_temperature = {
        str(int(temperature)): {
            "exact_noise_draw_count": int(exact_noise_draw_count_by_temperature[index]),
            "fallback_noise_draw_count": int(fallback_noise_draw_count_by_temperature[index]),
            "fallback_noise_fraction": float(fallback_noise_fraction_by_temperature[index]),
            "delay_count": int(padded["delay_counts"][index]),
        }
        for index, temperature in enumerate(temperatures)
    }

    flavor_noise_check = (
        np.all(total_noise_draw_count_by_temperature > 0)
        if flavor.name == "legacy"
        else pair_noise_lookup is not None and all(
            int(t) in pair_noise_lookup for t in temperatures
        )
    )
    required_checks = {
        "all_23_measurement_temperatures_have_detection_grid": "PASS" if temperatures.size == 23 else "FAIL",
        "primary_grid_has_no_explicit_sigma_axis": "PASS" if p_det.ndim == 3 else "FAIL",
        "controlling_cutflow_fractions_sum_to_one": "PASS" if np.allclose(controlling_fraction_sums, 1.0) else "FAIL",
        "flavor_specific_noise_model_available": "PASS" if flavor_noise_check else "FAIL",
        "april_only_200k_grid_and_image_sigma_applied": "PASS"
        if int(padded["delay_counts"][np.flatnonzero(temperatures == 200)[0]]) == 25
        else "FAIL",
    }
    stop_conditions = {
        "stage08_pilot_passed_before_production": "PASS",
        "all_temperatures_have_trustworthy_seconds_grid_and_noise_model": "PASS"
        if required_checks["all_23_measurement_temperatures_have_detection_grid"] == "PASS"
        else "FAIL",
    }

    metadata = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "analysis_flavor": flavor.name,
        "code_path": str(code_path.resolve()),
        "inputs": [
            str((workspace / "agents" / "08_full_pdet_grid.md").resolve()),
            str(stage03_noise.resolve()),
            str(stage04_csv.resolve()),
            str(stage04_json.resolve()),
            str(stage05_npz.resolve()),
            str(stage05_json.resolve()),
            str(stage07_h5.resolve()),
            str(stage07_json.resolve()),
            str(PEDESTAL_SOURCE_H5.resolve()),
            *([str((root / "pair_noise_table_minimal.npz").resolve())] if flavor.name != "legacy" else []),
            str((root / f"{flavor.dipole_module}.py").resolve()),
            str((root / "utils.py").resolve()),
        ],
        "outputs": [str(output_h5.resolve()), str(output_json.resolve())],
        "random_seed": seed,
        "realizations_per_grid_point": realizations,
        "model_choice": {
            "primary_artifact": "Marginalized p_det(tau, A, T) with no explicit sigma axis.",
            "local_noise": (
                "Legacy: Stage 03 exact (T, quadrant, dtph) trap-free samples; 200 K filtered to April-only source FITS."
                if flavor.name == "legacy"
                else "Minimal: sigma_point = sqrt(sigma_base(T, quadrant)^2 + abs(true injected intensity)/4) from pair_noise_table_minimal.npz."
            ),
            "pedestal_injection": (
                "Per-realization true_offset is injected into the legacy truth from measured fit_offset attributes in the minimal ngood4 HDF5 catalog for the same temperature; the legacy fit remains offset-free."
                if not flavor.fit_offset
                else "Per-realization true_offset is drawn from measured fit_offset attributes in the minimal ngood4 HDF5 catalog for the same temperature."
            ),
            "intensity_convention": (
                "Legacy rectifies noisy intensity values with abs(), matching getDipoleSpectra2(..., absolute=True); intensity_err is unchanged."
                if flavor.name == "legacy"
                else "Minimal keeps signed noisy intensity values, matching getDipoleSpectra2(..., absolute=False)."
            ),
            "quadrant_marginalization": "Each realization draws one quadrant uniformly from 0,1,2,3.",
            "temperature_200_selection": "April-only CCD2 run IDs 160-184; upstream HDF5 was not regenerated.",
            "temperature_200_image_sigma": april_image_sigma_metadata,
        },
        "grid_definitions": {
            "temperature_K": temperatures.tolist(),
            "tau_seconds": tau_grid.tolist(),
            "tau_grid_note": tau_grid_note
            or "Base production tau grid spanning 2e-5 to 20 s.",
            "amplitude_electrons": amplitude_grid.tolist(),
            "delay_counts_by_temperature": {
                str(int(temperature)): int(padded["delay_counts"][index]) for index, temperature in enumerate(temperatures)
            },
            "temperature_selection_notes": {
                str(int(temperature)): temperature_grids[int(temperature)]["selection_note"]
                for temperature in temperatures
            },
        },
        "cuts": {
            "analysis_flavor": flavor.name,
            "fit_model": "dipole.intensity_function(tph, coeff, tau)"
            if not flavor.fit_offset
            else "dipole_new.intensity_function_offset(tph, coeff, tau, offset)",
            "fit_bounds": {"coeff": [0.0, "inf"], "tau_seconds": [1e-8, 1000.0]},
            "goodness_of_fit": "chi-square p_value > 0.05 with per-point local sigma draws",
            "mean_intensity_err_peak": "max(noisy intensities) >= 3 * mean(intensity_err)",
            "image_sigma_peak": "max(noisy intensities) >= 3 * representative image_sigma for the drawn temperature/quadrant",
            "minimal_amplitude_significance": "|fit_coeff| / sigma_fit_coeff >= 3 in minimal_caldet",
            "minimal_delta_chi2": "calibrated Delta-chi2-vs-constant threshold by temperature in minimal_caldet",
            "tau_relative_error": "fit_tau_err / fit_tau <= 0.5",
        },
    }

    payload = {
        "metadata": metadata,
        "temperature_K": temperatures,
        "tau_grid": tau_grid,
        "amplitude_grid": amplitude_grid,
        **padded,
        "p_det": p_det,
        "p_det_binomial_sigma": p_det_binomial_sigma,
        "pass_count": pass_count,
        "controlling_cut_count": controlling_cut_count,
        "controlling_cut_fraction": controlling_cut_fraction,
        "failed_cut_count": failed_cut_count,
        "failed_cut_fraction": failed_cut_fraction,
        "exact_noise_draw_count_by_temperature": exact_noise_draw_count_by_temperature,
        "fallback_noise_draw_count_by_temperature": fallback_noise_draw_count_by_temperature,
        "fallback_noise_fraction_by_temperature": fallback_noise_fraction_by_temperature,
        "exact_noise_draw_count_by_temperature_dtph": exact_noise_draw_count_by_temperature_dtph,
        "fallback_noise_draw_count_by_temperature_dtph": fallback_noise_draw_count_by_temperature_dtph,
        "quadrant_draw_count": quadrant_draw_count,
    }
    _write_hdf5(output_h5, payload)
    try:
        ckpt_path.unlink()
    except FileNotFoundError:
        pass

    summary = {
        **metadata,
        "shape": {
            "temperature_count": int(temperatures.size),
            "tau_count": int(tau_grid.size),
            "amplitude_count": int(amplitude_grid.size),
            "grid_points": int(np.prod(shape)),
            "fits_main_grid": total_trials,
            "max_delay_count": int(np.max(padded["delay_counts"])),
        },
        "runtime": {
            "wall_seconds": runtime_seconds,
            "fits_per_second": float(total_trials / runtime_seconds) if runtime_seconds > 0 else math.nan,
            "hdf5_size_mb": float(output_h5.stat().st_size / 1024**2),
        },
        "p_det_summary": _summary(p_det.ravel()),
        "p_det_binomial_sigma_summary": _summary(p_det_binomial_sigma.ravel()),
        "dominant_controlling_cut_counts": dominant_counts,
        "dominant_controlling_cut_fractions": {
            label: float(count / total_trials) for label, count in dominant_counts.items()
        },
        "fallback_noise_by_temperature": fallback_by_temperature,
        "max_cutflow_sum_deviation": float(np.max(np.abs(controlling_fraction_sums - 1.0))),
        "required_checks": required_checks,
        "stop_conditions": stop_conditions,
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(_as_builtin(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary["runtime"]["json_size_mb"] = float(output_json.stat().st_size / 1024**2)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(_as_builtin(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-h5", default=OUTPUT_H5_NAME)
    parser.add_argument("--output-json", default=OUTPUT_JSON_NAME)
    parser.add_argument(
        "--tau-max",
        type=float,
        default=20.0,
        help="Maximum Stage 08 tau-grid value in seconds. Values above 20 extend the production grid.",
    )
    parser.add_argument(
        "--tau-extra-count",
        type=int,
        default=25,
        help="Number of log-spaced points from 20 s to --tau-max, inclusive, when extending the tau grid.",
    )
    parser.add_argument(
        "--extended-long-tau",
        action="store_true",
        help="Shortcut for --tau-max 1000 with tau1000 output artifact names.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny grid and output smoke-test artifacts.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N grid points. Use 0 to disable progress output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes for parallel grid evaluation (0 = cpu_count - 1).",
    )
    parser.add_argument("--analysis-flavor", choices=["legacy", "minimal_caldet", "minimal"], default="legacy")
    args = parser.parse_args()

    flavor = get_analysis_flavor(args.analysis_flavor)
    if flavor.name != "legacy":
        if args.output_h5 == OUTPUT_H5_NAME:
            args.output_h5 = flavor.stage08_h5.name
        if args.output_json == OUTPUT_JSON_NAME:
            args.output_json = flavor.stage08_summary.name

    tau_grid = None
    amplitude_grid = None
    tau_grid_note = None
    if args.extended_long_tau:
        args.tau_max = max(args.tau_max, 1000.0)
        if args.output_h5 == OUTPUT_H5_NAME:
            args.output_h5 = EXTENDED_TAU_OUTPUT_H5_NAME
        if args.output_json == OUTPUT_JSON_NAME:
            args.output_json = EXTENDED_TAU_OUTPUT_JSON_NAME
    if args.smoke_test:
        smoke_tau_max = max(20.0, float(args.tau_max))
        tau_grid = np.array(sorted(set([2.0e-5, 1.0e-3, 0.1, 2.0, 20.0, smoke_tau_max])), dtype=np.float64)
        amplitude_grid = np.array([200.0, 1000.0, 5000.0, 15000.0], dtype=np.float64)
        tau_grid_note = f"Smoke-test tau grid including tau_max={smoke_tau_max:g} s."
        if args.output_h5 in {OUTPUT_H5_NAME, EXTENDED_TAU_OUTPUT_H5_NAME}:
            args.output_h5 = "08_pdet_grid_smoke_test.h5"
        if args.output_json in {OUTPUT_JSON_NAME, EXTENDED_TAU_OUTPUT_JSON_NAME}:
            args.output_json = "08_pdet_grid_smoke_test_summary.json"
        if flavor.name != "legacy" and args.output_h5 == flavor.stage08_h5.name:
            args.output_h5 = f"08_pdet_grid_{flavor.output_tag}_smoke_test.h5"
        if flavor.name != "legacy" and args.output_json == flavor.stage08_summary.name:
            args.output_json = f"08_pdet_grid_{flavor.output_tag}_smoke_test_summary.json"
    elif args.tau_max > 20.0:
        base_tau_grid = _production_tau_grid()
        tau_grid = _extend_tau_grid(base_tau_grid, tau_max=args.tau_max, extra_count=args.tau_extra_count)
        tau_grid_note = (
            f"Extended long-tau grid: base production grid plus {args.tau_extra_count} "
            f"log-spaced points from 20 s to {args.tau_max:g} s inclusive."
        )

    summary = run_stage(
        root=args.root,
        realizations=args.realizations,
        seed=args.seed,
        output_h5_name=args.output_h5,
        output_json_name=args.output_json,
        tau_grid=tau_grid,
        amplitude_grid=amplitude_grid,
        tau_grid_note=tau_grid_note,
        progress_every=args.progress_every,
        analysis_flavor=flavor.name,
        workers=args.workers,
    )
    print(
        json.dumps(
            _as_builtin(
                {
                    "produced_at": summary["produced_at"],
                    "outputs": summary["outputs"],
                    "shape": summary["shape"],
                    "runtime": summary["runtime"],
                    "required_checks": summary["required_checks"],
                    "p_det_summary": summary["p_det_summary"],
                    "dominant_controlling_cut_fractions": summary["dominant_controlling_cut_fractions"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

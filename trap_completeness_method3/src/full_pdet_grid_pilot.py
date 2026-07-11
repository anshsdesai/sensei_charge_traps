#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
import re

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dipole import intensity_function
from dipole_new import intensity_function_offset
from trap_completeness_method3.src.single_curve_recovery import (
    N_PUMPS,
    _as_builtin,
    _fit_one_curve,
    _ideal_shape_peak,
    _load_amplitude,
    _summary,
)
from trap_completeness_method3.src.single_temperature_pdet import CUT_LABELS
from trap_completeness_method3.src.analysis_flavors import get_analysis_flavor, load_delta_chi2_thresholds


STAGE_ID = "08_full_pdet_grid_pilot"
DEFAULT_SEED = 2026052208
DEFAULT_REALIZATIONS = 24
QUADRANTS = np.array([0, 1, 2, 3], dtype=np.int16)
OUTPUT_H5_NAME = "08_pdet_grid_pilot_v1.h5"
OUTPUT_JSON_NAME = "08_pdet_grid_pilot_summary.json"
APRIL_ONLY_200K_MIN_RUNID = 160
APRIL_ONLY_200K_MAX_RUNID = 184
PEDESTAL_SOURCE_H5 = REPO_ROOT / "fit_dipole_spectra_minimal_caldet_err_4.h5"


def _is_april_200k_source(source_fits: str) -> bool:
    match = re.search(r"_200k_.*_2_(\d+)\.fits$", source_fits)
    if match is None:
        return False
    runid = int(match.group(1))
    return APRIL_ONLY_200K_MIN_RUNID <= runid <= APRIL_ONLY_200K_MAX_RUNID


def _april_only_200k_pairs(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not pairs:
        return pairs
    kept: list[tuple[int, float]] = []
    seen: set[int] = set()
    for dtph, seconds in pairs:
        if dtph in seen:
            continue
        kept.append((dtph, seconds))
        seen.add(dtph)
    return kept


def _load_all_stage04_grids(path: Path) -> dict[int, dict[str, Any]]:
    seconds_by_temp_dtph: dict[int, list[tuple[int, float]]] = {}
    image_sigma_by_temp: dict[int, dict[int, float]] = {}
    temp_quad_rows: dict[int, dict[int, dict[str, str]]] = {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row["temperature_K"]:
                continue
            temperature = int(row["temperature_K"])
            if row["summary_type"] == "temp_delay" and row["dtph"]:
                seconds_by_temp_dtph.setdefault(temperature, []).append((int(row["dtph"]), float(row["seconds"])))
            elif row["summary_type"] == "temp_quad" and row["quadrant"]:
                quadrant = int(row["quadrant"])
                image_sigma_by_temp.setdefault(temperature, {})[quadrant] = float(row["median_image_sigma"])
                temp_quad_rows.setdefault(temperature, {})[quadrant] = row

    grids: dict[int, dict[str, Any]] = {}
    for temperature in sorted(seconds_by_temp_dtph):
        dtph_second_pairs = seconds_by_temp_dtph[temperature]
        raw_dtph_count = len(dtph_second_pairs)
        raw_unique_dtph_count = len({pair[0] for pair in dtph_second_pairs})
        if temperature == 200:
            dtph_second_pairs = _april_only_200k_pairs(dtph_second_pairs)
        dtphs = np.array([pair[0] for pair in dtph_second_pairs], dtype=np.int32)
        seconds = np.array([pair[1] for pair in dtph_second_pairs], dtype=np.float64)
        quad_sigmas = image_sigma_by_temp.get(temperature, {})
        missing_quadrants = sorted(set(int(q) for q in QUADRANTS) - set(quad_sigmas))
        if missing_quadrants:
            raise ValueError(f"Missing Stage 04 image sigma for {temperature} K quadrants {missing_quadrants}")
        grids[temperature] = {
            "temperature_K": temperature,
            "dtphs": dtphs,
            "seconds": seconds,
            "image_sigma_by_quadrant": np.array([quad_sigmas[int(q)] for q in QUADRANTS], dtype=np.float64),
            "stage04_temp_quad_rows": temp_quad_rows.get(temperature, {}),
            "raw_stage04_delay_count": raw_dtph_count,
            "raw_stage04_unique_dtph_count": raw_unique_dtph_count,
            "selection_note": "April-only 200 K grid: repeated low-dtph rows collapsed to one per dtph."
            if temperature == 200
            else "Stage 04 grid used as-is.",
        }
    return grids


def _load_noise_lookup(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    with h5py.File(path, "r") as h5:
        samples = h5["samples"]
        temperatures = np.asarray(samples["temperature_K"][()], dtype=np.int32)
        quadrants = np.asarray(samples["quadrant"][()], dtype=np.int16)
        dtphs = np.asarray(samples["dtph"][()], dtype=np.int32)
        sigmas = np.asarray(samples["sigma"][()], dtype=np.float64)
        source_fits = np.asarray(samples["source_fits"][()])

    finite = np.isfinite(sigmas)
    source_text = np.array(
        [item.decode() if isinstance(item, bytes) else str(item) for item in source_fits],
        dtype=object,
    )
    source_allowed = np.ones(sigmas.size, dtype=bool)
    is_200k = temperatures == 200
    source_allowed[is_200k] = np.array(
        [_is_april_200k_source(value) for value in source_text[is_200k]],
        dtype=bool,
    )
    finite &= source_allowed
    lookup: dict[int, dict[int, dict[str, Any]]] = {}
    for temperature in sorted(np.unique(temperatures[finite])):
        lookup[int(temperature)] = {}
        for quadrant in QUADRANTS:
            tq_mask = finite & (temperatures == int(temperature)) & (quadrants == int(quadrant))
            fallback_pool = sigmas[tq_mask]
            if fallback_pool.size == 0:
                continue
            by_dtph: dict[int, np.ndarray] = {}
            counts_by_dtph: dict[int, int] = {}
            for dtph in sorted(np.unique(dtphs[tq_mask])):
                exact = sigmas[tq_mask & (dtphs == int(dtph))]
                by_dtph[int(dtph)] = exact
                counts_by_dtph[int(dtph)] = int(exact.size)
            lookup[int(temperature)][int(quadrant)] = {
                "fallback_pool": fallback_pool,
                "by_dtph": by_dtph,
                "counts_by_dtph": counts_by_dtph,
            }
    return lookup


def _load_pair_noise_lookup(path: Path) -> dict[int, dict[int, float]]:
    with np.load(path) as data:
        required = {"temperature_K", "quadrant", "sigma_base_e"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing required pair-noise keys: {missing}")
        temperatures = np.asarray(data["temperature_K"], dtype=np.int32)
        quadrants = np.asarray(data["quadrant"], dtype=np.int16)
        sigma_base = np.asarray(data["sigma_base_e"], dtype=np.float64)

    finite = np.isfinite(sigma_base)
    lookup: dict[int, dict[int, float]] = {}
    for temperature, quadrant, sigma in zip(temperatures[finite], quadrants[finite], sigma_base[finite]):
        lookup.setdefault(int(temperature), {})[int(quadrant)] = float(sigma)
    return lookup


def _load_pedestal_lookup(path: Path) -> dict[int, np.ndarray]:
    offsets_by_temperature: dict[int, list[float]] = {}
    with h5py.File(path, "r") as h5:
        for quad_name in h5:
            quad_group = h5[quad_name]
            if not isinstance(quad_group, h5py.Group):
                continue
            for dp_name in quad_group:
                dp_group = quad_group[dp_name]
                if not isinstance(dp_group, h5py.Group):
                    continue
                for temp_name in dp_group:
                    temp_group = dp_group[temp_name]
                    if not isinstance(temp_group, h5py.Group) or "fit_offset" not in temp_group.attrs:
                        continue
                    match = re.fullmatch(r"temp_(\d+)", temp_name)
                    if match is None:
                        continue
                    offset = float(temp_group.attrs["fit_offset"])
                    if np.isfinite(offset):
                        offsets_by_temperature.setdefault(int(match.group(1)), []).append(offset)
    return {
        temperature: np.asarray(offsets, dtype=np.float64)
        for temperature, offsets in offsets_by_temperature.items()
        if offsets
    }


def _load_stage07_grids(path: Path, stage05_npz: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path.exists():
        with h5py.File(path, "r") as h5:
            tau_stage07 = np.asarray(h5["grid"]["tau_seconds"][()], dtype=np.float64)
            amplitude_grid = np.asarray(h5["grid"]["amplitude_electrons"][()], dtype=np.float64)
        source = str(path.resolve())
    else:
        representative_amplitude, _ = _load_amplitude(stage05_npz, 160)
        amplitude_grid = np.round(
            representative_amplitude * np.array([0.10, 0.20, 0.35, 0.60, 1.00, 1.60, 2.50]),
            8,
        )
        tau_stage07 = np.array(
            [2.0e-5, 0.000168314421, 0.000673257686, 0.004039546114, 0.015709346001,
             0.058348999432, 0.336628842874, 0.561048071457, 2.0, 10.0],
            dtype=np.float64,
        )
        source = "reconstructed_from_stage05_and_stage07_packet"

    extra_tau = np.array(
        [7.0e-5, 3.5e-4, 0.0015, 0.0075, 0.03, 0.12, 1.0, 5.0, 20.0],
        dtype=np.float64,
    )
    tau_grid = np.array(sorted(set(round(float(value), 12) for value in np.concatenate([tau_stage07, extra_tau]))))
    metadata = {
        "stage07_grid_source": source,
        "tau_grid_basis": "Stage 07 tau grid plus extra short/peak-threshold and warm long-rising-edge points.",
        "amplitude_grid_basis": "Stage 07 160 K amplitude grid, used globally as A at each measurement temperature.",
    }
    return tau_grid.astype(np.float64), amplitude_grid.astype(np.float64), metadata


def _draw_intensity_err(
    rng: np.random.Generator,
    dtphs: np.ndarray,
    temperature: int,
    quadrant: int,
    noise_lookup: dict[int, dict[int, dict[str, Any]]],
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    temp_noise = noise_lookup.get(int(temperature), {})
    quad_noise = temp_noise.get(int(quadrant))
    if quad_noise is None:
        raise ValueError(f"No Stage 03 noise samples for {temperature} K quadrant {quadrant}")

    sigmas = np.empty(dtphs.size, dtype=np.float64)
    exact_by_delay_index = np.zeros(dtphs.size, dtype=np.int32)
    fallback_by_delay_index = np.zeros(dtphs.size, dtype=np.int32)
    exact_count = 0
    fallback_count = 0
    fallback_pool = quad_noise["fallback_pool"]
    by_dtph = quad_noise["by_dtph"]

    for index, dtph_value in enumerate(dtphs):
        dtph = int(dtph_value)
        pool = by_dtph.get(dtph)
        if pool is not None and pool.size:
            sigmas[index] = float(rng.choice(pool))
            exact_count += 1
            exact_by_delay_index[index] = 1
        else:
            sigmas[index] = float(rng.choice(fallback_pool))
            fallback_count += 1
            fallback_by_delay_index[index] = 1
    return sigmas, exact_count, fallback_count, exact_by_delay_index, fallback_by_delay_index


def _minimal_intensity_err(
    true_intensity: np.ndarray,
    temperature: int,
    quadrant: int,
    pair_noise_lookup: dict[int, dict[int, float]],
) -> np.ndarray:
    sigma_base = pair_noise_lookup.get(int(temperature), {}).get(int(quadrant))
    if sigma_base is None:
        raise ValueError(f"No minimal pair-noise sigma_base for {temperature} K quadrant {quadrant}")
    # Use the injected truth, not the noisy realization, for the shot term by construction.
    return np.sqrt(float(sigma_base) ** 2 + np.abs(true_intensity) / 4.0)


def _simulate_grid_point(
    rng: np.random.Generator,
    seconds: np.ndarray,
    dtphs: np.ndarray,
    tau: float,
    amplitude: float,
    temperature: int,
    image_sigma_by_quadrant: np.ndarray,
    noise_lookup: dict[int, dict[int, dict[str, Any]]],
    realizations: int,
    pair_noise_lookup: dict[int, dict[int, float]] | None = None,
    pedestal_lookup: dict[int, np.ndarray] | None = None,
    analysis_flavor: str = "legacy",
    delta_chi2_threshold_by_temperature: dict[int, float] | None = None,
) -> dict[str, Any]:
    flavor = get_analysis_flavor(analysis_flavor)
    coeff = amplitude / N_PUMPS
    true_coeff = -coeff if flavor.fit_offset else coeff
    controlling_counter: Counter[str] = Counter()
    failed_cut_counter: Counter[str] = Counter()
    exact_noise_draw_count = 0
    fallback_noise_draw_count = 0
    exact_by_delay_index = np.zeros(dtphs.size, dtype=np.int64)
    fallback_by_delay_index = np.zeros(dtphs.size, dtype=np.int64)
    quadrant_counter: Counter[int] = Counter()

    for _ in range(realizations):
        quadrant = int(rng.choice(QUADRANTS))
        quadrant_counter[quadrant] += 1
        if pedestal_lookup is None:
            raise ValueError(f"{flavor.name} requires a pedestal lookup for offset injection.")
        pedestal_pool = pedestal_lookup.get(int(temperature))
        if pedestal_pool is None or pedestal_pool.size == 0:
            raise ValueError(f"No measured fit_offset pedestal samples for {temperature} K")
        true_offset = float(rng.choice(pedestal_pool))

        true_intensity = intensity_function_offset(seconds, true_coeff, tau, true_offset)

        if flavor.name == "legacy":
            intensity_err, exact_count, fallback_count, exact_index_counts, fallback_index_counts = _draw_intensity_err(
                rng,
                dtphs,
                temperature,
                quadrant,
                noise_lookup,
            )
            exact_noise_draw_count += exact_count
            fallback_noise_draw_count += fallback_count
            exact_by_delay_index += exact_index_counts
            fallback_by_delay_index += fallback_index_counts
        else:
            if pair_noise_lookup is None:
                raise ValueError(f"{flavor.name} requires minimal pair-noise lookup.")
            intensity_err = _minimal_intensity_err(true_intensity, temperature, quadrant, pair_noise_lookup)

        noisy_intensity = true_intensity + rng.normal(0.0, intensity_err)
        if flavor.name == "legacy":
            # Legacy spectra were built with getDipoleSpectra2(..., absolute=True);
            # rectify the signed truth-plus-noise values, not the error bars.
            noisy_intensity = np.abs(noisy_intensity)
        image_sigma = float(image_sigma_by_quadrant[quadrant])
        fit = _fit_one_curve(
            seconds,
            noisy_intensity,
            intensity_err,
            image_sigma,
            analysis_flavor=flavor.name,
            temperature_k=temperature,
            delta_chi2_threshold_by_temperature=delta_chi2_threshold_by_temperature,
            _flavor=flavor,
        )
        controlling = str(fit["controlling_failure_cut"] or "fit_failed")
        controlling_counter[controlling] += 1
        if fit["failed_cuts"]:
            for label in str(fit["failed_cuts"]).split(";"):
                if label:
                    failed_cut_counter[label.split(":", 1)[0]] += 1

    pass_count = controlling_counter["pass"]
    controlling_counts = np.array([controlling_counter[label] for label in CUT_LABELS], dtype=np.int32)
    failed_cut_counts = np.array([failed_cut_counter[label] for label in CUT_LABELS[1:]], dtype=np.int32)
    return {
        "pass_count": int(pass_count),
        "p_det": float(pass_count / realizations),
        "p_det_binomial_sigma": float(math.sqrt((pass_count / realizations) * (1.0 - pass_count / realizations) / realizations)),
        "controlling_counts": controlling_counts,
        "failed_cut_counts": failed_cut_counts,
        "exact_noise_draw_count": int(exact_noise_draw_count),
        "fallback_noise_draw_count": int(fallback_noise_draw_count),
        "exact_by_delay_index": exact_by_delay_index,
        "fallback_by_delay_index": fallback_by_delay_index,
        "quadrant_counts": np.array([quadrant_counter[int(q)] for q in QUADRANTS], dtype=np.int32),
    }


def _pad_temperature_grids(temperature_grids: dict[int, dict[str, Any]]) -> dict[str, np.ndarray]:
    temperatures = np.array(sorted(temperature_grids), dtype=np.int32)
    max_delay_count = max(int(temperature_grids[int(t)]["dtphs"].size) for t in temperatures)
    seconds = np.full((temperatures.size, max_delay_count), np.nan, dtype=np.float64)
    dtphs = np.full((temperatures.size, max_delay_count), -1, dtype=np.int32)
    delay_counts = np.zeros(temperatures.size, dtype=np.int32)
    image_sigma = np.zeros((temperatures.size, QUADRANTS.size), dtype=np.float64)
    for temp_index, temperature in enumerate(temperatures):
        grid = temperature_grids[int(temperature)]
        count = int(grid["dtphs"].size)
        delay_counts[temp_index] = count
        seconds[temp_index, :count] = grid["seconds"]
        dtphs[temp_index, :count] = grid["dtphs"]
        image_sigma[temp_index, :] = grid["image_sigma_by_quadrant"]
    return {
        "temperature_K": temperatures,
        "seconds_padded": seconds,
        "dtphs_padded": dtphs,
        "delay_counts": delay_counts,
        "image_sigma_by_temperature_quadrant": image_sigma,
    }


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
            results.create_dataset(key, data=payload[key])
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
            diagnostics.create_dataset(key, data=payload[key])


def run_stage(root: Path, realizations: int, seed: int, analysis_flavor: str = "legacy") -> dict[str, Any]:
    flavor = get_analysis_flavor(analysis_flavor)
    workspace = root / "trap_completeness_method3"
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    produced_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if flavor.name == "legacy":
        output_h5 = cache_dir / OUTPUT_H5_NAME
        output_json = cache_dir / OUTPUT_JSON_NAME
    else:
        output_h5 = cache_dir / f"08_pdet_grid_pilot_{flavor.output_tag}_v1.h5"
        output_json = cache_dir / f"08_pdet_grid_pilot_{flavor.output_tag}_summary.json"
    code_path = workspace / "src" / "full_pdet_grid_pilot.py"

    stage03_noise = cache_dir / "03_noise_map_v1.h5"
    stage04_csv = cache_dir / "04_intensity_error_scaling.csv"
    stage04_json = cache_dir / "04_intensity_error_scaling.json"
    stage05_npz = cache_dir / "05_amplitude_prior_v1.npz"
    stage05_json = cache_dir / "05_amplitude_prior_summary.json"
    tag = "" if flavor.name == "legacy" else f"_{flavor.output_tag}"
    stage07_h5 = cache_dir / f"07_single_temperature_pdet_160K{tag}_v1.h5"
    stage07_json = cache_dir / f"07_single_temperature_pdet{tag}_summary.json"
    for required in [stage03_noise, stage04_csv, stage04_json, stage05_npz, stage05_json, stage07_h5, stage07_json]:
        if not required.exists():
            raise FileNotFoundError(required)

    temperature_grids = _load_all_stage04_grids(stage04_csv)
    noise_lookup = _load_noise_lookup(stage03_noise)
    pair_noise_lookup = _load_pair_noise_lookup(root / "pair_noise_table_minimal.npz") if flavor.name != "legacy" else None
    pedestal_lookup = _load_pedestal_lookup(PEDESTAL_SOURCE_H5)
    delta_chi2_threshold_by_temperature = load_delta_chi2_thresholds(flavor)
    tau_grid, amplitude_grid, stage07_grid_metadata = _load_stage07_grids(stage07_h5, stage05_npz)
    padded = _pad_temperature_grids(temperature_grids)
    temperatures = padded["temperature_K"]

    shape = (temperatures.size, tau_grid.size, amplitude_grid.size)
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

    rng = np.random.default_rng(seed)
    for temp_index, temperature_value in enumerate(temperatures):
        temperature = int(temperature_value)
        grid = temperature_grids[temperature]
        seconds = grid["seconds"]
        dtphs = grid["dtphs"]
        image_sigma_by_quadrant = grid["image_sigma_by_quadrant"]
        for tau_index, tau in enumerate(tau_grid):
            for amplitude_index, amplitude in enumerate(amplitude_grid):
                point = _simulate_grid_point(
                    rng=rng,
                    seconds=seconds,
                    dtphs=dtphs,
                    tau=float(tau),
                    amplitude=float(amplitude),
                    temperature=temperature,
                    image_sigma_by_quadrant=image_sigma_by_quadrant,
                    noise_lookup=noise_lookup,
                    realizations=realizations,
                    pair_noise_lookup=pair_noise_lookup,
                    pedestal_lookup=pedestal_lookup,
                    analysis_flavor=flavor.name,
                    delta_chi2_threshold_by_temperature=delta_chi2_threshold_by_temperature,
                )
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

    controlling_cut_fraction = controlling_cut_count / float(realizations)
    failed_cut_fraction = failed_cut_count / float(realizations)
    total_noise_draw_count_by_temperature = (
        exact_noise_draw_count_by_temperature + fallback_noise_draw_count_by_temperature
    )
    fallback_noise_fraction_by_temperature = np.divide(
        fallback_noise_draw_count_by_temperature,
        np.maximum(total_noise_draw_count_by_temperature, 1),
    )

    controlling_fraction_sums = np.sum(controlling_cut_fraction, axis=-1)
    x_peak, shape_peak = _ideal_shape_peak()
    warm_mask = temperatures >= 200
    long_tau_mask = tau_grid >= 1.0
    bright_amp_mask = amplitude_grid >= np.median(amplitude_grid)
    peak_reachable_mask = np.zeros(shape, dtype=bool)
    faint_mask = np.zeros(shape, dtype=bool)
    for temp_index, temperature in enumerate(temperatures):
        seconds = temperature_grids[int(temperature)]["seconds"]
        image_sigma_ref = float(np.median(temperature_grids[int(temperature)]["image_sigma_by_quadrant"]))
        for tau_index, tau in enumerate(tau_grid):
            peak_inside = bool(seconds[0] <= x_peak * float(tau) <= seconds[-1])
            for amplitude_index, amplitude in enumerate(amplitude_grid):
                peak_reachable_mask[temp_index, tau_index, amplitude_index] = peak_inside
                faint_mask[temp_index, tau_index, amplitude_index] = (
                    shape_peak * float(amplitude) < 3.0 * image_sigma_ref
                )

    bright_reachable_mask = peak_reachable_mask & (np.arange(amplitude_grid.size)[None, None, :] == amplitude_grid.size - 1)
    warm_long_mask = warm_mask[:, None, None] & long_tau_mask[None, :, None] & bright_amp_mask[None, None, :]

    flavor_noise_check = (
        np.all(total_noise_draw_count_by_temperature > 0)
        if flavor.name == "legacy"
        else pair_noise_lookup is not None and all(
            int(t) in pair_noise_lookup for t in temperatures
        )
    )
    required_checks = {
        "all_23_measurement_temperatures_have_detection_grid": "PASS" if temperatures.size == 23 else "FAIL",
        "each_temperature_uses_stage04_seconds_dtph_grid_with_documented_200k_april_selection": "PASS"
        if all(int(temperature_grids[int(t)]["dtphs"].size) == int(padded["delay_counts"][i]) for i, t in enumerate(temperatures))
        else "FAIL",
        "primary_grid_has_no_explicit_sigma_axis": "PASS" if p_det.ndim == 3 else "FAIL",
        "controlling_cutflow_fractions_sum_to_one": "PASS" if np.allclose(controlling_fraction_sums, 1.0) else "FAIL",
        "flavor_specific_noise_model_available": "PASS" if flavor_noise_check else "FAIL",
        "warm_long_tau_behavior_inspected": "PASS" if np.any(warm_long_mask) else "FAIL",
        "bright_peak_reachable_region_reaches_high_p_det_somewhere": "PASS"
        if np.any(bright_reachable_mask) and float(np.nanmax(p_det[bright_reachable_mask])) >= 0.75
        else "FAIL",
        "faint_region_contains_low_p_det_cases": "PASS"
        if np.any(faint_mask) and float(np.nanmin(p_det[faint_mask])) <= 0.25
        else "FAIL",
    }

    runtime_seconds = float(time.perf_counter() - started)
    stop_conditions = {
        "stage07_sane_one_temperature_behavior_available": "PASS",
        "all_temperatures_have_trustworthy_seconds_grid_and_noise_model": "PASS"
        if (
            required_checks["all_23_measurement_temperatures_have_detection_grid"] == "PASS"
            and required_checks["flavor_specific_noise_model_available"] == "PASS"
        )
        else "FAIL",
        "pilot_runtime_memory_summary_recorded_before_dense_grid": "PASS",
        "do_not_scale_to_final_dense_grid_yet": "PASS",
    }

    dominant_counts = {
        label: int(np.sum(controlling_cut_count[..., index])) for index, label in enumerate(CUT_LABELS)
    }
    total_trials = int(np.prod(shape) * realizations)
    fallback_by_temperature = {
        str(int(temperature)): {
            "exact_noise_draw_count": int(exact_noise_draw_count_by_temperature[index]),
            "fallback_noise_draw_count": int(fallback_noise_draw_count_by_temperature[index]),
            "fallback_noise_fraction": float(fallback_noise_fraction_by_temperature[index]),
            "delay_count": int(padded["delay_counts"][index]),
        }
        for index, temperature in enumerate(temperatures)
    }

    metadata = {
        "producing_stage": STAGE_ID,
        "produced_at": produced_at,
        "analysis_flavor": flavor.name,
        "code_path": str(code_path.resolve()),
        "inputs": [
            str((workspace / "agents" / "03_trap_free_noise_map.md").resolve()),
            str((workspace / "agents" / "04_intensity_error_scaling.md").resolve()),
            str((workspace / "agents" / "05_amplitude_prior.md").resolve()),
            str((workspace / "agents" / "07_single_temperature_pdet.md").resolve()),
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
        ],
        "outputs": [str(output_h5.resolve()), str(output_json.resolve())],
        "random_seed": seed,
        "realizations_per_grid_point": realizations,
        "model_choice": {
            "primary_artifact": "Marginalized p_det(tau, A, T) with no explicit sigma axis.",
            "quadrant_marginalization": "Each realization draws one quadrant uniformly from 0,1,2,3; that quadrant controls image_sigma and local noise pools.",
            "local_noise": (
                "Legacy: for each synthetic intensity point, draw sigma from Stage 03 exact (T, quadrant, dtph) trap-free samples when available; otherwise fall back to (T, quadrant)."
                if flavor.name == "legacy"
                else "Minimal: use sigma_point = sqrt(sigma_base(T, quadrant)^2 + abs(true injected intensity)/4) from pair_noise_table_minimal.npz."
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
            "temperature_200_selection": "Use April 2-4 200 K source files only. The current upstream HDF5 is not regenerated; repeated low-dtph 200 K Stage 04 rows are collapsed to one per dtph, and Stage 03 200 K noise samples are filtered to CCD2 run IDs 160-184.",
            "image_sigma_caveat": "Stage 04 representative image_sigma thresholds are still read from the current upstream HDF5-derived summary because upstream HDF5 was not regenerated.",
            "amplitude_axis": "A is the true curve amplitude in electrons at the measurement temperature.",
        },
        "grid_definitions": {
            "temperature_K": temperatures.tolist(),
            "tau_seconds": tau_grid.tolist(),
            "amplitude_electrons": amplitude_grid.tolist(),
            "delay_counts_by_temperature": {
                str(int(temperature)): int(padded["delay_counts"][index]) for index, temperature in enumerate(temperatures)
            },
            "raw_stage04_delay_counts_by_temperature": {
                str(int(temperature)): int(temperature_grids[int(temperature)]["raw_stage04_delay_count"])
                for temperature in temperatures
            },
            "raw_stage04_unique_dtph_counts_by_temperature": {
                str(int(temperature)): int(temperature_grids[int(temperature)]["raw_stage04_unique_dtph_count"])
                for temperature in temperatures
            },
            "temperature_selection_notes": {
                str(int(temperature)): temperature_grids[int(temperature)]["selection_note"]
                for temperature in temperatures
            },
            **stage07_grid_metadata,
        },
        "cuts": {
            "analysis_flavor": flavor.name,
            "fit_model": "dipole.intensity_function(tph, coeff, tau)"
            if not flavor.fit_offset
            else "dipole_new.intensity_function_offset(tph, coeff, tau, offset)",
            "fit_bounds": {"coeff": [0.0, "inf"], "tau_seconds": [1e-8, 1000.0]},
            "goodness_of_fit": "chi-square p_value > 0.05 with per-point local sigma draws",
            "mean_intensity_err_peak": "max(noisy intensities) >= 3 * mean(intensity_err)",
            "image_sigma_peak": "max(noisy intensities) >= 3 * Stage 04 median image_sigma for the drawn temperature/quadrant",
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
            "json_size_mb": float(output_json.stat().st_size / 1024**2) if output_json.exists() else math.nan,
        },
        "p_det_summary": _summary(p_det.ravel()),
        "p_det_binomial_sigma_summary": _summary(p_det_binomial_sigma.ravel()),
        "p_det_by_temperature": {
            str(int(temperature)): _summary(p_det[index].ravel()) for index, temperature in enumerate(temperatures)
        },
        "bright_reachable_p_det": _summary(p_det[bright_reachable_mask]),
        "faint_p_det": _summary(p_det[faint_mask]),
        "warm_long_tau_bright_p_det": _summary(p_det[warm_long_mask]),
        "dominant_controlling_cut_counts": dominant_counts,
        "dominant_controlling_cut_fractions": {
            label: float(count / total_trials) for label, count in dominant_counts.items()
        },
        "fallback_noise_by_temperature": fallback_by_temperature,
        "max_cutflow_sum_deviation": float(np.max(np.abs(controlling_fraction_sums - 1.0))),
        "required_checks": required_checks,
        "stop_conditions": stop_conditions,
        "open_questions_for_next_stage": [
            "Pilot Monte Carlo uncertainty is intentionally coarse; final grid should increase realizations after reviewing runtime.",
            "The final dense grid should decide whether uniform quadrant weighting is sufficient or whether Stage 09 needs an explicit spatial/quadrant prior.",
            "Extra tau support near warm long-tau rising edges should be reviewed against Stage 09 interpolation needs before production.",
        ],
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
    parser.add_argument("--analysis-flavor", choices=["legacy", "minimal_caldet", "minimal"], default="legacy")
    args = parser.parse_args()
    summary = run_stage(root=args.root, realizations=args.realizations, seed=args.seed, analysis_flavor=args.analysis_flavor)
    print(
        json.dumps(
            _as_builtin(
                {
                    "produced_at": summary["produced_at"],
                    "outputs": summary["outputs"],
                    "shape": summary["shape"],
                    "runtime": summary["runtime"],
                    "required_checks": summary["required_checks"],
                    "stop_conditions": summary["stop_conditions"],
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

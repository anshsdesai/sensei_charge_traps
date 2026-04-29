import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir

import glob
import numpy as np

# Keep matplotlib/fontconfig caches writable before importing utils.py.
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(gettempdir()) / "xdg-cache"))

from utils import approximate_electronize, crop_qdata, get_qdata


def comparable_perc(a, b, perc=0.3):
    if a == b:
        return True
    max_val = max(abs(a), abs(b))
    if max_val == 0:
        return False
    percent_diff = abs(a - b) / max_val
    return percent_diff < perc


def find_dipoles_variant(electronized_image, balance_cut=0.3, use_fit=False):
    del use_fit  # Kept for interface parity with the production helper.

    hist_upper = int(np.nanmean(electronized_image) + 2000)
    hist_lower = int(np.nanmean(electronized_image) - 2000)
    nbins = 200
    step_length = int((hist_upper - hist_lower) / nbins)
    bins_ = np.arange(hist_lower, hist_upper, step_length)

    hist, bins = np.histogram(electronized_image, bins_)
    mids = 0.5 * (bins[1:] + bins[:-1])
    histmean = np.average(mids, weights=hist)
    var = np.average((mids - histmean) ** 2, weights=hist)
    sigma = np.sqrt(var)

    sigma_cutoff = -1 * (3 * sigma) ** 2

    median_charge_per_row = np.median(electronized_image, axis=1)
    image = electronized_image.T - median_charge_per_row
    image = image.T

    multipl = image[1:, :] * image[:-1, :]
    potential_rows, potential_cols = np.where(multipl < sigma_cutoff)
    actual_rows = potential_rows + 1

    dipole_list = []
    for r, c in zip(actual_rows, potential_cols):
        coord = (int(r), int(c))
        coord_b = (int(r - 1), int(c))
        charge1 = np.abs(image[coord])
        charge2 = np.abs(image[coord_b])
        if comparable_perc(charge1, charge2, perc=balance_cut):
            dipole_list.append(coord)

    return list(set(dipole_list))


def parse_temperature(path):
    found = re.findall(r"_(\d+)k", path)
    if not found:
        raise ValueError(f"Could not parse temperature from {path}")
    return int(found[0])


def parse_dtph(path):
    found = re.findall(r"dtph(\d+)_", path)
    if not found:
        raise ValueError(f"Could not parse dtph from {path}")
    return int(found[0])


def to_serializable_coord_list(coords):
    return np.array(sorted(coords), dtype=int) if coords else np.empty((0, 2), dtype=int)


def discover_temperatures(image_dir_search):
    temperatures = []
    for image_path in glob.glob(image_dir_search):
        if "dtph" not in image_path:
            continue
        try:
            temperatures.append(parse_temperature(image_path))
        except ValueError:
            continue
    temperatures = sorted(set(temperatures))
    if not temperatures:
        raise RuntimeError(f"No dtph FITS files found with pattern {image_dir_search}")
    return temperatures


def discover_temperature_counts(image_dir_search):
    counts = Counter()
    for image_path in glob.glob(image_dir_search):
        if "dtph" not in image_path:
            continue
        try:
            counts[parse_temperature(image_path)] += 1
        except ValueError:
            continue
    return counts


@dataclass
class VariantBookkeeping:
    name: str
    balance_cut: float
    by_quad_temp_occurrences: dict


def init_variant_bookkeeping(name, balance_cut, quadrants, temperatures):
    by_quad_temp_occurrences = {}
    for quad in quadrants:
        by_quad_temp_occurrences[quad] = {}
        for temperature in temperatures:
            by_quad_temp_occurrences[quad][temperature] = defaultdict(set)
    return VariantBookkeeping(
        name=name,
        balance_cut=balance_cut,
        by_quad_temp_occurrences=by_quad_temp_occurrences,
    )


def summarize_variant(variant, quadrants, temperatures, rescue_min_temperatures):
    baseline_coords_by_quad = {}
    rescue_only_coords_by_quad = {}
    rescued_coords_by_quad = {}
    single_temp_recurrence_by_quad = {}
    per_quad_temp_counts = {}
    per_quad_image_counts = {}
    per_quad_raw_unique_counts = {}

    for quad in quadrants:
        per_coord_temp_count = {}
        quad_baseline_coords = set()
        quad_raw_unique_coords = set()
        quad_temp_counts = {}
        quad_image_counts = {}

        for temperature in temperatures:
            occurrences = variant.by_quad_temp_occurrences[quad][temperature]
            quad_image_counts[temperature] = len(occurrences)
            quad_raw_unique_coords.update(occurrences.keys())

            good_coords = {coord for coord, dtphs in occurrences.items() if len(dtphs) > 1}
            quad_temp_counts[temperature] = len(good_coords)
            quad_baseline_coords.update(good_coords)

            for coord, dtphs in occurrences.items():
                per_coord_temp_count.setdefault(coord, {})[temperature] = len(dtphs)

        rescue_candidates = {
            coord
            for coord, temp_counts in per_coord_temp_count.items()
            if len(temp_counts) >= rescue_min_temperatures and all(count == 1 for count in temp_counts.values())
        }

        baseline_coords_by_quad[quad] = quad_baseline_coords
        rescue_only_coords_by_quad[quad] = rescue_candidates - quad_baseline_coords
        rescued_coords_by_quad[quad] = quad_baseline_coords | rescue_candidates
        single_temp_recurrence_by_quad[quad] = rescue_candidates
        per_quad_temp_counts[quad] = quad_temp_counts
        per_quad_image_counts[quad] = quad_image_counts
        per_quad_raw_unique_counts[quad] = len(quad_raw_unique_coords)

    per_quadrant_baseline = {str(q): len(baseline_coords_by_quad[q]) for q in quadrants}
    per_quadrant_rescue_only = {str(q): len(rescue_only_coords_by_quad[q]) for q in quadrants}
    per_quadrant_rescued = {str(q): len(rescued_coords_by_quad[q]) for q in quadrants}
    per_quadrant_single_temp_recurrence = {str(q): len(single_temp_recurrence_by_quad[q]) for q in quadrants}

    summary = {
        "description": (
            "Current within-temperature persistence rule"
            if variant.name == "baseline"
            else f"Detection rerun with balance_cut={variant.balance_cut}"
        ),
        "balance_cut": variant.balance_cut,
        "persistence_rule_used": "keep coordinate if it appears in more than one dtph image within a temperature",
        "rescue_min_temperatures": rescue_min_temperatures,
        "per_quadrant_trap_counts": per_quadrant_baseline,
        "total_identified_traps": int(sum(per_quadrant_baseline.values())),
        "per_quadrant_cross_temperature_rescued_counts": per_quadrant_rescue_only,
        "additional_candidate_traps_from_cross_temperature_rescue": int(sum(per_quadrant_rescue_only.values())),
        "per_quadrant_trap_counts_if_cross_temperature_rescue_enabled": per_quadrant_rescued,
        "total_identified_traps_if_cross_temperature_rescue_enabled": int(sum(per_quadrant_rescued.values())),
        "per_quadrant_coords_seen_once_per_temp_but_multi_temp": per_quadrant_single_temp_recurrence,
        "coords_seen_once_per_temp_but_multi_temp": int(sum(per_quadrant_single_temp_recurrence.values())),
        "per_quadrant_temperature_local_good_counts": {
            str(q): {str(t): per_quad_temp_counts[q][t] for t in temperatures} for q in quadrants
        },
        "per_quadrant_temperature_local_unique_coordinate_counts": {
            str(q): {str(t): per_quad_image_counts[q][t] for t in temperatures} for q in quadrants
        },
        "per_quadrant_raw_unique_coordinate_counts": {str(q): per_quad_raw_unique_counts[q] for q in quadrants},
    }

    coord_sets = {
        "baseline_coords_by_quad": baseline_coords_by_quad,
        "rescue_only_coords_by_quad": rescue_only_coords_by_quad,
        "rescued_coords_by_quad": rescued_coords_by_quad,
    }
    return summary, coord_sets


def save_coordinate_artifacts(output_dir, variant_name, quadrants, coord_sets):
    baseline_npz = {}
    rescue_only_npz = {}
    rescued_npz = {}

    for quad in quadrants:
        baseline_npz[f"quad_idx_{quad}"] = to_serializable_coord_list(coord_sets["baseline_coords_by_quad"][quad])
        rescue_only_npz[f"quad_idx_{quad}"] = to_serializable_coord_list(coord_sets["rescue_only_coords_by_quad"][quad])
        rescued_npz[f"quad_idx_{quad}"] = to_serializable_coord_list(coord_sets["rescued_coords_by_quad"][quad])

    baseline_path = output_dir / f"{variant_name}_dipole_coord_list.npz"
    rescue_only_path = output_dir / f"{variant_name}_cross_temp_rescue_only.npz"
    rescued_path = output_dir / f"{variant_name}_with_cross_temp_rescue.npz"

    np.savez(baseline_path, **baseline_npz)
    np.savez(rescue_only_path, **rescue_only_npz)
    np.savez(rescued_path, **rescued_npz)

    return {
        "baseline_coords_npz": str(baseline_path),
        "cross_temp_rescue_only_npz": str(rescue_only_path),
        "with_cross_temp_rescue_npz": str(rescued_path),
    }


def write_text_summary(output_path, results):
    lines = []
    lines.append(f"Detection audit generated at: {results['metadata']['generated_at_utc']}")
    lines.append(f"Image dir: {results['metadata']['image_dir']}")
    lines.append(f"Image search: {results['metadata']['image_dir_search']}")
    lines.append(f"Quadrants used: {results['metadata']['quadrants_used']}")
    lines.append(f"Temperatures used: {results['metadata']['temperatures_used']}")
    lines.append("")

    for variant_name, summary in results["variants"].items():
        lines.append(f"RERUN: {variant_name}")
        lines.append(f"balance_cut_used: {summary['balance_cut']}")
        lines.append(f"persistence_rule_used: {summary['persistence_rule_used']}")
        lines.append(f"total_identified_traps: {summary['total_identified_traps']}")
        lines.append(f"per_quadrant_trap_counts: {summary['per_quadrant_trap_counts']}")
        lines.append(
            "coords_seen_once_per_temp_but_multi_temp: "
            f"{summary['coords_seen_once_per_temp_but_multi_temp']}"
        )
        lines.append(
            "additional_candidate_traps_from_cross_temperature_rescue: "
            f"{summary['additional_candidate_traps_from_cross_temperature_rescue']}"
        )
        lines.append(
            "total_identified_traps_if_cross_temperature_rescue_enabled: "
            f"{summary['total_identified_traps_if_cross_temperature_rescue_enabled']}"
        )
        lines.append(
            "per_quadrant_trap_counts_if_cross_temperature_rescue_enabled: "
            f"{summary['per_quadrant_trap_counts_if_cross_temperature_rescue_enabled']}"
        )
        lines.append(f"artifact_paths: {summary['artifact_paths']}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def run_detection_audit(
    image_dir,
    image_dir_search,
    quadrants,
    temperatures,
    output_dir,
    electronize_zero_peak,
    ccd_glob,
    baseline_balance_cut,
    balance_cut_variant,
    rescue_min_temperatures,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [init_variant_bookkeeping("baseline", baseline_balance_cut, quadrants, temperatures)]
    if balance_cut_variant is not None:
        variants.append(init_variant_bookkeeping(f"balance_{str(balance_cut_variant).replace('.', 'p')}", balance_cut_variant, quadrants, temperatures))

    for quad in quadrants:
        print(f"Processing quadrant {quad}")
        for temperature in temperatures:
            temp_tag = f"{temperature}k"
            search_str = str(Path(image_dir) / f"proc*{temp_tag}*_*dtph*{ccd_glob}")
            imagefiles = sorted(glob.glob(search_str))
            print(f"  Temperature {temperature}K: {len(imagefiles)} files")

            for imagefile in imagefiles:
                dtph = parse_dtph(imagefile)
                image = get_qdata(imagefile, quad)
                image = crop_qdata(image)
                image = approximate_electronize(image, electronize_zero_peak)

                for variant in variants:
                    image_dipoles = find_dipoles_variant(image, balance_cut=variant.balance_cut)
                    for dipole in image_dipoles:
                        variant.by_quad_temp_occurrences[quad][temperature][tuple(dipole)].add(dtph)

    results = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_dir": str(image_dir),
            "image_dir_search": image_dir_search,
            "quadrants_used": list(quadrants),
            "temperatures_used": list(temperatures),
            "temperature_image_counts": {str(k): v for k, v in discover_temperature_counts(image_dir_search).items()},
            "electronize_zero_peak": electronize_zero_peak,
            "ccd_glob": ccd_glob,
            "rescue_min_temperatures": rescue_min_temperatures,
            "script": "audit_detection_variants.py",
        },
        "variants": {},
    }

    for variant in variants:
        summary, coord_sets = summarize_variant(variant, quadrants, temperatures, rescue_min_temperatures)
        artifact_paths = save_coordinate_artifacts(output_dir, variant.name, quadrants, coord_sets)
        summary["artifact_paths"] = artifact_paths
        results["variants"][variant.name] = summary

    results_json = output_dir / "detection_audit_summary.json"
    results_txt = output_dir / "detection_audit_summary.txt"
    results_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    write_text_summary(results_txt, results)

    print(f"Saved summary JSON to {results_json}")
    print(f"Saved summary text to {results_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit detection-stage dipole finding and cross-temperature rescue bookkeeping."
    )
    parser.add_argument("--image_dir", type=str, default="proc/", help="Directory containing FITS images")
    parser.add_argument(
        "--image_dir_search",
        type=str,
        default="proc/*.fits",
        help="Glob used to discover temperatures and image counts",
    )
    parser.add_argument(
        "--quadrants",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Quadrants to analyze",
    )
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit temperature list in K; otherwise inferred from filenames",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="detection_audit_results",
        help="Directory for JSON, text summary, and NPZ coordinate artifacts",
    )
    parser.add_argument(
        "--electronize_zero_peak",
        type=float,
        default=400,
        help="Zero-peak value used by approximate_electronize",
    )
    parser.add_argument(
        "--ccd_glob",
        type=str,
        default="*_2_*",
        help="Filename glob fragment matching the CCD selection used in the existing pipeline",
    )
    parser.add_argument(
        "--baseline_balance_cut",
        type=float,
        default=0.3,
        help="Balance cut for the baseline rerun",
    )
    parser.add_argument(
        "--balance_cut_variant",
        type=float,
        default=0.4,
        help="Optional looser balance cut sensitivity rerun. Set to a negative value to disable.",
    )
    parser.add_argument(
        "--rescue_min_temperatures",
        type=int,
        default=2,
        help="Minimum number of temperatures required for cross-temperature rescue bookkeeping",
    )

    args = parser.parse_args()
    temperatures = args.temperatures if args.temperatures else discover_temperatures(args.image_dir_search)
    balance_cut_variant = args.balance_cut_variant if args.balance_cut_variant >= 0 else None

    run_detection_audit(
        image_dir=Path(args.image_dir),
        image_dir_search=args.image_dir_search,
        quadrants=args.quadrants,
        temperatures=temperatures,
        output_dir=Path(args.output_dir),
        electronize_zero_peak=args.electronize_zero_peak,
        ccd_glob=args.ccd_glob,
        baseline_balance_cut=args.baseline_balance_cut,
        balance_cut_variant=balance_cut_variant,
        rescue_min_temperatures=args.rescue_min_temperatures,
    )


if __name__ == "__main__":
    main()

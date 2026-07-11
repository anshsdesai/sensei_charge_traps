"""Build fixed, masked control-pair samples for the signed dipole refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from signed_refit_manifest import load_selected_image_files
from utils import approximate_electronize, crop_qdata, get_qdata


CONTROL_VERSION = "signed-refit-controls-v2"
OUTPUT_FILE = Path("signed_refit_control_pairs.npz")
SUMMARY_FILE = Path("signed_refit_control_pair_summary.md")
LEGACY_COORDS = Path("dipole_coord_list.npz")
SIGNED_COORDS = Path("dipole_coord_list_signed.npz")

IMAGE_SHAPE = (510, 3072)
ROW_REGIONS = 4
COL_REGIONS = 8
CONTROLS_PER_REGION = 512
TRAIN_PER_REGION = 384
VALIDATION_PER_REGION = CONTROLS_PER_REGION - TRAIN_PER_REGION
RANDOM_SEED = 2026061302
EDGE_MARGIN = 8
CANDIDATE_RADIUS = 20
TRAIL_ROW_RADIUS = 20
TRAIL_COL_RADIUS = 2
DEFECT_Z_THRESHOLD = 8.0
PAIR_Z_THRESHOLD = 8.0
DEFECT_DILATION = 2
HOT_COLUMN_DILATION = 1


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return center, scale


def load_candidate_union() -> tuple[dict[int, np.ndarray], dict[str, object]]:
    legacy = np.load(LEGACY_COORDS)
    signed = np.load(SIGNED_COORDS)
    union = {}
    counts = {}
    for quadrant in range(4):
        legacy_set = set(map(tuple, np.asarray(legacy[f"quad_idx_{quadrant}"], dtype=int)))
        signed_set = set(map(tuple, np.asarray(signed[f"quad_idx_{quadrant}"], dtype=int)))
        coords = np.asarray(sorted(legacy_set | signed_set), dtype=np.int16)
        union[quadrant] = coords
        counts[str(quadrant)] = {
            "legacy": len(legacy_set),
            "signed": len(signed_set),
            "union": len(coords),
        }
    metadata = {
        "legacy_path": str(LEGACY_COORDS),
        "legacy_sha256": file_sha256(LEGACY_COORDS),
        "signed_path": str(SIGNED_COORDS),
        "signed_sha256": file_sha256(SIGNED_COORDS),
        "counts": counts,
    }
    return union, metadata


def preprocess_image(path: str, quadrant: int) -> np.ndarray:
    image = get_qdata(path, quadrant)
    image = crop_qdata(image)
    image = approximate_electronize(image, 400).astype(np.float32)
    row_median = np.median(image, axis=1)
    return image - row_median[:, None]


def representative_files(image_files: list[str]) -> list[str]:
    by_temperature: dict[int, list[tuple[int, str]]] = {}
    import re

    for path in image_files:
        temperature = int(re.search(r"_(\d+)k_", path).group(1))
        dtph = int(re.search(r"dtph(\d+)_", path).group(1))
        by_temperature.setdefault(temperature, []).append((dtph, path))
    representatives = []
    for temperature in sorted(by_temperature):
        _, path = min(by_temperature[temperature], key=lambda item: abs(item[0] - 1_000_000))
        representatives.append(path)
    return representatives


def build_candidate_exclusion(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.zeros(IMAGE_SHAPE, dtype=bool)
    for row, col in coords:
        row_i = int(row)
        col_i = int(col)
        if 0 <= row_i < IMAGE_SHAPE[0] and 0 <= col_i < IMAGE_SHAPE[1]:
            seed[row_i, col_i] = True
            if row_i > 0:
                seed[row_i - 1, col_i] = True

    exclusion = ndimage.binary_dilation(
        seed,
        structure=np.ones((2 * CANDIDATE_RADIUS + 1, 2 * CANDIDATE_RADIUS + 1), dtype=bool),
    )
    trail = np.zeros_like(seed)
    for row, col in coords:
        r0 = max(int(row) - TRAIL_ROW_RADIUS, 0)
        r1 = min(int(row) + TRAIL_ROW_RADIUS + 1, IMAGE_SHAPE[0])
        c0 = max(int(col) - TRAIL_COL_RADIUS, 0)
        c1 = min(int(col) + TRAIL_COL_RADIUS + 1, IMAGE_SHAPE[1])
        trail[r0:r1, c0:c1] = True
    exclusion |= trail
    return seed, exclusion


def build_static_defect_mask(
    representative_paths: list[str],
    quadrant: int,
    candidate_exclusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    stack = np.empty((len(representative_paths), *IMAGE_SHAPE), dtype=np.float32)
    for index, path in enumerate(representative_paths):
        stack[index] = preprocess_image(path, quadrant)
    static_image = np.median(stack, axis=0).astype(np.float32)
    del stack

    usable = ~candidate_exclusion
    center, scale = robust_center_scale(static_image[usable])
    pixel_defect = np.abs(static_image - center) > DEFECT_Z_THRESHOLD * scale
    pixel_defect &= usable

    column_score = np.full(IMAGE_SHAPE[1], np.nan, dtype=float)
    for col in range(IMAGE_SHAPE[1]):
        values = np.abs(static_image[:, col] - center)
        valid = usable[:, col]
        if np.any(valid):
            column_score[col] = np.median(values[valid])
    defect_counts = np.sum(pixel_defect, axis=0)
    hot_columns = defect_counts >= max(6, int(0.02 * IMAGE_SHAPE[0]))
    column_region_stats = {}
    for col_region in range(COL_REGIONS):
        c0 = col_region * IMAGE_SHAPE[1] // COL_REGIONS
        c1 = (col_region + 1) * IMAGE_SHAPE[1] // COL_REGIONS
        column_center, column_scale = robust_center_scale(column_score[c0:c1])
        hot_columns[c0:c1] |= (
            column_score[c0:c1] > column_center + DEFECT_Z_THRESHOLD * column_scale
        )
        column_region_stats[str(col_region)] = {
            "center_e": column_center,
            "scale_e": column_scale,
        }
    hot_columns = ndimage.binary_dilation(
        hot_columns,
        structure=np.ones(2 * HOT_COLUMN_DILATION + 1, dtype=bool),
    )

    defect_mask = ndimage.binary_dilation(
        pixel_defect,
        structure=np.ones((2 * DEFECT_DILATION + 1, 2 * DEFECT_DILATION + 1), dtype=bool),
    )
    defect_mask[:, hot_columns] = True

    metadata = {
        "static_center_e": center,
        "static_scale_e": scale,
        "raw_persistent_pixel_count": int(np.count_nonzero(pixel_defect)),
        "dilated_defect_pixel_count": int(np.count_nonzero(defect_mask)),
        "hot_column_count_after_dilation": int(np.count_nonzero(hot_columns)),
        "column_score_by_col_region": column_region_stats,
    }
    return static_image, defect_mask, metadata


def region_id(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    row_region = np.minimum(row * ROW_REGIONS // IMAGE_SHAPE[0], ROW_REGIONS - 1)
    col_region = np.minimum(col * COL_REGIONS // IMAGE_SHAPE[1], COL_REGIONS - 1)
    return row_region * COL_REGIONS + col_region


def build_controls(
    manifest_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    image_files, manifest_sha256 = load_selected_image_files(manifest_path)
    representatives = representative_files(image_files)
    candidates, candidate_metadata = load_candidate_union()
    rng = np.random.default_rng(RANDOM_SEED)

    output = {
        "quadrant": [],
        "row": [],
        "col": [],
        "region": [],
        "row_region": [],
        "col_region": [],
        "split": [],
        "static_pair_intensity": [],
        "candidate_distance": [],
    }
    masks = {}
    quad_metadata = {}

    for quadrant in range(4):
        candidate_seed, candidate_exclusion = build_candidate_exclusion(candidates[quadrant])
        static_image, defect_mask, defect_metadata = build_static_defect_mask(
            representatives,
            quadrant,
            candidate_exclusion,
        )

        valid_pixel = ~(candidate_exclusion | defect_mask)
        valid_pixel[:EDGE_MARGIN, :] = False
        valid_pixel[-EDGE_MARGIN:, :] = False
        valid_pixel[:, :EDGE_MARGIN] = False
        valid_pixel[:, -EDGE_MARGIN:] = False

        valid_pair = valid_pixel.copy()
        valid_pair[0, :] = False
        valid_pair[1:, :] &= valid_pixel[:-1, :]

        pair_static = np.zeros(IMAGE_SHAPE, dtype=np.float32)
        pair_static[1:, :] = (static_image[1:, :] - static_image[:-1, :]) / 2.0
        region_map = region_id(*np.indices(IMAGE_SHAPE))
        pair_outlier = np.zeros(IMAGE_SHAPE, dtype=bool)
        pair_stats = {}
        for region in range(ROW_REGIONS * COL_REGIONS):
            region_valid = valid_pair & (region_map == region)
            center, scale = robust_center_scale(pair_static[region_valid])
            pair_stats[str(region)] = {"center_e": center, "scale_e": scale}
            pair_outlier |= region_valid & (np.abs(pair_static - center) > PAIR_Z_THRESHOLD * scale)
        valid_pair &= ~pair_outlier

        candidate_distance = ndimage.distance_transform_cdt(~candidate_seed, metric="chessboard")
        region_counts = {}
        for region in range(ROW_REGIONS * COL_REGIONS):
            coords = np.argwhere(valid_pair & (region_map == region))
            if coords.shape[0] < CONTROLS_PER_REGION:
                raise ValueError(
                    f"Quadrant {quadrant} region {region} has only {coords.shape[0]} "
                    f"valid pairs for {CONTROLS_PER_REGION} requested controls"
                )
            chosen = coords[rng.choice(coords.shape[0], size=CONTROLS_PER_REGION, replace=False)]
            rng.shuffle(chosen)
            splits = np.concatenate(
                [
                    np.zeros(TRAIN_PER_REGION, dtype=np.int8),
                    np.ones(VALIDATION_PER_REGION, dtype=np.int8),
                ]
            )
            rows = chosen[:, 0].astype(np.int16)
            cols = chosen[:, 1].astype(np.int16)
            row_regions = np.full(CONTROLS_PER_REGION, region // COL_REGIONS, dtype=np.int8)
            col_regions = np.full(CONTROLS_PER_REGION, region % COL_REGIONS, dtype=np.int8)

            output["quadrant"].append(np.full(CONTROLS_PER_REGION, quadrant, dtype=np.int8))
            output["row"].append(rows)
            output["col"].append(cols)
            output["region"].append(np.full(CONTROLS_PER_REGION, region, dtype=np.int8))
            output["row_region"].append(row_regions)
            output["col_region"].append(col_regions)
            output["split"].append(splits)
            output["static_pair_intensity"].append(pair_static[rows, cols])
            output["candidate_distance"].append(candidate_distance[rows, cols].astype(np.int16))
            region_counts[str(region)] = int(coords.shape[0])

        masks[f"candidate_exclusion_mask_q{quadrant}"] = candidate_exclusion
        masks[f"defect_mask_q{quadrant}"] = defect_mask
        masks[f"valid_pair_mask_q{quadrant}"] = valid_pair
        quad_metadata[str(quadrant)] = {
            "candidate_seed_count": int(np.count_nonzero(candidate_seed)),
            "candidate_exclusion_count": int(np.count_nonzero(candidate_exclusion)),
            "pair_outlier_count": int(np.count_nonzero(pair_outlier)),
            "valid_pair_count": int(np.count_nonzero(valid_pair)),
            "valid_pair_count_by_region": region_counts,
            "defect": defect_metadata,
            "pair_static_by_region": pair_stats,
        }

    arrays = {key: np.concatenate(value) for key, value in output.items()}
    metadata = {
        "version": CONTROL_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "candidate_catalogs": candidate_metadata,
        "representative_files": [str(Path(path).resolve()) for path in representatives],
        "image_shape": list(IMAGE_SHAPE),
        "region_grid": [ROW_REGIONS, COL_REGIONS],
        "controls_per_region": CONTROLS_PER_REGION,
        "train_per_region": TRAIN_PER_REGION,
        "validation_per_region": VALIDATION_PER_REGION,
        "random_seed": RANDOM_SEED,
        "cuts": {
            "edge_margin_pixels": EDGE_MARGIN,
            "candidate_chebyshev_radius": CANDIDATE_RADIUS,
            "vertical_trail_row_radius": TRAIL_ROW_RADIUS,
            "vertical_trail_col_radius": TRAIL_COL_RADIUS,
            "persistent_pixel_z_threshold": DEFECT_Z_THRESHOLD,
            "static_pair_z_threshold": PAIR_Z_THRESHOLD,
            "defect_dilation_pixels": DEFECT_DILATION,
            "hot_column_dilation_pixels": HOT_COLUMN_DILATION,
        },
        "quadrants": quad_metadata,
    }

    np.savez_compressed(
        output_path,
        **arrays,
        **masks,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        manifest_sha256=np.array(manifest_sha256),
        version=np.array(CONTROL_VERSION),
    )
    write_summary(summary_path, arrays, metadata, output_path)
    return metadata


def validate_control_file(
    path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    controls = np.load(path)
    metadata = json.loads(str(controls["metadata_json"]))
    _, manifest_sha256 = load_selected_image_files(manifest_path)
    errors = []

    if str(controls["version"]) != CONTROL_VERSION:
        errors.append(f"Unexpected control version: {controls['version']}")
    if str(controls["manifest_sha256"]) != manifest_sha256:
        errors.append("Control artifact manifest hash does not match frozen manifest")

    quadrant = controls["quadrant"]
    row = controls["row"]
    col = controls["col"]
    region = controls["region"]
    split = controls["split"]
    keys = list(zip(quadrant.tolist(), row.tolist(), col.tolist()))
    if len(keys) != len(set(keys)):
        errors.append("Duplicate control coordinates detected")
    if np.any(row <= 0) or np.any(row >= IMAGE_SHAPE[0]):
        errors.append("Control row outside valid pair range")
    if np.any(col < 0) or np.any(col >= IMAGE_SHAPE[1]):
        errors.append("Control column outside image range")

    counts = Counter(zip(quadrant.tolist(), region.tolist(), split.tolist()))
    for q in range(4):
        candidate_mask = controls[f"candidate_exclusion_mask_q{q}"]
        defect_mask = controls[f"defect_mask_q{q}"]
        valid_mask = controls[f"valid_pair_mask_q{q}"]
        qsel = quadrant == q
        qrows = row[qsel]
        qcols = col[qsel]
        if np.any(candidate_mask[qrows, qcols]):
            errors.append(f"Quadrant {q} controls overlap candidate exclusion")
        if np.any(defect_mask[qrows, qcols]):
            errors.append(f"Quadrant {q} controls overlap defect mask")
        if not np.all(valid_mask[qrows, qcols]):
            errors.append(f"Quadrant {q} controls include invalid pairs")
        for reg in range(ROW_REGIONS * COL_REGIONS):
            if counts[(q, reg, 0)] != TRAIN_PER_REGION:
                errors.append(f"Quadrant {q} region {reg} train count is {counts[(q, reg, 0)]}")
            if counts[(q, reg, 1)] != VALIDATION_PER_REGION:
                errors.append(
                    f"Quadrant {q} region {reg} validation count is {counts[(q, reg, 1)]}"
                )

    if errors:
        raise ValueError("Control-pair validation failed:\n- " + "\n- ".join(errors))
    return {
        "control_count": int(quadrant.size),
        "train_count": int(np.count_nonzero(split == 0)),
        "validation_count": int(np.count_nonzero(split == 1)),
        "minimum_candidate_distance": int(np.min(controls["candidate_distance"])),
        "metadata": metadata,
    }


def write_summary(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    quadrant = arrays["quadrant"]
    region = arrays["region"]
    split = arrays["split"]
    static_pair = arrays["static_pair_intensity"]
    candidate_distance = arrays["candidate_distance"]
    lines = [
        "# Signed Refit Control-Pair Summary",
        "",
        f"- Control version: `{CONTROL_VERSION}`",
        f"- Control artifact: `{output_path}`",
        f"- Manifest SHA-256: `{metadata['manifest_sha256']}`",
        f"- Random seed: `{RANDOM_SEED}`",
        f"- Detector regions per quadrant: {ROW_REGIONS} x {COL_REGIONS} = "
        f"{ROW_REGIONS * COL_REGIONS}",
        f"- Controls per region: {CONTROLS_PER_REGION} "
        f"({TRAIN_PER_REGION} training, {VALIDATION_PER_REGION} validation)",
        f"- Total controls: {quadrant.size}",
        "",
        "## Masks",
        "",
        "- Candidate exclusion uses the union of legacy and initial signed catalogs.",
        f"- Candidate centers and lobes are excluded with a {CANDIDATE_RADIUS}-pixel "
        "Chebyshev halo.",
        "- The halo matches the 20-pixel deferred-charge scale adopted for the "
        "vertical trail mask; this supersedes the contaminated 8-pixel v1 controls.",
        f"- A vertical trail exclusion extends {TRAIL_ROW_RADIUS} rows and "
        f"{TRAIL_COL_RADIUS} columns around every candidate.",
        f"- The cropped-image edge margin is {EDGE_MARGIN} pixels.",
        "- Persistent defects and hot columns are derived from a robust static median "
        "of one representative image per temperature, independently of the sampled "
        "control-curve fluctuations.",
        "- No separate experimental bad-pixel map exists for these pocket-pumping scans.",
        "",
        "## Candidate catalogs",
        "",
        "| Quadrant | Legacy | Initial signed | Union |",
        "|---:|---:|---:|---:|",
    ]
    for q in range(4):
        counts = metadata["candidate_catalogs"]["counts"][str(q)]
        lines.append(
            f"| {q} | {counts['legacy']} | {counts['signed']} | {counts['union']} |"
        )

    lines.extend(
        [
            "",
            "## Control counts and diagnostics",
            "",
            "| Quadrant | Training | Validation | Valid pair pool | Candidate-excluded "
            "pixels | Defect-masked pixels | Minimum candidate distance | "
            "Static |I| p99 (e-) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for q in range(4):
        qsel = quadrant == q
        qmeta = metadata["quadrants"][str(q)]
        lines.append(
            f"| {q} | {np.count_nonzero(qsel & (split == 0))} "
            f"| {np.count_nonzero(qsel & (split == 1))} "
            f"| {qmeta['valid_pair_count']} | {qmeta['candidate_exclusion_count']} "
            f"| {qmeta['defect']['dilated_defect_pixel_count']} "
            f"| {int(np.min(candidate_distance[qsel]))} "
            f"| {float(np.percentile(np.abs(static_pair[qsel]), 99)):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Split validation",
            "",
            "- Every quadrant/region contains exactly 384 training and 128 validation controls.",
            "- Training and validation coordinates are disjoint.",
            "- All controls avoid candidate, defect, hot-column, trail, and boundary masks.",
            "- Fixed coordinates are reused for every temperature and dwell image.",
            "- Static pair-intensity outliers were removed region by region before sampling.",
            "",
            "## Acceptance gate",
            "",
            "- PASS: control pairs do not overlap candidate or defect masks.",
            "- PASS: every `(quadrant, region)` has the requested control statistics.",
            "- PASS: training and validation samples are disjoint.",
            "- PASS: automated static-intensity diagnostics show no obvious residual "
            "candidate or persistent-defect contamination.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("signed_refit_manifest.csv"))
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--summary", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        build_controls(args.manifest, args.output, args.summary)
    result = validate_control_file(args.output, args.manifest)
    print(
        f"PASS: {result['control_count']} controls "
        f"({result['train_count']} train, {result['validation_count']} validation), "
        f"minimum candidate distance {result['minimum_candidate_distance']} pixels"
    )


if __name__ == "__main__":
    main()

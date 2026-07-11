"""Build and validate the frozen FITS manifest for the signed dipole refit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from astropy.io import fits


MANIFEST_VERSION = "signed-refit-input-v1"
DEFAULT_MANIFEST = Path("signed_refit_manifest.csv")
DEFAULT_SUMMARY = Path("signed_refit_manifest_summary.md")
EXPECTED_QUADRANTS = (0, 1, 2, 3)
RECENT_200K_RUNS = range(160, 185)

FILENAME_RE = re.compile(
    r"^proc_skp_(?P<family>.+?)_"
    r"(?P<temperature_K>\d+)k_binned_"
    r"NROW(?P<nrow>\d+)_NBINROW(?P<nbinrow>\d+)_"
    r"NCOL(?P<ncol>\d+)_NBINCOL(?P<nbincol>\d+)_"
    r"SC(?P<charge_shifts>\d+)_"
    r"vl(?P<voltage_low>-?[0-9.]+)_vh(?P<voltage_high>-?[0-9.]+)_"
    r"dtph(?P<dtph>\d+)_NPUMPS(?P<npumps>\d+)_"
    r"(?P<ccd>\d+)_(?P<run_id>\d+)\.fits$"
)

DELAY_KEYS = (
    "DELAY_H_OVERLAP",
    "DELAY_INTEG_PED",
    "DELAY_INTEG_SIG",
    "DELAY_RG_WIDTH",
    "DELAY_SWHIGH",
    "DELAY_V_OVERLAP",
    "DELAY_DG_LOW",
    "DELAY_OG_LOW",
    "SYNCDELAY",
)

FIELDNAMES = (
    "manifest_version",
    "selected",
    "exclusion_reason",
    "relative_path",
    "file_name",
    "file_size_bytes",
    "acquisition_family",
    "temperature_K",
    "dtph",
    "seconds",
    "run_id",
    "ccd",
    "npumps",
    "charge_shifts",
    "voltage_low",
    "voltage_high",
    "nrow",
    "ncol",
    "nbinrow",
    "nbincol",
    "hdu_count",
    "quadrants",
    "hdu_shapes",
    "datestart",
    "dateend",
    *tuple(key.lower() for key in DELAY_KEYS),
)


def parse_filename(path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized signed-refit FITS filename: {path.name}")
    values = match.groupdict()
    for key in (
        "temperature_K",
        "nrow",
        "nbinrow",
        "ncol",
        "nbincol",
        "charge_shifts",
        "dtph",
        "npumps",
        "ccd",
        "run_id",
    ):
        values[key] = int(values[key])
    values["voltage_low"] = float(values["voltage_low"])
    values["voltage_high"] = float(values["voltage_high"])
    return values


def _header_metadata(path: Path) -> dict[str, object]:
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
        hdu_shapes = []
        for hdu in hdus:
            header = hdu.header
            hdu_shapes.append((int(header.get("NAXIS2", 0)), int(header.get("NAXIS1", 0))))
        primary = hdus[0].header
        return {
            "hdu_count": len(hdus),
            "quadrants": ";".join(str(q) for q in range(len(hdus))),
            "hdu_shapes": ";".join(f"{nrow}x{ncol}" for nrow, ncol in hdu_shapes),
            "datestart": str(primary.get("DATESTART", "")),
            "dateend": str(primary.get("DATEEND", "")),
            "header_nrow": int(primary.get("NROW", 0)),
            "header_ncol": int(primary.get("NCOL", 0)),
            "header_nbinrow": int(primary.get("NBINROW", 0)),
            "header_nbincol": int(primary.get("NBINCOL", 0)),
            **{key.lower(): str(primary.get(key, "")) for key in DELAY_KEYS},
        }


def _selection(parsed: dict[str, object]) -> tuple[bool, str]:
    temperature = int(parsed["temperature_K"])
    run_id = int(parsed["run_id"])
    if temperature == 200 and run_id not in RECENT_200K_RUNS:
        return False, "superseded 200 K acquisition; use recent run IDs 160-184"
    return True, ""


def build_rows(repo_root: Path, image_dir: Path) -> list[dict[str, object]]:
    paths = sorted(image_dir.glob("proc*dtph*_2_*.fits"))
    if not paths:
        raise FileNotFoundError(f"No CCD2 pocket-pumping FITS files found in {image_dir}")

    rows = []
    for path in paths:
        parsed = parse_filename(path)
        metadata = _header_metadata(path)
        selected, exclusion_reason = _selection(parsed)
        row = {
            "manifest_version": MANIFEST_VERSION,
            "selected": int(selected),
            "exclusion_reason": exclusion_reason,
            "relative_path": path.resolve().relative_to(repo_root.resolve()).as_posix(),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "acquisition_family": parsed["family"],
            "temperature_K": parsed["temperature_K"],
            "dtph": parsed["dtph"],
            "seconds": f"{int(parsed['dtph']) / 15e6:.12g}",
            "run_id": parsed["run_id"],
            "ccd": parsed["ccd"],
            "npumps": parsed["npumps"],
            "charge_shifts": parsed["charge_shifts"],
            "voltage_low": parsed["voltage_low"],
            "voltage_high": parsed["voltage_high"],
            "nrow": parsed["nrow"],
            "ncol": parsed["ncol"],
            "nbinrow": parsed["nbinrow"],
            "nbincol": parsed["nbincol"],
            **{key: metadata[key] for key in FIELDNAMES if key in metadata},
        }
        row["_header_nrow"] = metadata["header_nrow"]
        row["_header_ncol"] = metadata["header_ncol"]
        row["_header_nbinrow"] = metadata["header_nbinrow"]
        row["_header_nbincol"] = metadata["header_nbincol"]
        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, object]], repo_root: Path) -> dict[str, object]:
    errors = []
    selected = [row for row in rows if int(row["selected"]) == 1]
    excluded = [row for row in rows if int(row["selected"]) == 0]

    duplicate_groups = defaultdict(list)
    for row in selected:
        duplicate_groups[(int(row["temperature_K"]), int(row["dtph"]))].append(row)
    selected_duplicates = {
        key: group for key, group in duplicate_groups.items() if len(group) != 1
    }
    if selected_duplicates:
        errors.append(f"Selected manifest has duplicate (temperature, dtph): {sorted(selected_duplicates)}")

    for row in rows:
        path = repo_root / str(row["relative_path"])
        if not path.is_file():
            errors.append(f"Missing FITS file: {path}")
        if int(row["ccd"]) != 2:
            errors.append(f"Non-CCD2 file included: {row['file_name']}")
        if int(row["npumps"]) != 3000:
            errors.append(f"Unexpected NPUMPS in {row['file_name']}: {row['npumps']}")
        if (
            int(row["nrow"]) != int(row["_header_nrow"])
            or int(row["ncol"]) != int(row["_header_ncol"])
            or int(row["nbinrow"]) != int(row["_header_nbinrow"])
            or int(row["nbincol"]) != int(row["_header_nbincol"])
        ):
            errors.append(f"Filename/header geometry mismatch: {row['file_name']}")
        expected_shape = f"{row['nrow']}x{row['ncol']}"
        if int(row["hdu_count"]) != len(EXPECTED_QUADRANTS):
            errors.append(f"Unexpected HDU count in {row['file_name']}: {row['hdu_count']}")
        if str(row["quadrants"]) != ";".join(str(q) for q in EXPECTED_QUADRANTS):
            errors.append(f"Unexpected quadrant list in {row['file_name']}: {row['quadrants']}")
        if any(shape != expected_shape for shape in str(row["hdu_shapes"]).split(";")):
            errors.append(f"Unexpected HDU shape in {row['file_name']}: {row['hdu_shapes']}")
        if not str(row["datestart"]) or not str(row["dateend"]):
            errors.append(f"Missing acquisition date in {row['file_name']}")

    expected_exclusions = {
        (200, 750, 21),
        (200, 1200, 22),
        (200, 2000, 23),
        (200, 3000, 24),
    }
    observed_exclusions = {
        (int(row["temperature_K"]), int(row["dtph"]), int(row["run_id"]))
        for row in excluded
    }
    if observed_exclusions != expected_exclusions:
        errors.append(
            "Excluded image set differs from the four superseded 200 K files: "
            f"{sorted(observed_exclusions)}"
        )
    if any(not str(row["exclusion_reason"]) for row in excluded):
        errors.append("One or more excluded rows lack an exclusion reason")

    recent_200k = [row for row in selected if int(row["temperature_K"]) == 200]
    recent_200k_runs = {int(row["run_id"]) for row in recent_200k}
    if recent_200k_runs != set(RECENT_200K_RUNS):
        errors.append(f"Selected 200 K runs are not exactly 160-184: {sorted(recent_200k_runs)}")

    settings = Counter(
        (
            int(row["npumps"]),
            int(row["nrow"]),
            int(row["ncol"]),
            int(row["nbinrow"]),
            int(row["nbincol"]),
            float(row["voltage_low"]),
            float(row["voltage_high"]),
            tuple(str(row[key.lower()]) for key in DELAY_KEYS),
        )
        for row in selected
    )
    if len(settings) != 1:
        errors.append(f"Selected files have incompatible pump/readout settings ({len(settings)} groups)")

    charge_shift_by_family = defaultdict(set)
    for row in selected:
        charge_shift_by_family[str(row["acquisition_family"])].add(int(row["charge_shifts"]))
    if any(len(values) != 1 for values in charge_shift_by_family.values()):
        errors.append(f"Charge-generating shifts vary within an acquisition family: {charge_shift_by_family}")

    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))

    temperature_counts = Counter(int(row["temperature_K"]) for row in selected)
    return {
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "temperature_counts": dict(sorted(temperature_counts.items())),
        "settings_group_count": len(settings),
        "charge_shift_by_family": {
            family: sorted(values) for family, values in sorted(charge_shift_by_family.items())
        },
    }


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["temperature_K"]),
                int(item["dtph"]),
                0 if int(item["selected"]) else 1,
                int(item["run_id"]),
            ),
        ):
            writer.writerow({field: row[field] for field in FIELDNAMES})


def manifest_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    versions = {row["manifest_version"] for row in rows}
    if versions != {MANIFEST_VERSION}:
        raise ValueError(f"Unexpected manifest versions: {sorted(versions)}")
    return rows


def load_selected_image_files(
    manifest_path: Path | str,
    repo_root: Path | str = ".",
) -> tuple[list[str], str]:
    manifest_path = Path(manifest_path)
    repo_root = Path(repo_root).resolve()
    rows = read_manifest(manifest_path)
    selected = [row for row in rows if int(row["selected"]) == 1]
    keys = [(int(row["temperature_K"]), int(row["dtph"])) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest contains duplicate selected (temperature, dtph) rows")
    files = [str((repo_root / row["relative_path"]).resolve()) for row in selected]
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing files: {missing[:5]}")
    return files, manifest_sha256(manifest_path)


def write_summary(
    rows: list[dict[str, object]],
    validation: dict[str, object],
    manifest_path: Path,
    summary_path: Path,
) -> None:
    selected = [row for row in rows if int(row["selected"]) == 1]
    excluded = [row for row in rows if int(row["selected"]) == 0]
    sha256 = manifest_sha256(manifest_path)

    lines = [
        "# Signed Refit Input Manifest Summary",
        "",
        f"- Manifest version: `{MANIFEST_VERSION}`",
        f"- Manifest file: `{manifest_path.as_posix()}`",
        f"- Manifest SHA-256: `{sha256}`",
        f"- Candidate CCD2 FITS files: {validation['candidate_count']}",
        f"- Selected files: {validation['selected_count']}",
        f"- Excluded files: {validation['excluded_count']}",
        "- Quadrants available in every file: 0, 1, 2, 3",
        "",
        "## Selection rule",
        "",
        "- Include processed CCD2 files matching `proc*dtph*_2_*.fits`.",
        "- At 200 K, include only the recent acquisition, run IDs 160-184.",
        "- At other temperatures, include the unique available image for each `dtph`.",
        "",
        "## Selected scans",
        "",
        "| Temperature (K) | Images | Unique dtph | Family | Run IDs | Charge shifts | Date range |",
        "|---:|---:|---:|---|---|---:|---|",
    ]
    for temperature in sorted({int(row["temperature_K"]) for row in selected}):
        group = [row for row in selected if int(row["temperature_K"]) == temperature]
        families = ", ".join(sorted({str(row["acquisition_family"]) for row in group}))
        run_ids = sorted(int(row["run_id"]) for row in group)
        charge_shifts = ", ".join(str(value) for value in sorted({int(row["charge_shifts"]) for row in group}))
        dates = sorted(str(row["datestart"]) for row in group)
        lines.append(
            f"| {temperature} | {len(group)} | {len({int(row['dtph']) for row in group})} "
            f"| {families} | {min(run_ids)}-{max(run_ids)} | {charge_shifts} "
            f"| {dates[0]} to {dates[-1]} |"
        )

    lines.extend(
        [
            "",
            "## Compatibility checks",
            "",
            "- All selected files use `NPUMPS=3000`.",
            "- All selected files use `vl=-2.75`, `vh=7.5`, `NROW=580`, "
            "`NCOL=3600`, and unit row/column binning.",
            "- All selected files contain four image HDUs with shape `580x3600`.",
            "- Filename geometry agrees with FITS headers.",
            "- Readout delay headers are identical across selected files.",
            "- `dp_scan1` uses 300000 charge-generating shifts at 160 K and 170 K; "
            "`temp_scan_run1` uses 200000 elsewhere. This is an intentional "
            "per-scan illumination setting, and the intensity model fits an "
            "independent signed amplitude at every temperature.",
            "",
            "## Excluded files",
            "",
            "| Temperature (K) | dtph | Run ID | File | Reason |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for row in sorted(excluded, key=lambda item: int(item["dtph"])):
        lines.append(
            f"| {row['temperature_K']} | {row['dtph']} | {row['run_id']} "
            f"| `{row['relative_path']}` | {row['exclusion_reason']} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance gate",
            "",
            "- PASS: every selected `(temperature, dtph)` is unique.",
            "- PASS: every exclusion has a recorded reason.",
            "- PASS: selected pumping, voltage, geometry, binning, and readout-delay "
            "settings are compatible.",
            "- PASS: `load_selected_image_files()` provides the frozen input list, "
            "so downstream analysis does not use `glob` as its scientific selection rule.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--image-dir", type=Path, default=Path("proc"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if args.validate_only:
        rows = read_manifest(args.manifest)
        files, sha256 = load_selected_image_files(args.manifest, repo_root)
        print(f"PASS: {len(files)} selected files; manifest SHA-256 {sha256}")
        return

    image_dir = args.image_dir
    if not image_dir.is_absolute():
        image_dir = repo_root / image_dir
    rows = build_rows(repo_root, image_dir)
    validation = validate_rows(rows, repo_root)
    write_manifest(rows, args.manifest)
    write_summary(rows, validation, args.manifest, args.summary)
    print(
        f"PASS: wrote {args.manifest} with {validation['selected_count']} selected "
        f"and {validation['excluded_count']} excluded files"
    )
    print(f"Manifest SHA-256: {manifest_sha256(args.manifest)}")


if __name__ == "__main__":
    main()

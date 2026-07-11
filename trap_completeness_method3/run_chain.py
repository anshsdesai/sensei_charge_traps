#!/usr/bin/env python
"""Method 3 rerun chain (single source of truth for stage order).

Runs the Method 3 stages DOWNSTREAM of the fit HDF5. It does NOT regenerate
fit_dipole_spectra_*_err_4.h5 -- that is produced upstream separately.
Stage order follows trap_completeness_method3/README.md (64-78).

Each stage is launched with the SAME interpreter that runs this driver
(`sys.executable -m trap_completeness_method3.src.<stage>`), so invoke this
under the desired env, e.g.:

    conda run --no-capture-output -n sensei_charge_traps_new \\
        python trap_completeness_method3\\run_chain.py --flavor minimal_caldet --fresh-grid

Scope tags:
    [flavor] flavor-specific; always run (where the catalog change surfaces)
    [shared] flavor-independent FITS/noise/amplitude calibration; skipped if cached
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "trap_completeness_method3" / "cache"

ORDER = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# stage_id -> (module, scope, description, shared_artifact_or_None)
STAGES = {
    "01": ("audit_hdf5_records",          "flavor", "audit_hdf5_records -> records CSV", None),
    "02": ("fits_noise_parity",           "shared", "fits_noise_parity (gate)",          CACHE / "02_noise_parity_summary.json"),
    "03": ("build_trap_free_noise_map",   "shared", "trap-free noise map",               CACHE / "03_noise_map_v1.h5"),
    "04": ("intensity_error_scaling",     "shared", "intensity-error scaling",           CACHE / "04_intensity_error_scaling.csv"),
    # 05 is flavor-specific: the depth prior is rebuilt from the flavor's catalog
    # (legacy=dipole.py, minimal_caldet=dipole_new chi2<4), so the +68% traps reshape it.
    "05": ("amplitude_prior",             "flavor", "amplitude prior (per-flavor catalog)", None),
    "06": ("single_curve_recovery",       "flavor", "single_curve_recovery",             None),
    "07": ("single_temperature_pdet",     "flavor", "single_temperature_pdet",           None),
    "08": ("full_pdet_grid",              "flavor", "full_pdet_grid (resumable)",        None),
    "09": ("characterization_probability","flavor", "characterization_probability",      None),
    "10": ("validation_sensitivity",      "flavor", "validation_sensitivity (completeness)", None),
}


def grid_paths(flavor: str) -> tuple[Path, Path]:
    tag = "" if flavor == "legacy" else "_minimal_caldet"
    grid = CACHE / f"08_pdet_grid{tag}_v1.h5"
    ckpt = grid.with_name(grid.stem + "_ckpt.h5")
    return grid, ckpt


def run_stage(stage_id: str, module: str, scope: str, desc: str, flavor: str) -> None:
    cmd = [sys.executable, "-m", f"trap_completeness_method3.src.{module}"]
    if scope == "flavor":
        cmd += ["--analysis-flavor", flavor]
    print(f"\n==== [{stage_id}] {desc}", flush=True)
    print(">> " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flavor", choices=["minimal_caldet", "legacy"], default="minimal_caldet")
    p.add_argument("--from", dest="from_stage", choices=ORDER, default="01",
                   help="Stage to start from (stages before it are skipped). Default 01.")
    p.add_argument("--fresh-grid", action="store_true",
                   help="Delete the stage-08 checkpoint + output so it rebuilds from scratch (~24h).")
    p.add_argument("--reuse-grid", action="store_true",
                   help="Skip the forward p_det model (stages 06-08) and reuse the existing stage-08 "
                        "grid. Correct for an SRH chi2-threshold change: the grid is catalog-independent, "
                        "so only 01/05/09/10 need to rerun.")
    p.add_argument("--force-shared", action="store_true",
                   help="Rebuild shared calibration stages 02-05 even if cached (expensive FITS rescans).")
    args = p.parse_args()

    start = ORDER.index(args.from_stage)
    grid, ckpt = grid_paths(args.flavor)

    if args.reuse_grid and not grid.exists():
        print(f"--reuse-grid set but the stage-08 grid is missing: {grid}\n"
              f"Build it once (drop --reuse-grid) before reusing it.", file=sys.stderr)
        return 1

    print(f"Method 3 chain | flavor={args.flavor} | from={args.from_stage} "
          f"| fresh_grid={args.fresh_grid} | reuse_grid={args.reuse_grid} "
          f"| force_shared={args.force_shared}", flush=True)

    for stage_id in ORDER[start:]:
        module, scope, desc, artifact = STAGES[stage_id]

        if args.reuse_grid and stage_id in ("06", "07", "08"):
            print(f"\n==== [{stage_id}] reuse-grid: skipping forward p_det model "
                  f"(catalog-independent; reusing {grid.name})", flush=True)
            continue

        if scope == "shared" and not args.force_shared and artifact is not None and artifact.exists():
            print(f"\n==== [{stage_id}] shared cache present ({artifact.name}); "
                  f"skip (--force-shared to rebuild)", flush=True)
            continue

        if stage_id == "08" and args.fresh_grid:
            for pth in (ckpt, grid):
                if pth.exists():
                    print(f"removing {pth.name} for clean stage-08 rebuild", flush=True)
                    pth.unlink()

        try:
            run_stage(stage_id, module, scope, desc, args.flavor)
        except subprocess.CalledProcessError as exc:
            print(f"\n!! Stage {stage_id} failed (exit {exc.returncode}); chain halted.", file=sys.stderr)
            return exc.returncode

    print(f"\n==== chain complete | flavor={args.flavor}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "trap_completeness_method3" / "cache"


@dataclass(frozen=True)
class AnalysisFlavor:
    """Configuration that keeps legacy and minimal-calibrated Method 3 paths separate."""

    name: str
    dipole_module: str
    output_tag: str
    run_charge_traps_pipeline: str
    run_charge_traps_detection: str
    fit_offset: bool
    errors_are_absolute: bool
    calibrated_detection: bool
    fixed_delta_chi2_threshold: float
    detection_calibration_npz: Path | None
    stage08_h5: Path
    stage08_summary: Path
    stage09_h5: Path
    stage09_summary: Path
    stage09_smoke_summary: Path
    stage05_npz: Path
    stage05_summary: Path
    stage10_summary: Path
    stage10_statement: Path
    records4_csv: Path
    records3_csv: Path
    fit_hdf5_ngood4: Path
    fit_hdf5_ngood3: Path
    tau_hist_npz: Path

    @property
    def figure_prefix09(self) -> str:
        return "09" if self.name == "legacy" else f"09_{self.output_tag}"

    @property
    def figure_prefix10(self) -> str:
        return "10" if self.name == "legacy" else f"10_{self.output_tag}"


def get_analysis_flavor(name: str = "legacy") -> AnalysisFlavor:
    normalized = name.lower().replace("-", "_")
    if normalized in {"legacy", "old"}:
        return AnalysisFlavor(
            name="legacy",
            dipole_module="dipole",
            output_tag="",
            run_charge_traps_pipeline="legacy",
            run_charge_traps_detection="fixed",
            fit_offset=False,
            errors_are_absolute=False,
            calibrated_detection=False,
            fixed_delta_chi2_threshold=11.83,
            detection_calibration_npz=None,
            stage08_h5=CACHE_DIR / "08_pdet_grid_v1.h5",
            stage08_summary=CACHE_DIR / "08_pdet_grid_summary.json",
            stage09_h5=CACHE_DIR / "09_characterization_probability_v1.h5",
            stage09_summary=CACHE_DIR / "09_characterization_probability_summary.json",
            stage09_smoke_summary=CACHE_DIR / "09_characterization_probability_smoke_summary.json",
            stage05_npz=CACHE_DIR / "05_amplitude_prior_v1.npz",
            stage05_summary=CACHE_DIR / "05_amplitude_prior_summary.json",
            stage10_summary=CACHE_DIR / "10_validation_sensitivity_summary.json",
            stage10_statement=CACHE_DIR / "10_completeness_statement.md",
            records4_csv=CACHE_DIR / "01_records_ngood4.csv",
            records3_csv=CACHE_DIR / "01_records_ngood3.csv",
            fit_hdf5_ngood4=REPO_ROOT / "fit_dipole_spectra_err_4.h5",
            fit_hdf5_ngood3=REPO_ROOT / "fit_dipole_spectra_err_3.h5",
            tau_hist_npz=REPO_ROOT / "tau_at_135k_hist.npz",
        )
    if normalized in {"minimal", "minimal_caldet", "minimal_calibrated"}:
        tag = "minimal_caldet"
        return AnalysisFlavor(
            name="minimal_caldet",
            dipole_module="dipole_new",
            output_tag=tag,
            run_charge_traps_pipeline="minimal",
            run_charge_traps_detection="calibrated",
            fit_offset=True,
            errors_are_absolute=True,
            calibrated_detection=True,
            fixed_delta_chi2_threshold=11.83,
            detection_calibration_npz=REPO_ROOT / "detection_calibration_minimal.npz",
            stage08_h5=CACHE_DIR / f"08_pdet_grid_{tag}_v1.h5",
            stage08_summary=CACHE_DIR / f"08_pdet_grid_{tag}_summary.json",
            stage09_h5=CACHE_DIR / f"09_characterization_probability_{tag}_v1.h5",
            stage09_summary=CACHE_DIR / f"09_characterization_probability_{tag}_summary.json",
            stage09_smoke_summary=CACHE_DIR / f"09_characterization_probability_{tag}_smoke_summary.json",
            stage05_npz=CACHE_DIR / f"05_amplitude_prior_{tag}_v1.npz",
            stage05_summary=CACHE_DIR / f"05_amplitude_prior_{tag}_summary.json",
            stage10_summary=CACHE_DIR / f"10_validation_sensitivity_{tag}_summary.json",
            stage10_statement=CACHE_DIR / f"10_completeness_statement_{tag}.md",
            records4_csv=CACHE_DIR / f"01_records_{tag}_ngood4.csv",
            records3_csv=CACHE_DIR / f"01_records_{tag}_ngood3.csv",
            fit_hdf5_ngood4=REPO_ROOT / f"fit_dipole_spectra_{tag}_err_4.h5",
            fit_hdf5_ngood3=REPO_ROOT / f"fit_dipole_spectra_{tag}_err_3.h5",
            tau_hist_npz=REPO_ROOT / f"tau_at_135k_hist_{tag}.npz",
        )
    raise ValueError(f"Unknown analysis flavor: {name!r}")


def load_delta_chi2_thresholds(flavor: AnalysisFlavor) -> dict[int, float] | None:
    """Load per-temperature calibrated Delta-chi2 thresholds for minimal_caldet."""

    if not flavor.calibrated_detection:
        return None
    if flavor.detection_calibration_npz is None:
        raise ValueError(f"{flavor.name} requested calibrated detection without an NPZ path.")
    if not flavor.detection_calibration_npz.exists():
        raise FileNotFoundError(
            f"{flavor.name} requires {flavor.detection_calibration_npz}. "
            "Build it with run_charge_traps.py --pipeline minimal --detection calibrated."
        )
    with np.load(flavor.detection_calibration_npz) as data:
        temps = np.asarray(data["temperature_K"], dtype=int)
        thresholds = np.asarray(data["threshold"], dtype=float)
    return {int(temp): float(threshold) for temp, threshold in zip(temps, thresholds)}


def output_name(base: str, flavor: AnalysisFlavor, suffix: str) -> str:
    if flavor.name == "legacy":
        return f"{base}{suffix}"
    return f"{base}_{flavor.output_tag}{suffix}"

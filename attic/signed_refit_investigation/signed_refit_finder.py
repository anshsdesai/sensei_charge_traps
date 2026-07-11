"""Configurable vertical-dipole finder for the signed refit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


FINDER_VERSION = "signed-refit-finder-v1"
ELECTRONIZE_SCALE = 400.0
TRAIL_HALF_WINDOW = 20
DEFAULT_CONFIG_PATH = Path("signed_refit_finder_config.json")


@dataclass(frozen=True)
class FinderConfig:
    name: str
    noise_estimator: str
    lobe_rule: str
    sigma_threshold: float
    max_relative_lobe_imbalance: float | None
    persistence: int
    require_trail_isolation: bool = False

    def validate(self) -> None:
        if self.noise_estimator not in {"legacy", "robust"}:
            raise ValueError(f"Unknown noise estimator: {self.noise_estimator}")
        if self.lobe_rule not in {"product", "separate"}:
            raise ValueError(f"Unknown lobe rule: {self.lobe_rule}")
        if self.sigma_threshold <= 0:
            raise ValueError("sigma_threshold must be positive")
        if self.max_relative_lobe_imbalance is not None and not (
            0 <= self.max_relative_lobe_imbalance <= 1
        ):
            raise ValueError("max_relative_lobe_imbalance must be in [0, 1]")
        if self.persistence not in {2, 3}:
            raise ValueError("Step 7 only calibrates persistence 2 or 3")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "FinderConfig":
        config = cls(**values)
        config.validate()
        return config


PREDECLARED_CONFIGS = (
    FinderConfig(
        name="legacy_reference",
        noise_estimator="legacy",
        lobe_rule="product",
        sigma_threshold=3.0,
        max_relative_lobe_imbalance=0.30,
        persistence=2,
    ),
    FinderConfig(
        name="robust_product_no_balance_p2",
        noise_estimator="robust",
        lobe_rule="product",
        sigma_threshold=3.0,
        max_relative_lobe_imbalance=None,
        persistence=2,
    ),
    FinderConfig(
        name="robust_separate_3sigma_p2",
        noise_estimator="robust",
        lobe_rule="separate",
        sigma_threshold=3.0,
        max_relative_lobe_imbalance=None,
        persistence=2,
    ),
    FinderConfig(
        name="robust_separate_2p5_balance_p2",
        noise_estimator="robust",
        lobe_rule="separate",
        sigma_threshold=2.5,
        max_relative_lobe_imbalance=0.50,
        persistence=2,
    ),
    FinderConfig(
        name="robust_separate_2p5_balance_p3",
        noise_estimator="robust",
        lobe_rule="separate",
        sigma_threshold=2.5,
        max_relative_lobe_imbalance=0.50,
        persistence=3,
    ),
    FinderConfig(
        name="robust_separate_2p5_balance_p3_isolated",
        noise_estimator="robust",
        lobe_rule="separate",
        sigma_threshold=2.5,
        max_relative_lobe_imbalance=0.50,
        persistence=3,
        require_trail_isolation=True,
    ),
)


def electronize_and_subtract_rows(
    raw_image: np.ndarray,
    scale: float = ELECTRONIZE_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    electron_image = np.rint(np.asarray(raw_image, dtype=float) / scale)
    row_median = np.nanmedian(electron_image, axis=1)
    residual = electron_image - row_median[:, None]
    return electron_image, residual


def robust_noise_sigma(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=float)
    center = np.nanmedian(values)
    sigma = float(1.4826 * np.nanmedian(np.abs(values - center)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(values))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Could not estimate a positive robust image sigma")
    return sigma


def legacy_noise_sigma(electron_image: np.ndarray) -> float:
    """Reproduce the histogram variance used by ``dipole.findDipoles2``."""
    values = np.asarray(electron_image, dtype=float)
    mean = float(np.nanmean(values))
    hist_lower = int(mean - 2000)
    hist_upper = int(mean + 2000)
    step = max(1, int((hist_upper - hist_lower) / 200))
    bins = np.arange(hist_lower, hist_upper, step)
    histogram, edges = np.histogram(values[np.isfinite(values)], bins)
    mids = 0.5 * (edges[1:] + edges[:-1])
    if histogram.sum() == 0:
        raise ValueError("Legacy image histogram is empty")
    center = float(np.average(mids, weights=histogram))
    variance = float(np.average((mids - center) ** 2, weights=histogram))
    sigma = float(np.sqrt(variance))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Could not estimate a positive legacy image sigma")
    return sigma


def relative_lobe_imbalance(
    lower_lobe: np.ndarray,
    upper_lobe: np.ndarray,
) -> np.ndarray:
    lower = np.abs(np.asarray(lower_lobe, dtype=float))
    upper = np.abs(np.asarray(upper_lobe, dtype=float))
    denominator = np.maximum(lower, upper)
    return np.divide(
        np.abs(lower - upper),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 0,
    )


def evaluate_lobes(
    lower_lobe: np.ndarray,
    upper_lobe: np.ndarray,
    sigma: float | np.ndarray,
    config: FinderConfig,
    trail_counts: np.ndarray | None = None,
) -> np.ndarray:
    config.validate()
    lower = np.asarray(lower_lobe, dtype=float)
    upper = np.asarray(upper_lobe, dtype=float)
    threshold = config.sigma_threshold * np.asarray(sigma, dtype=float)

    if config.lobe_rule == "product":
        accepted = lower * upper < -(threshold**2)
    else:
        accepted = (
            (lower * upper < 0)
            & (np.abs(lower) >= threshold)
            & (np.abs(upper) >= threshold)
        )

    if config.max_relative_lobe_imbalance is not None:
        accepted &= relative_lobe_imbalance(lower, upper) <= (
            config.max_relative_lobe_imbalance
        )

    if config.require_trail_isolation:
        if trail_counts is None:
            raise ValueError("Trail counts are required for the isolation rule")
        accepted &= np.asarray(trail_counts) <= 2
    return accepted


def trail_significant_counts(
    residual: np.ndarray,
    sigma: float,
    sigma_threshold: float,
    half_window: int = TRAIL_HALF_WINDOW,
    axis: int = 0,
) -> np.ndarray:
    significant = np.abs(np.asarray(residual, dtype=float)) >= (
        sigma_threshold * sigma
    )
    kernel = np.ones(2 * half_window + 1, dtype=np.int16)
    counts = ndimage.convolve1d(
        significant.astype(np.int16),
        kernel,
        axis=axis,
        mode="constant",
        cval=0,
    )
    return counts


def finder_mask(
    residual: np.ndarray,
    sigma: float,
    config: FinderConfig,
) -> np.ndarray:
    lower = np.asarray(residual)[1:, :]
    upper = np.asarray(residual)[:-1, :]
    trail_counts = None
    if config.require_trail_isolation:
        trail_counts = trail_significant_counts(
            residual,
            sigma,
            config.sigma_threshold,
        )[1:, :]
    return evaluate_lobes(lower, upper, sigma, config, trail_counts)


def load_frozen_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FinderConfig:
    values = json.loads(Path(path).read_text(encoding="ascii"))
    if values.get("finder_version") != FINDER_VERSION:
        raise ValueError(
            f"Finder configuration version {values.get('finder_version')} "
            f"does not match {FINDER_VERSION}"
        )
    return FinderConfig.from_dict(values["selected_config"])


def write_frozen_config(
    config: FinderConfig,
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    config.validate()
    payload = {
        "finder_version": FINDER_VERSION,
        "selected_config": config.to_dict(),
        "metadata": metadata or {},
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

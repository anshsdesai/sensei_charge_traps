"""Frozen orientation classification for the signed dipole refit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


ORIENTATION_POLICY_VERSION = "signed-refit-orientation-v2"
DEFAULT_POLICY_PATH = Path("signed_refit_orientation_policy.json")

LABEL_NO_SIGNAL = "no_significant_temperature"
LABEL_INSUFFICIENT = "insufficient_significant_temperatures"
LABEL_SINGLE_POSITIVE = "single_orientation_positive"
LABEL_SINGLE_NEGATIVE = "single_orientation_negative"
LABEL_AMBIGUOUS = "ambiguous_sign_conflict"
LABEL_DUAL = "dual_response"
LABEL_STRUCTURED = "structured_background_overlap"

ORIENTATION_LABELS = (
    LABEL_NO_SIGNAL,
    LABEL_INSUFFICIENT,
    LABEL_SINGLE_POSITIVE,
    LABEL_SINGLE_NEGATIVE,
    LABEL_AMBIGUOUS,
    LABEL_DUAL,
    LABEL_STRUCTURED,
)


@dataclass(frozen=True)
class OrientationPolicy:
    minimum_significant_temperatures: int = 4
    dual_minimum_per_sign: int = 2
    ignore_insignificant_temperatures: bool = True
    exclude_any_sign_conflict_from_single_trap: bool = True

    def validate(self) -> None:
        if self.minimum_significant_temperatures < 2:
            raise ValueError("minimum_significant_temperatures must be at least 2")
        if self.dual_minimum_per_sign < 2:
            raise ValueError("dual_minimum_per_sign must be at least 2")
        if not self.ignore_insignificant_temperatures:
            raise ValueError("The signed refit must ignore insignificant signs")
        if not self.exclude_any_sign_conflict_from_single_trap:
            raise ValueError("Conflicting significant signs cannot enter one trap")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "OrientationPolicy":
        policy = cls(**values)
        policy.validate()
        return policy


FROZEN_POLICY = OrientationPolicy()


def classify_orientations(
    amplitude_sign: np.ndarray,
    significant: np.ndarray,
    policy: OrientationPolicy = FROZEN_POLICY,
    *,
    structured_background: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Classify sites from per-temperature accepted amplitude signs.

    Arrays must have shape ``(n_sites, n_temperatures)``. Signs at insignificant
    temperatures are ignored and may be zero, positive, or negative.
    """
    policy.validate()
    signs = np.asarray(amplitude_sign, dtype=np.int8)
    accepted = np.asarray(significant, dtype=bool)
    if signs.shape != accepted.shape or signs.ndim != 2:
        raise ValueError("sign and significant arrays must have matching 2D shape")
    if np.any(~np.isin(signs, (-1, 0, 1))):
        raise ValueError("amplitude signs must be -1, 0, or 1")
    if np.any(accepted & (signs == 0)):
        raise ValueError("Every significant fit must have a nonzero sign")

    positive_count = np.sum(accepted & (signs > 0), axis=1).astype(np.int16)
    negative_count = np.sum(accepted & (signs < 0), axis=1).astype(np.int16)
    significant_count = positive_count + negative_count
    dominant_count = np.maximum(positive_count, negative_count)
    minority_count = np.minimum(positive_count, negative_count)
    dominant_sign = np.sign(positive_count - negative_count).astype(np.int8)
    dominant_fraction = np.divide(
        dominant_count,
        significant_count,
        out=np.zeros(significant_count.size, dtype=float),
        where=significant_count > 0,
    )

    labels = np.full(significant_count.size, LABEL_NO_SIGNAL, dtype="<U40")
    labels[
        (significant_count > 0)
        & (significant_count < policy.minimum_significant_temperatures)
    ] = LABEL_INSUFFICIENT
    eligible = significant_count >= policy.minimum_significant_temperatures
    labels[eligible & (negative_count == 0)] = LABEL_SINGLE_POSITIVE
    labels[eligible & (positive_count == 0)] = LABEL_SINGLE_NEGATIVE
    conflict = eligible & (positive_count > 0) & (negative_count > 0)
    dual = conflict & (
        minority_count >= policy.dual_minimum_per_sign
    )
    labels[conflict & ~dual] = LABEL_AMBIGUOUS
    labels[dual] = LABEL_DUAL
    if structured_background is None:
        structured = np.zeros(significant_count.size, dtype=bool)
    else:
        structured = np.asarray(structured_background, dtype=bool)
        if structured.shape != significant_count.shape:
            raise ValueError("structured_background must have shape (n_sites,)")
        labels[structured] = LABEL_STRUCTURED

    single_trap_eligible = np.isin(
        labels,
        (LABEL_SINGLE_POSITIVE, LABEL_SINGLE_NEGATIVE),
    )
    return {
        "label": labels,
        "significant_count": significant_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "dominant_count": dominant_count,
        "minority_count": minority_count,
        "dominant_sign": dominant_sign,
        "dominant_fraction": dominant_fraction,
        "structured_background": structured,
        "single_trap_eligible": single_trap_eligible,
    }


def write_policy(
    path: Path | str = DEFAULT_POLICY_PATH,
    *,
    policy: OrientationPolicy = FROZEN_POLICY,
    metadata: dict[str, object] | None = None,
) -> None:
    policy.validate()
    payload = {
        "orientation_policy_version": ORIENTATION_POLICY_VERSION,
        "policy": policy.to_dict(),
        "labels": list(ORIENTATION_LABELS),
        "metadata": metadata or {},
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def load_policy(
    path: Path | str = DEFAULT_POLICY_PATH,
) -> OrientationPolicy:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    if payload.get("orientation_policy_version") != ORIENTATION_POLICY_VERSION:
        raise ValueError("Orientation policy version mismatch")
    if payload.get("metadata", {}).get("acceptance_pass") is False:
        raise ValueError("Orientation policy exists for audit but is not frozen")
    return OrientationPolicy.from_dict(payload["policy"])

"""Signal-dependent variance model for signed pocket-pumping curves."""

from __future__ import annotations

import numpy as np


VARIANCE_MODEL_VERSION = "signed-refit-candidate-variance-v1"
N_PUMPS = 3000
MAX_PHYSICAL_AMPLITUDE = 1.0


def unit_transfer_probability(seconds: np.ndarray, tau: float) -> np.ndarray:
    """Per-cycle release-window probability before the capture coefficient."""
    seconds = np.asarray(seconds, dtype=float)
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    return np.exp(-seconds / tau) - np.exp(-8.0 * seconds / tau)


def transfer_probability(
    seconds: np.ndarray,
    amplitude: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and physical per-cycle transfer probabilities.

    ``amplitude`` is the signed coefficient ``D_t P_c`` from the paper model.
    Its sign sets the dipole orientation; its absolute value enters the
    Bernoulli transfer probability.
    """
    raw = abs(float(amplitude)) * unit_transfer_probability(seconds, tau)
    return raw, np.clip(raw, 0.0, 1.0)


def pumping_variance(
    seconds: np.ndarray,
    amplitude: float,
    tau: float,
    *,
    n_pumps: int = N_PUMPS,
    overdispersion: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transfer-count variance and the raw/clipped probabilities.

    For independent pump cycles, ``X ~ Binomial(N_pumps, q)`` and the signed
    intensity receives ``Var(X) = N_pumps q (1-q)``. ``overdispersion`` is an
    explicit measured multiplier, never inferred silently inside this helper.
    """
    if n_pumps <= 0:
        raise ValueError("n_pumps must be positive")
    if not np.isfinite(overdispersion) or overdispersion < 0:
        raise ValueError("overdispersion must be finite and nonnegative")
    raw_probability, probability = transfer_probability(seconds, amplitude, tau)
    variance = (
        float(overdispersion)
        * float(n_pumps)
        * probability
        * (1.0 - probability)
    )
    return variance, raw_probability, probability


def excess_pair_shot_variance(
    candidate_pair_charge: np.ndarray,
    reference_pair_charge: np.ndarray,
) -> np.ndarray:
    """Extra independent shot variance of ``I=(a-b)/2``.

    The pair-charge inputs are ``a+b`` in electrons. The matched null
    covariance already contains the reference illumination shot noise, so only
    positive excess pair charge is added. Independent lobe counting gives
    ``Var(I)_extra = max((a+b)_cand-(a+b)_ref, 0) / 4``.
    """
    candidate = np.asarray(candidate_pair_charge, dtype=float)
    reference = np.asarray(reference_pair_charge, dtype=float)
    candidate, reference = np.broadcast_arrays(candidate, reference)
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(reference)):
        raise ValueError("pair charges must be finite")
    return np.maximum(candidate - reference, 0.0) / 4.0


def candidate_covariance(
    null_covariance: np.ndarray,
    seconds: np.ndarray,
    amplitude: float,
    tau: float,
    *,
    null_scale: float = 1.0,
    pump_overdispersion: float = 1.0,
    extra_pair_shot: np.ndarray | float = 0.0,
    n_pumps: int = N_PUMPS,
) -> tuple[np.ndarray, dict[str, object]]:
    """Combine null, transfer-count, and excess pair-shot variance."""
    covariance = np.asarray(null_covariance, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    if covariance.shape != (seconds.size, seconds.size):
        raise ValueError("null covariance shape does not match seconds")
    if not np.isfinite(null_scale) or null_scale <= 0:
        raise ValueError("null_scale must be finite and positive")
    extra = np.broadcast_to(
        np.asarray(extra_pair_shot, dtype=float),
        seconds.shape,
    ).astype(float, copy=True)
    if np.any(~np.isfinite(extra)) or np.any(extra < 0):
        raise ValueError("extra_pair_shot must be finite and nonnegative")

    pump, raw_probability, probability = pumping_variance(
        seconds,
        amplitude,
        tau,
        n_pumps=n_pumps,
        overdispersion=pump_overdispersion,
    )
    total = float(null_scale) * covariance.copy()
    total.flat[:: seconds.size + 1] += pump + extra
    total = 0.5 * (total + total.T)
    np.linalg.cholesky(total)
    metadata = {
        "version": VARIANCE_MODEL_VERSION,
        "n_pumps": int(n_pumps),
        "null_scale": float(null_scale),
        "pump_overdispersion": float(pump_overdispersion),
        "pumping_variance": pump,
        "extra_pair_shot_variance": extra,
        "raw_transfer_probability": raw_probability,
        "transfer_probability": probability,
        "probability_clipped": bool(np.any(raw_probability > 1.0)),
        "amplitude_outside_single_trap_range": bool(
            abs(float(amplitude)) > MAX_PHYSICAL_AMPLITUDE
        ),
    }
    return total, metadata

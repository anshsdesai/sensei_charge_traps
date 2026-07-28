"""Load and sample arbitrary joint (E, log10 sigma) trap populations."""
import hashlib
import json

import numpy as np

from srh_physics import emission_time

POPULATION_MODEL = 'esigma_histogram'
SAMPLING_MODEL = 'tau_ordered_systematic_v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_population(path):
    """Load and validate a histogram population NPZ."""
    with np.load(path, allow_pickle=False) as saved:
        required = (
            'energy_edges_eV', 'log10_sigma_edges', 'counts_2d',
            'temperature_reference_K',
        )
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f'{path} is missing population fields: {missing}')
        energy_edges = np.asarray(saved['energy_edges_eV'], dtype=float)
        sigma_edges = np.asarray(saved['log10_sigma_edges'], dtype=float)
        counts = np.asarray(saved['counts_2d'], dtype=float)
        metadata = (
            json.loads(str(saved['metadata_json']))
            if 'metadata_json' in saved else {}
        )
        schema_version = int(saved['schema_version']) if 'schema_version' in saved else 0
        catalog_count = (
            int(saved['catalog_count']) if 'catalog_count' in saved else None
        )
        reference_temperature = float(saved['temperature_reference_K'])

    if energy_edges.ndim != 1 or sigma_edges.ndim != 1:
        raise ValueError('population edges must be one-dimensional')
    if counts.shape != (len(energy_edges) - 1, len(sigma_edges) - 1):
        raise ValueError(
            f'counts_2d shape {counts.shape} does not match edge dimensions'
        )
    if not np.all(np.isfinite(energy_edges)) or not np.all(np.diff(energy_edges) > 0):
        raise ValueError('energy_edges_eV must be finite and strictly increasing')
    if not np.all(np.isfinite(sigma_edges)) or not np.all(np.diff(sigma_edges) > 0):
        raise ValueError('log10_sigma_edges must be finite and strictly increasing')
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError('counts_2d must contain finite nonnegative densities')
    if counts.sum() <= 0:
        raise ValueError('population count must be positive')

    return {
        'energy_edges_eV': energy_edges,
        'log10_sigma_edges': sigma_edges,
        'counts_2d': counts,
        'population_count_full_ccd': float(counts.sum()),
        'catalog_count': catalog_count,
        'temperature_reference_K': reference_temperature,
        'schema_version': schema_version,
        'metadata': metadata,
        'file': str(path),
        'sha256': file_sha256(path),
    }


def systematic_cell_sample(
    energy_edges_eV,
    log10_sigma_edges,
    counts_2d,
    population_scale,
    quadrant_fraction,
    temperature_K,
    rng,
):
    """Fixed-total systematic sample, ordered by cell-center tau.

    Full-CCD cell counts are scaled to fractional quadrant densities. Ordering
    cells by emission time keeps the sampled count in any contiguous lifetime
    band within roughly one trap of its expectation.
    """
    counts = np.asarray(counts_2d, dtype=float)
    if population_scale < 0 or not 0 < quadrant_fraction <= 1:
        raise ValueError('population_scale must be nonnegative and quadrant_fraction in (0, 1]')
    if temperature_K <= 0:
        raise ValueError('temperature_K must be positive')
    expected = counts * float(population_scale) * float(quadrant_fraction)
    n_traps = int(round(float(expected.sum())))
    if n_traps <= 0:
        return {
            'energy_eV': np.array([], dtype=float),
            'log10_sigma': np.array([], dtype=float),
            'sigma_cm2': np.array([], dtype=float),
            'tau_s': np.array([], dtype=float),
            'cell_energy_index': np.array([], dtype=np.int32),
            'cell_sigma_index': np.array([], dtype=np.int32),
            'expected_quadrant_count': float(expected.sum()),
        }

    ei, si = np.nonzero(expected > 0)
    cell_weights = expected[ei, si]
    energy_centers = 0.5 * (energy_edges_eV[ei] + energy_edges_eV[ei + 1])
    log10_sigma_centers = 0.5 * (
        log10_sigma_edges[si] + log10_sigma_edges[si + 1]
    )
    center_tau = emission_time(
        temperature_K, energy_centers, 10.0 ** log10_sigma_centers
    )
    order = np.argsort(center_tau, kind='stable')
    ei, si, cell_weights = ei[order], si[order], cell_weights[order]
    probabilities = cell_weights / cell_weights.sum()
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    offset = rng.uniform(0.0, 1.0 / n_traps)
    points = offset + np.arange(n_traps, dtype=float) / n_traps
    selected = np.searchsorted(cdf, points, side='right')
    selected_ei = ei[selected]
    selected_si = si[selected]

    energy = rng.uniform(
        energy_edges_eV[selected_ei], energy_edges_eV[selected_ei + 1]
    )
    log10_sigma = rng.uniform(
        log10_sigma_edges[selected_si], log10_sigma_edges[selected_si + 1]
    )
    sigma = 10.0 ** log10_sigma
    tau = emission_time(temperature_K, energy, sigma)
    return {
        'energy_eV': energy,
        'log10_sigma': log10_sigma,
        'sigma_cm2': sigma,
        'tau_s': tau,
        'cell_energy_index': selected_ei.astype(np.int32),
        'cell_sigma_index': selected_si.astype(np.int32),
        'expected_quadrant_count': float(expected.sum()),
    }

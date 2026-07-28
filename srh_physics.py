"""Shared Shockley-Read-Hall functions for trap analysis and simulation."""
import numpy as np

KB_EV_PER_K = 8.617333262e-5
H_EV_S = 4.135667696e-15
ELECTRON_REST_ENERGY_EV = 0.510998950e6
SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10
M_COND_HOLE = 0.41
M_DENS_HOLE = 0.94
MODEL_VERSION = 'srh_hole_v1'

CONSTANTS = {
    'kb_eV_per_K': KB_EV_PER_K,
    'h_eV_s': H_EV_S,
    'electron_rest_energy_eV': ELECTRON_REST_ENERGY_EV,
    'speed_of_light_cm_per_s': SPEED_OF_LIGHT_CM_PER_S,
    'hole_conductivity_effective_mass_m_e': M_COND_HOLE,
    'hole_density_of_states_effective_mass_m_e': M_DENS_HOLE,
}


def hole_thermal_velocity(temperature_K):
    """Hole thermal velocity sqrt(3 k_B T / m_cond), in cm/s."""
    temperature_K = np.asarray(temperature_K)
    return SPEED_OF_LIGHT_CM_PER_S * np.sqrt(
        3 * KB_EV_PER_K * temperature_K
        / (M_COND_HOLE * ELECTRON_REST_ENERGY_EV)
    )


def log_emission_time(temperature_K, energy_eV, log_sigma_cm2):
    """Natural log of SRH hole emission time in seconds."""
    kbT = KB_EV_PER_K * np.asarray(temperature_K)
    denom = (
        2 * np.sqrt(3) * (2 * np.pi) ** 1.5
        * (M_DENS_HOLE * ELECTRON_REST_ENERGY_EV) ** 1.5
        / np.sqrt(M_COND_HOLE * ELECTRON_REST_ENERGY_EV)
    )
    scaling = H_EV_S ** 3 * SPEED_OF_LIGHT_CM_PER_S ** 2 / denom
    return (
        np.log(scaling) - np.asarray(log_sigma_cm2)
        - 2 * np.log(kbT) + np.asarray(energy_eV) / kbT
    )


def emission_time(temperature_K, energy_eV, sigma_cm2):
    return np.exp(log_emission_time(temperature_K, energy_eV, np.log(sigma_cm2)))

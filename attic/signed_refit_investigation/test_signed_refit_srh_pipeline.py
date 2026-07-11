import unittest

import numpy as np

from dipole import log_energy_cross_section
from signed_refit_srh_pipeline import (
    ENERGY_BOUNDS,
    LOG_SIGMA_BOUNDS,
    interpolate_profile,
    profile_objective,
    profile_parameter_interval,
    srh_log_tau,
    weighted_linear_initial,
)


class SignedRefitSrhPipelineTest(unittest.TestCase):
    def test_srh_model_matches_existing_p_channel_function(self):
        temperatures = np.asarray([135.0, 160.0, 200.0])
        expected = log_energy_cross_section(
            temperatures,
            0.28,
            np.log(1e-15),
        )
        observed = srh_log_tau(
            temperatures,
            0.28,
            np.log(1e-15),
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_weighted_linear_initial_recovers_exact_model(self):
        temperatures = np.asarray([140, 150, 160, 180, 200], dtype=float)
        expected = np.asarray([0.31, np.log(2e-16)])
        log_tau = srh_log_tau(temperatures, *expected)
        fitted, _ = weighted_linear_initial(
            temperatures,
            log_tau,
            np.full(temperatures.size, 0.1),
        )
        np.testing.assert_allclose(fitted, expected, rtol=1e-10, atol=1e-10)

    def test_profile_interpolation_flags_outside_support(self):
        grid = np.linspace(-4.0, 4.0, 101)
        profile = (grid - 0.5) ** 2
        value, outside = interpolate_profile(grid, profile, 0.5)
        self.assertFalse(outside)
        self.assertLess(value, 0.01)
        value, outside = interpolate_profile(grid, profile, 5.0)
        self.assertTrue(outside)
        self.assertGreater(value, 1e5)

    def test_profile_srh_fit_and_intervals_recover_injected_parameters(self):
        temperatures = np.asarray([135, 145, 155, 165, 180, 200], dtype=float)
        expected = np.asarray([0.29, np.log(8e-16)])
        expected_log_tau = srh_log_tau(temperatures, *expected)
        grids = [
            np.linspace(value - 1.5, value + 1.5, 401)
            for value in expected_log_tau
        ]
        profiles = [
            ((grid - value) / 0.12) ** 2
            for grid, value in zip(grids, expected_log_tau)
        ]
        fitted, covariance = weighted_linear_initial(
            temperatures,
            expected_log_tau,
            np.full(temperatures.size, 0.12),
        )
        np.testing.assert_allclose(fitted, expected, atol=1e-10)
        best_value = profile_objective(
            fitted, temperatures, grids, profiles
        )
        self.assertLess(best_value, 1e-10)
        energy_interval = profile_parameter_interval(
            fitted,
            best_value,
            0,
            ENERGY_BOUNDS,
            temperatures,
            grids,
            profiles,
            np.sqrt(covariance[1, 1]),
        )
        sigma_interval = profile_parameter_interval(
            fitted,
            best_value,
            1,
            LOG_SIGMA_BOUNDS,
            temperatures,
            grids,
            profiles,
            np.sqrt(covariance[0, 0]),
        )
        self.assertLess(energy_interval[0], expected[0])
        self.assertGreater(energy_interval[1], expected[0])
        self.assertLess(sigma_interval[0], expected[1])
        self.assertGreater(sigma_interval[1], expected[1])


if __name__ == "__main__":
    unittest.main()

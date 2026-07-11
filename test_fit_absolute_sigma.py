import copy
import unittest

import numpy as np

from dipole_new import fitTrapIntensity, intensity_function_offset


def make_synthetic_trap():
    seconds = np.geomspace(5e-5, 0.2, 25)
    noise_shape = np.array(
        [0.2, -0.4, 0.7, -0.1, -0.6] * 5,
        dtype=float,
    )
    trap = {}
    for temperature, tau in [(145, 0.020), (150, 0.012), (155, 0.007), (160, 0.004)]:
        sigma = np.full(seconds.size, 20.0)
        intensities = intensity_function_offset(seconds, 0.8, tau, -150.0)
        intensities = intensities + 4.0 * noise_shape
        trap[temperature] = {
            "seconds": seconds.copy(),
            "intensities": intensities,
            "intensity_err": sigma,
            "poisson_err": sigma.copy(),
            "image_sigma": 20.0,
        }
    return {0: {(10, 10): trap}}


class AbsoluteSigmaTest(unittest.TestCase):
    def test_absolute_errors_are_not_rescaled_by_reduced_chi_square(self):
        relative_fit = make_synthetic_trap()
        absolute_fit = copy.deepcopy(relative_fit)

        fitTrapIntensity(
            relative_fit,
            fit_offset=True,
            errors_are_absolute=False,
        )
        fitTrapIntensity(
            absolute_fit,
            fit_offset=True,
            errors_are_absolute=True,
        )

        relative_result = relative_fit[0][(10, 10)][150]
        absolute_result = absolute_fit[0][(10, 10)][150]
        reduced_chi2 = absolute_result["fit_reduced_chi_squared"]

        self.assertAlmostEqual(
            relative_result["fit_tau"],
            absolute_result["fit_tau"],
            places=10,
        )
        self.assertAlmostEqual(
            relative_result["fit_tau_err"] / absolute_result["fit_tau_err"],
            np.sqrt(reduced_chi2),
            places=6,
        )
        self.assertFalse(relative_result["fit_errors_are_absolute"])
        self.assertTrue(absolute_result["fit_errors_are_absolute"])


if __name__ == "__main__":
    unittest.main()

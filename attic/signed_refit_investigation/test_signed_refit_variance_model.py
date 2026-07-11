import unittest

import numpy as np

from signed_refit_profile_fitter import SignalDependentProfileTauFitter, intensity_model
from signed_refit_variance_model import (
    N_PUMPS,
    candidate_covariance,
    excess_pair_shot_variance,
    pumping_variance,
    unit_transfer_probability,
)


class SignedRefitVarianceModelTest(unittest.TestCase):
    def setUp(self):
        self.seconds = np.geomspace(5e-5, 0.2, 25)
        index = np.arange(self.seconds.size)
        self.null_covariance = 35.0**2 * 0.25 ** np.abs(
            index[:, None] - index[None, :]
        )

    def test_binomial_transfer_variance(self):
        amplitude = 0.2
        tau = 0.01
        variance, raw, probability = pumping_variance(
            self.seconds,
            amplitude,
            tau,
        )
        expected_probability = amplitude * unit_transfer_probability(
            self.seconds, tau
        )
        np.testing.assert_allclose(raw, expected_probability)
        np.testing.assert_allclose(probability, expected_probability)
        np.testing.assert_allclose(
            variance,
            N_PUMPS * expected_probability * (1.0 - expected_probability),
        )

    def test_excess_pair_shot_term(self):
        candidate = np.asarray([1200.0, 800.0, 500.0])
        reference = np.asarray([800.0, 900.0, 500.0])
        np.testing.assert_allclose(
            excess_pair_shot_variance(candidate, reference),
            np.asarray([100.0, 0.0, 0.0]),
        )

    def test_candidate_covariance_adds_only_diagonal_signal_terms(self):
        covariance, metadata = candidate_covariance(
            self.null_covariance,
            self.seconds,
            0.2,
            0.01,
            null_scale=1.05,
            extra_pair_shot=25.0,
        )
        difference = covariance - 1.05 * self.null_covariance
        np.testing.assert_allclose(
            difference - np.diag(np.diag(difference)),
            0.0,
            atol=1e-10,
        )
        self.assertTrue(np.all(np.diag(difference) > 25.0))
        self.assertFalse(metadata["probability_clipped"])

    def test_signal_dependent_fit_recovers_noiseless_curve(self):
        template = 5.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, 25))
        observed = intensity_model(
            self.seconds,
            -0.18,
            0.012,
            140.0,
            template,
        )
        fitter = SignalDependentProfileTauFitter(
            self.seconds,
            self.null_covariance,
            null_template=template,
            null_scale=1.03,
            extra_pair_shot=20.0,
        )
        result = fitter.fit(observed)
        self.assertAlmostEqual(result["tau"], 0.012, places=7)
        self.assertAlmostEqual(result["amplitude"], -0.18, places=7)
        self.assertTrue(result["variance_converged"])
        self.assertTrue(np.all(result["pumping_variance"] >= 0))
        self.assertEqual(
            result["effective_covariance"].shape,
            self.null_covariance.shape,
        )


if __name__ == "__main__":
    unittest.main()

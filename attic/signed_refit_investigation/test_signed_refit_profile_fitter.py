import unittest
from pathlib import Path

import numpy as np

from signed_refit_profile_fitter import (
    PROFILE_FITTER_VERSION,
    ProfileTauFitter,
    detector_region,
    intensity_model,
    load_scan_calibration,
    profile_mode_indices,
)
from signed_refit_profile_fitter_validation import validate_output


class SignedRefitProfileFitterTest(unittest.TestCase):
    def setUp(self):
        self.seconds = np.geomspace(5e-5, 0.2, 25)
        index = np.arange(self.seconds.size)
        correlation = 0.35 ** np.abs(index[:, None] - index[None, :])
        self.covariance = 35.0**2 * correlation
        self.template = 8.0 * np.sin(np.linspace(0, 2 * np.pi, self.seconds.size))
        self.fitter = ProfileTauFitter(
            self.seconds,
            self.covariance,
            null_template=self.template,
            tau_bounds=(5e-6, 2.0),
        )

    def test_noiseless_signed_fit_recovers_parameters(self):
        for amplitude in (0.22, -0.22):
            intensities = intensity_model(
                self.seconds,
                amplitude,
                0.012,
                -375.0,
                self.template,
            )
            result = self.fitter.fit(intensities)
            self.assertEqual(result["version"], PROFILE_FITTER_VERSION)
            self.assertAlmostEqual(result["tau"], 0.012, places=8)
            self.assertAlmostEqual(result["amplitude"], amplitude, places=8)
            self.assertAlmostEqual(result["offset"], -375.0, places=7)
            self.assertEqual(result["amplitude_sign"], int(np.sign(amplitude)))
            self.assertFalse(result["boundary_limited"])
            self.assertFalse(result["multimodal"])
            self.assertFalse(result["initial_guess_used"])

    def test_complete_profile_and_asymmetric_interval(self):
        rng = np.random.default_rng(17)
        truth = intensity_model(
            self.seconds,
            0.16,
            0.018,
            125.0,
            self.template,
        )
        observed = truth + rng.multivariate_normal(
            np.zeros(self.seconds.size), self.covariance
        )
        result = self.fitter.fit(observed)
        self.assertEqual(result["tau_grid"].shape, result["chi2_profile"].shape)
        self.assertEqual(
            result["tau_grid"].shape,
            result["amplitude_profile"].shape,
        )
        self.assertIsNotNone(result["tau_interval_lower"])
        self.assertIsNotNone(result["tau_interval_upper"])
        self.assertLess(result["tau_interval_lower"], result["tau"])
        self.assertGreater(result["tau_interval_upper"], result["tau"])
        self.assertNotAlmostEqual(
            result["tau_error_lower"],
            result["tau_error_upper"],
            places=6,
        )

    def test_batch_statistic_matches_saved_profile_grid(self):
        curves = np.asarray(
            [
                intensity_model(
                    self.seconds,
                    amplitude,
                    tau,
                    offset,
                    self.template,
                )
                for amplitude, tau, offset in (
                    (0.15, 0.004, -100.0),
                    (-0.12, 0.03, 250.0),
                )
            ]
        )
        batch = self.fitter.batch_profile_statistic(curves)
        for index, curve in enumerate(curves):
            result = self.fitter.fit(curve)
            grid_best = int(np.argmin(result["chi2_profile"]))
            grid_delta = result["constant_chi2"] - result["chi2_profile"][grid_best]
            self.assertAlmostEqual(batch["delta_chi2"][index], grid_delta, places=9)
            self.assertAlmostEqual(
                batch["tau"][index],
                result["tau_grid"][grid_best],
                places=12,
            )
            self.assertEqual(batch["amplitude"][index] > 0, result["amplitude"] > 0)

    def test_boundary_case_is_not_given_two_sided_error(self):
        intensities = intensity_model(
            self.seconds,
            0.20,
            20.0,
            -100.0,
            self.template,
        )
        result = self.fitter.fit(intensities)
        self.assertTrue(result["boundary_limited"])
        self.assertTrue(result["at_upper_boundary"])
        self.assertTrue(result["interval_upper_limited"])
        self.assertIsNone(result["tau_error_upper"])

    def test_low_signal_multimodal_case_is_flagged(self):
        observed = self.template + np.random.default_rng(5).normal(
            0.0, 35.0, self.seconds.size
        )
        result = self.fitter.fit(observed)
        self.assertTrue(result["multimodal"])
        self.assertGreaterEqual(len(result["competitive_modes"]), 2)

    def test_profile_mode_helper_includes_boundaries(self):
        profile = np.asarray([0.0, 2.0, 1.0, 3.0, 0.5])
        np.testing.assert_array_equal(
            profile_mode_indices(profile),
            np.asarray([0, 2, 4]),
        )

    def test_noise_model_mapping_and_provenance(self):
        root = Path(__file__).resolve().parent
        calibration = load_scan_calibration(
            root / "signed_refit_noise_model.h5",
            183,
            2,
            row=300,
            col=1200,
        )
        self.assertEqual(calibration.region, detector_region(300, 1200))
        self.assertEqual(calibration.seconds.size, 25)
        self.assertEqual(calibration.covariance.shape, (25, 25))
        self.assertEqual(calibration.null_template.shape, (25,))
        self.assertEqual(len(calibration.noise_model_sha256), 64)

    def test_validation_artifact_passes(self):
        root = Path(__file__).resolve().parent
        metrics = validate_output(
            root / "signed_refit_profile_fitter_validation.npz",
            root / "signed_refit_noise_model.h5",
        )
        self.assertEqual(metrics["acceptance"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

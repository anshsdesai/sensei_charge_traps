import unittest
from pathlib import Path

from signed_refit_noise_model import (
    NOISE_MODEL_VERSION,
    validate_noise_model,
)


class SignedRefitNoiseModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.root = root
        cls.result = validate_noise_model(
            root / "signed_refit_noise_model.h5",
            root / "signed_refit_manifest.csv",
            root / "signed_refit_control_pairs.npz",
        )

    def test_covariance_count(self):
        self.assertEqual(self.result["covariance_count"], 23 * 4 * 32)

    def test_positive_definite_and_conditioned(self):
        self.assertGreater(self.result["minimum_eigenvalue"], 0)
        self.assertLess(self.result["condition_max"], 1e8)

    def test_version_constant(self):
        self.assertEqual(NOISE_MODEL_VERSION, "signed-refit-noise-v2")

    def test_null_template_has_negligible_pump_projection(self):
        projection = self.result["template_projection"]
        self.assertEqual(projection["count"], 23 * 4 * 32)
        self.assertLess(projection["delta_chi2_max"], 1.0)
        self.assertLess(projection["abs_z_max"], 1.0)


if __name__ == "__main__":
    unittest.main()

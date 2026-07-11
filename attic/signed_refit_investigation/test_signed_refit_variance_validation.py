import unittest
from pathlib import Path

from signed_refit_variance_validation import validate_output


class SignedRefitVarianceValidationTest(unittest.TestCase):
    def test_validation_artifact_passes(self):
        root = Path(__file__).resolve().parent
        metrics = validate_output(
            root / "signed_refit_variance_validation.npz",
            root / "signed_refit_noise_model.h5",
        )
        self.assertEqual(metrics["acceptance"]["status"], "PASS")
        self.assertEqual(metrics["template_projection"]["count"], 23 * 4 * 32)


if __name__ == "__main__":
    unittest.main()

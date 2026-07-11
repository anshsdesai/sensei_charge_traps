import unittest
from pathlib import Path

from signed_refit_noise_closure import validate_closure


class SignedRefitNoiseClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.metrics = validate_closure(
            root / "signed_refit_noise_closure.npz",
            root / "signed_refit_noise_model.h5",
        )

    def test_held_out_count(self):
        self.assertEqual(self.metrics["curve_count"], 23 * 4 * 32 * 128)

    def test_acceptance(self):
        self.assertEqual(self.metrics["acceptance"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

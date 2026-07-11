import unittest
from pathlib import Path

from signed_refit_controls import (
    CANDIDATE_RADIUS,
    CONTROLS_PER_REGION,
    CONTROL_VERSION,
    COL_REGIONS,
    ROW_REGIONS,
    TRAIN_PER_REGION,
    VALIDATION_PER_REGION,
    validate_control_file,
)


class SignedRefitControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        cls.result = validate_control_file(
            root / "signed_refit_control_pairs.npz",
            root / "signed_refit_manifest.csv",
        )

    def test_counts(self):
        expected = 4 * ROW_REGIONS * COL_REGIONS * CONTROLS_PER_REGION
        self.assertEqual(self.result["control_count"], expected)
        self.assertEqual(
            self.result["train_count"],
            4 * ROW_REGIONS * COL_REGIONS * TRAIN_PER_REGION,
        )
        self.assertEqual(
            self.result["validation_count"],
            4 * ROW_REGIONS * COL_REGIONS * VALIDATION_PER_REGION,
        )

    def test_version(self):
        self.assertEqual(self.result["metadata"]["version"], CONTROL_VERSION)

    def test_candidate_clearance(self):
        self.assertGreater(
            self.result["minimum_candidate_distance"],
            CANDIDATE_RADIUS,
        )


if __name__ == "__main__":
    unittest.main()

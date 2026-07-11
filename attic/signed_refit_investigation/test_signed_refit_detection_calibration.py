import unittest
from pathlib import Path

import numpy as np

from signed_refit_detection_calibration import (
    TARGET_PER_FIT_FPR,
    empirical_p_value,
    empirical_threshold,
    passes_detection_threshold,
    validate_calibration,
)


class SignedRefitDetectionCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.path = cls.root / "signed_refit_detection_calibration.npz"
        cls.metrics = validate_calibration(
            cls.path,
            cls.root / "signed_refit_noise_model.h5",
            cls.root / "signed_refit_control_pairs.npz",
        )
        cls.data = np.load(cls.path, allow_pickle=False)

    def test_acceptance_and_reference_shape(self):
        self.assertEqual(self.metrics["acceptance"]["status"], "PASS")
        self.assertEqual(self.data["reference_statistics"].shape, (23, 8192))
        self.assertEqual(self.data["thresholds"].shape, (23,))
        self.assertEqual(
            self.data["ordinary_empirical_pvalue"].shape,
            self.data["ordinary_statistics"].shape,
        )

    def test_finite_sample_threshold_meets_stated_pvalue(self):
        for temperature, threshold in zip(
            self.data["temperatures"],
            self.data["thresholds"],
        ):
            pvalue = float(
                empirical_p_value(threshold, int(temperature), self.path)
            )
            self.assertLessEqual(pvalue, TARGET_PER_FIT_FPR)
            self.assertTrue(
                bool(
                    passes_detection_threshold(
                        threshold,
                        int(temperature),
                        self.path,
                    )
                )
            )

    def test_threshold_is_lowest_attainable_observed_value(self):
        reference = np.arange(8192, dtype=float)
        threshold = empirical_threshold(reference, TARGET_PER_FIT_FPR)
        self.assertEqual(threshold, 8185.0)
        pvalue = (np.count_nonzero(reference >= threshold) + 1) / 8193
        lower_pvalue = (
            np.count_nonzero(reference >= (threshold - 1)) + 1
        ) / 8193
        self.assertLessEqual(pvalue, TARGET_PER_FIT_FPR)
        self.assertGreater(lower_pvalue, TARGET_PER_FIT_FPR)

    def test_calibration_and_evaluation_sites_are_balanced(self):
        split = self.data["ordinary_calibration_split"]
        self.assertEqual(np.count_nonzero(split == 0), 8192)
        self.assertEqual(np.count_nonzero(split == 1), 8192)
        coordinates = np.column_stack(
            (
                self.data["ordinary_quadrant"],
                self.data["ordinary_row"],
                self.data["ordinary_col"],
            )
        )
        self.assertEqual(np.unique(coordinates, axis=0).shape[0], coordinates.shape[0])

    def test_structured_controls_are_present(self):
        self.assertGreater(self.data["horizontal_statistics"].shape[0], 0)
        self.assertGreater(self.data["near_defect_statistics"].shape[0], 10000)
        self.assertEqual(self.data["horizontal_statistics"].shape[1], 23)
        self.assertEqual(self.data["near_defect_statistics"].shape[1], 23)
        self.assertEqual(
            self.data["horizontal_empirical_pvalue"].shape,
            self.data["horizontal_statistics"].shape,
        )
        self.assertEqual(
            self.data["near_defect_empirical_pvalue"].shape,
            self.data["near_defect_statistics"].shape,
        )


if __name__ == "__main__":
    unittest.main()

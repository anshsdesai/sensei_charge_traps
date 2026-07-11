import unittest

import numpy as np

from signed_refit_orientation import (
    FROZEN_POLICY,
    LABEL_AMBIGUOUS,
    LABEL_DUAL,
    LABEL_INSUFFICIENT,
    LABEL_NO_SIGNAL,
    LABEL_SINGLE_NEGATIVE,
    LABEL_SINGLE_POSITIVE,
    LABEL_STRUCTURED,
    classify_orientations,
)


class SignedRefitOrientationTests(unittest.TestCase):
    def test_classifications_are_exhaustive(self):
        signs = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [-1, -1, -1, -1, 0],
                [1, 1, 1, -1, 0],
                [1, 1, -1, -1, 0],
            ],
            dtype=np.int8,
        )
        accepted = signs != 0
        result = classify_orientations(signs, accepted)
        self.assertEqual(
            result["label"].tolist(),
            [
                LABEL_NO_SIGNAL,
                LABEL_INSUFFICIENT,
                LABEL_SINGLE_POSITIVE,
                LABEL_SINGLE_NEGATIVE,
                LABEL_AMBIGUOUS,
                LABEL_DUAL,
            ],
        )

    def test_insignificant_opposite_sign_is_ignored(self):
        signs = np.asarray([[1, 1, 1, 1, -1]], dtype=np.int8)
        accepted = np.asarray([[True, True, True, True, False]])
        result = classify_orientations(signs, accepted)
        self.assertEqual(result["label"][0], LABEL_SINGLE_POSITIVE)
        self.assertTrue(result["single_trap_eligible"][0])

    def test_any_significant_conflict_is_not_single_trap(self):
        signs = np.asarray([[1, 1, 1, 1, -1]], dtype=np.int8)
        accepted = np.ones_like(signs, dtype=bool)
        result = classify_orientations(signs, accepted)
        self.assertEqual(result["label"][0], LABEL_AMBIGUOUS)
        self.assertFalse(result["single_trap_eligible"][0])

    def test_significant_zero_is_invalid(self):
        with self.assertRaises(ValueError):
            classify_orientations(
                np.asarray([[1, 1, 1, 0]], dtype=np.int8),
                np.ones((1, 4), dtype=bool),
            )

    def test_frozen_policy_is_strict(self):
        FROZEN_POLICY.validate()
        self.assertEqual(FROZEN_POLICY.minimum_significant_temperatures, 4)
        self.assertTrue(FROZEN_POLICY.exclude_any_sign_conflict_from_single_trap)

    def test_structured_background_overrides_single_orientation(self):
        signs = np.asarray([[1, 1, 1, 1]], dtype=np.int8)
        accepted = np.ones_like(signs, dtype=bool)
        result = classify_orientations(
            signs,
            accepted,
            structured_background=np.asarray([True]),
        )
        self.assertEqual(result["label"][0], LABEL_STRUCTURED)
        self.assertFalse(result["single_trap_eligible"][0])


if __name__ == "__main__":
    unittest.main()

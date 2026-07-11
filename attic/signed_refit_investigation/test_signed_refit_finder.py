import json
import unittest
from pathlib import Path

import numpy as np

from signed_refit_finder import (
    FINDER_VERSION,
    PREDECLARED_CONFIGS,
    FinderConfig,
    evaluate_lobes,
    finder_mask,
    load_frozen_config,
    relative_lobe_imbalance,
    robust_noise_sigma,
    write_frozen_config,
)
from signed_refit_finder_calibration import N_PUMPS, transfer_probability
from signed_refit_profile_fitter import pump_shape


class SignedRefitFinderTests(unittest.TestCase):
    def test_predeclared_configs_are_valid_and_unique(self):
        names = []
        for config in PREDECLARED_CONFIGS:
            config.validate()
            names.append(config.name)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 6)

    def test_product_rule_can_accept_a_subthreshold_lobe(self):
        config = PREDECLARED_CONFIGS[1]
        accepted = evaluate_lobes(
            np.asarray([10.0]),
            np.asarray([-1.0]),
            1.0,
            config,
        )
        self.assertTrue(bool(accepted[0]))

        separate = PREDECLARED_CONFIGS[2]
        accepted = evaluate_lobes(
            np.asarray([10.0]),
            np.asarray([-1.0]),
            1.0,
            separate,
        )
        self.assertFalse(bool(accepted[0]))

    def test_balance_is_lobe_magnitude_mismatch(self):
        values = relative_lobe_imbalance(
            np.asarray([6.0, 6.0]),
            np.asarray([-6.0, -2.0]),
        )
        np.testing.assert_allclose(values, [0.0, 2.0 / 3.0])
        balanced = PREDECLARED_CONFIGS[3]
        accepted = evaluate_lobes(
            np.asarray([6.0, 6.0]),
            np.asarray([-6.0, -2.0]),
            1.0,
            balanced,
        )
        np.testing.assert_array_equal(accepted, [True, False])

    def test_trail_isolation_rejects_extra_significant_pixels(self):
        isolated = PREDECLARED_CONFIGS[5]
        accepted = evaluate_lobes(
            np.asarray([6.0, 6.0]),
            np.asarray([-6.0, -6.0]),
            1.0,
            isolated,
            trail_counts=np.asarray([2, 3]),
        )
        np.testing.assert_array_equal(accepted, [True, False])

    def test_finder_mask_uses_lower_row_coordinate(self):
        residual = np.zeros((5, 3), dtype=float)
        residual[2, 1] = 8.0
        residual[1, 1] = -8.0
        config = FinderConfig(
            name="test",
            noise_estimator="robust",
            lobe_rule="separate",
            sigma_threshold=3.0,
            max_relative_lobe_imbalance=None,
            persistence=2,
        )
        mask = finder_mask(residual, 2.0, config)
        self.assertTrue(mask[1, 1])
        self.assertEqual(int(np.count_nonzero(mask)), 1)

    def test_robust_sigma_is_stable_to_one_outlier(self):
        rng = np.random.default_rng(7)
        values = rng.normal(0.0, 4.0, size=(200, 200))
        baseline = robust_noise_sigma(values)
        values[0, 0] = 1e6
        contaminated = robust_noise_sigma(values)
        self.assertLess(abs(contaminated - baseline) / baseline, 0.01)

    def test_frozen_config_round_trip(self):
        path = Path("test_signed_refit_finder_config.tmp.json")
        try:
            write_frozen_config(
                PREDECLARED_CONFIGS[3],
                path,
                metadata={"test": True},
            )
            values = json.loads(path.read_text(encoding="ascii"))
            self.assertEqual(values["finder_version"], FINDER_VERSION)
            self.assertEqual(load_frozen_config(path), PREDECLARED_CONFIGS[3])
        finally:
            path.unlink(missing_ok=True)

    def test_injection_probability_does_not_double_count_pumps(self):
        probability = transfer_probability(0.1, 0.3, 0.03)
        expected = 0.03 * float(pump_shape(0.1, 0.3)) / N_PUMPS
        self.assertAlmostEqual(probability, expected)
        self.assertLess(probability, 0.03)


if __name__ == "__main__":
    unittest.main()

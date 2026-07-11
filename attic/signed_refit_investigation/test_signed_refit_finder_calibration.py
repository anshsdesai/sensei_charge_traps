import json
import unittest

import numpy as np

from signed_refit_finder_calibration import (
    CHARACTERIZABLE_PEAK_SNR,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    MAX_ORDINARY_FPR,
    MAX_STRUCTURED_FPR,
    validate_outputs,
)


class SignedRefitFinderCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(DEFAULT_OUTPUT, allow_pickle=False)
        cls.selected = int(cls.data["selected_index"])

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_artifact_validates(self):
        result = validate_outputs(DEFAULT_OUTPUT, DEFAULT_REPORT, DEFAULT_CONFIG)
        self.assertTrue(result["acceptance_pass"])

    def test_selected_rule_requires_both_lobes(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="ascii"))
        selected = config["selected_config"]
        self.assertEqual(selected["lobe_rule"], "separate")
        self.assertEqual(selected["persistence"], 2)

    def test_selection_did_not_use_candidate_count(self):
        metadata = json.loads(str(self.data["metadata_json"]))
        self.assertFalse(
            metadata["selection_rule"]["actual_candidate_count_used"]
        )

    def test_strong_injection_definition_is_stored(self):
        mask = np.asarray(self.data["characterizable_mask"], dtype=bool)
        peak_snr = np.asarray(self.data["injection_peak_snr"], dtype=float)
        np.testing.assert_array_equal(
            mask,
            peak_snr >= CHARACTERIZABLE_PEAK_SNR,
        )
        self.assertTrue(np.all(np.any(mask, axis=(1, 2))))

    def test_selected_configuration_passes_completeness_gate(self):
        by_temperature = self.data[
            "characterizable_completeness_by_temperature"
        ][self.selected]
        self.assertGreaterEqual(float(np.min(by_temperature)), 0.50)
        self.assertGreater(float(self.data["characterizable_completeness"][self.selected]), 0.90)

    def test_selected_configuration_passes_null_gates(self):
        rates = self.data["null_end_to_end_rates"][self.selected]
        self.assertLessEqual(float(rates[0]), MAX_ORDINARY_FPR)
        self.assertTrue(np.all(rates[1:] <= MAX_STRUCTURED_FPR))

    def test_horizontal_axis_negative_control_is_present(self):
        names = self.data["class_names"].tolist()
        self.assertIn("horizontal_axis", names)
        horizontal = names.index("horizontal_axis")
        self.assertGreater(
            int(np.sum(self.data["null_trials"][self.selected, horizontal])),
            0,
        )

    def test_finite_sample_upper_bounds_cover_observed_rates(self):
        rates = self.data["null_end_to_end_rates"][self.selected]
        upper = self.data["null_end_to_end_upper_95"][self.selected]
        self.assertTrue(np.all(upper >= rates))


if __name__ == "__main__":
    unittest.main()

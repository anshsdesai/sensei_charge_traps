import json
import unittest

import numpy as np

from signed_refit_orientation import (
    LABEL_AMBIGUOUS,
    LABEL_DUAL,
    LABEL_SINGLE_NEGATIVE,
    LABEL_SINGLE_POSITIVE,
    LABEL_STRUCTURED,
    load_policy,
)
from signed_refit_orientation_validation import (
    CONTROL_CLASSES,
    DEFAULT_COORDS,
    DEFAULT_OUTPUT,
    DEFAULT_POLICY,
    DEFAULT_REPORT,
    MAX_END_TO_END_NULL_SINGLE_ORIENTATION_RATE,
    MIN_INJECTION_ORIENTATION_EFFICIENCY,
    MIN_INJECTION_SIGN_ACCURACY,
    validate_outputs,
)


class SignedRefitOrientationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(DEFAULT_OUTPUT, allow_pickle=False)

    @classmethod
    def tearDownClass(cls):
        cls.data.close()

    def test_artifact_validates_and_policy_loads(self):
        result = validate_outputs(
            DEFAULT_OUTPUT,
            DEFAULT_POLICY,
            DEFAULT_REPORT,
            DEFAULT_COORDS,
        )
        self.assertTrue(result["acceptance_pass"])
        load_policy(DEFAULT_POLICY).validate()

    def test_artifact_has_expected_provenance(self):
        metadata = json.loads(str(self.data["metadata_json"]))
        self.assertTrue(metadata["acceptance_pass"])
        self.assertEqual(metadata["candidate_fit_role"].split(";")[0], "policy calibration only")
        self.assertEqual(self.data["candidate_row"].size, 8241)

    def test_single_trap_labels_have_no_conflicting_sign(self):
        labels = self.data["candidate_orientation_label"]
        single = self.data["candidate_single_trap_eligible"].astype(bool)
        self.assertTrue(
            np.all(
                np.isin(
                    labels[single],
                    (LABEL_SINGLE_POSITIVE, LABEL_SINGLE_NEGATIVE),
                )
            )
        )
        self.assertTrue(
            np.all(
                (self.data["candidate_positive_count"][single] == 0)
                | (self.data["candidate_negative_count"][single] == 0)
            )
        )
        self.assertFalse(
            np.any(
                single
                & np.isin(labels, (LABEL_AMBIGUOUS, LABEL_DUAL))
            )
        )

    def test_structured_overlap_is_never_single_trap(self):
        structured = self.data["candidate_structured_background"].astype(bool)
        labels = self.data["candidate_orientation_label"]
        single = self.data["candidate_single_trap_eligible"].astype(bool)
        self.assertEqual(int(np.count_nonzero(structured)), 2)
        self.assertTrue(np.all(labels[structured] == LABEL_STRUCTURED))
        self.assertFalse(np.any(single[structured]))
        self.assertEqual(
            int(np.sum(self.data["injection_structured_background"])),
            0,
        )

    def test_injection_efficiency_and_sign_accuracy_pass(self):
        eligible = self.data["injection_orientation_eligible"].astype(bool)
        correct = self.data["injection_correct_single"].astype(bool)
        efficiency = float(np.mean(correct[eligible]))
        self.assertGreaterEqual(
            efficiency,
            MIN_INJECTION_ORIENTATION_EFFICIENCY,
        )
        active = self.data["injection_active"].astype(bool)
        accepted = self.data["injection_significant"].astype(bool)
        selected = active & accepted
        truth = np.broadcast_to(
            self.data["injection_true_sign"][:, None],
            selected.shape,
        )
        accuracy = float(
            np.mean(self.data["injection_sign"][selected] == truth[selected])
        )
        self.assertGreaterEqual(accuracy, MIN_INJECTION_SIGN_ACCURACY)

    def test_production_control_chain_has_no_survivors(self):
        for name in CONTROL_CLASSES[:3]:
            finder = self.data[f"{name}_finder_selected"].astype(bool)
            single = self.data[f"{name}_single_trap_eligible"].astype(bool)
            rate = float(np.mean(finder & single))
            self.assertLessEqual(
                rate,
                MAX_END_TO_END_NULL_SINGLE_ORIENTATION_RATE,
            )

    def test_horizontal_axis_raw_stress_result_is_retained(self):
        raw = self.data["horizontal_axis_raw_orientation_label"]
        raw_single = np.isin(
            raw,
            (LABEL_SINGLE_POSITIVE, LABEL_SINGLE_NEGATIVE),
        )
        self.assertEqual(int(np.count_nonzero(raw_single)), 38)
        structured = self.data[
            "horizontal_axis_structured_background"
        ].astype(bool)
        self.assertTrue(np.all(structured))


if __name__ == "__main__":
    unittest.main()

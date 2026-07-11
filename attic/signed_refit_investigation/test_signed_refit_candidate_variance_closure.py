import unittest
from pathlib import Path

import numpy as np
import json


class SignedRefitCandidateVarianceClosureTest(unittest.TestCase):
    def test_crossfit_closure_passes(self):
        root = Path(__file__).resolve().parent
        metrics = json.loads(
            str(
                np.load(
                    root / "signed_refit_candidate_variance_closure_v2.npz",
                    allow_pickle=False,
                )["metadata_json"]
            )
        )
        self.assertEqual(metrics["acceptance"]["status"], "PASS")
        self.assertFalse(metrics["calibration"]["hit_upper_bound"])
        self.assertEqual(
            len(metrics["calibration"]["crossfit_folds"]),
            2,
        )
        self.assertLess(
            metrics["evaluation"]["width_spread"],
            0.12,
        )
        self.assertEqual(
            metrics["lobe_order_contract"],
            "I=(image[row,col]-image[row-1,col])/2",
        )


if __name__ == "__main__":
    unittest.main()

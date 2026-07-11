import unittest

import numpy as np

from signed_refit_intensity_pipeline import (
    REJECTION_CODE,
    _factor_for_amplitude,
    _quality_code,
)


class SignedRefitIntensityPipelineTest(unittest.TestCase):
    def test_amplitude_bin_factor_is_frozen_from_null_fit(self):
        calibration = {
            "edges": np.asarray([0.25, 0.50, 0.75]),
            "factors": np.asarray([8.0, 16.0, 30.0, 23.0]),
        }
        self.assertEqual(_factor_for_amplitude(0.10, calibration), 8.0)
        self.assertEqual(_factor_for_amplitude(-0.40, calibration), 16.0)
        self.assertEqual(_factor_for_amplitude(0.60, calibration), 30.0)
        self.assertEqual(_factor_for_amplitude(-0.90, calibration), 23.0)

    def test_quality_gate_requires_characterizable_profile(self):
        result = {
            "variance_converged": True,
            "amplitude": 0.5,
            "tau_interval_lower": 1.0,
            "tau_interval_upper": 2.0,
            "boundary_limited": False,
            "multimodal": False,
        }
        self.assertEqual(_quality_code(result), REJECTION_CODE["accepted"])
        result["multimodal"] = True
        self.assertEqual(
            _quality_code(result),
            REJECTION_CODE["multimodal_profile"],
        )


if __name__ == "__main__":
    unittest.main()

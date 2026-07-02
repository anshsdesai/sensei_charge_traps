import unittest

import numpy as np

from ccd_simulation import drain_traps_empty_numba, fast_clear_numba, seed_numba


TRAP_ROWS = np.array([1], dtype=np.int64)
TRAP_COLS = np.array([0], dtype=np.int64)
ZERO_EMIT = np.array([0.0], dtype=np.float64)


def run_one_clear(q, capture_alpha, emit_prob=0.0, initially_trapped=False,
                  is_v3=True):
    image = np.zeros((2, 1), dtype=np.float64)
    image[0, 0] = q
    trapped = np.array([1.0 if initially_trapped else 0.0], dtype=np.float64)
    emit = np.array([emit_prob], dtype=np.float64)
    alpha = np.array([capture_alpha], dtype=np.float64)
    phase = np.array([1 if is_v3 else 0], dtype=np.uint8)
    fast_clear_numba(
        image,
        1,
        1.0,
        0,
        1.0,
        TRAP_ROWS,
        TRAP_COLS,
        emit,
        emit,
        alpha,
        phase,
        trapped,
    )
    return image, trapped


class PhaseLimitedTransportTest(unittest.TestCase):
    def test_empty_trap_capture_frequency_matches_phase_limited_probability(self):
        seed_numba(12345)
        n_trials = 8000
        q = 1.0
        target_p = 0.30
        alpha = -np.log1p(-target_p) / q
        captures = 0
        for _ in range(n_trials):
            _, trapped = run_one_clear(q, alpha)
            captures += int(trapped[0] > 0.0)
        observed = captures / n_trials
        self.assertAlmostEqual(observed, target_p, delta=0.025)

    def test_large_packet_capture_frequency_uses_q_scaled_probability(self):
        seed_numba(23456)
        n_trials = 8000
        q = 2000.0
        target_p = 0.70
        alpha = -np.log1p(-target_p) / q
        captures = 0
        for _ in range(n_trials):
            _, trapped = run_one_clear(q, alpha)
            captures += int(trapped[0] > 0.0)
        observed = captures / n_trials
        self.assertAlmostEqual(observed, target_p, delta=0.025)

    def test_occupied_trap_emission_frequency_matches_full_dwell_probability(self):
        seed_numba(34567)
        n_trials = 8000
        emit_prob = 0.40
        releases = 0
        for _ in range(n_trials):
            _, trapped = run_one_clear(0.0, 0.0, emit_prob=emit_prob, initially_trapped=True)
            releases += int(trapped[0] == 0.0)
        observed = releases / n_trials
        self.assertAlmostEqual(observed, emit_prob, delta=0.025)

    def test_zero_phase_window_cannot_capture_but_can_emit(self):
        seed_numba(45678)
        for _ in range(100):
            _, trapped = run_one_clear(2000.0, 0.0)
            self.assertEqual(trapped[0], 0.0)

        image, trapped = run_one_clear(0.0, 0.0, emit_prob=1.0, initially_trapped=True)
        self.assertEqual(trapped[0], 0.0)
        self.assertEqual(image.sum(), 1.0)

    def test_charge_conservation_inside_clear_transport_boundary(self):
        image, trapped = run_one_clear(3.0, 100.0)
        self.assertEqual(image.sum() + trapped.sum(), 3.0)

        image, trapped = run_one_clear(0.0, 0.0, emit_prob=1.0, initially_trapped=True)
        self.assertEqual(image.sum() + trapped.sum(), 1.0)

    def test_v3_emission_faces_same_step_recapture(self):
        seed_numba(56789)
        # alpha -> inf: a V3 trap's own emission is always recaptured on exit.
        for _ in range(100):
            image, trapped = run_one_clear(
                0.0, 50.0, emit_prob=1.0, initially_trapped=True, is_v3=True)
            self.assertEqual(trapped[0], 1.0)
            self.assertEqual(image.sum(), 0.0)

    def test_v1_emission_always_escapes(self):
        seed_numba(67890)
        # Same alpha -> inf, but a V1 trap's emitted carrier exits over V3
        # without recrossing the trap: it must always escape.
        for _ in range(100):
            image, trapped = run_one_clear(
                0.0, 50.0, emit_prob=1.0, initially_trapped=True, is_v3=False)
            self.assertEqual(trapped[0], 0.0)
            self.assertEqual(image.sum(), 1.0)

    def test_v1_capture_frequency_matches_phase_limited_probability(self):
        seed_numba(78901)
        n_trials = 8000
        q = 1.0
        target_p = 0.30
        alpha = -np.log1p(-target_p) / q
        captures = 0
        for _ in range(n_trials):
            _, trapped = run_one_clear(q, alpha, is_v3=False)
            captures += int(trapped[0] > 0.0)
        observed = captures / n_trials
        self.assertAlmostEqual(observed, target_p, delta=0.025)

    def test_drain_is_recapture_free_for_v1_and_thinned_for_v3(self):
        seed_numba(89012)
        n_trials = 8000
        tau = np.array([1.0], dtype=np.float64)
        alpha = np.array([np.log(2.0)], dtype=np.float64)  # exp(-alpha) = 1/2
        # total dwell = tau*ln2 -> V1 P_drain = 1/2, V3 P_drain = 1-2^{-1/2}
        dwell, n_shifts = np.log(2.0), 1
        for is_v3, expected in ((0, 0.5), (1, 1.0 - 2.0 ** -0.5)):
            phase = np.array([is_v3], dtype=np.uint8)
            drains = 0
            for _ in range(n_trials):
                trapped = np.array([1.0], dtype=np.float64)
                drain_traps_empty_numba(tau, alpha, phase, trapped, dwell, n_shifts)
                drains += int(trapped[0] == 0.0)
            self.assertAlmostEqual(drains / n_trials, expected, delta=0.02)

    def test_representative_phase_window_is_not_single_e_saturated_but_flat_fields_are(self):
        t_phase = 300.0 / 15.0e6
        kc = 1.0e3
        p_single = 1.0 - np.exp(-kc * t_phase)
        p_flat = 1.0 - np.exp(-2000.0 * kc * t_phase)
        self.assertLess(p_single, 0.05)
        self.assertGreater(p_flat, 0.999999)


if __name__ == "__main__":
    unittest.main()

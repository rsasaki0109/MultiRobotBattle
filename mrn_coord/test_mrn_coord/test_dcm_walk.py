"""Tests for DCM walking control (Englsberger et al., IEEE T-RO 2015).

Contracts: the backward recursion ``xi_ini = p + (xi_eos - p) e^{-wT}`` builds a
continuous DCM reference that ends at rest on the last foot and stays bounded
inside the feet span (while a free single-foot DCM diverges); the trailing CoM
walks the full stride and settles; the instantaneous DCM is exactly the capture
point of that CoM; and the tracking law drives the DCM error to zero at the
chosen gain ``k_xi`` (faster for larger gain), freezes it at ``k_xi = 0``, and
diverges at rate ``omega`` with the feedback term removed.
"""

import math
import unittest

from mrn_coord.mapf.capture_point import capture_point, omega0, simulate_lipm
from mrn_coord.mapf.dcm_walk import (
    plan_dcm_reference,
    track_dcm,
    vrp_command,
)

Z = 0.8
T = 0.7
FEET = [0.0, 0.3, 0.6, 0.9, 1.2, 1.2, 1.2]


class TestDCMReference(unittest.TestCase):
    def setUp(self):
        self.w = omega0(Z)
        self.plan = plan_dcm_reference(FEET, T, Z, dt=0.01)

    def test_backward_recursion_exact(self):
        decay = math.exp(-self.w * T)
        for i, p in enumerate(FEET):
            expect = p + (self.plan.xi_eos[i] - p) * decay
            self.assertAlmostEqual(self.plan.xi_ini[i], expect, places=12)

    def test_reference_continuous_and_terminal_rest(self):
        for i in range(len(FEET) - 1):
            self.assertAlmostEqual(self.plan.xi_eos[i], self.plan.xi_ini[i + 1],
                                   places=12)
        # ends at rest on the last foot
        self.assertAlmostEqual(self.plan.xi_eos[-1], FEET[-1], places=12)

    def test_dcm_and_com_bounded(self):
        # the planned DCM never leaves the feet span (divergence is caught)
        self.assertLess(self.plan.dcm_excursion(), 1e-6)
        self.assertTrue(self.plan.com_in_support_span(margin=1e-6))
        # ... unlike a free single-foot DCM, which blows up
        free = simulate_lipm(0.0, 0.5, FEET[0], Z, duration=T * len(FEET))
        self.assertGreater(max(abs(v) for v in free.xi), 100.0)

    def test_com_walks_and_settles(self):
        span = max(self.plan.com) - min(self.plan.com)
        self.assertGreaterEqual(span, 0.99 * (max(FEET) - min(FEET)))
        self.assertAlmostEqual(self.plan.com[-1], FEET[-1], places=3)

    def test_instantaneous_dcm_is_capture_point(self):
        i = len(self.plan.t) // 2
        v = (self.plan.xi[i] - self.plan.com[i]) * self.w
        self.assertAlmostEqual(capture_point(self.plan.com[i], v, Z),
                               self.plan.xi[i], places=9)

    def test_deterministic(self):
        again = plan_dcm_reference(FEET, T, Z, dt=0.01)
        self.assertEqual(again.xi, self.plan.xi)


class TestDCMTracking(unittest.TestCase):
    def setUp(self):
        self.w = omega0(Z)
        self.plan = plan_dcm_reference(FEET, T, Z, dt=0.01)
        self.xi0 = self.plan.xi[0] + 0.1

    def test_feedback_converges_at_chosen_rate(self):
        res = track_dcm(self.plan, self.xi0, k_xi=3.0)
        self.assertTrue(res.converged())
        self.assertAlmostEqual(res.decay_rate(), 3.0, delta=0.2)

    def test_higher_gain_converges_faster(self):
        slow = track_dcm(self.plan, self.xi0, k_xi=1.0).decay_rate()
        fast = track_dcm(self.plan, self.xi0, k_xi=3.0).decay_rate()
        self.assertGreater(fast, slow + 1.0)

    def test_zero_gain_freezes_error(self):
        # k_xi = 0 cancels the natural omega-divergence but does not pull back
        res = track_dcm(self.plan, self.xi0, k_xi=0.0)
        self.assertAlmostEqual(res.err[-1], 0.1, places=6)
        self.assertAlmostEqual(res.decay_rate(), 0.0, delta=1e-2)

    def test_open_loop_diverges_at_omega(self):
        res = track_dcm(self.plan, self.xi0, k_xi=3.0, feedback=False)
        self.assertGreater(res.err[-1], 100.0)
        # the error envelope grows as e^{+omega t}
        self.assertAlmostEqual(res.decay_rate(), -self.w, delta=0.05)

    def test_control_law_formula(self):
        # r_cmd = r_ref + (1 + k/omega)(xi - xi_ref)
        r = vrp_command(0.5, 0.3, 0.1, 3.0, self.w)
        self.assertAlmostEqual(r, 0.1 + (1.0 + 3.0 / self.w) * (0.5 - 0.3),
                               places=12)


if __name__ == "__main__":
    unittest.main()

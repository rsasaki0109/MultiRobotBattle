"""Tests for Capture Point push recovery (Pratt et al., Humanoids 2006).

Contracts: the instantaneous capture point is exactly ``x + v/omega0``; stepping
the foot there brings the CoM to rest (the push is captured) while stepping short
or long lets the LIPM diverge (the robot falls); a push beyond one step's reach
is not one-step capturable and the foot clamps to the longest step; and bigger
pushes need monotonically more steps to capture.
"""

import math
import unittest

from mrn_coord.mapf.capture_point import (
    capture_point,
    n_step_capture,
    omega0,
    recover_step,
    simulate_lipm,
)

Z = 0.8


class TestCapturePoint(unittest.TestCase):
    def test_omega_and_icp_formula(self):
        self.assertAlmostEqual(omega0(Z), math.sqrt(9.8 / Z), places=9)
        self.assertAlmostEqual(capture_point(0.0, 0.5, Z), 0.5 / omega0(Z),
                               places=12)
        # capture point shifts with both position and velocity
        self.assertAlmostEqual(capture_point(0.1, 0.0, Z), 0.1, places=12)

    def test_step_to_capture_point_captures(self):
        v = 0.5
        xi = capture_point(0.0, v, Z)
        traj = simulate_lipm(0.0, v, xi, Z)
        self.assertTrue(traj.captured())
        self.assertLess(traj.max_excursion(), 0.2)
        # CoM ends at rest over the foot
        self.assertAlmostEqual(traj.x[-1], xi, places=2)
        self.assertAlmostEqual(traj.x_dot[-1], 0.0, places=1)

    def test_short_and_long_steps_fall(self):
        v = 0.5
        xi = capture_point(0.0, v, Z)
        for foot in (0.6 * xi, 1.4 * xi):
            traj = simulate_lipm(0.0, v, foot, Z)
            self.assertFalse(traj.captured())
            self.assertGreater(traj.max_excursion(), 1.0)   # diverges (falls)

    def test_one_step_capturable_and_clamp(self):
        small = recover_step(0.0, 0.5, Z, max_step=0.4)
        self.assertTrue(small.one_step_capturable)
        self.assertTrue(small.trajectory.captured())
        big = recover_step(0.0, 2.0, Z, max_step=0.4)
        self.assertFalse(big.one_step_capturable)
        self.assertAlmostEqual(big.foot, 0.4, places=9)     # clamped to reach

    def test_n_step_capturability_monotone(self):
        ns = [n_step_capture(0.0, v, Z, max_step=0.4, step_time=0.3)[0]
              for v in (0.5, 1.5, 2.5)]
        self.assertEqual(ns[0], 1)                          # small push: 1 step
        self.assertTrue(ns[0] <= ns[1] <= ns[2])            # monotone margin
        # all eventually captured
        for v in (0.5, 1.5, 2.5):
            _, captured, _ = n_step_capture(0.0, v, Z, max_step=0.4,
                                            step_time=0.3)
            self.assertTrue(captured)


if __name__ == "__main__":
    unittest.main()

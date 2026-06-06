"""Tests for humanoid push recovery decision surfaces (Stephens 2007)."""

import math
import unittest

from mrn_coord.mapf.capture_point import capture_point as cp_capture_point
from mrn_coord.mapf.push_recovery import (
    StrategyParams,
    capture_point,
    classify,
    hip_recovery_boundary,
    simulate_ankle,
    simulate_hip,
)


def _params():
    return StrategyParams()


class TestDecisionSurfaces(unittest.TestCase):
    def test_ankle_surface_is_capture_point_in_foot(self):
        # eq. (4): the ankle (CoP-balancing) region is exactly ξ in [δ⁻, δ⁺]
        p = _params()
        for x in (-0.05, 0.0, 0.05):
            for j in range(-60, 61):
                v = j * 0.01
                xi = capture_point(x, v, p)
                in_foot = p.delta_back <= xi <= p.delta_front
                self.assertEqual(classify(x, v, p) == "ankle", in_foot)

    def test_strategies_nest(self):
        p = _params()
        self.assertGreater(p.delta_hip, 0.0)
        self.assertLess(p.delta_front, p.delta_front + p.delta_hip)
        self.assertLess(p.delta_front + p.delta_hip, p.max_step)

    def test_classify_covers_all_regions(self):
        p = _params()
        w = p.omega
        self.assertEqual(classify(0.0, 0.05 * w, p), "ankle")
        self.assertEqual(classify(0.0, 0.121 * w, p), "hip")
        self.assertEqual(classify(0.0, 0.30 * w, p), "step")
        self.assertEqual(classify(0.0, 0.80 * w, p), "fall")


class TestHipBoundary(unittest.TestCase):
    def test_closed_form_matches_simulation(self):
        # the headline: Δ_hip = (τ_max/mg)(1−e^{−ωT_max})² matches the exact
        # bang-bang LIPPF simulation to machine precision
        p = _params()
        sim = hip_recovery_boundary(p)
        self.assertAlmostEqual(sim, p.delta_front + p.delta_hip, places=6)

    def test_printed_eq15_form_is_a_typo(self):
        # the paper's printed (e^{ωT}−1)² does NOT match the simulation
        p = _params()
        sim = hip_recovery_boundary(p)
        printed = p.delta_front + p.cmp_shift * (math.exp(p.omega * p.t_max) - 1.0) ** 2
        self.assertGreater(abs(sim - printed), 1e-3)


class TestRecovery(unittest.TestCase):
    def test_ankle_recovers_in_foot_fails_beyond(self):
        p = _params()
        inside = simulate_ankle(0.0, 0.25, p)    # ξ ≈ 0.080
        beyond = simulate_ankle(0.0, 0.38, p)    # ξ ≈ 0.121
        self.assertTrue(inside.captured())
        self.assertFalse(beyond.captured())

    def test_hip_recovers_what_ankle_cannot(self):
        p = _params()
        ankle = simulate_ankle(0.0, 0.38, p)
        hip = simulate_hip(0.0, 0.38, p)
        self.assertFalse(ankle.captured())
        self.assertTrue(hip.captured())
        self.assertTrue(hip.theta_within_limit())

    def test_hip_fails_beyond_band(self):
        p = _params()
        beyond = simulate_hip(0.0, 0.55, p)      # ξ ≈ 0.176, step region
        self.assertFalse(beyond.captured())

    def test_flywheel_returns_to_rest(self):
        p = _params()
        hip = simulate_hip(0.0, 0.38, p)
        n_pulse = 2 * int(round(p.t_max / 0.002))
        rest = hip.theta[n_pulse:]
        self.assertLess(max(rest) - min(rest), 1e-9)  # θ constant ⇒ θ̇ = 0

    def test_step_defers_to_capture_point(self):
        p = _params()
        w = p.omega
        x, v = 0.0, 0.30 * w
        self.assertEqual(classify(x, v, p), "step")
        xi_pr = capture_point(x, v, p)
        xi_cp = cp_capture_point(x, v, p.z_com, g=p.g)
        self.assertAlmostEqual(xi_pr, xi_cp, places=9)

    def test_deterministic(self):
        p = _params()
        a = simulate_hip(0.0, 0.38, p)
        b = simulate_hip(0.0, 0.38, p)
        self.assertEqual(a.x, b.x)


if __name__ == "__main__":
    unittest.main()

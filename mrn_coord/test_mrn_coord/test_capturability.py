"""Tests for N-step capturability analysis (Koolen et al. 2012)."""

import math
import unittest

from mrn_coord.mapf.capture_point import capture_point as cp_capture_point
from mrn_coord.mapf.capturability import (
    MODELS,
    CaptureParams,
    capturability_margin,
    inf_step_region,
    n_step_region,
    simulate_greedy,
)
from mrn_coord.mapf.push_recovery import StrategyParams


class TestCaptureRegions(unittest.TestCase):
    def test_regions_nested_and_below_limit(self):
        p = CaptureParams(foot_half=0.08)
        for model in MODELS:
            regs = [n_step_region(p, n, model=model) for n in range(10)]
            self.assertTrue(all(regs[i] < regs[i + 1] for i in range(len(regs) - 1)))
            limit = inf_step_region(p, model=model)
            self.assertTrue(all(r < limit for r in regs))
            self.assertAlmostEqual(regs[-1], limit, places=3)

    def test_three_models_nest(self):
        # Koolen model 1 ⊂ 2 ⊂ 3: point ⊂ finite foot ⊂ reaction mass
        p = CaptureParams(foot_half=0.08, reaction_shift=0.03)
        for n in range(6):
            self.assertLessEqual(n_step_region(p, n, model="point"),
                                 n_step_region(p, n, model="foot"))
            self.assertLessEqual(n_step_region(p, n, model="foot"),
                                 n_step_region(p, n, model="reaction"))

    def test_point_foot_zero_step_is_a_point(self):
        p = CaptureParams(foot_half=0.08)
        self.assertEqual(n_step_region(p, 0, model="point"), 0.0)
        self.assertEqual(n_step_region(p, 0, model="foot"), p.foot_half)

    def test_closed_form_geometric_series(self):
        p = CaptureParams(foot_half=0.0)
        w, T, l = p.omega, p.step_time, p.l_max
        for n in range(1, 6):
            series = sum(l * math.exp(-w * T * k) for k in range(1, n + 1))
            self.assertAlmostEqual(n_step_region(p, n, model="point"), series, places=12)


class TestCapturabilityLimit(unittest.TestCase):
    def test_limit_is_finite_and_unrecoverable_beyond(self):
        p = CaptureParams(foot_half=0.0)
        limit = inf_step_region(p, model="point")
        self.assertTrue(math.isinf(capturability_margin(limit * 1.02, p)))
        self.assertFalse(simulate_greedy(limit * 1.02, p, max_steps=400).captured)
        # just inside the limit: still (eventually) capturable
        self.assertTrue(simulate_greedy(limit * 0.999, p, max_steps=400).captured)

    def test_limit_monotonic(self):
        infs_T = [inf_step_region(CaptureParams(step_time=t, foot_half=0.0))
                  for t in (0.25, 0.35, 0.5)]
        self.assertGreater(infs_T[0], infs_T[1])
        self.assertGreater(infs_T[1], infs_T[2])
        infs_L = [inf_step_region(CaptureParams(l_max=l, foot_half=0.0))
                  for l in (0.3, 0.5, 0.7)]
        self.assertLess(infs_L[0], infs_L[1])
        self.assertLess(infs_L[1], infs_L[2])


class TestMarginVsSimulation(unittest.TestCase):
    def test_margin_matches_greedy_at_region_midpoints(self):
        p = CaptureParams(foot_half=0.08, reaction_shift=0.03)
        for model in MODELS:
            for n in range(1, 7):
                lo = n_step_region(p, n - 1, model=model)
                hi = n_step_region(p, n, model=model)
                xi = 0.5 * (lo + hi)
                self.assertEqual(simulate_greedy(xi, p, model=model).margin(), n)
                self.assertEqual(capturability_margin(xi, p, model=model), n)

    def test_margin_matches_greedy_dense_sweep(self):
        p = CaptureParams(foot_half=0.08)
        for j in range(60):
            xi = j * 0.005
            self.assertEqual(capturability_margin(xi, p, model="foot"),
                             simulate_greedy(xi, p, model="foot").margin())

    def test_already_balanced_needs_zero_steps(self):
        p = CaptureParams(foot_half=0.08)
        self.assertEqual(capturability_margin(0.05, p, model="foot"), 0)
        self.assertEqual(simulate_greedy(0.05, p, model="foot").num_steps, 0)


class TestCrossModuleLinks(unittest.TestCase):
    def test_capture_point_formula_matches_module(self):
        p = CaptureParams()
        x, v = 0.0, 0.4
        self.assertAlmostEqual(x + v / p.omega,
                               cp_capture_point(x, v, p.z_com, g=p.g), places=12)

    def test_reaction_model_equals_push_recovery_hip(self):
        # Koolen model 3's in-place region == push_recovery ankle band + hip widening
        sp = StrategyParams()
        p = CaptureParams(z_com=sp.z_com, g=sp.g,
                          foot_half=sp.delta_front, reaction_shift=sp.delta_hip)
        self.assertAlmostEqual(n_step_region(p, 0, model="reaction"),
                               sp.delta_front + sp.delta_hip, places=12)

    def test_deterministic(self):
        p = CaptureParams(foot_half=0.08)
        a = simulate_greedy(0.20, p, model="foot")
        b = simulate_greedy(0.20, p, model="foot")
        self.assertEqual(a.feet, b.feet)


if __name__ == "__main__":
    unittest.main()

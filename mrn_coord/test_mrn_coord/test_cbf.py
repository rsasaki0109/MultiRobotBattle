"""Tests for Control Barrier Function safety certificates — Wang, Ames & Egerstedt (2017)."""

import math
import random
import unittest

from mrn_coord.mapf.cbf import (
    barrier_constraints,
    min_separation,
    nominal_control,
    safe_control,
    simulate,
)


R = 0.2


class TestBarrier(unittest.TestCase):
    def test_constraint_is_active_pairwise(self):
        cons = barrier_constraints([(0, 0), (0.5, 0)], R, gamma=1.0, margin=0.0)
        self.assertEqual(len(cons), 1)
        a, b = cons[0]
        # h = 0.25 - (0.4)^2 = 0.25 - 0.16 = 0.09
        self.assertAlmostEqual(b, 0.09, places=6)

    def test_nominal_clipped_to_vmax(self):
        u = nominal_control([(0, 0)], [(100, 0)], gain=1.0, v_max=1.0)
        self.assertAlmostEqual(math.hypot(u[0][0], u[0][1]), 1.0)

    def test_stopping_is_feasible_when_safe(self):
        # robots safely apart: zero control trivially satisfies every certificate
        cons = barrier_constraints([(0, 0), (4, 0)], R, gamma=1.0, margin=0.1)
        for a, b in cons:
            self.assertGreaterEqual(b, 0.0)        # gamma*h >= 0 -> u=0 feasible


class TestSafeControl(unittest.TestCase):
    def test_minimally_invasive_when_far(self):
        pos = [(0, 0), (10, 10)]
        un = nominal_control(pos, [(5, 0), (5, 10)], gain=1.5, v_max=1.0)
        us = safe_control(pos, un, R, gamma=1.0, margin=0.1)
        for i in range(2):
            self.assertAlmostEqual(us[i][0], un[i][0], places=9)
            self.assertAlmostEqual(us[i][1], un[i][1], places=9)

    def test_projection_respects_certificate(self):
        # two robots driving straight at each other: filter must satisfy the
        # barrier inequality a.u <= b
        pos = [(0, 0), (0.6, 0)]
        un = [(1.0, 0.0), (-1.0, 0.0)]
        us = safe_control(pos, un, R, gamma=1.0, margin=0.1)
        cons = barrier_constraints(pos, R, gamma=1.0, margin=0.1)
        flat = [c for uv in us for c in uv]
        for a, b in cons:
            self.assertLessEqual(sum(a[k] * flat[k] for k in range(len(flat))),
                                 b + 1e-6)


class TestSimulate(unittest.TestCase):
    def test_random_arrives_collision_free(self):
        rng = random.Random(200)
        pts = []
        while len(pts) < 8:
            p = (rng.uniform(0, 4), rng.uniform(0, 4))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= 3 * R for q in pts):
                pts.append(p)
        res = simulate(pts[:4], pts[4:], R, dt=0.05, goal_radius=0.12,
                       max_steps=900)
        self.assertTrue(res.arrived)
        self.assertGreaterEqual(min_separation(res.paths, R), -1e-6)

    def test_symmetric_crossing_safe(self):
        res = simulate([(0, 0), (4, 4), (4, 0), (0, 4)],
                       [(4, 4), (0, 0), (0, 4), (4, 0)], R, dt=0.05,
                       goal_radius=0.1, max_steps=800)
        self.assertGreaterEqual(min_separation(res.paths, R), -1e-6)

    def test_headon_safe_but_deadlocks(self):
        res = simulate([(0, 0), (4, 0)], [(4, 0), (0, 0)], R, dt=0.05,
                       goal_radius=0.1, max_steps=600)
        self.assertGreaterEqual(min_separation(res.paths, R), -1e-6)
        self.assertLess(res.num_arrived, 2)

    def test_deterministic(self):
        a = simulate([(0, 0), (4, 4)], [(4, 4), (0, 0)], R, dt=0.05,
                     max_steps=400)
        b = simulate([(0, 0), (4, 4)], [(4, 4), (0, 0)], R, dt=0.05,
                     max_steps=400)
        self.assertEqual(a.paths, b.paths)
        self.assertEqual(a.steps, b.steps)


if __name__ == "__main__":
    unittest.main()

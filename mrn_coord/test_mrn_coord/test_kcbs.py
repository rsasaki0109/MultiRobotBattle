"""Tests for Kinodynamic CBS (K-CBS) — Kottinger, Almagor & Lahijanian (IROS 2022)."""

import math
import random
import unittest

from mrn_coord.mapf.kcbs import (
    DubinsCar,
    first_conflict,
    kcbs,
    min_separation,
    plan_trajectory,
    propagate,
    trajectory_feasible,
)


BOUNDS = (0.0, 8.0, 0.0, 8.0)
CAR = DubinsCar(speed=1.0, omega_max=1.5, radius=0.3)
DT = 0.1
KW = dict(prim_steps=5, goal_radius=0.5)


class TestDynamics(unittest.TestCase):
    def test_propagate_straight(self):
        # zero turn rate -> straight line at constant speed
        s = propagate(CAR, (0.0, 0.0, 0.0), 0.0, 0.5)
        self.assertAlmostEqual(s[0], 0.5)
        self.assertAlmostEqual(s[1], 0.0)
        self.assertAlmostEqual(s[2], 0.0)

    def test_propagate_arc_radius(self):
        # turning at omega_max traces the minimum-radius circle
        s0 = (0.0, 0.0, 0.0)
        s = propagate(CAR, s0, CAR.omega_max, DT)
        # displacement magnitude over dt is exactly speed*dt (constant speed)
        self.assertAlmostEqual(math.hypot(s[0], s[1]),
                               2 * CAR.min_turn_radius
                               * math.sin(CAR.omega_max * DT / 2), places=6)

    def test_min_turn_radius(self):
        self.assertAlmostEqual(CAR.min_turn_radius, 1.0 / 1.5)


class TestLowLevel(unittest.TestCase):
    def test_single_plan_feasible(self):
        tr = plan_trajectory(CAR, (1, 1, 0), (7, 7, 0), BOUNDS, [], [],
                             rng=random.Random(1), dt=DT, **KW)
        self.assertIsNotNone(tr)
        self.assertTrue(trajectory_feasible(CAR, tr, DT))
        # ends within the goal radius
        self.assertLessEqual(math.hypot(tr[-1][1] - 7, tr[-1][2] - 7), 0.5)

    def test_heading_away_must_curve(self):
        tr = plan_trajectory(CAR, (4, 4, math.pi), (6, 4, 0), BOUNDS, [], [],
                             rng=random.Random(2), dt=DT, **KW)
        self.assertIsNotNone(tr)
        ths = [s[3] for s in tr]
        self.assertGreater(max(ths) - min(ths), 0.5)
        self.assertTrue(trajectory_feasible(CAR, tr, DT))

    def test_constraint_is_respected(self):
        from mrn_coord.mapf.kcbs import Constraint
        # forbid a tube straddling the straight path; the plan must detour
        con = [Constraint(4.0, 4.0, 0.0, 100.0, 1.0)]
        tr = plan_trajectory(CAR, (1, 4, 0), (7, 4, 0), BOUNDS, [], con,
                             rng=random.Random(3), dt=DT, **KW)
        self.assertIsNotNone(tr)
        for (_t, x, y, _th) in tr:
            self.assertGreaterEqual(math.hypot(x - 4.0, y - 4.0), 1.0 - 1e-6)


class TestConflictDetection(unittest.TestCase):
    def test_first_conflict_and_separation(self):
        a = [(k * DT, k * DT, 0.0, 0.0) for k in range(20)]
        b = [(k * DT, 2.0 - k * DT, 0.0, math.pi) for k in range(20)]
        c = first_conflict(a, b, 0.3, 0.3, DT)
        self.assertIsNotNone(c)
        self.assertLess(min_separation(a, b, DT), 0.6)

    def test_no_conflict_far_apart(self):
        a = [(k * DT, k * DT, 0.0, 0.0) for k in range(20)]
        b = [(k * DT, k * DT, 5.0, 0.0) for k in range(20)]
        self.assertIsNone(first_conflict(a, b, 0.3, 0.3, DT))


class TestKCBS(unittest.TestCase):
    def test_perpendicular_crossing_resolved(self):
        cars = {0: CAR, 1: CAR}
        s = {0: (1, 4, 0), 1: (4, 1, math.pi / 2)}
        g = {0: (7, 4, 0), 1: (4, 7, math.pi / 2)}
        # uncoordinated roots collide
        r0 = plan_trajectory(CAR, s[0], g[0], BOUNDS, [], [],
                             rng=random.Random(hash(0) & 0xFFFFFFFF), dt=DT, **KW)
        r1 = plan_trajectory(CAR, s[1], g[1], BOUNDS, [], [],
                             rng=random.Random(hash(1) & 0xFFFFFFFF), dt=DT, **KW)
        self.assertIsNotNone(first_conflict(r0, r1, 0.3, 0.3, DT))
        sol = kcbs(cars, s, g, BOUNDS, [], dt=DT, window=0.4,
                   max_expansions=400, rng=random.Random(7), **KW)
        self.assertIsNotNone(sol)
        sep = min_separation(sol.trajectories[0], sol.trajectories[1], DT)
        self.assertGreaterEqual(sep, 2 * CAR.radius - 1e-6)
        for i in cars:
            self.assertTrue(trajectory_feasible(CAR, sol.trajectories[i], DT))
        self.assertGreater(sol.high_level_expansions, 1)

    def test_three_cars(self):
        cars = {i: CAR for i in range(3)}
        s = {0: (1, 1, 0), 1: (7, 1, math.pi), 2: (4, 7, -math.pi / 2)}
        g = {0: (7, 7, 0), 1: (1, 7, math.pi), 2: (4, 1, -math.pi / 2)}
        sol = kcbs(cars, s, g, BOUNDS, [], dt=DT, window=0.3,
                   max_expansions=400, rng=random.Random(5), **KW)
        self.assertIsNotNone(sol)
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertGreaterEqual(
                    min_separation(sol.trajectories[a], sol.trajectories[b], DT),
                    2 * CAR.radius - 1e-6)

    def test_deterministic(self):
        cars = {0: CAR, 1: CAR}
        s = {0: (1, 4, 0), 1: (4, 1, math.pi / 2)}
        g = {0: (7, 4, 0), 1: (4, 7, math.pi / 2)}
        a = kcbs(cars, s, g, BOUNDS, [], dt=DT, window=0.4,
                 max_expansions=400, rng=random.Random(7), **KW)
        b = kcbs(cars, s, g, BOUNDS, [], dt=DT, window=0.4,
                 max_expansions=400, rng=random.Random(7), **KW)
        self.assertEqual(a.high_level_expansions, b.high_level_expansions)
        self.assertAlmostEqual(a.cost, b.cost, places=9)


if __name__ == "__main__":
    unittest.main()

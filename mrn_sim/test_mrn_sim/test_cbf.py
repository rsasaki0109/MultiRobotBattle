"""Tests for the Control Barrier Function (CBF) safety filter.

The contract: the filter (1) returns the nominal command unchanged when it is
safe, (2) solves the minimal-deviation QP exactly (no feasible command is closer
to nominal than the one returned), and (3) keeps the safe set forward-invariant
— driving a robot straight at an obstacle, the filtered command never lets it
penetrate the safety boundary.
"""

import math
import random
import unittest

from mrn_sim.cbf import CBFConfig, _feasible, _solve_qp, cbf_filter
from mrn_sim.kinematics import unicycle_step


class TestQP(unittest.TestCase):
    def test_optimal_vs_sampling(self):
        rng = random.Random(0)
        tested = 0
        for _ in range(1500):
            rows = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-2, 2))
                    for _ in range(rng.randint(1, 5))]
            rows += [(1, 0, -5), (-1, 0, -5), (0, 1, -5), (0, -1, -5)]
            u_nom = (rng.uniform(-3, 3), rng.uniform(-3, 3))
            sampled_best = None
            for _ in range(500):
                p = (rng.uniform(-5, 5), rng.uniform(-5, 5))
                if _feasible(p, rows):
                    d = (p[0] - u_nom[0]) ** 2 + (p[1] - u_nom[1]) ** 2
                    sampled_best = d if sampled_best is None else min(sampled_best, d)
            if sampled_best is None:
                continue                          # empty polytope, skip
            sol = _solve_qp(u_nom, rows)
            self.assertTrue(_feasible(sol, rows))
            d_sol = (sol[0] - u_nom[0]) ** 2 + (sol[1] - u_nom[1]) ** 2
            self.assertLessEqual(d_sol, sampled_best + 1e-6)
            tested += 1
        self.assertGreater(tested, 500)


class TestCBFFilter(unittest.TestCase):
    def test_passes_through_when_safe(self):
        cfg = CBFConfig()
        out = cbf_filter((0.0, 0.0, 0.0), (1.0, 0.2), [(10.0, 10.0, 0.5)], cfg)
        self.assertAlmostEqual(out[0], 1.0, places=6)
        self.assertAlmostEqual(out[1], 0.2, places=6)

    def test_forward_invariant_against_static_obstacle(self):
        cfg = CBFConfig()
        pose = (0.0, 0.0, 0.0)
        obs = (2.5, 0.0, 0.5)
        worst = float("inf")
        dt = 0.05
        for _ in range(300):
            u = cbf_filter(pose, (1.5, 0.0), [obs], cfg)   # charge straight in
            pose = unicycle_step(pose, u[0], u[1], dt)
            worst = min(worst, math.hypot(pose[0] - obs[0], pose[1] - obs[1])
                        - obs[2] - cfg.robot_radius)
        self.assertGreater(worst, -1e-3)                   # never penetrates

    def test_respects_velocity_limits(self):
        cfg = CBFConfig()
        v, omega = cbf_filter((0.0, 0.0, 0.0), (99.0, 99.0), [], cfg)
        self.assertLessEqual(v, cfg.max_v + 1e-9)
        self.assertLessEqual(abs(omega), cfg.max_omega + 1e-9)


class TestCBFInPolicy(unittest.TestCase):
    def test_mpc_cbf_is_collision_free_on_doorway(self):
        import os

        from mrn_sim.benchmark import load_scenario, mpc_policy, run_scenario
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sc = load_scenario(os.path.join(here, "scenarios", "doorway.yaml"))
        try:
            policy = mpc_policy(sc, safety="cbf")   # builds the global plan (mrn_coord)
        except ModuleNotFoundError:
            self.skipTest("mrn_coord not importable (needs colcon build)")
        r = run_scenario(sc, policy, dt=0.1, max_steps=600)
        d = r.as_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["collisions"], 0)


if __name__ == "__main__":
    unittest.main()

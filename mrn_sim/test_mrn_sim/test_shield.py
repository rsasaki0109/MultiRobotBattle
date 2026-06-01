"""Tests for the certified runtime safety shield.

The contract is stronger than the look-ahead CBF filter's: (1) the guarantee is
on the **robot body**, not a look-ahead point; (2) it holds in discrete time
under the acceleration limit — a feasible safe command (brake) always exists, so
the QP never traps the robot; (3) under an adversary that drives straight at the
nearest obstacle at full speed the body never crosses the boundary, whereas the
same command unshielded collides every time.
"""

import math
import random
import unittest

from mrn_sim.kinematics import unicycle_step
from mrn_sim.shield import ShieldConfig, braking_speed_cap, shield_step


def _body_clearance(pose, obstacles, rr):
    return min(math.hypot(pose[0] - o[0], pose[1] - o[1]) - o[2] - rr
               for o in obstacles)


class TestShieldBasics(unittest.TestCase):
    def test_passes_through_when_safe(self):
        cfg = ShieldConfig()
        v, omega = shield_step((0.0, 0.0, 0.0, 1.0), (1.0, 0.2),
                               [(10.0, 10.0, 0.5)], 0.1, cfg)
        self.assertAlmostEqual(v, 1.0, places=6)       # at max accel reach v_des
        self.assertAlmostEqual(omega, 0.2, places=6)

    def test_respects_limits(self):
        cfg = ShieldConfig()
        v, omega = shield_step((0.0, 0.0, 0.0, 0.0), (99.0, 99.0), [], 0.1, cfg)
        self.assertLessEqual(v, cfg.max_v + 1e-9)
        self.assertLessEqual(v, 0.0 + cfg.a_max * 0.1 + 1e-9)   # accel-limited
        self.assertLessEqual(abs(omega), cfg.max_omega + 1e-9)

    def test_braking_cap_is_zero_at_boundary(self):
        cfg = ShieldConfig()
        d = 0.5 + cfg.robot_radius + cfg.safety_margin
        # exactly on the safety boundary -> no speed is safe
        cap = braking_speed_cap((0.0, 0.0, 0.0), [(d, 0.0, 0.5)], cfg)
        self.assertAlmostEqual(cap, 0.0, places=6)
        # far away -> capped only by max_v
        self.assertAlmostEqual(
            braking_speed_cap((0.0, 0.0, 0.0), [(50.0, 0.0, 0.5)], cfg),
            cfg.max_v, places=6)


class TestBodyForwardInvariance(unittest.TestCase):
    def test_head_on_charge_never_penetrates_body(self):
        cfg = ShieldConfig()
        pose, v = (0.0, 0.0, 0.0), 0.0
        obs = (3.0, 0.0, 0.5)
        worst = float("inf")
        for _ in range(400):
            v, omega = shield_step((pose[0], pose[1], pose[2], v),
                                   (cfg.max_v, 0.0), [obs], 0.1, cfg)
            pose = unicycle_step(pose, v, omega, 0.1)
            worst = min(worst, _body_clearance(pose, [obs], cfg.robot_radius))
        self.assertGreater(worst, -1e-3)               # body never crosses

    def test_always_feasible_when_boxed_in(self):
        # surrounded by obstacles, any nominal command -> still returns a finite,
        # limit-respecting command (the polytope is never empty: brake exists)
        cfg = ShieldConfig()
        obstacles = [(1.5, 0.0, 0.5), (-1.5, 0.0, 0.5),
                     (0.0, 1.5, 0.5), (0.0, -1.5, 0.5)]
        for _ in range(50):
            v, omega = shield_step((0.0, 0.0, 0.3, 1.0), (1.6, 2.0),
                                   obstacles, 0.1, cfg)
            self.assertTrue(math.isfinite(v) and math.isfinite(omega))
            self.assertLessEqual(v, cfg.max_v + 1e-9)
            self.assertLessEqual(abs(omega), cfg.max_omega + 1e-9)


class TestAdversarialCertificate(unittest.TestCase):
    def test_shield_beats_the_adversary(self):
        cfg = ShieldConfig()
        rng = random.Random(0)
        shield_coll, unshielded_coll, ran = 0, 0, 0
        for _ in range(120):
            obstacles = [(rng.uniform(1.0, 9.0), rng.uniform(-3.0, 3.0),
                          rng.uniform(0.3, 0.8)) for _ in range(rng.randint(1, 4))]
            theta0 = rng.uniform(-0.3, 0.3)
            if _body_clearance((0.0, 0.0), obstacles, cfg.robot_radius) < 0.2:
                continue
            ran += 1
            for shielded in (True, False):
                pose, v, worst = (0.0, 0.0, theta0), 0.0, float("inf")
                for _k in range(200):
                    near = min(obstacles, key=lambda o: math.hypot(
                        pose[0] - o[0], pose[1] - o[1]))
                    ang = math.atan2(near[1] - pose[1], near[0] - pose[0])
                    derr = math.atan2(math.sin(ang - pose[2]),
                                      math.cos(ang - pose[2]))
                    u = (cfg.max_v, 4.0 * derr)
                    if shielded:
                        cv, cw = shield_step((pose[0], pose[1], pose[2], v),
                                             u, obstacles, 0.1, cfg)
                    else:
                        cv = max(-cfg.max_v, min(cfg.max_v, u[0]))
                        cw = max(-cfg.max_omega, min(cfg.max_omega, u[1]))
                    pose = unicycle_step(pose, cv, cw, 0.1)
                    v = cv
                    worst = min(worst, _body_clearance(
                        pose, obstacles, cfg.robot_radius))
                if worst < -1e-3:
                    if shielded:
                        shield_coll += 1
                    else:
                        unshielded_coll += 1
        self.assertGreater(ran, 80)
        self.assertEqual(shield_coll, 0)               # certificate holds
        self.assertGreater(unshielded_coll, 0)         # the attack is real


class TestShieldInPolicy(unittest.TestCase):
    def test_mpc_shield_is_collision_free_on_doorway(self):
        import os

        from mrn_sim.benchmark import load_scenario, mpc_policy, run_scenario
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sc = load_scenario(os.path.join(here, "scenarios", "doorway.yaml"))
        try:
            policy = mpc_policy(sc, safety="shield")
        except ModuleNotFoundError:
            self.skipTest("mrn_coord not importable (needs colcon build)")
        r = run_scenario(sc, policy, dt=0.1, max_steps=600).as_dict()
        self.assertTrue(r["success"])
        self.assertEqual(r["collisions"], 0)


if __name__ == "__main__":
    unittest.main()

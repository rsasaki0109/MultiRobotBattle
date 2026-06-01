"""Tests for the iLQR Model Predictive Control local controller."""

import math
import unittest

from mrn_sim import Obstacle, Robot, World
from mrn_sim.kinematics import unicycle_step
from mrn_sim.mpc import MPCConfig, mpc_command, solve_ilqr


def _carrot(pose, path, lookahead, start_index):
    """Forward-tracking carrot point (monotonic index), mirroring the DWA test."""
    nearest, best = start_index, float("inf")
    for i in range(start_index, len(path)):
        d = math.hypot(path[i][0] - pose[0], path[i][1] - pose[1])
        if d < best:
            best, nearest = d, i
    for i in range(nearest, len(path)):
        if math.hypot(path[i][0] - pose[0], path[i][1] - pose[1]) >= lookahead:
            return path[i], nearest
    return path[-1], nearest


class TestILQR(unittest.TestCase):
    def test_open_space_progress_and_limits(self):
        # From rest in open space, the first command drives toward the goal
        # within the velocity/accel limits.
        cfg = MPCConfig()
        world = World(12.0, 8.0, {}, [])
        (v, omega), us = mpc_command((1.0, 4.0, 0.0), 0.0, 0.0, (3.0, 4.0),
                                     [], world, cfg)
        self.assertGreater(v, 0.0)
        self.assertLessEqual(v, cfg.accel_v * cfg.dt + 1e-9)   # accel-limited
        self.assertLessEqual(abs(omega), cfg.max_omega + 1e-9)
        self.assertEqual(len(us), cfg.horizon)

    def test_reduces_cost_vs_zero_controls(self):
        # The optimized sequence has no higher cost than doing nothing.
        cfg = MPCConfig()
        world = World(12.0, 8.0, {}, [])
        obstacles = [(2.5, 4.0, 0.6)]
        _, xs, cost = solve_ilqr((1.0, 4.0, 0.0), (5.0, 4.5), obstacles, world,
                                 cfg)
        _, _, zero_cost = solve_ilqr((1.0, 4.0, 0.0), (5.0, 4.5), obstacles,
                                     world, cfg, u_init=[(0.0, 0.0)] * cfg.horizon)
        self.assertLessEqual(cost, zero_cost + 1e-6)

    def test_bends_around_obstacle(self):
        # An obstacle near (just off) the straight line bends the optimized
        # trajectory around it rather than through it. (A perfectly head-on
        # obstacle is a symmetric local minimum for any gradient-based local
        # controller — iLQR like DWA — which is why these follow a global plan
        # that breaks the symmetry; see test_tracks_planned_path_*.)
        cfg = MPCConfig(horizon=25)
        world = World(12.0, 8.0, {}, [])
        obstacles = [(3.0, 4.3, 0.6)]
        _, xs, _ = solve_ilqr((1.0, 4.0, 0.0), (5.0, 4.0), obstacles, world, cfg)
        # no trajectory point penetrates the obstacle surface
        worst = min(math.hypot(x[0] - 3.0, x[1] - 4.3) - 0.6 for x in xs)
        self.assertGreater(worst, -1e-6)
        # and it had to dip off the y=4 line to do it
        self.assertGreater(max(abs(x[1] - 4.0) for x in xs), 0.1)

    def test_warm_start_is_cheaper_to_refine(self):
        # Warm-starting from a shifted solution yields a no-worse cost than cold.
        cfg = MPCConfig()
        world = World(12.0, 8.0, {}, [])
        us, _, _ = solve_ilqr((1.0, 4.0, 0.0), (4.0, 4.0), [], world, cfg)
        warm = us[1:] + [us[-1]]
        _, _, cold = solve_ilqr((1.1, 4.0, 0.0), (4.0, 4.0), [], world, cfg)
        _, _, warmed = solve_ilqr((1.1, 4.0, 0.0), (4.0, 4.0), [], world, cfg,
                                  u_init=warm)
        self.assertLessEqual(warmed, cold + 1e-6)

    def test_tracks_planned_path_to_goal_collision_free(self):
        from mrn_sim.kinodynamic import plan_kinodynamic
        from mrn_sim.world import step

        cfg = MPCConfig()
        world = World(12.0, 8.0, {"r": Robot("r", (1.0, 4.0, 0.0))},
                      [Obstacle(6.0, 4.0, 1.2)])
        obstacles = [(6.0, 4.0, 1.2)]
        goal = (11.0, 4.0)
        plan = plan_kinodynamic(world, (1.0, 4.0, 0.0), (11.0, 4.0, 0.0),
                                turn_radius=1.0)
        self.assertIsNotNone(plan)
        path = plan.waypoints
        v = omega = 0.0
        warm = None
        idx = 0
        reached = False
        for _ in range(400):
            p = world.robots["r"].pose
            if math.hypot(p[0] - goal[0], p[1] - goal[1]) <= cfg.goal_tolerance:
                reached = True
                break
            local_goal, idx = _carrot(p, path, 1.2, idx)
            (v, omega), us = mpc_command(p, v, omega, local_goal, obstacles,
                                         world, cfg, warm)
            warm = us[1:] + [us[-1]]
            world = step(world, {"r": (v, omega)}, cfg.dt)
            self.assertTrue(world.is_free(*world.robots["r"].pose[:2], 0.25))
        self.assertTrue(reached)

    def test_moving_obstacle_prediction_avoided(self):
        # A moving obstacle predicted to sweep across the path is avoided in
        # space-time: the trajectory keeps clear of it at the matching step.
        cfg = MPCConfig(horizon=20)
        world = World(12.0, 8.0, {}, [])
        # one obstacle marching down through the corridor the robot crosses;
        # `traj` is its predicted (x, y, radius) indexed by timestep.
        traj = [(4.0, 6.0 - 0.25 * t, 0.5) for t in range(cfg.horizon + 1)]
        _, xs, _ = solve_ilqr((1.0, 4.0, 0.0), (8.0, 4.0), [], world, cfg,
                              moving=[traj])
        worst = min(math.hypot(xs[t][0] - traj[t][0], xs[t][1] - traj[t][1])
                    - traj[t][2] for t in range(len(xs)))
        self.assertGreater(worst, -1e-6)


if __name__ == "__main__":
    unittest.main()

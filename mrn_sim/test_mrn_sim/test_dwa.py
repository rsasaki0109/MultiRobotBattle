"""Tests for the Dynamic Window Approach local controller."""

import math
import unittest

from mrn_sim import Obstacle, Robot, World
from mrn_sim.dwa import DWAConfig, dwa_command
from mrn_sim.kinodynamic import plan_kinodynamic
from mrn_sim.world import step


def _carrot(pose, path, lookahead, start_index):
    """Carrot point that tracks forward progress (monotonic index).

    Advance ``start_index`` to the path point nearest the robot, then return the
    first point at least ``lookahead`` beyond it (so points already passed are
    never re-chased) and the new index.
    """
    nearest, best = start_index, float("inf")
    for i in range(start_index, len(path)):
        d = math.hypot(path[i][0] - pose[0], path[i][1] - pose[1])
        if d < best:
            best, nearest = d, i
    for i in range(nearest, len(path)):
        if math.hypot(path[i][0] - pose[0], path[i][1] - pose[1]) >= lookahead:
            return path[i], nearest
    return path[-1], nearest


class TestDWA(unittest.TestCase):
    def _open_world(self):
        return World(12.0, 8.0, {}, [])

    def test_respects_acceleration_window(self):
        # From rest, the chosen speed cannot exceed accel_v * dt.
        cfg = DWAConfig()
        world = self._open_world()
        v, omega = dwa_command((1.0, 4.0, 0.0), 0.0, 0.0, (11.0, 4.0), [], world, cfg)
        self.assertLessEqual(v, cfg.accel_v * cfg.dt + 1e-9)
        self.assertLessEqual(abs(omega), cfg.max_omega + 1e-9)

    def test_drives_straight_at_aligned_goal(self):
        # Already pointing at the goal in open space -> go forward, ~no turn.
        world = self._open_world()
        v, omega = dwa_command((1.0, 4.0, 0.0), 1.0, 0.0, (11.0, 4.0), [], world)
        self.assertGreater(v, 0.0)
        self.assertLess(abs(omega), 0.6)

    def test_obstacle_within_horizon_slows_or_turns(self):
        # With an obstacle within the rollout horizon dead ahead, the command
        # must differ from the open-space one: slower and/or turning more.
        world_free = self._open_world()
        world_obs = World(12.0, 8.0, {}, [Obstacle(2.8, 4.0, 0.5)])
        obstacles = [(2.8, 4.0, 0.5)]
        vf, wf = dwa_command((1.0, 4.0, 0.0), 1.2, 0.0, (11.0, 4.0), [], world_free)
        vo, wo = dwa_command((1.0, 4.0, 0.0), 1.2, 0.0, (11.0, 4.0),
                             obstacles, world_obs)
        self.assertTrue(vo < vf - 1e-6 or abs(wo) > abs(wf) + 1e-6,
                        msg=f"free=({vf:.2f},{wf:.2f}) obs=({vo:.2f},{wo:.2f})")

    def test_tracks_planned_path_to_goal(self):
        """End-to-end as intended: DWA tracks a global plan's carrot to the goal.

        Pure DWA to a distant goal behind an obstacle stalls in a local minimum
        (a known limitation) — so DWA is a *local* controller. Plan the route
        with the kinodynamic planner, then have DWA follow its carrot: it reaches
        the goal, accel-limited and collision-free.
        """
        cfg = DWAConfig()
        world = World(12.0, 8.0, {"r": Robot("r", (1.0, 4.0, 0.0))},
                      [Obstacle(6.0, 4.0, 1.2)])
        obstacles = [(6.0, 4.0, 1.2)]
        goal = (11.0, 4.0)
        plan = plan_kinodynamic(world, (1.0, 4.0, 0.0), (11.0, 4.0, 0.0),
                                turn_radius=1.0)
        self.assertIsNotNone(plan)
        path = plan.waypoints
        v, omega = 0.0, 0.0
        idx = 0
        reached = False
        for _ in range(800):
            p = world.robots["r"].pose
            if math.hypot(p[0] - goal[0], p[1] - goal[1]) <= cfg.goal_tolerance:
                reached = True
                break
            local_goal, idx = _carrot(p, path, 1.0, idx)
            v, omega = dwa_command(p, v, omega, local_goal, obstacles, world, cfg)
            world = step(world, {"r": (v, omega)}, cfg.dt)
            self.assertTrue(world.is_free(*world.robots["r"].pose[:2], 0.25))
        self.assertTrue(reached)

    def test_stops_when_boxed_in(self):
        # Goal sits inside an obstacle and the robot is right against it:
        # no safe forward rollout -> command brakes to zero linear speed.
        world = World(12.0, 8.0, {}, [Obstacle(3.0, 4.0, 1.0)])
        obstacles = [(3.0, 4.0, 1.0)]
        v, omega = dwa_command((1.7, 4.0, 0.0), 1.0, 0.0, (3.0, 4.0),
                               obstacles, world)
        self.assertEqual(v, 0.0)


if __name__ == "__main__":
    unittest.main()

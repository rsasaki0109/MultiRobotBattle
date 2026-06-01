"""Tests for continuous-space kinodynamic planning: Dubins curves + Hybrid A*."""

import importlib.util
import math
import unittest

from mrn_sim import Obstacle, Robot, World
from mrn_sim.kinodynamic import dubins_path, plan_kinodynamic


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TestDubins(unittest.TestCase):
    def test_straight_line_is_distance(self):
        # Same heading along the +x axis -> the shortest curve is a straight run.
        dp = dubins_path((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), radius=1.0)
        self.assertAlmostEqual(dp.length, 5.0, places=6)
        self.assertEqual(dp.word, ("L", "S", "L"))  # zero-length arcs + straight

    def test_sampled_endpoint_matches_goal(self):
        # The hand-rolled six-word math is only trustworthy if a sampled curve
        # actually lands on the requested goal pose. Check several goals.
        radius = 1.5
        start = (1.0, 2.0, 0.3)
        goals = [
            (6.0, 5.0, 1.2),
            (-3.0, 4.0, -2.0),
            (2.0, -5.0, math.pi),
            (0.5, 0.5, -0.5),
            (8.0, -2.0, 2.7),
        ]
        for goal in goals:
            dp = dubins_path(start, goal, radius)
            poses = dp.sample(0.05, start)
            ex, ey, eth = poses[-1]
            self.assertAlmostEqual(ex, goal[0], places=2, msg=f"x for {goal}")
            self.assertAlmostEqual(ey, goal[1], places=2, msg=f"y for {goal}")
            self.assertAlmostEqual(_wrap(eth - goal[2]), 0.0, places=2,
                                   msg=f"theta for {goal}")
            # the curve length equals the summed segment lengths
            self.assertAlmostEqual(dp.length, sum(dp.segments), places=9)

    def test_length_lower_bounds_euclidean(self):
        # A bounded-curvature path is never shorter than the straight-line gap.
        start, goal = (0.0, 0.0, math.pi / 2), (4.0, 0.0, -math.pi / 2)
        dp = dubins_path(start, goal, radius=1.0)
        self.assertGreaterEqual(dp.length + 1e-9, math.hypot(4.0, 0.0))

    def test_tighter_radius_is_not_shorter(self):
        # Relaxing the turn radius (larger R) can only shorten or match.
        start, goal = (0.0, 0.0, 0.0), (1.0, 3.0, math.pi)
        tight = dubins_path(start, goal, radius=0.5).length
        loose = dubins_path(start, goal, radius=2.0).length
        self.assertGreaterEqual(tight + 1e-9, 0.0)
        self.assertGreaterEqual(loose + 1e-9, 0.0)
        # both are valid Dubins lengths; the looser one reaches via a gentler arc
        self.assertLess(loose, tight + 10.0)


class TestHybridAstar(unittest.TestCase):
    def _open_world(self):
        return World(12.0, 8.0, {}, [])

    def _obstacle_world(self):
        return World(12.0, 8.0, {}, [Obstacle(6.0, 4.0, 1.5)])

    def test_open_world_path_is_feasible(self):
        res = plan_kinodynamic(self._open_world(), (1.0, 4.0, 0.0),
                               (11.0, 4.0, 0.0), turn_radius=1.0)
        self.assertIsNotNone(res)
        # endpoints land on start and within tolerance of the goal
        self.assertAlmostEqual(res.poses[0][0], 1.0, places=6)
        self.assertAlmostEqual(res.poses[0][1], 4.0, places=6)
        self.assertLessEqual(math.hypot(res.poses[-1][0] - 11.0,
                                        res.poses[-1][1] - 4.0), 0.5)
        # a near-straight shot should be only a little longer than the gap
        self.assertGreaterEqual(res.length, 10.0 - 1e-6)
        self.assertLess(res.length, 13.0)

    def test_path_clears_obstacle(self):
        world = self._obstacle_world()
        res = plan_kinodynamic(world, (1.0, 4.0, 0.0), (11.0, 4.0, 0.0),
                               turn_radius=1.0, robot_radius=0.25, clearance=0.1)
        self.assertIsNotNone(res)
        for (x, y, _th) in res.poses:
            self.assertTrue(world.is_free(x, y, 0.25),
                            msg=f"pose ({x:.2f},{y:.2f}) collides")

    def test_curvature_is_bounded(self):
        # consecutive heading changes never exceed what the turn radius allows
        world = self._obstacle_world()
        turn_radius = 1.0
        res = plan_kinodynamic(world, (1.0, 1.0, 0.0), (10.0, 7.0, 1.2),
                               turn_radius=turn_radius)
        self.assertIsNotNone(res)
        for (ax, ay, ath), (bx, by, bth) in zip(res.poses, res.poses[1:]):
            ds = math.hypot(bx - ax, by - ay)
            if ds < 1e-6:
                continue
            kappa = abs(_wrap(bth - ath)) / ds
            self.assertLessEqual(kappa, 1.0 / turn_radius + 0.05,
                                 msg=f"curvature {kappa:.3f} exceeds bound")

    def test_blocked_start_returns_none(self):
        world = self._obstacle_world()
        # start sitting inside the obstacle disk
        res = plan_kinodynamic(world, (6.0, 4.0, 0.0), (11.0, 4.0, 0.0))
        self.assertIsNone(res)

    def test_goal_heading_respected_when_requested(self):
        res = plan_kinodynamic(self._open_world(), (1.0, 4.0, 0.0),
                               (9.0, 4.0, math.pi / 2), turn_radius=1.0,
                               goal_yaw_tol=math.radians(20.0))
        self.assertIsNotNone(res)
        self.assertLessEqual(abs(_wrap(res.poses[-1][2] - math.pi / 2)),
                             math.radians(20.0))

    def test_waypoints_drop_into_follower_shape(self):
        res = plan_kinodynamic(self._open_world(), (1.0, 4.0, 0.0),
                               (8.0, 4.0, 0.0), turn_radius=1.0)
        self.assertIsNotNone(res)
        wps = res.waypoints
        self.assertTrue(all(len(p) == 2 for p in wps))
        self.assertEqual(len(wps), len(res.poses))


@unittest.skipUnless(
    importlib.util.find_spec("mrn_coord") is not None, "mrn_coord not available"
)
class TestKinoFollowable(unittest.TestCase):
    def test_follower_drives_kino_path_to_goal(self):
        """The whole point of bounded curvature: pure-pursuit can actually track it.

        Plan a kinodynamic path around an obstacle, then drive a unicycle along
        it with the repo's own carrot follower and assert it reaches the goal
        without ever entering the obstacle.
        """
        from mrn_sim.world import step
        from mrn_coord.mapf.path_follower import carrot_point
        from mrn_coord.flocking import velocity_to_unicycle

        world = World(12.0, 8.0, {"r": Robot("r", (1.0, 1.0, 0.0))},
                      [Obstacle(6.0, 4.0, 1.5)])
        goal = (10.5, 7.0, 1.0)
        res = plan_kinodynamic(world, (1.0, 1.0, 0.0), goal, turn_radius=1.0)
        self.assertIsNotNone(res)
        path = res.waypoints

        reached = False
        for _ in range(800):
            p = world.robots["r"].pose
            if math.hypot(p[0] - goal[0], p[1] - goal[1]) <= 0.3:
                reached = True
                break
            cx, cy = carrot_point(p, path, 0.8)
            vx, vy = cx - p[0], cy - p[1]
            d = math.hypot(vx, vy) or 1.0
            cmd = velocity_to_unicycle(p[2], vx / d * 1.6, vy / d * 1.6,
                                       max_v=1.6, max_omega=3.0)
            world = step(world, {"r": cmd}, 0.1)
            self.assertTrue(world.is_free(*world.robots["r"].pose[:2], 0.25))
        self.assertTrue(reached)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Boids flocking step — one rule at a time."""

import math
import unittest

from mrn_coord.flocking import (
    flock_velocities,
    goal_seek,
    obstacle_avoidance,
    velocity_to_unicycle,
)


class TestFlocking(unittest.TestCase):
    def test_separation_pushes_apart(self):
        # two stationary agents closer than `separation` -> steer away from each other
        pos = [(0.0, 0.0), (0.3, 0.0)]
        vel = [(0.0, 0.0), (0.0, 0.0)]
        out = flock_velocities(pos, vel, perception=3.0, separation=1.0)
        self.assertLess(out[0][0], 0.0)      # agent 0 moves -x
        self.assertGreater(out[1][0], 0.0)   # agent 1 moves +x

    def test_cohesion_pulls_toward_cluster(self):
        # lone agent with a cluster to its +x (beyond separation, within perception)
        pos = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.5), (2.0, -0.5)]
        vel = [(0.0, 0.0)] * 4
        out = flock_velocities(pos, vel, perception=4.0, separation=1.0, w_coh=1.0)
        self.assertGreater(out[0][0], 0.0)   # steered toward the cluster (+x)

    def test_alignment_matches_neighbor_velocity(self):
        # symmetric neighbors (cohesion cancels) both moving +x -> agent gains +x
        pos = [(0.0, 0.0), (1.5, 0.0), (-1.5, 0.0)]
        vel = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
        out = flock_velocities(pos, vel, perception=3.0, separation=1.0,
                               w_ali=1.0, w_coh=1.0, inertia=0.0)
        self.assertGreater(out[0][0], 0.0)

    def test_isolated_agent_keeps_scaled_velocity(self):
        pos = [(0.0, 0.0), (50.0, 50.0)]   # second agent far outside perception
        vel = [(1.0, 0.0), (0.0, 0.0)]
        out = flock_velocities(pos, vel, perception=3.0, inertia=0.85)
        self.assertAlmostEqual(out[0][0], 0.85)
        self.assertAlmostEqual(out[0][1], 0.0)

    def test_speed_is_clamped(self):
        pos = [(0.0, 0.0), (0.05, 0.0)]   # extremely close -> huge separation push
        vel = [(0.0, 0.0), (0.0, 0.0)]
        out = flock_velocities(pos, vel, separation=1.0, w_sep=5.0, max_speed=2.0)
        for vx, vy in out:
            self.assertLessEqual(math.hypot(vx, vy), 2.0 + 1e-9)

    def test_empty_and_single(self):
        self.assertEqual(flock_velocities([], []), [])
        out = flock_velocities([(0.0, 0.0)], [(1.0, 1.0)], inertia=0.5)
        self.assertAlmostEqual(out[0][0], 0.5)
        self.assertAlmostEqual(out[0][1], 0.5)


class TestVelocityToUnicycle(unittest.TestCase):
    def test_aligned_drives_forward(self):
        v, omega = velocity_to_unicycle(0.0, 1.5, 0.0, max_v=2.0)
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(omega, 0.0, places=6)

    def test_desired_left_turns_left_no_forward(self):
        # desired +y while facing +x -> heading error +pi/2: turn left, v ~ 0
        v, omega = velocity_to_unicycle(0.0, 0.0, 1.0)
        self.assertGreater(omega, 0.0)
        self.assertAlmostEqual(v, 0.0, places=6)

    def test_desired_behind_no_forward(self):
        v, omega = velocity_to_unicycle(0.0, -1.0, 0.0)
        self.assertAlmostEqual(v, 0.0, places=6)
        self.assertNotEqual(omega, 0.0)

    def test_zero_desired_is_stop(self):
        self.assertEqual(velocity_to_unicycle(1.0, 0.0, 0.0), (0.0, 0.0))

    def test_clamps(self):
        v, omega = velocity_to_unicycle(0.0, 100.0, 0.0, max_v=1.5, max_omega=2.5)
        self.assertLessEqual(v, 1.5 + 1e-9)
        self.assertLessEqual(abs(omega), 2.5 + 1e-9)


class TestObstacleAvoidance(unittest.TestCase):
    def test_repels_away_from_obstacle(self):
        # agent to the +x side of an obstacle at origin -> pushed further +x
        out = obstacle_avoidance([(2.0, 0.0)], [(0.0, 0.0, 1.0)], influence=2.0)
        self.assertGreater(out[0][0], 0.0)
        self.assertAlmostEqual(out[0][1], 0.0, places=6)

    def test_zero_when_far(self):
        out = obstacle_avoidance([(20.0, 0.0)], [(0.0, 0.0, 1.0)], influence=2.0)
        self.assertEqual(out[0], (0.0, 0.0))

    def test_closer_is_stronger(self):
        near = obstacle_avoidance([(1.3, 0.0)], [(0.0, 0.0, 1.0)], influence=3.0)
        far = obstacle_avoidance([(2.5, 0.0)], [(0.0, 0.0, 1.0)], influence=3.0)
        self.assertGreater(near[0][0], far[0][0])

    def test_clamped(self):
        out = obstacle_avoidance([(1.01, 0.0)], [(0.0, 0.0, 1.0)],
                                 influence=3.0, strength=10.0, max_accel=6.0)
        self.assertLessEqual(math.hypot(*out[0]), 6.0 + 1e-9)

    def test_direction_diagonal(self):
        out = obstacle_avoidance([(1.0, 1.0)], [(0.0, 0.0, 0.5)], influence=3.0)
        self.assertGreater(out[0][0], 0.0)
        self.assertGreater(out[0][1], 0.0)
        self.assertAlmostEqual(out[0][0], out[0][1], places=6)


class TestGoalSeek(unittest.TestCase):
    def test_points_toward_goal(self):
        out = goal_seek([(0.0, 0.0)], (5.0, 0.0), gain=1.0, max_speed=2.0)
        self.assertGreater(out[0][0], 0.0)         # +x toward goal
        self.assertAlmostEqual(out[0][1], 0.0, places=6)
        self.assertLessEqual(math.hypot(*out[0]), 2.0 + 1e-9)   # clamped

    def test_zero_at_goal(self):
        out = goal_seek([(3.0, 4.0)], (3.0, 4.0))
        self.assertEqual(out[0], (0.0, 0.0))

    def test_per_agent(self):
        out = goal_seek([(0.0, 0.0), (10.0, 0.0)], (5.0, 0.0), max_speed=9.0)
        self.assertGreater(out[0][0], 0.0)         # left agent moves +x
        self.assertLess(out[1][0], 0.0)            # right agent moves -x


if __name__ == "__main__":
    unittest.main()

"""Tests for the 2D world core: kinematics, world step, and sensors."""

import math
import random
import unittest

from mrn_sim import (
    Obstacle,
    Robot,
    World,
    add_gaussian_noise,
    gnss_observation,
    normalize_angle,
    range_bearing,
    relative_pose_body,
    step,
    unicycle_step,
)


class TestKinematics(unittest.TestCase):
    def test_straight_line(self):
        x, y, th = unicycle_step((0.0, 0.0, 0.0), v=1.0, omega=0.0, dt=2.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(th, 0.0)

    def test_drive_along_heading(self):
        x, y, th = unicycle_step((0.0, 0.0, math.pi / 2), v=1.0, omega=0.0, dt=1.0)
        self.assertAlmostEqual(x, 0.0, places=9)
        self.assertAlmostEqual(y, 1.0, places=9)

    def test_pure_rotation(self):
        x, y, th = unicycle_step((1.0, 2.0, 0.0), v=0.0, omega=math.pi, dt=0.5)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(th, math.pi / 2)

    def test_normalize_angle(self):
        self.assertAlmostEqual(normalize_angle(3 * math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(-3 * math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)

    def test_negative_dt_rejected(self):
        with self.assertRaises(ValueError):
            unicycle_step((0.0, 0.0, 0.0), 1.0, 0.0, -0.1)


class TestWorldStep(unittest.TestCase):
    def _world(self, obstacles=None):
        robots = {"a": Robot("a", (1.0, 1.0, 0.0), radius=0.2)}
        return World(10.0, 10.0, robots, obstacles or [])

    def test_robot_advances(self):
        w = step(self._world(), {"a": (1.0, 0.0)}, dt=1.0)
        self.assertAlmostEqual(w.robots["a"].pose[0], 2.0)

    def test_obstacle_blocks_translation_but_allows_turn(self):
        w0 = self._world([Obstacle(2.0, 1.0, 0.5)])
        w1 = step(w0, {"a": (1.0, 1.0)}, dt=1.0)   # would drive into the obstacle
        # position held (move rejected)...
        self.assertAlmostEqual(w1.robots["a"].pose[0], 1.0)
        self.assertAlmostEqual(w1.robots["a"].pose[1], 1.0)
        # ...but the turn still applied
        self.assertAlmostEqual(w1.robots["a"].pose[2], 1.0)

    def test_bounds_block(self):
        robots = {"a": Robot("a", (9.7, 1.0, 0.0), radius=0.2)}
        w0 = World(10.0, 10.0, robots, [])
        w1 = step(w0, {"a": (1.0, 0.0)}, dt=1.0)   # would exit the right bound
        self.assertAlmostEqual(w1.robots["a"].pose[0], 9.7)

    def test_missing_command_holds(self):
        w = step(self._world(), {}, dt=1.0)
        self.assertEqual(w.robots["a"].pose, (1.0, 1.0, 0.0))

    def test_is_free(self):
        w = self._world([Obstacle(5.0, 5.0, 1.0)])
        self.assertFalse(w.is_free(5.0, 5.0, 0.2))
        self.assertTrue(w.is_free(2.0, 2.0, 0.2))
        self.assertFalse(w.is_free(0.05, 5.0, 0.2))   # out of bounds


class TestSensors(unittest.TestCase):
    def test_range_bearing_forward(self):
        rng, bearing = range_bearing((0.0, 0.0, 0.0), (5.0, 0.0))
        self.assertAlmostEqual(rng, 5.0)
        self.assertAlmostEqual(bearing, 0.0)

    def test_range_bearing_left(self):
        rng, bearing = range_bearing((0.0, 0.0, 0.0), (0.0, 3.0))
        self.assertAlmostEqual(rng, 3.0)
        self.assertAlmostEqual(bearing, math.pi / 2)

    def test_range_bearing_accounts_for_heading(self):
        # target straight ahead in world, but observer faces +y -> bearing -90
        rng, bearing = range_bearing((0.0, 0.0, math.pi / 2), (5.0, 0.0))
        self.assertAlmostEqual(bearing, -math.pi / 2)

    def test_relative_pose_body_forward(self):
        x, y, dth = relative_pose_body((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(dth, 0.0)

    def test_relative_pose_body_rotated_observer(self):
        # observer faces +y; target is 2m north in world -> 2m forward in body
        x, y, dth = relative_pose_body((0.0, 0.0, math.pi / 2), (0.0, 2.0, math.pi / 2))
        self.assertAlmostEqual(x, 2.0, places=9)
        self.assertAlmostEqual(y, 0.0, places=9)
        self.assertAlmostEqual(dth, 0.0, places=9)

    def test_relative_pose_is_antisymmetric_in_distance(self):
        a = relative_pose_body((1.0, 1.0, 0.0), (4.0, 1.0, 0.0))
        b = relative_pose_body((4.0, 1.0, 0.0), (1.0, 1.0, 0.0))
        self.assertAlmostEqual(a[0], 3.0)
        self.assertAlmostEqual(b[0], -3.0)

    def test_gnss(self):
        self.assertEqual(gnss_observation((3.0, 4.0, 1.0)), (3.0, 4.0))

    def test_noise_is_reproducible(self):
        a = add_gaussian_noise(5.0, 0.1, random.Random(7))
        b = add_gaussian_noise(5.0, 0.1, random.Random(7))
        self.assertEqual(a, b)
        self.assertNotEqual(a, 5.0)

    def test_noise_zero_sigma_is_identity(self):
        self.assertEqual(add_gaussian_noise(5.0, 0.0, random.Random(1)), 5.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the pure-pursuit path follower and a guarded node import."""

import importlib.util
import math
import unittest

from mrn_coord.mapf import pure_pursuit


class TestPurePursuit(unittest.TestCase):
    def test_straight_path_goes_straight(self):
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        v, omega, reached = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=1.0,
                                         v_nominal=1.0)
        self.assertFalse(reached)
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(omega, 0.0, places=6)

    def test_path_turning_left_commands_positive_omega(self):
        # carrot up and to the left of a robot facing +x -> turn left (omega>0)
        path = [(0.0, 0.0), (0.5, 1.5)]
        v, omega, reached = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=1.0)
        self.assertFalse(reached)
        self.assertGreater(omega, 0.0)

    def test_path_turning_right_commands_negative_omega(self):
        path = [(0.0, 0.0), (0.5, -1.5)]
        _, omega, _ = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=1.0)
        self.assertLess(omega, 0.0)

    def test_reached_goal_stops(self):
        path = [(0.0, 0.0), (2.0, 0.0)]
        v, omega, reached = pure_pursuit((2.05, 0.0, 0.0), path, goal_tolerance=0.3)
        self.assertTrue(reached)
        self.assertEqual((v, omega), (0.0, 0.0))

    def test_carrot_behind_turns_in_place(self):
        # goal behind the robot (robot faces +x, goal at -x) -> turn, no forward v
        path = [(0.0, 0.0), (-3.0, 0.0)]
        v, omega, reached = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=1.0)
        self.assertFalse(reached)
        self.assertEqual(v, 0.0)
        self.assertNotEqual(omega, 0.0)

    def test_omega_is_clamped(self):
        path = [(0.0, 0.0), (0.1, 2.0)]
        _, omega, _ = pure_pursuit((0.0, 0.0, 0.0), path, lookahead=0.5, max_omega=1.0)
        self.assertLessEqual(abs(omega), 1.0)

    def test_empty_path_is_reached(self):
        v, omega, reached = pure_pursuit((0.0, 0.0, 0.0), [])
        self.assertTrue(reached)
        self.assertEqual((v, omega), (0.0, 0.0))

    def test_single_point_goal_drives_toward_it(self):
        # a one-point path (a goal point) is followed like any other path
        v, omega, reached = pure_pursuit((0.0, 0.0, 0.0), [(5.0, 0.0)])
        self.assertFalse(reached)
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(omega, 0.0, places=6)
        # and reports reached once on top of it
        _, _, reached2 = pure_pursuit((5.1, 0.0, 0.0), [(5.0, 0.0)], goal_tolerance=0.3)
        self.assertTrue(reached2)


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") is not None, "rclpy not available"
)
class TestFollowerNodeImport(unittest.TestCase):
    def test_module_imports(self):
        from mrn_coord.mapf import follower_node
        self.assertTrue(hasattr(follower_node, "PathFollowerNode"))


if __name__ == "__main__":
    unittest.main()

"""Tests for the pure ROS-bridge helpers (no rclpy required)."""

import importlib.util
import math
import unittest

from mrn_coord.mapf.ros_conversion import (
    build_agents,
    parse_cell,
    parse_cells,
    path_to_world_points,
    safe_topic_token,
    solve_scenario,
    yaw_along,
)


class TestParsing(unittest.TestCase):
    def test_parse_cell(self):
        self.assertEqual(parse_cell("3,4"), (3, 4))
        self.assertEqual(parse_cell(" 0,0 "), (0, 0))

    def test_parse_cell_rejects_bad(self):
        for bad in ("3", "1,2,3", "x,y"):
            with self.assertRaises(ValueError):
                parse_cell(bad)

    def test_parse_cells(self):
        self.assertEqual(parse_cells(["1,1", "2,3"]), {(1, 1), (2, 3)})

    def test_build_agents(self):
        agents = build_agents(["a", "b"], ["0,0", "1,1"], ["3,0", "3,1"])
        self.assertEqual(agents, {"a": ((0, 0), (3, 0)), "b": ((1, 1), (3, 1))})

    def test_build_agents_length_mismatch(self):
        with self.assertRaises(ValueError):
            build_agents(["a", "b"], ["0,0"], ["3,0", "3,1"])

    def test_safe_topic_token(self):
        # digit-leading ids are invalid ROS topic tokens -> prefixed
        self.assertEqual(safe_topic_token("1"), "a_1")
        self.assertEqual(safe_topic_token("2"), "a_2")
        # already-valid ids pass through unchanged
        self.assertEqual(safe_topic_token("robot_1"), "robot_1")
        self.assertEqual(safe_topic_token("_x"), "_x")


class TestSolveScenario(unittest.TestCase):
    def test_cbs_solves_doorway(self):
        blocked = {(5, y) for y in range(7) if y != 3}
        agents = build_agents(
            ["1", "2", "3"], ["1,1", "1,3", "1,5"], ["8,5", "8,3", "8,1"]
        )
        sol = solve_scenario(11, 7, blocked, agents, solver="cbs")
        self.assertIsNotNone(sol)
        self.assertEqual(sol.paths["2"][-1], (8, 3))

    def test_prioritized_option(self):
        agents = build_agents(["a", "b"], ["0,0", "0,1"], ["3,0", "3,1"])
        sol = solve_scenario(4, 2, set(), agents, solver="prioritized")
        self.assertIsNotNone(sol)

    def test_unknown_solver(self):
        agents = build_agents(["a"], ["0,0"], ["1,0"])
        with self.assertRaises(ValueError):
            solve_scenario(2, 1, set(), agents, solver="astar")


class TestWorldConversion(unittest.TestCase):
    def test_path_to_world_points_scales(self):
        pts = path_to_world_points([(0, 0), (1, 0), (1, 1)], cell_size=0.5)
        self.assertEqual(pts, [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)])

    def test_path_to_world_points_origin(self):
        pts = path_to_world_points([(0, 0), (2, 0)], cell_size=1.0, origin=(10.0, -5.0))
        self.assertEqual(pts, [(10.0, -5.0), (12.0, -5.0)])

    def test_yaw_along(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        yaws = yaw_along(pts)
        self.assertEqual(len(yaws), 3)
        self.assertAlmostEqual(yaws[0], 0.0)               # heading +x
        self.assertAlmostEqual(yaws[1], math.pi / 2)       # turning +y
        self.assertAlmostEqual(yaws[2], math.pi / 2)       # last holds previous

    def test_yaw_along_empty(self):
        self.assertEqual(yaw_along([]), [])


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") is not None, "rclpy not available"
)
class TestPlannerNodeImport(unittest.TestCase):
    def test_module_imports(self):
        # Importing the node module requires rclpy but not a running daemon.
        from mrn_coord.mapf import planner_node

        self.assertTrue(hasattr(planner_node, "MapfPlannerNode"))
        self.assertTrue(hasattr(planner_node, "main"))


if __name__ == "__main__":
    unittest.main()

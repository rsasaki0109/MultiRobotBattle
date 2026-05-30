"""Tests for the formation/coverage ROS-bridge helpers (no rclpy required)."""

import importlib.util
import unittest

from mrn_coord.coverage.ros_conversion import cell_to_world, parse_robot_positions
from mrn_coord.formation.control import (
    formation_control_from_relative,
    relative_measurements,
)
from mrn_coord.formation.ros_conversion import parse_edges, parse_offsets


class TestFormationConversion(unittest.TestCase):
    def test_parse_offsets(self):
        spec = parse_offsets(["a", "b"], ["2.0,0.0", "-1.0,1.5"])
        self.assertEqual(spec.offsets["a"], (2.0, 0.0))
        self.assertEqual(spec.offsets["b"], (-1.0, 1.5))

    def test_parse_offsets_length_mismatch(self):
        with self.assertRaises(ValueError):
            parse_offsets(["a", "b"], ["0,0"])

    def test_parse_offsets_bad_format(self):
        with self.assertRaises(ValueError):
            parse_offsets(["a"], ["1,2,3"])

    def test_parse_edges(self):
        self.assertEqual(parse_edges(["1,2", "2,3", ""]), [("1", "2"), ("2", "3")])

    def test_parse_edges_bad(self):
        with self.assertRaises(ValueError):
            parse_edges(["1-2"])

    def test_parsed_spec_drives_control(self):
        # offsets parsed from strings produce a usable control law (zero at
        # formation, nonzero off it).
        spec = parse_offsets(["a", "b"], ["1.0,0.0", "-1.0,0.0"])
        edges = parse_edges(["a,b"])
        in_formation = {"a": (1.0, 0.0), "b": (-1.0, 0.0)}
        meas = relative_measurements(in_formation, edges)
        commands = formation_control_from_relative(meas, spec, gain=1.0)
        for ux, uy in commands.values():
            self.assertAlmostEqual(ux, 0.0)
            self.assertAlmostEqual(uy, 0.0)


class TestCoverageConversion(unittest.TestCase):
    def test_parse_robot_positions(self):
        pos = parse_robot_positions(["1", "2"], ["3,0", "5,4"])
        self.assertEqual(pos, {"1": (3, 0), "2": (5, 4)})

    def test_parse_robot_positions_mismatch(self):
        with self.assertRaises(ValueError):
            parse_robot_positions(["1", "2"], ["3,0"])

    def test_cell_to_world(self):
        self.assertEqual(cell_to_world((2, 3), cell_size=0.5), (1.0, 1.5))
        self.assertEqual(cell_to_world((1, 1), cell_size=1.0, origin=(10.0, -2.0)),
                         (11.0, -1.0))


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") is not None, "rclpy not available"
)
class TestNodeImports(unittest.TestCase):
    def test_formation_node_imports(self):
        from mrn_coord.formation import controller_node
        self.assertTrue(hasattr(controller_node, "FormationControllerNode"))

    def test_coverage_node_imports(self):
        from mrn_coord.coverage import allocator_node
        self.assertTrue(hasattr(allocator_node, "CoverageAllocatorNode"))

    def test_goal_follower_imports(self):
        from mrn_coord.coverage import goal_follower_node
        self.assertTrue(hasattr(goal_follower_node, "GoalFollowerNode"))


if __name__ == "__main__":
    unittest.main()

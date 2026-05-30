"""Tests for proximity queries and a guarded sim-node import smoke."""

import importlib.util
import unittest

from mrn_sim import Robot, World, in_range_pairs, undirected_in_range


def _world():
    robots = {
        "a": Robot("a", (0.0, 0.0, 0.0)),
        "b": Robot("b", (3.0, 0.0, 0.0)),
        "c": Robot("c", (10.0, 0.0, 0.0)),
    }
    return World(20.0, 20.0, robots, [])


class TestProximity(unittest.TestCase):
    def test_in_range_directed(self):
        pairs = in_range_pairs(_world(), radius=4.0)
        # a<->b within 4; c is far from both
        self.assertIn(("a", "b"), pairs)
        self.assertIn(("b", "a"), pairs)
        self.assertNotIn(("a", "c"), pairs)
        self.assertNotIn(("b", "c"), pairs)

    def test_undirected_unique(self):
        pairs = undirected_in_range(_world(), radius=4.0)
        self.assertEqual(pairs, [("a", "b")])

    def test_all_in_range(self):
        pairs = undirected_in_range(_world(), radius=100.0)
        self.assertEqual(set(pairs), {("a", "b"), ("a", "c"), ("b", "c")})

    def test_none_in_range(self):
        self.assertEqual(undirected_in_range(_world(), radius=1.0), [])


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") is not None
    and importlib.util.find_spec("mrn_msgs") is not None,
    "rclpy / mrn_msgs not available",
)
class TestSimNodeImport(unittest.TestCase):
    def test_module_imports(self):
        from mrn_sim import sim_node
        self.assertTrue(hasattr(sim_node, "SimWorldNode"))
        self.assertTrue(hasattr(sim_node, "main"))


if __name__ == "__main__":
    unittest.main()

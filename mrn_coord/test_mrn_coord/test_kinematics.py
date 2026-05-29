"""Tests for the kinematic integrator and the agent-sim node import."""

import importlib.util
import math
import unittest

from mrn_coord.kinematics import euler_step


class TestEulerStep(unittest.TestCase):
    def test_basic_integration(self):
        self.assertEqual(euler_step((0.0, 0.0), (1.0, 2.0), 0.5), (0.5, 1.0))

    def test_zero_velocity_holds(self):
        self.assertEqual(euler_step((3.0, 4.0), (0.0, 0.0), 0.1), (3.0, 4.0))

    def test_speed_clamp(self):
        # velocity magnitude 10 clamped to max_speed 2 over dt=1
        new = euler_step((0.0, 0.0), (6.0, 8.0), 1.0, max_speed=2.0)
        self.assertAlmostEqual(math.hypot(*new), 2.0, places=9)
        # direction preserved
        self.assertAlmostEqual(new[1] / new[0], 8.0 / 6.0, places=9)

    def test_no_clamp_when_under_limit(self):
        new = euler_step((0.0, 0.0), (1.0, 0.0), 1.0, max_speed=5.0)
        self.assertEqual(new, (1.0, 0.0))

    def test_negative_dt_rejected(self):
        with self.assertRaises(ValueError):
            euler_step((0.0, 0.0), (1.0, 0.0), -0.1)


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") is not None, "rclpy not available"
)
class TestAgentSimImport(unittest.TestCase):
    def test_module_imports(self):
        from mrn_coord import agent_sim_node
        self.assertTrue(hasattr(agent_sim_node, "AgentSimNode"))


if __name__ == "__main__":
    unittest.main()

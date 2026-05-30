"""Import smoke for the estimate->coordination bridge node (rclpy-guarded)."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") is not None
    and importlib.util.find_spec("mrn_msgs") is not None,
    "rclpy / mrn_msgs not available",
)
class TestPoseBridgeImport(unittest.TestCase):
    def test_module_imports(self):
        from mrn_coord import pose_bridge_node
        self.assertTrue(hasattr(pose_bridge_node, "PoseBridgeNode"))
        self.assertTrue(hasattr(pose_bridge_node, "main"))


if __name__ == "__main__":
    unittest.main()

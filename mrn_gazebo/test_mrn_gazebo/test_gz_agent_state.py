"""Tests for the Gazebo pose -> AgentState builder (guarded on mrn_msgs)."""

import importlib.util
import math
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("mrn_msgs") is not None, "mrn_msgs not available"
)
class TestBuildAgentState(unittest.TestCase):
    def test_fields_and_stamp(self):
        from mrn_gazebo.gz_agent_state import build_agent_state

        msg = build_agent_state(
            "robot_1", 3.0, 4.0, math.pi / 2,
            frame_id="map", stamp_sec=12.5, ttl_sec=0.3,
        )
        self.assertEqual(msg.agent_id, "robot_1")
        self.assertEqual(msg.map_frame, "map")
        self.assertAlmostEqual(msg.pose.pose.position.x, 3.0)
        self.assertAlmostEqual(msg.pose.pose.position.y, 4.0)
        # yaw pi/2 -> quaternion z = sin(pi/4)
        self.assertAlmostEqual(msg.pose.pose.orientation.z, math.sin(math.pi / 4))
        # stamped with a TTL so freshness gates accept it
        self.assertEqual(msg.packet.header.stamp.sec, 12)
        self.assertEqual(msg.packet.ttl.sec, 0)
        self.assertEqual(msg.packet.ttl.nanosec, 300_000_000)
        self.assertEqual(msg.packet.sender_agent_id, "robot_1")

    def test_status_ok(self):
        from mrn_gazebo.gz_agent_state import build_agent_state
        from mrn_msgs.msg import AgentState

        msg = build_agent_state("a", 0.0, 0.0, 0.0)
        self.assertEqual(msg.status, AgentState.STATUS_OK)


if __name__ == "__main__":
    unittest.main()

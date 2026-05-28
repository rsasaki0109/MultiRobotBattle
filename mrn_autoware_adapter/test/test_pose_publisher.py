"""Pure-function tests for ``build_pose_message``.

The full publisher node spins up rclpy, but the message-construction
helper is pure and can be exercised against real geometry_msgs without a
running node.
"""

import unittest

from geometry_msgs.msg import PoseWithCovarianceStamped
from mrn_msgs.msg import CooperativePose

from mrn_autoware_adapter.pose_publisher_node import build_pose_message


def _cooperative_pose(
    *, sec: int = 5, nanosec: int = 250_000_000, x: float = 1.0, y: float = 2.0
) -> CooperativePose:
    msg = CooperativePose()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = "robot_1/base"
    msg.agent_id = "robot_1"
    msg.map_frame = "map"
    msg.odom_frame = "odom"
    msg.base_frame = "base"
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0
    msg.pose.pose.orientation.w = 1.0
    # Distinctive covariance pattern so we can verify it survives unchanged.
    covariance = list(msg.pose.covariance)
    covariance[0] = 0.04   # var(x)
    covariance[7] = 0.04   # var(y)
    covariance[35] = 0.01  # var(yaw)
    msg.pose.covariance = covariance
    msg.status = 0  # STATUS_OK
    msg.quality = 0.9
    return msg


class TestBuildPoseMessage(unittest.TestCase):
    def test_returns_pose_with_covariance_stamped(self):
        msg = _cooperative_pose()
        output = build_pose_message(msg, "map")
        self.assertIsInstance(output, PoseWithCovarianceStamped)

    def test_target_frame_is_used(self):
        msg = _cooperative_pose()
        output = build_pose_message(msg, "world")
        self.assertEqual(output.header.frame_id, "world")

    def test_stamp_is_forwarded(self):
        msg = _cooperative_pose(sec=10, nanosec=500_000_000)
        output = build_pose_message(msg, "map")
        self.assertEqual(output.header.stamp.sec, 10)
        self.assertEqual(output.header.stamp.nanosec, 500_000_000)

    def test_position_is_copied(self):
        msg = _cooperative_pose(x=3.5, y=4.5)
        output = build_pose_message(msg, "map")
        self.assertAlmostEqual(output.pose.pose.position.x, 3.5)
        self.assertAlmostEqual(output.pose.pose.position.y, 4.5)

    def test_covariance_is_copied_intact(self):
        msg = _cooperative_pose()
        output = build_pose_message(msg, "map")
        self.assertAlmostEqual(output.pose.covariance[0], 0.04)
        self.assertAlmostEqual(output.pose.covariance[7], 0.04)
        self.assertAlmostEqual(output.pose.covariance[35], 0.01)


if __name__ == "__main__":
    unittest.main()

"""``mrn_gz_pose_adapter``: Gazebo model poses -> AgentState.

Per agent, subscribes to the bridged Gazebo model pose (a
``geometry_msgs/PoseStamped`` produced by ``ros_gz_bridge`` from
``/model/<name>/pose``) and republishes it as ``/<id>/mrn/agent_state``
(``mrn_msgs/AgentState``) so the cooperative-localization stack consumes the
Gazebo world exactly as it consumes ``mrn_sim``. Optionally adds reproducible
GNSS-like noise. Thin shell over the pure :func:`build_agent_state`.
"""

from __future__ import annotations

import math
import random

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from mrn_gazebo.gz_agent_state import build_agent_state


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GzPoseAdapter(Node):
    def __init__(self) -> None:
        super().__init__("mrn_gz_pose_adapter")

        self.declare_parameter("agent_ids", ["robot_1"])
        self.declare_parameter("gz_pose_topic_template", "/model/{id}/pose")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("gnss_sigma", 0.0)

        self._agent_ids = [str(a) for a in self.get_parameter("agent_ids").value]
        src_tmpl = str(self.get_parameter("gz_pose_topic_template").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._sigma = float(self.get_parameter("gnss_sigma").value)
        self._rng = random.Random(0)

        from mrn_msgs.msg import AgentState
        self._pubs = {}
        for a in self._agent_ids:
            self._pubs[a] = self.create_publisher(
                AgentState, f"/{a}/mrn/agent_state", 10)
            self.create_subscription(
                PoseStamped, src_tmpl.format(id=a), self._make_cb(a), 10)
        self.get_logger().info(f"gz pose adapter: {len(self._agent_ids)} agents")

    def _make_cb(self, agent):
        def cb(msg: PoseStamped) -> None:
            now = self.get_clock().now().to_msg()
            x = msg.pose.position.x
            y = msg.pose.position.y
            if self._sigma > 0.0:
                x += self._rng.gauss(0.0, self._sigma)
                y += self._rng.gauss(0.0, self._sigma)
            state = build_agent_state(
                agent, x, y, _yaw(msg.pose.orientation),
                frame_id=self._frame_id,
                stamp_sec=now.sec + now.nanosec * 1e-9,
            )
            self._pubs[agent].publish(state)
        return cb


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GzPoseAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

"""``mrn_agent_sim``: a minimal kinematic agent simulator for closed-loop demos.

Holds a position per agent, publishes it as ``geometry_msgs/PoseStamped`` on
``formation/pose/<id>`` (the input the formation controller expects) plus a
``visualization_msgs/MarkerArray`` on ``coordination/markers`` for RViz, and
integrates ``geometry_msgs/Twist`` commands from ``formation/cmd_vel/<id>``.

Run together with ``mrn_formation_controller`` it closes the loop entirely
inside ROS: poses out, velocity commands back, integrate, repeat — the agents
converge into the commanded formation. It is a stand-in for the real plant /
localization estimate, deliberately simple (single integrator).
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from mrn_coord.kinematics import euler_step
from mrn_coord.mapf.ros_conversion import safe_topic_token

# A small fixed palette so each agent gets a stable color in RViz.
_PALETTE = [
    (0.22, 0.74, 0.93),  # sky
    (0.96, 0.45, 0.71),  # pink
    (0.64, 0.90, 0.21),  # lime
    (0.98, 0.62, 0.25),  # amber
]


def _parse_xy(text: str) -> tuple:
    parts = str(text).split(",")
    if len(parts) != 2:
        raise ValueError(f"expected 'x,y', got {text!r}")
    return (float(parts[0]), float(parts[1]))


class AgentSimNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_agent_sim")

        self.declare_parameter("agent_ids", ["1", "2", "3"])
        self.declare_parameter("initial_positions", ["0.0,0.0", "5.0,1.0", "1.0,4.0"])
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_speed", 3.0)

        self._agent_ids = [str(a) for a in self._param("agent_ids")]
        init = [str(p) for p in self._param("initial_positions")]
        if len(init) != len(self._agent_ids):
            raise ValueError("initial_positions must match agent_ids in length")
        self._pos = {a: _parse_xy(p) for a, p in zip(self._agent_ids, init)}
        self._vel = {a: (0.0, 0.0) for a in self._agent_ids}
        self._frame_id = str(self._param("frame_id"))
        self._max_speed = float(self._param("max_speed"))
        self._rate = float(self._param("rate_hz"))
        self._dt = 1.0 / max(self._rate, 1e-3)

        self._pose_pubs = {}
        for i, a in enumerate(self._agent_ids):
            token = safe_topic_token(a)
            self._pose_pubs[a] = self.create_publisher(
                PoseStamped, f"formation/pose/{token}", 10
            )
            self.create_subscription(
                Twist, f"formation/cmd_vel/{token}", self._make_cmd_cb(a), 10
            )
        self._marker_pub = self.create_publisher(MarkerArray, "coordination/markers", 10)
        self._timer = self.create_timer(self._dt, self._step)
        self.get_logger().info(
            f"agent sim: {len(self._agent_ids)} agents @ {self._rate} Hz"
        )

    def _param(self, name):
        return self.get_parameter(name).value

    def _make_cmd_cb(self, agent):
        def cb(msg: Twist) -> None:
            self._vel[agent] = (msg.linear.x, msg.linear.y)
        return cb

    def _step(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for a in self._agent_ids:
            self._pos[a] = euler_step(self._pos[a], self._vel[a], self._dt, self._max_speed)
        self._publish(stamp)

    def _publish(self, stamp) -> None:
        markers = MarkerArray()
        for i, a in enumerate(self._agent_ids):
            x, y = self._pos[a]
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            self._pose_pubs[a].publish(pose)

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self._frame_id
            m.ns = "agents"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.6
            r, g, b = _PALETTE[i % len(_PALETTE)]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            markers.markers.append(m)
        self._marker_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgentSimNode()
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

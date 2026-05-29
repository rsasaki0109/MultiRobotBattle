"""``mrn_formation_controller``: a thin ROS wrapper around the formation law.

Subscribes to each agent's current pose on ``formation/pose/<id>``
(``geometry_msgs/PoseStamped``), and on a timer computes the decentralized
displacement-based command for every agent from the relative positions of its
neighbors, publishing ``geometry_msgs/Twist`` on ``formation/cmd_vel/<id>``.

The node holds no control logic of its own — the relative-measurement law lives
in the pure, CI-tested :mod:`mrn_coord.formation.control`; this shell only wires
poses in and velocity commands out. In a live system the poses would come from
the cooperative-localization estimate (or the V2V relative-pose constraints
could feed the law directly).
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from mrn_coord.formation.control import (
    formation_control_from_relative,
    relative_measurements,
)
from mrn_coord.formation.ros_conversion import parse_edges, parse_offsets
from mrn_coord.mapf.ros_conversion import safe_topic_token


class FormationControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_formation_controller")

        self.declare_parameter("agent_ids", ["1", "2", "3"])
        self.declare_parameter("formation_offsets", ["2.0,0.0", "-1.0,1.7", "-1.0,-1.7"])
        self.declare_parameter("edges", ["1,2", "2,3", "1,3"])
        self.declare_parameter("gain", 1.0)
        self.declare_parameter("control_rate_hz", 10.0)

        self._agent_ids = [str(a) for a in self._param("agent_ids")]
        self._spec = parse_offsets(self._agent_ids, [str(s) for s in self._param("formation_offsets")])
        self._edges = parse_edges([str(e) for e in self._param("edges")])
        self._gain = float(self._param("gain"))

        self._positions: dict = {}
        self._cmd_pubs = {}
        self._pose_subs = []
        for a in self._agent_ids:
            token = safe_topic_token(a)
            self._cmd_pubs[a] = self.create_publisher(
                Twist, f"formation/cmd_vel/{token}", 10
            )
            self._pose_subs.append(self.create_subscription(
                PoseStamped, f"formation/pose/{token}",
                self._make_pose_cb(a), 10,
            ))

        rate = float(self._param("control_rate_hz"))
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._control_step)
        self.get_logger().info(
            f"formation controller: {len(self._agent_ids)} agents, "
            f"{len(self._edges)} edges, gain={self._gain}"
        )

    def _param(self, name):
        return self.get_parameter(name).value

    def _make_pose_cb(self, agent):
        def cb(msg: PoseStamped) -> None:
            self._positions[agent] = (msg.pose.position.x, msg.pose.position.y)
        return cb

    def _control_step(self) -> None:
        # Need a position for every agent that appears in an edge.
        needed = {a for e in self._edges for a in e}
        if not needed.issubset(self._positions):
            return
        meas = relative_measurements(self._positions, self._edges)
        commands = formation_control_from_relative(meas, self._spec, self._gain)
        for agent, (ux, uy) in commands.items():
            twist = Twist()
            twist.linear.x = float(ux)
            twist.linear.y = float(uy)
            self._cmd_pubs[agent].publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FormationControllerNode()
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

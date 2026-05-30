"""``mrn_path_follower``: drive unicycle robots along planned MAPF paths.

Per agent, subscribes to a ``nav_msgs/Path`` (e.g. ``mapf/path/<id>`` from
``mrn_mapf_planner``, latched) and the robot's pose
(``geometry_msgs/PoseStamped``), and on a timer publishes a
``geometry_msgs/Twist`` (``v = linear.x``, ``omega = angular.z``) that tracks the
path via pure pursuit. The tracking math is the pure, CI-tested
:func:`mrn_coord.mapf.path_follower.pure_pursuit`; this node is the thin shell.

Pair it with ``mrn_sim_world`` (pose in, cmd_vel out) to drive a MAPF plan
through the simulator — closing planning → world for a non-holonomic robot.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from mrn_coord.mapf.path_follower import pure_pursuit
from mrn_coord.mapf.ros_conversion import safe_topic_token


def _latched_qos() -> QoSProfile:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PathFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_path_follower")

        self.declare_parameter("agent_ids", ["1", "2", "3"])
        self.declare_parameter("path_topic_template", "mapf/path/{token}")
        self.declare_parameter("pose_topic_template", "/{token}/ground_truth/pose")
        self.declare_parameter("cmd_vel_topic_template", "/{token}/cmd_vel")
        self.declare_parameter("lookahead", 1.2)
        self.declare_parameter("v_nominal", 1.0)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("rate_hz", 20.0)

        self._agent_ids = [str(a) for a in self._param("agent_ids")]
        self._lookahead = float(self._param("lookahead"))
        self._v_nominal = float(self._param("v_nominal"))
        self._goal_tol = float(self._param("goal_tolerance"))
        path_tmpl = str(self._param("path_topic_template"))
        pose_tmpl = str(self._param("pose_topic_template"))
        cmd_tmpl = str(self._param("cmd_vel_topic_template"))

        self._paths: dict = {}
        self._poses: dict = {}
        self._cmd_pubs = {}
        for a in self._agent_ids:
            token = safe_topic_token(a)
            self._cmd_pubs[a] = self.create_publisher(
                Twist, cmd_tmpl.format(id=a, token=token), 10)
            self.create_subscription(
                Path, path_tmpl.format(id=a, token=token),
                self._make_path_cb(a), _latched_qos())
            self.create_subscription(
                PoseStamped, pose_tmpl.format(id=a, token=token),
                self._make_pose_cb(a), 10)

        rate = float(self._param("rate_hz"))
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._control_step)
        self.get_logger().info(f"path follower: {len(self._agent_ids)} agents")

    def _param(self, name):
        return self.get_parameter(name).value

    def _make_path_cb(self, agent):
        def cb(msg: Path) -> None:
            self._paths[agent] = [
                (p.pose.position.x, p.pose.position.y) for p in msg.poses
            ]
        return cb

    def _make_pose_cb(self, agent):
        def cb(msg: PoseStamped) -> None:
            self._poses[agent] = (
                msg.pose.position.x, msg.pose.position.y, _yaw(msg.pose.orientation)
            )
        return cb

    def _control_step(self) -> None:
        for a in self._agent_ids:
            path = self._paths.get(a)
            pose = self._poses.get(a)
            if not path or pose is None:
                continue
            v, omega, _ = pure_pursuit(
                pose, path, lookahead=self._lookahead,
                v_nominal=self._v_nominal, goal_tolerance=self._goal_tol,
            )
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(omega)
            self._cmd_pubs[a].publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathFollowerNode()
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

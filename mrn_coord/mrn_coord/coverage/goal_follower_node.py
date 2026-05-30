"""``mrn_goal_follower``: drive unicycle robots to coverage goal points.

Per agent, subscribes to a ``geometry_msgs/PointStamped`` goal (e.g.
``coverage/goal/<id>`` from ``mrn_coverage_allocator``, latched) and the robot's
``geometry_msgs/PoseStamped`` pose, and publishes a ``geometry_msgs/Twist`` that
drives toward the goal — reusing the pure-pursuit core with a single-point path.

Paired with ``mrn_sim_world`` it closes coverage → world: the allocator assigns
each robot a frontier, and the robot drives there. (Iterative re-mapping as
frontiers are reached is a separate, larger loop; this executes one allocation.)
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
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


class GoalFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_goal_follower")

        self.declare_parameter("agent_ids", ["robot_1", "robot_2"])
        self.declare_parameter("goal_topic_template", "coverage/goal/{token}")
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
        goal_tmpl = str(self._param("goal_topic_template"))
        pose_tmpl = str(self._param("pose_topic_template"))
        cmd_tmpl = str(self._param("cmd_vel_topic_template"))

        self._goals: dict = {}
        self._poses: dict = {}
        self._cmd_pubs = {}
        for a in self._agent_ids:
            token = safe_topic_token(a)
            self._cmd_pubs[a] = self.create_publisher(
                Twist, cmd_tmpl.format(id=a, token=token), 10)
            self.create_subscription(
                PointStamped, goal_tmpl.format(id=a, token=token),
                self._make_goal_cb(a), _latched_qos())
            self.create_subscription(
                PoseStamped, pose_tmpl.format(id=a, token=token),
                self._make_pose_cb(a), 10)

        rate = float(self._param("rate_hz"))
        self._timer = self.create_timer(1.0 / max(rate, 1e-3), self._control_step)
        self.get_logger().info(f"goal follower: {len(self._agent_ids)} agents")

    def _param(self, name):
        return self.get_parameter(name).value

    def _make_goal_cb(self, agent):
        def cb(msg: PointStamped) -> None:
            self._goals[agent] = (msg.point.x, msg.point.y)
        return cb

    def _make_pose_cb(self, agent):
        def cb(msg: PoseStamped) -> None:
            self._poses[agent] = (
                msg.pose.position.x, msg.pose.position.y, _yaw(msg.pose.orientation))
        return cb

    def _control_step(self) -> None:
        for a in self._agent_ids:
            goal = self._goals.get(a)
            pose = self._poses.get(a)
            if goal is None or pose is None:
                continue
            v, omega, _ = pure_pursuit(
                pose, [goal], lookahead=self._lookahead,
                v_nominal=self._v_nominal, goal_tolerance=self._goal_tol)
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(omega)
            self._cmd_pubs[a].publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalFollowerNode()
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

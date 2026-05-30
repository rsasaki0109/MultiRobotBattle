"""``mrn_sim_world``: the true 2D world as a ROS node.

Holds the world, integrates the per-robot ``geometry_msgs/Twist`` commands it
receives (``v = linear.x``, ``omega = angular.z``), and publishes the
measurements the rest of the stack consumes:

- ``/<id>/mrn/agent_state`` (``mrn_msgs/AgentState``) — the per-agent estimate
  (true pose plus reproducible GNSS-like noise), which the localization stack
  and the ``mrn_pose_bridge`` already consume.
- ``/<id>/ground_truth/pose`` (``geometry_msgs/PoseStamped``) — the noiseless
  truth, for evaluation.
- ``sim/markers`` (``visualization_msgs/MarkerArray``) — robots, obstacles, and
  in-range V2V links for RViz.

Because it both *emits* the localization messages and *accepts* velocity
commands, it closes the world → localization → coordination → world loop
entirely in ROS. The world model and proximity/sensor math are the pure,
CI-tested core; this node is the thin shell.
"""

from __future__ import annotations

import math
import random

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from mrn_msgs.msg import AgentState, RelativePoseConstraint
from mrn_sim.proximity import in_range_pairs, undirected_in_range
from mrn_sim.sensors import add_gaussian_noise, relative_pose_observation
from mrn_sim.v2v import build_relative_constraint
from mrn_sim.world import Obstacle, Robot, World, step

_PALETTE = [
    (0.22, 0.74, 0.93), (0.96, 0.45, 0.71),
    (0.64, 0.90, 0.21), (0.98, 0.62, 0.25),
]


def _safe_token(agent_id: str) -> str:
    s = str(agent_id)
    return s if s and (s[0].isalpha() or s[0] == "_") else "a_" + s


def _parse_pose(text: str):
    p = [float(v) for v in str(text).split(",")]
    if len(p) != 3:
        raise ValueError(f"expected 'x,y,theta', got {text!r}")
    return (p[0], p[1], p[2])


def _parse_obstacle(text: str) -> Obstacle:
    p = [float(v) for v in str(text).split(",")]
    if len(p) != 3:
        raise ValueError(f"expected 'x,y,r', got {text!r}")
    return Obstacle(p[0], p[1], p[2])


class SimWorldNode(Node):
    def __init__(self) -> None:
        super().__init__("mrn_sim_world")

        self.declare_parameter("agent_ids", ["robot_1", "robot_2", "robot_3"])
        self.declare_parameter("initial_poses", ["1.0,1.0,0.0", "11.0,1.0,3.14", "1.0,7.0,-1.57"])
        self.declare_parameter("obstacles", ["6.0,4.0,1.3"])
        self.declare_parameter("width", 12.0)
        self.declare_parameter("height", 8.0)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("gnss_sigma", 0.1)
        self.declare_parameter("sense_radius", 5.0)
        self.declare_parameter("max_speed", 2.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("cmd_vel_topic_template", "/{token}/cmd_vel")
        self.declare_parameter("publish_constraints", True)
        self.declare_parameter("v2v_xy_sigma", 0.1)
        self.declare_parameter("v2v_yaw_sigma", 0.05)
        self.declare_parameter("state_ttl_sec", 0.3)
        self.declare_parameter("degraded_agents", [""])
        self.declare_parameter("degraded_gnss_sigma", 1.2)

        agent_ids = [str(a) for a in self._param("agent_ids")]
        poses = [str(p) for p in self._param("initial_poses")]
        robots = {
            a: Robot(a, _parse_pose(p), 0.28)
            for a, p in zip(agent_ids, poses)
        }
        obstacles = [_parse_obstacle(o) for o in self._param("obstacles") if str(o)]
        self._world = World(
            float(self._param("width")), float(self._param("height")), robots, obstacles
        )
        self._agent_ids = agent_ids
        self._frame_id = str(self._param("frame_id"))
        self._gnss_sigma = float(self._param("gnss_sigma"))
        self._sense_radius = float(self._param("sense_radius"))
        self._max_speed = float(self._param("max_speed"))
        self._dt = 1.0 / max(float(self._param("rate_hz")), 1e-3)
        self._rng = random.Random(0)
        self._cmd = {a: (0.0, 0.0) for a in agent_ids}
        self._publish_constraints = bool(self._param("publish_constraints"))
        self._v2v_xy_sigma = float(self._param("v2v_xy_sigma"))
        self._v2v_yaw_sigma = float(self._param("v2v_yaw_sigma"))
        ttl = float(self._param("state_ttl_sec"))
        self._ttl = Duration(sec=int(ttl), nanosec=int(round((ttl - int(ttl)) * 1e9)))
        self._degraded = {str(a) for a in self._param("degraded_agents") if str(a)}
        self._degraded_sigma = float(self._param("degraded_gnss_sigma"))
        self._seq = 0

        cmd_tmpl = str(self._param("cmd_vel_topic_template"))
        self._agent_pubs = {}
        self._truth_pubs = {}
        self._constraint_pubs = {}
        for a in agent_ids:
            token = _safe_token(a)
            self._agent_pubs[a] = self.create_publisher(
                AgentState, f"/{token}/mrn/agent_state", 10)
            self._truth_pubs[a] = self.create_publisher(
                PoseStamped, f"/{token}/ground_truth/pose", 10)
            self._constraint_pubs[a] = self.create_publisher(
                RelativePoseConstraint, f"/{token}/mrn/relative_constraints", 10)
            self.create_subscription(
                Twist, cmd_tmpl.format(id=a, token=token), self._make_cmd_cb(a), 10
            )
        self._marker_pub = self.create_publisher(MarkerArray, "sim/markers", 10)
        self._timer = self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f"sim world: {len(agent_ids)} agents, {len(obstacles)} obstacle(s)"
        )

    def _param(self, name):
        return self.get_parameter(name).value

    def _make_cmd_cb(self, agent):
        def cb(msg: Twist) -> None:
            v = max(-self._max_speed, min(self._max_speed, msg.linear.x))
            self._cmd[agent] = (v, msg.angular.z)
        return cb

    def _tick(self) -> None:
        self._world = step(self._world, self._cmd, self._dt)
        stamp = self.get_clock().now().to_msg()
        for a in self._agent_ids:
            x, y, theta = self._world.robots[a].pose
            # ground truth
            gt = PoseStamped()
            gt.header.stamp = stamp
            gt.header.frame_id = self._frame_id
            gt.pose.position.x, gt.pose.position.y = x, y
            gt.pose.orientation.z = math.sin(0.5 * theta)
            gt.pose.orientation.w = math.cos(0.5 * theta)
            self._truth_pubs[a].publish(gt)
            # agent state estimate (truth + GNSS-like noise); a degraded agent
            # gets a much larger position error and a DEGRADED status, simulating
            # a GNSS outage that cooperative localization should rescue.
            degraded = a in self._degraded
            sigma = self._degraded_sigma if degraded else self._gnss_sigma
            est = AgentState()
            est.packet.header.stamp = stamp
            est.packet.header.frame_id = self._frame_id
            est.packet.sender_agent_id = a
            est.packet.measurement_time = stamp
            est.packet.source_publish_time = stamp
            est.packet.ttl = self._ttl
            est.agent_id = a
            est.map_frame = self._frame_id
            est.pose.pose.position.x = add_gaussian_noise(x, sigma, self._rng)
            est.pose.pose.position.y = add_gaussian_noise(y, sigma, self._rng)
            est.pose.pose.orientation.z = math.sin(0.5 * theta)
            est.pose.pose.orientation.w = math.cos(0.5 * theta)
            est.status = AgentState.STATUS_DEGRADED if degraded else AgentState.STATUS_OK
            est.quality = 0.45 if degraded else 0.9
            self._agent_pubs[a].publish(est)
        if self._publish_constraints:
            self._publish_v2v_constraints(stamp)
        self._publish_markers(stamp)

    def _publish_v2v_constraints(self, stamp) -> None:
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        for src, dst in in_range_pairs(self._world, self._sense_radius):
            x, y, yaw, cov = relative_pose_observation(
                self._world.robots[src].pose, self._world.robots[dst].pose,
                self._v2v_xy_sigma, self._v2v_yaw_sigma, self._rng,
            )
            self._seq += 1
            msg = build_relative_constraint(
                from_agent_id=src, to_agent_id=dst,
                from_frame=f"{src}/base_link", to_frame=f"{dst}/base_link",
                x=x, y=y, yaw=yaw, covariance=cov,
                stamp_sec=stamp_sec, sequence_id=self._seq, confidence=0.9,
            )
            self._constraint_pubs[src].publish(msg)

    def _publish_markers(self, stamp) -> None:
        arr = MarkerArray()
        for i, o in enumerate(self._world.obstacles):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, self._frame_id
            m.ns, m.id, m.type, m.action = "obstacles", i, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y = o.x, o.y
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2.0 * o.radius
            m.scale.z = 0.2
            m.color.r = m.color.g = m.color.b = 0.4
            m.color.a = 0.9
            arr.markers.append(m)
        for i, a in enumerate(self._agent_ids):
            x, y, _ = self._world.robots[a].pose
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, self._frame_id
            m.ns, m.id, m.type, m.action = "agents", i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y = x, y
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.56
            r, g, b = _PALETTE[i % len(_PALETTE)]
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            arr.markers.append(m)
        link = Marker()
        link.header.stamp, link.header.frame_id = stamp, self._frame_id
        link.ns, link.id, link.type, link.action = "v2v", 0, Marker.LINE_LIST, Marker.ADD
        link.scale.x = 0.04
        link.color.r = link.color.g = link.color.b = 0.9
        link.color.a = 0.5
        link.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        for ia, ib in undirected_in_range(self._world, self._sense_radius):
            pa, pb = self._world.robots[ia].pose, self._world.robots[ib].pose
            link.points.append(Point(x=pa[0], y=pa[1], z=0.0))
            link.points.append(Point(x=pb[0], y=pb[1], z=0.0))
        arr.markers.append(link)
        self._marker_pub.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimWorldNode()
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

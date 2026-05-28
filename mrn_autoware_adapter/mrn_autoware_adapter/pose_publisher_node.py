"""Cooperative pose to Autoware initialpose-style PoseWithCovarianceStamped.

Subscribes to ``/<agent_id>/mrn/cooperative_pose``, runs the SE(2) safety
gates from :mod:`mrn_autoware_adapter.correction_gate`, and (only when
the gate accepts) republishes the cooperative pose + covariance as a
``geometry_msgs/PoseWithCovarianceStamped`` on a configurable topic.

The default output topic is ``/<agent_id>/initialpose``. In a real
Autoware deployment, point the remap to whichever input the local
Autoware stack treats as a re-localization hypothesis (typically
``/initialpose`` or the ``pose_initializer_node`` input).

Optional periodic republish (``publish_rate_hz > 0``) re-emits the last
accepted pose at the configured rate, which is what tooling like RViz's
initialpose plug-in expects as a heartbeat. ``publish_rate_hz == 0``
makes the adapter emit one message per accepted cooperative pose, which
is the safer default in real Autoware deployments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from mrn_msgs.msg import CooperativePose

from mrn_autoware_adapter.correction_gate import (
    CorrectionGateConfig,
    CorrectionGateInput,
    CorrectionGateResult,
    CorrectionGateStatus,
    Pose2D,
    evaluate,
)


@dataclass
class _PoseCache:
    last_cooperative: Optional[Pose2D] = None
    last_accepted_message: Optional[PoseWithCovarianceStamped] = None


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def build_pose_message(
    msg: CooperativePose, target_frame: str
) -> PoseWithCovarianceStamped:
    """Convert a ``CooperativePose`` to ``PoseWithCovarianceStamped``.

    Pure-function so unit tests can verify the conversion without spinning
    a ROS node. The Autoware contract expects the pose in ``target_frame``
    (typically ``map``); the timestamp and covariance are forwarded
    untouched from the cooperative source.
    """
    output = PoseWithCovarianceStamped()
    output.header.stamp = msg.header.stamp
    output.header.frame_id = target_frame
    output.pose.pose = msg.pose.pose
    output.pose.covariance = msg.pose.covariance
    return output


class AutowarePosePublisher(Node):
    def __init__(self) -> None:
        super().__init__("mrn_autoware_pose_publisher")

        self.declare_parameter("agent_id", "robot_1")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("initialpose_topic", "")
        self.declare_parameter("max_pose_age_sec", 1.0)
        self.declare_parameter("max_translation_jump_m", 1.5)
        self.declare_parameter("max_rotation_jump_deg", 20.0)
        self.declare_parameter("accept_degraded", False)
        self.declare_parameter("publish_rate_hz", 0.0)

        self._agent_id = str(self.get_parameter("agent_id").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        explicit_topic = str(self.get_parameter("initialpose_topic").value).strip()
        self._initialpose_topic = (
            explicit_topic if explicit_topic else f"/{self._agent_id}/initialpose"
        )
        max_rotation_jump_rad = math.radians(
            float(self.get_parameter("max_rotation_jump_deg").value)
        )
        self._config = CorrectionGateConfig(
            max_pose_age_sec=float(self.get_parameter("max_pose_age_sec").value),
            max_translation_jump_m=float(
                self.get_parameter("max_translation_jump_m").value
            ),
            max_rotation_jump_rad=max_rotation_jump_rad,
            accept_degraded=bool(self.get_parameter("accept_degraded").value),
        )

        reliable_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        cooperative_topic = f"/{self._agent_id}/mrn/cooperative_pose"

        self._cache = _PoseCache()
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self._initialpose_topic, reliable_qos
        )

        self.create_subscription(
            CooperativePose,
            cooperative_topic,
            self._on_cooperative_pose,
            reliable_qos,
        )

        self._diag_pub = self.create_publisher(String, "~/diagnostics", reliable_qos)

        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate > 0.0:
            self.create_timer(1.0 / publish_rate, self._republish_last_accepted)

        self.get_logger().info(
            f"mrn_autoware_pose_publisher running for agent_id={self._agent_id}, "
            f"initialpose_topic={self._initialpose_topic}, "
            f"map_frame={self._map_frame}, publish_rate_hz={publish_rate}"
        )

    def _on_cooperative_pose(self, msg: CooperativePose) -> None:
        pose = msg.pose.pose
        candidate_pose = Pose2D(
            x=pose.position.x,
            y=pose.position.y,
            yaw=_yaw_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
        )
        now = self.get_clock().now().nanoseconds * 1e-9
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        candidate = CorrectionGateInput(
            stamp_sec=stamp_sec,
            now_sec=now,
            status=int(msg.status),
            pose=candidate_pose,
            previous_pose=self._cache.last_cooperative,
        )

        result = evaluate(candidate, self._config)
        self._publish_diagnostic(result)

        if not result.accepted:
            return

        self._cache.last_cooperative = candidate_pose
        output = build_pose_message(msg, self._map_frame)
        self._cache.last_accepted_message = output
        self._pose_pub.publish(output)

    def _republish_last_accepted(self) -> None:
        if self._cache.last_accepted_message is None:
            return
        last = self._cache.last_accepted_message
        # Re-emit with a current stamp so downstream stale gates see a
        # fresh message; pose and covariance are unchanged.
        republished = PoseWithCovarianceStamped()
        republished.header.stamp = self.get_clock().now().to_msg()
        republished.header.frame_id = last.header.frame_id
        republished.pose = last.pose
        self._pose_pub.publish(republished)

    def _publish_diagnostic(self, result: CorrectionGateResult) -> None:
        message = String()
        translation = (
            "nan"
            if result.translation_jump_m is None
            else f"{result.translation_jump_m:.3f}"
        )
        rotation = (
            "nan"
            if result.rotation_jump_rad is None
            else f"{math.degrees(result.rotation_jump_rad):.2f}"
        )
        message.data = (
            f"agent={self._agent_id} status={result.status.value} "
            f"age_sec={result.pose_age_sec:.3f} "
            f"translation_jump_m={translation} "
            f"rotation_jump_deg={rotation}"
        )
        self._diag_pub.publish(message)

        if result.status is CorrectionGateStatus.ACCEPT:
            return
        self.get_logger().warn(message.data)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutowarePosePublisher()
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

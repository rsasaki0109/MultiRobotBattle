"""Cooperative pose to Nav2 map->odom correction broadcaster.

Subscribes to ``/<agent_id>/mrn/cooperative_pose`` and ``/<agent_id>/odom``,
runs the SE(2) safety gates from :mod:`mrn_nav2_adapter.correction_gate`,
and (only when the gate accepts) publishes the resulting ``map -> odom``
transform via ``tf2_ros`` plus a diagnostic message on ``~/diagnostics``.

When the gate rejects a candidate, the last accepted transform is held.
That keeps Nav2 frame consistency while making the rejection visible in the
diagnostic stream so it can show up in reports the same way constraint-gate
rejections do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from mrn_msgs.msg import CooperativePose
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

from mrn_nav2_adapter.correction_gate import (
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
    last_odom: Optional[Pose2D] = None
    last_accepted_correction: Optional[TransformStamped] = None


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class CorrectionBroadcaster(Node):
    def __init__(self) -> None:
        super().__init__("mrn_nav2_correction_broadcaster")

        self.declare_parameter("agent_id", "robot_1")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("max_pose_age_sec", 1.0)
        self.declare_parameter("max_translation_jump_m", 1.5)
        self.declare_parameter("max_rotation_jump_deg", 20.0)
        self.declare_parameter("accept_degraded", False)
        self.declare_parameter("publish_rate_hz", 10.0)

        self._agent_id = str(self.get_parameter("agent_id").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._accept_degraded = bool(self.get_parameter("accept_degraded").value)
        max_rotation_jump_rad = math.radians(
            float(self.get_parameter("max_rotation_jump_deg").value)
        )
        self._config = CorrectionGateConfig(
            max_pose_age_sec=float(self.get_parameter("max_pose_age_sec").value),
            max_translation_jump_m=float(
                self.get_parameter("max_translation_jump_m").value
            ),
            max_rotation_jump_rad=max_rotation_jump_rad,
            accept_degraded=self._accept_degraded,
        )

        reliable_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        cooperative_topic = f"/{self._agent_id}/mrn/cooperative_pose"
        odom_topic = f"/{self._agent_id}/odom"

        self._cache = _PoseCache()
        self._broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            CooperativePose,
            cooperative_topic,
            self._on_cooperative_pose,
            reliable_qos,
        )
        self.create_subscription(Odometry, odom_topic, self._on_odom, reliable_qos)

        self._diag_pub = self.create_publisher(String, "~/diagnostics", reliable_qos)

        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / max(publish_rate, 0.1), self._republish_last_correction)

        self.get_logger().info(
            f"mrn_nav2_correction_broadcaster running for agent_id={self._agent_id}, "
            f"map_frame={self._map_frame}, odom_frame={self._odom_frame}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self._cache.last_odom = Pose2D(
            x=pose.position.x,
            y=pose.position.y,
            yaw=_yaw_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
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
        if self._cache.last_odom is None:
            return

        transform = self._compute_correction(candidate_pose, self._cache.last_odom, msg)
        self._cache.last_accepted_correction = transform
        self._broadcaster.sendTransform(transform)

    def _compute_correction(
        self,
        cooperative_pose: Pose2D,
        odom_pose: Pose2D,
        msg: CooperativePose,
    ) -> TransformStamped:
        # map_T_odom = map_T_base * (odom_T_base)^-1 in SE(2).
        dx = cooperative_pose.x - (
            odom_pose.x * math.cos(cooperative_pose.yaw - odom_pose.yaw)
            - odom_pose.y * math.sin(cooperative_pose.yaw - odom_pose.yaw)
        )
        dy = cooperative_pose.y - (
            odom_pose.x * math.sin(cooperative_pose.yaw - odom_pose.yaw)
            + odom_pose.y * math.cos(cooperative_pose.yaw - odom_pose.yaw)
        )
        dyaw = cooperative_pose.yaw - odom_pose.yaw

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self._map_frame
        transform.child_frame_id = f"{self._agent_id}/{self._odom_frame}"
        transform.transform.translation.x = dx
        transform.transform.translation.y = dy
        transform.transform.translation.z = 0.0
        qx, qy, qz, qw = _quaternion_from_yaw(dyaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        return transform

    def _republish_last_correction(self) -> None:
        if self._cache.last_accepted_correction is None:
            return
        last = self._cache.last_accepted_correction
        republished = TransformStamped()
        republished.header.stamp = self.get_clock().now().to_msg()
        republished.header.frame_id = last.header.frame_id
        republished.child_frame_id = last.child_frame_id
        republished.transform = last.transform
        self._broadcaster.sendTransform(republished)

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
    node = CorrectionBroadcaster()
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

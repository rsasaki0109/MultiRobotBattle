#!/usr/bin/env python3
"""Relative-anchor cooperative localization baseline.

This is intentionally not a full factor graph. It is a deterministic bridge
between the pass-through dummy backend and a future fixed-lag optimizer:
degraded agents are corrected from recent relative pose constraints to
non-degraded anchor agents.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import math
import time

from constraint_gate import validate_relative_pose_constraint
from geometry_msgs.msg import Point, PoseWithCovariance
from mrn_sync.time_gate import TimeGateConfig, duration_to_sec, validate_receive_age
import rclpy
from mrn_msgs.msg import (
    AgentState,
    ClockOffsetEstimate,
    ConstraintGraph,
    CooperativePose,
    RelativePoseConstraint,
)
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


Pose2 = tuple[float, float, float]


@dataclass
class AgentGraphState:
    state: AgentState | None = None
    received_monotonic_sec: float = 0.0
    accepted_constraints: int = 0
    rejected_constraints: int = 0
    cooperative_trail: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class StoredConstraint:
    message: RelativePoseConstraint
    received_monotonic_sec: float


@dataclass
class StoredClockOffset:
    message: ClockOffsetEstimate
    received_monotonic_sec: float


class RelativeAnchorGraphNode(Node):
    def __init__(self) -> None:
        super().__init__("relative_anchor_graph")
        self.declare_parameter("agent_ids", ["robot_1", "robot_2", "robot_3"])
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("stale_timeout_sec", 1.0)
        self.declare_parameter("max_constraint_age_sec", 2.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("use_clock_offset_gate", True)
        self.declare_parameter("reject_unknown_clock_offset", False)
        self.declare_parameter("max_clock_offset_sec", 0.05)
        self.declare_parameter("max_offset_uncertainty_sec", 0.01)
        self.declare_parameter("clock_status_timeout_sec", 2.0)

        self.agent_ids = [
            str(agent_id)
            for agent_id in self.get_parameter("agent_ids").get_parameter_value().string_array_value
        ]
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.stale_timeout_sec = float(self.get_parameter("stale_timeout_sec").value)
        self.max_constraint_age_sec = float(self.get_parameter("max_constraint_age_sec").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.use_clock_offset_gate = bool(self.get_parameter("use_clock_offset_gate").value)
        self.clock_status_timeout_sec = float(self.get_parameter("clock_status_timeout_sec").value)
        self.time_gate_config = TimeGateConfig(
            max_message_age_sec=None,
            max_clock_offset_sec=float(self.get_parameter("max_clock_offset_sec").value),
            max_offset_uncertainty_sec=float(
                self.get_parameter("max_offset_uncertainty_sec").value
            ),
            reject_if_unknown_offset=bool(
                self.get_parameter("reject_unknown_clock_offset").value
            ),
        )

        self.agent_states = {agent_id: AgentGraphState() for agent_id in self.agent_ids}
        self.constraints: dict[tuple[str, str], StoredConstraint] = {}
        self.clock_offsets: dict[tuple[str, str], StoredClockOffset] = {}
        self.accepted_constraints_total = 0
        self.rejected_constraints_total = 0
        self.rejection_reasons: Counter[str] = Counter()
        self.last_rejection_reason = ""
        self.last_corrected_count = 0

        self.odom_pubs = {
            agent_id: self.create_publisher(Odometry, f"/{agent_id}/mrn/cooperative_odom", 10)
            for agent_id in self.agent_ids
        }
        self.pose_pubs = {
            agent_id: self.create_publisher(
                CooperativePose, f"/{agent_id}/mrn/cooperative_pose", 10
            )
            for agent_id in self.agent_ids
        }
        self.graph_status_pub = self.create_publisher(ConstraintGraph, "/mrn/graph/status", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/mrn/graph/markers", 10)

        self._subscriptions = []
        for agent_id in self.agent_ids:
            self._subscriptions.append(
                self.create_subscription(
                    AgentState,
                    f"/{agent_id}/mrn/agent_state",
                    lambda msg, agent_id=agent_id: self._agent_state_callback(agent_id, msg),
                    10,
                )
            )
            self._subscriptions.append(
                self.create_subscription(
                    RelativePoseConstraint,
                    f"/{agent_id}/mrn/relative_constraints",
                    self._constraint_callback,
                    10,
                )
            )
            self._subscriptions.append(
                self.create_subscription(
                    ClockOffsetEstimate,
                    f"/{agent_id}/mrn/clock_status",
                    self._clock_status_callback,
                    10,
                )
            )

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_outputs)
        self.get_logger().info(
            "relative-anchor graph started: "
            f"agents={','.join(self.agent_ids)} max_constraint_age={self.max_constraint_age_sec:.2f}s"
        )

    def _agent_state_callback(self, agent_id: str, msg: AgentState) -> None:
        self.agent_states[agent_id].state = msg
        self.agent_states[agent_id].received_monotonic_sec = time.monotonic()

    def _clock_status_callback(self, msg: ClockOffsetEstimate) -> None:
        if not msg.local_agent_id or not msg.remote_agent_id:
            return
        self.clock_offsets[(msg.local_agent_id, msg.remote_agent_id)] = StoredClockOffset(
            message=msg,
            received_monotonic_sec=time.monotonic(),
        )

    def _constraint_callback(self, msg: RelativePoseConstraint) -> None:
        gate = validate_relative_pose_constraint(
            msg,
            self.agent_ids,
            clock_offset_estimate=self._clock_offset_for_constraint(msg),
            time_config=self.time_gate_config,
        )
        if gate.accepted:
            self.accepted_constraints_total += 1
            self.constraints[(msg.from_agent_id, msg.to_agent_id)] = StoredConstraint(
                message=msg,
                received_monotonic_sec=time.monotonic(),
            )
        else:
            self.rejected_constraints_total += 1
            self.rejection_reasons[gate.reason] += 1
            self.last_rejection_reason = gate.reason

        for agent_id in [msg.from_agent_id, msg.to_agent_id]:
            if agent_id not in self.agent_states:
                continue
            if gate.accepted:
                self.agent_states[agent_id].accepted_constraints += 1
            else:
                self.agent_states[agent_id].rejected_constraints += 1

    def _publish_outputs(self) -> None:
        now = self.get_clock().now()
        graph = ConstraintGraph()
        graph.header = Header(stamp=now.to_msg(), frame_id=self.map_frame)
        graph.graph_id = "relative_anchor_baseline"
        graph.backend_name = "relative_anchor"
        marker_array = MarkerArray()
        marker_id = 0
        self.last_corrected_count = 0

        for agent_id, runtime in self.agent_states.items():
            state = runtime.state
            if state is None:
                continue

            stale = self._state_is_stale(runtime)
            output_pose = deepcopy(state.pose)
            corrected = False
            if not stale and state.status == AgentState.STATUS_DEGRADED:
                candidate = self._cooperative_candidate(agent_id)
                if candidate is not None:
                    output_pose = candidate
                    corrected = True
                    self.last_corrected_count += 1

            status = self._output_status(state, stale, corrected)
            quality = self._output_quality(state, stale, corrected)
            self._publish_agent_output(agent_id, state, output_pose, status, quality, runtime)

            graph.agent_ids.append(agent_id)
            graph.agent_poses.append(output_pose)
            if stale:
                graph.stale_constraint_count += 1

            runtime.cooperative_trail.append(
                (output_pose.pose.position.x, output_pose.pose.position.y)
            )
            runtime.cooperative_trail = runtime.cooperative_trail[-180:]
            marker_array.markers.append(
                self._trail_marker(
                    marker_id,
                    now.to_msg(),
                    f"{agent_id}_cooperative",
                    runtime.cooperative_trail,
                    corrected,
                )
            )
            marker_id += 1
            if corrected:
                marker_array.markers.append(
                    self._text_marker(
                        marker_id,
                        now.to_msg(),
                        f"{agent_id}_corrected_text",
                        output_pose.pose.position.x,
                        output_pose.pose.position.y,
                        "COOP",
                    )
                )
                marker_id += 1

        graph.accepted_constraint_count = self.accepted_constraints_total
        graph.rejected_constraint_count = self.rejected_constraints_total
        self._fill_rejection_summary(graph)
        marker_array.markers.append(self._rejection_summary_marker(marker_id, now.to_msg()))
        self.graph_status_pub.publish(graph)
        self.marker_pub.publish(marker_array)

    def _cooperative_candidate(self, target_agent_id: str) -> PoseWithCovariance | None:
        candidates: list[tuple[Pose2, float]] = []
        now = time.monotonic()
        for stored in self.constraints.values():
            msg = stored.message
            age_gate = validate_receive_age(
                stored.received_monotonic_sec,
                now,
                duration_to_sec(msg.packet.ttl),
                max_age_sec=self.max_constraint_age_sec,
            )
            if not age_gate.accepted:
                continue
            if target_agent_id not in [msg.from_agent_id, msg.to_agent_id]:
                continue

            other_agent_id = (
                msg.to_agent_id if msg.from_agent_id == target_agent_id else msg.from_agent_id
            )
            other_runtime = self.agent_states.get(other_agent_id)
            if other_runtime is None or other_runtime.state is None:
                continue
            if self._state_is_stale(other_runtime):
                continue
            if other_runtime.state.status != AgentState.STATUS_OK:
                continue

            anchor_pose = self._pose_msg_to_pose2(other_runtime.state.pose)
            relative_pose = self._pose_msg_to_pose2(msg.relative_pose)
            if msg.from_agent_id == other_agent_id and msg.to_agent_id == target_agent_id:
                candidate = self._compose(anchor_pose, relative_pose)
            else:
                candidate = self._compose(anchor_pose, self._inverse(relative_pose))
            candidates.append((candidate, max(0.01, float(msg.confidence))))

        if not candidates:
            return None

        corrected = deepcopy(self.agent_states[target_agent_id].state.pose)
        averaged = self._weighted_average(candidates)
        corrected.pose.position.x = averaged[0]
        corrected.pose.position.y = averaged[1]
        corrected.pose.orientation.z = math.sin(averaged[2] / 2.0)
        corrected.pose.orientation.w = math.cos(averaged[2] / 2.0)
        corrected.covariance = self._corrected_covariance(corrected.covariance, len(candidates))
        return corrected

    def _publish_agent_output(
        self,
        agent_id: str,
        state: AgentState,
        output_pose: PoseWithCovariance,
        status: int,
        quality: float,
        runtime: AgentGraphState,
    ) -> None:
        coop_pose = CooperativePose()
        coop_pose.header = Header(stamp=state.packet.header.stamp, frame_id=state.map_frame)
        coop_pose.agent_id = agent_id
        coop_pose.map_frame = state.map_frame or self.map_frame
        coop_pose.odom_frame = state.odom_frame
        coop_pose.base_frame = state.base_frame
        coop_pose.pose = output_pose
        coop_pose.source_local_pose = state.pose
        coop_pose.twist = state.twist
        coop_pose.status = status
        coop_pose.accepted_constraints = runtime.accepted_constraints
        coop_pose.rejected_constraints = runtime.rejected_constraints
        coop_pose.quality = float(quality)
        self.pose_pubs[agent_id].publish(coop_pose)

        odom = Odometry()
        odom.header = coop_pose.header
        odom.child_frame_id = state.base_frame
        odom.pose = output_pose
        odom.twist = state.twist
        self.odom_pubs[agent_id].publish(odom)

    def _output_status(self, state: AgentState, stale: bool, corrected: bool) -> int:
        if stale:
            return CooperativePose.STATUS_STALE
        if corrected:
            return CooperativePose.STATUS_OK
        if state.status == AgentState.STATUS_OK:
            return CooperativePose.STATUS_OK
        if state.status == AgentState.STATUS_DEGRADED:
            return CooperativePose.STATUS_DEGRADED
        if state.status == AgentState.STATUS_STALE:
            return CooperativePose.STATUS_STALE
        return CooperativePose.STATUS_INVALID

    def _output_quality(self, state: AgentState, stale: bool, corrected: bool) -> float:
        if stale:
            return 0.0
        if corrected:
            return max(float(state.quality), 0.75)
        return float(state.quality)

    def _state_is_stale(self, runtime: AgentGraphState) -> bool:
        if runtime.state is None:
            return True
        if runtime.received_monotonic_sec <= 0.0:
            return True
        gate = validate_receive_age(
            runtime.received_monotonic_sec,
            time.monotonic(),
            duration_to_sec(runtime.state.packet.ttl),
            max_age_sec=self.stale_timeout_sec,
        )
        return not gate.accepted

    def _clock_offset_for_constraint(self, msg: RelativePoseConstraint):
        if not self.use_clock_offset_gate:
            return None
        keys = [
            (msg.from_agent_id, msg.to_agent_id),
            (msg.packet.sender_agent_id, msg.packet.receiver_agent_id),
            (msg.to_agent_id, msg.from_agent_id),
            (msg.packet.receiver_agent_id, msg.packet.sender_agent_id),
        ]
        now = time.monotonic()
        for key in keys:
            if not key[0] or not key[1]:
                continue
            stored = self.clock_offsets.get(key)
            if stored is None:
                continue
            if now - stored.received_monotonic_sec > self.clock_status_timeout_sec:
                continue
            return stored.message
        return None

    def _fill_rejection_summary(self, graph: ConstraintGraph) -> None:
        for reason, count in self.rejection_reasons.most_common():
            graph.rejection_reasons.append(reason)
            graph.rejection_reason_counts.append(int(count))
        graph.last_rejection_reason = self.last_rejection_reason

    def _rejection_summary_marker(self, marker_id: int, stamp) -> Marker:
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=self.map_frame)
        marker.ns = "rejection_summary"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = Point(x=-3.6, y=-3.2, z=1.2)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.24
        marker.color = self._rejection_color()
        marker.text = self._rejection_summary_text()
        marker.lifetime.sec = 1
        return marker

    def _rejection_rate(self) -> float:
        total = (
            self.accepted_constraints_total
            + self.rejected_constraints_total
        )
        if total == 0:
            return 0.0
        return self.rejected_constraints_total / total

    def _rejection_color(self) -> ColorRGBA:
        rate = self._rejection_rate()
        if rate <= 0.05:
            return ColorRGBA(r=0.10, g=0.95, b=0.30, a=0.95)
        if rate <= 0.25:
            return ColorRGBA(r=1.0, g=0.85, b=0.18, a=0.95)
        return ColorRGBA(r=1.0, g=0.24, b=0.16, a=0.95)

    def _rejection_summary_text(self) -> str:
        rate = self._rejection_rate()
        header_line = (
            f"accepted {self.accepted_constraints_total}  "
            f"rejected {self.rejected_constraints_total}  "
            f"rate {rate:.2f}"
        )
        if not self.rejection_reasons:
            return header_line + "\nrejects: none"
        items = [
            f"{reason}: {count}"
            for reason, count in self.rejection_reasons.most_common(4)
        ]
        return header_line + "\nrejects\n" + "\n".join(items)

    def _pose_msg_to_pose2(self, pose: PoseWithCovariance) -> Pose2:
        orientation = pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        return pose.pose.position.x, pose.pose.position.y, yaw

    def _compose(self, lhs: Pose2, rhs: Pose2) -> Pose2:
        x1, y1, yaw1 = lhs
        x2, y2, yaw2 = rhs
        cos_yaw = math.cos(yaw1)
        sin_yaw = math.sin(yaw1)
        return (
            x1 + cos_yaw * x2 - sin_yaw * y2,
            y1 + sin_yaw * x2 + cos_yaw * y2,
            self._normalize_angle(yaw1 + yaw2),
        )

    def _inverse(self, pose: Pose2) -> Pose2:
        x, y, yaw = pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            -cos_yaw * x - sin_yaw * y,
            sin_yaw * x - cos_yaw * y,
            self._normalize_angle(-yaw),
        )

    def _weighted_average(self, candidates: list[tuple[Pose2, float]]) -> Pose2:
        total_weight = sum(weight for _, weight in candidates)
        x = sum(pose[0] * weight for pose, weight in candidates) / total_weight
        y = sum(pose[1] * weight for pose, weight in candidates) / total_weight
        sin_sum = sum(math.sin(pose[2]) * weight for pose, weight in candidates)
        cos_sum = sum(math.cos(pose[2]) * weight for pose, weight in candidates)
        yaw = math.atan2(sin_sum, cos_sum)
        return x, y, yaw

    def _corrected_covariance(self, source: list[float], candidate_count: int) -> list[float]:
        covariance = list(source)
        xy_var = 0.10 if candidate_count == 1 else 0.06
        covariance[0] = min(max(covariance[0], 0.02), xy_var)
        covariance[7] = min(max(covariance[7], 0.02), xy_var)
        covariance[35] = min(max(covariance[35], 0.01), 0.04)
        return covariance

    def _trail_marker(
        self,
        marker_id: int,
        stamp,
        ns: str,
        trail: list[tuple[float, float]],
        corrected: bool,
    ) -> Marker:
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=self.map_frame)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.09 if corrected else 0.06
        marker.color = (
            ColorRGBA(r=0.05, g=0.95, b=0.25, a=0.95)
            if corrected
            else ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.65)
        )
        marker.lifetime.sec = 1
        for x, y in trail:
            marker.points.append(Point(x=x, y=y, z=0.16))
        return marker

    def _text_marker(self, marker_id: int, stamp, ns: str, x: float, y: float, text: str) -> Marker:
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=self.map_frame)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.28
        marker.color = ColorRGBA(r=0.05, g=1.0, b=0.25, a=1.0)
        marker.text = text
        marker.lifetime.sec = 1
        return marker

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main() -> None:
    rclpy.init()
    node = RelativeAnchorGraphNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

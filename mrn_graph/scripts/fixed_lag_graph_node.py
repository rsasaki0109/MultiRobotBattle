#!/usr/bin/env python3
"""Fixed-lag cooperative localization graph node (pure-Python backend).

A thin rclpy shell over :class:`graph_backend.FixedLagBackend`: it ingests
the same topics as ``relative_anchor_graph_node`` through the same
constraint gate and time gate, converts the current window of agent states
and accepted constraints into the backend's plain dataclasses, runs the
Gauss-Newton optimizer, and publishes the standard cooperative-pose /
cooperative-odom / graph-status / marker topics.

This keeps the optimization and diagnostics logic in the unit-tested
``graph_backend`` / ``pose_graph_solver`` / ``factor_graph`` modules; the
node only does ROS message conversion. It is an opt-in backend
(``graph_executable:=fixed_lag_graph_node.py``); the default remains
``relative_anchor`` so the CI smoke path is unchanged.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

from constraint_gate import validate_relative_pose_constraint
from geometry_msgs.msg import Point, PoseWithCovariance
from graph_backend import (
    AgentInput,
    FixedLagBackend,
    FixedLagBackendConfig,
    RelativeInput,
)
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
_FLOOR_VAR = 1e-4


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


def _yaw_from_quaternion(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


def _pose2_from_msg(pose: PoseWithCovariance) -> Pose2:
    return (
        pose.pose.position.x,
        pose.pose.position.y,
        _yaw_from_quaternion(pose.pose.orientation),
    )


def _diag_cov3_from_msg(covariance) -> list[list[float]]:
    """Extract a positive-definite diagonal 3x3 (x, y, yaw) from a 6x6 covariance.

    The node uses the diagonal of the reported covariance; non-finite or
    non-positive entries fall back to a small floor so the backend's
    factor validity check always passes for a present agent/constraint.
    """

    def _pos(value: float) -> float:
        return value if math.isfinite(value) and value > 0.0 else _FLOOR_VAR

    return [
        [_pos(covariance[0]), 0.0, 0.0],
        [0.0, _pos(covariance[7]), 0.0],
        [0.0, 0.0, _pos(covariance[35])],
    ]


class FixedLagGraphNode(Node):
    def __init__(self) -> None:
        super().__init__("fixed_lag_graph")
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
        self.declare_parameter("huber_delta", 1.0)

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
        self.backend = FixedLagBackend(
            FixedLagBackendConfig(
                max_constraint_age_sec=self.max_constraint_age_sec,
                huber_delta=float(self.get_parameter("huber_delta").value),
            )
        )

        self.agent_states = {agent_id: AgentGraphState() for agent_id in self.agent_ids}
        self.constraints: dict[tuple[str, str], StoredConstraint] = {}
        self.clock_offsets: dict[tuple[str, str], StoredClockOffset] = {}
        self.accepted_constraints_total = 0
        self.rejected_constraints_total = 0
        self.rejection_reasons: Counter[str] = Counter()
        self.last_rejection_reason = ""

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
            "fixed-lag graph started: "
            f"agents={','.join(self.agent_ids)} backend={self.backend.name}"
        )

    # --- ingest (same gating contract as relative_anchor) ---

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

    # --- solve + publish ---

    def _publish_outputs(self) -> None:
        now = self.get_clock().now()
        agents: list[AgentInput] = []
        stale_flags: dict[str, bool] = {}
        for agent_id, runtime in self.agent_states.items():
            state = runtime.state
            if state is None:
                continue
            stale = self._state_is_stale(runtime)
            stale_flags[agent_id] = stale
            degraded = stale or state.status != AgentState.STATUS_OK
            agents.append(
                AgentInput(
                    agent_id=agent_id,
                    pose=_pose2_from_msg(state.pose),
                    covariance=_diag_cov3_from_msg(state.pose.covariance),
                    degraded=degraded,
                )
            )

        relatives = self._active_relatives()
        graph = ConstraintGraph()
        graph.header = Header(stamp=now.to_msg(), frame_id=self.map_frame)
        graph.graph_id = "fixed_lag_python"
        graph.backend_name = self.backend.name

        if not agents:
            graph.accepted_constraint_count = self.accepted_constraints_total
            graph.rejected_constraint_count = self.rejected_constraints_total
            self._fill_rejection_summary(graph)
            self.graph_status_pub.publish(graph)
            return

        estimates, _diag = self.backend.step(agents, relatives)
        estimate_by_id = {e.agent_id: e for e in estimates}

        marker_array = MarkerArray()
        marker_id = 0
        for agent_id, runtime in self.agent_states.items():
            state = runtime.state
            if state is None or agent_id not in estimate_by_id:
                continue
            estimate = estimate_by_id[agent_id]
            stale = stale_flags[agent_id]
            corrected = (not stale) and state.status != AgentState.STATUS_OK and (
                estimate.accepted_constraints > 0
            )
            output_pose = self._estimate_to_pose_msg(estimate, state.pose, corrected)
            status = self._output_status(state, stale, corrected)
            quality = 0.0 if stale else estimate.quality
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
                    marker_id, now.to_msg(), f"{agent_id}_cooperative",
                    runtime.cooperative_trail, corrected,
                )
            )
            marker_id += 1

        graph.accepted_constraint_count = self.accepted_constraints_total
        graph.rejected_constraint_count = self.rejected_constraints_total
        self._fill_rejection_summary(graph)
        marker_array.markers.append(self._rejection_summary_marker(marker_id, now.to_msg()))
        self.graph_status_pub.publish(graph)
        self.marker_pub.publish(marker_array)

    def _active_relatives(self) -> list[RelativeInput]:
        now = time.monotonic()
        relatives: list[RelativeInput] = []
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
            if msg.from_agent_id not in self.agent_states:
                continue
            if msg.to_agent_id not in self.agent_states:
                continue
            relatives.append(
                RelativeInput(
                    from_id=msg.from_agent_id,
                    to_id=msg.to_agent_id,
                    measured=_pose2_from_msg(msg.relative_pose),
                    covariance=_diag_cov3_from_msg(msg.relative_pose.covariance),
                    age_sec=now - stored.received_monotonic_sec,
                )
            )
        return relatives

    def _estimate_to_pose_msg(self, estimate, source_pose, corrected) -> PoseWithCovariance:
        from copy import deepcopy

        output = deepcopy(source_pose)
        x, y, yaw = estimate.pose
        output.pose.position.x = x
        output.pose.position.y = y
        output.pose.orientation.z = math.sin(yaw / 2.0)
        output.pose.orientation.w = math.cos(yaw / 2.0)
        if corrected:
            covariance = list(output.covariance)
            covariance[0] = min(max(covariance[0], 0.02), 0.10)
            covariance[7] = min(max(covariance[7], 0.02), 0.10)
            covariance[35] = min(max(covariance[35], 0.01), 0.04)
            output.covariance = covariance
        return output

    def _publish_agent_output(self, agent_id, state, output_pose, status, quality, runtime) -> None:
        coop_pose = CooperativePose()
        coop_pose.header = Header(stamp=state.packet.header.stamp, frame_id=state.map_frame or self.map_frame)
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

    def _output_status(self, state, stale, corrected) -> int:
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

    def _state_is_stale(self, runtime: AgentGraphState) -> bool:
        if runtime.state is None or runtime.received_monotonic_sec <= 0.0:
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

    def _rejection_rate(self) -> float:
        total = self.accepted_constraints_total + self.rejected_constraints_total
        return self.rejected_constraints_total / total if total else 0.0

    def _rejection_color(self) -> ColorRGBA:
        rate = self._rejection_rate()
        if rate <= 0.05:
            return ColorRGBA(r=0.10, g=0.95, b=0.30, a=0.95)
        if rate <= 0.25:
            return ColorRGBA(r=1.0, g=0.85, b=0.18, a=0.95)
        return ColorRGBA(r=1.0, g=0.24, b=0.16, a=0.95)

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
        rate = self._rejection_rate()
        marker.text = (
            f"[fixed_lag] accepted {self.accepted_constraints_total}  "
            f"rejected {self.rejected_constraints_total}  rate {rate:.2f}"
        )
        marker.lifetime.sec = 1
        return marker

    def _trail_marker(self, marker_id, stamp, ns, trail, corrected) -> Marker:
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=self.map_frame)
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.09 if corrected else 0.06
        marker.color = (
            ColorRGBA(r=0.05, g=0.55, b=0.95, a=0.95)
            if corrected
            else ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.65)
        )
        marker.lifetime.sec = 1
        for x, y in trail:
            marker.points.append(Point(x=x, y=y, z=0.16))
        return marker


def main() -> None:
    rclpy.init()
    node = FixedLagGraphNode()
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

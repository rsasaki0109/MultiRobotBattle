#!/usr/bin/env python3
"""Pass-through cooperative graph backend for early demos."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time

from constraint_gate import validate_relative_pose_constraint
from geometry_msgs.msg import Point
import rclpy
from mrn_msgs.msg import (
    AgentState,
    ClockOffsetEstimate,
    ConstraintGraph,
    CooperativePose,
    RelativePoseConstraint,
)
from mrn_sync.time_gate import TimeGateConfig, duration_to_sec, validate_receive_age
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class AgentGraphState:
    state: AgentState | None = None
    received_monotonic_sec: float = 0.0
    accepted_constraints: int = 0
    rejected_constraints: int = 0


@dataclass
class StoredClockOffset:
    message: ClockOffsetEstimate
    received_monotonic_sec: float


class DummyGraphNode(Node):
    """Publish cooperative outputs using local AgentState as the initial baseline."""

    def __init__(self) -> None:
        super().__init__("dummy_graph")
        self.declare_parameter("agent_ids", ["robot_1", "robot_2", "robot_3"])
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("stale_timeout_sec", 1.0)
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
        self.publish_rate_hz = self.get_parameter("publish_rate_hz").value
        self.stale_timeout_sec = self.get_parameter("stale_timeout_sec").value
        self.map_frame = self.get_parameter("map_frame").value
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

        self.agent_states = {
            agent_id: AgentGraphState()
            for agent_id in self.agent_ids
        }
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
            "dummy graph started: "
            f"agents={','.join(self.agent_ids)} publish_rate={self.publish_rate_hz:.1f}Hz"
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
        graph.graph_id = "dummy_pass_through"
        graph.backend_name = "dummy"
        marker_array = MarkerArray()

        for agent_id, runtime in self.agent_states.items():
            state = runtime.state
            if state is None:
                continue

            stale = self._state_is_stale(runtime)
            status = CooperativePose.STATUS_STALE if stale else self._map_status(state.status)

            coop_pose = CooperativePose()
            coop_pose.header = Header(stamp=state.packet.header.stamp, frame_id=state.map_frame)
            coop_pose.agent_id = agent_id
            coop_pose.map_frame = state.map_frame or self.map_frame
            coop_pose.odom_frame = state.odom_frame
            coop_pose.base_frame = state.base_frame
            coop_pose.pose = state.pose
            coop_pose.source_local_pose = state.pose
            coop_pose.twist = state.twist
            coop_pose.status = status
            coop_pose.accepted_constraints = runtime.accepted_constraints
            coop_pose.rejected_constraints = runtime.rejected_constraints
            coop_pose.quality = 0.0 if stale else state.quality
            self.pose_pubs[agent_id].publish(coop_pose)

            odom = Odometry()
            odom.header = coop_pose.header
            odom.child_frame_id = state.base_frame
            odom.pose = state.pose
            odom.twist = state.twist
            self.odom_pubs[agent_id].publish(odom)

            graph.agent_ids.append(agent_id)
            graph.agent_poses.append(state.pose)

        graph.accepted_constraint_count = self.accepted_constraints_total
        graph.rejected_constraint_count = self.rejected_constraints_total
        self._fill_rejection_summary(graph)
        self.graph_status_pub.publish(graph)
        marker_array.markers.append(self._rejection_summary_marker(0, now.to_msg()))
        self.marker_pub.publish(marker_array)

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

    def _map_status(self, status: int) -> int:
        if status == AgentState.STATUS_OK:
            return CooperativePose.STATUS_OK
        if status == AgentState.STATUS_DEGRADED:
            return CooperativePose.STATUS_DEGRADED
        if status == AgentState.STATUS_STALE:
            return CooperativePose.STATUS_STALE
        return CooperativePose.STATUS_INVALID


def main() -> None:
    rclpy.init()
    node = DummyGraphNode()
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

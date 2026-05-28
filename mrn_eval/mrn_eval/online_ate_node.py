#!/usr/bin/env python3
"""Online ATE evaluation for synthetic cooperative localization demos."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from mrn_eval.metrics import StreamingAte
from mrn_msgs.msg import AgentState, CooperativePose, EvaluationSummary
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from std_msgs.msg import Header


Point2 = tuple[float, float]


@dataclass
class MethodTrack:
    metric: StreamingAte
    samples_total: int = 0
    last_error: float = math.nan
    last_update_monotonic_sec: float = 0.0


@dataclass
class AgentEvalState:
    truth: Point2 | None = None
    truth_monotonic_sec: float = 0.0
    local: MethodTrack | None = None
    cooperative: MethodTrack | None = None


class OnlineAteNode(Node):
    def __init__(self) -> None:
        super().__init__("online_ate")
        self.declare_parameter("agent_ids", ["robot_1", "robot_2", "robot_3"])
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("max_samples", 2000)
        self.declare_parameter("max_truth_age_sec", 0.25)
        self.declare_parameter("experiment_name", "synthetic_online_ate")

        self.agent_ids = [
            str(agent_id)
            for agent_id in self.get_parameter("agent_ids").get_parameter_value().string_array_value
        ]
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.max_samples = int(self.get_parameter("max_samples").value)
        self.max_truth_age_sec = float(self.get_parameter("max_truth_age_sec").value)
        self.experiment_name = str(self.get_parameter("experiment_name").value)

        self.agent_states = {
            agent_id: AgentEvalState(
                local=MethodTrack(StreamingAte(self.max_samples)),
                cooperative=MethodTrack(StreamingAte(self.max_samples)),
            )
            for agent_id in self.agent_ids
        }

        self.summary_pub = self.create_publisher(EvaluationSummary, "/mrn/eval/summary", 10)
        self._subscriptions = []
        for agent_id in self.agent_ids:
            self._subscriptions.append(
                self.create_subscription(
                    PoseWithCovarianceStamped,
                    f"/{agent_id}/ground_truth/pose",
                    lambda msg, agent_id=agent_id: self._truth_callback(agent_id, msg),
                    10,
                )
            )
            self._subscriptions.append(
                self.create_subscription(
                    AgentState,
                    f"/{agent_id}/mrn/agent_state",
                    lambda msg, agent_id=agent_id: self._local_callback(agent_id, msg),
                    10,
                )
            )
            self._subscriptions.append(
                self.create_subscription(
                    CooperativePose,
                    f"/{agent_id}/mrn/cooperative_pose",
                    lambda msg, agent_id=agent_id: self._cooperative_callback(agent_id, msg),
                    10,
                )
            )

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_summaries)
        self.get_logger().info(
            "online ATE evaluator started: "
            f"agents={','.join(self.agent_ids)} max_samples={self.max_samples}"
        )

    def _truth_callback(self, agent_id: str, msg: PoseWithCovarianceStamped) -> None:
        state = self.agent_states[agent_id]
        state.truth = self._pose_to_point(msg.pose)
        state.truth_monotonic_sec = time.monotonic()

    def _local_callback(self, agent_id: str, msg: AgentState) -> None:
        self._push_estimate(agent_id, "local_only", self._pose_to_point(msg.pose))

    def _cooperative_callback(self, agent_id: str, msg: CooperativePose) -> None:
        self._push_estimate(agent_id, "cooperative", self._pose_to_point(msg.pose))

    def _push_estimate(self, agent_id: str, method: str, estimate: Point2) -> None:
        state = self.agent_states[agent_id]
        if state.truth is None:
            return
        if time.monotonic() - state.truth_monotonic_sec > self.max_truth_age_sec:
            return

        track = state.local if method == "local_only" else state.cooperative
        track.metric.push(estimate, state.truth)
        track.samples_total += 1
        track.last_error = math.dist(estimate, state.truth)
        track.last_update_monotonic_sec = time.monotonic()

    def _publish_summaries(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for agent_id, state in self.agent_states.items():
            self._publish_summary(agent_id, "local_only", state.local, stamp)
            self._publish_summary(agent_id, "cooperative", state.cooperative, stamp)

    def _publish_summary(self, agent_id: str, method: str, track: MethodTrack, stamp) -> None:
        if track.metric.count == 0:
            return

        msg = EvaluationSummary()
        msg.header = Header(stamp=stamp, frame_id="map")
        msg.experiment_name = self.experiment_name
        msg.method_name = f"{agent_id}/{method}"
        msg.ate_rmse = track.metric.rmse()
        msg.rpe_rmse = math.nan
        msg.heading_rmse = math.nan
        msg.nees_mean = math.nan
        msg.nis_mean = math.nan
        msg.localization_availability = 1.0
        msg.packet_loss_rate = math.nan
        msg.stale_ratio = math.nan
        msg.useful_constraint_ratio = math.nan
        msg.graph_solve_time_ms = 0.0
        msg.dropped_factor_count = 0
        self.summary_pub.publish(msg)

    def _pose_to_point(self, pose_with_covariance) -> Point2:
        return (
            float(pose_with_covariance.pose.position.x),
            float(pose_with_covariance.pose.position.y),
        )


def main() -> None:
    rclpy.init()
    node = OnlineAteNode()
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

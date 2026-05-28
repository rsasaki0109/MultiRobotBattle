#!/usr/bin/env python3
"""Synthetic three-robot cooperative localization demo publisher."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import PoseWithCovarianceStamped
from mrn_msgs.msg import (
    AgentState,
    ClockOffsetEstimate,
    CommStatus,
    RelativePoseConstraint,
    V2VPacketHeader,
)
from mrn_gnss import FixQualitySchedule
from mrn_netem.profile import load_network_profile
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
import yaml


@dataclass
class AgentConfig:
    agent_id: str
    odom_frame: str
    base_frame: str
    phase: float
    clock_offset_sec: float = 0.0


@dataclass
class AgentRuntime:
    local_drift_x: float = 0.0
    local_drift_y: float = 0.0
    last_truth: tuple[float, float, float] | None = None
    truth_trail: list[tuple[float, float]] = field(default_factory=list)
    local_trail: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PendingConstraint:
    release_time_sec: float
    publisher_agent_id: str
    message: RelativePoseConstraint
    latency_sec: float


class SyntheticWorldNode(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_world")
        self.declare_parameter("scenario_path", "")
        self.declare_parameter("network_profile_path", "")
        scenario_path = self.get_parameter("scenario_path").get_parameter_value().string_value
        network_profile_path = (
            self.get_parameter("network_profile_path").get_parameter_value().string_value
        )
        self.config = self._load_config(scenario_path)
        self._apply_network_profile(network_profile_path)

        scenario = self.config.get("scenario", {})
        faults = self.config.get("faults", {})
        self.seed = int(scenario.get("seed", 42))
        self.rng = random.Random(self.seed)
        self.update_rate_hz = float(scenario.get("update_rate_hz", 20.0))
        self.constraint_rate_hz = float(scenario.get("constraint_rate_hz", 5.0))
        self.marker_rate_hz = float(scenario.get("marker_rate_hz", 5.0))
        self.packet_loss = float(faults.get("packet_loss_percent", 0.0)) / 100.0
        self.latency_mean_sec = float(faults.get("latency_ms_mean", 0.0)) / 1000.0
        self.latency_stddev_sec = float(faults.get("latency_ms_stddev", 15.0)) / 1000.0
        self.clock_drift_sec = float(faults.get("clock_drift_ms", 0.0)) / 1000.0
        self.qos_profile_name = str(faults.get("qos_profile_name", "relative_constraint"))
        self.gnss_outage = faults.get("gnss_outage", {})
        self.gnss_quality_schedules = self._load_quality_schedules(
            faults.get("gnss_quality_schedule", {})
        )

        self.agents = self._load_agents(self.config.get("agents", []))
        self.agent_state = {agent.agent_id: AgentRuntime() for agent in self.agents}
        self.pending_constraints: list[PendingConstraint] = []
        self.sequence = 0
        self.last_constraint_tick = -1
        self.last_marker_tick = -1
        self.link_stats: dict[tuple[str, str], dict[str, float]] = {}

        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/mrn/viz/markers", 10)
        self.agent_pubs = {
            agent.agent_id: self.create_publisher(
                AgentState, f"/{agent.agent_id}/mrn/agent_state", 10
            )
            for agent in self.agents
        }
        self.odom_pubs = {
            agent.agent_id: self.create_publisher(
                Odometry, f"/{agent.agent_id}/local/odometry", 10
            )
            for agent in self.agents
        }
        self.truth_pubs = {
            agent.agent_id: self.create_publisher(
                PoseWithCovarianceStamped, f"/{agent.agent_id}/ground_truth/pose", 10
            )
            for agent in self.agents
        }
        self.gnss_pubs = {
            agent.agent_id: self.create_publisher(
                PoseWithCovarianceStamped, f"/{agent.agent_id}/local/gnss_pose", 10
            )
            for agent in self.agents
        }
        self.constraint_pubs = {
            agent.agent_id: self.create_publisher(
                RelativePoseConstraint,
                f"/{agent.agent_id}/mrn/relative_constraints",
                10,
            )
            for agent in self.agents
        }
        self.comm_pubs = {
            agent.agent_id: self.create_publisher(
                CommStatus, f"/{agent.agent_id}/mrn/comm_status", 10
            )
            for agent in self.agents
        }
        self.clock_status_pubs = {
            agent.agent_id: self.create_publisher(
                ClockOffsetEstimate, f"/{agent.agent_id}/mrn/clock_status", 10
            )
            for agent in self.agents
        }

        self.start_ros_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.update_rate_hz, self._on_timer)
        self.get_logger().info(
            "synthetic world started: "
            f"agents={','.join(agent.agent_id for agent in self.agents)} "
            f"loss={self.packet_loss:.2f} latency_mean={self.latency_mean_sec:.3f}s "
            f"latency_stddev={self.latency_stddev_sec:.3f}s "
            f"qos={self.qos_profile_name}"
        )

    def _load_config(self, scenario_path: str) -> dict[str, Any]:
        if not scenario_path:
            raise RuntimeError("scenario_path parameter is required")
        path = Path(scenario_path)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid scenario file: {path}")
        return data

    def _apply_network_profile(self, network_profile_path: str) -> None:
        if not network_profile_path:
            return
        profile = load_network_profile(network_profile_path)
        faults = self.config.setdefault("faults", {})
        if not isinstance(faults, dict):
            raise RuntimeError("scenario faults must be a mapping")
        if profile.model != "random":
            self.get_logger().warn(
                f"network profile model '{profile.model}' is loaded as static synthetic rates"
            )
        if profile.model == "burst" and profile.packet_loss_percent <= 0.0:
            faults["packet_loss_percent"] = (
                profile.loss_percent_when_good + profile.loss_percent_when_bad
            ) / 2.0
        else:
            faults["packet_loss_percent"] = profile.packet_loss_percent
        faults["latency_ms_mean"] = profile.latency_ms_mean
        faults["latency_ms_stddev"] = (
            profile.latency_ms_stddev
            if profile.latency_ms_stddev > 0.0
            else profile.jitter_ms
        )
        self.get_logger().info(
            "applied network profile: "
            f"path={network_profile_path} model={profile.model} "
            f"loss={float(faults['packet_loss_percent']):.1f}% "
            f"latency_mean={profile.latency_ms_mean:.1f}ms"
        )

    def _load_agents(self, entries: list[dict[str, Any]]) -> list[AgentConfig]:
        agents: list[AgentConfig] = []
        for index, entry in enumerate(entries):
            clock_offset = self.clock_drift_sec if entry.get("id") == "robot_2" else 0.0
            agents.append(
                AgentConfig(
                    agent_id=str(entry["id"]),
                    odom_frame=str(entry["odom_frame"]),
                    base_frame=str(entry["base_frame"]),
                    phase=2.0 * math.pi * index / max(len(entries), 1),
                    clock_offset_sec=clock_offset,
                )
            )
        return agents

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        sim_time = (now - self.start_ros_time).nanoseconds * 1e-9
        stamp = self._to_time_msg(sim_time)
        self.clock_pub.publish(Clock(clock=stamp))

        poses = {agent.agent_id: self._truth_pose(agent, sim_time) for agent in self.agents}
        local_poses = {
            agent.agent_id: self._local_pose(agent, sim_time, poses[agent.agent_id])
            for agent in self.agents
        }

        for agent in self.agents:
            truth_pose = poses[agent.agent_id]
            local_pose = local_poses[agent.agent_id]
            self._publish_ground_truth(agent, stamp, truth_pose)
            self._publish_agent_state(agent, sim_time, local_pose)
            self._publish_odometry(agent, stamp, local_pose)
            if not self._gnss_is_out(agent.agent_id, sim_time):
                xy_var = self._gnss_xy_var(agent.agent_id, sim_time)
                if xy_var is not None:
                    self._publish_gnss(agent, stamp, truth_pose, xy_var)

        constraint_tick = int(sim_time * self.constraint_rate_hz)
        if constraint_tick != self.last_constraint_tick:
            self.last_constraint_tick = constraint_tick
            self._generate_constraints(sim_time, poses)
            self._publish_link_diagnostics(sim_time)

        self._publish_due_constraints(sim_time)

        marker_tick = int(sim_time * self.marker_rate_hz)
        if marker_tick != self.last_marker_tick:
            self.last_marker_tick = marker_tick
            self._publish_markers(stamp, poses, local_poses, sim_time)

    def _truth_pose(self, agent: AgentConfig, t: float) -> tuple[float, float, float]:
        radius = 5.0
        speed = 0.20
        theta = speed * t + agent.phase
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        yaw = self._normalize_angle(theta + math.pi / 2.0)
        return x, y, yaw

    def _local_pose(
        self, agent: AgentConfig, t: float, truth_pose: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        runtime = self.agent_state[agent.agent_id]
        x, y, yaw = truth_pose
        if agent.agent_id == "robot_2" and self._gnss_is_out(agent.agent_id, t):
            runtime.local_drift_x += 0.010
            runtime.local_drift_y -= 0.006
        elif agent.agent_id == "robot_2":
            runtime.local_drift_x *= 0.92
            runtime.local_drift_y *= 0.92
        else:
            runtime.local_drift_x *= 0.98
            runtime.local_drift_y *= 0.98

        noisy_yaw = yaw + (0.015 * math.sin(t + agent.phase))
        return x + runtime.local_drift_x, y + runtime.local_drift_y, noisy_yaw

    def _publish_agent_state(
        self, agent: AgentConfig, sim_time: float, pose: tuple[float, float, float]
    ) -> None:
        xy_var, yaw_var = self._agent_uncertainty(agent.agent_id, sim_time)
        msg = AgentState()
        msg.packet = self._packet(agent.agent_id, "", sim_time, "map", ttl_sec=0.3)
        msg.agent_id = agent.agent_id
        msg.map_frame = "map"
        msg.odom_frame = agent.odom_frame
        msg.base_frame = agent.base_frame
        msg.pose = self._pose_with_covariance(pose, xy_var=xy_var, yaw_var=yaw_var)
        msg.twist.twist.linear.x = 1.0
        msg.twist.twist.angular.z = 0.2
        msg.twist.covariance = self._covariance(0.25, 0.25, 0.25)
        msg.status = (
            AgentState.STATUS_DEGRADED
            if self._gnss_is_out(agent.agent_id, sim_time)
            else AgentState.STATUS_OK
        )
        msg.quality = 0.45 if msg.status == AgentState.STATUS_DEGRADED else 0.95
        self.agent_pubs[agent.agent_id].publish(msg)

    def _publish_odometry(
        self, agent: AgentConfig, stamp: Time, pose: tuple[float, float, float]
    ) -> None:
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id=agent.odom_frame)
        msg.child_frame_id = agent.base_frame
        msg.pose.pose = self._pose_with_covariance(pose, 0.20, 0.05).pose
        msg.pose.covariance = self._covariance(0.20, 0.20, 0.05)
        msg.twist.twist.linear.x = 1.0
        msg.twist.twist.angular.z = 0.2
        msg.twist.covariance = self._covariance(0.25, 0.25, 0.10)
        self.odom_pubs[agent.agent_id].publish(msg)

    def _publish_ground_truth(
        self, agent: AgentConfig, stamp: Time, pose: tuple[float, float, float]
    ) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header = Header(stamp=stamp, frame_id="map")
        msg.pose = self._pose_with_covariance(pose, xy_var=0.001, yaw_var=0.001)
        self.truth_pubs[agent.agent_id].publish(msg)

    def _publish_gnss(
        self,
        agent: AgentConfig,
        stamp: Time,
        pose: tuple[float, float, float],
        xy_var: float = 0.09,
    ) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header = Header(stamp=stamp, frame_id="map")
        msg.pose = self._pose_with_covariance(pose, xy_var=xy_var, yaw_var=0.50)
        self.gnss_pubs[agent.agent_id].publish(msg)

    def _generate_constraints(
        self, sim_time: float, poses: dict[str, tuple[float, float, float]]
    ) -> None:
        pairs = [(self.agents[0], self.agents[1]), (self.agents[1], self.agents[2]), (self.agents[0], self.agents[2])]
        for from_agent, to_agent in pairs:
            key = (from_agent.agent_id, to_agent.agent_id)
            stats = self.link_stats.setdefault(
                key,
                {
                    "received": 0.0,
                    "lost": 0.0,
                    "last_sequence": 0.0,
                    "latency_sum": 0.0,
                    "latency_sq_sum": 0.0,
                    "latency_max": 0.0,
                    "last_event": 1.0,
                    "last_event_time": 0.0,
                    "last_latency": 0.0,
                },
            )
            self.sequence += 1
            stats["last_sequence"] = float(self.sequence)
            stats["last_event_time"] = sim_time
            if self.rng.random() < self.packet_loss:
                stats["lost"] += 1.0
                stats["last_event"] = 0.0
                continue

            latency = max(0.0, self.rng.gauss(self.latency_mean_sec, self.latency_stddev_sec))
            msg = self._relative_constraint(from_agent, to_agent, sim_time, poses)
            stats["received"] += 1.0
            stats["latency_sum"] += latency
            stats["latency_sq_sum"] += latency * latency
            stats["latency_max"] = max(stats["latency_max"], latency)
            stats["last_event"] = 1.0
            stats["last_latency"] = latency
            self.pending_constraints.append(
                PendingConstraint(
                    release_time_sec=sim_time + latency,
                    publisher_agent_id=from_agent.agent_id,
                    message=msg,
                    latency_sec=latency,
                )
            )

    def _publish_due_constraints(self, sim_time: float) -> None:
        ready = [item for item in self.pending_constraints if item.release_time_sec <= sim_time]
        self.pending_constraints = [
            item for item in self.pending_constraints if item.release_time_sec > sim_time
        ]
        for item in ready:
            self.constraint_pubs[item.publisher_agent_id].publish(item.message)

    def _relative_constraint(
        self,
        from_agent: AgentConfig,
        to_agent: AgentConfig,
        sim_time: float,
        poses: dict[str, tuple[float, float, float]],
    ) -> RelativePoseConstraint:
        from_pose = poses[from_agent.agent_id]
        to_pose = poses[to_agent.agent_id]
        relative = self._relative_pose(from_pose, to_pose)
        noisy_relative = (
            relative[0] + self.rng.gauss(0.0, 0.04),
            relative[1] + self.rng.gauss(0.0, 0.04),
            relative[2] + self.rng.gauss(0.0, 0.01),
        )

        msg = RelativePoseConstraint()
        msg.packet = self._packet(
            from_agent.agent_id,
            to_agent.agent_id,
            sim_time,
            from_agent.base_frame,
            ttl_sec=2.0,
        )
        msg.from_agent_id = from_agent.agent_id
        msg.to_agent_id = to_agent.agent_id
        msg.from_frame = from_agent.base_frame
        msg.to_frame = to_agent.base_frame
        msg.from_state_time = msg.packet.measurement_time
        msg.to_state_time = msg.packet.measurement_time
        msg.relative_pose = self._pose_with_covariance(noisy_relative, xy_var=0.04, yaw_var=0.01)
        msg.source_type = RelativePoseConstraint.SOURCE_FAKE_GROUND_TRUTH
        msg.confidence = 0.90
        msg.registration_score = 0.95
        return msg

    def _publish_link_diagnostics(self, sim_time: float) -> None:
        stamp = self._to_time_msg(sim_time)
        for from_agent, to_agent in self.link_stats:
            stats = self.link_stats[(from_agent, to_agent)]
            total = stats["received"] + stats["lost"]
            loss_rate = stats["lost"] / total if total else 0.0
            latency_mean = stats["latency_sum"] / stats["received"] if stats["received"] else 0.0
            variance = 0.0
            if stats["received"]:
                variance = stats["latency_sq_sum"] / stats["received"] - latency_mean * latency_mean

            comm = CommStatus()
            comm.header = Header(stamp=stamp, frame_id="map")
            comm.local_agent_id = from_agent
            comm.remote_agent_id = to_agent
            comm.last_sequence_id = int(stats["last_sequence"])
            comm.received_count = int(stats["received"])
            comm.lost_count = int(stats["lost"])
            comm.loss_rate = float(loss_rate)
            comm.latency_mean = self._to_duration_msg(latency_mean)
            comm.latency_stddev = self._to_duration_msg(math.sqrt(max(0.0, variance)))
            comm.max_latency = self._to_duration_msg(stats["latency_max"])
            comm.qos_profile_name = self.qos_profile_name
            comm.transport_name = "synthetic_loopback"
            self.comm_pubs[from_agent].publish(comm)

            local = self._agent_by_id(from_agent)
            remote = self._agent_by_id(to_agent)
            clock = ClockOffsetEstimate()
            clock.header = Header(stamp=stamp, frame_id="map")
            clock.local_agent_id = from_agent
            clock.remote_agent_id = to_agent
            clock.estimated_offset = self._to_duration_msg(remote.clock_offset_sec - local.clock_offset_sec)
            clock.offset_uncertainty = self._to_duration_msg(0.005)
            clock.round_trip_time = self._to_duration_msg(latency_mean * 2.0)
            clock.samples = int(stats["received"])
            clock.quality = 0.9 if stats["received"] else 0.0
            self.clock_status_pubs[from_agent].publish(clock)

    def _publish_markers(
        self,
        stamp: Time,
        poses: dict[str, tuple[float, float, float]],
        local_poses: dict[str, tuple[float, float, float]],
        sim_time: float,
    ) -> None:
        marker_array = MarkerArray()
        marker_id = 0
        colors = {
            "robot_1": ColorRGBA(r=0.1, g=0.45, b=1.0, a=1.0),
            "robot_2": ColorRGBA(r=1.0, g=0.25, b=0.15, a=1.0),
            "robot_3": ColorRGBA(r=0.1, g=0.75, b=0.35, a=1.0),
        }
        for agent in self.agents:
            runtime = self.agent_state[agent.agent_id]
            truth = poses[agent.agent_id]
            local = local_poses[agent.agent_id]
            runtime.truth_trail.append((truth[0], truth[1]))
            runtime.local_trail.append((local[0], local[1]))
            runtime.truth_trail = runtime.truth_trail[-160:]
            runtime.local_trail = runtime.local_trail[-160:]

            color = colors.get(agent.agent_id, ColorRGBA(r=0.8, g=0.8, b=0.8, a=1.0))
            marker_array.markers.append(
                self._sphere_marker(marker_id, stamp, agent.agent_id, truth, color, scale=0.35)
            )
            marker_id += 1
            marker_array.markers.append(
                self._trail_marker(marker_id, stamp, f"{agent.agent_id}_truth", runtime.truth_trail, color, 0.05)
            )
            marker_id += 1
            local_color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.35)
            marker_array.markers.append(
                self._trail_marker(marker_id, stamp, f"{agent.agent_id}_local", runtime.local_trail, local_color, 0.08)
            )
            marker_id += 1
            xy_var, _ = self._agent_uncertainty(agent.agent_id, sim_time)
            marker_array.markers.append(
                self._covariance_marker(
                    marker_id,
                    stamp,
                    f"{agent.agent_id}_covariance",
                    local,
                    xy_var,
                    local_color,
                )
            )
            marker_id += 1
            if self._gnss_is_out(agent.agent_id, sim_time):
                marker_array.markers.append(
                    self._text_marker(marker_id, stamp, agent.agent_id, truth, "GNSS OUT", ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0))
                )
                marker_id += 1

        for from_agent, to_agent in [(self.agents[0], self.agents[1]), (self.agents[1], self.agents[2]), (self.agents[0], self.agents[2])]:
            stats = self.link_stats.get((from_agent.agent_id, to_agent.agent_id), {})
            marker_array.markers.append(
                self._edge_marker(
                    marker_id,
                    stamp,
                    from_agent.agent_id,
                    to_agent.agent_id,
                    poses[from_agent.agent_id],
                    poses[to_agent.agent_id],
                    stats,
                )
            )
            marker_id += 1
            marker_array.markers.append(
                self._link_text_marker(
                    marker_id,
                    stamp,
                    from_agent.agent_id,
                    to_agent.agent_id,
                    poses[from_agent.agent_id],
                    poses[to_agent.agent_id],
                    stats,
                )
            )
            marker_id += 1

        self.marker_pub.publish(marker_array)

    def _packet(
        self,
        sender: str,
        receiver: str,
        sim_time: float,
        frame_id: str,
        ttl_sec: float,
    ) -> V2VPacketHeader:
        sender_clock = sim_time + self._agent_by_id(sender).clock_offset_sec
        packet = V2VPacketHeader()
        packet.header = Header(stamp=self._to_time_msg(sender_clock), frame_id=frame_id)
        packet.sender_agent_id = sender
        packet.receiver_agent_id = receiver
        packet.sequence_id = self.sequence
        packet.measurement_time = self._to_time_msg(sender_clock)
        packet.source_publish_time = self._to_time_msg(sender_clock)
        packet.ttl = self._to_duration_msg(ttl_sec)
        if self.qos_profile_name == "agent_state_fast":
            packet.reliability_class = V2VPacketHeader.RELIABILITY_BEST_EFFORT
        else:
            packet.reliability_class = (
                V2VPacketHeader.RELIABILITY_IMPORTANT
                if receiver
                else V2VPacketHeader.RELIABILITY_BEST_EFFORT
            )
        packet.frame_convention_version = "mrn.frames.v1"
        return packet

    def _agent_by_id(self, agent_id: str) -> AgentConfig:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise KeyError(agent_id)

    def _load_quality_schedules(
        self, raw: dict[str, Any]
    ) -> dict[str, FixQualitySchedule]:
        schedules: dict[str, FixQualitySchedule] = {}
        if not isinstance(raw, dict):
            raise RuntimeError("faults.gnss_quality_schedule must be a mapping")
        for agent_id, entries in raw.items():
            schedules[agent_id] = FixQualitySchedule.from_config(entries)
        return schedules

    def _gnss_is_out(self, agent_id: str, sim_time: float) -> bool:
        outage = self.gnss_outage.get(agent_id)
        if not outage:
            return False
        start = float(outage.get("start_sec", 0.0))
        duration = float(outage.get("duration_sec", 0.0))
        return start <= sim_time <= start + duration

    def _gnss_xy_var(self, agent_id: str, sim_time: float) -> float | None:
        """Horizontal variance for the GNSS pose, or None to skip publishing.

        With a quality schedule, the variance is driven by mrn_gnss's
        fix-quality covariance, and an INVALID quality (outage) returns
        None so no GNSS pose is published. Without a schedule the legacy
        fixed variance is used.
        """
        schedule = self.gnss_quality_schedules.get(agent_id)
        if schedule is None:
            return 0.09
        var = schedule.covariance_at(sim_time)[0][0]
        if not math.isfinite(var):
            return None
        return var

    def _agent_uncertainty(self, agent_id: str, sim_time: float) -> tuple[float, float]:
        outage = self.gnss_outage.get(agent_id)
        if not outage or not self._gnss_is_out(agent_id, sim_time):
            return 0.16, 0.04

        start = float(outage.get("start_sec", 0.0))
        outage_age = max(0.0, sim_time - start)
        xy_var = min(3.0, 0.16 + 0.055 * outage_age)
        yaw_var = min(0.50, 0.04 + 0.012 * outage_age)
        return xy_var, yaw_var

    def _pose_with_covariance(
        self, pose: tuple[float, float, float], xy_var: float, yaw_var: float
    ):
        msg = PoseWithCovarianceStamped().pose
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(pose[2] / 2.0)
        msg.pose.orientation.w = math.cos(pose[2] / 2.0)
        msg.covariance = self._covariance(xy_var, xy_var, yaw_var)
        return msg

    def _covariance(self, x_var: float, y_var: float, yaw_var: float) -> list[float]:
        covariance = [0.0] * 36
        covariance[0] = x_var
        covariance[7] = y_var
        covariance[14] = 1.0
        covariance[21] = 1.0
        covariance[28] = 1.0
        covariance[35] = yaw_var
        return covariance

    def _relative_pose(
        self, from_pose: tuple[float, float, float], to_pose: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        fx, fy, fyaw = from_pose
        tx, ty, tyaw = to_pose
        dx = tx - fx
        dy = ty - fy
        cos_yaw = math.cos(-fyaw)
        sin_yaw = math.sin(-fyaw)
        x = cos_yaw * dx - sin_yaw * dy
        y = sin_yaw * dx + cos_yaw * dy
        yaw = self._normalize_angle(tyaw - fyaw)
        return x, y, yaw

    def _sphere_marker(
        self,
        marker_id: int,
        stamp: Time,
        ns: str,
        pose: tuple[float, float, float],
        color: ColorRGBA,
        scale: float,
    ) -> Marker:
        marker = self._base_marker(marker_id, stamp, ns, Marker.SPHERE)
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        return marker

    def _trail_marker(
        self,
        marker_id: int,
        stamp: Time,
        ns: str,
        points: list[tuple[float, float]],
        color: ColorRGBA,
        width: float,
    ) -> Marker:
        marker = self._base_marker(marker_id, stamp, ns, Marker.LINE_STRIP)
        marker.scale.x = width
        marker.color = color
        for x, y in points:
            marker.points.append(self._point(x, y, 0.0))
        return marker

    def _covariance_marker(
        self,
        marker_id: int,
        stamp: Time,
        ns: str,
        pose: tuple[float, float, float],
        xy_var: float,
        color: ColorRGBA,
    ) -> Marker:
        marker = self._base_marker(marker_id, stamp, ns, Marker.LINE_STRIP)
        marker.scale.x = 0.035
        marker.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.70)
        radius = 2.0 * math.sqrt(max(0.0, xy_var))
        for index in range(49):
            angle = 2.0 * math.pi * index / 48.0
            marker.points.append(
                self._point(
                    pose[0] + radius * math.cos(angle),
                    pose[1] + radius * math.sin(angle),
                    0.06,
                )
            )
        return marker

    def _edge_marker(
        self,
        marker_id: int,
        stamp: Time,
        from_agent: str,
        to_agent: str,
        from_pose: tuple[float, float, float],
        to_pose: tuple[float, float, float],
        stats: dict[str, float],
    ) -> Marker:
        marker = self._base_marker(marker_id, stamp, f"{from_agent}_to_{to_agent}", Marker.LINE_LIST)
        total = stats.get("received", 0.0) + stats.get("lost", 0.0)
        loss_rate = stats.get("lost", 0.0) / total if total else 0.0
        last_event_delivered = stats.get("last_event", 1.0) > 0.5
        marker.scale.x = 0.05 if last_event_delivered else 0.08
        if not last_event_delivered:
            marker.color = ColorRGBA(r=1.0, g=0.05, b=0.02, a=0.85)
        elif loss_rate > 0.25:
            marker.color = ColorRGBA(r=1.0, g=0.65, b=0.05, a=0.75)
        else:
            marker.color = ColorRGBA(r=0.2, g=1.0, b=0.8, a=0.65)
        marker.points.append(self._point(from_pose[0], from_pose[1], 0.2))
        marker.points.append(self._point(to_pose[0], to_pose[1], 0.2))
        return marker

    def _link_text_marker(
        self,
        marker_id: int,
        stamp: Time,
        from_agent: str,
        to_agent: str,
        from_pose: tuple[float, float, float],
        to_pose: tuple[float, float, float],
        stats: dict[str, float],
    ) -> Marker:
        total = stats.get("received", 0.0) + stats.get("lost", 0.0)
        loss_rate = stats.get("lost", 0.0) / total if total else 0.0
        latency_ms = stats.get("last_latency", 0.0) * 1000.0
        delivered = stats.get("last_event", 1.0) > 0.5
        text = f"{from_agent}->{to_agent} loss {loss_rate * 100:.0f}%"
        text += f" {latency_ms:.0f}ms" if delivered else " DROP"
        midpoint = (
            0.5 * (from_pose[0] + to_pose[0]),
            0.5 * (from_pose[1] + to_pose[1]),
            0.0,
        )
        color = (
            ColorRGBA(r=1.0, g=0.05, b=0.02, a=1.0)
            if not delivered
            else ColorRGBA(r=0.65, g=0.95, b=1.0, a=0.90)
        )
        marker = self._text_marker(marker_id, stamp, f"{from_agent}_to_{to_agent}_status", midpoint, text, color)
        marker.pose.position.z = 0.45
        marker.scale.z = 0.22
        return marker

    def _text_marker(
        self,
        marker_id: int,
        stamp: Time,
        ns: str,
        pose: tuple[float, float, float],
        text: str,
        color: ColorRGBA,
    ) -> Marker:
        marker = self._base_marker(marker_id, stamp, f"{ns}_text", Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0.7
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.35
        marker.color = color
        marker.text = text
        return marker

    def _base_marker(self, marker_id: int, stamp: Time, ns: str, marker_type: int) -> Marker:
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id="map")
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = self._to_duration_msg(0.4)
        return marker

    def _point(self, x: float, y: float, z: float):
        from geometry_msgs.msg import Point

        point = Point()
        point.x = x
        point.y = y
        point.z = z
        return point

    def _to_time_msg(self, seconds: float) -> Time:
        sec = math.floor(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Time(sec=int(sec), nanosec=nanosec)

    def _to_duration_msg(self, seconds: float) -> Duration:
        sec = math.floor(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=int(sec), nanosec=nanosec)

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main() -> None:
    rclpy.init()
    node = SyntheticWorldNode()
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

"""Build a gate-valid ``RelativePoseConstraint`` from a sim V2V measurement.

The simulator observes the relative pose between in-range agents (truth plus
noise); this turns that observation into the V2V constraint message the
cooperative-localization graph consumes — the same contract as the UWB
constraint source. ROS message types are imported lazily so the rest of the
sim core stays usable without a sourced ROS environment.

The source type is ``SOURCE_FAKE_GROUND_TRUTH``: it is honestly derived from the
simulator's ground truth (plus configured noise), not a real sensor.
"""

from __future__ import annotations

import math


def build_relative_constraint(
    *,
    from_agent_id: str,
    to_agent_id: str,
    from_frame: str,
    to_frame: str,
    x: float,
    y: float,
    yaw: float,
    covariance,
    stamp_sec: float,
    sequence_id: int = 0,
    ttl_sec: float = 0.5,
    confidence: float = 0.9,
):
    """Assemble a ``mrn_msgs/RelativePoseConstraint`` (lazy ROS imports)."""
    from builtin_interfaces.msg import Duration, Time
    from mrn_msgs.msg import RelativePoseConstraint

    sec = int(stamp_sec)
    nanosec = int(round((stamp_sec - sec) * 1e9))
    stamp = Time(sec=sec, nanosec=nanosec)
    ttl_s = int(ttl_sec)
    ttl_ns = int(round((ttl_sec - ttl_s) * 1e9))

    msg = RelativePoseConstraint()
    msg.packet.header.stamp = stamp
    msg.packet.header.frame_id = from_frame
    msg.packet.sender_agent_id = from_agent_id
    msg.packet.receiver_agent_id = to_agent_id
    msg.packet.sequence_id = int(sequence_id)
    msg.packet.measurement_time = stamp
    msg.packet.source_publish_time = stamp
    msg.packet.ttl = Duration(sec=ttl_s, nanosec=ttl_ns)

    msg.from_agent_id = from_agent_id
    msg.to_agent_id = to_agent_id
    msg.from_frame = from_frame
    msg.to_frame = to_frame
    msg.from_state_time = stamp
    msg.to_state_time = stamp

    msg.relative_pose.pose.position.x = float(x)
    msg.relative_pose.pose.position.y = float(y)
    msg.relative_pose.pose.orientation.z = math.sin(0.5 * yaw)
    msg.relative_pose.pose.orientation.w = math.cos(0.5 * yaw)
    msg.relative_pose.covariance = list(covariance)

    msg.source_type = RelativePoseConstraint.SOURCE_FAKE_GROUND_TRUTH
    msg.confidence = float(confidence)
    msg.registration_score = 0.0
    return msg

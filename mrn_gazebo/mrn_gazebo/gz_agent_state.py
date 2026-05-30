"""Build an ``AgentState`` from a Gazebo model pose (pure-ish, lazy ROS import).

The Gazebo adapter's job mirrors ``mrn_sim``: turn a robot's world pose into the
``mrn_msgs/AgentState`` the localization stack consumes. The only difference is
the source of the pose — here it is Gazebo's physics rather than the in-house 2D
world. Like the sim, the emitted message carries a stamped packet header and a
TTL so downstream freshness gates accept it (a lesson learned the hard way).
"""

from __future__ import annotations

import math


def build_agent_state(
    agent_id: str,
    x: float,
    y: float,
    yaw: float,
    *,
    frame_id: str = "map",
    stamp_sec: float = 0.0,
    ttl_sec: float = 0.3,
    quality: float = 0.9,
):
    """Assemble a gate-fresh ``mrn_msgs/AgentState`` for a planar pose."""
    from builtin_interfaces.msg import Duration, Time
    from mrn_msgs.msg import AgentState

    sec = int(stamp_sec)
    nanosec = int(round((stamp_sec - sec) * 1e9))
    stamp = Time(sec=sec, nanosec=nanosec)
    ttl_s = int(ttl_sec)
    ttl_ns = int(round((ttl_sec - ttl_s) * 1e9))

    msg = AgentState()
    msg.packet.header.stamp = stamp
    msg.packet.header.frame_id = frame_id
    msg.packet.sender_agent_id = agent_id
    msg.packet.measurement_time = stamp
    msg.packet.source_publish_time = stamp
    msg.packet.ttl = Duration(sec=ttl_s, nanosec=ttl_ns)

    msg.agent_id = agent_id
    msg.map_frame = frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.orientation.z = math.sin(0.5 * yaw)
    msg.pose.pose.orientation.w = math.cos(0.5 * yaw)
    msg.status = AgentState.STATUS_OK
    msg.quality = float(quality)
    return msg

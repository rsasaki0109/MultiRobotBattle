#!/usr/bin/env python3
"""Worked example: a UWB range-bearing cooperative constraint source.

This is the v1.0 "adding a new constraint source" example, the producer-side
counterpart to the offline-evaluator example in ``docs/offline_ate.md``. It
shows how to turn a raw sensor observation into a ``RelativePoseConstraint``
that the graph backends accept — i.e. one that passes ``constraint_gate``.

A UWB (ultra-wideband) radio between two agents measures range and bearing
to the other agent in the observer's body frame. That maps to an SE(2)
relative *position* ``T_from_to`` (range/bearing observe where the other
agent is, not its orientation), so the relative yaw is left effectively
unobserved with a large-but-bounded variance.

The sensor math (:func:`range_bearing_to_relative`) is pure — no ROS — so it
is unit-tested directly and its covariance is validated against
``constraint_gate.validate_pose_covariance``. :func:`build_uwb_constraint`
wraps it into the actual message with the correct frames, packet header,
``T_from_to`` direction, source type, and TTL; a thin ROS node would call it
per measurement and publish on ``/<i>/mrn/relative_constraints``.
"""

from __future__ import annotations

import math

# Index of each variance in a row-major 6x6 PoseWithCovariance covariance.
_XX, _XY, _YX, _YY, _YAW = 0, 1, 6, 7, 35

# A range-bearing radio does not observe relative orientation. Use a large but
# gate-bounded yaw variance (constraint_gate caps it at 10.0) to say "yaw
# unobserved" without tripping the validity check.
DEFAULT_UNOBSERVED_YAW_VAR = 9.0


def range_bearing_to_relative(
    range_m: float,
    bearing_rad: float,
    range_sigma_m: float,
    bearing_sigma_rad: float,
    yaw_var: float = DEFAULT_UNOBSERVED_YAW_VAR,
) -> tuple[float, float, float, list[float]]:
    """Convert a UWB range-bearing measurement to an SE(2) relative pose.

    Returns ``(x, y, yaw, covariance)`` where ``covariance`` is a row-major
    6x6 list. The position covariance is propagated from the measurement
    noise through the polar-to-Cartesian Jacobian

        J = [[cos b, -r sin b],
             [sin b,  r cos b]]
        Sigma_xy = J diag(range_sigma^2, bearing_sigma^2) J^T

    so a small bearing error becomes a larger tangential position error at
    longer range — the correct, range-dependent uncertainty.
    """
    if range_m < 0.0:
        raise ValueError("range_m must be non-negative")
    if range_sigma_m <= 0.0 or bearing_sigma_rad <= 0.0:
        raise ValueError("measurement sigmas must be positive")
    if yaw_var <= 0.0:
        raise ValueError("yaw_var must be positive")

    cos_b = math.cos(bearing_rad)
    sin_b = math.sin(bearing_rad)
    x = range_m * cos_b
    y = range_m * sin_b

    var_r = range_sigma_m * range_sigma_m
    var_b = bearing_sigma_rad * bearing_sigma_rad
    # Sigma_xy = J diag(var_r, var_b) J^T
    sxx = cos_b * cos_b * var_r + (range_m * sin_b) ** 2 * var_b
    syy = sin_b * sin_b * var_r + (range_m * cos_b) ** 2 * var_b
    sxy = cos_b * sin_b * var_r - (range_m * range_m) * sin_b * cos_b * var_b

    covariance = [0.0] * 36
    covariance[_XX] = sxx
    covariance[_YY] = syy
    covariance[_XY] = sxy
    covariance[_YX] = sxy  # keep the covariance symmetric
    covariance[14] = 1.0   # z (unused in SE(2)) — finite placeholder
    covariance[21] = 1.0   # roll
    covariance[28] = 1.0   # pitch
    covariance[_YAW] = yaw_var
    return x, y, 0.0, covariance


def build_uwb_constraint(
    *,
    from_agent_id: str,
    to_agent_id: str,
    from_frame: str,
    to_frame: str,
    range_m: float,
    bearing_rad: float,
    range_sigma_m: float,
    bearing_sigma_rad: float,
    stamp_sec: float,
    sequence_id: int = 0,
    ttl_sec: float = 2.0,
    confidence: float = 0.9,
    yaw_var: float = DEFAULT_UNOBSERVED_YAW_VAR,
):
    """Build a gate-valid ``RelativePoseConstraint`` from a UWB measurement.

    Imported lazily-friendly: the ROS message types are imported here, not at
    module top, so :func:`range_bearing_to_relative` stays usable (and
    testable) without a sourced ROS environment.
    """
    from builtin_interfaces.msg import Duration, Time
    from mrn_msgs.msg import RelativePoseConstraint

    x, y, yaw, covariance = range_bearing_to_relative(
        range_m, bearing_rad, range_sigma_m, bearing_sigma_rad, yaw_var
    )

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

    msg.relative_pose.pose.position.x = x
    msg.relative_pose.pose.position.y = y
    msg.relative_pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.relative_pose.pose.orientation.w = math.cos(yaw / 2.0)
    msg.relative_pose.covariance = covariance

    msg.source_type = RelativePoseConstraint.SOURCE_UWB
    msg.confidence = float(confidence)
    msg.registration_score = 0.0
    return msg

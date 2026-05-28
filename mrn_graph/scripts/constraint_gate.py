"""Validation gates for relative pose constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from mrn_sync.time_gate import (
    TimeGateConfig,
    duration_to_sec,
    validate_clock_offset,
    validate_v2v_packet_time,
)


@dataclass(frozen=True)
class ConstraintGateConfig:
    min_confidence: float = 0.01
    min_position_variance: float = 1e-6
    min_yaw_variance: float = 1e-6
    max_position_variance: float = 100.0
    max_yaw_variance: float = 10.0
    symmetry_tolerance: float = 1e-9


@dataclass(frozen=True)
class ConstraintGateResult:
    accepted: bool
    reason: str = "accepted"


def validate_relative_pose_constraint(
    msg,
    known_agent_ids: Iterable[str] | None = None,
    config: ConstraintGateConfig = ConstraintGateConfig(),
    receive_time_sec: float | None = None,
    clock_offset_estimate=None,
    time_config: TimeGateConfig | None = None,
) -> ConstraintGateResult:
    known = set(known_agent_ids or [])
    if not msg.from_agent_id:
        return ConstraintGateResult(False, "missing_from_agent_id")
    if not msg.to_agent_id:
        return ConstraintGateResult(False, "missing_to_agent_id")
    if msg.from_agent_id == msg.to_agent_id:
        return ConstraintGateResult(False, "self_constraint")
    if known and msg.from_agent_id not in known:
        return ConstraintGateResult(False, "unknown_from_agent")
    if known and msg.to_agent_id not in known:
        return ConstraintGateResult(False, "unknown_to_agent")
    if not msg.from_frame:
        return ConstraintGateResult(False, "missing_from_frame")
    if not msg.to_frame:
        return ConstraintGateResult(False, "missing_to_frame")
    if float(msg.confidence) < config.min_confidence:
        return ConstraintGateResult(False, "low_confidence")
    ttl_sec = duration_to_sec(msg.packet.ttl)
    if ttl_sec <= 0.0:
        return ConstraintGateResult(False, "invalid_ttl")
    if clock_offset_estimate is not None or (
        time_config is not None and time_config.reject_if_unknown_offset
    ):
        clock_gate = validate_clock_offset(
            clock_offset_estimate,
            time_config or TimeGateConfig(reject_if_unknown_offset=False),
        )
        if not clock_gate.accepted:
            return ConstraintGateResult(False, clock_gate.reason)
    if receive_time_sec is not None:
        gate = validate_v2v_packet_time(
            msg.packet,
            receive_time_sec,
            clock_offset_estimate=clock_offset_estimate,
            config=time_config or TimeGateConfig(
                max_message_age_sec=None,
                reject_if_unknown_offset=False,
            ),
        )
        if not gate.accepted:
            return ConstraintGateResult(False, gate.reason)
    covariance_result = validate_pose_covariance(msg.relative_pose.covariance, config)
    if not covariance_result.accepted:
        return covariance_result
    return ConstraintGateResult(True)


def validate_pose_covariance(
    covariance: Iterable[float],
    config: ConstraintGateConfig = ConstraintGateConfig(),
) -> ConstraintGateResult:
    values = list(covariance)
    if len(values) != 36:
        return ConstraintGateResult(False, "invalid_covariance_size")
    if not all(math.isfinite(value) for value in values):
        return ConstraintGateResult(False, "nonfinite_covariance")
    for row in range(6):
        for col in range(row + 1, 6):
            lhs = values[row * 6 + col]
            rhs = values[col * 6 + row]
            if abs(lhs - rhs) > config.symmetry_tolerance:
                return ConstraintGateResult(False, "nonsymmetric_covariance")
    for index in [0, 7]:
        variance = values[index]
        if variance < config.min_position_variance:
            return ConstraintGateResult(False, "position_variance_too_small")
        if variance > config.max_position_variance:
            return ConstraintGateResult(False, "position_variance_too_large")
    yaw_variance = values[35]
    if yaw_variance < config.min_yaw_variance:
        return ConstraintGateResult(False, "yaw_variance_too_small")
    if yaw_variance > config.max_yaw_variance:
        return ConstraintGateResult(False, "yaw_variance_too_large")
    return ConstraintGateResult(True)

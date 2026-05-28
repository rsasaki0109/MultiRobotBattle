"""Shared time gates for V2V packets and replay diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimeGateConfig:
    max_message_age_sec: float | None = 0.3
    max_clock_offset_sec: float = 0.020
    max_offset_uncertainty_sec: float = 0.010
    max_future_skew_sec: float = 0.050
    reject_if_unknown_offset: bool = True


@dataclass(frozen=True)
class TimeGateResult:
    accepted: bool
    reason: str = "accepted"
    age_sec: float | None = None
    clock_offset_sec: float | None = None
    offset_uncertainty_sec: float | None = None


def validate_v2v_packet_time(
    packet,
    receive_time_sec: float,
    clock_offset_estimate=None,
    config: TimeGateConfig = TimeGateConfig(),
) -> TimeGateResult:
    """Validate a V2V packet in the receiver's clock domain.

    `clock_offset_estimate.estimated_offset` follows `ClockOffsetEstimate`:
    `remote_time = local_time + estimated_offset`.
    """

    ttl_sec = duration_to_sec(packet.ttl)
    if ttl_sec <= 0.0:
        return TimeGateResult(False, "invalid_ttl")
    if not math.isfinite(receive_time_sec):
        return TimeGateResult(False, "invalid_receive_time")

    measurement_time_sec = stamp_to_sec(packet.measurement_time)
    publish_time_sec = stamp_to_sec(packet.source_publish_time)
    if not math.isfinite(measurement_time_sec) or not math.isfinite(publish_time_sec):
        return TimeGateResult(False, "invalid_packet_time")
    if publish_time_sec + config.max_future_skew_sec < measurement_time_sec:
        return TimeGateResult(False, "publish_before_measurement")

    offset_result = validate_clock_offset(clock_offset_estimate, config)
    if not offset_result.accepted:
        return offset_result

    clock_offset_sec = offset_result.clock_offset_sec or 0.0
    local_publish_time_sec = publish_time_sec - clock_offset_sec
    age_sec = receive_time_sec - local_publish_time_sec
    if age_sec < -config.max_future_skew_sec:
        return TimeGateResult(
            False,
            "packet_from_future",
            age_sec=age_sec,
            clock_offset_sec=clock_offset_sec,
            offset_uncertainty_sec=offset_result.offset_uncertainty_sec,
        )
    if age_sec > ttl_sec:
        return TimeGateResult(
            False,
            "stale_ttl",
            age_sec=age_sec,
            clock_offset_sec=clock_offset_sec,
            offset_uncertainty_sec=offset_result.offset_uncertainty_sec,
        )
    if config.max_message_age_sec is not None and age_sec > config.max_message_age_sec:
        return TimeGateResult(
            False,
            "stale_max_age",
            age_sec=age_sec,
            clock_offset_sec=clock_offset_sec,
            offset_uncertainty_sec=offset_result.offset_uncertainty_sec,
        )
    return TimeGateResult(
        True,
        age_sec=age_sec,
        clock_offset_sec=clock_offset_sec,
        offset_uncertainty_sec=offset_result.offset_uncertainty_sec,
    )


def validate_receive_age(
    received_time_sec: float,
    now_sec: float,
    ttl_sec: float,
    max_age_sec: float | None = None,
) -> TimeGateResult:
    """Validate how long a packet has been kept after receipt."""

    if ttl_sec <= 0.0:
        return TimeGateResult(False, "invalid_ttl")
    if not math.isfinite(received_time_sec) or not math.isfinite(now_sec):
        return TimeGateResult(False, "invalid_receive_time")
    age_sec = now_sec - received_time_sec
    if age_sec < 0.0:
        return TimeGateResult(False, "negative_receive_age", age_sec=age_sec)
    allowed_age_sec = ttl_sec if max_age_sec is None else min(ttl_sec, max_age_sec)
    if age_sec > allowed_age_sec:
        return TimeGateResult(False, "stale_receive_age", age_sec=age_sec)
    return TimeGateResult(True, age_sec=age_sec)


def duration_to_sec(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def validate_clock_offset(
    clock_offset_estimate,
    config: TimeGateConfig = TimeGateConfig(),
) -> TimeGateResult:
    """Validate a pairwise clock-offset estimate without packet receive time."""

    if clock_offset_estimate is None:
        if config.reject_if_unknown_offset:
            return TimeGateResult(False, "unknown_clock_offset")
        return TimeGateResult(True, clock_offset_sec=0.0, offset_uncertainty_sec=None)

    offset_sec = duration_to_sec(clock_offset_estimate.estimated_offset)
    uncertainty_sec = duration_to_sec(clock_offset_estimate.offset_uncertainty)
    if not math.isfinite(offset_sec) or not math.isfinite(uncertainty_sec):
        return TimeGateResult(False, "invalid_clock_offset")
    if abs(offset_sec) > config.max_clock_offset_sec:
        return TimeGateResult(
            False,
            "clock_offset_too_large",
            clock_offset_sec=offset_sec,
            offset_uncertainty_sec=uncertainty_sec,
        )
    if uncertainty_sec > config.max_offset_uncertainty_sec:
        return TimeGateResult(
            False,
            "clock_uncertainty_too_large",
            clock_offset_sec=offset_sec,
            offset_uncertainty_sec=uncertainty_sec,
        )
    return TimeGateResult(
        True,
        clock_offset_sec=offset_sec,
        offset_uncertainty_sec=uncertainty_sec,
    )

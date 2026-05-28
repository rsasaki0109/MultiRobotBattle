from types import SimpleNamespace

from constraint_gate import (
    ConstraintGateConfig,
    duration_to_sec,
    validate_pose_covariance,
    validate_relative_pose_constraint,
)
from mrn_sync.time_gate import TimeGateConfig


def _duration(sec=1, nanosec=0):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _offset(sec=0, nanosec=0, uncertainty_nsec=5_000_000):
    return SimpleNamespace(
        estimated_offset=_duration(sec=sec, nanosec=nanosec),
        offset_uncertainty=_duration(nanosec=uncertainty_nsec),
    )


def _constraint(covariance=None, confidence=0.9, ttl=None):
    if covariance is None:
        covariance = [0.0] * 36
        covariance[0] = 0.04
        covariance[7] = 0.04
        covariance[14] = 1.0
        covariance[21] = 1.0
        covariance[28] = 1.0
        covariance[35] = 0.01
    return SimpleNamespace(
        from_agent_id="robot_1",
        to_agent_id="robot_2",
        from_frame="robot_1/base_link",
        to_frame="robot_2/base_link",
        confidence=confidence,
        packet=SimpleNamespace(ttl=ttl or _duration(sec=2)),
        relative_pose=SimpleNamespace(covariance=covariance),
    )


def test_accepts_valid_constraint():
    result = validate_relative_pose_constraint(_constraint(), ["robot_1", "robot_2"])
    assert result.accepted


def test_rejects_unknown_agent():
    result = validate_relative_pose_constraint(_constraint(), ["robot_1"])
    assert not result.accepted
    assert result.reason == "unknown_to_agent"


def test_rejects_invalid_ttl():
    result = validate_relative_pose_constraint(_constraint(ttl=_duration(sec=0)))
    assert not result.accepted
    assert result.reason == "invalid_ttl"


def test_rejects_large_clock_offset_without_receive_time():
    result = validate_relative_pose_constraint(
        _constraint(),
        ["robot_1", "robot_2"],
        clock_offset_estimate=_offset(nanosec=80_000_000),
        time_config=TimeGateConfig(max_clock_offset_sec=0.050),
    )
    assert not result.accepted
    assert result.reason == "clock_offset_too_large"


def test_can_require_known_clock_offset():
    result = validate_relative_pose_constraint(
        _constraint(),
        ["robot_1", "robot_2"],
        time_config=TimeGateConfig(reject_if_unknown_offset=True),
    )
    assert not result.accepted
    assert result.reason == "unknown_clock_offset"


def test_rejects_too_confident_covariance():
    covariance = [0.0] * 36
    covariance[0] = 0.0
    covariance[7] = 0.04
    covariance[35] = 0.01
    result = validate_pose_covariance(covariance)
    assert not result.accepted
    assert result.reason == "position_variance_too_small"


def test_rejects_nonsymmetric_covariance():
    covariance = [0.0] * 36
    covariance[0] = 0.04
    covariance[1] = 0.2
    covariance[7] = 0.04
    covariance[35] = 0.01
    result = validate_pose_covariance(covariance, ConstraintGateConfig())
    assert not result.accepted
    assert result.reason == "nonsymmetric_covariance"


def test_duration_to_sec():
    assert duration_to_sec(_duration(sec=2, nanosec=500000000)) == 2.5

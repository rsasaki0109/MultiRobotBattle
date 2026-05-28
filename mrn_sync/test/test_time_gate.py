from types import SimpleNamespace

from mrn_sync.time_gate import (
    TimeGateConfig,
    duration_to_sec,
    stamp_to_sec,
    validate_clock_offset,
    validate_receive_age,
    validate_v2v_packet_time,
)


def _time(sec=0, nanosec=0):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _duration(sec=0, nanosec=0):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _packet(publish_sec=10, measurement_sec=10, ttl_sec=2):
    return SimpleNamespace(
        measurement_time=_time(sec=measurement_sec),
        source_publish_time=_time(sec=publish_sec),
        ttl=_duration(sec=ttl_sec),
    )


def _offset(offset_ns=0, uncertainty_ns=5_000_000):
    return SimpleNamespace(
        estimated_offset=_duration(nanosec=offset_ns),
        offset_uncertainty=_duration(nanosec=uncertainty_ns),
    )


def test_duration_and_stamp_to_sec():
    assert duration_to_sec(_duration(sec=2, nanosec=500_000_000)) == 2.5
    assert stamp_to_sec(_time(sec=7, nanosec=250_000_000)) == 7.25


def test_accepts_packet_with_clock_offset():
    result = validate_v2v_packet_time(
        _packet(publish_sec=10, ttl_sec=2),
        receive_time_sec=10.1,
        clock_offset_estimate=_offset(offset_ns=20_000_000),
    )
    assert result.accepted
    assert round(result.age_sec, 2) == 0.12


def test_rejects_unknown_clock_offset_by_default():
    result = validate_v2v_packet_time(_packet(), receive_time_sec=10.1)
    assert not result.accepted
    assert result.reason == "unknown_clock_offset"


def test_can_allow_unknown_clock_offset_for_replay():
    result = validate_v2v_packet_time(
        _packet(),
        receive_time_sec=10.1,
        config=TimeGateConfig(reject_if_unknown_offset=False),
    )
    assert result.accepted


def test_rejects_stale_packet_by_ttl():
    result = validate_v2v_packet_time(
        _packet(publish_sec=10, ttl_sec=1),
        receive_time_sec=11.2,
        clock_offset_estimate=_offset(),
        config=TimeGateConfig(max_message_age_sec=None),
    )
    assert not result.accepted
    assert result.reason == "stale_ttl"


def test_rejects_high_offset_and_uncertainty():
    high_offset = validate_clock_offset(_offset(offset_ns=30_000_000))
    assert not high_offset.accepted
    assert high_offset.reason == "clock_offset_too_large"

    high_uncertainty = validate_clock_offset(_offset(uncertainty_ns=15_000_000))
    assert not high_uncertainty.accepted
    assert high_uncertainty.reason == "clock_uncertainty_too_large"


def test_clock_offset_threshold_is_configurable():
    result = validate_clock_offset(
        _offset(offset_ns=30_000_000),
        TimeGateConfig(max_clock_offset_sec=0.050),
    )
    assert result.accepted


def test_rejects_future_packet():
    result = validate_v2v_packet_time(
        _packet(publish_sec=10),
        receive_time_sec=9.8,
        clock_offset_estimate=_offset(),
    )
    assert not result.accepted
    assert result.reason == "packet_from_future"


def test_receive_age_uses_ttl_and_max_age():
    accepted = validate_receive_age(
        received_time_sec=10.0,
        now_sec=10.5,
        ttl_sec=2.0,
        max_age_sec=1.0,
    )
    assert accepted.accepted

    stale = validate_receive_age(
        received_time_sec=10.0,
        now_sec=11.2,
        ttl_sec=2.0,
        max_age_sec=1.0,
    )
    assert not stale.accepted
    assert stale.reason == "stale_receive_age"

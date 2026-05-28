import math

from comm_backend import (
    REASON_DELIVERED,
    REASON_DROPPED,
    DeliveryRecord,
    LinkDiagnostics,
    LoopbackBackend,
    LoopbackConfig,
    RecordingBackend,
    ReplayBackend,
    comm_status_fields,
    trace_from_dicts,
    trace_to_dicts,
)


class TestLoopbackConfig:
    def test_rejects_out_of_range_loss(self):
        for bad in (-0.1, 1.1):
            try:
                LoopbackConfig(loss_rate=bad)
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError")

    def test_rejects_negative_latency(self):
        for kwargs in ({"latency_mean_sec": -1.0}, {"latency_stddev_sec": -1.0}):
            try:
                LoopbackConfig(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError")


class TestIdealLoopback:
    def test_perfect_link_delivers_everything(self):
        backend = LoopbackBackend()
        for seq in range(1, 11):
            d = backend.transmit("robot_1", "robot_2", seq, float(seq))
            assert d.delivered
            assert d.reason == REASON_DELIVERED
            assert d.latency_sec == 0.0
            assert d.deliver_time_sec == float(seq)
        diag = backend.diagnostics("robot_1", "robot_2")
        assert diag.received_count == 10
        assert diag.lost_count == 0
        assert diag.loss_rate == 0.0
        assert diag.last_sequence_id == 10
        assert diag.transport_name == "loopback"


class TestLoss:
    def test_total_loss_drops_everything(self):
        backend = LoopbackBackend(LoopbackConfig(loss_rate=1.0))
        for seq in range(1, 6):
            d = backend.transmit("a", "b", seq, 0.0)
            assert not d.delivered
            assert d.reason == REASON_DROPPED
            assert d.latency_sec is None
            assert d.deliver_time_sec is None
        diag = backend.diagnostics("a", "b")
        assert diag.received_count == 0
        assert diag.lost_count == 5
        assert diag.loss_rate == 1.0

    def test_partial_loss_is_deterministic_for_a_seed(self):
        a = LoopbackBackend(LoopbackConfig(loss_rate=0.5, seed=42))
        b = LoopbackBackend(LoopbackConfig(loss_rate=0.5, seed=42))
        mask_a = [a.transmit("x", "y", s, 0.0).delivered for s in range(50)]
        mask_b = [b.transmit("x", "y", s, 0.0).delivered for s in range(50)]
        assert mask_a == mask_b
        # A different seed should produce a different mask (overwhelmingly likely).
        c = LoopbackBackend(LoopbackConfig(loss_rate=0.5, seed=7))
        mask_c = [c.transmit("x", "y", s, 0.0).delivered for s in range(50)]
        assert mask_a != mask_c
        # Observed loss rate should be roughly the configured rate.
        observed = a.diagnostics("x", "y").loss_rate
        assert 0.3 < observed < 0.7


class TestLatencyStats:
    def test_constant_latency_stats(self):
        backend = LoopbackBackend(LoopbackConfig(latency_mean_sec=0.02))
        for seq in range(1, 6):
            d = backend.transmit("a", "b", seq, 1.0)
            assert math.isclose(d.latency_sec, 0.02, abs_tol=1e-12)
            assert math.isclose(d.deliver_time_sec, 1.02, abs_tol=1e-12)
        diag = backend.diagnostics("a", "b")
        assert math.isclose(diag.latency_mean_sec, 0.02, abs_tol=1e-12)
        assert diag.latency_stddev_sec == 0.0
        assert math.isclose(diag.max_latency_sec, 0.02, abs_tol=1e-12)

    def test_jittered_latency_is_nonnegative_with_spread(self):
        backend = LoopbackBackend(
            LoopbackConfig(latency_mean_sec=0.05, latency_stddev_sec=0.02, seed=3)
        )
        latencies = [backend.transmit("a", "b", s, 0.0).latency_sec for s in range(200)]
        assert all(v >= 0.0 for v in latencies)
        diag = backend.diagnostics("a", "b")
        assert diag.latency_stddev_sec > 0.0
        assert diag.max_latency_sec >= diag.latency_mean_sec


class TestLinkIsolation:
    def test_directed_links_are_tracked_independently(self):
        backend = LoopbackBackend()
        backend.transmit("robot_1", "robot_2", 1, 0.0)
        backend.transmit("robot_1", "robot_2", 2, 0.0)
        backend.transmit("robot_2", "robot_1", 9, 0.0)
        fwd = backend.diagnostics("robot_1", "robot_2")
        rev = backend.diagnostics("robot_2", "robot_1")
        assert fwd.received_count == 2
        assert fwd.last_sequence_id == 2
        assert rev.received_count == 1
        assert rev.last_sequence_id == 9


class TestCommStatusMapping:
    def test_fields_match_diagnostics(self):
        backend = LoopbackBackend(LoopbackConfig(latency_mean_sec=0.01))
        backend.transmit("a", "b", 5, 0.0)
        diag = backend.diagnostics("a", "b")
        fields = comm_status_fields(diag, qos_profile_name="agent_state_fast")
        assert fields["local_agent_id"] == "a"
        assert fields["remote_agent_id"] == "b"
        assert fields["last_sequence_id"] == 5
        assert fields["received_count"] == 1
        assert fields["lost_count"] == 0
        assert fields["transport_name"] == "loopback"
        assert fields["qos_profile_name"] == "agent_state_fast"
        assert math.isclose(fields["latency_mean_sec"], 0.01, abs_tol=1e-12)

    def test_build_comm_status_populates_message(self):
        from comm_backend import build_comm_status

        backend = LoopbackBackend(LoopbackConfig(loss_rate=0.0, latency_mean_sec=0.25))
        backend.transmit("robot_1", "robot_2", 3, 0.0)
        diag = backend.diagnostics("robot_1", "robot_2")
        msg = build_comm_status(
            diag, stamp_sec=12.5, qos_profile_name="relative_constraint"
        )
        assert msg.local_agent_id == "robot_1"
        assert msg.remote_agent_id == "robot_2"
        assert msg.last_sequence_id == 3
        assert msg.received_count == 1
        assert msg.transport_name == "loopback"
        assert msg.qos_profile_name == "relative_constraint"
        assert msg.header.stamp.sec == 12
        # 0.25 s latency -> 250 ms in the Duration.
        assert msg.latency_mean.sec == 0
        assert msg.latency_mean.nanosec == 250_000_000


def _drive(backend, n=40):
    """Run a fixed two-link workload through a backend and return per-link diag."""
    for seq in range(n):
        backend.transmit("robot_1", "robot_2", seq, float(seq) * 0.1)
        backend.transmit("robot_2", "robot_1", seq, float(seq) * 0.1 + 0.05)
    return (
        backend.diagnostics("robot_1", "robot_2"),
        backend.diagnostics("robot_2", "robot_1"),
    )


class TestRecordAndReplay:
    def test_record_then_replay_reproduces_diagnostics(self):
        source = LoopbackBackend(
            LoopbackConfig(loss_rate=0.3, latency_mean_sec=0.04,
                           latency_stddev_sec=0.01, seed=11)
        )
        recorder = RecordingBackend(source)
        rec_fwd, rec_rev = _drive(recorder)

        replay = ReplayBackend(recorder.records)
        rep_fwd, rep_rev = _drive(replay)

        for original, replayed in ((rec_fwd, rep_fwd), (rec_rev, rep_rev)):
            assert replayed.received_count == original.received_count
            assert replayed.lost_count == original.lost_count
            assert math.isclose(replayed.loss_rate, original.loss_rate, abs_tol=1e-12)
            assert math.isclose(
                replayed.latency_mean_sec, original.latency_mean_sec, abs_tol=1e-12
            )
            assert math.isclose(
                replayed.max_latency_sec, original.max_latency_sec, abs_tol=1e-12
            )
            assert replayed.last_sequence_id == original.last_sequence_id
        # The replay transport carries its own label, not the recorded one.
        assert rep_fwd.transport_name == "replay"

    def test_recording_backend_is_transparent(self):
        reference = LoopbackBackend(LoopbackConfig(loss_rate=0.5, seed=5))
        direct = [reference.transmit("a", "b", s, 0.0).delivered for s in range(20)]
        source = LoopbackBackend(LoopbackConfig(loss_rate=0.5, seed=5))
        recorder = RecordingBackend(source)
        wrapped = [recorder.transmit("a", "b", s, 0.0).delivered for s in range(20)]
        assert wrapped == direct
        assert recorder.name == "loopback"
        assert len(recorder.records) == 20

    def test_replay_is_strict_about_missing_records(self):
        replay = ReplayBackend([
            DeliveryRecord("a", "b", 0, 0.0, delivered=True, latency_sec=0.01),
        ])
        assert replay.transmit("a", "b", 0, 0.0).delivered
        try:
            replay.transmit("a", "b", 1, 0.0)
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for an unrecorded packet")

    def test_dropped_record_replays_as_drop(self):
        replay = ReplayBackend([
            DeliveryRecord("a", "b", 0, 0.0, delivered=False),
        ])
        d = replay.transmit("a", "b", 0, 7.0)
        assert not d.delivered
        assert d.reason == REASON_DROPPED
        assert d.deliver_time_sec is None

    def test_replay_reapplies_latency_to_caller_send_time(self):
        replay = ReplayBackend([
            DeliveryRecord("a", "b", 0, 0.0, delivered=True, latency_sec=0.25),
        ])
        d = replay.transmit("a", "b", 0, 100.0)
        assert math.isclose(d.latency_sec, 0.25, abs_tol=1e-12)
        assert math.isclose(d.deliver_time_sec, 100.25, abs_tol=1e-12)


class TestTraceSerialization:
    def test_dict_round_trip(self):
        records = [
            DeliveryRecord("a", "b", 0, 0.0, delivered=True, latency_sec=0.02),
            DeliveryRecord("a", "b", 1, 0.1, delivered=False, latency_sec=None),
        ]
        restored = trace_from_dicts(trace_to_dicts(records))
        assert restored == records


def test_record_does_not_count_dropped_latency():
    diag = LinkDiagnostics("a", "b", "loopback")
    from comm_backend import Delivery

    diag.record(1, Delivery(delivered=False, reason=REASON_DROPPED))
    assert diag.lost_count == 1
    assert diag.latency_mean_sec == 0.0
    assert diag.loss_rate == 1.0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
"""Communication backend abstraction and a loopback reference backend.

This is the v1.0 "communication backend interface" (PLAN 15.3 / 22 / 26). A
communication backend is the thing that carries V2V packets between agents and
reports *transport diagnostics* (delivery, loss, latency) — without ever
changing message semantics. Plain ROS 2 DDS is the implicit default transport;
this module adds the first explicit backend so transport behavior can be
swapped and benchmarked deterministically, and so a future Zenoh backend has a
contract to satisfy.

The order of backends (PLAN 26) is: plain DDS, then this loopback backend,
then a rosbag/replay backend, then a Zenoh experiment. They must all expose the
same diagnostics so a benchmark compares transports without touching graph
semantics.

Everything here is pure Python — no ROS, no DDS, no Zenoh — so the transport
model and its diagnostics are unit-testable directly. :func:`build_comm_status`
is the only ROS-aware helper and imports message types lazily, mirroring the
constraint-source example in ``mrn_graph/scripts/uwb_constraint_source.py``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Delivery:
    """Outcome of transmitting a single packet over a backend.

    ``latency_sec`` and ``deliver_time_sec`` are ``None`` for a dropped packet.
    ``reason`` is a stable, machine-readable vocabulary so diagnostics and
    tests can assert on it.
    """

    delivered: bool
    reason: str
    latency_sec: float | None = None
    deliver_time_sec: float | None = None


# Stable delivery reason vocabulary.
REASON_DELIVERED = "delivered"
REASON_DROPPED = "dropped"


class _LatencyAccumulator:
    """Welford online mean/variance for delivered-packet latency."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._max = 0.0

    def add(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)
        if value > self._max:
            self._max = value

    @property
    def mean(self) -> float:
        return self._mean if self._n else 0.0

    @property
    def stddev(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._n - 1))

    @property
    def max(self) -> float:
        return self._max


@dataclass
class LinkDiagnostics:
    """Per-(local, remote) running transport diagnostics.

    The fields map one-to-one onto ``mrn_msgs/msg/CommStatus`` so any backend
    can feed the same status topic regardless of the underlying transport.
    """

    local_agent_id: str
    remote_agent_id: str
    transport_name: str
    last_sequence_id: int = 0
    received_count: int = 0
    lost_count: int = 0
    _latency: _LatencyAccumulator = field(default_factory=_LatencyAccumulator)

    @property
    def total_count(self) -> int:
        return self.received_count + self.lost_count

    @property
    def loss_rate(self) -> float:
        total = self.total_count
        return self.lost_count / total if total else 0.0

    @property
    def latency_mean_sec(self) -> float:
        return self._latency.mean

    @property
    def latency_stddev_sec(self) -> float:
        return self._latency.stddev

    @property
    def max_latency_sec(self) -> float:
        return self._latency.max

    def record(self, sequence_id: int, delivery: Delivery) -> None:
        if sequence_id > self.last_sequence_id:
            self.last_sequence_id = sequence_id
        if delivery.delivered:
            self.received_count += 1
            if delivery.latency_sec is not None:
                self._latency.add(delivery.latency_sec)
        else:
            self.lost_count += 1


class CommunicationBackend(Protocol):
    """Transport contract shared by every backend.

    A backend transmits a packet identified by ``(local, remote, sequence_id)``
    sent at ``send_time_sec`` and returns a :class:`Delivery`. It also exposes
    accumulated :class:`LinkDiagnostics` per directed link. Implementations
    must not alter payload semantics — they only model carriage and report
    diagnostics.
    """

    @property
    def name(self) -> str:
        ...

    def transmit(
        self,
        local_agent_id: str,
        remote_agent_id: str,
        sequence_id: int,
        send_time_sec: float,
    ) -> Delivery:
        ...

    def diagnostics(
        self, local_agent_id: str, remote_agent_id: str
    ) -> LinkDiagnostics:
        ...


@dataclass(frozen=True)
class LoopbackConfig:
    """Deterministic loss/latency model for the loopback backend.

    ``loss_rate`` is the per-packet independent drop probability. Latency is
    drawn from a Gaussian clamped at zero (``latency_mean_sec`` /
    ``latency_stddev_sec``). ``seed`` makes the whole sequence reproducible.
    """

    loss_rate: float = 0.0
    latency_mean_sec: float = 0.0
    latency_stddev_sec: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.loss_rate <= 1.0:
            raise ValueError("loss_rate must be in [0.0, 1.0]")
        if self.latency_mean_sec < 0.0 or self.latency_stddev_sec < 0.0:
            raise ValueError("latency parameters must be non-negative")


class LoopbackBackend:
    """In-process reference backend with a deterministic transport model.

    Useful as the baseline transport in tests and benchmarks: with the default
    zero-loss, zero-latency config it is a perfect link; with a configured
    ``LoopbackConfig`` it injects reproducible loss and latency so transport
    diagnostics can be exercised without a network. Determinism depends on the
    sequence of :meth:`transmit` calls, not on wall-clock time.
    """

    def __init__(self, config: LoopbackConfig | None = None) -> None:
        self._config = config or LoopbackConfig()
        self._rng = random.Random(self._config.seed)
        self._links: dict[tuple[str, str], LinkDiagnostics] = {}

    @property
    def name(self) -> str:
        return "loopback"

    @property
    def config(self) -> LoopbackConfig:
        return self._config

    def diagnostics(
        self, local_agent_id: str, remote_agent_id: str
    ) -> LinkDiagnostics:
        key = (local_agent_id, remote_agent_id)
        link = self._links.get(key)
        if link is None:
            link = LinkDiagnostics(
                local_agent_id=local_agent_id,
                remote_agent_id=remote_agent_id,
                transport_name=self.name,
            )
            self._links[key] = link
        return link

    def transmit(
        self,
        local_agent_id: str,
        remote_agent_id: str,
        sequence_id: int,
        send_time_sec: float,
    ) -> Delivery:
        link = self.diagnostics(local_agent_id, remote_agent_id)
        # Draw loss first, then latency, so the RNG stream is well-defined
        # regardless of branch taken.
        dropped = self._rng.random() < self._config.loss_rate
        latency = self._sample_latency()
        if dropped:
            delivery = Delivery(delivered=False, reason=REASON_DROPPED)
        else:
            delivery = Delivery(
                delivered=True,
                reason=REASON_DELIVERED,
                latency_sec=latency,
                deliver_time_sec=send_time_sec + latency,
            )
        link.record(sequence_id, delivery)
        return delivery

    def _sample_latency(self) -> float:
        mean = self._config.latency_mean_sec
        stddev = self._config.latency_stddev_sec
        if stddev <= 0.0:
            sample = mean
        else:
            sample = self._rng.gauss(mean, stddev)
        return sample if sample > 0.0 else 0.0


@dataclass(frozen=True)
class DeliveryRecord:
    """A single recorded transmission outcome — one entry in a transport trace.

    Identifies the directed link and sequence id so a replay can match it back,
    and stores the outcome (``delivered`` / ``latency_sec``). ``latency_sec`` is
    ``None`` for a dropped packet. A list of these is the unit a rosbag/replay
    backend carries.
    """

    local_agent_id: str
    remote_agent_id: str
    sequence_id: int
    send_time_sec: float
    delivered: bool
    latency_sec: float | None = None


class RecordingBackend:
    """Transparent wrapper that captures another backend's deliveries.

    Delegates :meth:`transmit`/:meth:`diagnostics` to ``inner`` unchanged — the
    transport behavior and diagnostics are exactly the wrapped backend's — while
    appending a :class:`DeliveryRecord` per packet. The captured
    :attr:`records` can be serialized (see :func:`trace_to_dicts`) and later
    fed to a :class:`ReplayBackend` to reproduce the run deterministically.
    """

    def __init__(self, inner: CommunicationBackend) -> None:
        self._inner = inner
        self._records: list[DeliveryRecord] = []

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def records(self) -> tuple[DeliveryRecord, ...]:
        return tuple(self._records)

    def diagnostics(
        self, local_agent_id: str, remote_agent_id: str
    ) -> LinkDiagnostics:
        return self._inner.diagnostics(local_agent_id, remote_agent_id)

    def transmit(
        self,
        local_agent_id: str,
        remote_agent_id: str,
        sequence_id: int,
        send_time_sec: float,
    ) -> Delivery:
        delivery = self._inner.transmit(
            local_agent_id, remote_agent_id, sequence_id, send_time_sec
        )
        self._records.append(
            DeliveryRecord(
                local_agent_id=local_agent_id,
                remote_agent_id=remote_agent_id,
                sequence_id=sequence_id,
                send_time_sec=send_time_sec,
                delivered=delivery.delivered,
                latency_sec=delivery.latency_sec,
            )
        )
        return delivery


class ReplayBackend:
    """Trace-driven backend that replays a recorded transport (PLAN 26 #3).

    Instead of generating loss/latency from a model, it replays a fixed list of
    :class:`DeliveryRecord` — e.g. captured by :class:`RecordingBackend` or read
    from a bag — so a real or previously-modeled network condition can be
    reproduced exactly inside a benchmark. Replay is *strict*: transmitting a
    ``(local, remote, sequence_id)`` with no matching record raises
    ``KeyError`` rather than silently inventing an outcome.

    The recorded latency is reapplied to whatever ``send_time_sec`` the caller
    passes, so ``deliver_time_sec`` stays consistent with the replay clock.
    """

    def __init__(
        self, records, transport_name: str = "replay"
    ) -> None:
        self._by_key: dict[tuple[str, str, int], DeliveryRecord] = {}
        for record in records:
            key = (
                record.local_agent_id,
                record.remote_agent_id,
                record.sequence_id,
            )
            self._by_key[key] = record
        self._transport_name = transport_name
        self._links: dict[tuple[str, str], LinkDiagnostics] = {}

    @property
    def name(self) -> str:
        return self._transport_name

    def diagnostics(
        self, local_agent_id: str, remote_agent_id: str
    ) -> LinkDiagnostics:
        key = (local_agent_id, remote_agent_id)
        link = self._links.get(key)
        if link is None:
            link = LinkDiagnostics(
                local_agent_id=local_agent_id,
                remote_agent_id=remote_agent_id,
                transport_name=self.name,
            )
            self._links[key] = link
        return link

    def transmit(
        self,
        local_agent_id: str,
        remote_agent_id: str,
        sequence_id: int,
        send_time_sec: float,
    ) -> Delivery:
        key = (local_agent_id, remote_agent_id, sequence_id)
        record = self._by_key.get(key)
        if record is None:
            raise KeyError(f"no recorded delivery for {key}")
        link = self.diagnostics(local_agent_id, remote_agent_id)
        if record.delivered:
            latency = record.latency_sec or 0.0
            delivery = Delivery(
                delivered=True,
                reason=REASON_DELIVERED,
                latency_sec=latency,
                deliver_time_sec=send_time_sec + latency,
            )
        else:
            delivery = Delivery(delivered=False, reason=REASON_DROPPED)
        link.record(sequence_id, delivery)
        return delivery


def trace_to_dicts(records) -> list[dict]:
    """Serialize a trace to plain dicts (for YAML/JSON / bag sidecar)."""
    return [
        {
            "local_agent_id": r.local_agent_id,
            "remote_agent_id": r.remote_agent_id,
            "sequence_id": int(r.sequence_id),
            "send_time_sec": float(r.send_time_sec),
            "delivered": bool(r.delivered),
            "latency_sec": (None if r.latency_sec is None else float(r.latency_sec)),
        }
        for r in records
    ]


def trace_from_dicts(dicts) -> list[DeliveryRecord]:
    """Inverse of :func:`trace_to_dicts`."""
    return [
        DeliveryRecord(
            local_agent_id=str(d["local_agent_id"]),
            remote_agent_id=str(d["remote_agent_id"]),
            sequence_id=int(d["sequence_id"]),
            send_time_sec=float(d["send_time_sec"]),
            delivered=bool(d["delivered"]),
            latency_sec=(
                None if d.get("latency_sec") is None else float(d["latency_sec"])
            ),
        )
        for d in dicts
    ]


def comm_status_fields(diag: LinkDiagnostics, qos_profile_name: str = "") -> dict:
    """Project link diagnostics onto the CommStatus field set (pure).

    Returns plain Python values (seconds for latency) so the mapping can be
    asserted in tests without constructing ROS messages.
    """
    return {
        "local_agent_id": diag.local_agent_id,
        "remote_agent_id": diag.remote_agent_id,
        "last_sequence_id": int(diag.last_sequence_id),
        "received_count": int(diag.received_count),
        "lost_count": int(diag.lost_count),
        "loss_rate": float(diag.loss_rate),
        "latency_mean_sec": float(diag.latency_mean_sec),
        "latency_stddev_sec": float(diag.latency_stddev_sec),
        "max_latency_sec": float(diag.max_latency_sec),
        "qos_profile_name": qos_profile_name,
        "transport_name": diag.transport_name,
    }


def _duration_from_sec(seconds: float):
    from builtin_interfaces.msg import Duration

    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    return Duration(sec=sec, nanosec=nanosec)


def build_comm_status(
    diag: LinkDiagnostics,
    stamp_sec: float,
    *,
    qos_profile_name: str = "",
    frame_id: str = "",
):
    """Build a ``mrn_msgs/msg/CommStatus`` from link diagnostics.

    ROS message types are imported lazily so :class:`LinkDiagnostics` and the
    loopback backend stay usable (and testable) without a sourced ROS
    environment.
    """
    from builtin_interfaces.msg import Time
    from mrn_msgs.msg import CommStatus

    fields = comm_status_fields(diag, qos_profile_name)

    sec = int(stamp_sec)
    nanosec = int(round((stamp_sec - sec) * 1e9))

    msg = CommStatus()
    msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = frame_id
    msg.local_agent_id = fields["local_agent_id"]
    msg.remote_agent_id = fields["remote_agent_id"]
    msg.last_sequence_id = fields["last_sequence_id"]
    msg.received_count = fields["received_count"]
    msg.lost_count = fields["lost_count"]
    msg.loss_rate = fields["loss_rate"]
    msg.latency_mean = _duration_from_sec(fields["latency_mean_sec"])
    msg.latency_stddev = _duration_from_sec(fields["latency_stddev_sec"])
    msg.max_latency = _duration_from_sec(fields["max_latency_sec"])
    msg.qos_profile_name = fields["qos_profile_name"]
    msg.transport_name = fields["transport_name"]
    return msg

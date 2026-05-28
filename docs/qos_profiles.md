# QoS Profiles

The project does not make every topic reliable. Stale reliable messages can be worse than dropped messages.

Initial profiles live in `mrn_comm/config/qos_profiles.yaml`.

## Policy

- high-rate agent state: best effort, short lifespan
- heartbeat: best effort, shallow history
- relative constraints: reliable candidate, explicit TTL
- static agent info: reliable and transient local

All profiles must be evaluated under packet loss, latency, and jitter.

## Replay Benchmark

`experiments/qos_best_effort_vs_reliable.yaml` is the first synthetic QoS
comparison. It runs the same cooperative graph method twice with generated
scenario overrides:

- `qos_profile_best_effort_fast`: `agent_state_fast`, 30% packet loss, 40 ms
  mean latency
- `qos_profile_reliable_constraints`: `relative_constraint`, 5% packet loss,
  100 ms mean latency

The synthetic transport is not a DDS implementation. It makes the network
tradeoff visible in replay artifacts: best-effort-like links should show higher
loss and lower latency; reliable-constraint-like links should show lower loss
and higher latency. The acceptance rules check the observed communication rows
by sweep case and QoS profile name.

## Communication Backend Interface

`mrn_comm/scripts/comm_backend.py` defines the transport contract shared by
every communication backend. A backend carries V2V packets between agents and
reports *transport diagnostics* — delivery, loss, and latency — **without ever
changing message semantics**. Plain ROS 2 DDS is the implicit default
transport; the explicit interface lets transport behavior be swapped and
benchmarked deterministically, and gives a future Zenoh backend a contract to
satisfy.

The contract (a `Protocol`) is small:

- `name` — a stable transport label that lands in `CommStatus.transport_name`
- `transmit(local, remote, sequence_id, send_time_sec) -> Delivery` — carry one
  packet; the `Delivery` records `delivered`, a stable `reason`
  (`"delivered"` / `"dropped"`), and, when delivered, `latency_sec` and
  `deliver_time_sec`
- `diagnostics(local, remote) -> LinkDiagnostics` — running per-directed-link
  counters whose fields map one-to-one onto `mrn_msgs/msg/CommStatus`
  (`received_count`, `lost_count`, `loss_rate`, latency mean/stddev/max)

The order of backends (see PLAN §26) is: plain DDS, then the loopback backend
below, then a rosbag/replay backend, then a Zenoh experiment. They must all
expose the same diagnostics so a benchmark compares transports without touching
graph semantics.

### Loopback reference backend

`LoopbackBackend` is the first concrete backend and the in-process baseline for
tests and benchmarks. Its `LoopbackConfig` injects a **deterministic** loss and
latency model:

- `loss_rate` — per-packet independent drop probability
- `latency_mean_sec` / `latency_stddev_sec` — Gaussian latency, clamped at zero
- `seed` — makes the whole delivery/latency stream reproducible

With the default zero-loss, zero-latency config it is a perfect link.
Determinism depends on the *sequence* of `transmit` calls, not on wall-clock
time, so a benchmark replays identically. Latency statistics use a Welford
online accumulator, and each directed link `(local, remote)` is tracked
independently.

`comm_status_fields(diag, qos_profile_name)` projects the diagnostics onto the
`CommStatus` field set as plain Python values (latency in seconds), and
`build_comm_status(diag, stamp_sec, ...)` wraps them into the actual message
(ROS message types imported lazily, so the transport model stays usable and
testable without a sourced ROS environment).

### Replay backend

`ReplayBackend` is the PLAN §26 step-3 transport: instead of generating loss
and latency from a model, it replays a fixed list of `DeliveryRecord` — a
recorded *transport trace* — so a real or previously-modeled network condition
can be reproduced exactly inside a benchmark. Replay is **strict**: transmitting
a `(local, remote, sequence_id)` with no matching record raises `KeyError`
rather than inventing an outcome. The recorded latency is reapplied to whatever
`send_time_sec` the caller passes, keeping `deliver_time_sec` consistent with
the replay clock.

Traces are captured with `RecordingBackend`, a transparent wrapper that
delegates `transmit`/`diagnostics` to an inner backend unchanged while
appending a `DeliveryRecord` per packet. Recording any backend and replaying
its `records` reproduces the original per-link diagnostics exactly (counts,
loss rate, latency mean/stddev/max). `trace_to_dicts` / `trace_from_dicts`
round-trip a trace through plain dicts for a YAML/JSON bag sidecar, so feeding
the trace from a recorded bag is just the wiring layer.

### Adding a new communication backend

Implement the `CommunicationBackend` protocol — `name`, `transmit`, and
`diagnostics`. Reuse `LinkDiagnostics` so the same `CommStatus` mapping applies,
and keep the transport model pure where possible so it can be unit-tested
without a network. A backend must not alter payload semantics; it only models
carriage and reports diagnostics. This is exactly what a Zenoh backend package
(an optional, no-core-dependency package per PLAN §26) would provide.

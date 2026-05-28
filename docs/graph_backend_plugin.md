# GraphBackend Plugin Design

Status: draft (target release v0.4.0)

This document describes the planned plugin interface for cooperative
localization graph backends. It is a design note, not a guarantee. The shape
of the interface is allowed to change while we still have only two reference
backends (`dummy_graph_node` and `relative_anchor_graph_node`); it stabilizes
no later than the `v1.0.0` cut described in `PLAN.md` §22.

## Goals

- Let users swap the cooperative graph backend (dummy, relative-anchor,
  GTSAM, Ceres, custom) without changing message contracts, frame
  conventions, time semantics, covariance rules, or report formats.
- Keep `dummy_graph_node.py` runnable forever as the CI smoke baseline.
- Make every backend produce the same `/mrn/graph/status`,
  `/mrn/graph/markers`, `/robot_i/mrn/cooperative_pose`, and
  `/robot_i/mrn/cooperative_odom` topics so the experiment runner, RViz,
  Foxglove, and reports do not need backend-specific code paths.
- Make rejection visible: backends must report accepted, rejected, and stale
  constraint counts plus rejection reasons. The marker color and report
  rejection rate added in `mrn_eval` / `mrn_graph` already assume this.

## Non-Goals

- Distributed optimization. Plugins are local to one graph node in v0.4.x.
- Pluggable message contracts. Backends consume and produce the existing
  `mrn_msgs` types; new fields go through the `mrn_msgs` review process.
- A C++-only API. Python and C++ backends are both first-class. The
  interface description below is language-neutral; concrete bindings live
  in `mrn_graph/include/mrn_graph/backend.hpp` (C++) and
  `mrn_graph/scripts/graph_backend.py` (Python).

## Python Backend Layer

`mrn_graph/scripts/graph_backend.py` is the Python mirror of the C++
`Backend` interface, kept ROS-free so the backend logic is unit-tested in
CI. `FixedLagBackend.step(agents, relatives, gnss)` consumes plain
dataclasses (`AgentInput`, `RelativeInput`, `GnssInput`) and returns
`CooperativeEstimate` per agent plus a `GraphDiagnostics` carrying the
accepted / rejected / stale counts and rejection-reason map this document
requires. Internally it builds prior + between factors and optimizes them
with `pose_graph_solver.gauss_newton` (see
[`graph_architecture.md`](graph_architecture.md) → "Reference Batch
Backend"). A thin rclpy node converts `AgentState` /
`RelativePoseConstraint` / GNSS messages to and from these dataclasses and
publishes the standard topics — keeping the ROS node trivial and the
optimization + diagnostics logic testable without a running graph.

## Inputs the Backend Consumes

Every backend subscribes to (or is fed by the runner with):

| Input | Source topic | Contract notes |
| --- | --- | --- |
| Local agent state | `/robot_i/mrn/agent_state` | `AgentState`, pose in `map_frame`, finite covariance |
| Relative constraints | `/robot_i/mrn/relative_constraints` | `RelativePoseConstraint`, `T_from_to` direction, validated through `constraint_gate` |
| Clock offsets | `/robot_i/mrn/clock_status` | `ClockOffsetEstimate`, used by the time gate |
| Communication status | `/robot_i/mrn/comm_status` | optional input, primarily for diagnostics |

All inputs go through the constraint gate (`constraint_gate.py`) and the time
gate (`mrn_sync.time_gate`) before reaching the backend. Backends MUST NOT
bypass these gates. A backend that wants to relax a rule must document the
relaxation in its own README and surface the relaxed acceptance count in
graph status.

## Outputs the Backend Produces

Every backend publishes:

| Output | Topic | Contract notes |
| --- | --- | --- |
| Cooperative pose | `/robot_i/mrn/cooperative_pose` | `CooperativePose`, pose in `map_frame`, status flag |
| Cooperative odometry | `/robot_i/mrn/cooperative_odom` | `nav_msgs/Odometry`, header matches cooperative pose |
| Graph status | `/mrn/graph/status` | `ConstraintGraph` with accepted/rejected/stale counts and `rejection_reasons[]` |
| Graph markers | `/mrn/graph/markers` | rejection-summary marker color follows rejection rate |

`backend_name` in `ConstraintGraph` MUST be unique per backend
implementation. The report collector keys graph rows on this string.

## Plugin Lifecycle

A backend is a ROS 2 node. The runner constructs one backend per launch via
the `graph_executable` argument used by `cooperative_localization.launch.py`.
Backends MUST:

1. Declare parameters with the same names as `relative_anchor_graph_node` for
   the parts that are common: `agent_ids`, `publish_rate_hz`,
   `stale_timeout_sec`, `max_constraint_age_sec`, `map_frame`,
   `use_clock_offset_gate`, `reject_unknown_clock_offset`,
   `max_clock_offset_sec`, `max_offset_uncertainty_sec`,
   `clock_status_timeout_sec`. Backends MAY add their own parameters.
2. Publish at least one `ConstraintGraph` message per `publish_rate_hz`
   interval. Backends with longer optimization cycles can publish a
   "pending" status snapshot between solves.
3. Stop cleanly on `KeyboardInterrupt` / `ExternalShutdownException` /
   ROS shutdown. The current backends use the same shutdown pattern.

## Diagnostics Contract

The diagnostics contract is the load-bearing part of the plugin interface.
A backend that solves but does not diagnose is not acceptable. Every
backend MUST expose:

- `accepted_constraint_count`, `rejected_constraint_count`,
  `stale_constraint_count` — monotonic counters since node start.
- `rejection_reasons[]` and `rejection_reason_counts[]` — paired arrays of
  every reason ever observed, sorted by count descending.
- `last_rejection_reason` — the most recent reason string.
- per-agent acceptance counts on `CooperativePose.accepted_constraints` and
  `CooperativePose.rejected_constraints`.
- `quality` on `CooperativePose` — backend-defined, but reset to 0.0 when
  the source state is stale.

Reason strings SHOULD be drawn from a stable vocabulary so reports can be
compared across runs:

- gate-side reasons defined in `constraint_gate.ConstraintGateResult`
  (`low_confidence`, `invalid_ttl`, `nonfinite_covariance`,
  `position_variance_too_small`, `position_variance_too_large`,
  `yaw_variance_too_small`, `yaw_variance_too_large`,
  `nonsymmetric_covariance`, `invalid_covariance_size`,
  `missing_from_agent_id`, `missing_to_agent_id`, `self_constraint`,
  `unknown_from_agent`, `unknown_to_agent`, `missing_from_frame`,
  `missing_to_frame`).
- time-gate reasons surfaced from `mrn_sync.time_gate`
  (`message_too_old`, `clock_offset_unknown`,
  `clock_offset_exceeds_threshold`, `offset_uncertainty_exceeds_threshold`).
- backend-specific reasons SHOULD be prefixed with the backend name, e.g.
  `gtsam.optimizer_diverged`, so they remain distinguishable in JSON
  reports and acceptance YAML.

## Failure Behavior

Backends MUST fail safe rather than silently degrade:

- If no constraints have been accepted, the backend SHOULD publish the
  local `AgentState` pose unchanged with `status = STATUS_OK` (the
  dummy backend already does this; the relative-anchor backend keeps the
  local pose when no candidate anchor is available).
- If a backend cannot solve, it MUST mark the corresponding agent as
  `CooperativePose.STATUS_DEGRADED` or `STATUS_INVALID` and bump
  rejection counters with a descriptive reason.
- Backends MUST NOT publish a finite cooperative pose with zero
  covariance. `mrn_core::isCovarianceValid` is available for output
  validation.

## C++ Skeleton

The C++ interface lives in `mrn_graph/include/mrn_graph/backend.hpp`. The
current shape, exercised by `mrn_graph/test/test_backend_interface.cpp`:

```cpp
namespace mrn_graph
{
struct GraphInputs
{
  std::vector<mrn_msgs::msg::AgentState> agent_states;
  std::vector<mrn_msgs::msg::RelativePoseConstraint> constraints;
  std::vector<mrn_msgs::msg::ClockOffsetEstimate> clock_offsets;
  std::vector<mrn_msgs::msg::CommStatus> comm_status;
};

struct GraphOutputs
{
  std::vector<mrn_msgs::msg::CooperativePose> cooperative_poses;
  std::vector<nav_msgs::msg::Odometry> cooperative_odoms;
  mrn_msgs::msg::ConstraintGraph status;
  visualization_msgs::msg::MarkerArray markers;
};

class Backend
{
public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  virtual void configure(rclcpp::Node & node) = 0;
  virtual GraphOutputs step(const GraphInputs & inputs, const rclcpp::Time & stamp) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;
}  // namespace mrn_graph
```

The shape is exercised by a no-op test backend so changes to the interface
break the gtest before they break a real backend. Python backends will
mirror the same `step()` shape via a small `mrn_graph.backend.PythonBackend`
ABC added alongside the first GTSAM/Ceres backend.

## Migration From `relative_anchor`

`relative_anchor_graph_node.py` is the current placeholder. Migration to a
real graph backend MUST:

1. Keep `cooperative_localization.launch.py
   graph_executable:=relative_anchor_graph_node.py` working as a CI
   baseline until at least one factor-graph backend has parity on the
   `gnss_outage_packet_loss.yaml` experiment.
2. Compare `local_only`, `relative_anchor`, and the new backend in the
   same experiment runner. The runner already supports per-method graph
   rows.
3. Preserve the rejection vocabulary so existing acceptance YAML
   (`min_rejection_reasons`, `max_rejected_constraints`,
   `max_stale_constraints`) continues to work.
4. Reuse the marker rejection-rate coloring rather than defining a new
   visual contract.

## Testing Expectations

A new backend SHOULD ship with:

- a unit test that feeds synthetic `RelativePoseConstraint` messages and
  asserts the output `ConstraintGraph` counts.
- an integration smoke that runs at least one experiment YAML using the
  backend, and produces a non-empty `graph_rows` section in the report.
- documentation of its known failure modes and the rejection reasons it
  introduces.

The dummy backend remains the smoke baseline. New backends MUST NOT remove
or rename the dummy backend's parameters; if a new backend takes the same
name, it MUST also pass the dummy backend's smoke test.

## Related Plans

- `PLAN.md` §15.5 — package plan for `mrn_graph`.
- `PLAN.md` §22 — full plugin plan, including `ConstraintSourcePlugin`,
  `CommunicationBackendPlugin`, `DatasetAdapterPlugin`, `EvaluatorPlugin`.
- `docs/graph_architecture.md` — current phased architecture.
- `docs/covariance.md` — covariance validity rules every backend must
  respect.
- `docs/time_sync.md` — time-gate rules that bound constraint acceptance.

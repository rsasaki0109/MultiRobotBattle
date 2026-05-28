# Interfaces

This document defines the initial ROS 2 message contracts.

## V2VPacketHeader

`V2VPacketHeader` wraps every V2V message with sender, receiver, sequence, timestamp, TTL, reliability class, and frame convention metadata.

Required semantics:

- `header.stamp` is the message publication timestamp in the sender's time domain.
- `measurement_time` is the time at which the underlying measurement was made.
- `source_publish_time` is the sender-side publish time.
- `ttl` is mandatory. A message older than TTL must be dropped.
- `receiver_agent_id` may be empty for broadcast.

## AgentState

`AgentState` describes one robot's current cooperative-localization input state.

Required semantics:

- `pose` is expressed in `map_frame`.
- `twist` is expressed in `base_frame` unless a later adapter explicitly documents otherwise.
- pose and twist covariance must be finite.
- unknown covariance must be represented as large covariance, not zeros.

## RelativePoseConstraint

`RelativePoseConstraint` represents a relative transform between two agent frames.

Transform convention:

```text
T_from_to maps points expressed in to_frame into from_frame.
p_in_from = T_from_to * p_in_to
```

Required fields:

- `from_agent_id`
- `to_agent_id`
- `from_frame`
- `to_frame`
- `from_state_time`
- `to_state_time`
- `relative_pose`
- covariance

Rejection rules:

- stale message
- stale receive age beyond TTL or graph buffer
- missing frame
- invalid covariance
- non-positive TTL
- invalid source publish / measurement time ordering when validated in a shared clock domain
- clock offset or offset uncertainty outside the configured gate
- unknown agent id
- low confidence
- missing state buffer at measurement time
- Mahalanobis distance over the configured gate

## CommStatus

`CommStatus` summarizes observed link quality between a local and remote agent.

Required semantics:

- `loss_rate` is in `[0.0, 1.0]`.
- latency fields use receive-time minus source-publish-time after known clock offset compensation.
- `qos_profile_name` must match a profile in `mrn_comm/config/qos_profiles.yaml` when applicable.

## ClockOffsetEstimate

`ClockOffsetEstimate` describes estimated offset from local time to remote time.

Convention:

```text
remote_time ~= local_time + estimated_offset
```

`offset_uncertainty` must be propagated into time gating. Unknown offset is not equivalent to zero offset.

## CooperativePose

`CooperativePose` is the graph output for an agent.

Required semantics:

- `pose` is expressed in `map_frame`.
- `source_local_pose` stores the local-only pose used by the graph for comparison.
- `status` indicates whether the cooperative solution is valid, degraded, stale, or invalid.

## ConstraintGraph

`ConstraintGraph` is a compact visualization and debugging message, not the canonical graph serialization format.

Rejection diagnostics:

- `rejection_reasons[i]` and `rejection_reason_counts[i]` form a parallel array.
- `last_rejection_reason` stores the latest gate failure reason observed by the backend.
- Reason strings are stable enough for dashboards and smoke tests, but not a public enum yet.

## EvaluationSummary

`EvaluationSummary` reports replay or benchmark metrics. It is designed for dashboards and CI smoke tests, not for full experiment archival.

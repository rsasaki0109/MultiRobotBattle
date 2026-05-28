# Autoware Adapter (Experimental)

Status: experimental (v0.3.0 skeleton; closes the v0.3.0 acceptance gate
"Autoware integration remains adapter-only").

`mrn_autoware_adapter` is a thin boundary node that republishes
cooperative poses as `geometry_msgs/PoseWithCovarianceStamped` on an
Autoware initialpose-style topic. It does not modify Autoware's own
localization output, does not subscribe to Autoware topics, and does not
reach inside any Autoware module — it is purely a one-way emit guarded
by the same SE(2) safety gates as the Nav2 adapter
([`docs/nav2_adapter.md`](nav2_adapter.md)).

This skeleton lands the contract, the safety gates, and the launch
wiring. End-to-end validation against a real Autoware stack is left for
later — until that happens, the adapter is verified by the unit tests in
`mrn_autoware_adapter/test/` and by the launch-time spawn behavior in
`mrn_demos`.

## Topic and Frame Contract

| Direction | Topic | Type | Notes |
| --- | --- | --- | --- |
| in | `/<agent_id>/mrn/cooperative_pose` | `mrn_msgs/CooperativePose` | reliable QoS, depth 10 |
| out | `/<agent_id>/initialpose` (overridable) | `geometry_msgs/PoseWithCovarianceStamped` | reliable QoS, depth 10 |
| out | `~/diagnostics` | `std_msgs/String` | one line per evaluated candidate |

- The output `header.frame_id` is set from the `map_frame` parameter
  (default `map`), matching Autoware's `pose_initializer_node`
  expectations.
- The covariance is forwarded **untouched** from the cooperative source
  — Autoware decides how aggressively to consume it.
- Orientation is forwarded untouched (full quaternion); the gate is
  evaluated in SE(2) but the published pose is 3D.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `agent_id` | `robot_1` | per-instance agent id; drives the input topic name |
| `map_frame` | `map` | output `header.frame_id` |
| `initialpose_topic` | `""` (→ `/<agent_id>/initialpose`) | explicit output topic override |
| `max_pose_age_sec` | `1.0` | stale gate; drops candidates whose `header.stamp` is older than this |
| `max_translation_jump_m` | `1.5` | rejects translation jumps above this between accepted candidates |
| `max_rotation_jump_deg` | `20.0` | rejects yaw jumps above this between accepted candidates |
| `accept_degraded` | `false` | when `false`, `CooperativePose.status == DEGRADED` is rejected |
| `publish_rate_hz` | `0.0` | `0` = emit one message per accepted candidate; `>0` = also periodically republish the last accepted message at this rate |

## Safety Gates

The gates are implemented in
`mrn_autoware_adapter/correction_gate.py` and are a verbatim copy of the
Nav2 adapter's rules; both adapters share the same correction-safety
contract:

- nonfinite pose (`NaN` / `Inf` in `x`, `y`, or yaw) → `NONFINITE_POSE`
- `now - header.stamp > max_pose_age_sec` → `STALE_COOPERATIVE_POSE`
- `CooperativePose.status` not in known set → `UNKNOWN_STATUS_FLAG`
- `status == INVALID` → `INVALID_COOPERATIVE_POSE`
- `status == STALE` → `STALE_COOPERATIVE_POSE`
- `status == DEGRADED` with `accept_degraded=false` →
  `DEGRADED_COOPERATIVE_POSE`
- translation jump from last accepted exceeds budget →
  `TRANSLATION_JUMP_TOO_LARGE`
- yaw jump from last accepted exceeds budget → `ROTATION_JUMP_TOO_LARGE`

Each evaluated candidate produces one line on `~/diagnostics` regardless
of accept/reject, so a downstream Foxglove / RViz panel can see rejection
rates the same way it sees graph-rejected-constraint rates.

## Launch

The cooperative-localization launch can spawn one publisher per agent on
demand:

```bash
ros2 launch mrn_demos cooperative_localization.launch.py \
  enable_autoware_correction:=true \
  autoware_correction_agents:=robot_1,robot_2
```

Other relevant `autoware_*` launch arguments mirror the `nav2_*`
arguments. The Nav2 broadcaster and the Autoware publisher can run
side-by-side — they observe the same cooperative topic and do not touch
each other's output.

Standalone usage (single agent):

```bash
ros2 run mrn_autoware_adapter mrn_autoware_pose_publisher \
  --ros-args -p agent_id:=robot_1 -p map_frame:=map
```

## What This Is Not

- Not a fork of Autoware. The adapter never imports Autoware code; it
  only speaks ROS 2 topic-level contracts.
- Not a pose hypothesis manager. It emits at most one candidate per
  accepted cooperative pose; merging multiple hypotheses or arbitrating
  between cooperative and local re-localization remains Autoware's job.
- Not a coverage report for Autoware's own re-localization. We measure
  the cooperative side (rejection rates, ATE) and trust Autoware to
  handle the rest.

## Related Documents

- [`docs/nav2_adapter.md`](nav2_adapter.md) — parallel Nav2 adapter.
  Both packages re-use the same correction-gate semantics.
- [`docs/interfaces.md`](interfaces.md) — `CooperativePose` field
  reference.
- [`PLAN.md`](../PLAN.md) §12 — v0.3.0 acceptance.
- [`PLAN.md`](../PLAN.md) §25 — Autoware adapter plan.

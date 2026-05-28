# Nav2 Adapter

Status: experimental (added during v0.2.0 work). The adapter is intentionally
conservative: it consumes cooperative pose, applies SE(2) safety gates, and
publishes a `map -> <agent_id>/odom` TF correction. It does not own planning,
control, costmaps, or the local estimator.

## What It Does

`mrn_nav2_correction_broadcaster` subscribes to:

- `/<agent_id>/mrn/cooperative_pose` (`mrn_msgs/CooperativePose`)
- `/<agent_id>/odom` (`nav_msgs/Odometry`)

For each cooperative pose update it runs the SE(2) correction gate in
`mrn_nav2_adapter/correction_gate.py`. Only when the gate accepts the
candidate does it broadcast a `map -> <agent_id>/odom` transform via
`tf2_ros::TransformBroadcaster`. Between accepted updates the last accepted
correction is republished at `publish_rate_hz` so Nav2 always sees a fresh
transform stamp.

The gate emits a `~/diagnostics` `std_msgs/String` for every candidate (both
accepted and rejected). Rejection reasons come from a stable vocabulary so
reports can compare runs:

- `stale_cooperative_pose` — pose age exceeded `max_pose_age_sec` **or** the
  cooperative pose's own status flag is `STATUS_STALE`.
- `degraded_cooperative_pose` — status flag is `STATUS_DEGRADED` and
  `accept_degraded` is `false`.
- `invalid_cooperative_pose` — status flag is `STATUS_INVALID`.
- `translation_jump_too_large` — Euclidean delta between the new candidate
  and the last accepted pose exceeded `max_translation_jump_m`.
- `rotation_jump_too_large` — shortest-arc yaw delta exceeded
  `max_rotation_jump_rad` (configured in degrees).
- `nonfinite_pose` — candidate contained NaN/Inf.
- `unknown_status_flag` — cooperative pose used a status value outside the
  `CooperativePose` enum.

## What It Does Not Do

- It does **not** replace the local estimator. The `<agent_id>/odom -> base`
  transform must still come from `robot_localization`, AMCL, an Autoware
  pose initializer, or whatever the platform already uses.
- It does **not** ingest LaserScan, IMU, or GNSS messages directly.
- It does **not** modify Nav2 costmaps, planners, controllers, or behavior
  trees.
- It does **not** publish `<agent_id>/odom -> base_link`. Publishing both
  the `map -> odom` correction and `odom -> base` would create a TF cycle.

## Frame Tree Assumption

```
map
 └── <agent_id>/odom         <- published by this adapter
       └── <agent_id>/base   <- published by the local stack (robot_localization etc.)
```

`<agent_id>` matches the `agent_id` parameter; the default frame names are
`map` and `odom`, prefixed with `<agent_id>/` to keep multi-robot launches
collision-free.

## Parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `agent_id` | `robot_1` | Suffix for topics and child frame |
| `map_frame` | `map` | Parent frame of the broadcast TF |
| `odom_frame` | `odom` | Child frame of the broadcast TF (prefixed with `<agent_id>/`) |
| `max_pose_age_sec` | `1.0` | Reject cooperative pose older than this many seconds |
| `max_translation_jump_m` | `1.5` | Reject corrections that move map->odom more than this |
| `max_rotation_jump_deg` | `20.0` | Reject corrections whose yaw delta exceeds this |
| `accept_degraded` | `false` | If true, accept `STATUS_DEGRADED` cooperative poses |
| `publish_rate_hz` | `10.0` | Rate at which the last accepted correction is rebroadcast |

## How to Launch

Standalone, per agent:

```bash
ros2 run mrn_nav2_adapter mrn_nav2_correction_broadcaster \
  --ros-args \
  -p agent_id:=robot_1 \
  -p max_pose_age_sec:=0.5
```

Via the cooperative demo launch (opt-in, off by default):

```bash
ros2 launch mrn_demos cooperative_localization.launch.py \
  enable_nav2_correction:=true \
  nav2_correction_agents:=robot_1,robot_2 \
  nav2_max_translation_jump_m:=0.5
```

Launch arguments map one-to-one onto the adapter parameters:

| Launch arg | Adapter parameter |
| --- | --- |
| `enable_nav2_correction` | (gates whether broadcasters are spawned at all) |
| `nav2_correction_agents` | one broadcaster per comma-separated `agent_id` |
| `nav2_max_pose_age_sec` | `max_pose_age_sec` |
| `nav2_max_translation_jump_m` | `max_translation_jump_m` |
| `nav2_max_rotation_jump_deg` | `max_rotation_jump_deg` |
| `nav2_accept_degraded` | `accept_degraded` |
| `nav2_publish_rate_hz` | `publish_rate_hz` |

To disable the correction path entirely without changing the planner stack,
omit `enable_nav2_correction:=true` (default) or, for the standalone case,
set `max_translation_jump_m:=0.0` so every candidate is rejected. The
`~/diagnostics` stream will explain each rejection.

## Interaction With robot_localization

`robot_localization` typically owns `odom -> base_link`. Configure it with
`world_frame: odom` (not `map`) so it never tries to publish a `map -> odom`
transform of its own. This adapter publishes the only `map -> odom`
transform in the system.

If `robot_localization` is already configured to publish `map -> odom`
(i.e., it consumes a global pose), disable that publisher before launching
this adapter. Two publishers on `map -> odom` would race.

## Acceptance Behavior

If no cooperative pose has been received yet, the adapter publishes nothing
and the existing TF tree is unchanged. This is intentional fail-safe
behavior: Nav2 should never see a partially-initialized map->odom
correction.

If the local `nav_msgs/Odometry` is missing, the adapter accepts cooperative
pose updates internally (so the gate state advances) but cannot compute a
correction. As soon as the first odom message arrives, the next accepted
cooperative pose triggers a broadcast.

## Related Documents

- `docs/graph_backend_plugin.md` — what produces `CooperativePose` upstream.
- `docs/graph_architecture.md` — overall cooperative graph topology.
- `docs/covariance.md` — covariance contract for cooperative poses.
- `docs/time_sync.md` — clock/time rules that bound staleness.

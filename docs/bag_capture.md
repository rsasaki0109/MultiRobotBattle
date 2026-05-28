# Two-Robot Bag Capture Procedure

Status: targets the v0.2.0 acceptance criterion that a bag replay produces
`AgentState`, `RelativePoseConstraint`, `CommStatus`, `ClockOffsetEstimate`,
`CooperativePose`, and `EvaluationSummary`. Recording a real two-robot bag
is the v0.2.0 deliverable; this document explains how to capture one in a
way that is replayable by the experiment runner.

## Goal

A bag is "MRN-replayable" when:

1. Every cooperative input topic listed in
   [`docs/interfaces.md`](interfaces.md) is recorded with the agreed QoS.
2. Timestamps are real wall-clock or PTP/Chrony-synced rosbag timestamps,
   not synthetic ones.
3. `agent_id` is unique per robot and matches the `frame_id` prefix.
4. The bag is recorded in MCAP. MCAP is the only container the experiment
   runner is currently expected to ingest.

## Required Topics

Record at least the following per agent ``<i>``:

| Topic | Type | Why |
| --- | --- | --- |
| `/<i>/mrn/agent_state` | `mrn_msgs/AgentState` | local pose, covariance, status |
| `/<i>/mrn/relative_constraints` | `mrn_msgs/RelativePoseConstraint` | V2V observations |
| `/<i>/mrn/comm_status` | `mrn_msgs/CommStatus` | rate, loss, latency context |
| `/<i>/mrn/clock_status` | `mrn_msgs/ClockOffsetEstimate` | offset/uncertainty for time gate |
| `/<i>/odom` | `nav_msgs/Odometry` | needed by `mrn_nav2_adapter` and online ATE |

Additional helpful topics:

- `/tf`, `/tf_static` (with `--all` semantics not recommended; record
  explicit frames)
- `/<i>/sensor/gnss/fix` if GNSS is on board (`sensor_msgs/NavSatFix`)
- raw inputs needed to **reproduce** `AgentState` rather than just replay it
  (laser scans, IMU) when storage budget allows

If the bag is meant to compare `local_only`, `relative_anchor`, and a
real-graph backend at replay time, the cooperative outputs
(`/<i>/mrn/cooperative_pose`, `/mrn/graph/status`) MUST NOT be recorded —
those are produced by the graph backend during replay, and recording them
hides whichever backend was running at capture time.

## QoS

Default to the profiles published in `mrn_comm/config/qos_profiles.yaml`
(validated in CI by `tools/validate_qos_profiles.py`). Concretely:

- `agent_state_fast` — high-rate local state. Best-effort, `keep_last`,
  depth 10.
- `relative_constraint` — V2V observations. Reliable, `keep_last`, depth
  20.
- `heartbeat` — `comm_status` / `clock_status`. Reliable, `keep_last`,
  depth 5.

Use `ros2 bag record --qos-profile-overrides-path` with a QoS overrides
file pinned to those profiles so the bag is self-describing. Otherwise the
recorded reader QoS is whatever `ros2 bag` defaulted to, and replay clients
silently mismatch.

## Time Sync

The clock gate (`mrn_sync.time_gate`) rejects messages whose clock offset
exceeds `max_clock_offset_sec` (default 0.05 s). For a clean recording:

- run `chronyc tracking` (or `ptp4l/timemaster` if PTP) on every robot
  before starting and capture the report in `bag/manifest.md`
- avoid recording across a daylight-saving boundary or NTP step
- if `/clock` is published (sim-time only), record it; otherwise the bag
  uses wall-clock timestamps

## Manifest

Each bag directory should contain a `manifest.md` with:

- bag name and short description
- date, location, weather/lighting if outdoor
- agents present (with `agent_id`, frames, sensors)
- start/end times in UTC
- network conditions (cabled/Wi-Fi/cellular)
- chrony or PTP sync state at capture start
- any non-default QoS overrides
- known issues (e.g., dropped sensor feed at minute 3)
- intended replay scenario (which experiment YAMLs it feeds)

Manifest fields are validated by `tools/validate_bag_manifest.py`. After
recording a bag, run `tools/validate_bag.py` to confirm the bag actually
contains every `required: true` topic with the expected message types:

```bash
python3 tools/validate_bag.py path/to/two_robot_demo_2026-06-01 \
  --manifest path/to/two_robot_demo_2026-06-01/manifest.yaml
```

The bag validator reads `<bag_dir>/metadata.yaml` (written by rosbag2) and
fails with a stable error message if storage is not MCAP, if any required
topic is missing, or if a recorded message type does not match the manifest.
Extra (non-manifest) topics are reported but do not cause a failure.

The same validator runs automatically inside `mrn_experiment run` when an
experiment YAML declares a `bag:` block (see
[`docs/experiments.md`](experiments.md) → "Bag Replay"). The runner then
switches to `bag_replay.launch.py`, which `ros2 bag play`s the directory
alongside the graph backend.

## Capture Workflow

```bash
# 1. Start the robots' local stacks. Confirm /mrn/agent_state is publishing
#    at the expected rate and that frame_id values are <agent_id>/base.

# 2. Record the bag in MCAP.
ros2 bag record \
  --storage mcap \
  --output two_robot_demo_2026-06-01 \
  --qos-profile-overrides-path mrn_comm/config/qos_profiles.yaml \
  /robot_1/mrn/agent_state \
  /robot_1/mrn/relative_constraints \
  /robot_1/mrn/comm_status \
  /robot_1/mrn/clock_status \
  /robot_1/odom \
  /robot_2/mrn/agent_state \
  /robot_2/mrn/relative_constraints \
  /robot_2/mrn/comm_status \
  /robot_2/mrn/clock_status \
  /robot_2/odom \
  /tf /tf_static

# 3. After capture, write manifest.md alongside the .mcap file.
```

## Replay Sanity Check

Before publishing a bag, confirm it round-trips through the experiment
runner:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/<bag_replay>.yaml \
  --duration 30 \
  --output-dir out/experiments/<bag_replay>
```

The resulting `report.md` should show non-zero rows for every recorded
agent and a non-empty `graph_rows` section. If `graph_rows` is empty, the
backend never received valid constraints — usually a QoS mismatch or
missing `relative_constraints` topic.

## Public Distribution

When a bag becomes public:

- include the manifest in the published archive
- include a SHA256 of the `.mcap` so replay reports can pin it
- record the bag's experiment YAML(s) in `experiments/` so anyone with the
  bag can reproduce the report
- the bag is not a substitute for the synthetic CI smoke — keep the
  synthetic path as the always-green baseline

## Related Documents

- [`docs/interfaces.md`](interfaces.md) — full topic and message inventory
- [`docs/qos_profiles.md`](qos_profiles.md) — the validated QoS profiles
- [`docs/time_sync.md`](time_sync.md) — clock-offset rules
- [`docs/experiments.md`](experiments.md) — how the experiment runner
  consumes bag-derived data
- [`docs/offline_ate.md`](offline_ate.md) — post-hoc ATE/RPE comparison
  against a separate RTK truth CSV (for bags without an in-bag ground truth
  topic)
- [`PLAN.md`](../PLAN.md) §11 — v0.2.0 goal and acceptance

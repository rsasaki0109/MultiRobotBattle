# Experiments

Every demo should also be a replayable benchmark.

## GNSS Outage With V2V Constraints

Run target:

```bash
ros2 launch mrn_demos cooperative_localization.launch.py scenario:=gnss_outage_3robots.yaml
ros2 launch mrn_viz rviz_graph.launch.py
```

Network-profile driven target:

```bash
ros2 launch mrn_demos gnss_outage_packet_loss.launch.py \
  network_profile:=/path/to/loss20_delay80.yaml
```

If `network_profile` is omitted, the launch uses
`mrn_netem/config/loss20_delay80.yaml` from the installed package.

Force clock-offset rejection diagnostics:

```bash
ros2 launch mrn_demos cooperative_localization.launch.py \
  scenario:=gnss_outage_3robots.yaml \
  max_clock_offset_sec:=0.01
```

Inspect `/mrn/graph/status` for `clock_offset_too_large` in
`rejection_reasons`.

Expected result:

- three agent poses move in `map`.
- robot 2 enters GNSS outage from 15s to 35s.
- robot 2 local covariance grows during the outage.
- `relative_anchor` publishes cooperative pose using robot 1 and robot 3 as anchors when robot 2 is degraded.
- relative constraint edges are cyan when delivered, orange when the link loss rate is high, and red when the latest packet was dropped.
- packet loss, latency, and clock offset are visible in diagnostics topics.

Known failure modes:

- clock offset over threshold causes constraint rejection.
- reversed relative transform direction causes divergence.
- invalid covariance causes overconfident graph output.

## GNSS Quality Transition (Outage and Reacquisition)

`gnss_outage_3robots.yaml` models GNSS as binary (in / out). For a more
realistic outage-then-reacquisition profile, `gnss_quality_transition_3robots.yaml`
drives robot 2's GNSS pose covariance from a fix-quality *schedule*:

```yaml
faults:
  gnss_quality_schedule:
    robot_2:
      - { start_sec: 0.0,  fix_quality: RTK_FIX }
      - { start_sec: 15.0, fix_quality: INVALID }    # outage
      - { start_sec: 30.0, fix_quality: SINGLE }     # coarse reacquisition
      - { start_sec: 38.0, fix_quality: SBAS }
      - { start_sec: 44.0, fix_quality: RTK_FLOAT }
      - { start_sec: 50.0, fix_quality: RTK_FIX }    # full reacquisition
```

The schedule is a step function (`mrn_gnss.FixQualitySchedule`): each entry's
`fix_quality` holds until the next entry. `INVALID` means no fix — the synthetic
world publishes no GNSS pose during that window. For every finite quality, the
published covariance is `mrn_gnss.position_covariance(quality)`, so the
fix-quality → covariance path is exercised end-to-end without real RTK data.
`fix_quality` accepts a NMEA GGA name (`RTK_FIX`, `SBAS`, …) or the integer
indicator. See [`docs/gnss.md`](gnss.md) for the quality table.

Run the experiment:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/gnss_quality_transition.yaml \
  --duration 60 \
  --output-dir out/experiments/gnss_quality_transition
```

Cooperative localization should beat local-only on robot 2 over the run,
since robot 2 spends 15 s–50 s without a usable RTK fix.

## Synthetic Demo Topics

```text
/robot_i/local/odometry
/robot_i/local/gnss_pose
/robot_i/ground_truth/pose
/robot_i/mrn/agent_state
/robot_i/mrn/cooperative_odom
/robot_i/mrn/cooperative_pose
/robot_i/mrn/relative_constraints
/robot_i/mrn/comm_status
/robot_i/mrn/clock_status
/mrn/graph/status
/mrn/graph/markers
/mrn/eval/summary
/mrn/viz/markers
```

## Offline Metric Helpers

`mrn_eval.metrics` provides deterministic helpers that operate on plain
sample sequences without running the demo:

- `ate_2d(estimate, truth)` — root-mean-square 2D ATE.
- `rpe_translation_2d(estimate, truth, delta=1)` — RMSE of the difference
  between estimated and ground-truth translation deltas across a fixed sample
  step. Useful for trajectories where ATE is dominated by a constant offset.
- `recovery_time(samples, degraded_threshold, recovered_threshold, hold_seconds)`
  — given `(time_sec, error)` pairs, returns the seconds between the first
  degraded sample and a sustained recovery, `0.0` when the trajectory never
  degrades, or `None` when it never recovers. Useful for measuring how long a
  GNSS outage keeps a robot above its acceptable error bound.

These helpers are unit-tested in `mrn_eval/test_mrn_eval/test_metrics.py` and
can be invoked from notebooks or replay scripts that load metrics.json. The
online ATE node still owns the streaming metric used in CI reports.

## Online ATE

`cooperative_localization.launch.py` starts `mrn_online_ate`. It compares:

- `local_only`: `/robot_i/mrn/agent_state`
- `cooperative`: `/robot_i/mrn/cooperative_pose`

against `/robot_i/ground_truth/pose` and publishes `mrn_msgs/EvaluationSummary`
messages on `/mrn/eval/summary`.

Collect a Markdown report from a running demo:

```bash
ros2 run mrn_eval mrn_report \
  --duration 45 \
  --output out/report.md \
  --json-output out/metrics.json
```

The report collector also subscribes to `/robot_i/mrn/comm_status` for the
configured agents and `/mrn/graph/status` for graph diagnostics. It writes
`Network Diagnostics` and `Graph Status` sections plus `network_rows` and
`graph_rows` in JSON. Network rows include loss rate, mean latency, jitter,
max latency, received/lost counts, QoS profile, and transport name. Graph rows
include accepted/rejected/stale constraint counts, total constraint count,
rejection rate (`rejected / total`), top rejection reasons, and the last
rejection reason. The graph rejection-summary RViz/Foxglove marker is colored
green/yellow/red based on the same rejection rate so a visual scan is enough
to see if the backend is silently dropping constraints.

Inspect a reusable network profile:

```bash
ros2 run mrn_netem mrn_netem \
  --profile mrn_netem/config/loss20_delay80.yaml \
  --packets 20 \
  --seed 42 \
  --json
```

The synthetic demo currently publishes network diagnostics from its scenario
fault settings; `mrn_netem` profiles provide the same fault contract for
benchmark configs and later live network namespace wrappers.

Or run the demo and report collector together:

```bash
scripts/run_benchmark.sh 45 out/report.md out/metrics.json
```

Run the benchmark helper against the network-profile launch:

```bash
MRN_LAUNCH_FILE=gnss_outage_packet_loss.launch.py \
  scripts/run_benchmark.sh 45 out/report.md out/metrics.json
```

Run the experiment YAML directly:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/gnss_outage_packet_loss.yaml \
  --duration 45 \
  --output-dir out/experiments/gnss_outage_packet_loss
```

The `run` command evaluates optional `acceptance` rules from the experiment
YAML, writes `acceptance.json`, and includes an `Acceptance` table in the root
`report.md`. A failed rule exits non-zero after preserving the generated report
artifacts. It also writes reproducibility metadata in `provenance.json` plus
human-readable `command.txt`, `git_info.txt`, `ros_distro.txt`,
`dependency_versions.txt`, and `environment.json`.

When `methods` are present, each method is run in sequence under
`out/experiments/<name>/methods/<method>/`. The root `metrics.json` keeps the
primary method rows in `rows`, all method rows in `method_rows`, and all method
network diagnostics in `method_network_rows`. Graph diagnostics are collected
in `graph_rows` and `method_graph_rows`.

## Parameter Sweeps

Use `sweeps` to run one experiment across multiple scenario values. The runner
copies the input scenario into `out/experiments/<name>/sweeps/<case>/scenario.yaml`,
applies dotted-path overrides, and launches each case sequentially.

```yaml
sweeps:
  - name: clock_drift_ms
    parameter: faults.clock_drift_ms
    values: [0, 10, 30, 50, 100]
```

Use `cases` when one sweep case needs multiple scenario overrides:

```yaml
sweeps:
  - name: qos_profile
    parameter: faults.qos_profile_name
    cases:
      - name: best_effort_fast
        value: agent_state_fast
        values:
          faults.qos_profile_name: agent_state_fast
          faults.packet_loss_percent: 30
          faults.latency_ms_mean: 40
      - name: reliable_constraints
        value: relative_constraint
        values:
          faults.qos_profile_name: relative_constraint
          faults.packet_loss_percent: 5
          faults.latency_ms_mean: 100
```

Run the clock drift sensitivity benchmark:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/clock_drift_sensitivity.yaml \
  --duration 45 \
  --output-dir out/experiments/clock_drift_sensitivity
```

The aggregate report includes `Sweep Cases`, method/network comparison tables,
and `Graph Status Comparison`. A clock drift above the graph gate should show
constraint rejections such as `clock_offset_too_large`.

Filter sweep cases for a shorter smoke run:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/clock_drift_sensitivity.yaml \
  --duration 10 \
  --sweep-case clock_drift_ms_50 \
  --sweep-case clock_drift_ms_100 \
  --output-dir out/experiments/clock_drift_smoke
```

## Bag Replay

The runner can also be pointed at a rosbag2 directory instead of the synthetic
world. This is the v0.2.0 path for replaying a real two-robot recording (see
`docs/bag_capture.md` for the capture procedure).

```yaml
experiment:
  name: bag_replay_smoke
bag:
  directory: path/to/two_robot_demo_2026-06-01
  manifest: path/to/two_robot_demo_2026-06-01/manifest.yaml  # optional
  play_rate: 1.0          # optional, default 1.0
  storage: mcap           # optional, default mcap
  agent_ids: [robot_1, robot_2]  # required if more than two robots
  enable_online_ate: false  # default false; only true if bag carries ground truth
  extra_play_args: []     # optional, appended to ros2 bag play
methods:
  - name: coop_graph
    config: mrn_graph/config/gtsam_fixed_lag.yaml
    graph_executable: relative_anchor_graph_node.py
```

When `bag.manifest` is set, the runner cross-checks the recording with
`tools/validate_bag.py` **before** spawning anything. A missing required topic
or message-type mismatch raises a `ValueError` and the run aborts.

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/bag_replay_smoke.yaml \
  --duration 20 \
  --output-dir out/experiments/bag_replay_smoke
```

The runner switches to `bag_replay.launch.py`, which `ExecuteProcess`-spawns
`ros2 bag play --storage <storage> --rate <play_rate> --clock <bag_dir>`
alongside the graph backend. `mrn_online_ate` is **off by default** because real
bags rarely carry `/<agent_id>/ground_truth/pose`; the graph status path
(`/mrn/graph/status` → `graph_rows`) is the primary smoke signal. The repo
ships a stub fixture under `experiments/fixtures/synthetic_bag/` so the YAML
loads in CI before a real bag exists; the fixture is not playable, only
loadable.

`bag` cannot be combined with `network.profile` or `sweeps` in this release —
network and clock-drift sweeps remain the synthetic-world path.

When sweep cases are filtered, network and graph acceptance rules for
unselected sweep cases are skipped. Rules for selected cases still fail the run
if their rows or thresholds are missing.

Run the QoS profile comparison:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/qos_best_effort_vs_reliable.yaml \
  --duration 25 \
  --output-dir out/experiments/qos_best_effort_vs_reliable
```

The aggregate report groups network rows by sweep case and QoS profile. This
benchmark intentionally compares communication behavior, not DDS vendor
internals: the synthetic scenario overrides make a best-effort-like profile
high-loss/low-latency and a reliable-constraint-like profile low-loss/higher
latency.

Example acceptance rules:

```yaml
acceptance:
  localization:
    - agent_id: robot_2
      method_run: coop_graph
      method: cooperative
      min_improvement_vs_local: 0.2
      max_ate_rmse: 0.3
      min_availability: 0.95
  network:
    method_run: coop_graph
    min_rows: 3
    min_observed_loss_rate: 0.1
    max_mean_latency_sec: 0.2
  graph:
    - method_run: coop_graph
      sweep_case: clock_drift_ms_50
      backend: relative_anchor
      max_rejected_constraints: 0
    - method_run: coop_graph
      sweep_case: clock_drift_ms_100
      backend: relative_anchor
      min_rejected_constraints: 1
      min_rejection_reasons:
        clock_offset_too_large: 1
```

Network acceptance can also be a list filtered by sweep case and QoS profile:

```yaml
acceptance:
  network:
    - method_run: coop_graph
      sweep_case: qos_profile_best_effort_fast
      qos_profile_name: agent_state_fast
      min_rows: 3
      min_observed_loss_rate: 0.2
      max_mean_latency_sec: 0.08
    - method_run: coop_graph
      sweep_case: qos_profile_reliable_constraints
      qos_profile_name: relative_constraint
      min_rows: 3
      max_observed_loss_rate: 0.12
      min_mean_latency_sec: 0.08
```

Inspect the resolved launch/report plan without running ROS nodes:

```bash
ros2 run mrn_eval mrn_experiment plan experiments/gnss_outage_packet_loss.yaml
```

## Outputs

Experiment reports should include:

- `metrics.json`
- `acceptance.json`
- `report.md`
- `plan.json`
- `experiment.yaml`
- `provenance.json`
- `command.txt`
- `git_info.txt`
- `ros_distro.txt`
- `dependency_versions.txt`
- `environment.json`
- plots
- result MCAP bag

CI uploads the Jazzy smoke reports as the `jazzy-smoke-artifacts` GitHub
Actions artifact. It includes the clock-drift sweep output, QoS sweep output,
and the cooperative smoke report/log files.

## Bag Manifest

The synthetic demo bag contract is versioned separately from any generated
MCAP file:

```bash
tools/validate_bag_manifest.py mrn_demos/bags/mrn_demo_3robots_manifest.yaml
scripts/record_demo_bag.sh --print-topics
scripts/record_demo_bag.sh bags/mrn_demo_3robots
```

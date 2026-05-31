# Project Plan

Status date: 2026-05-28

This document is the long-form execution plan for `multirobot-navigation`.
It is intentionally more detailed than the README and roadmap. The README
explains the project quickly; this file explains what to build, what not to
build, how to stage the work, and what evidence is required before each release.

## 1. Position

`multirobot-navigation` should not become a "multi-robot Nav2" or an
everything-included navigation stack. That path would put the project in direct
competition with Nav2, Autoware, Open-RMF, Isaac, CARLA, SUMO, and OpenCDA.

The stronger position is:

> ROS 2-native cooperative robotics infrastructure for localization,
> V2V constraints, time synchronization diagnostics, network fault injection,
> multi-agent replay, and benchmarkable multi-robot experiments.

The short README phrase remains:

> ROS 2-native cooperative localization and multi-robot navigation
> infrastructure.

The important word is `infrastructure`. Planning, control, fleet task
allocation, and perception models are not the center of this repository. The
center is the layer that makes cooperative localization experiments reliable
enough to run, replay, debug, compare, and integrate with existing robotics
stacks.

## 2. Core Claim

The missing layer in ROS 2 multi-robot autonomy is not another planner. It is
the practical layer around:

- time semantics
- frame semantics
- covariance semantics
- communication observability
- QoS and stale message handling
- network fault injection
- cooperative relative constraints
- replayable experiments
- benchmark reports

If this project stays focused on those boundaries, it can be useful to Nav2
users, Autoware users, ROS 2 robotics researchers, UGV teams, delivery robots,
warehouse fleets, and V2X/cooperative localization researchers without trying
to replace any of their existing stacks.

## 3. What This Project Is

The project should be presented as:

- a ROS 2 message contract for cooperative localization inputs and outputs
- a V2V constraint exchange layer
- a clock and network diagnostics layer
- a cooperative localization backend interface
- a replayable experiment runner
- a synthetic multi-agent benchmark harness
- a visualization and report generator
- a future adapter layer for Nav2, Autoware, Zenoh, datasets, and real bags

This gives researchers an interface they can plug algorithms into, while giving
real robot developers diagnostics for the things that break before the
algorithm even matters.

## 4. What This Project Is Not

The project should explicitly avoid becoming:

- a Nav2 planner/controller replacement
- an Autoware full-stack replacement
- an Open-RMF fleet dashboard
- a cooperative perception model zoo
- a CARLA/SUMO-only research repository
- a distributed SLAM implementation in the MVP
- a raw LiDAR sharing framework in the MVP
- a cloud-first robotics dashboard
- a Rust-first ROS 2 core stack

Adapters are welcome. Replacements are not.

## 5. Current Repository State

The current scaffold already contains the essential foundation packages:

| Package | Current role |
| --- | --- |
| `mrn_msgs` | message contracts for agent state, V2V packet headers, constraints, comm status, clock status, graph status, cooperative pose, and eval summaries |
| `mrn_core` | C++ utility placeholder for version/covariance/frame helpers |
| `mrn_comm` | QoS profile configuration and profile-name constants |
| `mrn_sync` | Python and C++ time gate helpers and tests |
| `mrn_graph` | dummy backend, relative-anchor backend, constraint gate logic, graph status output |
| `mrn_netem` | synthetic packet loss/profile utilities and CLI |
| `mrn_eval` | online ATE, report collector, experiment runner, acceptance checks, provenance output |
| `mrn_viz` | RViz and Foxglove visualization assets |
| `mrn_demos` | synthetic 3-robot world, launch files, scenarios, bag manifest |

The repository also has:

- Jazzy GitHub Actions build/test workflow
- docs workflow for required docs and bag manifest validation
- smoke experiments for clock drift and QoS profile comparison
- smoke artifact upload from CI
- README badges and artifact inspection guidance
- `v0.1.0-alpha` release checklist

The current cooperative backend is `relative_anchor`, not a real factor graph.
That is acceptable for the current stage because the immediate goal is to make
the contracts, replay, diagnostics, and benchmark pipeline solid before adding
a heavy optimizer.

## 6. Current Working Demos

### 6.1 Synthetic Cooperative Localization

Command:

```bash
ros2 launch mrn_demos cooperative_localization.launch.py \
  scenario:=gnss_outage_3robots.yaml
```

Expected behavior:

- three robots move in `map`
- each robot publishes local odometry, agent state, GNSS pose, and ground truth
- robot 2 enters a GNSS outage in the base scenario
- relative constraints are published between robot pairs
- `relative_anchor` publishes cooperative pose and odometry
- online ATE publishes `/mrn/eval/summary`
- RViz/Foxglove can display poses, paths, graph edges, and diagnostics

### 6.2 GNSS Outage With Network Faults

Command:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/gnss_outage_packet_loss.yaml \
  --duration 25 \
  --output-dir out/experiments/gnss_outage_packet_loss
```

Expected evidence:

- local-only output drifts during robot 2 GNSS outage
- cooperative output improves robot 2 ATE
- network diagnostics show nonzero loss and latency
- graph diagnostics show accepted/rejected/stale counts
- root report contains acceptance, method comparison, network comparison, and
  graph status comparison

### 6.3 Clock Drift Sensitivity

Command:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/clock_drift_sensitivity.yaml \
  --duration 10 \
  --sweep-case clock_drift_ms_50 \
  --sweep-case clock_drift_ms_100 \
  --output-dir out/experiments/clock_drift_smoke
```

Expected evidence:

- `clock_drift_ms_50` stays under the current graph rejection gate
- `clock_drift_ms_100` triggers `clock_offset_too_large`
- acceptance verifies graph rejection counts and reasons

### 6.4 QoS Profile Comparison

Command:

```bash
ros2 run mrn_eval mrn_experiment run \
  experiments/qos_best_effort_vs_reliable.yaml \
  --duration 10 \
  --output-dir out/experiments/qos_smoke
```

Expected evidence:

- best-effort-like case reports higher loss and lower latency
- reliable-constraint-like case reports lower loss and higher latency
- acceptance filters network rows by sweep case and QoS profile name

This is still synthetic transport behavior. It is not a DDS vendor benchmark
yet. The value is that the experiment runner and report contract can compare
communication profiles in a deterministic replay harness.

## 7. Release Strategy

The project should release early, but the first alpha must be honest. The alpha
should prove infrastructure contracts, not algorithmic completeness.

Recommended release sequence:

| Release | Theme | Primary proof |
| --- | --- | --- |
| `v0.1.0-alpha` | synthetic infrastructure alpha | Jazzy build/test, synthetic demo, reports, acceptance checks, CI artifacts |
| `v0.2.0` | real-robot replay and Nav2 adapter | two-robot bag, correction publication, Nav2 integration notes |
| `v0.3.0` | GNSS/Autoware/dataset bridge | ENU utilities, Autoware adapter, first dataset converter prototype |
| `v0.4.0` | first real graph backend | GTSAM/Ceres backend with odom/GNSS/relative factors |
| `v1.0.0` | stable infrastructure contract | stable messages, real bags, docs, plugin API, known limitations |

The release numbers can change, but the order should not. Real robot replay and
contract quality should come before distributed optimization.

## 8. v0.1.0-alpha Goal

The alpha goal:

> A user can build the workspace on ROS 2 Jazzy, launch a synthetic 3-robot
> cooperative localization demo, run benchmark experiments with packet loss,
> latency, clock drift, and QoS profile variation, and inspect Markdown/JSON
> reports that clearly show localization, network, time, and graph diagnostics.

The alpha does not need:

- GTSAM
- Ceres
- Nav2 adapter
- Autoware adapter
- Zenoh backend
- real robot bag
- MCAP recording automation beyond manifest/topic contracts

The alpha does need:

- consistent message semantics
- consistent frame semantics
- consistent time semantics
- covariance rules documented
- deterministic smoke tests
- clear README and docs
- known limitations stated plainly

## 9. v0.1.0-alpha Acceptance Gate

The release is acceptable when all of these are true:

- `colcon build --symlink-install` passes on Jazzy
- `colcon test --event-handlers console_direct+` passes
- `colcon test-result --verbose` reports zero failures
- `scripts/smoke_cooperative_demo.sh 20 out/smoke_report.md` passes
- `tools/validate_bag_manifest.py mrn_demos/bags/mrn_demo_3robots_manifest.yaml`
  passes
- `mrn_experiment run experiments/gnss_outage_packet_loss.yaml` passes
- `mrn_experiment run experiments/clock_drift_sensitivity.yaml` passes for the
  selected CI smoke cases
- `mrn_experiment run experiments/qos_best_effort_vs_reliable.yaml` passes
- CI uploads `jazzy-smoke-artifacts`
- README Quick Start is current
- `docs/interfaces.md` describes every public message contract used in the
  demos
- `docs/frames.md` defines transform directions and frame ownership
- `docs/time_sync.md` defines message time semantics and clock rejection rules
- `docs/covariance.md` defines covariance validity and rejection rules
- release notes mention that `relative_anchor` is not a real factor graph
- release notes mention synthetic relative pose uses fake ground truth plus
  noise

## 10. v0.1.0-alpha Work Breakdown

### 10.1 Documentation Freeze

Tasks:

- ensure README Quick Start matches the actual launch files
- ensure README says what the project is not
- add or update release notes for `v0.1.0-alpha`
- make `docs/release_checklist.md` reflect all current smoke experiments
- ensure `docs/interfaces.md` matches message definitions
- ensure `docs/experiments.md` documents all experiment YAML features
- ensure `docs/qos_profiles.md` explains synthetic-vs-real QoS limitations
- ensure `docs/graph_architecture.md` clearly labels `relative_anchor` as a
  temporary baseline

Acceptance:

- a new user can identify the fastest demo path in under 60 seconds
- a researcher can identify the replay/benchmark path in under 5 minutes
- a developer can find message semantics without reading source code

### 10.2 CI Hardening

Tasks:

- keep build/test on Jazzy
- keep docs workflow validating required docs and bag manifest
- keep clock drift smoke in CI
- keep QoS smoke in CI
- upload smoke reports as artifacts
- add a workflow note in README and docs
- consider adding a lightweight markdown link check later

Acceptance:

- a failed acceptance check preserves report artifacts
- a launch failure preserves smoke logs
- artifact names are stable
- CI logs print the first section of each report

### 10.3 Experiment Runner Hardening

Tasks:

- keep `plan` command deterministic and useful
- keep method comparison stable
- keep sweep case filtering stable
- keep network acceptance filters stable
- keep graph acceptance filters stable
- keep provenance output stable
- add tests for any new YAML syntax before adding new experiment configs

Acceptance:

- bad metrics fail with a clear acceptance check name
- filtered sweep cases skip only the rules for unselected cases
- generated scenario YAMLs are saved under the experiment output directory
- `plan.json` always describes what was launched

### 10.4 Demo Quality

Tasks:

- keep the synthetic motion visually readable
- keep covariance ellipses visible but not noisy
- keep rejected/stale/accepted constraints visually distinguishable
- keep graph status markers and report rows aligned
- produce a short GIF or video after the alpha contract is stable

Acceptance:

- one screenshot or short clip communicates: GNSS outage, packet loss, V2V
  constraints, and cooperative recovery
- report numbers line up with the visual story

### 10.5 Known Limitations

These limitations must be stated in release notes:

- current cooperative backend is `relative_anchor`, not a factor graph
- synthetic relative pose constraints come from fake ground truth plus noise
- online ATE is the primary localization metric today
- NEES/NIS are roadmap items, not complete metrics yet
- QoS comparison is synthetic transport behavior, not a DDS vendor benchmark
- no real robot bag is included yet
- no Nav2 or Autoware adapter is implemented yet

## 11. v0.2.0 Goal

The v0.2 goal:

> Prove that the infrastructure works beyond synthetic replay by supporting a
> small real or real-like two-robot bag and a basic Nav2 adapter path.

### 11.1 Scaffolding (done; CI-verified without a real bag)

- [x] document bag capture procedure
  ([`docs/bag_capture.md`](docs/bag_capture.md))
- [x] bag manifest schema + validator
  (`tools/validate_bag_manifest.py`, `tools/validate_bag.py`)
- [x] bag replay launch + experiment runner integration
  (`mrn_demos/launch/bag_replay.launch.py`,
  `experiments/bag_replay_smoke.yaml`)
- [x] replay benchmark independent of the synthetic world node
  (`mrn_experiment run` switches to `bag_replay.launch.py` when a `bag:`
  block is present in the experiment YAML)
- [x] add `mrn_nav2_adapter` with safety gates for stale, large, or
  missing-TF corrections (`mrn_nav2_adapter/correction_gate.py`,
  `correction_broadcaster_node.py`,
  [`docs/nav2_adapter.md`](docs/nav2_adapter.md))
- [x] correction broadcaster wired into the cooperative-localization launch
  via `enable_nav2_correction`/`nav2_correction_agents` args (off by default)
- [x] Linux network-namespace netem smoke path
  (`mrn_netem/mrn_netem/netns.py`, `netns_cli.py`,
  [`docs/netem_netns.md`](docs/netem_netns.md))
- [x] offline ATE/RPE helper for bags without an in-bag truth topic
  (`mrn_eval/mrn_eval/offline_ate.py`, `offline_ate_cli.py`,
  [`docs/offline_ate.md`](docs/offline_ate.md))
- [x] bag-to-CSV exporter that feeds `mrn_eval_offline_ate --estimated`
  from a recorded topic (CooperativePose/AgentState/Odometry/PoseStamped/
  PoseWithCovarianceStamped); extractors are pure-function and CI-tested
  without rosbag2_py
  (`mrn_eval/mrn_eval/bag_to_csv.py`, `bag_to_csv_cli.py`)

### 11.2 Real-bag dependent (pending; blocks acceptance)

- [ ] record a small two-robot real bag (field work; requires hardware
  access and synced clocks per `docs/bag_capture.md`)
- [ ] RTK-to-CSV exporter that feeds `mrn_eval_offline_ate --truth` from a
  separate RTK logger; frame convention (ENU origin, axis order) will be
  pinned together with the v0.3.0 GNSS work (§12)
- [ ] end-to-end smoke that round-trips a recorded bag through
  `mrn_eval_bag_to_csv` → `mrn_eval_offline_ate` against an RTK truth CSV;
  blocked on the real bag + RTK exporter above
- [ ] publish the recorded bag + its `manifest.md` + experiment YAML(s) so
  external users can reproduce the report

Non-goals:

- full distributed graph optimization
- full fleet management
- production-ready Nav2 deployment

Acceptance:

- a bag replay produces `AgentState`, `RelativePoseConstraint`, `CommStatus`,
  `ClockOffsetEstimate`, `CooperativePose`, and `EvaluationSummary`
- the Nav2 adapter can be launched without changing core message contracts
- the adapter has safety gates for stale correction, large correction, and
  missing TF
- docs explain how to disable the correction path

## 12. v0.3.0 Goal

The v0.3 goal:

> Make the project credible for outdoor localization and autonomous vehicle
> research by adding GNSS/ENU utilities, RTK metadata handling, and early
> Autoware/dataset adapter prototypes.

### 12.1 Scaffolding (done; CI-verified)

- [x] WGS84 geodetic ↔ ECEF conversion (Bowring closed-form ECEF→geodetic)
  (`mrn_gnss/mrn_gnss/wgs84.py`)
- [x] local ENU frame conversion with cached-trig origin
  (`mrn_gnss/mrn_gnss/enu.py`)
- [x] NMEA GGA fix-quality enum + horizontal/vertical sigma tables +
  diagonal 3×3 ENU position covariance helper, plus
  `FixQuality.from_navsatstatus` for `sensor_msgs/NavSatStatus` callers
  (`mrn_gnss/mrn_gnss/fix_quality.py`)
- [x] docs for frame conventions, origin choice, units, and the
  covariance heuristic ([`docs/gnss.md`](docs/gnss.md))
- [x] `mrn_eval_rtk_to_csv` CLI: converts an RTK logger CSV
  (`stamp_sec,lat_deg,lon_deg,alt_m,fix_quality`) into the offline-ATE
  truth CSV via `mrn_gnss` local-ENU linearization, writes a
  `<output>.origin.yaml` sidecar, and filters by minimum fix quality
  (default RTK_FLOAT)
  (`mrn_eval/mrn_eval/rtk_to_csv.py`, `rtk_to_csv_cli.py`)
- [x] `mrn_autoware_adapter` skeleton: CooperativePose →
  `PoseWithCovarianceStamped` publisher on a configurable
  initialpose-style topic, gated by the same SE(2) safety rules as the
  Nav2 adapter (verbatim copy of `correction_gate.py` for now; refactor
  to a shared location deferred to v0.4). Wired into
  `cooperative_localization.launch.py` via
  `enable_autoware_correction` / `autoware_correction_agents`
  (`mrn_autoware_adapter/`, [`docs/autoware_adapter.md`](docs/autoware_adapter.md))
- [x] NavSatFix → RTK CSV path in `mrn_eval_bag_to_csv`: when the topic is
  `sensor_msgs/msg/NavSatFix`, emit the geodetic RTK input schema
  (`stamp_sec,lat_deg,lon_deg,alt_m,fix_quality`) for `mrn_eval_rtk_to_csv`
  to linearize. NavSatStatus maps via `FixQuality.from_navsatstatus`, so
  the quality ceiling is DGPS (NavSatStatus cannot express RTK)
  (`mrn_eval/mrn_eval/bag_to_csv.py`, `rtk_to_csv.py:write_rtk_csv`)
- [x] GNSS outage / reacquisition experiment with quality transitions:
  `mrn_gnss.FixQualitySchedule` (pure, step-function, CI-tested) drives
  the synthetic world's GNSS covariance via `position_covariance`;
  scenario `gnss_quality_transition_3robots.yaml`
  (RTK_FIX → INVALID → SINGLE → SBAS → RTK_FLOAT → RTK_FIX) and experiment
  `experiments/gnss_quality_transition.yaml`
  (`mrn_gnss/mrn_gnss/quality_schedule.py`,
  [`docs/experiments.md`](docs/experiments.md) → "GNSS Quality Transition")
- [x] dataset adapter prototype: `mrn_eval_tum_to_csv` converts a TUM
  trajectory (`timestamp tx ty tz [qx qy qz qw]`, the TUM RGB-D / EuRoC
  export convention) into the offline-ATE CSV, making `mrn_eval_offline_ate`
  usable against public benchmark datasets with no ROS dependency
  (`mrn_eval/mrn_eval/tum_to_csv.py`,
  [`docs/offline_ate.md`](docs/offline_ate.md) → "Feeding from a Public Dataset")
- [x] `docs/frames.md` extended with `earth` (ECEF/WGS84), `map` (local-ENU
  tangent plane = `earth -> map` via `mrn_gnss.EnuOrigin`), shared-origin
  requirement, and ellipsoidal-altitude rules ([`docs/frames.md`](docs/frames.md))

### 12.2 Pending

v0.3.0 scaffolding is complete; the remaining work is real-data dependent:

- [ ] validate the Autoware adapter against a real Autoware stack / AWSIM
- [ ] ingest one concrete public dataset end-to-end (download, convert,
  run `mrn_eval_offline_ate`, record the report) once a target dataset is
  chosen

Non-goals:

- full Autoware stack replacement
- full OPV2V/OpenCOOD ingestion
- cooperative perception model training

Acceptance:

- GNSS covariance is not silently trusted
- ENU frame origin and transform direction are documented
- Autoware integration remains adapter-only

## 13. v0.4.0 Goal

The v0.4 goal:

> Replace the temporary cooperative backend with the first real graph backend
> while preserving the existing replay, report, and acceptance contracts.

### 13.1 Scaffolding (done; CI-verified, solver-independent)

- [x] solver-independent factor core in pure Python (no GTSAM/numpy), so
  the factor math is proven before the solver dependency is decided:
  SE(2) pose ops, covariance→information inversion, Mahalanobis weighting,
  Huber robust loss, between/GNSS residuals, and `evaluate_factor` with
  accept / non-finite / invalid-covariance / stale reasons shared with the
  ingest-side `constraint_gate`
  (`mrn_graph/scripts/factor_graph.py`,
  [`docs/graph_architecture.md`](docs/graph_architecture.md) →
  "Solver-Independent Factor Core")
- [x] pure-Python Gauss-Newton batch backend on top of the factor core (no
  GTSAM/numpy): prior factors (pose + GNSS-style 2D position), between
  factors (odometry + relative-pose), covariance-weighted normal equations,
  per-factor Huber robust loss, numerical SE(2) Jacobians, and a per-factor
  report reusing `FactorReason`. This is the CI-green reference backend the
  GTSAM backend must match
  (`mrn_graph/scripts/pose_graph_solver.py`,
  [`docs/graph_architecture.md`](docs/graph_architecture.md) →
  "Reference Batch Backend")
- [x] GTSAM dependency verified: `ros-jazzy-gtsam` (4.x) imports with the
  full SE(2) factor-graph API locally, but is absent from the `build_jazzy`
  CI image — a GTSAM-backed node is gated on adding it to CI (documented in
  graph_architecture.md → "GTSAM dependency status")
- [x] Python `GraphBackend` layer + `FixedLagBackend`: ROS-free mirror of the
  C++ `Backend` interface that wraps the solver and produces the diagnostics
  contract (accepted / rejected / stale counts, stable rejection-reason
  vocabulary, per-agent accepted/rejected, degraded→0 quality). Anchors
  non-degraded agents, regularizes degraded ones, gates relatives by age and
  covariance, applies GNSS priors
  (`mrn_graph/scripts/graph_backend.py`,
  [`docs/graph_backend_plugin.md`](docs/graph_backend_plugin.md) →
  "Python Backend Layer")

### 13.2 Backend integration

- [x] `fixed_lag_graph_node.py` rclpy shell: ingests the same topics through
  the same constraint gate + time gate as `relative_anchor`, converts the
  agent-state / accepted-constraint window into the `graph_backend.py`
  dataclasses, runs `FixedLagBackend.step`, and publishes the standard
  cooperative-pose / cooperative-odom / `/mrn/graph/status` / marker topics
  (`backend_name = fixed_lag_python`). Opt-in via
  `graph_executable:=fixed_lag_graph_node.py`; the default stays
  `relative_anchor` so the CI smoke path is unchanged
  (`mrn_graph/scripts/fixed_lag_graph_node.py`)

- [x] three-way comparison in the experiment runner:
  `experiments/backend_comparison.yaml` runs `local_only` /
  `relative_anchor` / `fixed_lag` on the GNSS-outage scenario. A new
  method-vs-method acceptance rule (`max_ate_rmse_ratio_vs_method` +
  `vs_method_run`) encodes "fixed_lag improves or matches relative_anchor"
  (ATE ratio ≤ 1.05); the rule logic is unit-tested in CI without launching
  (`mrn_eval/mrn_eval/experiment_cli.py:evaluate_acceptance`,
  `experiments/backend_comparison.yaml`,
  [`docs/experiments.md`](docs/experiments.md) → "Backend Comparison")

- [x] GTSAM-backed backend: `gtsam_backend.GtsamBackend` is a drop-in for
  `FixedLagBackend` (same `step()` + diagnostics) using a GTSAM
  `NonlinearFactorGraph` with robust Huber noise and Levenberg-Marquardt,
  sharing the gating / prior / estimate helpers so it differs only in the
  optimizer. Selected via `fixed_lag_graph_node.py -p backend:=gtsam`
  (GTSAM imported lazily; default path and CI unaffected).
  `test_gtsam_backend.py` asserts equivalence with the pure reference,
  skipped wherever GTSAM is absent
  (`mrn_graph/scripts/gtsam_backend.py`,
  [`docs/graph_architecture.md`](docs/graph_architecture.md) → "GTSAM backend")

Deliberately deferred (CI-ops budget, not code):

- the default `build_jazzy` image stays lean (no `ros-jazzy-gtsam`); the
  GTSAM equivalence test runs anywhere GTSAM is installed and would move to a
  dedicated CI job rather than bloat the smoke job. Ceres remains the
  documented fallback if a GTSAM CI job proves unstable
- running `backend_comparison.yaml` as a full 3-launch CI smoke (the
  acceptance logic is already CI-tested; the launch run stays local/manual
  until added to the smoke budget)
- keep dummy backend for smoke tests

Acceptance:

- graph backend improves or matches `relative_anchor` on synthetic outage
  scenarios
- graph backend rejects invalid covariance and stale constraints
- graph backend publishes stable graph status
- CI still passes without requiring large datasets

## 14. v1.0.0 Goal

The v1.0 goal:

> Establish a stable ROS 2 cooperative localization infrastructure contract
> with real replay evidence, adapter paths, documented semantics, and a clear
> extension model.

Requirements:

- stable public message contracts or a documented migration path
- real bag replay benchmark
- synthetic benchmark suite
- graph backend plugin API
- [x] communication backend abstraction — `mrn_comm/scripts/comm_backend.py`
  defines the `CommunicationBackend` protocol (`name` / `transmit` /
  `diagnostics`) and a deterministic `LoopbackBackend` reference
  implementation. `LinkDiagnostics` maps one-to-one onto
  `mrn_msgs/msg/CommStatus`, and `build_comm_status` wraps it (ROS imported
  lazily). Backends carry packets and report loss/latency diagnostics without
  changing message semantics; documented in
  [`docs/qos_profiles.md`](docs/qos_profiles.md) → "Communication Backend
  Interface"
- Nav2 adapter
- documented Autoware adapter status
- complete frame/time/covariance docs
- reproducible reports
- CI artifacts for smoke experiments
- [x] examples for adding new constraint sources — the UWB range-bearing
  source (`uwb_constraint_source.py`) is the worked example: pure
  range/bearing → SE(2) relative pose with Jacobian covariance propagation,
  a `build_uwb_constraint` message builder, and tests asserting the output
  passes `constraint_gate`. Pattern documented in
  [`docs/graph_backend_plugin.md`](docs/graph_backend_plugin.md) → "Adding a
  New Constraint Source"
- [x] examples for adding new evaluators — the offline drift-rate metric
  (`compute_drift_rate`) is the worked example, with the extension pattern
  documented in [`docs/offline_ate.md`](docs/offline_ate.md) → "Adding a New
  Evaluator" (pure function over `AlignedPair` → `ErrorStats` → CLI wiring,
  opt-in `--drift-segment-m`, CI-tested without a bag)

Non-requirements:

- fully distributed graph optimization
- raw cooperative perception fusion
- cloud dashboard

## 15. Package Plans

### 15.1 `mrn_msgs`

Current role:

- owns public message contracts

Near-term tasks:

- review every message against docs/interfaces.md
- add comments to `.msg` files only where ROS message comments improve generated
  docs
- decide whether `ConstraintGraph` reason strings should stay strings or become
  constants
- preserve compatibility through the alpha

Rules:

- no algorithm dependencies
- no package should redefine these contracts locally
- every message semantic change needs docs and tests

### 15.2 `mrn_core`

Current role:

- placeholder for C++ utility contracts

Near-term tasks:

- add covariance validation helpers
- add frame-id validation helpers
- add transform direction helper docs
- add unit tests for covariance finite/positive checks

Later tasks:

- covariance adjoint transform helpers
- SE(2)/SE(3) utility functions
- C++ time gate mirror if Python helper becomes insufficient

### 15.3 `mrn_comm`

Current role:

- owns QoS profile names and YAML config

Near-term tasks:

- keep `agent_state_fast`, `relative_constraint`, `heartbeat`, and
  `static_agent_info` documented
- add a validation script for QoS profile YAML
- define which topics use each profile

Later tasks:

- [x] communication backend interface — `scripts/comm_backend.py`
  (`CommunicationBackend` protocol + `LinkDiagnostics` → `CommStatus`)
- DDS backend config loader
- [x] loopback backend — `LoopbackBackend` with a deterministic, seeded
  loss/latency model (`scripts/comm_backend.py`)
- Zenoh backend package

### 15.4 `mrn_sync`

Current role:

- time gate helper and tests

Near-term tasks:

- keep clock offset rejection behavior stable
- add docs for time jump behavior
- add tests for unknown offset and high uncertainty cases

Later tasks:

- NTP-like offset estimator
- known-offset simulator estimator
- PTP/GNSS PPS status ingestion adapter

### 15.5 `mrn_graph`

Current role:

- dummy backend
- relative-anchor backend
- constraint gate
- graph status output

Near-term tasks:

- keep `dummy_graph_node.py` working
- keep `relative_anchor_graph_node.py` documented as baseline
- add graph status to every backend
- add rejected constraint publication if useful for visualization

Next major task:

- introduce real factor graph backend

Rules:

- never accept invalid covariance silently
- never insert late constraints at "now" if they were measured in the past
- never hide rejected constraints from reports

### 15.6 `mrn_netem`

Current role:

- synthetic network profile model and CLI

Near-term tasks:

- keep random and burst profile tests
- document mapping from YAML profile to synthetic world faults
- add packet duplication/corruption fields only after loss/latency semantics are
  stable

Later tasks:

- Linux `tc netem` wrapper
- Docker/network namespace helper
- live profile monitor

### 15.7 `mrn_eval`

Current role:

- online ATE
- Markdown/JSON report collector
- experiment runner
- acceptance checks
- provenance output

Near-term tasks:

- keep report schema stable
- add recovery time metric
- add RPE
- add NEES/NIS when covariance output is meaningful
- add schema version to metrics JSON if needed

Rules:

- every experiment should be runnable from YAML
- every failure should preserve artifacts
- every acceptance rule should have a clear name in `acceptance.json`

### 15.8 `mrn_viz`

Current role:

- RViz config
- Foxglove layout

Near-term tasks:

- keep visualization aligned with report fields
- add markers for rejected constraints if not already visible enough
- ensure covariance markers are readable

Later tasks:

- Foxglove panel layout for latency, clock offset, ATE, graph status, and
  acceptance summary

### 15.9 `mrn_demos`

Current role:

- synthetic world node
- launch files
- scenario YAML
- bag manifest

Near-term tasks:

- keep scenario fields documented
- keep launch arguments stable
- keep synthetic demo deterministic under fixed seed
- add a small generated MCAP only when storage policy is decided

Rules:

- fake algorithm inputs are acceptable
- fake timestamp/frame/covariance/replay semantics are not acceptable

### 15.10 `mrn_coord`

The coordination / navigation layer — the counterpart to the localization
stack. Where localization answers *where are we*, this answers *how do we move
and what do we do together*. Same pattern: pure, ROS-free algorithm cores,
unit-tested in CI, with thin ROS/CLI wiring on top. Documented in
[`docs/coordination.md`](docs/coordination.md).

Current role:

- [x] MAPF (multi-agent path finding): `GridWorld`, space-time A* with
  vertex/edge constraints, conflict detection, optimal Conflict-Based Search,
  and prioritized planning; `mrn_mapf_demo` CLI renders solutions as an ASCII
  timeline. A `pure_pursuit` path follower (`mrn_path_follower`) tracks the
  planned paths with a unicycle, closing planning -> world through `mrn_sim`
- [x] decentralized formation control reusing the V2V relative-pose
  constraints: displacement-based consensus law over relative measurements,
  `FormationSpec` shape builders, closed-loop simulation, and an
  `mrn_formation_demo` CLI
- [x] cooperative coverage / exploration: three-state `OccupancyGrid`, frontier
  detection and clustering, BFS travel cost, and task allocation by greedy
  auction or optimal Hungarian assignment (cross-checked against brute force);
  `mrn_coverage_demo` CLI
- [x] swarm flocking: `flocking.flock_velocities` (pure Boids — separation /
  alignment / cohesion) scaling to tens-hundreds of agents; demo GIF

Later tasks:

- [x] closed-loop ROS demo: `mrn_agent_sim` (single-integrator plant publishing
  poses + RViz markers, integrating cmd_vel) plus `formation_closed_loop.launch.py`
  drive the formation controller to convergence entirely inside ROS, verified
  end-to-end; RViz config included
- [x] thin ROS nodes for all three modules, each a shell over the pure core
  (parsing/conversion CI-tested, nodes launch-smoke-tested):
  `mrn_mapf_planner` publishes `nav_msgs/Path` per agent;
  `mrn_formation_controller` subscribes to poses and publishes
  `geometry_msgs/Twist` per agent; `mrn_coverage_allocator` publishes a
  `geometry_msgs/PointStamped` frontier goal per robot. Launch files for each.
- continuous-space / kinematic extensions beyond the grid model
- [x] bridge the cooperative-localization estimate into the coordination layer:
  `mrn_pose_bridge` republishes per-agent `AgentState`/`CooperativePose` as
  `geometry_msgs/PoseStamped` on `formation/pose/<id>`. `estimate_to_formation.launch.py`
  wires the synthetic world -> bridge -> formation controller end-to-end
  (one-way: estimate -> coordination). Feeding MAPF starts / coverage cells
  from the estimate remains.

### 15.11 `mrn_sim`

The simulation foundation — a deterministic 2D *true world model* both halves
plug into (localization consumes its noisy sensor measurements; coordination's
commands drive its robots). Same pattern: pure, ROS-free cores, unit-tested in
CI. Documented in [`docs/simulation.md`](docs/simulation.md).

Current role:

- [x] 2D world core: unicycle kinematics, `World`/`Obstacle` with a
  collision-aware `step`, and geometric sensor models (range/bearing,
  body-frame relative pose, GNSS) with a reproducible Gaussian-noise helper;
  `scripts/make_sim_gif.py` renders a real-simulator demo GIF

- [x] `mrn_sim_world` ROS node + `sim_world.launch.py` (+ RViz config):
  integrates per-robot `Twist` commands and publishes `AgentState` (truth +
  reproducible GNSS noise), ground truth, and an RViz `MarkerArray`
  (robots / obstacles / in-range V2V links); `proximity.py` is the CI-tested
  pure helper, the node is launch-smoke-tested

- [x] emit `mrn_msgs/RelativePoseConstraint` for in-range pairs (V2V) on
  `/<id>/mrn/relative_constraints`, feeding the cooperative-localization graph
  directly: `relative_pose_observation` (pure, covariance from sigmas) +
  `v2v.build_relative_constraint`, `source_type=SOURCE_FAKE_GROUND_TRUTH`,
  verified to pass `constraint_gate` in the tests
- [x] end-to-end sim -> localization: `sim_localization.launch.py` feeds the
  sim's `AgentState` + V2V constraints into the relative-anchor graph; with
  `robot_2` GNSS-degraded, cooperative localization pulls its estimate back to
  within ~0.3 m of truth (`status = OK`), verified by an isolated-domain run.
  Also stamps the emitted `AgentState` (header + TTL) so freshness gates accept it

- [x] a unicycle path-follower so a MAPF plan can be driven through this world:
  `mrn_coord` `pure_pursuit` (pure) + `mrn_path_follower` node; the
  `mapf_through_sim.launch.py` closed loop (planner -> follower -> world) drives
  all three robots along their CBS paths to within ~0.3 m of their goals
- [x] coverage executed in the world: `mrn_goal_follower` drives each robot to
  its allocated frontier; `coverage_through_sim.launch.py` (allocator ->
  follower -> world) reaches the frontiers to within ~0.3 m (one allocation;
  iterative re-mapping is future work)

Later tasks:

- [x] swarm-scale runs (tens to hundreds of agents) for emergent behavior:
  `mrn_coord.flocking.flock_velocities` (pure Boids) drives a 70-agent flock in
  `scripts/make_swarm_gif.py`; `obstacle_avoidance` + `mrn_sim.swarm.flock_in_world`
  flock a unicycle swarm *through* the collision-aware world (Boids ->
  velocity_to_unicycle -> world.step), deterministically verified in CI
  (in-bounds, obstacle-clear, moving) — the testable twin of the Gazebo swarm.
  With `goal_seek` migration the flock travels to a goal around the obstacles
  (verified: the flock centroid closes most of the distance), and with
  `predator_evasion` the flock flees a pursuer (verified: mean distance from the
  predator grows while staying in bounds); `scripts/make_predator_gif.py`.
  `leader_follow` (followers track a leader) and multi-predator evasion compose
  into a multi-phase swarm mission (regroup -> migrate via waypoints -> evade ->
  reach goal), verified deterministically and shown in `make_mission_gif.py`
- [x] an optional Gazebo (`gz sim`) adapter for full 3D physics, kept out of the
  pure / CI core: `mrn_gazebo` — a validated diff-drive SDF world, a
  `ros_gz_bridge` config, and `mrn_gz_pose_adapter` republishing the bridged
  model pose as `AgentState`. Verified headless end-to-end (Gazebo pose →
  bridge → adapter → `/<id>/mrn/agent_state`). Documented in `docs/gazebo.md`
- [x] Gazebo multi-robot swarm: `swarm.launch.py` spawns N differential-drive
  vehicles and `mrn_gz_swarm_controller` flocks them (Boids ->
  `velocity_to_unicycle` -> diff-drive). The math is unit-tested and spawn +
  single-robot drive are verified; the full N-robot run targets a real machine
  (DDS discovery was unreliable in the CI-less sandbox — see docs/gazebo.md)

## 16. Message Contract Freeze Plan

The message contracts should not be frozen too early, but they should be stable
enough for the alpha. Freeze them in stages:

### Stage A: Alpha Stability

Allowed:

- add fields if required for diagnostics
- add constants
- clarify docs

Avoid:

- renaming fields
- changing transform direction
- changing covariance interpretation
- changing time semantics

### Stage B: Pre-v1 Migration

Allowed:

- make breaking changes with migration notes
- add version fields if needed
- split visualization messages from canonical messages if required

Required:

- changelog
- migration examples
- updated bag manifest

### Stage C: v1 Stability

Allowed:

- additive changes only where possible
- new messages for major semantic changes

Required:

- documented compatibility policy

## 17. Frame Plan

The frame contract remains:

```text
earth
  map
    robot_i/odom
      robot_i/base_link
```

Rules:

- `earth` is future-facing and may map to WGS84/ECEF concepts later
- `map` is the local mission frame for the MVP
- each robot owns its own `odom -> base_link`
- cooperative correction should publish `map -> robot_i/odom`
- V2V messages must not rely on blindly merging remote TF trees
- relative pose direction must remain documented as `T_from_to`

Planned tasks:

- add frame validator utility
- add tests for namespace-prefixed frames
- add docs for static transforms and adapter responsibilities
- add warning/report rows for unknown frames

## 18. Time Plan

Every relevant cooperative message should preserve:

- measurement time
- source publish time
- receive time where available
- graph insert time where available
- TTL
- clock offset estimate
- offset uncertainty

Near-term tasks:

- keep `ClockOffsetEstimate` docs current
- keep clock offset rejection visible in graph status
- add diagnostics for stale receive age and TTL drops
- ensure `/clock` replay behavior is documented

Later tasks:

- time jump handling tests
- offset estimator plugins
- latency compensation beyond synthetic transport

## 19. Covariance Plan

Covariance quality is a core differentiator. The project should be stricter
than typical quick demos.

Rules:

- covariance must be finite
- unknown covariance should be represented as large covariance, not zeros
- zero covariance should be treated as invalid unless explicitly documented
- relative pose covariance must be interpreted in the documented frame
- graph factors should convert covariance to information consistently
- outlier rejection should be covariance-aware

Near-term tasks:

- add covariance validation utility in `mrn_core`
- add tests for NaN, Inf, zero, negative, and overconfident covariance
- add docs for large-covariance fallback policy

Later tasks:

- adjoint covariance transform helper
- covariance consistency metrics
- NEES/NIS report sections

## 20. Network and QoS Plan

QoS and network faults should remain first-class, not side effects.

Current profiles:

- `agent_state_fast`
- `relative_constraint`
- `heartbeat`
- `static_agent_info`

Current benchmark:

- synthetic `qos_best_effort_vs_reliable.yaml`

Near-term tasks:

- validate QoS YAML in CI
- document topic-to-profile mapping
- add stale ratio to report when graph/drop data is available
- add useful constraint ratio

Later tasks:

- DDS profile examples for Cyclone DDS and Fast DDS
- Linux `tc netem` runner
- network namespace Docker demo
- Zenoh backend experiment

Important limitation:

Synthetic QoS comparison is not a DDS benchmark. It is a deterministic replay
contract for comparing communication behavior. Real DDS behavior should be a
separate benchmark when the profile runner is ready.

## 21. Evaluation Plan

Current metrics:

- online ATE RMSE
- localization availability
- improvement vs local-only
- network loss/latency/jitter/max latency
- graph accepted/rejected/stale counts
- rejection reason counts

Near-term metrics:

- recovery time after GNSS outage
- RPE
- useful constraint ratio
- stale/drop ratio
- graph solve time placeholder

Later metrics:

- NEES
- NIS
- CPU/memory
- bandwidth
- end-to-end delay
- per-factor rejection statistics

Rules:

- every demo should have a report path
- every report should have JSON and Markdown
- every benchmark should have deterministic seed/config/provenance
- every CI smoke should upload artifacts

## 22. Plugin Plan

The project should become plugin-oriented, but not before the contracts are
clear.

Planned plugin categories:

- `ConstraintSourcePlugin`
- `GraphBackendPlugin`
- `CommunicationBackendPlugin`
- `DatasetAdapterPlugin`
- `EvaluatorPlugin`

Initial implementations:

- dummy graph backend
- relative-anchor graph backend
- future GTSAM/Ceres backend
- synthetic dataset adapter
- rosbag/MCAP adapter
- [x] loopback communication backend (`mrn_comm/scripts/comm_backend.py`)

Rules:

- plugins must not change message semantics
- plugins must expose diagnostics
- plugins must have small acceptance tests

## 23. Real Robot Plan

The first real robot milestone should be modest.

Target:

- two robots
- local odometry
- GNSS or motion-capture ground truth if available
- one relative pose source, even if simple
- packet loss/latency diagnostics
- replayable bag

Minimum bag topics:

- `/clock` if replay uses simulated time
- `/tf`
- `/tf_static`
- `/robot_i/local/odometry`
- `/robot_i/local/gnss_pose` or equivalent prior
- `/robot_i/mrn/agent_state`
- `/robot_i/mrn/relative_constraints`
- `/robot_i/mrn/comm_status`
- `/robot_i/mrn/clock_status`
- `/robot_i/mrn/cooperative_odom`
- `/mrn/graph/status`
- `/mrn/eval/summary`

Acceptance:

- bag manifest validates
- report can be regenerated from bag/config
- known failures are documented

## 24. Nav2 Adapter Plan

The Nav2 adapter should be conservative.

Responsibilities:

- consume cooperative pose/correction
- publish or support `map -> robot_i/odom` correction
- enforce stale correction gate
- enforce max translation/rotation jump gate
- optionally expose diagnostics

Non-responsibilities:

- planner replacement
- controller replacement
- behavior tree replacement
- costmap ownership

Acceptance:

- adapter can be launched in isolation
- adapter fails safe when cooperative pose is stale
- docs explain interaction with robot_localization/Nav2 frame tree

## 25. Autoware Adapter Plan

The Autoware adapter should come after Nav2 and GNSS utilities.

Responsibilities:

- map Autoware localization outputs into `AgentState`
- map V2V/relative constraints into MRN contracts
- preserve covariance and timestamps
- document frame assumptions

Non-responsibilities:

- replacing Autoware localization
- replacing planning/control
- solving full V2X perception

Acceptance:

- adapter skeleton compiles
- one replay config demonstrates message conversion
- known limitations are clear

## 26. Zenoh Plan

Zenoh should not be required for the MVP.

Order:

1. plain ROS 2 DDS
2. [x] loopback backend (`mrn_comm/scripts/comm_backend.py` — `LoopbackBackend`
   behind the `CommunicationBackend` protocol)
3. [x] rosbag/replay backend (`ReplayBackend` replays a recorded
   `DeliveryRecord` trace; `RecordingBackend` captures one transparently;
   `trace_to_dicts`/`trace_from_dicts` round-trip it for a bag sidecar)
4. Zenoh backend experiment

Acceptance for first Zenoh milestone:

- optional package
- no core dependency
- docs explain setup
- benchmark compares transport diagnostics without changing graph semantics

## 27. Dataset Plan

Dataset support should be adapter-based and staged.

Near-term:

- MCAP/rosbag-first
- synthetic bag manifest
- real robot bag manifest

Later:

- Autoware bags
- UrbanNav or GNSS datasets
- OPV2V/V2X-Sim prototype only after localization contract is stable

Rules:

- do not chase perception datasets before cooperative localization is solid
- do not make CARLA/SUMO required for core demos

## 28. Demo and Public Launch Plan

The first public demo should show:

- three robots
- robot 2 GNSS outage
- packet loss
- clock drift or clock rejection
- V2V relative constraints
- cooperative recovery
- covariance change
- latency/loss plots or report table
- one-command launch or experiment run

Assets to produce:

- 60-second video
- short GIF for README
- screenshot of RViz/Foxglove
- benchmark report
- acceptance JSON
- architecture diagram

Message:

> One robot loses GNSS. The network drops packets. Clocks drift. V2V relative
> constraints keep the cooperative localization pipeline observable and
> replayable.

## 29. GitHub Growth Plan

The repository should be easy to understand and easy to contribute to.

Required:

- clear README
- badges
- quick start
- "what it is not"
- architecture diagram
- benchmark reports
- issue templates
- release checklist
- first alpha tag
- small issues with acceptance criteria

Recommended near-term issues:

- add covariance validator tests
- add QoS profile YAML validator
- add recovery time metric
- add RPE metric
- add rejected constraint marker topic
- add release notes for `v0.1.0-alpha`
- add first demo GIF
- add `GraphBackend` interface skeleton
- add bag replay smoke placeholder
- add Nav2 adapter design doc

## 30. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Project drifts into navigation stack replacement | very high | keep "what it is not" prominent; adapters only |
| Message semantics change too often | high | document contracts; add migration notes |
| Frame direction ambiguity | high | keep `T_from_to` docs and tests |
| Covariance treated casually | high | validation utilities and rejection tests |
| CI smoke becomes flaky | high | deterministic seeds, artifact upload, short selected sweeps |
| Synthetic demo overclaims real-world behavior | medium | label synthetic limitations clearly |
| GTSAM dependency complicates CI | medium | keep dummy/relative-anchor backends; consider Ceres/batch fallback |
| QoS benchmark confused with DDS benchmark | medium | document synthetic-vs-real clearly |
| README becomes too long | medium | keep PLAN/docs for long detail |
| Real robot data blocked | medium | start with internal/private bag manifest and synthetic parity |

## 31. AI Coding Workflow

This repository should stay friendly to AI-assisted development.

Rules:

- one package, one responsibility
- small issues with explicit acceptance criteria
- docs updated with code
- tests updated with behavior
- experiment YAMLs are source of truth for replay
- generated artifacts stay under `out/`
- avoid broad refactors during feature work
- preserve dummy backends for smoke tests

Good issue shape:

```text
Task: Add covariance validator for PoseWithCovariance.

Acceptance:
- rejects NaN and Inf
- rejects negative variance
- rejects all-zero covariance unless allow_unknown=false is overridden
- unit tests cover valid, invalid, and large-covariance cases
- docs/covariance.md updated
```

Bad issue shape:

```text
Implement cooperative localization.
```

## 32. Immediate Next Queue

The next practical tasks, in recommended order:

1. Update `docs/release_checklist.md` for clock drift/QoS CI smoke and artifact
   upload.
2. Add `CHANGELOG.md` or `docs/release_notes_v0.1.0-alpha.md`.
3. Add QoS profile YAML validation test.
4. Add covariance validation utilities in `mrn_core`.
5. Add recovery time metric in `mrn_eval`.
6. Add RPE metric in `mrn_eval`.
7. Add graph rejected-constraint marker/report improvements.
8. Add README demo screenshot/GIF placeholder.
9. Add `GraphBackend` plugin design doc.
10. Prepare `v0.1.0-alpha` known limitations section.

## 33. Definition of Done for New Features

A new feature is done when:

- code is implemented
- tests pass
- docs are updated
- experiment or smoke path exists when relevant
- failure modes are visible in diagnostics
- artifact output is preserved when relevant
- README is updated if the user-facing workflow changed

For this project, a feature that cannot be replayed or diagnosed is usually not
done.

## 34. Final Operating Principle

Keep the project boring in the right places:

- boring message contracts
- boring frame rules
- boring time rules
- boring covariance validation
- boring CI artifacts
- boring replay commands

That boring layer is what lets the interesting cooperative localization
research and real robot integration happen on top.


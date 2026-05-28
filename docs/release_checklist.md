# v0.1.0-alpha Release Checklist

The alpha release should prove the infrastructure contract, not algorithmic completeness.

## Build and Test

- [ ] `colcon build --symlink-install` passes on ROS 2 Jazzy.
- [ ] `colcon test --event-handlers console_direct+` passes.
- [ ] `colcon test-result --verbose` reports zero failures.

## Smoke Experiments

- [ ] `scripts/smoke_cooperative_demo.sh 20 out/smoke_report.md` passes locally.
- [ ] `tools/validate_bag_manifest.py mrn_demos/bags/mrn_demo_3robots_manifest.yaml` passes.
- [ ] `ros2 run mrn_eval mrn_experiment run experiments/gnss_outage_packet_loss.yaml --duration 25 --output-dir out/experiments/gnss_outage_packet_loss` produces a report with acceptance pass.
- [ ] `ros2 run mrn_eval mrn_experiment run experiments/clock_drift_sensitivity.yaml --duration 10 --sweep-case clock_drift_ms_50 --sweep-case clock_drift_ms_100 --output-dir out/experiments/clock_drift_smoke` passes acceptance.
- [ ] `ros2 run mrn_eval mrn_experiment run experiments/qos_best_effort_vs_reliable.yaml --duration 10 --output-dir out/experiments/qos_smoke` passes acceptance.

## CI Evidence

- [ ] `build-jazzy` workflow run is green on the release commit.
- [ ] `docs` workflow run is green on the release commit.
- [ ] `jazzy-smoke-artifacts` is uploaded from the release run and contains:
  - `out/experiments/clock_drift_smoke/report.md` and `acceptance.json`
  - `out/experiments/qos_smoke/report.md` and `acceptance.json`
  - `out/smoke_report.md`, `out/smoke_metrics.json`, `out/smoke_launch.log`, `out/smoke_summary.yaml`
- [ ] At least one CI log section shows the first ~140 lines of each smoke report.

## Documentation

- [ ] README Quick Start matches actual launch files and experiment commands.
- [ ] README CI Smoke Artifacts section lists the current artifact paths.
- [ ] `docs/interfaces.md` describes every public message contract used in the demos.
- [ ] `docs/frames.md` defines transform directions and frame ownership.
- [ ] `docs/time_sync.md` defines message time semantics and clock rejection rules.
- [ ] `docs/covariance.md` defines covariance validity and rejection rules.
- [ ] `docs/experiments.md` documents all experiment YAML features used by the alpha demos.
- [ ] `docs/qos_profiles.md` explains synthetic-vs-real QoS limitations.
- [ ] `docs/graph_architecture.md` labels `relative_anchor` as a temporary baseline.
- [ ] `docs/release_notes_v0.1.0-alpha.md` exists and states the known limitations below.

## Known Limitations To State

The release notes must state the following plainly:

- The cooperative backend is `relative_anchor`, not a real factor graph.
- Synthetic relative pose constraints come from fake ground truth plus noise.
- Online ATE is the primary localization metric today.
- NEES/NIS are roadmap items, not complete metrics yet.
- Recovery time and RPE metrics are roadmap items.
- QoS comparison is synthetic transport behavior, not a DDS vendor benchmark.
- No real robot bag is included yet.
- No Nav2 or Autoware adapter is implemented yet.

## Suggested Tag Command

```bash
git tag -a v0.1.0-alpha -m "v0.1.0-alpha"
```

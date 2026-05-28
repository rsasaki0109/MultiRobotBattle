# v0.1.0-alpha Release Notes

Status date: 2026-05-28

## Theme

The `v0.1.0-alpha` release is a *synthetic infrastructure alpha*. The goal is
to prove the message, frame, time, covariance, replay, and reporting contracts
that the rest of the project will be built on. Algorithmic completeness is not
the goal of this release.

A user should be able to:

1. Build the workspace on ROS 2 Jazzy.
2. Launch a synthetic 3-robot cooperative localization demo.
3. Run benchmark experiments with packet loss, latency, clock drift, and QoS
   profile variation.
4. Inspect Markdown and JSON reports that show localization, network, time,
   and graph diagnostics.
5. Download the same artifacts from CI.

## Highlights

- ROS 2 Jazzy build and test workflow with `jazzy-smoke-artifacts` upload.
- Public message contracts for agent state, V2V packet headers, constraints,
  comm status, clock status, graph status, cooperative pose, and evaluation
  summaries (`mrn_msgs`).
- Constraint gate and `relative_anchor` cooperative backend with rejected and
  stale constraint accounting (`mrn_graph`).
- Synthetic network fault model (`mrn_netem`) with random and burst profiles.
- Time gate utilities for clock offset rejection (`mrn_sync`).
- Experiment runner with deterministic YAML configs, sweep filtering,
  acceptance checks, and provenance output (`mrn_eval`).
- Online ATE node and Markdown/JSON report collector.
- Three smoke experiments in CI:
  - `experiments/gnss_outage_packet_loss.yaml`
  - `experiments/clock_drift_sensitivity.yaml`
  - `experiments/qos_best_effort_vs_reliable.yaml`
- RViz and Foxglove visualization assets (`mrn_viz`).
- Synthetic 3-robot demo world, launch files, scenarios, and bag manifest
  (`mrn_demos`).
- Documentation set covering interfaces, frames, time sync, covariance, QoS
  profiles, graph architecture, experiments, demo storyboard, and roadmap.

## Evidence

The release is acceptable when the checklist in
[release_checklist.md](release_checklist.md) is fully checked. CI artifacts
under `jazzy-smoke-artifacts` provide the primary evidence:

- `out/experiments/clock_drift_smoke/{report.md,acceptance.json}`
- `out/experiments/qos_smoke/{report.md,acceptance.json}`
- `out/smoke_report.md`, `out/smoke_metrics.json`, `out/smoke_launch.log`,
  `out/smoke_summary.yaml`

Typical synthetic cooperative result on the GNSS outage scenario:

| Agent | Method | ATE RMSE [m] | Improvement vs Local [m] |
| --- | --- | ---: | ---: |
| robot_2 | local_only | 1.076 | |
| robot_2 | cooperative | 0.057 | 1.019 |

These numbers come from the synthetic world. They are not a claim about real
robot performance.

## Known Limitations

This release is intentionally narrow. The following limitations apply:

- **Cooperative backend is a placeholder.** The current backend is
  `relative_anchor`. It is not a factor graph and not a real optimizer. It
  exists so the message, frame, time, and replay contracts can be exercised
  before the heavyweight optimizer arrives in `v0.4.0`.
- **Synthetic relative pose constraints.** Constraints come from fake ground
  truth perturbed by noise inside the demo world node. They do not come from
  real perception or real V2V hardware.
- **Online ATE is the primary metric.** Recovery time and RPE are roadmap
  items. NEES and NIS will arrive once covariance output is meaningful.
- **QoS comparison is synthetic.** The `qos_best_effort_vs_reliable`
  experiment compares synthetic transport behavior through `mrn_netem`. It is
  not a DDS vendor benchmark. A real DDS comparison will live in a separate
  benchmark once the profile runner is ready.
- **No real robot bag is included.** A two-robot real-or-real-like bag is the
  primary target of `v0.2.0`.
- **No Nav2 or Autoware adapter.** Adapters are planned for `v0.2.0` (Nav2)
  and `v0.3.0` (Autoware). The current release deliberately stops at the
  cooperative correction contract.
- **No distributed graph optimization.** Distributed optimization is out of
  scope until at least `v1.0.0`.
- **No Zenoh backend.** Zenoh remains an optional future experiment.
- **MCAP recording is manifest-only.** Bag capture procedure is documented
  via `mrn_demos/bags/mrn_demo_3robots_manifest.yaml`, but the alpha does not
  automate MCAP recording.

## What This Release Is Not

Per `PLAN.md` section 4, this release is not a Nav2 replacement, Autoware
replacement, Open-RMF fleet dashboard, cooperative perception zoo, or
distributed SLAM implementation.

## Upgrade and Stability Notes

- Message contracts in `mrn_msgs` are in *Stage A: Alpha Stability* per
  `PLAN.md` section 16. Additive changes and clarifications are allowed
  during the alpha; renames, frame direction changes, covariance
  interpretation changes, and time semantic changes are deferred to a future
  pre-`v1.0.0` migration.
- Experiment YAML schema follows the fields documented in
  `docs/experiments.md`. New experiment fields will be added with tests
  before new configs depend on them.

## Suggested Tag Command

```bash
git tag -a v0.1.0-alpha -m "v0.1.0-alpha"
git push origin v0.1.0-alpha
```

## Next Release

`v0.2.0` will focus on real-or-real-like two-robot bag replay and a basic
Nav2 adapter that publishes `map -> robot_i/odom` corrections with stale and
jump gates. See `PLAN.md` section 11 for the full v0.2 scope.

# Roadmap

Status legend: ✅ landed & CI-green · 🟡 scaffolding landed, pending real data / CI-ops · ⬜ planned.
See [PLAN.md](../PLAN.md) §11–§14 for the per-milestone checklists.

## Q1: MVP / Alpha — ✅

- ✅ ROS 2 Jazzy baseline
- ✅ message contracts
- ✅ synthetic three-robot demo
- ✅ centralized cooperative localization skeleton
- ✅ packet loss, latency, and clock drift profiles
- ✅ RViz and Foxglove visualization assets
- ✅ MCAP bag manifest format
- ✅ benchmark report scaffold
- ✅ CI smoke demo for `/mrn/eval/summary`

Release gate: see `docs/release_checklist.md`.

## Q2: Real Robots, Nav2, Zenoh — 🟡 (pending a recorded two-robot bag)

- 🟡 two-robot real bag — capture procedure + manifest validator landed
  (`docs/bag_capture.md`, `tools/validate_bag.py`); the bag itself is field work
- ✅ Nav2 adapter (`mrn_nav2_adapter`, `docs/nav2_adapter.md`)
- ✅ `map -> robot_i/odom` correction integration (with stale/jump/TF gates)
- ✅ bag-replay experiment runner + Linux-netns netem path
  (`experiments/bag_replay_smoke.yaml`, `docs/netem_netns.md`)
- ✅ offline ATE/RPE helper for bags without an in-bag truth topic
  (`docs/offline_ate.md`)
- ⬜ Zenoh backend experiment
- 🟡 real packet loss tests — synthetic + netns paths landed; real-link runs pending

## Q3: Autoware, GNSS, Datasets — 🟡 (pending an outdoor RTK dataset)

- ✅ Autoware adapter (`mrn_autoware_adapter`, `docs/autoware_adapter.md`)
- ✅ GNSS/ENU utilities (`mrn_gnss`: WGS84 ↔ ECEF ↔ local ENU, `docs/gnss.md`)
- ✅ RTK quality handling (NMEA GGA fix-quality → covariance, quality schedule)
- ✅ dataset adapter prototypes (RTK / bag-NavSatFix / TUM → offline-ATE CSV)
- ✅ robust graph backend — factor core + Gauss-Newton + GTSAM backend
  with Huber robust loss (`docs/graph_architecture.md`); see Q-extra below

## Q-extra: Cooperative Graph Backend — ✅ (v0.4.0 code, CI-green)

- ✅ solver-independent factor core (SE(2) residuals, covariance weighting,
  Huber loss, factor evaluation) — `mrn_graph/scripts/factor_graph.py`
- ✅ pure-Python Gauss-Newton fixed-lag reference backend —
  `pose_graph_solver.py`, `graph_backend.py`, `fixed_lag_graph_node.py`
- ✅ GTSAM-backed backend (opt-in `-p backend:=gtsam`), equivalence-tested
  against the reference where GTSAM is installed — `gtsam_backend.py`
- ✅ three-way backend comparison with a method-vs-method acceptance rule —
  `experiments/backend_comparison.yaml`
- 🟡 GTSAM CI job / launch-smoke budget — deferred CI-ops decision, not code

## Q4: Distributed and Shared World Hooks — ⬜

- ⬜ federated graph summary exchange
- ⬜ landmark and submap constraints
- ⬜ cooperative perception message hooks
- ⬜ RSU agent support
- ⬜ edge gateway experiments

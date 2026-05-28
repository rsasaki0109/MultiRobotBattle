# Graph Architecture

The MVP uses a centralized graph server. This is a baseline, not the final distributed architecture.

## Phase 1: Centralized Graph

```text
robot_1 ----\
robot_2 ----- graph server ---> /robot_i/mrn/cooperative_odom
robot_3 ----/
```

Inputs:

- local odometry
- GNSS prior
- relative pose constraint
- communication status
- clock status

Factors:

- odometry factor
- GNSS prior factor
- relative pose factor
- optional map prior factor

## Current Baselines

`dummy_graph_node.py` is a pass-through backend. It republishes local
`AgentState` as cooperative pose and odometry, while counting accepted and
rejected relative constraints.

`relative_anchor_graph_node.py` is a lightweight cooperative baseline. When an
agent is degraded, it uses recent relative pose constraints to non-degraded
agents as anchors. This is not a factor graph, but it makes the replay and
visualization contract testable before GTSAM/Ceres integration.

Run:

```bash
ros2 launch mrn_demos cooperative_localization.launch.py
ros2 launch mrn_demos cooperative_localization.launch.py graph_executable:=dummy_graph_node.py
```

See [`graph_backend_plugin.md`](graph_backend_plugin.md) for the plugin
contract that future GTSAM/Ceres backends will follow.

## Solver-Independent Factor Core

`mrn_graph/scripts/factor_graph.py` holds the factor math a real backend
needs regardless of which solver is wired in (GTSAM, Ceres, or a
hand-rolled fixed-lag optimizer). It is pure Python — no ROS, no numpy, no
GTSAM — so the residuals, weighting, and robust loss are unit-tested in CI
before any solver packaging is decided. This deliberately de-risks the v0.4
backend dependency question (see [`PLAN.md`](../PLAN.md) §13): the math is
proven first; the solver is an implementation detail layered on top.

The v0.4 factor families map onto these helpers:

| v0.4 plan item | helper |
| --- | --- |
| odometry between-factor | `between_residual(pose_i, pose_j, measured)` |
| relative-pose factor | `between_residual` (same SE(2) math, cross-robot) |
| GNSS prior factor | `gnss_prior_residual(pose, measured_xy)` |
| covariance-aware weighting | `information_from_covariance`, `mahalanobis_norm` |
| robust loss | `huber_weight(norm, delta)` |
| rejected factors and reasons | `evaluate_factor` → `FactorReason` |

`evaluate_factor(residual, covariance, *, age_sec, max_age_sec, huber_delta)`
ties them together: it rejects non-finite measurements, invalid
covariances, and stale factors (the same rejection categories the
`constraint_gate` enforces upstream), and for accepted factors returns the
covariance-weighted residual norm plus a Huber robust weight. A backend
builds a candidate factor, calls `evaluate_factor`, and uses the result to
decide whether to add the factor and how to weight it — the same decision
surface whether the optimizer is GTSAM or Ceres.

Poses are SE(2) `(x, y, yaw)`; covariances are 3×3 for pose factors and 2×2
for the position-only GNSS prior. The module shares the rejection vocabulary
with `constraint_gate.py` so graph-status reports can attribute a dropped
factor to the same reasons whether it was gated at ingest or at factor
construction.

## Reference Batch Backend

`mrn_graph/scripts/pose_graph_solver.py` is a pure-Python Gauss-Newton
optimizer for a 2D pose graph, built on the factor core above. It is a
dependency-free reference backend: `PriorFactor` (pose or GNSS-style 2D
position), `BetweenFactor` (odometry or relative-pose), covariance-weighted
normal equations, optional per-factor Huber weighting, and a per-factor
report that reuses the `FactorReason` vocabulary. Jacobians are numerical
(finite differences with the same body-frame SE(2) retraction used for the
update), which keeps a reference implementation free of hand-derived
Jacobian bugs. At least one prior is required to fix the gauge; a graph with
none reports non-convergence rather than returning a degenerate result.

This backend is what the synthetic-outage acceptance can be validated
against in CI, and it is the reference the eventual GTSAM backend must
match.

`fixed_lag_graph_node.py` is the thin rclpy shell over it: it ingests the
same topics through the same constraint gate and time gate as
`relative_anchor_graph_node`, converts the agent-state / accepted-constraint
window into the backend dataclasses, runs `FixedLagBackend.step`, and
publishes the standard cooperative-pose / cooperative-odom /
`/mrn/graph/status` / marker topics with `backend_name = fixed_lag_python`.
It is opt-in and does not change the default:

```bash
ros2 launch mrn_demos cooperative_localization.launch.py \
  graph_executable:=fixed_lag_graph_node.py
```

The default `graph_executable` stays `relative_anchor_graph_node.py`, so the
CI smoke path is unchanged.

### GTSAM backend

The same node can run a GTSAM-backed optimizer instead of the pure-Python
one by setting `backend:=gtsam`:

```bash
ros2 run mrn_graph fixed_lag_graph_node.py --ros-args -p backend:=gtsam
```

`gtsam_backend.GtsamBackend` is a drop-in for `FixedLagBackend` with the same
`step()` interface and diagnostics contract; it builds a GTSAM
`NonlinearFactorGraph` (`PriorFactorPose2`, `BetweenFactorPose2`, robust
Huber noise, Levenberg-Marquardt) but shares the gating, prior-covariance
selection, and estimate assembly helpers (`classify_relatives`,
`prior_covariance_for_agent`, `build_estimates`) with the Python backend, so
the two differ only in the optimizer. GTSAM is imported lazily — only when
`backend:=gtsam` is selected — so the default path and CI are unaffected.

Requires `ros-jazzy-gtsam` (verified importable as the `gtsam` Python
module). `test_gtsam_backend.py` asserts the GTSAM backend agrees with the
pure-Python reference on the synthetic scenarios; it is skipped wherever
GTSAM is absent (the default CI image), so the always-green path never
depends on GTSAM while the equivalence is still checked anywhere GTSAM is
installed.

### GTSAM dependency status

GTSAM is the intended high-performance backend. Verified locally:
`ros-jazzy-gtsam` (4.x) imports as the `gtsam` Python module and exposes
`Pose2`, `NonlinearFactorGraph`, `BetweenFactorPose2`,
`noiseModel.Diagonal` / `noiseModel.Robust`, and
`LevenbergMarquardtOptimizer` — everything a fixed-lag backend needs.

However, the `build_jazzy` CI image installs only
`python3-colcon-common-extensions` and `python3-yaml`, so GTSAM is **not**
present in CI. Wiring a GTSAM-backed node is therefore gated on adding
`ros-jazzy-gtsam` to the CI image (and keeping the default
`graph_executable` on `relative_anchor` so the smoke path stays green). The
pure-Python solver above intentionally lands the backend contract without
that dependency, satisfying the v0.4 acceptance that "CI still passes
without requiring large datasets."

## Phase 2: Federated Graph

Robots keep local graphs and exchange keyframes, marginal covariance, and relative constraints.

## Phase 3: Distributed Graph

Future work:

- subgraph exchange
- separator variables
- marginal priors
- distributed loop closure
- asynchronous optimization
- distributed robust outlier rejection

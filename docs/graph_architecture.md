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

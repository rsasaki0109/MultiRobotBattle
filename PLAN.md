# Plan — multirobot-navigation

Scope: **multi-robot simulation, navigation, and coordination**. A deterministic
2D world plus the planning/control/swarm algorithms that move robots through it,
all as pure, CI-tested cores with thin ROS/CLI wiring.

Cooperative **localization** is out of scope here — it lives in the companion
repo [`multirobot-localization`](https://github.com/rsasaki0109/multirobot-localization)
(rosbag-centric, real-data benchmarks). The two meet only at the message
contract (`mrn_msgs/AgentState`, `RelativePoseConstraint`): this repo's
simulator emits them; that repo consumes them.

## Packages

- `mrn_msgs` — message contracts (the interface to the localization consumer).
- `mrn_sim` — deterministic 2D world (unicycle kinematics, obstacles, collision,
  V2V/GNSS/range sensors), point-to-point navigation, and the swarm driver; a
  thin `mrn_sim_world` ROS node.
- `mrn_coord` — coordination: MAPF (CBS / prioritized), formation control,
  coverage (frontier + greedy/Hungarian), swarm flocking.
- `mrn_gazebo` — optional Gazebo (`gz sim`) adapter (requires Gazebo; not in CI).

## Done

- [x] `mrn_sim` 2D world core: unicycle kinematics, collision-aware `step`,
  sensor models, proximity queries — pure, CI-tested; ROS node emitting
  `AgentState` / `RelativePoseConstraint` (stamped) and accepting `cmd_vel`.
- [x] MAPF: space-time A*, conflict detection, Conflict-Based Search (optimal),
  prioritized planning; `mrn_mapf_demo` + `mrn_mapf_planner` node + pure-pursuit
  path follower; `mapf_through_sim` closed loop.
- [x] Formation control: displacement-based consensus over relative
  measurements; `mrn_formation_demo` + controller node + `mrn_agent_sim`
  closed loop.
- [x] Coverage: occupancy grid, frontier detection/clustering, BFS cost, greedy
  / Hungarian allocation (cross-checked vs brute force); `mrn_coverage_demo` +
  allocator + goal-follower nodes; `coverage_through_sim`.
- [x] Swarm flocking: separation / alignment / cohesion + obstacle avoidance +
  migration + predator evasion (one or many) + leader following; deterministic
  swarm-in-world driver + multi-phase mission; CI-verified.
- [x] Navigation: occupancy grid + grid A* + pure pursuit; reciprocal
  multi-robot collision avoidance; replanning around dynamic obstacles.
- [x] Optional Gazebo adapter: validated diff-drive world, `ros_gz_bridge`,
  pose→`AgentState`, multi-robot spawn + swarm controller (real machine; not CI).
- [x] CI: build + `colcon test` over all packages + coordination CLI demos.

## Next ideas

- [x] reusable benchmark environment (`mrn_sim.benchmark`): `Scenario`
  (YAML/dict) + `run_scenario(scenario, policy) -> BenchmarkResult` with
  standard metrics (success, makespan, path length, min clearance, min
  inter-robot distance, collisions); a scenario library (`mrn_sim/scenarios/`),
  a baseline `navigate_policy`, and a `mrn_sim_bench` CLI. External
  planners/controllers plug in as a `policy(world) -> commands` callable.
- Standard MAPF benchmarks (movingai) with comparable solve-rate / runtime.
- A scenario-driven CI benchmark gate (expected-metrics regression like the
  localization repo's).
- Continuous-space / kinematic planning beyond the grid; deadlock resolution
  for reciprocal avoidance (priorities / ORCA).
- Real-robot bring-up (separate effort; the localization repo is rosbag-first).

## Rules

- Pure algorithm cores are ROS-free and unit-tested; ROS nodes are thin shells
  smoke-tested via launch.
- Demos are synthetic, deterministic, and reproducible; GIFs are driven by the
  real algorithms.
- State honestly what is verified vs. pending (e.g. Gazebo multi-robot runs on a
  real machine, not in the CI sandbox).

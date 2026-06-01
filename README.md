# multirobot-navigation

<p align="center">
  <img src="docs/media/coordination_demo.gif" alt="Three robots funnel through a one-cell doorway without colliding via Conflict-Based Search, then converge into a triangle via decentralized formation control" width="720">
</p>

<p align="center">
  <em>Conflict-Based Search plans a collision-free doorway crossing, then a consensus controller assembles a formation — driven by the real algorithms. Regenerate with <code>scripts/make_coordination_gif.py</code>.</em>
</p>

[![build-jazzy](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/build_jazzy.yaml/badge.svg)](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/build_jazzy.yaml)
[![docs](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/docs.yaml/badge.svg)](https://github.com/rsasaki0109/multirobot-navigation/actions/workflows/docs.yaml)

ROS 2-native **multi-robot simulation, navigation, and coordination** — a
deterministic, pure-core, CI-tested stack for developing and benchmarking
multi-robot motion algorithms without hardware.

> **Localization lives in a companion repo:** cooperative multi-agent
> localization (rosbag-centric, real-data benchmarks on UTIAS MR.CLAM / KITTI)
> is [**multirobot-localization**](https://github.com/rsasaki0109/multirobot-localization).
> This repo answers *how the robots move*; that one answers *where they are*.
> They meet at the message contract (`mrn_msgs/AgentState`,
> `RelativePoseConstraint`): the simulator here emits them, that repo consumes
> them.

## What It Is

A 2D, deterministic multi-robot world plus the coordination and navigation that
moves robots through it. Every layer is a pure, ROS-free algorithm core
unit-tested in CI, with thin ROS/CLI wiring on top.

- **Simulation** (`mrn_sim`) — a deterministic 2D world: unicycle kinematics,
  circular obstacles with collision, and V2V / GNSS / range-bearing sensor
  models. It emits the localization message contract and accepts `cmd_vel`, so
  it is the plant the rest of the stack drives. An optional **Gazebo** adapter
  (`mrn_gazebo`) runs the same contract on a 3D physics world.
- **Navigation** (`mrn_sim.navigate`, `mrn_sim.kinodynamic`) — point-to-point
  navigation: occupancy grid from the obstacles, grid A* planning, pure-pursuit
  following, with **reciprocal multi-robot collision avoidance** and
  **replanning around dynamic obstacles** — plus a continuous-space
  **Hybrid A\*** kinodynamic planner (bounded turning radius, Dubins curves +
  analytic expansion) for smooth, feasibly-followable paths, **DWA** /
  **MPC (iLQR)** optimizing local controllers for accel-limited tracking, a
  **Control Barrier Function** QP safety filter for provable collision-free
  steering, and a **certified body-true safety shield** whose braking speed cap
  keeps the robot body — not a look-ahead point — collision-free under the accel
  limit, even against moving obstacles, and **reciprocally** for several shielded
  robots in adversarial mutual pursuit with no shared coordination
  (`scripts/certify_shield.py`).
- **Coordination** (`mrn_coord`) — multi-agent path finding (optimal
  Conflict-Based Search, **bounded-suboptimal ECBS**, **complete satisficing
  LaCAM**, and **anytime MAPF-LNS** that scale further, prioritized planning,
  all over a **space-time A\*** or drop-in **SIPP** safe-interval low level,
  plus **lifelong / online MAPF** stepped by
  **PIBT** with **auction / Hungarian** task allocation for warehouse-style
  endless-task throughput), **ORCA** reciprocal
  local collision avoidance, decentralized formation control, cooperative
  coverage (frontier + greedy/Hungarian allocation), and swarm flocking (Boids:
  separation / alignment / cohesion + obstacle avoidance + migration + predator
  evasion + leader following).
- **Benchmark environment** (`mrn_sim.benchmark`) — plug your own multi-robot
  policy into a `Scenario` and get comparable metrics (success, makespan, path
  length, clearance, inter-robot distance, collisions). Five baseline policies
  ship for comparison — grid A* + pursuit, **Hybrid A\*** kinodynamic, **DWA**
  local control, **MPC** (iLQR receding-horizon optimization, space-time
  avoidance), and **ORCA** — and
  and an **end-to-end MAPF executor** (`mrn_sim.mapf_exec`) runs a discrete
  grid plan in the continuous world — exposing where the discrete guarantee
  breaks down and bridging it with a Temporal-Plan-Graph schedule —
  `scripts/compare_planners.py` tabulates them all across the bundled scenarios
  ([`benchmarks/comparison.md`](benchmarks/comparison.md)). `ros2 run mrn_sim
  mrn_sim_bench crossing` runs a bundled scenario with a baseline policy. MAPF
  also loads the standard **MovingAI** `.map`/`.scen` format
  (`ros2 run mrn_coord mrn_mapf_bench`), so the planners can be evaluated on the
  community benchmark suite.

## Architecture

```
            ┌──────────── mrn_sim — the deterministic world ───────────┐
            │  unicycle kinematics · obstacles · collision · sensors    │
   cmd_vel ─▶  navigate (A* + pursuit + avoidance) / swarm / coordination│
            │  ─▶ AgentState · RelativePoseConstraint · ground truth     │
            └───────────────────────────────┬───────────────────────────┘
                                             │ message contract (mrn_msgs)
                          ▼                                ▼
              localization consumer                  Gazebo (mrn_gazebo)
              (multirobot-localization repo)         3D physics, same contract
```

The layers connect only through message contracts and pure interfaces, so each
is testable and replaceable in isolation. The simulator emits exactly what a
cooperative-localization consumer (the companion repo) ingests.

## Demos

All animations are driven by the real algorithms (no hand-drawn paths) and are
deterministic; regenerate with the matching `scripts/make_*_gif.py`.

**Coordination** — MAPF (Conflict-Based Search / prioritized), formation
control, frontier coverage. Each has a CLI demo (`mrn_mapf_demo`,
`mrn_formation_demo`, `mrn_coverage_demo`) and a thin ROS node; the top GIF
shows CBS + formation.

**Simulation & swarm** — the `mrn_sim` 2D world and Boids flocking, the same
foundation from a handful of robots to a swarm:

<p align="center">
  <img src="docs/media/sim_demo.gif" alt="Robots roam a 2D world with obstacles, exchanging V2V links" width="410">
  <img src="docs/media/swarm_demo.gif" alt="Seventy agents flock via separation, alignment, and cohesion" width="410">
</p>

Flocking *through* the collision-aware world — migrate to a goal, flee a
predator, and a multi-phase mission (regroup → migrate → evade → reach):

<p align="center">
  <img src="docs/media/swarm_sim_demo.gif" alt="A flock migrating to a goal around obstacles" width="270">
  <img src="docs/media/predator_demo.gif" alt="A flock fleeing a pursuing predator" width="270">
  <img src="docs/media/mission_demo.gif" alt="A swarm carrying out a multi-phase mission" width="270">
</p>

**Navigation** — grid A* plan + pure-pursuit follow to a goal, with reciprocal
multi-robot avoidance and replanning around a moving obstacle:

<p align="center">
  <img src="docs/media/nav_demo.gif" alt="Robots planning A* paths around obstacles to their goals" width="270">
  <img src="docs/media/recip_nav_demo.gif" alt="Robots navigating to crossing goals while avoiding each other" width="270">
  <img src="docs/media/replan_demo.gif" alt="A robot replanning around a moving obstacle" width="270">
</p>

**ORCA** — Optimal Reciprocal Collision Avoidance: two crowds walk straight at
each other and pass through, collision-free, each picking the velocity closest
to its goal that stays provably safe (`mrn_coord.orca`, regenerate with
`scripts/make_orca_gif.py`). Our port is checked against the reference RVO2
library — same scenarios, same velocity to ~1e-5
([`benchmarks/orca_rvo2.md`](benchmarks/orca_rvo2.md)):

<p align="center">
  <img src="docs/media/orca_demo.gif" alt="Two crowds of agents walk into each other and pass through collision-free via ORCA reciprocal avoidance" width="560">
</p>

**3D physics — Gazebo** — the same navigation, but in the `mrn_gazebo` (`gz sim`,
Harmonic) **3D** world: three robots cross an obstacle arena under the repo's own
A\* grid planning + pure-pursuit + reciprocal avoidance, driven over `cmd_vel`.
Each carries a **360° LiDAR** whose live returns are overlaid on the render, so
you can watch the lasers trace the obstacles and the other robots. Rendered and
recorded **fully offscreen on the GPU** (no GUI, no desktop window) by
`scripts/record_gazebo_gif.py` — the 3D counterpart to the deterministic 2D
demos, driven by the same algorithms:

<p align="center">
  <img src="docs/media/gazebo_demo.gif" alt="Three robots cross a 3D Gazebo arena of cylindrical obstacles via the repo's A* + pure-pursuit + reciprocal-avoidance navigation, each sweeping a 360-degree LiDAR whose returns trace the obstacles" width="660">
</p>

The same offscreen seam runs the other layers in 3D too — **ORCA** crowds passing
through each other, **Boids** swarming past obstacles (with the flock's LiDAR
point cloud), and **CBS + formation** funneling through a doorway — each driven by
the matching `mrn_coord` algorithm and sharing one recording harness
(`scripts/_gz_record.py`, `scripts/record_gazebo_{orca,swarm,coord}_gif.py`):

<p align="center">
  <img src="docs/media/gazebo_orca_demo.gif" alt="Two streams of robots in a 3D Gazebo world pass through each other collision-free via ORCA" width="265">
  <img src="docs/media/gazebo_swarm_demo.gif" alt="Twelve robots flock through a 3D Gazebo arena past obstacles via Boids rules, their LiDAR returns drawn as a point cloud" width="265">
  <img src="docs/media/gazebo_coord_demo.gif" alt="Three robots funnel through a doorway via Conflict-Based Search then assemble a formation in 3D Gazebo, their LiDAR tracing the wall" width="265">
</p>

## Packages

| Package | Role |
| --- | --- |
| `mrn_msgs` | message contracts (agent state, V2V relative-pose constraints, …) — the interface to the localization consumer |
| `mrn_sim` | deterministic 2D world (kinematics, obstacles, sensors), point-to-point navigation, and the swarm driver; a `mrn_sim_world` ROS node |
| `mrn_coord` | coordination: MAPF (CBS / prioritized), formation control, coverage, swarm flocking — pure cores, CLI demos, and thin ROS nodes |
| `mrn_gazebo` | optional Gazebo (`gz sim`) adapter: bridges model poses into `AgentState` so a 3D physics world can be the plant (requires Gazebo; not in CI) |

## Quick Start

Build with ROS 2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run the coordination CLI demos (pure, no ROS daemon):

```bash
ros2 run mrn_coord mrn_mapf_demo            # Conflict-Based Search
ros2 run mrn_coord mrn_formation_demo       # formation control
ros2 run mrn_coord mrn_coverage_demo        # frontier allocation
```

Drive a planned path through the simulator, or close a formation loop in ROS:

```bash
ros2 launch mrn_sim mapf_through_sim.launch.py use_rviz:=true
ros2 launch mrn_coord formation_closed_loop.launch.py use_rviz:=true
```

Regenerate any demo GIF: `python3 scripts/make_<name>_gif.py`.

See [docs/simulation.md](docs/simulation.md), [docs/coordination.md](docs/coordination.md),
and [docs/gazebo.md](docs/gazebo.md).

## Continuous Integration

Every push builds the workspace and runs `colcon test` over all packages (the
pure algorithm cores), then exercises the coordination CLI demos end-to-end. A
final **benchmark gate** (`scripts/benchmark_gate.py`) runs the bundled
scenarios and the MovingAI MAPF example and compares their metrics against
checked-in expectations in `benchmarks/expected_metrics/`, so a regression that
drops a goal, introduces a collision, or worsens a makespan / sum-of-costs fails
the build — the benchmarks are a guarded contract, not decoration.

## License

Apache-2.0.
